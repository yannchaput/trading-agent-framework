from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderEventError

logger = logging.getLogger(__name__)


class SafeList[T]:
    """A thread-safe list, guarded by a single re-entrant lock.

    All mutating and reading operations take the lock, so concurrent access from
    multiple broker stream/polling threads is safe. `snapshot()` is the only way
    callers should iterate the contents: it returns an independent copy so the
    caller can never observe (or corrupt) the live list mid-iteration.
    """

    def __init__(self) -> None:
        self._items: list[T] = []
        self._lock = threading.RLock()

    def append(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def remove(self, item: T, key: str = "identifier") -> T | None:
        """Remove and return the item whose `key` attribute matches `item`'s.

        Matches by attribute equality (not object identity), so a different
        instance carrying the same key value is treated as the same logical item.
        Returns None (never raises) if no match is found.
        """
        target = getattr(item, key)
        with self._lock:
            for index, candidate in enumerate(self._items):
                if getattr(candidate, key) == target:
                    return self._items.pop(index)
        return None

    def get_by(self, key: str, value: Any) -> T | None:
        with self._lock:
            for candidate in self._items:
                if getattr(candidate, key) == value:
                    return candidate
        return None

    def snapshot(self) -> list[T]:
        """Return an independent copy of the current contents."""
        with self._lock:
            return list(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, item: object) -> bool:
        with self._lock:
            return item in self._items


class OrderTracker:
    """Broker-agnostic order state machine.

    Orders move between six thread-safe buckets as trade events arrive:
    unprocessed -> new -> (partially_filled ->)* filled, or new -> canceled/error.
    """

    def __init__(self) -> None:
        self.unprocessed: SafeList[Order] = SafeList()
        self.new: SafeList[Order] = SafeList()
        self.partially_filled: SafeList[Order] = SafeList()
        self.filled: SafeList[Order] = SafeList()
        self.canceled: SafeList[Order] = SafeList()
        self.error: SafeList[Order] = SafeList()
        self.listeners: list[Callable[[Order, OrderEvent], None]] = []
        self._transition_lock = threading.RLock()

    def _buckets(self) -> tuple[SafeList[Order], ...]:
        return (
            self.unprocessed,
            self.new,
            self.partially_filled,
            self.filled,
            self.canceled,
            self.error,
        )

    def track_unprocessed(self, order: Order) -> None:
        self.unprocessed.append(order)

    def get_tracked_order(self, identifier: str) -> Order | None:
        for bucket in self._buckets():
            found = bucket.get_by("identifier", identifier)
            if found is not None:
                return found
        return None

    def get_all_tracked_orders(self) -> list[Order]:
        orders: list[Order] = []
        for bucket in self._buckets():
            orders.extend(bucket.snapshot())
        return orders

    def get_active_orders(self) -> list[Order]:
        active_buckets = (self.unprocessed, self.new, self.partially_filled)
        orders: list[Order] = []
        for bucket in active_buckets:
            orders.extend(bucket.snapshot())
        return orders

    def mark_replaced(self, old: Order, new: Order) -> None:
        """Record that the broker replaced `old` with `new` (Alpaca PATCH /orders/{id}).

        No listener fires: the stream's own `replaced` event for `old` maps to
        MODIFIED (a no-op), and `new` reports its own events once tracked.
        """
        with self._transition_lock:
            self._process_canceled(old)
            self.unprocessed.append(new)

    def _remove_from_all_buckets(self, order: Order) -> None:
        for bucket in self._buckets():
            bucket.remove(order)

    def _process_new(self, order: Order) -> None:
        self._remove_from_all_buckets(order)
        order.status = OrderStatus.NEW
        self.new.append(order)

    def _process_partial_fill(self, order: Order, price: Decimal, filled_quantity: Decimal) -> None:
        self._remove_from_all_buckets(order)
        order.add_transaction(price, filled_quantity)
        order.status = OrderStatus.PARTIAL_FILL
        self.partially_filled.append(order)

    def _process_fill(self, order: Order, price: Decimal, filled_quantity: Decimal) -> None:
        self._remove_from_all_buckets(order)
        order.add_transaction(price, filled_quantity)
        order.status = OrderStatus.FILL
        self.filled.append(order)

    def _process_canceled(self, order: Order) -> None:
        self._remove_from_all_buckets(order)
        order.status = OrderStatus.CANCELED
        self.canceled.append(order)

    def _process_error(self, order: Order) -> None:
        self._remove_from_all_buckets(order)
        order.status = OrderStatus.ERROR
        self.error.append(order)

    def _notify_listeners(self, order: Order, event: OrderEvent) -> None:
        for listener in self.listeners:
            try:
                listener(order, event)
            except Exception:
                logger.exception(
                    "Order tracker listener raised for order %s on event %s",
                    order.identifier,
                    event,
                )

    def process_trade_event(
        self,
        order: Order,
        event: OrderEvent,
        *,
        price: Decimal | None = None,
        filled_quantity: Decimal | None = None,
    ) -> None:
        """Apply a broker trade event to `order`, updating buckets and status.

        Validation happens before any bucket is touched: a FILLED or
        PARTIALLY_FILLED event missing `price` or `filled_quantity` raises
        `OrderEventError` and leaves all bucket state untouched.
        """
        with self._transition_lock:
            if event in (OrderEvent.FILLED, OrderEvent.PARTIALLY_FILLED) and (
                price is None or filled_quantity is None
            ):
                raise OrderEventError(
                    f"{event} event for order {order.identifier} requires both "
                    "price and filled_quantity"
                )

            if event == OrderEvent.MODIFIED:
                logger.info("Order %s modified; no bucket change", order.identifier)
                return

            if event == OrderEvent.NEW:
                self._process_new(order)
            elif event == OrderEvent.PARTIALLY_FILLED:
                # Guaranteed non-None by the guard above (FILLED/PARTIALLY_FILLED events
                # with a missing price or filled_quantity raise before reaching here).
                assert price is not None
                assert filled_quantity is not None
                self._process_partial_fill(order, price, filled_quantity)
            elif event == OrderEvent.FILLED:
                assert price is not None
                assert filled_quantity is not None
                self._process_fill(order, price, filled_quantity)
            elif event == OrderEvent.CANCELED:
                self._process_canceled(order)
            elif event == OrderEvent.ERROR:
                self._process_error(order)
            else:
                raise OrderEventError(f"Unhandled order event: {event}")

            self._notify_listeners(order, event)

"""`TastytradeOrderEvents`: applies Tastytrade order snapshots (account streamer or REST) to `OrderTracker`.

Feeds the tracker only -- never strategy hooks. Snapshots are full order objects, so applying one
twice is harmless: fills are counted once by `fill-id`. Orders are found by Tastytrade order id
(falling back to `external-identifier`). A replace gives the order a new id; the old one is
"superseded" from the moment the replace is sent: its fills still count, its statuses are ignored,
so the original's `Cancelled` never cancels the modified order.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order

logger = logging.getLogger(__name__)

_NEW_FROM = frozenset({OrderStatus.UNPROCESSED, OrderStatus.SUBMITTED})


class TastytradeOrderEvents:
    def __init__(self, tracker: OrderTracker) -> None:
        self._tracker = tracker
        # Reentrant, held across a whole apply(): the stream thread and the strategy thread
        # (submit, reconcile) may apply the same snapshot concurrently.
        self._lock = threading.RLock()
        self._current: dict[str, str] = {}  # our identifier -> current Tastytrade id
        self._orders: dict[str, Order] = {}  # every registered Tastytrade id -> our order
        self._superseded: set[str] = set()
        self._seen_fills: set[tuple[str, str]] = set()  # (Tastytrade order id, fill id)

    def register(self, order: Order, tt_id: str, *, snapshot: Mapping[str, Any] | None = None) -> None:
        """Map `order` to its Tastytrade id; `snapshot` marks its fills as already counted (an adopted order)."""
        with self._lock:
            self._current[order.identifier] = tt_id
            self._orders[tt_id] = order
            if snapshot is not None:
                self._seen_fills.update((tt_id, f.fill_id) for f in orders.parse_fills(snapshot))

    def tt_id_of(self, order: Order) -> str | None:
        with self._lock:
            return self._current.get(order.identifier)

    def begin_replace(self, order: Order) -> str | None:
        """Supersede the current id before a replace is sent; returns it (None when unknown)."""
        with self._lock:
            tt_id = self._current.get(order.identifier)
            if tt_id is not None:
                self._superseded.add(tt_id)
            return tt_id

    def abort_replace(self, tt_id: str) -> None:
        with self._lock:
            self._superseded.discard(tt_id)

    def finish_replace(self, order: Order, new_tt_id: str) -> None:
        with self._lock:
            old = self._current.get(order.identifier)
            self._current[order.identifier] = new_tt_id
            self._orders[new_tt_id] = order
            if new_tt_id == old:  # replaced in place: still the live id
                self._superseded.discard(new_tt_id)

    def apply(self, data: Mapping[str, Any]) -> bool:
        """Apply one order snapshot; True when it changed one of our orders."""
        tt_id = str(data.get("id", ""))
        status = str(data.get("status", ""))
        with self._lock:
            order = self._orders.get(tt_id)
            if order is None:
                external = data.get("external-identifier")
                order = self._tracker.get_tracked_order_by_client_order_id(str(external)) if external else None
                if order is None:
                    return False
                self._adopt(order, tt_id, data)
            changed = self._apply_fills(order, tt_id, data, status)
            if tt_id in self._superseded:
                return changed
            return self._apply_status(order, data, status) or changed

    def _adopt(self, order: Order, tt_id: str, data: Mapping[str, Any]) -> None:
        """Register an order found by external identifier; fills already in its `filled_quantity` count as seen."""
        self._current[order.identifier] = tt_id
        self._orders[tt_id] = order
        reflected = Decimal(0)
        for fill in orders.parse_fills(data):
            if reflected + fill.quantity > order.filled_quantity:
                break
            reflected += fill.quantity
            self._seen_fills.add((tt_id, fill.fill_id))

    def _apply_fills(self, order: Order, tt_id: str, data: Mapping[str, Any], status: str) -> bool:
        if order.notional is not None and status != "Filled":
            return False  # a notional order has no quantity to compare fills against: wait for Filled
        every = orders.parse_fills(data)
        new = [f for f in every if (tt_id, f.fill_id) not in self._seen_fills]
        counted = [f for f in every if f not in new]  # reflected in the order before this snapshot
        self._seen_fills.update((tt_id, f.fill_id) for f in new)
        for index, fill in enumerate(new):
            if not order.is_active():
                # The order already ended: a tracker fill event would move it out of its terminal bucket.
                logger.warning("Fill %s of %s arrived after order %s ended (%s); counted without changing its status", fill.fill_id, fill.quantity, order.identifier, order.status)
                self._update_average(order, fill, counted)
                order.add_transaction(fill.price, fill.quantity)
                continue
            if order.notional is not None:
                complete = index == len(new) - 1
            else:
                complete = order.quantity is not None and order.filled_quantity + fill.quantity >= order.quantity
            event = OrderEvent.FILLED if complete else OrderEvent.PARTIALLY_FILLED
            self._update_average(order, fill, counted)
            self._tracker.process_trade_event(order, event, price=fill.price, filled_quantity=fill.quantity)
        if new:
            order.update_raw(dict(data))
        return bool(new)

    @staticmethod
    def _update_average(order: Order, fill: orders.Fill, counted: list[orders.Fill]) -> None:
        """Fold `fill` into `order.avg_fill_price` (call before it is added to `filled_quantity`).

        An adopted order may already carry an average (parse_order) or only a quantity; in the
        latter case it is rebuilt from the snapshot fills already reflected in that quantity.
        """
        before = order.filled_quantity
        average = order.avg_fill_price
        if before > 0 and average is None:
            basis = sum((f.quantity for f in counted), Decimal(0))
            if basis > 0:
                average = sum((f.quantity * f.price for f in counted), Decimal(0)) / basis
        if before <= 0 or average is None:
            order.avg_fill_price = fill.price
        else:
            order.avg_fill_price = (average * before + fill.price * fill.quantity) / (before + fill.quantity)

    def _apply_status(self, order: Order, data: Mapping[str, Any], status: str) -> bool:
        if status == "Live" and order.status in _NEW_FROM:
            event = OrderEvent.NEW
        elif status in orders.CANCELED_STATUSES and order.is_active():
            event = OrderEvent.CANCELED
        elif status == "Rejected" and order.is_active():
            order.set_error(str(data.get("reject-reason") or "rejected by Tastytrade"))
            event = OrderEvent.ERROR
        elif status == "Cancel Requested" and order.is_active() and order.status is not OrderStatus.CANCELLING:
            order.status = OrderStatus.CANCELLING
            order.update_raw(dict(data))
            return True
        else:
            return False
        order.update_raw(dict(data))
        self._tracker.process_trade_event(order, event)
        return True

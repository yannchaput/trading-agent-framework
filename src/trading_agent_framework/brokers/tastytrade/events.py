"""`TastytradeOrderEvents`: applies Tastytrade order snapshots (account streamer or REST) to `OrderTracker`.

Feeds the tracker only -- never strategy hooks. Snapshots are full order objects, so applying one
twice is harmless: fills are counted once by `fill-id`. Orders are found by Tastytrade order id
(falling back to `external-identifier`). A replace gives the order a new id; the old one is
"superseded" from the moment the replace is sent: its fills still count, its statuses are ignored,
so the original's `Cancelled` never cancels the modified order.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order

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
        self._seen_fills: set[str] = set()

    def register(self, order: Order, tt_id: str, *, snapshot: Mapping[str, Any] | None = None) -> None:
        """Map `order` to its Tastytrade id; `snapshot` marks its fills as already counted (an adopted order)."""
        with self._lock:
            self._current[order.identifier] = tt_id
            self._orders[tt_id] = order
            if snapshot is not None:
                self._seen_fills.update(f.fill_id for f in orders.parse_fills(snapshot))

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
            changed = self._apply_fills(order, data, status)
            if tt_id in self._superseded:
                return changed
            return self._apply_status(order, data, status) or changed

    def _apply_fills(self, order: Order, data: Mapping[str, Any], status: str) -> bool:
        if order.notional is not None and status != "Filled":
            return False  # a notional order has no quantity to compare fills against: wait for Filled
        new = [f for f in orders.parse_fills(data) if f.fill_id not in self._seen_fills]
        self._seen_fills.update(f.fill_id for f in new)
        for index, fill in enumerate(new):
            if order.notional is not None:
                complete = index == len(new) - 1
            else:
                complete = order.quantity is not None and order.filled_quantity + fill.quantity >= order.quantity
            event = OrderEvent.FILLED if complete else OrderEvent.PARTIALLY_FILLED
            self._tracker.process_trade_event(order, event, price=fill.price, filled_quantity=fill.quantity)
        if new:
            order.update_raw(dict(data))
        return bool(new)

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

"""IBKR order-event handlers: `ib_async` order-status, execution and error events -> `OrderTracker`.

They run on `IbkrConnection`'s loop thread and only feed the tracker (never strategy hooks),
exactly like `AlpacaTradeStream`. Fills come from executions (price and quantity per fill);
status updates carry NEW / CANCELED / ERROR only.
"""

from __future__ import annotations

import logging
import threading
from decimal import Decimal
from typing import Any

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order

logger = logging.getLogger(__name__)

# "connection OK"-style notices IB Gateway sends with an error code; never order failures.
INFO_CODES: frozenset[int] = frozenset({2100, 2104, 2106, 2107, 2108, 2119, 2150, 2158})

_NEW_FROM = frozenset({OrderStatus.UNPROCESSED, OrderStatus.SUBMITTED})


class IbkrOrderEvents:
    def __init__(self, tracker: OrderTracker) -> None:
        self._tracker = tracker
        self._applied_exec_ids: set[str] = set()
        self._lock = threading.Lock()

    def register(self, ib: Any) -> None:
        ib.orderStatusEvent += self.on_order_status
        ib.execDetailsEvent += self.on_exec_details
        ib.errorEvent += self.on_error

    def unregister(self, ib: Any) -> None:
        ib.orderStatusEvent -= self.on_order_status
        ib.execDetailsEvent -= self.on_exec_details
        ib.errorEvent -= self.on_error

    def _find(self, order_ref: str) -> Order | None:
        if not order_ref:
            return None
        return self._tracker.get_tracked_order_by_client_order_id(order_ref)

    def on_order_status(self, trade: Any) -> bool:
        event = orders.map_status_event(trade.orderStatus.status)
        order = self._find(trade.order.orderRef)
        if event is None or order is None:
            return False
        if event is OrderEvent.NEW and order.status not in _NEW_FROM:
            return False
        if event in (OrderEvent.ERROR, OrderEvent.CANCELED) and order.status is OrderStatus.UNPROCESSED:
            return False  # _submit_order is still waiting on this order and reports it itself
        if event is OrderEvent.CANCELED and order.status is OrderStatus.CANCELED:
            return False
        if event is OrderEvent.ERROR:
            order.set_error(orders.rejection_message(trade) or "rejected by IBKR")
        order.update_raw(trade)
        self._tracker.process_trade_event(order, event)
        return True

    def on_exec_details(self, trade: Any, fill: Any) -> bool:
        return self.apply_fill(fill)

    def apply_fill(self, fill: Any) -> bool:
        """Apply one execution (live, or replayed by the post-reconnect reconcile) exactly once."""
        execution = fill.execution
        order = self._find(execution.orderRef)
        if order is None:
            return False
        with self._lock:
            if execution.execId in self._applied_exec_ids:
                return False
            self._applied_exec_ids.add(execution.execId)
        price = orders.to_decimal(execution.price) or Decimal(0)
        shares = orders.to_decimal(execution.shares) or Decimal(0)
        cumulative = orders.to_decimal(execution.cumQty) or Decimal(0)
        average = orders.to_decimal(execution.avgPrice)
        if average:
            order.avg_fill_price = average
        complete = order.quantity is not None and cumulative >= order.quantity
        event = OrderEvent.FILLED if complete else OrderEvent.PARTIALLY_FILLED
        self._tracker.process_trade_event(order, event, price=price, filled_quantity=shares)
        return True

    def on_error(self, req_id: int, error_code: int, error_string: str, contract: Any) -> None:
        if error_code in INFO_CODES:
            logger.debug("IB Gateway notice %s: %s", error_code, error_string)
            return
        logger.warning("IB Gateway error %s (request %s): %s", error_code, req_id, error_string)

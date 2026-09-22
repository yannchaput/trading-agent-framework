from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import FakeIB, make_ib_fill, make_ib_trade

from trading_agent_framework.brokers.ibkr.events import IbkrOrderEvents
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order


def _tracked(tracker: OrderTracker, status: OrderStatus = OrderStatus.SUBMITTED) -> Order:
    order = Order("s", Asset("AAPL"), OrderSide.BUY, quantity=Decimal(10), identifier="abc", client_order_id="s:abc")
    tracker.track_unprocessed(order)
    order.status = status
    return order


def test_submitted_moves_the_order_to_new_once() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="PreSubmitted")) is True
    assert events.on_order_status(make_ib_trade(status="Submitted")) is False
    assert order.status is OrderStatus.NEW
    assert tracker.new.snapshot() == [order]


def test_untracked_orders_are_ignored() -> None:
    assert IbkrOrderEvents(OrderTracker()).on_order_status(make_ib_trade(order_ref="other:x")) is False


def test_inactive_records_the_reason_as_an_error() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)

    IbkrOrderEvents(tracker).on_order_status(make_ib_trade(status="Inactive", log_message="margin required", log_error_code=201))

    assert order.status is OrderStatus.ERROR
    assert order.error_message == "margin required (IBKR error 201)"


def test_errors_and_cancels_during_submission_are_left_to_submit_order() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker, status=OrderStatus.UNPROCESSED)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="Inactive")) is False
    assert events.on_order_status(make_ib_trade(status="Cancelled")) is False
    assert order.status is OrderStatus.UNPROCESSED


def test_cancel_is_applied_once() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="Cancelled")) is True
    assert events.on_order_status(make_ib_trade(status="ApiCancelled")) is False
    assert order.status is OrderStatus.CANCELED


def test_partial_then_full_fill() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    events.on_exec_details(make_ib_trade(), make_ib_fill(exec_id="e1", shares=4.0, price=100.0, cum_qty=4.0, avg_price=100.0))
    assert order.status is OrderStatus.PARTIAL_FILL
    events.on_exec_details(make_ib_trade(), make_ib_fill(exec_id="e2", shares=6.0, price=101.0, cum_qty=10.0, avg_price=100.6))

    assert order.status is OrderStatus.FILL
    assert order.filled_quantity == Decimal(10)
    assert [(t.quantity, t.price) for t in order.transactions] == [(Decimal(4), Decimal(100)), (Decimal(6), Decimal(101))]
    assert order.avg_fill_price == Decimal("100.6")


def test_a_replayed_execution_is_ignored() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)
    fill = make_ib_fill(exec_id="e1", shares=4.0, cum_qty=4.0)

    assert events.apply_fill(fill) is True
    assert events.apply_fill(fill) is False
    assert order.filled_quantity == Decimal(4)


def test_informational_codes_are_debug_only(caplog: pytest.LogCaptureFixture) -> None:
    events = IbkrOrderEvents(OrderTracker())

    events.on_error(-1, 2104, "Market data farm connection is OK:usfarm", None)
    events.on_error(12, 201, "Order rejected", None)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == ["IB Gateway error 201 (request 12): Order rejected"]


def test_register_wires_and_unregister_unwires_the_handlers() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)
    ib = FakeIB()

    events.register(ib)
    ib.orderStatusEvent.emit(make_ib_trade(status="Submitted"))
    assert order.status is OrderStatus.NEW

    events.unregister(ib)
    ib.orderStatusEvent.emit(make_ib_trade(status="Cancelled"))
    assert order.status is OrderStatus.NEW

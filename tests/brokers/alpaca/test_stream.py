from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderEventError


def _make_order(identifier: str = "order-1") -> Order:
    return Order(
        strategy_name="momentum",
        asset=Asset(symbol="AAPL"),
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("10"),
        identifier=identifier,
    )


def _order_stub(**overrides: object) -> SimpleNamespace:
    defaults: dict[str, object] = {"id": "order-1", "filled_avg_price": None}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _trade_update(
    event: str,
    order: SimpleNamespace | None = None,
    price: object = None,
    qty: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        event=event,
        order=order if order is not None else _order_stub(),
        price=price,
        qty=qty,
    )


# --- Task 15: handle_trade_update -------------------------------------------


# Test 80
def test_handle_trade_update_fill_moves_order_to_filled_and_sets_avg_price() -> None:
    tracker = OrderTracker()
    order = _make_order()
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(
        event="fill",
        order=_order_stub(id="order-1", filled_avg_price="151.23"),
        price="151.20",
        qty="10",
    )

    result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is True
    assert order in tracker.filled.snapshot()
    assert order.avg_fill_price == Decimal("151.23")


# Test 81
def test_handle_trade_update_untracked_order_returns_false_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracker = OrderTracker()
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(event="fill", order=_order_stub(id="unknown"), price="10", qty="1")

    with caplog.at_level(logging.DEBUG, logger="trading_agent_framework.brokers.alpaca.stream"):
        result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is False
    assert any("untracked order" in record.message for record in caplog.records)
    assert tracker.get_all_tracked_orders() == []


def test_handle_trade_update_falls_back_to_client_order_id_and_promotes_identifier() -> None:
    tracker = OrderTracker()
    order = _make_order(identifier="local-uuid")
    order.client_order_id = "momentum:local-uuid"
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(
        event="new",
        order=_order_stub(id="broker-order-1", client_order_id="momentum:local-uuid"),
    )

    result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is True
    assert order.identifier == "broker-order-1"
    assert order in tracker.new.snapshot()
    assert tracker.get_tracked_order("broker-order-1") is order


# Test 82
def test_handle_trade_update_partial_fill_records_transaction() -> None:
    tracker = OrderTracker()
    order = _make_order()
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(
        event="partial_fill", order=_order_stub(id="order-1"), price="150.00", qty="4"
    )

    result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is True
    assert order in tracker.partially_filled.snapshot()
    assert len(order.transactions) == 1
    assert order.transactions[0].price == Decimal("150.00")
    assert order.transactions[0].quantity == Decimal("4")


# Test 83
def test_handle_trade_update_ignored_event_returns_false_without_state_change() -> None:
    tracker = OrderTracker()
    order = _make_order()
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(event="restated", order=_order_stub(id="order-1"))

    result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is False
    assert order in tracker.unprocessed.snapshot()


# Test 84
def test_handle_trade_update_unknown_event_returns_false_without_raising() -> None:
    tracker = OrderTracker()
    order = _make_order()
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(event="totally_made_up_event", order=_order_stub(id="order-1"))

    result = asyncio.run(stream.handle_trade_update(trade_update))

    assert result is False
    assert order in tracker.unprocessed.snapshot()


# Test 85
def test_handle_trade_update_fill_missing_price_raises_order_event_error() -> None:
    tracker = OrderTracker()
    order = _make_order()
    tracker.track_unprocessed(order)
    stream = AlpacaTradeStream(tracker)
    trade_update = _trade_update(
        event="fill", order=_order_stub(id="order-1"), price=None, qty="10"
    )

    with pytest.raises(OrderEventError):
        asyncio.run(stream.handle_trade_update(trade_update))


# Test 87
def test_handle_trade_update_is_coroutine_function() -> None:
    stream = AlpacaTradeStream(OrderTracker())

    # inspect.iscoroutinefunction is the forward-compatible spelling (asyncio's version
    # is deprecated since 3.14). For a plain `async def` like handle_trade_update, both
    # agree -- so this still guards what matters: alpaca-py's internal
    # _ensure_coroutine() (stream.py) calls asyncio.iscoroutinefunction(handler) and
    # would reject handle_trade_update the same way if it weren't a coroutine function.
    assert inspect.iscoroutinefunction(stream.handle_trade_update)


# --- Task 16: thread lifecycle -----------------------------------------------


# Test 86
def test_start_subscribes_handler_spawns_daemon_thread_and_is_running() -> None:
    release = threading.Event()
    stream = MagicMock()
    stream.run.side_effect = lambda: release.wait(timeout=2)
    trade_stream = AlpacaTradeStream(OrderTracker(), stream)

    trade_stream.start()
    try:
        stream.subscribe_trade_updates.assert_called_once_with(trade_stream.handle_trade_update)
        assert trade_stream._thread is not None
        assert trade_stream._thread.daemon is True
        assert trade_stream.is_running is True
    finally:
        release.set()
        trade_stream._thread.join(timeout=2)


# Test 88
def test_stop_before_start_does_not_raise() -> None:
    stream = MagicMock()
    stream._loop = None
    trade_stream = AlpacaTradeStream(OrderTracker(), stream)

    trade_stream.stop()

    stream.stop.assert_not_called()


# Test 89
def test_stop_is_idempotent() -> None:
    stream = MagicMock()
    stream._loop = None
    trade_stream = AlpacaTradeStream(OrderTracker(), stream)

    trade_stream.stop()
    trade_stream.stop()

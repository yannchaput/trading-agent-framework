from __future__ import annotations

import threading
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from tests.fakes import FakeTradingClient, make_alpaca_order, make_alpaca_position, make_api_error

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from trading_agent_framework.entities.order import Order

_BROKER_ORDER_ID = "11111111-1111-1111-1111-111111111111"


def _make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "strategy_name": "momentum",
        "asset": Asset(symbol="AAPL"),
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("1"),
        "time_in_force": TimeInForce.DAY,
    }
    defaults.update(overrides)
    return Order(**defaults)


# Test 71
def test_submit_order_conforms_before_building_request() -> None:
    client = FakeTradingClient()
    client.submit_response = make_alpaca_order(id=_BROKER_ORDER_ID)
    broker = AlpacaBroker("momentum", client)
    order = _make_order(
        order_type=OrderType.LIMIT,
        limit_price=Decimal("1.005"),
    )

    broker.submit_order(order)

    assert len(client.submitted) == 1
    assert client.submitted[0].limit_price == 1.01


# Test 72
def test_submit_order_success_updates_identifier_status_raw_and_tracker() -> None:
    client = FakeTradingClient()
    response = make_alpaca_order(id=_BROKER_ORDER_ID, status="new")
    client.submit_response = response
    broker = AlpacaBroker("momentum", client)
    order = _make_order()
    local_identifier = order.identifier

    result = broker.submit_order(order)

    assert result.identifier == _BROKER_ORDER_ID
    assert result.identifier != local_identifier
    assert result.status is OrderStatus.NEW
    assert result.raw is response
    assert result in broker.tracker.unprocessed.snapshot()


# Test 73
def test_submit_order_raises_sets_error_and_tracks_nowhere() -> None:
    client = FakeTradingClient()
    boom = RuntimeError("boom")
    client.raise_on_submit = boom
    broker = AlpacaBroker("momentum", client)
    order = _make_order()
    local_identifier = order.identifier

    with pytest.raises(RuntimeError, match="boom"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert order.error_message == "boom"
    assert broker.tracker.get_tracked_order(local_identifier) is None
    assert all(
        len(bucket) == 0
        for bucket in (
            broker.tracker.unprocessed,
            broker.tracker.new,
            broker.tracker.partially_filled,
            broker.tracker.filled,
            broker.tracker.canceled,
            broker.tracker.error,
        )
    )


# Test 74
def test_cancel_order_calls_cancel_order_by_id_once() -> None:
    client = FakeTradingClient()
    broker = AlpacaBroker("momentum", client)
    order = _make_order()
    order.set_identifier("broker-1")

    broker.cancel_order(order)

    assert client.canceled == ["broker-1"]


# Test 75
def test_pull_orders_builds_get_orders_request_with_status_all_and_limit() -> None:
    client = FakeTradingClient()
    client.orders_response = []
    broker = AlpacaBroker("momentum", client)

    broker.pull_orders(limit=100)

    request = client.last_orders_request
    assert isinstance(request, GetOrdersRequest)
    assert request.status is QueryOrderStatus.ALL
    assert request.limit == 100


# Test 76
def test_pull_orders_returns_parsed_orders_with_broker_strategy_name() -> None:
    client = FakeTradingClient()
    client.orders_response = [make_alpaca_order(), make_alpaca_order()]
    broker = AlpacaBroker("momentum", client)

    results = broker.pull_orders()

    assert len(results) == 2
    assert all(o.strategy_name == "momentum" for o in results)


# Test 77
def test_pull_order_returns_none_on_404_and_reraises_on_500() -> None:
    client = FakeTradingClient()
    broker = AlpacaBroker("momentum", client)

    client.get_order_by_id_raises = make_api_error(404)
    assert broker.pull_order("missing") is None

    client.get_order_by_id_raises = make_api_error(500)
    with pytest.raises(Exception):  # noqa: B017 - alpaca APIError, re-raised as-is
        broker.pull_order("broken")


# Test 78
def test_pull_positions_maps_through_parse_broker_position() -> None:
    client = FakeTradingClient()
    client.positions_response = [make_alpaca_position(side="short")]
    broker = AlpacaBroker("momentum", client)

    positions = broker.pull_positions()

    assert len(positions) == 1
    assert positions[0].strategy_name == "momentum"
    assert positions[0].side is PositionSide.SHORT


# Test 79
def test_submit_orders_submits_each_in_order() -> None:
    client = FakeTradingClient()
    client.submit_response = make_alpaca_order(id=_BROKER_ORDER_ID)
    broker = AlpacaBroker("momentum", client)
    order_a = _make_order()
    order_b = _make_order()

    results = broker.submit_orders([order_a, order_b])

    assert len(results) == 2
    assert len(client.submitted) == 2


# Test 79b
def test_submit_order_stamps_client_order_id_only_when_unset() -> None:
    client = FakeTradingClient()
    client.submit_response = make_alpaca_order(id=_BROKER_ORDER_ID)
    broker = AlpacaBroker("momentum", client)

    unstamped = _make_order()
    local_identifier = unstamped.identifier
    broker.submit_order(unstamped)
    assert unstamped.client_order_id == f"momentum:{local_identifier}"

    preset = _make_order(client_order_id="preset-id")
    broker.submit_order(preset)
    assert preset.client_order_id == "preset-id"


# Task 16: start_stream / stop_stream delegation
def test_start_stream_subscribes_handler_and_starts_thread() -> None:
    client = FakeTradingClient()
    stream = MagicMock()
    release = threading.Event()
    stream.run.side_effect = lambda: release.wait(timeout=2)
    broker = AlpacaBroker("momentum", client, stream=stream)

    broker.start_stream()
    try:
        stream.subscribe_trade_updates.assert_called_once()
    finally:
        release.set()
        broker.stop_stream()


def test_stop_stream_before_start_does_not_raise() -> None:
    client = FakeTradingClient()
    stream = MagicMock()
    stream._loop = None
    broker = AlpacaBroker("momentum", client, stream=stream)

    broker.stop_stream()

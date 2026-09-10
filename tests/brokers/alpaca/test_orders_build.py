from __future__ import annotations

from decimal import Decimal

import alpaca.trading.enums as alpaca_enums
import pytest
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)

from trading_agent_framework.brokers.alpaca.orders import build_order_request
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderValidationError


def make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "strategy_name": "momentum",
        "asset": Asset(symbol="AAPL"),
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
    }
    defaults.update(overrides)
    return Order(**defaults)  # ty: ignore[invalid-argument-type]


# Test 47
def test_build_order_request_market_with_qty() -> None:
    order = make_order(order_type=OrderType.MARKET, quantity=Decimal("10"))
    request = build_order_request(order)
    assert isinstance(request, MarketOrderRequest)
    assert request.qty == 10.0
    assert request.notional is None


# Test 48
def test_build_order_request_market_with_notional() -> None:
    order = make_order(order_type=OrderType.MARKET, quantity=None, notional=Decimal("500"))
    request = build_order_request(order)
    assert request.notional == 500.0
    assert request.qty is None


# Test 49
def test_build_order_request_fractional_quantity_survives() -> None:
    order = make_order(order_type=OrderType.MARKET, quantity=Decimal("0.001"))
    request = build_order_request(order)
    assert request.qty == 0.001


# Test 50
def test_build_order_request_limit() -> None:
    order = make_order(
        order_type=OrderType.LIMIT,
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
    )
    request = build_order_request(order)
    assert isinstance(request, LimitOrderRequest)
    assert request.limit_price == 100.0


# Test 51
def test_build_order_request_stop() -> None:
    order = make_order(
        order_type=OrderType.STOP,
        quantity=Decimal("10"),
        stop_price=Decimal("95"),
    )
    request = build_order_request(order)
    assert isinstance(request, StopOrderRequest)
    assert request.stop_price == 95.0


# Test 52 (load-bearing)
def test_build_order_request_stop_limit_mapping() -> None:
    order = make_order(
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("10"),
        stop_price=Decimal("50"),
        stop_limit_price=Decimal("49.5"),
    )
    request = build_order_request(order)
    assert isinstance(request, StopLimitOrderRequest)
    assert request.stop_price == 50.0
    assert request.limit_price == 49.5


# Test 53
def test_build_order_request_stop_limit_missing_stop_limit_price_raises() -> None:
    order = make_order(
        order_type=OrderType.STOP_LIMIT,
        quantity=Decimal("10"),
        stop_price=Decimal("50"),
    )
    with pytest.raises(OrderValidationError):
        build_order_request(order)


# Test 54
def test_build_order_request_trailing_with_trail_percent_only() -> None:
    order = make_order(
        order_type=OrderType.TRAIL,
        quantity=Decimal("10"),
        trail_percent=Decimal("1.5"),
    )
    request = build_order_request(order)
    assert isinstance(request, TrailingStopOrderRequest)
    assert request.trail_percent == 1.5
    assert request.trail_price is None


# Test 55
def test_build_order_request_trailing_with_both_raises() -> None:
    order = make_order(
        order_type=OrderType.TRAIL,
        quantity=Decimal("10"),
        trail_percent=Decimal("1.5"),
        trail_price=Decimal("1"),
    )
    with pytest.raises(OrderValidationError):
        build_order_request(order)


# Test 56
def test_build_order_request_trailing_with_neither_raises() -> None:
    order = make_order(order_type=OrderType.TRAIL, quantity=Decimal("10"))
    with pytest.raises(OrderValidationError):
        build_order_request(order)


# Test 57
@pytest.mark.parametrize("tif", list(TimeInForce))
def test_build_order_request_time_in_force_parametrized(tif: TimeInForce) -> None:
    order = make_order(order_type=OrderType.MARKET, quantity=Decimal("10"), time_in_force=tif)
    request = build_order_request(order)
    assert request.time_in_force == alpaca_enums.TimeInForce(tif.value)


# Test 58
@pytest.mark.parametrize("side", list(OrderSide))
def test_build_order_request_side_parametrized(side: OrderSide) -> None:
    order = make_order(order_type=OrderType.MARKET, quantity=Decimal("10"), side=side)
    request = build_order_request(order)
    assert request.side == alpaca_enums.OrderSide(side.value)


# Test 59 (lumibot bug #2 falsy regression)
def test_build_order_request_to_request_fields_keeps_falsy_and_drops_none() -> None:
    order = make_order(
        order_type=OrderType.MARKET,
        quantity=Decimal("10"),
        extended_hours=False,
    )
    request = build_order_request(order)
    fields = request.to_request_fields()
    assert "symbol" in fields
    assert "stop_price" not in fields
    assert "extended_hours" in fields
    assert fields["extended_hours"] is False


# Test 60
def test_build_order_request_limit_to_request_fields_key_set() -> None:
    order = make_order(
        order_type=OrderType.LIMIT,
        quantity=Decimal("10"),
        limit_price=Decimal("100"),
        client_order_id="abc-123",
    )
    request = build_order_request(order)
    fields = request.to_request_fields()
    assert set(fields.keys()) == {
        "symbol",
        "qty",
        "side",
        "type",
        "time_in_force",
        "extended_hours",
        "client_order_id",
        "limit_price",
    }

from __future__ import annotations

import logging
from decimal import Decimal

import alpaca.trading.enums as alpaca_enums
import pytest
from tests.fakes import _NOW, make_alpaca_order, make_alpaca_position

from trading_agent_framework.brokers.alpaca.orders import (
    parse_broker_order,
    parse_broker_orders,
    parse_broker_position,
)
from trading_agent_framework.entities.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)


# Test 61
def test_parse_broker_order_full_market_response_maps_every_field() -> None:
    response = make_alpaca_order()
    order = parse_broker_order(response, strategy_name="momentum")

    assert order is not None
    assert order.strategy_name == "momentum"
    assert order.asset.symbol == "AAPL"
    assert order.side is OrderSide.BUY
    assert order.order_type is OrderType.MARKET
    assert order.quantity == Decimal("10")
    assert order.notional is None
    assert order.filled_quantity == Decimal("0")
    assert order.time_in_force is TimeInForce.DAY
    assert order.status is OrderStatus.NEW
    assert order.client_order_id == "cid-1"
    assert order.created_at == _NOW
    assert order.updated_at == _NOW
    assert order.raw is response


# Test 62
def test_parse_broker_order_qty_string_becomes_decimal() -> None:
    response = make_alpaca_order(qty="10")
    order = parse_broker_order(response, strategy_name="momentum")
    assert order is not None
    assert order.quantity == Decimal("10")
    assert isinstance(order.quantity, Decimal)


# Test 63
def test_parse_broker_order_stop_limit_inbound_mapping() -> None:
    response = make_alpaca_order(
        type=alpaca_enums.OrderType.STOP_LIMIT,
        stop_price="50",
        limit_price="49.5",
    )
    order = parse_broker_order(response, strategy_name="momentum")
    assert order is not None
    assert order.order_type is OrderType.STOP_LIMIT
    assert order.stop_price == Decimal("50")
    assert order.stop_limit_price == Decimal("49.5")
    assert order.limit_price is None


# Test 64
def test_parse_broker_order_trailing_stop_type_and_trail_percent_decimal() -> None:
    response = make_alpaca_order(
        type=alpaca_enums.OrderType.TRAILING_STOP,
        trail_percent="1.5",
    )
    order = parse_broker_order(response, strategy_name="momentum")
    assert order is not None
    assert order.order_type is OrderType.TRAIL
    assert order.trail_percent == Decimal("1.5")
    assert isinstance(order.trail_percent, Decimal)


# Test 65
def test_parse_broker_order_returns_none_and_warns_when_qty_and_notional_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = make_alpaca_order(qty=None, notional=None)
    with caplog.at_level(logging.WARNING):
        order = parse_broker_order(response, strategy_name="momentum")
    assert order is None
    assert any("qty" in record.message or "notional" in record.message for record in caplog.records)


# Test 66 (deliberate divergence from lumibot)
def test_parse_broker_order_qty_none_but_notional_set_returns_valid_order() -> None:
    response = make_alpaca_order(qty=None, notional="500")
    order = parse_broker_order(response, strategy_name="momentum")
    assert order is not None
    assert order.quantity is None
    assert order.notional == Decimal("500")


# Test 67
def test_parse_broker_order_dict_response_parses_identically() -> None:
    model_response = make_alpaca_order()
    dict_response = model_response.model_dump()
    order_from_model = parse_broker_order(model_response, strategy_name="momentum")
    order_from_dict = parse_broker_order(dict_response, strategy_name="momentum")

    assert order_from_model is not None
    assert order_from_dict is not None
    assert order_from_dict.asset.symbol == order_from_model.asset.symbol
    assert order_from_dict.quantity == order_from_model.quantity
    assert order_from_dict.side == order_from_model.side
    assert order_from_dict.order_type == order_from_model.order_type
    assert order_from_dict.status == order_from_model.status


# Test 68
def test_parse_broker_orders_skips_nones() -> None:
    good = make_alpaca_order()
    bad = make_alpaca_order(qty=None, notional=None)
    orders = parse_broker_orders([good, bad, good], strategy_name="momentum")
    assert len(orders) == 2


# Test 69
def test_parse_broker_orders_propagates_strategy_name() -> None:
    orders = parse_broker_orders([make_alpaca_order()], strategy_name="my-strategy")
    assert orders[0].strategy_name == "my-strategy"


# Test 70
def test_parse_broker_position_maps_fields_including_short_side() -> None:
    response = make_alpaca_position(side=alpaca_enums.PositionSide.SHORT)
    position = parse_broker_position(response, strategy_name="momentum")

    assert position.strategy_name == "momentum"
    assert position.asset.symbol == "AAPL"
    assert position.quantity == Decimal("10")
    assert position.side is PositionSide.SHORT
    assert position.avg_fill_price == Decimal("100.00")
    assert position.current_price == Decimal("101.00")
    assert position.market_value == Decimal("1010.00")
    assert position.unrealized_pnl == Decimal("10.00")
    assert position.raw is response

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from trading_agent_framework.brokers.alpaca.orders import conform_order
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order


def make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "strategy_name": "momentum",
        "asset": Asset(symbol="AAPL"),
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
    }
    defaults.update(overrides)
    return Order(**defaults)  # ty: ignore[invalid-argument-type]


# Test 38
def test_conform_order_limit_price_already_conformed_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("1.00"))
    with caplog.at_level(logging.WARNING):
        result = conform_order(order)
    assert result.limit_price == Decimal("1.00")
    assert caplog.records == []


# Test 39
def test_conform_order_limit_price_rounds_half_up_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("1.005"))
    with caplog.at_level(logging.WARNING):
        conform_order(order)
    assert order.limit_price == Decimal("1.01")
    assert any(record.levelname == "WARNING" for record in caplog.records)


# Test 40: the load-bearing boundary case
def test_conform_order_limit_price_bucket_from_prerounded_value() -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("0.99999"))
    conform_order(order)
    assert order.limit_price == Decimal("1.0000")


# Test 41
def test_conform_order_limit_price_sub_dollar_rounds_to_four_places() -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("0.123456"))
    conform_order(order)
    assert order.limit_price == Decimal("0.1235")


# Test 42
def test_conform_order_limit_price_unchanged_above_dollar_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("12.34"))
    with caplog.at_level(logging.WARNING):
        conform_order(order)
    assert order.limit_price == Decimal("12.34")
    assert caplog.records == []


# Test 43
def test_conform_order_stop_price_rounds_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    order = make_order(order_type=OrderType.STOP, stop_price=Decimal("50.5678"))
    with caplog.at_level(logging.WARNING):
        conform_order(order)
    assert order.stop_price == Decimal("50.57")
    assert any(record.levelname == "WARNING" for record in caplog.records)


# Test 44
def test_conform_order_stop_limit_rounds_both_prices() -> None:
    order = make_order(
        order_type=OrderType.STOP_LIMIT,
        stop_price=Decimal("50.5678"),
        stop_limit_price=Decimal("51.5678"),
    )
    conform_order(order)
    assert order.stop_price == Decimal("50.57")
    assert order.stop_limit_price == Decimal("51.57")


# Test 45
def test_conform_order_market_order_no_prices_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    order = make_order(order_type=OrderType.MARKET)
    with caplog.at_level(logging.WARNING):
        conform_order(order)
    assert order.limit_price is None
    assert order.stop_price is None
    assert order.stop_limit_price is None
    assert caplog.records == []


# Test 46
def test_conform_order_mutates_in_place_and_returns_same_object() -> None:
    order = make_order(order_type=OrderType.LIMIT, limit_price=Decimal("1.005"))
    result = conform_order(order)
    assert result is order
    assert order.limit_price == Decimal("1.01")

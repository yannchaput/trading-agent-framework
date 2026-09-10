from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers.alpaca.orders import validate_order
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


# Rule 1: notional set and order_type is not MARKET.
def test_validate_order_rejects_notional_on_non_market_order() -> None:
    order = make_order(
        quantity=None,
        notional=Decimal("100"),
        order_type=OrderType.LIMIT,
    )
    with pytest.raises(OrderValidationError, match="notional"):
        validate_order(order)


# Rule 2: fractional quantity and order_type is not MARKET.
def test_validate_order_rejects_fractional_quantity_on_non_market_order() -> None:
    order = make_order(
        quantity=Decimal("1.5"),
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
    )
    with pytest.raises(OrderValidationError, match="quantity"):
        validate_order(order)


# Rule 3: fractional quantity and time_in_force is not DAY.
def test_validate_order_rejects_fractional_quantity_with_non_day_tif() -> None:
    order = make_order(
        quantity=Decimal("1.5"),
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
    )
    with pytest.raises(OrderValidationError, match="quantity"):
        validate_order(order)


# Rule 4: time_in_force in {OPG, CLS} and order_type not in {MARKET, LIMIT}.
def test_validate_order_rejects_opg_cls_tif_with_non_market_limit_order_type() -> None:
    order = make_order(
        quantity=Decimal("1"),
        order_type=OrderType.STOP,
        time_in_force=TimeInForce.OPG,
    )
    with pytest.raises(OrderValidationError, match="time_in_force"):
        validate_order(order)


def test_validate_order_passes_for_plain_market_day_whole_quantity_order() -> None:
    order = make_order(
        quantity=Decimal("10"),
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY,
    )
    validate_order(order)  # must not raise

from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

AAPL = Asset("AAPL")


def _order(**overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": Decimal(10)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def test_contract_is_a_smart_routed_usd_stock() -> None:
    contract = orders.build_contract(AAPL)

    assert (contract.secType, contract.symbol, contract.exchange, contract.currency) == ("STK", "AAPL", "SMART", "USD")


def test_client_order_id_prefixes_the_strategy() -> None:
    assert orders.client_order_id_for("news_binary", "abc") == "news_binary:abc"


def test_conform_rejects_notional_orders() -> None:
    with pytest.raises(OrderValidationError, match="notional"):
        orders.conform_order(_order(quantity=None, notional=Decimal(100)))


def test_conform_floors_fractional_quantities(caplog: pytest.LogCaptureFixture) -> None:
    order = orders.conform_order(_order(quantity=Decimal("3.7")))

    assert order.quantity == Decimal(3)
    assert "whole shares" in caplog.text


@pytest.mark.parametrize("quantity", [Decimal("0.4"), Decimal(0), Decimal(-1), None])
def test_conform_rejects_quantities_below_one_share(quantity: Decimal | None) -> None:
    with pytest.raises(OrderValidationError):
        orders.conform_order(_order(quantity=quantity))


def test_market_day_order() -> None:
    ib_order = orders.build_order(_order(client_order_id="s:1"))

    assert (ib_order.action, ib_order.totalQuantity, ib_order.orderType, ib_order.tif) == ("BUY", 10.0, "MKT", "DAY")
    assert ib_order.orderRef == "s:1"
    assert ib_order.outsideRth is False
    assert ib_order.transmit is True


def test_limit_sell_gtc_outside_regular_hours() -> None:
    ib_order = orders.build_order(
        _order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=Decimal("101.25"), time_in_force=TimeInForce.GTC, extended_hours=True)
    )

    assert (ib_order.action, ib_order.orderType, ib_order.lmtPrice, ib_order.tif, ib_order.outsideRth) == ("SELL", "LMT", 101.25, "GTC", True)


def test_stop_and_stop_limit() -> None:
    stop = orders.build_order(_order(order_type=OrderType.STOP, stop_price=Decimal(95)))
    stop_limit = orders.build_order(_order(order_type=OrderType.STOP_LIMIT, stop_price=Decimal(95), stop_limit_price=Decimal("94.5")))

    assert (stop.orderType, stop.auxPrice) == ("STP", 95.0)
    assert (stop_limit.orderType, stop_limit.auxPrice, stop_limit.lmtPrice) == ("STP LMT", 95.0, 94.5)


def test_trailing_stop_by_amount_or_percent() -> None:
    by_amount = orders.build_order(_order(order_type=OrderType.TRAIL, trail_price=Decimal(2)))
    by_percent = orders.build_order(_order(order_type=OrderType.TRAIL, trail_percent=Decimal("1.5")))

    assert (by_amount.orderType, by_amount.auxPrice) == ("TRAIL", 2.0)
    assert (by_percent.orderType, by_percent.trailingPercent) == ("TRAIL", 1.5)


@pytest.mark.parametrize(("order_type", "expected"), [(OrderType.MARKET, "MOC"), (OrderType.LIMIT, "LOC")])
def test_at_the_close_orders(order_type: OrderType, expected: str) -> None:
    ib_order = orders.build_order(_order(order_type=order_type, limit_price=Decimal(100), time_in_force=TimeInForce.CLS))

    assert (ib_order.orderType, ib_order.tif) == (expected, "DAY")


@pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.FOK, TimeInForce.OPG])
def test_other_time_in_force_values_map_directly(tif: TimeInForce) -> None:
    assert orders.build_order(_order(time_in_force=tif)).tif == tif.value.upper()


@pytest.mark.parametrize(
    "overrides",
    [
        {"order_type": OrderType.LIMIT},
        {"order_type": OrderType.STOP},
        {"order_type": OrderType.STOP_LIMIT, "stop_price": Decimal(1)},
        {"order_type": OrderType.TRAIL},
        {"order_type": OrderType.STOP, "stop_price": Decimal(1), "time_in_force": TimeInForce.CLS},
    ],
)
def test_missing_prices_and_unsupported_combinations_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(OrderValidationError):
        orders.build_order(_order(**overrides))

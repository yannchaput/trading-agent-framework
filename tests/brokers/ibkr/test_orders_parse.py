from __future__ import annotations

import math
from decimal import Decimal

import ib_async
import pytest
from tests.fakes import make_ib_portfolio_item, make_ib_trade

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PendingSubmit", OrderStatus.NEW),
        ("PreSubmitted", OrderStatus.NEW),
        ("Submitted", OrderStatus.NEW),
        ("Filled", OrderStatus.FILL),
        ("Cancelled", OrderStatus.CANCELED),
        ("ApiCancelled", OrderStatus.CANCELED),
        ("Inactive", OrderStatus.ERROR),
        ("PendingCancel", OrderStatus.CANCELLING),
        ("ApiPending", OrderStatus.UNKNOWN),
    ],
)
def test_map_status(status: str, expected: OrderStatus) -> None:
    assert orders.map_status(status) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PreSubmitted", OrderEvent.NEW),
        ("Submitted", OrderEvent.NEW),
        ("Cancelled", OrderEvent.CANCELED),
        ("ApiCancelled", OrderEvent.CANCELED),
        ("Inactive", OrderEvent.ERROR),
        ("Filled", None),  # fills come from executions, with price and quantity
        ("PendingSubmit", None),
        ("PendingCancel", None),
    ],
)
def test_map_status_event(status: str, expected: OrderEvent | None) -> None:
    assert orders.map_status_event(status) is expected


def test_identifier_from_order_ref() -> None:
    assert orders.identifier_from_order_ref("news_binary:abc", "news_binary") == "abc"
    assert orders.identifier_from_order_ref("other:abc", "news_binary") is None
    assert orders.identifier_from_order_ref("news_binary:", "news_binary") is None
    assert orders.identifier_from_order_ref("", "news_binary") is None


@pytest.mark.parametrize("value", [None, math.nan, ib_async.util.UNSET_DOUBLE])
def test_to_decimal_treats_unset_values_as_none(value: float | None) -> None:
    assert orders.to_decimal(value) is None


def test_to_decimal_is_exact_for_the_printed_float() -> None:
    assert orders.to_decimal(101.25) == Decimal("101.25")


def test_parse_trade_for_this_strategy() -> None:
    trade = make_ib_trade(order_ref="s:abc", order_type="LMT", lmt_price=99.5, tif="GTC", status="Submitted", filled=4.0, avg_fill_price=99.4)

    order = orders.parse_trade(trade, "s")

    assert order.identifier == "abc"
    assert order.client_order_id == "s:abc"
    assert (order.asset, order.side, order.order_type, order.time_in_force) == (Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, TimeInForce.GTC)
    assert (order.quantity, order.limit_price, order.filled_quantity, order.avg_fill_price) == (Decimal(10), Decimal("99.5"), Decimal(4), Decimal("99.4"))
    assert order.status is OrderStatus.NEW
    assert order.raw is trade


def test_parse_trade_of_another_client_uses_the_perm_id() -> None:
    order = orders.parse_trade(make_ib_trade(order_ref="", perm_id=555), "s")

    assert order.identifier == "ibkr:555"
    assert order.client_order_id is None


def test_parse_trade_at_the_close_and_stop_limit() -> None:
    moc = orders.parse_trade(make_ib_trade(order_type="MOC"), "s")
    stop_limit = orders.parse_trade(make_ib_trade(order_type="STP LMT", aux_price=95.0, lmt_price=94.5), "s")

    assert (moc.order_type, moc.time_in_force) == (OrderType.MARKET, TimeInForce.CLS)
    assert (stop_limit.order_type, stop_limit.stop_price, stop_limit.stop_limit_price) == (OrderType.STOP_LIMIT, Decimal(95), Decimal("94.5"))


def test_parse_portfolio_item() -> None:
    position = orders.parse_portfolio_item(make_ib_portfolio_item(), "s")

    assert position is not None
    assert (position.asset, position.quantity, position.side) == (Asset("AAPL"), Decimal(10), PositionSide.LONG)
    assert (position.avg_fill_price, position.current_price, position.market_value, position.unrealized_pnl) == (
        Decimal(100), Decimal(101), Decimal(1010), Decimal(10)
    )


def test_a_flat_portfolio_item_is_no_position() -> None:
    assert orders.parse_portfolio_item(make_ib_portfolio_item(position=0.0), "s") is None


def test_rejection_message() -> None:
    rejected = make_ib_trade(status="Inactive", log_message="Order rejected - reason: no trading permissions", log_error_code=201)

    assert orders.rejection_message(rejected) == "Order rejected - reason: no trading permissions (IBKR error 201)"
    assert orders.rejection_message(make_ib_trade(status="Submitted")) is None
    assert orders.rejection_message(make_ib_trade(status="Cancelled")) == "order Cancelled by IBKR"

from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from tests.fakes import make_tt_fill, make_tt_order, make_tt_position

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.utils.errors import BrokerError

D = Decimal


@pytest.mark.parametrize(
    ("status", "has_fills", "expected"),
    [
        ("Received", False, OrderStatus.SUBMITTED),
        ("Routed", False, OrderStatus.SUBMITTED),
        ("In Flight", False, OrderStatus.SUBMITTED),
        ("Contingent", False, OrderStatus.SUBMITTED),
        ("Live", False, OrderStatus.NEW),
        ("Live", True, OrderStatus.PARTIAL_FILL),
        ("Replace Requested", False, OrderStatus.NEW),
        ("Cancel Requested", False, OrderStatus.CANCELLING),
        ("Filled", True, OrderStatus.FILL),
        ("Cancelled", False, OrderStatus.CANCELED),
        ("Removed", False, OrderStatus.CANCELED),
        ("Partially Removed", True, OrderStatus.CANCELED),
        ("Expired", False, OrderStatus.EXPIRED),
        ("Rejected", False, OrderStatus.ERROR),
    ],
)
def test_status_map(status: str, has_fills: bool, expected: OrderStatus) -> None:
    assert orders.map_status(status, has_fills=has_fills) is expected


def test_an_unknown_status_is_unknown_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert orders.map_status("Teleported", has_fills=False) is OrderStatus.UNKNOWN
    assert "Teleported" in caplog.text


def test_parse_a_live_limit_buy_of_ours() -> None:
    order = orders.parse_order(make_tt_order(), "s")

    assert (order.identifier, order.client_order_id) == ("abc", "s:abc")
    assert (order.asset, order.side, order.order_type, order.time_in_force) == (Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, TimeInForce.DAY)
    assert (order.quantity, order.limit_price, order.status) == (D(10), D(150), OrderStatus.NEW)
    assert order.raw["id"] == 1001


def test_another_strategys_order_is_identified_by_its_tastytrade_id() -> None:
    assert orders.parse_order(make_tt_order(external="other:xyz"), "s").identifier == "1001"
    assert orders.parse_order(make_tt_order(external=None), "s").client_order_id is None


def test_parse_a_stop_limit_sell() -> None:
    order = orders.parse_order(make_tt_order(action="Sell to Close", **{"order-type": "Stop Limit", "price": "149.5", "stop-trigger": "150"}), "s")

    assert (order.side, order.order_type, order.stop_price, order.stop_limit_price, order.limit_price) == (OrderSide.SELL, OrderType.STOP_LIMIT, D(150), D("149.5"), None)


def test_parse_a_notional_order() -> None:
    order = orders.parse_order(make_tt_order(quantity=None, **{"order-type": "Notional Market", "value": "250.0", "value-effect": "Debit"}), "s")

    assert (order.order_type, order.notional, order.quantity) == (OrderType.MARKET, D(250), None)


def test_fills_set_the_filled_quantity_and_average_price() -> None:
    order = orders.parse_order(make_tt_order(status="Filled", fills=[make_tt_fill("f1", "4", "150"), make_tt_fill("f2", "6", "151")]), "s")

    assert (order.filled_quantity, order.avg_fill_price, order.status) == (D(10), D("150.6"), OrderStatus.FILL)


def test_a_rejected_order_carries_its_reason() -> None:
    order = orders.parse_order(make_tt_order(status="Rejected", **{"reject-reason": "Insufficient buying power"}), "s")

    assert (order.status, order.error_message) == (OrderStatus.ERROR, "Insufficient buying power")


def test_an_extended_hours_order() -> None:
    order = orders.parse_order(make_tt_order(**{"time-in-force": "Ext"}), "s")

    assert (order.time_in_force, order.extended_hours) == (TimeInForce.DAY, True)


def test_parse_order_and_position_map_class_share_symbols_back() -> None:
    assert orders.parse_order(make_tt_order(symbol="BRK/B"), "s").asset == Asset("BRK.B")
    position = orders.parse_position(make_tt_position(symbol="BRK/B"), "s")
    assert position is not None
    assert position.asset == Asset("BRK.B")


def test_unsupported_orders_are_not_parsable() -> None:
    assert orders.is_parsable_order(make_tt_order())
    assert not orders.is_parsable_order(make_tt_order(**{"order-type": "Marketable Limit"}))
    option = make_tt_order()
    option["legs"][0]["instrument-type"] = "Equity Option"
    assert not orders.is_parsable_order(option)
    with pytest.raises(BrokerError, match="Marketable Limit"):
        orders.parse_order(make_tt_order(**{"order-type": "Marketable Limit"}), "s")


def test_parse_fills_reads_every_leg_fill_and_rejects_a_fill_without_an_id() -> None:
    assert orders.parse_fills(make_tt_order(fills=[make_tt_fill("f1", "4", "150.5")])) == [orders.Fill("f1", D(4), D("150.5"))]
    broken = make_tt_order(fills=[{"quantity": "1", "fill-price": "1"}])
    with pytest.raises(BrokerError, match="fill-id"):
        orders.parse_fills(broken)


def test_parse_a_fractional_long_position() -> None:
    position = orders.parse_position(make_tt_position(quantity="10.5", average_open_price="150.25"), "s")

    assert position is not None
    assert (position.strategy_name, position.quantity, position.side, position.avg_fill_price) == ("s", D("10.5"), PositionSide.LONG, D("150.25"))


def test_a_short_position_has_a_positive_quantity() -> None:
    position = orders.parse_position(make_tt_position(quantity="5", direction="Short"), "s")

    assert position is not None
    assert (position.quantity, position.side) == (D(5), PositionSide.SHORT)


def test_closed_and_non_equity_positions_are_dropped() -> None:
    assert orders.parse_position(make_tt_position(quantity="0", direction="Zero"), "s") is None
    assert orders.parse_position(make_tt_position(instrument_type="Equity Option"), "s") is None

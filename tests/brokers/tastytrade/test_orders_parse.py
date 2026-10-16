from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from tests.fakes import make_tt_fill, make_tt_order, make_tt_position

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import ACTIVE_ORDER_STATUSES, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
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


# --- strict parsing (Task 5 review) ---------------------------------------------------------


def test_every_terminal_status_maps_to_a_terminal_order_status() -> None:
    for status in orders.TERMINAL_STATUSES:
        assert orders.map_status(status, has_fills=True) not in ACTIVE_ORDER_STATUSES, status
    assert orders.CANCELED_STATUSES <= orders.TERMINAL_STATUSES


def test_an_option_leg_never_parses_into_a_stock_order() -> None:
    option = make_tt_order(order_id=77)
    option["legs"][0]["instrument-type"] = "Equity Option"
    with pytest.raises(BrokerError, match="77"):
        orders.parse_order(option, "s")


def test_a_two_leg_order_never_parses() -> None:
    spread = make_tt_order(order_id=78)
    spread["legs"].append(dict(spread["legs"][0]))
    assert not orders.is_parsable_order(spread)
    with pytest.raises(BrokerError, match="78"):
        orders.parse_order(spread, "s")


@pytest.mark.parametrize(
    ("action", "side"),
    [("Buy to Open", OrderSide.BUY), ("Buy to Close", OrderSide.BUY), ("Sell to Open", OrderSide.SELL), ("Sell to Close", OrderSide.SELL)],
)
def test_the_four_actions_map_to_a_side(action: str, side: OrderSide) -> None:
    assert orders.parse_order(make_tt_order(action=action), "s").side is side


@pytest.mark.parametrize("action", [None, "", "Hold", "buy to open", 3])
def test_an_unknown_action_raises_instead_of_failing_open_to_sell(action: object) -> None:
    order = make_tt_order()
    order["legs"][0]["action"] = action
    with pytest.raises(BrokerError, match="action"):
        orders.parse_order(order, "s")


@pytest.mark.parametrize("direction", [None, "", "Sideways", "long"])
def test_an_unknown_quantity_direction_raises_instead_of_failing_open_to_long(direction: object) -> None:
    position = make_tt_position()
    position["quantity-direction"] = direction
    with pytest.raises(BrokerError, match="quantity-direction"):
        orders.parse_position(position, "s")


def test_a_position_without_a_direction_key_raises() -> None:
    position = make_tt_position()
    del position["quantity-direction"]
    with pytest.raises(BrokerError, match="quantity-direction"):
        orders.parse_position(position, "s")


def test_an_unknown_time_in_force_falls_back_to_day_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        order = orders.parse_order(make_tt_order(**{"time-in-force": "Moon"}), "s")
    assert (order.time_in_force, order.extended_hours) == (TimeInForce.DAY, False)
    assert "Moon" in caplog.text


def test_a_filled_snapshot_with_only_zero_quantity_fills_has_no_average_price() -> None:
    order = orders.parse_order(make_tt_order(status="Filled", fills=[make_tt_fill("f1", "0", "150")]), "s")

    assert (order.filled_quantity, order.avg_fill_price) == (D(0), None)


def test_a_null_fill_entry_raises_a_broker_error() -> None:
    with pytest.raises(BrokerError, match="1001"):
        orders.parse_fills(make_tt_order(fills=[None]))  # type: ignore[list-item]


def test_a_non_dict_leg_is_not_parsable_and_raises_a_broker_error() -> None:
    broken = make_tt_order()
    broken["legs"] = ["nonsense"]
    assert not orders.is_parsable_order(broken)
    with pytest.raises(BrokerError, match="1001"):
        orders.parse_order(broken, "s")
    with pytest.raises(BrokerError, match="1001"):
        orders.parse_fills(broken)


def test_a_non_string_external_identifier_is_treated_as_absent() -> None:
    order = orders.parse_order(make_tt_order(external=None, **{"external-identifier": 12345}), "s")

    assert (order.identifier, order.client_order_id) == ("1001", None)

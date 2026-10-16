from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from trading_agent_framework.brokers.tastytrade import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

D = Decimal
AAPL = Asset("AAPL")


def _order(quantity: str | None = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": D(quantity) if quantity is not None else None, "client_order_id": "s:abc"}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def _conform(order: Order, *, eligible: bool = False, last_price: str | None = None) -> Order:
    return orders.conform_order(order, fractional_eligible=eligible, last_price=D(last_price) if last_price is not None else None)


# --- identifiers and symbols -------------------------------------------------------------


def test_external_identifier_round_trip() -> None:
    external = orders.external_identifier_for("s", "abc")

    assert external == "s:abc"
    assert orders.identifier_from_external(external, "s") == "abc"
    assert orders.identifier_from_external(external, "other") is None
    assert orders.identifier_from_external("s:", "s") is None
    assert orders.identifier_from_external(None, "s") is None


def test_class_shares_use_tastytrades_slash_symbol() -> None:
    assert orders.to_tastytrade_symbol("BRK.B") == "BRK/B"
    assert orders.from_tastytrade_symbol("BRK/B") == "BRK.B"
    assert orders.build_order_json(_order(asset=Asset("BRK.B")))["legs"][0]["symbol"] == "BRK/B"


# --- conform: unsupported orders -------------------------------------------------------------


def test_trailing_stops_are_rejected() -> None:
    with pytest.raises(OrderValidationError, match="no trailing stops"):
        _conform(_order(order_type=OrderType.TRAIL, trail_percent=D(5)), eligible=True)


@pytest.mark.parametrize("tif", [TimeInForce.OPG, TimeInForce.CLS, TimeInForce.FOK])
def test_unsupported_time_in_force_is_rejected(tif: TimeInForce) -> None:
    with pytest.raises(OrderValidationError, match="time_in_force"):
        _conform(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=tif))


def test_a_gtc_market_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="GTC market"):
        _conform(_order(time_in_force=TimeInForce.GTC))


def test_an_extended_hours_ioc_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="extended-hours"):
        _conform(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=TimeInForce.IOC, extended_hours=True))


# --- conform: fractional quantities ------------------------------------------------------------


def test_a_whole_quantity_is_unchanged() -> None:
    assert _conform(_order("10")).quantity == D(10)


def test_an_eligible_fractional_market_order_keeps_its_fraction_floored_to_the_precision() -> None:
    assert _conform(_order("2.123456"), eligible=True).quantity == D("2.12345")


def test_a_non_eligible_fractional_market_order_is_floored_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        conformed = _conform(_order("2.7"))

    assert conformed.quantity == D(2)
    assert "whole shares" in caplog.text


def test_a_non_eligible_fraction_below_one_share_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="below one whole share"):
        _conform(_order("0.5"))


@pytest.mark.parametrize("eligible", [True, False])
@pytest.mark.parametrize(
    ("order_type", "prices"),
    [
        (OrderType.LIMIT, {"limit_price": D(150)}),
        (OrderType.STOP, {"stop_price": D(140)}),
        (OrderType.STOP_LIMIT, {"stop_price": D(140), "stop_limit_price": D(139)}),
    ],
)
def test_a_fractional_non_market_order_is_rejected(eligible: bool, order_type: OrderType, prices: dict[str, Decimal]) -> None:
    with pytest.raises(OrderValidationError, match="only supported for MARKET"):
        _conform(_order("4.57", order_type=order_type, **prices), eligible=eligible)


# --- conform: notional ---------------------------------------------------------------------------


def test_an_eligible_notional_market_order_is_unchanged() -> None:
    conformed = _conform(_order(None, notional=D(250)), eligible=True)

    assert (conformed.notional, conformed.quantity) == (D(250), None)


def test_a_non_eligible_notional_order_becomes_whole_shares_at_the_last_price(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        conformed = _conform(_order(None, notional=D(1000)), last_price="150")

    assert (conformed.quantity, conformed.notional) == (D(6), None)
    assert "whole shares" in caplog.text


def test_a_non_eligible_notional_below_one_share_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="below one whole share"):
        _conform(_order(None, notional=D(100)), last_price="150")


def test_a_non_eligible_notional_without_a_price_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="no last price"):
        _conform(_order(None, notional=D(1000)))


def test_a_notional_limit_order_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="notional is only supported for MARKET"):
        _conform(_order(None, notional=D(1000), order_type=OrderType.LIMIT, limit_price=D(150)), eligible=True)


# --- order JSON ----------------------------------------------------------------------------------


def test_a_market_buy() -> None:
    assert orders.build_order_json(_order()) == {
        "time-in-force": "Day",
        "source": "trading-agent-framework",
        "external-identifier": "s:abc",
        "order-type": "Market",
        "legs": [{"instrument-type": "Equity", "symbol": "AAPL", "action": "Buy to Open", "quantity": "10"}],
    }


def test_a_limit_sell_rounds_its_price_and_takes_a_credit() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=D("150.126")))

    assert (body["order-type"], body["price"], body["price-effect"]) == ("Limit", "150.13", "Credit")
    assert body["legs"][0]["action"] == "Sell to Close"


def test_a_price_below_one_dollar_keeps_four_decimals() -> None:
    assert orders.round_price(D("0.12345")) == D("0.1235")


def test_a_stop_order_sends_only_a_trigger() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=D(140)))

    assert (body["order-type"], body["stop-trigger"]) == ("Stop", "140.00")
    assert "price" not in body
    assert "price-effect" not in body


def test_a_stop_limit_order_uses_the_stop_limit_price_as_its_price() -> None:
    body = orders.build_order_json(_order(side=OrderSide.SELL, order_type=OrderType.STOP_LIMIT, stop_price=D(140), stop_limit_price=D(139)))

    assert (body["order-type"], body["stop-trigger"], body["price"], body["price-effect"]) == ("Stop Limit", "140.00", "139.00", "Credit")


def test_a_notional_order_sends_a_value_and_no_leg_quantity() -> None:
    body = orders.build_order_json(_order(None, notional=D("250.009")))

    assert (body["order-type"], body["value"], body["value-effect"]) == ("Notional Market", "250.00", "Debit")
    assert "quantity" not in body["legs"][0]


def test_a_fractional_quantity_is_sent_as_a_plain_decimal_string() -> None:
    assert orders.build_order_json(_order("4.57"))["legs"][0]["quantity"] == "4.57"


@pytest.mark.parametrize(("tif", "expected"), [(TimeInForce.DAY, "Ext"), (TimeInForce.GTC, "GTC Ext")])
def test_extended_hours_time_in_force(tif: TimeInForce, expected: str) -> None:
    body = orders.build_order_json(_order(order_type=OrderType.LIMIT, limit_price=D(150), time_in_force=tif, extended_hours=True))

    assert body["time-in-force"] == expected


def test_a_limit_order_without_a_limit_price_is_rejected() -> None:
    with pytest.raises(OrderValidationError, match="needs limit_price"):
        orders.build_order_json(_order(order_type=OrderType.LIMIT))


def test_an_order_without_a_client_order_id_sends_no_external_identifier() -> None:
    assert "external-identifier" not in orders.build_order_json(_order(client_order_id=None))


def test_a_replace_body_drops_the_legs_and_keeps_prices_and_identifier() -> None:
    body = orders.build_replace_json(_order(order_type=OrderType.LIMIT, limit_price=D(151)))

    assert "legs" not in body
    assert (body["price"], body["external-identifier"]) == ("151.00", "s:abc")

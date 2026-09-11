from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from alpaca.trading.enums import QueryOrderStatus
from tests.fakes import (
    make_alpaca_order,
    make_close_position_response,
    make_failed_close_details,
)

from trading_agent_framework.brokers.alpaca import orders
from trading_agent_framework.errors import OrderValidationError

_ORDER_ID = "22222222-2222-2222-2222-222222222222"


def test_get_orders_request_defaults_to_all_statuses() -> None:
    request = orders.build_get_orders_request(50)
    assert request.status == QueryOrderStatus.ALL
    assert request.limit == 50


def test_get_orders_request_can_ask_for_open_orders_only() -> None:
    request = orders.build_get_orders_request(500, open_only=True)
    assert request.status == QueryOrderStatus.OPEN
    assert request.limit == 500


def test_replace_request_rounds_prices_to_alpaca_ticks() -> None:
    request = orders.build_replace_order_request(
        limit_price=Decimal("101.005"), stop_price=Decimal("99.999")
    )
    assert request.limit_price == 101.01
    assert request.stop_price == 100.0


def test_replace_request_needs_a_price() -> None:
    with pytest.raises(OrderValidationError, match="limit_price"):
        orders.build_replace_order_request()


def test_replace_request_wraps_sdk_validation_errors() -> None:
    with pytest.raises(OrderValidationError):
        orders.build_replace_order_request(stop_price=Decimal("-1"))


@pytest.mark.parametrize(
    ("fraction", "percentage"),
    [(Decimal(1), "100.000000000"), (Decimal("0.25"), "25.000000000")],
)
def test_close_position_request_uses_a_percentage(fraction: Decimal, percentage: str) -> None:
    request = orders.build_close_position_request(fraction)
    assert request.percentage == percentage
    assert request.qty is None


@pytest.mark.parametrize("fraction", [Decimal(0), Decimal("-0.5"), Decimal("1.5")])
def test_close_position_request_rejects_fractions_outside_zero_one(fraction: Decimal) -> None:
    with pytest.raises(OrderValidationError, match="fraction"):
        orders.build_close_position_request(fraction)


def test_close_position_request_bounds_percentage_to_nine_decimal_places() -> None:
    """Alpaca's `percentage` field caps at 9dp; a repeating fraction must be quantized,
    not passed through with arbitrary Decimal precision."""
    request = orders.build_close_position_request(Decimal(1) / Decimal(3))
    assert request.percentage is not None
    whole, _, decimals = request.percentage.partition(".")
    assert whole == "33"
    assert len(decimals) <= 9


def test_parse_close_all_responses_keeps_orders_and_logs_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = [
        make_close_position_response(make_alpaca_order(id=_ORDER_ID, symbol="AAPL"), "AAPL"),
        make_close_position_response(make_failed_close_details("TSLA"), "TSLA"),
    ]
    with caplog.at_level(logging.WARNING):
        closed = orders.parse_close_all_responses(responses, "momentum")
    assert [order.identifier for order in closed] == [_ORDER_ID]
    assert closed[0].strategy_name == "momentum"
    assert "TSLA" in caplog.text

from __future__ import annotations

import json

import pytest

from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1.23456789, 6, 1.234567),  # floored, never rounded up
        (1.9999999, 6, 1.999999),
        (5.0, 6, 5.0),
        (0.0000009, 6, 0.0),  # below the precision: nothing to buy
        (12.345, 2, 12.34),
        (-1.5, 6, -1.5),
    ],
)
def test_fractional_qty_floors_to_the_given_number_of_decimals(value: float, decimals: int, expected: float) -> None:
    assert fractional_qty(value, decimals) == expected


def test_fractional_qty_defaults_to_six_decimals() -> None:
    assert fractional_qty(2.0000019) == 2.000001


def test_the_cost_of_a_floored_quantity_never_exceeds_the_budget() -> None:
    price, budget = 33.3333, 1000.0

    assert fractional_qty(budget / price) * price <= budget


def test_parse_insufficient_buying_power_reads_the_brokers_real_figure() -> None:
    error = Exception(json.dumps({"buying_power": "132.45", "code": 40310000, "message": "insufficient buying power"}))

    assert parse_insufficient_buying_power(error) == 132.45


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps(["a", "list"]),
        json.dumps({"message": "something else", "buying_power": "5"}),
        json.dumps({"message": "insufficient buying power"}),
        json.dumps({"message": "insufficient buying power", "buying_power": "n/a"}),
    ],
)
def test_parse_insufficient_buying_power_is_none_for_any_other_error(body: str) -> None:
    assert parse_insufficient_buying_power(Exception(body)) is None

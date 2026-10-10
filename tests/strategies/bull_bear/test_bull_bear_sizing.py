from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.sizing import capped_inverse_volatility


def _size(volatilities: dict[str, float]) -> dict[str, float]:
    return capped_inverse_volatility(volatilities, total=0.98, min_weight=0.04, max_weight=0.20)


def test_equal_volatilities_share_the_investable_money_equally() -> None:
    weights = _size({symbol: 0.3 for symbol in "ABCDE"})

    assert weights == pytest.approx({symbol: 0.196 for symbol in "ABCDE"})


def test_a_very_calm_stock_is_capped_and_the_rest_share_the_remainder() -> None:
    weights = _size({"A": 0.05, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5})

    assert weights["A"] == 0.20
    assert [weights[s] for s in "BCDE"] == pytest.approx([0.195] * 4)


def test_a_very_volatile_stock_is_floored() -> None:
    volatilities = {f"S{i}": 0.2 for i in range(9)} | {"WILD": 5.0}

    weights = _size(volatilities)

    assert weights["WILD"] == 0.04
    assert [weights[f"S{i}"] for i in range(9)] == pytest.approx([0.94 / 9] * 9)


def test_the_weights_sum_to_the_total_within_the_bounds_and_keep_the_input_order() -> None:
    volatilities = {"Z": 0.9, "A": 0.15, "M": 0.4, "B": 0.25, "Q": 0.6, "C": 0.33, "D": 0.21}

    weights = _size(volatilities)

    assert list(weights) == list(volatilities)  # the rebalancer buys in this order
    assert sum(weights.values()) == pytest.approx(0.98, abs=1e-9)
    assert all(0.04 <= weight <= 0.20 for weight in weights.values())
    assert weights["A"] == weights["D"] == 0.20  # both capped once the excess is redistributed
    assert weights["D"] > weights["B"] > weights["C"] > weights["M"] > weights["Q"] > weights["Z"]  # calmer gets more


@pytest.mark.parametrize("volatility", [0.0, -0.1, float("nan"), float("inf")])
def test_a_volatility_that_is_not_finite_and_positive_is_a_programming_error(volatility: float) -> None:
    with pytest.raises(ValueError, match="volatility of B"):
        _size({"A": 0.2, "B": volatility, "C": 0.2, "D": 0.2, "E": 0.2})


def test_bounds_that_cannot_hold_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot sum to"):
        _size({"A": 0.2, "B": 0.2, "C": 0.2})  # 3 x 0.20 < 0.98


def test_no_stock_is_refused() -> None:
    with pytest.raises(ValueError, match="no stock"):
        _size({})

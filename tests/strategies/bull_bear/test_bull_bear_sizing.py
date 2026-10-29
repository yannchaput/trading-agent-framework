from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.sizing import strength_weights

W = {"strong": 0.14, "standard": 0.09}


def test_each_stock_gets_its_strengths_weight_in_input_order() -> None:
    assert strength_weights({"B": "standard", "A": "strong"}, weights=W, total=0.98) == pytest.approx({"B": 0.09, "A": 0.14})
    assert list(strength_weights({"B": "standard", "A": "strong"}, weights=W, total=0.98)) == ["B", "A"]


def test_ten_strong_stocks_are_scaled_to_the_total_ratios_kept() -> None:
    w = strength_weights({f"S{i}": "strong" for i in range(10)}, weights=W, total=0.98)

    assert sum(w.values()) == pytest.approx(0.98) and all(x == pytest.approx(0.098) for x in w.values())


def test_seven_strong_three_standard_scale_by_one_factor() -> None:
    w = strength_weights({**{f"S{i}": "strong" for i in range(7)}, **{f"T{i}": "standard" for i in range(3)}}, weights=W, total=0.98)

    assert w["T0"] == pytest.approx(0.09 * 0.98 / 1.25) and w["S0"] == pytest.approx(0.14 * 0.98 / 1.25)


def test_a_book_under_the_total_is_not_scaled_up() -> None:
    assert sum(strength_weights({"A": "standard", "B": "standard"}, weights=W, total=0.98).values()) == pytest.approx(0.18)


def test_no_stock_is_an_empty_book_and_an_unknown_or_weak_strength_is_an_error() -> None:
    assert strength_weights({}, weights=W, total=0.98) == {}
    for bad in ("high", "weak"):
        with pytest.raises(ValueError, match=bad):
            strength_weights({"A": bad}, weights=W, total=0.98)

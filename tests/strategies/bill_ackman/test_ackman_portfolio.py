from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bill_ackman.portfolio import target_portfolio


def test_the_remainder_after_the_stocks_and_the_cash_buffer_goes_to_parking() -> None:
    target = target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02)

    assert target.weights == {"A": 0.35, "B": 0.25}
    assert target.parking_weight == pytest.approx(0.38)


def test_an_empty_portfolio_parks_everything_but_the_cash_buffer() -> None:
    target = target_portfolio({}, cash_buffer=0.02)

    assert target.weights == {}
    assert target.parking_weight == pytest.approx(0.98)


def test_a_fully_invested_portfolio_parks_nothing() -> None:
    target = target_portfolio({"A": 0.35, "B": 0.35, "C": 0.28}, cash_buffer=0.02)

    assert target.parking_weight == pytest.approx(0.0, abs=1e-9)


def test_weights_above_the_investable_share_are_refused() -> None:
    with pytest.raises(ValueError, match="0.98"):
        target_portfolio({"A": 0.35, "B": 0.35, "C": 0.3}, cash_buffer=0.02)


def test_the_input_is_copied() -> None:
    weights = {"A": 0.3}
    target = target_portfolio(weights, cash_buffer=0.02)
    weights["B"] = 0.3

    assert target.weights == {"A": 0.3}

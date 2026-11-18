"""Tests for the bear put spread pricing and cost model."""

import math

import pytest

from trading_agent_framework.strategies.bear_put_spread.pricing import CostModel, VolModel, put_price, spread_mid


def test_at_the_money_put_matches_black_scholes() -> None:
    assert put_price(100.0, 100.0, 0.2, 1.0) == pytest.approx(7.96557, abs=1e-4)


def test_put_tends_to_intrinsic_at_expiry() -> None:
    assert put_price(90.0, 100.0, 0.3, 1e-6) == pytest.approx(10.0, abs=1e-3)
    assert put_price(110.0, 100.0, 0.3, 1e-6) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_bad_inputs_raise(bad: float) -> None:
    for args in ((bad, 100, 0.2, 1), (100, bad, 0.2, 1), (100, 100, bad, 1), (100, 100, 0.2, bad)):
        with pytest.raises(ValueError):
            put_price(*args)


def test_skew_raises_vol_below_spot_only() -> None:
    model = VolModel()
    assert model.vol(600, 600, 20) == pytest.approx(0.20)
    assert model.vol(600, 660, 20) == pytest.approx(0.20)
    assert model.vol(600, 540, 20) == pytest.approx(0.20 + 0.8 * math.log(600 / 540))


@pytest.mark.parametrize("spot,vix", [(600, 18), (600, 80), (300, 80), (400, 80)])
def test_spread_value_stays_between_zero_and_width(spot: float, vix: float) -> None:
    value = spread_mid(spot, 588.0, 583.0, vix, 30 / 365, VolModel())
    assert 0.0 <= value <= 5.0


def test_costs_per_contract() -> None:
    costs = CostModel()
    assert costs.open_cost(1) == pytest.approx(12.2)
    assert costs.close_cost(1) == pytest.approx(10.2)
    assert costs.open_cost(3) == pytest.approx(36.6)

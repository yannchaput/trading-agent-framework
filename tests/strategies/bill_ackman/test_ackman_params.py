from __future__ import annotations

import dataclasses
import math

import pytest

from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.screen import ScreenParams


def test_the_defaults_are_the_spec_values() -> None:
    params = AckmanParams()

    assert params.screen == ScreenParams()
    assert params.screen.top_n == 15
    assert (params.research_top_n, params.max_positions) == (8, 5)
    assert (params.max_weight, params.min_weight, params.cash_buffer) == (0.35, 0.05, 0.02)
    assert (params.forced_exit_fails, params.rebalance_band, params.min_trade_pct) == (2, 0.05, 0.005)
    assert (params.parking_symbol, params.max_consecutive_abandoned, params.reason_max_chars) == ("SHV", 3, 300)
    assert (params.reentry_cooldown_reviews, params.agent_temperature) == (4, 0.3)


def test_the_largest_total_weight_is_one_minus_the_cash_buffer() -> None:
    assert AckmanParams().max_total_weight == pytest.approx(0.98)
    assert AckmanParams(cash_buffer=0.1).max_total_weight == pytest.approx(0.9)


def test_the_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        AckmanParams().max_weight = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"research_top_n": 0},
        {"max_positions": 0},
        {"min_weight": 0.0},
        {"min_weight": -0.01},
        {"max_weight": 1.01},
        {"min_weight": 0.4, "max_weight": 0.35, "rebalance_band": 0.05},
        {"min_weight": 0.04, "rebalance_band": 0.05},  # a chosen stock could sit below the band and never be bought
        {"cash_buffer": -0.01},
        {"cash_buffer": 1.0},
        {"forced_exit_fails": 0},
        {"rebalance_band": 0.0},
        {"rebalance_band": 1.0},
        {"min_trade_pct": -0.001},
        {"min_trade_pct": 1.0},
        {"max_consecutive_abandoned": 0},
        {"reason_max_chars": 0},
        {"parking_symbol": " "},
        {"max_weight": math.nan},
        {"cash_buffer": math.inf},
        {"max_positions": 30, "min_weight": 0.05},  # 30 x 5% cannot fit in 98%
        {"reentry_cooldown_reviews": -1},
        {"agent_temperature": -0.1},
        {"agent_temperature": 2.01},
        {"agent_temperature": math.nan},
    ],
)
def test_an_out_of_range_value_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="AckmanParams"):
        AckmanParams(**overrides)  # type: ignore[arg-type]


def test_boundary_values_are_accepted() -> None:
    AckmanParams(min_weight=0.05, max_weight=0.05, rebalance_band=0.05, cash_buffer=0.0, max_positions=20, forced_exit_fails=1)


def test_a_cooldown_of_zero_and_the_temperature_bounds_are_accepted() -> None:
    AckmanParams(reentry_cooldown_reviews=0, agent_temperature=0.0)
    AckmanParams(agent_temperature=2.0)
    AckmanParams(agent_temperature=None)

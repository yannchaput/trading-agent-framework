from __future__ import annotations

import dataclasses
import math

import pytest

from trading_agent_framework.strategies.congress_trades.parameters import CongressParams


def test_the_defaults_are_the_spec_values() -> None:
    params = CongressParams()

    assert params.politician == "Nancy Pelosi"
    assert (params.max_holdings, params.max_positions) == (20, 15)
    assert (params.max_total_weight, params.max_position_weight, params.min_weight) == (0.95, 0.15, 0.01)
    assert params.tier_weight_base == 1.5
    assert (params.rebalance_band, params.min_trade_pct) == (0.01, 0.005)
    assert (params.max_trade_retries, params.max_consecutive_abandoned, params.reason_max_chars) == (3, 3, 300)
    assert (params.agent_temperature, params.order_wait_seconds) == (0.3, 60.0)


def test_the_params_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CongressParams().min_weight = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"politician": ""},
        {"politician": "   "},
        {"max_holdings": 0},
        {"max_positions": 0},
        {"max_total_weight": 0.0},
        {"max_total_weight": 1.01},
        {"min_weight": 0.0},
        {"min_weight": -0.01},
        {"max_position_weight": 0.96},  # above max_total_weight
        {"min_weight": 0.2},  # above max_position_weight
        {"min_weight": 0.005},  # below the rebalance band: such a stock would never be bought
        {"rebalance_band": 0.0},
        {"rebalance_band": 1.0},
        {"min_trade_pct": -0.001},
        {"min_trade_pct": 1.0},
        {"order_wait_seconds": 0.0},
        {"order_wait_seconds": -1.0},
        {"max_trade_retries": -1},
        {"max_consecutive_abandoned": 0},
        {"reason_max_chars": 0},
        {"max_positions": 100},  # 100 x 1% does not fit in 95%
        {"agent_temperature": -0.1},
        {"agent_temperature": 2.1},
        {"agent_temperature": math.nan},
        {"max_total_weight": math.nan},
        {"tier_weight_base": 0.9},  # a higher tier would weigh less
        {"tier_weight_base": math.nan},
        {"min_weight": math.inf},
        {"rebalance_band": math.nan},
        {"order_wait_seconds": math.inf},
    ],
)
def test_an_invalid_value_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="CongressParams"):
        CongressParams(**overrides)  # type: ignore[arg-type]


def test_every_problem_is_reported_at_once() -> None:
    with pytest.raises(ValueError) as caught:
        CongressParams(politician="", max_holdings=0)

    assert "politician" in str(caught.value) and "max_holdings" in str(caught.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"agent_temperature": None},
        {"agent_temperature": 0.0},
        {"max_trade_retries": 0},
        {"min_trade_pct": 0.0},
        {"max_positions": 95},
        {"politician": "Someone Else"},
        {"tier_weight_base": 1.0},
        {"tier_weight_base": 3.0},
    ],
)
def test_valid_edge_values_are_accepted(overrides: dict[str, object]) -> None:
    CongressParams(**overrides)  # type: ignore[arg-type]

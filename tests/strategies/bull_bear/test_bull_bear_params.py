from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.shortlist_size, params.retention_rank, params.min_picks, params.max_picks) == (15, 35, 5, 10)
    assert (params.min_weight, params.max_weight, params.cash_buffer, params.rebalance_band) == (0.04, 0.20, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"min_weight": 0.1}, "max_picks x min_weight must fit"),
        ({"min_picks": 4}, "min_picks x max_weight must reach"),
        ({"retention_rank": 15}, "retention_rank must be above shortlist_size"),
        ({"max_picks": 16}, "1 <= min_picks <= max_picks <= shortlist_size"),
        ({"rebalance_time": "9:00"}, "rebalance_time"),
        ({"rebalance_weekday": 5}, "rebalance_weekday must be a weekday"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
        ({"parking_symbol": " "}, "parking_symbol must not be blank"),
        ({"min_yahoo_coverage": 0.0}, "min_yahoo_coverage must be in (0, 1]"),
        ({"yahoo_retry_delays": (60.0, -1.0)}, "yahoo_retry_delays"),
        ({"research_tool_budget": 0}, "research_tool_budget must be at least 1"),
    ],
)
def test_an_invalid_value_is_refused_with_a_reason(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):  # messages contain "(0, 1]"
        BullBearParams(**overrides)

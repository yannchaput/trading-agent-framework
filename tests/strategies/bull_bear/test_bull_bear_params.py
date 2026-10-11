from __future__ import annotations

import re
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.candidate_count, params.max_book_size, params.retention_rank) == (15, 10, 35)
    assert (params.min_weight, params.max_weight, params.cash_buffer, params.rebalance_band) == (0.04, 0.20, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")
    assert params.reentry_cooldown_reviews == 4
    assert params.feedback_reviews == 4
    assert params.judge_memory_budget == 6
    assert params.research_news is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"max_book_size": 30}, "max_book_size x min_weight must fit"),
        ({"candidate_count": 35}, "retention_rank must be above candidate_count"),
        ({"retention_rank": 15}, "retention_rank must be above candidate_count"),
        ({"candidate_count": 0}, "candidate_count and max_book_size must be at least 1"),
        ({"max_book_size": 0}, "candidate_count and max_book_size must be at least 1"),
        ({"rebalance_time": "9:00"}, "rebalance_time"),
        ({"rebalance_weekday": 5}, "rebalance_weekday must be a weekday"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
        ({"parking_symbol": " "}, "parking_symbol must not be blank"),
        ({"min_yahoo_coverage": 0.0}, "min_yahoo_coverage must be in (0, 1]"),
        ({"yahoo_retry_delays": (60.0, -1.0)}, "yahoo_retry_delays"),
        ({"research_tool_budget": 0}, "research_tool_budget must be at least 1"),
        ({"reentry_cooldown_reviews": -1}, "reentry_cooldown_reviews must be at least 0"),
        ({"feedback_reviews": 0}, "feedback_reviews must be at least 1"),
        ({"judge_memory_budget": -1}, "judge_memory_budget must be at least 0"),
    ],
)
def test_an_invalid_value_is_refused_with_a_reason(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):  # messages contain "(0, 1]"
        BullBearParams(**overrides)


@pytest.mark.parametrize("field", ["shortlist_size", "min_picks", "max_picks", "book_size", "max_vetoes"])
def test_the_picker_and_veto_parameters_are_gone(field: str) -> None:
    overrides: dict[str, Any] = {field: 5}

    with pytest.raises(TypeError):
        BullBearParams(**overrides)


def test_there_is_no_derived_debate_size() -> None:
    assert not hasattr(BullBearParams(), "debate_size")

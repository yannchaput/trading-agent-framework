from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.candidate_count, params.max_book_size, params.retention_rank) == (15, 10, 35)
    assert (params.stock_weight, params.cash_buffer, params.rebalance_band) == (0.098, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert params.max_book_size * params.stock_weight == pytest.approx(params.investable)  # a full book is fully invested
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")
    assert params.reentry_cooldown_reviews == 4
    assert params.feedback_reviews == 4
    assert params.judge_memory_budget == 3
    assert params.research_news is True
    assert params.max_agent_steps == 60  # the worst normal run, a researcher using its 5 tool calls, needs a limit of 21; each rejected resubmission adds 3
    assert (params.note_max_chars, params.note_hard_max_chars) == (500, 600)  # the researcher is asked for 500, a note up to 600 is kept
    assert params.max_output_tokens == 36864  # the thinking budget plus 8,192 for the submission
    assert params.judge_max_output_tokens == 45056  # the judge's own cap: its budget plus 8,192; ~20k prompt + 45,056 fits the 81,920 window
    assert params.thinking_token_budget == 28672  # a safety net set high on purpose: a healthy turn never reaches it
    assert params.judge_thinking_token_budget == 36864  # the judge's own budget
    assert params.max_output_tokens - params.thinking_token_budget == 8192  # room after a cut for the submission
    assert params.judge_max_output_tokens - params.judge_thinking_token_budget == 8192


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"stock_weight": 0.0}, "stock_weight must be in (0, 1]"),
        ({"stock_weight": 1.2}, "stock_weight must be in (0, 1]"),
        ({"stock_weight": 0.1}, "max_book_size x stock_weight must fit in 1 - cash_buffer"),  # 10 x 10% > 98%
        ({"stock_weight": 0.03}, "stock_weight must be above rebalance_band"),
        ({"stock_weight": math.nan}, "must be finite"),
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
        ({"max_agent_steps": 0}, "max_agent_steps must be at least 1"),
        ({"max_output_tokens": 0}, "max_output_tokens must be None or at least 1"),
        ({"judge_max_output_tokens": 0}, "judge_max_output_tokens must be None or at least 1"),
        ({"thinking_token_budget": 0}, "thinking_token_budget must be None or at least 1"),
        ({"judge_thinking_token_budget": 0}, "judge_thinking_token_budget must be None or at least 1"),
        ({"note_hard_max_chars": 499}, "note_hard_max_chars must be at least note_max_chars"),
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


def test_there_is_no_min_weight_max_weight_or_weak_weight() -> None:
    names = {field.name for field in dataclasses.fields(BullBearParams)}

    assert not names & {"min_weight", "max_weight", "weak_weight"}

from __future__ import annotations

import re
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.book_size, params.max_vetoes, params.debate_size, params.retention_rank) == (10, 5, 15, 35)
    assert (params.min_weight, params.max_weight, params.cash_buffer, params.rebalance_band) == (0.04, 0.20, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")


def test_the_book_defaults_debate_the_top_15() -> None:
    params = BullBearParams()

    assert (params.book_size, params.max_vetoes, params.debate_size) == (10, 5, 15)
    assert BullBearParams(book_size=8, max_vetoes=7).debate_size == 15


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"book_size": 30, "retention_rank": 40}, "book_size x min_weight must fit"),
        ({"book_size": 4}, "book_size x max_weight must reach"),
        ({"max_vetoes": 25}, "retention_rank must be above debate_size"),
        ({"retention_rank": 15}, "retention_rank must be above debate_size"),
        ({"book_size": 0}, "book_size and max_vetoes must be at least 1"),
        ({"max_vetoes": 0}, "book_size and max_vetoes must be at least 1"),
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


@pytest.mark.parametrize("field", ["shortlist_size", "min_picks", "max_picks"])
def test_the_pickers_parameters_are_gone(field: str) -> None:
    overrides: dict[str, Any] = {field: 5}

    with pytest.raises(TypeError):
        BullBearParams(**overrides)

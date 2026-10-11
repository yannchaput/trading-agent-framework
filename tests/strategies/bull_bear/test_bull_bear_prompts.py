from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear import prompts


@pytest.mark.parametrize(
    ("system", "tool"),
    [
        (prompts.RESEARCHER_SYSTEM, "submit_note"),
        (prompts.BULL_SYSTEM, "submit_bull_case"),
        (prompts.BEAR_SYSTEM, "submit_bear_case"),
        (prompts.JUDGE_SYSTEM, "submit_verdicts"),
    ],
)
def test_every_system_prompt_requires_english_and_names_its_submit_tool(system: str, tool: str) -> None:
    assert "English" in system and tool in system


@pytest.mark.parametrize("system", [prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM])
def test_no_prompt_states_the_review_cadence(system: str) -> None:
    assert not any(word in system.lower() for word in ("weekly", "every week", "tuesday", "daily", "every day"))


def test_only_the_bear_prompt_lists_the_concerns() -> None:
    assert "momentum_exhaustion" in prompts.BEAR_SYSTEM and "momentum_exhaustion" not in prompts.BULL_SYSTEM


def test_the_bear_may_not_call_a_strong_trend_or_a_stretched_price_a_risk() -> None:
    """Run 2026-10-10_085928: 244 `momentum_exhaustion` flags, 88% of them 'high' on ranks 1-3, returned +2.2% the next week against +1.4% overall."""
    text = prompts.BEAR_SYSTEM

    assert "far above its 200-day average" in text and "large three-month gain" in text  # named as NOT a risk
    assert "price far above its 200-day average)" not in text  # no longer offered as an example of exhaustion
    assert "one-month return far above its three-month pace" not in text


def test_momentum_exhaustion_needs_a_reversal_already_under_way() -> None:
    text = prompts.BEAR_SYSTEM

    # the fact sheet's drawdown is NEGATIVE below the high (-21.0 for 21% down), so the threshold must be stated as -15
    assert "negative one-month return" in text and "drawdown_from_52w_high_pct of -15 or below" in text
    assert "still rising is never momentum_exhaustion" in text
    assert "concrete risk" in text  # a stock is rated high only for a risk the bear can name


def test_the_researcher_task_names_the_symbol_and_the_note_limit() -> None:
    task = prompts.researcher_task("AAA", 500)

    assert "AAA" in task and "500" in task


@pytest.mark.parametrize("task", [prompts.bull_task, prompts.bear_task, prompts.judge_task])
def test_the_debate_tasks_state_the_character_cap_they_are_given(task) -> None:  # noqa: ANN001
    text = task(234)

    assert "234 characters" in text and "rejects the whole submission" in text


def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("submit_verdicts", "11 stocks in: put at most 10 in and the others out")

    assert "submit_verdicts" in text and "put at most 10 in" in text


def test_the_judge_puts_each_stock_in_or_out_under_a_cap_and_a_trend_alone_is_no_reason_for_out() -> None:
    text = prompts.JUDGE_SYSTEM

    assert "in or out" in text and "max_in" in text
    assert "not a reason for out by itself" in text and "already reacted to" in text
    assert "veto" not in text.lower() and "veto" not in prompts.judge_task(300).lower()


def test_no_prompt_mentions_forced_exits() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM, prompts.bull_task(300), prompts.bear_task(300), prompts.judge_task(300)):
        assert "forced" not in text.lower()


def test_no_prompt_explains_held_or_weight() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM):
        assert "held" not in text and "weight_pct" not in text


def test_the_fact_sheet_explains_the_one_week_return() -> None:
    assert "return_1w_pct" in prompts.BULL_SYSTEM and "return_1w_pct" in prompts.BEAR_SYSTEM


def test_the_researcher_reads_the_prior_year_and_is_not_asked_for_a_trend_it_cannot_see() -> None:
    assert "prior_year" in prompts.RESEARCHER_SYSTEM and "trend of revenue" not in prompts.RESEARCHER_SYSTEM


def test_the_bear_may_not_rate_news_the_price_already_absorbed_high_by_itself() -> None:
    assert "already reacted to" in prompts.BEAR_SYSTEM and "lasting consequence" in prompts.BEAR_SYSTEM

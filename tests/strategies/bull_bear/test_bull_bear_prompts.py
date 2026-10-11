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


def test_the_researcher_task_names_the_symbol_and_the_note_limit() -> None:
    task = prompts.researcher_task("AAA", 500)

    assert "AAA" in task and "500" in task


@pytest.mark.parametrize("task", [prompts.bull_task, prompts.bear_task, prompts.judge_task])
def test_the_debate_tasks_state_the_character_cap_they_are_given(task) -> None:  # noqa: ANN001
    text = task(234)

    assert "234 characters" in text and "rejects the whole submission" in text


def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("submit_verdicts", "6 vetoes: veto at most 5 stocks and keep the others")

    assert "submit_verdicts" in text and "veto at most 5" in text


def test_the_judge_gives_a_verdict_per_stock_and_a_trend_alone_is_no_reason_to_veto() -> None:
    assert "keep or veto" in prompts.JUDGE_SYSTEM and "max_vetoes" in prompts.JUDGE_SYSTEM
    assert "not a reason to veto" in prompts.JUDGE_SYSTEM
    assert "pick" not in prompts.JUDGE_SYSTEM.lower() and "pick" not in prompts.judge_task(300).lower()


def test_no_prompt_mentions_forced_exits() -> None:
    for text in (prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM, prompts.bull_task(300), prompts.bear_task(300), prompts.judge_task(300)):
        assert "forced" not in text.lower()

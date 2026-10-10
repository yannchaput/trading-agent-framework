from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear import prompts


@pytest.mark.parametrize(
    ("system", "tool"),
    [
        (prompts.RESEARCHER_SYSTEM, "submit_note"),
        (prompts.BULL_SYSTEM, "submit_bull_case"),
        (prompts.BEAR_SYSTEM, "submit_bear_case"),
        (prompts.JUDGE_SYSTEM, "submit_picks"),
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


def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("submit_picks", "pick between 5 and 10 stocks, got 4")

    assert "submit_picks" in text and "got 4" in text

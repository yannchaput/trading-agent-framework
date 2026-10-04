from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.bill_ackman import prompts
from trading_agent_framework.strategies.bill_ackman.handoff import CONCERNS

# The review cadence is the strategy's `sleeptime`, which changes: no prompt may say how often it runs.
_CADENCE_WORDING = re.compile(r"\b(each day|every day|daily|yesterday|today)\b", re.IGNORECASE)


@pytest.mark.parametrize(
    "name",
    ["RESEARCHER_SYSTEM", "SHORT_SELLER_SYSTEM", "TRADER_SYSTEM", "SHORT_SELLER_TASK", "TRADER_TASK"],
)
def test_no_prompt_hardcodes_the_review_cadence(name: str) -> None:
    assert _CADENCE_WORDING.search(getattr(prompts, name)) is None


def test_the_researcher_is_not_told_to_prefer_holdings_or_shown_the_last_ranking() -> None:
    assert "Prefer a stock we already hold" not in prompts.RESEARCHER_SYSTEM
    assert "previous review's ranking" not in prompts.RESEARCHER_SYSTEM


def test_the_researcher_is_warned_about_value_traps_and_concentration() -> None:
    assert "price_return_12m" in prompts.RESEARCHER_SYSTEM and "check why" in prompts.RESEARCHER_SYSTEM
    assert "different businesses and industries" in prompts.RESEARCHER_SYSTEM


def test_the_short_seller_knows_the_concerns_its_memory_and_the_news_rule() -> None:
    for concern in CONCERNS:
        assert concern in prompts.SHORT_SELLER_SYSTEM
    for phrase in ("previous_verdict", "what_changed", "A news event alone", "cheaper, not riskier"):
        assert phrase in prompts.SHORT_SELLER_SYSTEM


def test_the_trader_must_keep_every_required_holding_and_is_no_longer_told_to_let_go() -> None:
    assert "let go" not in prompts.TRADER_SYSTEM
    assert "failed once while held" not in prompts.TRADER_SYSTEM
    assert "current holdings that have not failed twice" in prompts.TRADER_SYSTEM
    assert "required" in prompts.TRADER_SYSTEM and "at least at the minimum weight" in prompts.TRADER_SYSTEM

from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.bill_ackman import prompts

# The review cadence is the strategy's `sleeptime`, which changes: no prompt may say how often it runs.
_CADENCE_WORDING = re.compile(r"\b(each day|every day|daily|yesterday|today)\b", re.IGNORECASE)


@pytest.mark.parametrize(
    "name",
    ["RESEARCHER_SYSTEM", "SHORT_SELLER_SYSTEM", "TRADER_SYSTEM", "SHORT_SELLER_TASK", "TRADER_TASK"],
)
def test_no_prompt_hardcodes_the_review_cadence(name: str) -> None:
    assert _CADENCE_WORDING.search(getattr(prompts, name)) is None

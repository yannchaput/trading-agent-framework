from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import DeskRig

from trading_agent_framework.strategies.earnings_drift.prompts import DRIFT_SYSTEM, TASK_PROMPT
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS, desk_tools, research_tools
from trading_agent_framework.utils.errors import ConfigurationError


def test_desk_tools_are_the_order_tools_with_one_line_docstrings(tmp_path: Path) -> None:
    tools = desk_tools(DeskRig(tmp_path).desk)
    assert tuple(tool.__name__ for tool in tools) == ORDER_TOOLS
    for tool in tools:
        assert tool.__doc__ and "\n" not in tool.__doc__.strip()
        assert all(p.annotation is not inspect.Parameter.empty for p in inspect.signature(tool).parameters.values())


def test_desk_tools_reach_the_desk(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    tools = {tool.__name__: tool for tool in desk_tools(rig.desk)}
    assert tools["buy"]("AAA", 10, 8.0, "beat")["symbol"] == "AAA"
    assert tools["skip"]("BBB", "guidance cut") == {"symbol": "BBB", "status": "skipped"}
    assert "no open position" in tools["sell"]("AAA", "x")["error"]  # pending, not open yet
    assert "no open position" in tools["set_trailing_stop"]("AAA", 5.0, "x")["error"]


def test_research_tools_are_news_two_filing_tools_and_market_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")
    names = [tool.__name__ for tool in research_tools(DeskRig(tmp_path).strategy)]
    assert names == ["search_news", "get_filings", "get_filing_document", "get_last_price", "get_quote", "get_bars"]


_HOLDING_LIMIT = re.compile(r"\b(?:\d+|ten)\s*-?\s*(?:trading\s+)?(?:sessions?|days?)\b")


def test_the_prompt_never_states_the_holding_limit_and_names_every_order_tool() -> None:
    text = DRIFT_SYSTEM.lower()
    for forbidden in ("10 sessions", "ten sessions", "max_holding", "max_hold", "holding period", "10 days", "trading days", "trading sessions", "weeks"):
        assert forbidden not in text
    assert _HOLDING_LIMIT.search(text) is None
    for tool in ORDER_TOOLS:
        assert f"{tool}(" in DRIFT_SYSTEM
    assert "do not recompute" in DRIFT_SYSTEM.lower()
    assert TASK_PROMPT.strip()


@pytest.mark.parametrize("phrase", ["held for 10 sessions", "sell after 10 trading days", "a 10-day limit", "ten sessions", "within 12 days"])
def test_the_holding_limit_pattern_catches_a_stated_limit(phrase: str) -> None:
    assert _HOLDING_LIMIT.search(phrase) is not None


def test_the_prompt_explains_slots_fills_stop_status_and_a_missing_atr() -> None:
    text = " ".join(DRIFT_SYSTEM.split())
    assert "balances.free_slots" in text and "skip the rest" in text
    assert "Buy and sell orders fill at the next open; a trailing stop change takes effect at once" in text
    assert "Every order you place fills at the next open" not in text
    for status in ('"working"', '"missing"', '"backstop"'):
        assert status in text
    assert "context.atr14_pct is null" in text and "about 8" in text


def test_research_tools_need_an_sec_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    with pytest.raises(ConfigurationError, match="SEC_EDGAR_USER_AGENT"):
        research_tools(DeskRig(tmp_path).strategy)

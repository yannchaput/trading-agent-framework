from __future__ import annotations

from typing import Any

import pytest

from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.news_builtin.grounding import require_search_news_before


def _fake_tool(name: str, calls: list[str], result: dict[str, Any] | None = None) -> Any:
    def tool(**kwargs: Any) -> dict[str, Any]:
        calls.append(name)
        return dict(result) if result is not None else {"ok": name}

    tool.__name__ = name
    tool.__doc__ = f"Fake {name}."
    return tool


def _fake_search_news(calls: list[str], result: dict[str, Any] | None = None) -> Any:
    """Mirrors the real `search_news` signature so grounding can bind positional args too."""

    def search_news(
        symbols: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        calls.append("search_news")
        return dict(result) if result is not None else {"count": 0, "articles": []}

    search_news.__doc__ = "Fake search_news."
    return search_news


def _tools(calls: list[str], *, news_result: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = [
        _fake_search_news(calls, news_result),
        _fake_tool("remember_decision", calls),
        _fake_tool("submit_order", calls),
        _fake_tool("get_positions", calls),  # untouched control tool
    ]
    wrapped = require_search_news_before(raw)
    return {tool.__name__: tool for tool in wrapped}


def test_remember_decision_is_refused_before_search_news_succeeds_this_run() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        result = tools["remember_decision"](text="KEEP")

    assert "error" in result
    assert "search_news" in result["error"]
    assert calls == []  # the real remember_decision was never invoked


def test_the_refusal_says_an_empty_result_still_counts() -> None:
    # Regression: glm-4.7-flash repeatedly misread "grounded in a specific article" as requiring the
    # article to actually have a body, and invented a "gating rule requires a full-text article" that
    # does not exist -- recorded verbatim into memory, where later runs' search_memory recalled it back.
    # Every place that tells the model about this gate must say outright that an empty response counts.
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        result = tools["remember_decision"](text="KEEP")

    assert "an empty result also counts" in result["error"]


def test_remember_decision_succeeds_after_an_include_content_search_news_call_in_the_same_run() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        tools["search_news"](symbols="SPY", include_content=True)
        result = tools["remember_decision"](text="KEEP")

    assert result == {"ok": "remember_decision"}
    assert calls == ["search_news", "remember_decision"]


def test_a_headline_only_search_news_call_does_not_ground_the_run() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        tools["search_news"](symbols="SPY")  # include_content defaults to False
        result = tools["remember_decision"](text="KEEP")

    assert "error" in result
    assert "include_content" in result["error"]
    assert calls == ["search_news"]  # remember_decision was never invoked


def test_include_content_passed_positionally_still_grounds_the_run() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        tools["search_news"]("SPY", None, None, 10, True)  # include_content is the 5th positional arg
        result = tools["remember_decision"](text="KEEP")

    assert result == {"ok": "remember_decision"}


def test_an_include_content_call_that_returns_no_content_still_grounds_the_run() -> None:
    # The point of this step is that the agent TRIED to read a specific article, not that the
    # provider happened to have a body for it -- some articles (terse data-print/quote wires) never
    # carry one no matter which one is picked or how the search window is narrowed. Refusing to
    # ground on that outcome previously deadlocked runs: the model retried different content-less
    # articles dozens to 100+ times in a row until it exhausted its turn budget with no decision
    # recorded (see grounding.py's docstring).
    calls: list[str] = []
    tools = _tools(calls, news_result={"count": 1, "articles": [{"id": 1, "headline": "h"}]})

    with agent_call_context(run_id="run-1"):
        tools["search_news"](symbols="SPY", include_content=True)
        result = tools["remember_decision"](text="KEEP")

    assert result == {"ok": "remember_decision"}


def test_submit_order_is_gated_the_same_way() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        before = tools["submit_order"](symbol="SPY", quantity=1, side="buy")
        tools["search_news"](symbols="SPY", include_content=True)
        after = tools["submit_order"](symbol="SPY", quantity=1, side="buy")

    assert "error" in before
    assert after == {"ok": "submit_order"}


def test_a_failed_search_news_call_does_not_count_as_grounding() -> None:
    calls: list[str] = []
    tools = _tools(calls, news_result={"error": "provider down"})

    with agent_call_context(run_id="run-1"):
        tools["search_news"](symbols="SPY", include_content=True)
        result = tools["remember_decision"](text="KEEP")

    assert "error" in result
    assert "search_news" in result["error"]


def test_grounding_does_not_carry_over_to_a_new_run() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    with agent_call_context(run_id="run-1"):
        tools["search_news"](symbols="SPY", include_content=True)
    with agent_call_context(run_id="run-2"):
        result = tools["remember_decision"](text="KEEP")

    assert "error" in result


def test_outside_an_agent_run_nothing_is_gated() -> None:
    calls: list[str] = []
    tools = _tools(calls)

    result = tools["remember_decision"](text="KEEP")

    assert result == {"ok": "remember_decision"}


def test_other_tools_are_returned_unchanged() -> None:
    calls: list[str] = []
    raw = [
        _fake_tool("search_news", calls),
        _fake_tool("remember_decision", calls),
        _fake_tool("get_positions", calls),
    ]
    wrapped = require_search_news_before(raw)

    assert wrapped[2] is raw[2]  # get_positions is not one of the gated tools


def test_missing_search_news_tool_raises() -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="search_news"):
        require_search_news_before([_fake_tool("remember_decision", calls)])


def test_wrapped_tools_keep_their_name_and_docstring() -> None:
    calls: list[str] = []
    raw = [_fake_tool("search_news", calls), _fake_tool("remember_decision", calls)]
    wrapped = {tool.__name__: tool for tool in require_search_news_before(raw)}

    assert wrapped["search_news"].__name__ == "search_news"
    assert wrapped["search_news"].__doc__ == "Fake search_news."
    assert wrapped["remember_decision"].__doc__ == "Fake remember_decision."

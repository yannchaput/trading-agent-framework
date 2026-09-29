from __future__ import annotations

import pytest
from tests.fakes import et

from trading_agent_framework.strategies.vwap_pullback.graph import build_tick_graph, route_after_classify, route_after_exit
from trading_agent_framework.utils.errors import FatalStrategyError

NOW = et(2026, 9, 1, 10, 0)


def test_routing() -> None:
    assert route_after_classify({"exit_due": ["AAA"], "entry_due": True}) == "exit_agent"
    assert route_after_classify({"exit_due": [], "entry_due": True}) == "entry_agent"
    assert route_after_classify({}) == "__end__"
    assert route_after_exit({"entry_due": True}) == "entry_agent"
    assert route_after_exit({"entry_due": False}) == "__end__"


def _graph(calls: list[str], *, classify_result: dict, exit_result: dict | None = None):
    def classify(state):
        calls.append("classify")
        assert state["now"] == NOW
        return classify_result

    def run_exit(state):
        calls.append("exit")
        return exit_result or {"runs": [{"agent": "exit"}]}

    def run_entry(state):
        calls.append("entry")
        return {"runs": [{"agent": "entry"}]}

    return build_tick_graph(classify=classify, run_exit=run_exit, run_entry=run_entry)


def test_a_quiet_tick_runs_only_the_classifier() -> None:
    calls: list[str] = []
    _graph(calls, classify_result={"exit_due": [], "entry_due": False}).invoke({"now": NOW})
    assert calls == ["classify"]


def test_exit_runs_before_entry_and_entry_follows_the_exit_nodes_update() -> None:
    calls: list[str] = []
    graph = _graph(calls, classify_result={"exit_due": ["AAA"], "entry_due": False}, exit_result={"runs": [{"agent": "exit"}], "entry_due": True})
    result = graph.invoke({"now": NOW})
    assert calls == ["classify", "exit", "entry"]
    assert result["runs"] == [{"agent": "exit"}, {"agent": "entry"}]


def test_entry_alone() -> None:
    calls: list[str] = []
    _graph(calls, classify_result={"exit_due": [], "entry_due": True}).invoke({"now": NOW})
    assert calls == ["classify", "entry"]


def test_a_fatal_error_in_a_node_propagates() -> None:
    def classify(state):
        raise FatalStrategyError("abort")

    graph = build_tick_graph(classify=classify, run_exit=lambda s: {}, run_entry=lambda s: {})
    with pytest.raises(FatalStrategyError, match="abort"):
        graph.invoke({"now": NOW})

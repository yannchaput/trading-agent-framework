from __future__ import annotations

import json
import time
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager, _tool_budget_message, _ToolBudget


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


def _calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{index}_{name}") for index, (name, args) in enumerate(calls)])


def _agent(seen: list[str], *messages: AIMessage, exempt_tools: list[str] | None = None) -> AgentHandle:
    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        seen.append(symbol)
        return f"{symbol} ok"

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        seen.append(f"submit:{answer}")
        return {"status": "recorded"}

    model = FakeToolCallingChatModel(messages=iter(list(messages)))
    return _manager().create(name="seller", system_prompt="x", model=model, tools=[lookup, submit], exempt_tools=exempt_tools)


def test_calls_past_the_budget_are_refused_and_never_run() -> None:
    seen: list[str] = []
    handle = _agent(
        seen,
        _calls(("lookup", {"symbol": "A"})),
        _calls(("lookup", {"symbol": "B"})),
        _calls(("lookup", {"symbol": "C"})),
        _calls(("submit", {"answer": "x"})),
        AIMessage(content="done"),
        exempt_tools=["submit"],
    )

    result = handle.run("go", tool_budget=2)

    assert seen == ["A", "B", "submit:x"]  # C was refused, the exempt submit still ran
    refused = [call for call in result.tool_calls if call.name == "lookup"][2]
    assert json.loads(refused.result) == {"error": "tool budget of 2 calls spent; lookup was not run. Finish now: call submit."}


def test_several_calls_in_one_turn_count_one_by_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # LangGraph runs one turn's tool calls on pool threads, so which two of A/B/C get the budget is
    # thread-order dependent: assert the count, not the names. Reading `used` is slowed down so that,
    # without the lock around check-and-increment, all three calls read 0 before any increments it.
    def slow_read(self: _ToolBudget) -> int:
        value = self._used
        time.sleep(0.05)
        return value

    def write(self: _ToolBudget, value: int) -> None:
        self._used = value

    monkeypatch.setattr(_ToolBudget, "used", property(slow_read, write), raising=False)
    seen: list[str] = []
    handle = _agent(seen, _calls(("lookup", {"symbol": "A"}), ("lookup", {"symbol": "B"}), ("lookup", {"symbol": "C"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=2)

    assert len(seen) == 2
    assert set(seen) <= {"A", "B", "C"}
    assert sum("tool budget of 2 calls spent" in call.result for call in result.tool_calls) == 1


def test_without_a_budget_nothing_is_refused() -> None:
    seen: list[str] = []
    handle = _agent(seen, _calls(("lookup", {"symbol": "A"})), _calls(("lookup", {"symbol": "B"})), _calls(("lookup", {"symbol": "C"})), AIMessage(content="done"))

    handle.run("go")

    assert seen == ["A", "B", "C"]


def test_the_budget_is_per_run() -> None:
    seen: list[str] = []
    handle = _agent(
        seen,
        _calls(("lookup", {"symbol": "A"})),
        _calls(("lookup", {"symbol": "B"})),
        AIMessage(content="first run done"),
        _calls(("lookup", {"symbol": "C"})),
        AIMessage(content="second run done"),
    )

    handle.run("go", tool_budget=1)
    handle.run("go again")  # no budget: a previous run's budget must not linger

    assert seen == ["A", "C"]


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="tool_budget"):
        _agent([], AIMessage(content="done")).run("go", tool_budget=-1)


def test_the_message_names_only_the_callers_tools() -> None:
    assert _tool_budget_message(12, "get_bars", ["submit_a", "submit_b"]) == "tool budget of 12 calls spent; get_bars was not run. Finish now: call submit_a or submit_b."
    assert _tool_budget_message(3, "get_bars", []) == "tool budget of 3 calls spent; get_bars was not run. Answer now without calling another tool."

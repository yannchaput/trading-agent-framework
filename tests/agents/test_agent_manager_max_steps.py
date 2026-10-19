from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.utils.errors import AgentError, AgentLoopError


def _calls(*calls: tuple[str, dict[str, Any]], tag: str = "") -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{tag}{index}_{name}") for index, (name, args) in enumerate(calls)])


def _agent(responses: Iterator[AIMessage]) -> AgentHandle:
    """An agent with a `lookup` tool and a `submit` tool that refuses everything but the answer "good"."""

    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return f"{symbol} ok"

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        return {"status": "recorded"} if answer == "good" else {"error": "too long"}

    manager = AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))
    return manager.create(name="researcher", system_prompt="x", model=FakeToolCallingChatModel(messages=responses), tools=[lookup, submit], final_tools=["submit"])


def _resubmitting_forever() -> Iterator[AIMessage]:
    for turn in range(10_000):
        yield _calls(("submit", {"answer": "bad"}), tag=f"{turn}_")


def test_a_run_that_keeps_resubmitting_a_rejected_call_stops_at_max_steps_with_an_agent_loop_error() -> None:
    handle = _agent(_resubmitting_forever())

    with pytest.raises(AgentLoopError, match="60 steps"):
        handle.run("go", max_steps=60)


def test_an_agent_loop_error_is_an_agent_error() -> None:
    assert issubclass(AgentLoopError, AgentError)


def test_a_run_within_max_steps_is_not_touched() -> None:
    responses = iter([_calls(("lookup", {"symbol": "A"}), tag="a"), _calls(("lookup", {"symbol": "B"}), tag="b"), _calls(("submit", {"answer": "good"}), tag="c")])

    result = _agent(responses).run("go", max_steps=60)

    assert [call.name for call in result.tool_calls] == ["lookup", "lookup", "submit"]


def test_max_steps_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        _agent(iter([])).run("go", max_steps=0)

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


def _calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{index}_{name}") for index, (name, args) in enumerate(calls)])


def _agent(manager: AgentManager, responses: list[AIMessage], *, final_tools: list[str] | None, submit_result: dict[str, str] | None = None) -> tuple[AgentHandle, list[AIMessage]]:
    """The agent plus the responses the model was never asked for (an unused response is a model call that did not happen)."""

    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return f"{symbol} ok"

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        return submit_result if submit_result is not None else {"status": "recorded"}

    unused = list(responses)
    model = FakeToolCallingChatModel(messages=iter(unused))
    handle = manager.create(name="writer", system_prompt="x", model=model, tools=[lookup, submit], final_tools=final_tools)
    return handle, unused


def test_a_recorded_submission_ends_the_run_without_another_model_call() -> None:
    closing = AIMessage(content="thinking about it again")
    handle, _ = _agent(_manager(), [_calls(("submit", {"answer": "x"})), closing], final_tools=["submit"])

    result = handle.run("go")

    assert [call.name for call in result.tool_calls] == ["submit"]
    assert closing.content not in result.output  # the model was never asked for its closing turn


def test_a_final_tool_that_returned_an_error_does_not_end_the_run() -> None:
    handle, _ = _agent(
        _manager(),
        [_calls(("submit", {"answer": "x"})), AIMessage(content="I will fix it")],
        final_tools=["submit"],
        submit_result={"error": "the answer is too long"},
    )

    result = handle.run("go")

    assert "I will fix it" in result.output  # the model got its next turn to correct the submission


def test_a_tool_that_is_not_final_does_not_end_the_run() -> None:
    handle, _ = _agent(_manager(), [_calls(("lookup", {"symbol": "A"})), AIMessage(content="done")], final_tools=["submit"])

    result = handle.run("go")

    assert "done" in result.output


def test_without_final_tools_the_model_still_gets_its_closing_turn() -> None:
    handle, _ = _agent(_manager(), [_calls(("submit", {"answer": "x"})), AIMessage(content="all done")], final_tools=None)

    result = handle.run("go")

    assert "all done" in result.output


def test_the_skipped_closing_turn_is_not_recorded_as_a_model_call() -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: datetime(2026, 1, 5, 12, 0, tzinfo=UTC))
    handle, _ = _agent(manager, [_calls(("submit", {"answer": "x"})), AIMessage(content="more thinking")], final_tools=["submit"])

    handle.run("go")

    assert manager.telemetry_summary()["writer"]["calls"] == 1

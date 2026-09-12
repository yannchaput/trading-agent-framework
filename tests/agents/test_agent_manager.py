from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.memory.store import MemoryStore
from trading_agent_framework.memory.tools import memory_tools
from trading_agent_framework.utils.errors import AgentError, ConfigurationError


def _credentials() -> LLMCredentials:
    return LLMCredentials(base_url="http://localhost:8000/v1", api_key="key", default_model="qwen3-8b")


def _fake_model(messages: list[AIMessage]) -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(messages=iter(messages))


def _manager() -> AgentManager:
    return AgentManager(_credentials)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_create_returns_a_handle_and_registers_it_by_name() -> None:
    manager = _manager()

    handle = manager.create(name="analyst", system_prompt="be helpful", model=_fake_model([AIMessage("hi")]))

    assert isinstance(handle, AgentHandle)
    assert handle.name == "analyst"
    assert manager["analyst"] is handle
    assert "analyst" in manager
    assert "someone_else" not in manager


def test_create_with_duplicate_name_raises_value_error() -> None:
    manager = _manager()
    manager.create(name="analyst", system_prompt="x", model=_fake_model([AIMessage("hi")]))

    with pytest.raises(ValueError, match="analyst"):
        manager.create(name="analyst", system_prompt="y", model=_fake_model([AIMessage("hi")]))


def test_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        _manager()["missing"]


def test_create_with_no_model_and_no_llm_model_env_raises_configuration_error() -> None:
    manager = AgentManager(lambda: LLMCredentials(base_url="http://x", api_key="k", default_model=None))

    with pytest.raises(ConfigurationError):
        manager.create(name="analyst", system_prompt="x")


def test_run_returns_a_text_only_result() -> None:
    manager = _manager()
    handle = manager.create(name="analyst", system_prompt="be helpful", model=_fake_model([AIMessage("The answer is 42.")]))

    result = handle.run("What is the answer?")

    assert result == AgentRunResult(output="The answer is 42.", tool_calls=[])


def test_run_with_a_tool_call_round_trips_through_the_tool() -> None:
    manager = _manager()
    model = _fake_model(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="add", args={"a": 1, "b": 2}, id="call_1")]),
            AIMessage(content="The answer is 3."),
        ]
    )
    handle = manager.create(name="calculator", system_prompt="you do math", model=model, tools=[add])

    result = handle.run("what is 1+2?")

    assert result == AgentRunResult(
        output="The answer is 3.",
        tool_calls=[ToolCallRecord(name="add", args={"a": 1, "b": 2}, result="3")],
    )


def test_a_memory_tool_is_accepted_and_actually_invocable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite", strategy_name="s", now=lambda: datetime.now(UTC))
    manager = _manager()
    model = _fake_model(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="remember", args={"text": "note"}, id="call_1")]),
            AIMessage(content="Remembered."),
        ]
    )
    handle = manager.create(name="analyst", system_prompt="x", model=model, tools=memory_tools(store))

    result = handle.run("remember this")

    assert result.output == "Remembered."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "remember"
    assert '"kind": "memory"' in result.tool_calls[0].result


def test_run_with_context_appends_it_to_the_prompt() -> None:
    manager = _manager()
    captured: dict[str, Any] = {}

    class RecordingModel(FakeToolCallingChatModel):
        def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Any:
            captured["last_human_content"] = messages[-1].content
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    handle = manager.create(name="analyst", system_prompt="x", model=RecordingModel(messages=iter([AIMessage("ok")])))

    handle.run("Decide.", context={"symbol": "SPY"})

    assert "Decide." in captured["last_human_content"]
    assert "SPY" in captured["last_human_content"]


def test_a_tool_exception_surfaces_as_agent_error() -> None:
    def broken(text: str) -> str:
        """Always fails."""
        raise ValueError("boom")

    manager = _manager()
    model = _fake_model(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="broken", args={"text": "x"}, id="call_1")]),
            AIMessage(content="unreachable"),
        ]
    )
    handle = manager.create(name="analyst", system_prompt="x", model=model, tools=[broken])

    with pytest.raises(AgentError, match="analyst"):
        handle.run("go")


def test_create_with_a_chat_model_instance_bypasses_credentials() -> None:
    manager = AgentManager(lambda: (_ for _ in ()).throw(AssertionError("credentials should not be read")))

    handle = manager.create(name="analyst", system_prompt="x", model=_fake_model([AIMessage("ok")]))

    assert handle.run("go") == AgentRunResult(output="ok", tool_calls=[])

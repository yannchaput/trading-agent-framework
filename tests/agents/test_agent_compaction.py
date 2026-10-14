from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall
from pydantic import Field
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.compaction import compaction_middleware
from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.utils.errors import ConfigurationError

SUMMARY = "SUMMARY: looked up A"


class _RecordingModel(FakeToolCallingChatModel):
    """Records the messages of every call, the summary call included."""

    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _manager(max_input_tokens: int | None = None) -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b", max_input_tokens=max_input_tokens))


def _lookups_then_answer(symbols: list[str], result_chars: int, *responses: AIMessage, max_input_tokens: int = 1000) -> tuple[_RecordingModel, Any]:
    """An agent that looks up each of `symbols` in turn (each result is `result_chars` of that symbol), then gives `responses`."""

    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return symbol * result_chars

    calls = [AIMessage(content="", tool_calls=[ToolCall(name="lookup", args={"symbol": symbol}, id=f"call_{symbol}")]) for symbol in symbols]
    model = _RecordingModel(messages=iter([*calls, *responses]), profile={"max_input_tokens": max_input_tokens})
    handle = _manager().create(name="researcher", system_prompt="x", model=model, tools=[lookup], compact_history=True)
    return model, handle


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def test_a_history_past_80_percent_of_the_window_is_summarized_before_the_next_model_call() -> None:
    # two ~500-token results against a 1000-token window: under the 800-token trigger after A, past it after B
    model, handle = _lookups_then_answer(["A", "B"], 2000, AIMessage(content=SUMMARY), AIMessage(content="done"))

    handle.run("research A and B")

    *_, summary_call, last_call = model.prompts
    assert "A" * 2000 in summary_call[0].text  # the summary call is given the older history...
    assert any(SUMMARY in text for text in _texts(last_call))  # ...the next turn sees its summary instead
    assert not any("A" * 2000 in text for text in _texts(last_call))
    assert any("B" * 2000 in text for text in _texts(last_call))  # the latest tool exchange is kept as is


def test_a_history_under_80_percent_of_the_window_is_left_alone() -> None:
    model, handle = _lookups_then_answer(["A", "B"], 400, AIMessage(content="done"))

    handle.run("research A and B")

    assert len(model.prompts) == 3  # the three agent turns, no summary call
    assert any("A" * 400 in text for text in _texts(model.prompts[-1]))


def test_without_compaction_a_long_history_reaches_the_model_unchanged() -> None:
    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return "x" * 4000

    first = AIMessage(content="", tool_calls=[ToolCall(name="lookup", args={"symbol": "A"}, id="call_0")])
    model = _RecordingModel(messages=iter([first, AIMessage(content="done")]), profile={"max_input_tokens": 1000})
    _manager().create(name="researcher", system_prompt="x", model=model, tools=[lookup]).run("research A")

    assert len(model.prompts) == 2
    assert any("x" * 4000 in text for text in _texts(model.prompts[-1]))


def test_compaction_without_a_known_window_is_refused() -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))

    with pytest.raises(ConfigurationError, match="LLM_MAX_INPUT_TOKENS"):
        compaction_middleware(model)


def test_create_refuses_compaction_for_a_model_without_a_known_window() -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))

    with pytest.raises(ConfigurationError, match="LLM_MAX_INPUT_TOKENS"):
        _manager().create(name="researcher", system_prompt="x", model=model, compact_history=True)


def test_the_env_window_reaches_a_resolved_chat_model() -> None:
    chat_model = _manager(max_input_tokens=32768)._resolve_model(None, None)

    assert chat_model.profile["max_input_tokens"] == 32768


def test_without_an_env_window_a_resolved_chat_model_keeps_its_own_profile() -> None:
    chat_model = _manager()._resolve_model(None, None)

    assert (chat_model.profile or {}).get("max_input_tokens") is None


def test_the_summary_replaces_the_history_as_a_human_message() -> None:
    model, handle = _lookups_then_answer(["A", "B"], 2000, AIMessage(content=SUMMARY), AIMessage(content="done"))

    handle.run("research A and B")

    summaries = [m for m in model.prompts[-1] if isinstance(m, HumanMessage) and SUMMARY in m.text]
    assert len(summaries) == 1

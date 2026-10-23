from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from pydantic import SecretStr
from tests.fakes import RecordingToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials, ReasoningEffort
from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.utils.errors import AgentError

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)

# vLLM's refusal of a `reasoning_effort` it does not accept (observed, shortened).
_EFFORT_REFUSAL = (
    "Error code: 400 - {'object': 'error', 'message': \"1 validation error for ChatCompletionRequest\\nreasoning_effort\\n"
    "  Input should be 'low', 'medium' or 'high'\", 'type': 'BadRequestError', 'code': 400}"
)


def _manager(effort: ReasoningEffort | None) -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b", reasoning_effort=effort))


def _with_model(manager: AgentManager, model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `create(model=None)` take the credentials path (where the effort applies) while running the fake."""
    monkeypatch.setattr(manager, "_resolve_model", lambda requested, *args, **kwargs: model if requested is None else requested)


def _lookup(symbol: str) -> str:
    """Look up a symbol."""
    return f"{symbol} ok"


def _lookup_call() -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name="_lookup", args={"symbol": "A"}, id="call_0")])


class _RefusingModel(RecordingToolCallingChatModel):
    """Raises `error` on its first call, then answers from `messages`."""

    error: Exception | None = None

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        if self.error is not None:
            error, self.error = self.error, None
            self.calls.append(kwargs)
            raise error
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _efforts(model: RecordingToolCallingChatModel) -> list[Any]:
    return [call.get("reasoning_effort") for call in model.calls]


def test_every_model_call_of_the_run_is_sent_the_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingToolCallingChatModel(messages=iter([_lookup_call(), AIMessage(content="done")]))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup])

    handle.run("go")

    assert _efforts(model) == ["xhigh", "xhigh"]


def test_without_an_effort_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))
    manager = _manager(None)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup])

    handle.run("go")

    assert "reasoning_effort" not in model.calls[0]


def test_a_prebuilt_model_gets_no_reasoning_effort() -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))
    handle = _manager(ReasoningEffort.XHIGH).create(name="a", system_prompt="x", model=model, tools=[_lookup])

    handle.run("go")

    assert "reasoning_effort" not in model.calls[0]


def test_a_refusal_naming_reasoning_effort_is_resent_without_it(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_EFFORT_REFUSAL))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup])

    with caplog.at_level(logging.INFO):
        result = handle.run("go")

    assert _efforts(model) == ["xhigh", None]
    assert result.output == "done"
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "reasoning_effort=xhigh" in warnings[0].getMessage()


def test_after_a_refusal_no_agent_of_the_manager_sends_it(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    refused = _RefusingModel(messages=iter([AIMessage(content="done"), AIMessage(content="again")]), error=ValueError(_EFFORT_REFUSAL))
    other = RecordingToolCallingChatModel(messages=iter([AIMessage(content="other")]))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, refused, monkeypatch)
    first = manager.create(name="a", system_prompt="x", tools=[_lookup])
    first.run("go")
    _with_model(manager, other, monkeypatch)
    second = manager.create(name="b", system_prompt="x", tools=[_lookup])
    caplog.clear()  # the refusal's own warning

    with caplog.at_level(logging.INFO):
        first.run("again")
        second.run("go")

    assert _efforts(refused) == ["xhigh", None, None]
    assert "reasoning_effort" not in other.calls[0]
    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


def test_the_resend_keeps_the_output_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_EFFORT_REFUSAL))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], max_output_tokens=4096)

    handle.run("go")

    assert [(c.get("max_tokens"), c.get("reasoning_effort")) for c in model.calls] == [(4096, "xhigh"), (4096, None)]


def test_any_other_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError("connection reset"))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup])

    with pytest.raises(AgentError, match="connection reset"):
        handle.run("go")

    assert len(model.calls) == 1


def test_the_telemetry_summary_records_each_agents_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager(ReasoningEffort.XHIGH)
    manager.enable_telemetry(now=lambda: T0)
    _with_model(manager, RecordingToolCallingChatModel(messages=iter([AIMessage(content="ok")])), monkeypatch)
    manager.create(name="built", system_prompt="x").run("go")
    prebuilt = RecordingToolCallingChatModel(messages=iter([AIMessage(content="ok")]))
    manager.create(name="prebuilt", system_prompt="x", model=prebuilt).run("go")

    summary = manager.telemetry_summary()

    assert summary["built"]["reasoning_effort"] == "xhigh"
    assert summary["prebuilt"]["reasoning_effort"] is None


def test_the_telemetry_summary_shows_no_effort_after_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager(ReasoningEffort.XHIGH)
    manager.enable_telemetry(now=lambda: T0)
    _with_model(manager, _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_EFFORT_REFUSAL)), monkeypatch)
    manager.create(name="a", system_prompt="x").run("go")

    assert manager.telemetry_summary()["a"]["reasoning_effort"] is None


def test_a_real_chat_openai_puts_reasoning_effort_at_the_top_level_of_the_chat_completions_payload() -> None:
    """The fakes only prove `model_settings` reaches `_generate`; this proves what ChatOpenAI then puts on the wire."""
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model="qwen3-8b", base_url="http://localhost:1/v1", api_key=SecretStr("k"))
    # `create_agent` binds `model_settings` to the model, which hands them to `_generate` as kwargs.
    bound = chat.bind(reasoning_effort="xhigh")

    payload = chat._get_request_payload([HumanMessage("hi")], **bound.kwargs)

    assert payload["reasoning_effort"] == "xhigh"
    assert "reasoning" not in payload
    assert chat._use_responses_api(payload) is False  # the Responses API path is the one that rewrites it into `reasoning={"effort": ...}`

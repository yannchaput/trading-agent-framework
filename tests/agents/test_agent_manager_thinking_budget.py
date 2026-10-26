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

# vLLM's refusal of a `thinking_token_budget` it does not accept (shortened).
_BUDGET_REFUSAL = "Error code: 400 - {'message': '`thinking_token_budget` must be a non-negative integer or -1 for unlimited.', 'code': 400}"


def _manager(effort: ReasoningEffort | None = None) -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b", reasoning_effort=effort))


def _with_model(manager: AgentManager, model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `create(model=None)` take the credentials path while running the fake."""
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


class _ModelWithExtraBody(RecordingToolCallingChatModel):
    extra_body: dict[str, Any] | None = None


def _budget(call: dict[str, Any]) -> Any:
    return (call.get("extra_body") or {}).get("thinking_token_budget")


def _budgets(model: RecordingToolCallingChatModel) -> list[Any]:
    return [_budget(call) for call in model.calls]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def test_every_model_call_of_the_run_is_sent_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingToolCallingChatModel(messages=iter([_lookup_call(), AIMessage(content="done")]))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=28672)

    handle.run("go")

    assert _budgets(model) == [28672, 28672]


def test_without_a_budget_nothing_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup])

    handle.run("go")

    assert "extra_body" not in model.calls[0] or "thinking_token_budget" not in model.calls[0]["extra_body"]


def test_a_prebuilt_model_gets_the_budget() -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], thinking_token_budget=500)

    handle.run("go")

    assert _budgets(model) == [500]


def test_the_budget_is_merged_into_an_existing_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ModelWithExtraBody(messages=iter([AIMessage(content="done")]), extra_body={"top_k": 20})
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=500)

    handle.run("go")

    assert model.calls[0]["extra_body"] == {"top_k": 20, "thinking_token_budget": 500}


def test_the_budget_is_merged_with_a_call_level_extra_body_and_the_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain.agents.middleware import wrap_model_call

    @wrap_model_call
    def add_call_level_extra_body(request: Any, handler: Any) -> Any:
        extra_body = {**(request.model_settings.get("extra_body") or {}), "top_p_note": 1}
        return handler(request.override(model_settings={**request.model_settings, "extra_body": extra_body}))

    model = _ModelWithExtraBody(messages=iter([AIMessage(content="done")]), extra_body={"top_k": 20})
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], middleware=[add_call_level_extra_body], thinking_token_budget=500)

    handle.run("go")

    assert model.calls[0]["extra_body"] == {"top_k": 20, "top_p_note": 1, "thinking_token_budget": 500}


def test_a_budget_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="thinking_token_budget must be at least 1, got 0"):
        _manager().create(name="a", system_prompt="x", thinking_token_budget=0)


def test_a_refusal_naming_thinking_token_budget_is_resent_without_it(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_BUDGET_REFUSAL))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=500)

    with caplog.at_level(logging.INFO):
        result = handle.run("go")

    assert _budgets(model) == [500, None]
    assert result.output == "done"
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "thinking_token_budget=500" in warnings[0].getMessage()


def test_after_a_refusal_no_agent_of_the_manager_sends_it(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    refused = _RefusingModel(messages=iter([AIMessage(content="done"), AIMessage(content="again")]), error=ValueError(_BUDGET_REFUSAL))
    other = RecordingToolCallingChatModel(messages=iter([AIMessage(content="other")]))
    manager = _manager()
    _with_model(manager, refused, monkeypatch)
    first = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=500)
    first.run("go")
    _with_model(manager, other, monkeypatch)
    second = manager.create(name="b", system_prompt="x", tools=[_lookup], thinking_token_budget=500)
    caplog.clear()  # the refusal's own warning

    with caplog.at_level(logging.INFO):
        first.run("again")
        second.run("go")

    assert _budgets(refused) == [500, None, None]
    assert _budgets(other) == [None]
    assert _warnings(caplog) == []


def test_the_resend_keeps_the_output_cap_and_the_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_BUDGET_REFUSAL))
    manager = _manager(ReasoningEffort.XHIGH)
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], max_output_tokens=4096, thinking_token_budget=500)

    handle.run("go")

    assert [(c.get("max_tokens"), c.get("reasoning_effort"), _budget(c)) for c in model.calls] == [(4096, "xhigh", 500), (4096, "xhigh", None)]


def test_any_other_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError("connection reset"))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=500)

    with pytest.raises(AgentError, match="connection reset"):
        handle.run("go")

    assert len(model.calls) == 1


def test_a_budget_not_below_the_cap_is_warned_about_at_creation(caplog: pytest.LogCaptureFixture) -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))

    with caplog.at_level(logging.INFO):
        handle = _manager().create(name="judge", system_prompt="x", model=model, max_output_tokens=4096, thinking_token_budget=4096)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "thinking budget of 4096 tokens is not below the output cap of 4096" in message
    assert "judge" in message
    assert handle.run("go").output == "done"


@pytest.mark.parametrize(("cap", "budget"), [(4096, 4095), (None, 500), (4096, None)])
def test_no_cap_warning_when_the_budget_is_below_the_cap_or_either_is_unset(cap: int | None, budget: int | None, caplog: pytest.LogCaptureFixture) -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))

    with caplog.at_level(logging.INFO):
        _manager().create(name="a", system_prompt="x", model=model, max_output_tokens=cap, thinking_token_budget=budget)

    assert _warnings(caplog) == []


def test_the_telemetry_summary_records_each_agents_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0)
    _with_model(manager, RecordingToolCallingChatModel(messages=iter([AIMessage(content="ok")])), monkeypatch)
    manager.create(name="with", system_prompt="x", thinking_token_budget=500).run("go")
    _with_model(manager, RecordingToolCallingChatModel(messages=iter([AIMessage(content="ok")])), monkeypatch)
    manager.create(name="without", system_prompt="x").run("go")

    summary = manager.telemetry_summary()

    assert summary["with"]["thinking_token_budget"] == 500
    assert summary["without"]["thinking_token_budget"] is None


def test_the_telemetry_summary_shows_no_budget_after_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0)
    _with_model(manager, _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_BUDGET_REFUSAL)), monkeypatch)
    manager.create(name="a", system_prompt="x", thinking_token_budget=500).run("go")

    assert manager.telemetry_summary()["a"]["thinking_token_budget"] is None


def test_a_real_chat_openai_puts_the_budget_in_the_request_body() -> None:
    """The fakes only prove `model_settings` reaches `_generate`; this proves what ChatOpenAI then puts on the wire."""
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model="qwen3-8b", base_url="http://localhost:1/v1", api_key=SecretStr("k"), extra_body={"top_k": 20})
    bound = chat.bind(extra_body={"top_k": 20, "thinking_token_budget": 500})

    payload = chat._get_request_payload([HumanMessage("hi")], **bound.kwargs)

    assert payload["extra_body"] == {"top_k": 20, "thinking_token_budget": 500}
    assert "thinking_token_budget" not in payload  # never a top-level OpenAI client argument


def _reasoning_reply(reasoning: int, content: str = "done") -> AIMessage:
    return AIMessage(
        content=content,
        usage_metadata={"input_tokens": 10, "output_tokens": reasoning, "total_tokens": reasoning + 10, "output_token_details": {"reasoning": reasoning}},
    )


def _run_one(reply: AIMessage, budget: int, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, name: str = "judge") -> Any:
    model = RecordingToolCallingChatModel(messages=iter([reply]))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name=name, system_prompt="x", tools=[_lookup], thinking_token_budget=budget)
    with caplog.at_level(logging.INFO):
        return handle.run("go")


def test_reasoning_at_the_budget_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _run_one(_reasoning_reply(936), 1000, monkeypatch, caplog)  # 1000 - max(64, 10)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "thinking budget of 1000 tokens reached (936 reasoning tokens): the server ended the thinking" in message
    assert "judge" in message


def test_reasoning_above_the_budget_says_the_server_did_not_end_the_thinking(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _run_one(_reasoning_reply(1200), 1000, monkeypatch, caplog)

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "thinking budget of 1000 tokens exceeded (1200 reasoning tokens): the server did not end the thinking" in message
    assert "reached" not in message
    assert "judge" in message


def test_reasoning_exactly_at_the_budget_is_still_reached(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _run_one(_reasoning_reply(1000), 1000, monkeypatch, caplog)

    message = _warnings(caplog)[0].getMessage()
    assert "thinking budget of 1000 tokens reached (1000 reasoning tokens): the server ended the thinking" in message


def test_reasoning_just_outside_the_margin_is_not_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _run_one(_reasoning_reply(935), 1000, monkeypatch, caplog)

    assert _warnings(caplog) == []


@pytest.mark.parametrize(("reasoning", "logged"), [(36496, True), (36495, False)])
def test_a_large_budget_uses_a_one_percent_margin(reasoning: int, logged: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _run_one(_reasoning_reply(reasoning), 36864, monkeypatch, caplog)  # 36864 - 368

    assert bool(_warnings(caplog)) is logged


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage(content="done"),
        AIMessage(content="done", usage_metadata={"input_tokens": 10, "output_tokens": 500, "total_tokens": 510}),
    ],
    ids=["no usage", "usage without reasoning details"],
)
def test_no_reported_reasoning_logs_nothing(reply: AIMessage, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    result = _run_one(reply, 500, monkeypatch, caplog)

    assert _warnings(caplog) == []
    assert result.output == "done"


def test_after_a_refusal_nothing_is_logged_even_at_the_old_budget(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = _RefusingModel(messages=iter([_reasoning_reply(500)]), error=ValueError(_BUDGET_REFUSAL))
    manager = _manager()
    _with_model(manager, model, monkeypatch)
    handle = manager.create(name="a", system_prompt="x", tools=[_lookup], thinking_token_budget=500)

    with caplog.at_level(logging.INFO):
        handle.run("go")

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "refused" in warnings[0].getMessage()

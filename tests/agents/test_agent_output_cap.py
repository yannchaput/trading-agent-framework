from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import RecordingToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.utils.errors import AgentError

# vLLM 0.30's refusal of a request whose prompt plus `max_tokens` is larger than the window (observed, shortened).
_WINDOW_REFUSAL = "This model's maximum context length is 32768 tokens. However, you requested 12288 output tokens and your prompt contains 90000 characters."


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


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


def test_every_model_call_of_the_run_is_sent_with_the_cap() -> None:
    model = RecordingToolCallingChatModel(messages=iter([_lookup_call(), AIMessage(content="done")]))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=12288)

    handle.run("go")

    assert [call.get("max_tokens") for call in model.calls] == [12288, 12288]


def test_without_a_cap_no_max_tokens_is_sent() -> None:
    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="done")]))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup])

    handle.run("go")

    assert "max_tokens" not in model.calls[0]


def test_a_call_refused_because_the_cap_does_not_fit_the_window_is_sent_again_without_it() -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError(_WINDOW_REFUSAL))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=12288)

    result = handle.run("go")

    assert [call.get("max_tokens") for call in model.calls] == [12288, None]
    assert result.output == "done"


def test_a_prompt_too_large_for_the_window_even_without_the_cap_fails_after_one_resend() -> None:
    class _AlwaysRefusing(RecordingToolCallingChatModel):
        def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise ValueError(_WINDOW_REFUSAL)

    model = _AlwaysRefusing(messages=iter([]))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=12288)

    with pytest.raises(AgentError, match="maximum context length"):
        handle.run("go")

    assert [call.get("max_tokens") for call in model.calls] == [12288, None]


def test_any_other_failure_is_not_retried() -> None:
    model = _RefusingModel(messages=iter([AIMessage(content="done")]), error=ValueError("connection reset"))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=12288)

    with pytest.raises(AgentError, match="connection reset"):
        handle.run("go")

    assert len(model.calls) == 1


def test_a_call_a_caller_middleware_makes_is_capped_too() -> None:
    @wrap_model_call
    def ask_twice(request: Any, handler: Callable[[Any], Any]) -> Any:
        handler(request)
        return handler(request)

    model = RecordingToolCallingChatModel(messages=iter([AIMessage(content="first"), AIMessage(content="second")]))
    handle = _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=4096, middleware=[ask_twice])

    result = handle.run("go")

    assert [call.get("max_tokens") for call in model.calls] == [4096, 4096]
    assert result.output == "second"


@pytest.mark.parametrize("cap", [0, -1])
def test_a_cap_below_one_token_is_refused(cap: int) -> None:
    model = RecordingToolCallingChatModel(messages=iter([]))

    with pytest.raises(ValueError, match="max_output_tokens"):
        _manager().create(name="a", system_prompt="x", model=model, tools=[_lookup], max_output_tokens=cap)

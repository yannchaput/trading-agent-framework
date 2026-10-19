from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import RecordingToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.strategies.bull_bear.thinking import thinking_guards

_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{index}_{name}") for index, (name, args) in enumerate(calls)])


def _at_the_cap(message: AIMessage | None = None) -> AIMessage:
    """A turn that spent the output cap: `finish_reason="length"`, nothing written but what `message` holds."""
    stopped = message or AIMessage(content="")
    stopped.response_metadata = {"finish_reason": "length"}
    return stopped


class _Agent:
    """A researcher-shaped agent with both guards, a `lookup` tool and its `submit` tool (a final tool)."""

    def __init__(self, responses: list[AIMessage], *, submit_results: list[dict[str, str]] | None = None, lookup_result: Any = "A ok", outer: list[Any] | None = None) -> None:
        self.logged: list[str] = []
        results = list(submit_results or [])

        def lookup(symbol: str) -> Any:
            """Look up a symbol."""
            return lookup_result

        def submit(answer: str) -> dict[str, str]:
            """Submit the answer."""
            return results.pop(0) if results else {"status": "recorded"}

        self.model = RecordingToolCallingChatModel(messages=iter(responses))
        guards = [*(outer or []), *thinking_guards("researcher", "submit", self.logged.append)]
        manager = AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))
        self.handle: AgentHandle = manager.create(name="researcher", system_prompt="x", model=self.model, tools=[lookup, submit], final_tools=["submit"], middleware=guards)

    def thinking(self) -> list[bool]:
        """Per model call, whether thinking was left on (no `extra_body` turning it off)."""
        return [call.get("extra_body") != _THINKING_OFF for call in self.model.calls]


# --- a turn that stops at the cap --------------------------------------------------------------------


def test_a_turn_stopped_at_the_cap_without_a_tool_call_is_asked_again_with_thinking_off_and_the_submit_tool_forced() -> None:
    agent = _Agent([_calls(("lookup", {"symbol": "A"})), _at_the_cap(), _calls(("submit", {"answer": "x"}))])

    result = agent.handle.run("go")

    assert agent.thinking() == [True, True, False]
    assert agent.model.calls[2]["tool_choice"] == "submit"
    assert [call.name for call in result.tool_calls] == ["lookup", "submit"]  # the research is still in the run
    assert len(agent.logged) == 1 and "researcher" in agent.logged[0]


def test_a_turn_stopped_at_the_cap_with_a_tool_call_is_kept() -> None:
    agent = _Agent([_at_the_cap(_calls(("lookup", {"symbol": "A"}))), AIMessage(content="done")])

    result = agent.handle.run("go")

    assert agent.thinking() == [True, True]
    assert result.output == "done"


def test_a_turn_that_ended_normally_is_not_asked_again() -> None:
    finished = AIMessage(content="done", response_metadata={"finish_reason": "stop"})
    agent = _Agent([finished, AIMessage(content="never asked")])

    result = agent.handle.run("go")

    assert len(agent.model.calls) == 1
    assert result.output == "done"


def test_a_turn_already_without_thinking_that_stops_at_the_cap_is_not_asked_again() -> None:
    rejected = {"error": "the note is 537 characters: keep it under 500"}
    agent = _Agent([_calls(("submit", {"answer": "long"})), _at_the_cap(), AIMessage(content="never asked")], submit_results=[rejected])

    agent.handle.run("go")

    assert agent.thinking() == [True, False]  # the turn after the rejection, then nothing more


# --- a rejected submission ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first_submission",
    [
        pytest.param({"answer": "long"}, id="refused by the handoff"),
        pytest.param({}, id="arguments the tool cannot take"),
    ],
)
def test_after_a_rejected_submission_the_next_turn_runs_with_thinking_off_and_the_submit_tool_forced(first_submission: dict[str, str]) -> None:
    refusals = [{"error": "the note is 537 characters: keep it under 500"}] if first_submission else []
    agent = _Agent([_calls(("submit", first_submission)), _calls(("submit", {"answer": "short"}))], submit_results=refusals)

    result = agent.handle.run("go")

    assert agent.thinking() == [True, False]
    assert agent.model.calls[1]["tool_choice"] == "submit"
    assert [call.name for call in result.tool_calls] == ["submit", "submit"]
    assert len(agent.logged) == 1 and "researcher" in agent.logged[0]


def test_a_run_whose_submissions_keep_being_rejected_ends_at_the_third_rejection() -> None:
    # Forced and without thinking, the model's only way out is a valid submission: without a limit (and with compaction
    # keeping the history in the window) a submission it cannot get right would loop up to LangGraph's 9,999 steps.
    rejected = {"error": "the note is 537 characters: keep it under 500"}
    agent = _Agent([_calls(("submit", {"answer": "long"})) for _ in range(6)], submit_results=[rejected] * 6)

    result = agent.handle.run("go")

    assert len(agent.model.calls) == 3  # the first try and two corrections
    assert [call.name for call in result.tool_calls] == ["submit", "submit", "submit"]
    assert result.output == ""  # the run ends with nothing recorded: the pipeline's forced retry, then abandonment, take over
    assert "researcher" in agent.logged[-1] and "3" in agent.logged[-1]


def test_the_rejection_warning_says_why_the_submission_was_rejected() -> None:
    # The warning is logged while the run is going, before the pipeline logs the run: it must name the stock and the reason itself.
    rejected = {"error": "the note for RKLB is 506 characters: keep it under 500"}
    agent = _Agent([_calls(("submit", {"answer": "long"})), _calls(("submit", {"answer": "short"}))], submit_results=[rejected])

    agent.handle.run("go")

    assert "the note for RKLB is 506 characters: keep it under 500" in agent.logged[0]


def test_the_warning_for_a_run_ended_by_rejections_says_why_the_last_one_was_rejected() -> None:
    rejected = [{"error": "the note for CRDO is 504 characters: keep it under 500"}] * 6
    agent = _Agent([_calls(("submit", {"answer": "long"})) for _ in range(6)], submit_results=rejected)

    agent.handle.run("go")

    assert "the note for CRDO is 504 characters: keep it under 500" in agent.logged[-1]


def test_a_very_long_rejection_reason_is_cut_in_the_warning() -> None:
    agent = _Agent([_calls(("submit", {"answer": "long"})), _calls(("submit", {"answer": "short"}))], submit_results=[{"error": "x" * 5000}])

    agent.handle.run("go")

    assert len(agent.logged[0]) < 600


def test_turning_thinking_off_keeps_the_other_settings_of_the_call() -> None:
    @wrap_model_call
    def server_options(request: Any, handler: Callable[[Any], Any]) -> Any:
        extra_body = {"top_k": 20, "chat_template_kwargs": {"preserve_thinking": True}}
        return handler(request.override(model_settings={"presence_penalty": 1.5, "extra_body": extra_body}))

    rejected = {"error": "the note is 537 characters: keep it under 500"}
    agent = _Agent([_calls(("submit", {"answer": "long"})), _calls(("submit", {"answer": "short"}))], submit_results=[rejected], outer=[server_options])

    agent.handle.run("go")

    correction = agent.model.calls[1]
    assert correction["presence_penalty"] == 1.5
    assert correction["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": False}}


def test_a_failed_call_of_another_tool_leaves_thinking_on() -> None:
    agent = _Agent([_calls(("lookup", {"symbol": "A"})), _calls(("submit", {"answer": "x"}))], lookup_result={"error": "no filings"})

    agent.handle.run("go")

    assert agent.thinking() == [True, True]
    assert agent.model.calls[1]["tool_choice"] is None
    assert agent.logged == []

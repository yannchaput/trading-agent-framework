"""A corrective retry (`news_binary`'s `on_trading_iteration`) asks the model to call `remember_decision`

by prompt alone -- observed unreliable with `glm-4.7-flash`: the model satisfied every prerequisite,
then wrote a closing summary claiming it had called `remember_decision` without ever emitting the tool
call. `AgentHandle.run(..., force_tool=...)` makes the run's first model turn a real API-level forced
tool choice instead of a text instruction, via a `tool_choice` override in `AgentManager`'s middleware.
Later turns in the same run are left unforced so the model can still close out normally afterward.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager


def _credentials() -> LLMCredentials:
    return LLMCredentials(base_url="http://localhost:8000/v1", api_key="key", default_model="qwen3-8b")


def _manager() -> AgentManager:
    return AgentManager(_credentials)


class _CapturingModel(FakeToolCallingChatModel):
    """`FakeToolCallingChatModel` that records every `tool_choice` its `bind_tools` receives."""

    def __init__(self, messages: list[AIMessage]) -> None:
        super().__init__(messages=iter(messages))
        object.__setattr__(self, "bind_tools_calls", [])

    def bind_tools(self, tools: object, **kwargs: object) -> _CapturingModel:
        self.bind_tools_calls.append(kwargs)
        return self


def remember_decision(text: str) -> dict[str, str]:
    """Record a decision."""
    return {"id": "decision_abc123", "status": "recorded"}


def test_force_tool_sets_tool_choice_on_the_runs_first_model_turn_only() -> None:
    model = _CapturingModel(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="remember_decision", args={"text": "KEEP"}, id="call_1")]),
            AIMessage(content="Recorded."),
        ]
    )
    manager = _manager()
    handle = manager.create(name="trader", system_prompt="x", tools=[remember_decision], model=model)

    handle.run("decide", force_tool="remember_decision")

    assert [call["tool_choice"] for call in model.bind_tools_calls] == ["remember_decision", None]


def test_without_force_tool_the_tool_choice_is_never_set() -> None:
    model = _CapturingModel([AIMessage(content="Hold.")])
    manager = _manager()
    handle = manager.create(name="trader", system_prompt="x", tools=[remember_decision], model=model)

    handle.run("decide")

    assert [call["tool_choice"] for call in model.bind_tools_calls] == [None]

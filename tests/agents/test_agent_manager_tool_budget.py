from __future__ import annotations

import itertools
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager, ToolBudget, _tool_budget_message


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


_call_ids = itertools.count()  # unique across turns: the run's record pairs a call with its answer by id


def _calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{next(_call_ids)}_{name}") for name, args in calls])


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
    return _manager().create(name="seller", system_prompt="x", model=model, tools=[lookup, submit], budget=ToolBudget(exempt=tuple(exempt_tools or ())))


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


def test_an_unknown_tool_name_does_not_use_up_the_budget() -> None:
    seen: list[str] = []
    handle = _agent(seen, _calls(("bogus", {})), _calls(("lookup", {"symbol": "A"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=1)

    assert seen == ["A"]  # bogus was not counted, so lookup still had the one slot
    assert not any("tool budget" in call.result for call in result.tool_calls)


def test_several_calls_in_one_turn_count_in_the_order_they_were_requested() -> None:
    seen: list[str] = []
    handle = _agent(seen, _calls(("lookup", {"symbol": "A"}), ("lookup", {"symbol": "B"}), ("lookup", {"symbol": "C"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=2)

    assert sorted(seen) == ["A", "B"]  # C, the third request, is the one refused
    refused = [call for call in result.tool_calls if "tool budget" in call.result]
    assert [call.args for call in refused] == [{"symbol": "C"}]


def test_an_exempt_call_in_a_turn_that_overruns_the_budget_still_runs() -> None:
    seen: list[str] = []
    handle = _agent(
        seen,
        _calls(("lookup", {"symbol": "A"}), ("lookup", {"symbol": "B"}), ("submit", {"answer": "x"}), ("lookup", {"symbol": "C"})),
        AIMessage(content="done"),
        exempt_tools=["submit"],
    )

    result = handle.run("go", tool_budget=2)

    assert sorted(seen) == ["A", "B", "submit:x"]
    assert [call.name for call in result.tool_calls if "tool budget" in call.result] == ["lookup"]


def test_a_refused_call_reaches_the_model_marked_as_an_error() -> None:
    # `_tool_succeeded` (the final-tools gate) and the provider both read `status`, so a refusal must not look like a success.
    class RecordingModel(FakeToolCallingChatModel):
        seen_inputs: list[list[BaseMessage]] = []

        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            self.seen_inputs.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return f"{symbol} ok"

    model = RecordingModel(messages=iter([_calls(("lookup", {"symbol": "A"})), _calls(("lookup", {"symbol": "B"})), AIMessage(content="done")]))
    model.seen_inputs = []
    handle = _manager().create(name="seller", system_prompt="x", model=model, tools=[lookup])

    handle.run("go", tool_budget=1)

    last_turn_tool_messages = [m for m in model.seen_inputs[-1] if isinstance(m, ToolMessage)]
    assert [m.status for m in last_turn_tool_messages] == ["success", "error"]


def test_each_refusal_answers_the_call_it_refuses_with_the_budget_error() -> None:
    # The refusal is rebuilt from the base class's messages, so its `tool_call_id` must still be the refused call's id.
    class RecordingModel(FakeToolCallingChatModel):
        seen_inputs: list[list[BaseMessage]] = []

        def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
            self.seen_inputs.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return f"{symbol} ok"

    refused_call = _calls(("lookup", {"symbol": "A"}), ("lookup", {"symbol": "B"}))
    model = RecordingModel(messages=iter([refused_call, AIMessage(content="done")]))
    model.seen_inputs = []
    handle = _manager().create(name="seller", system_prompt="x", model=model, tools=[lookup])

    handle.run("go", tool_budget=1)

    last_turn_tool_messages = [m for m in model.seen_inputs[-1] if isinstance(m, ToolMessage)]
    by_id = {m.tool_call_id: m for m in last_turn_tool_messages}
    allowed_id, refused_id = (call["id"] for call in refused_call.tool_calls)
    assert set(by_id) == {allowed_id, refused_id}
    assert by_id[allowed_id].content == "A ok"
    assert json.loads(str(by_id[refused_id].content)) == {"error": "tool budget of 1 calls spent; lookup was not run. Answer now without calling another tool."}


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


def test_the_refusal_names_the_final_tools_not_every_exempt_tool() -> None:
    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        return f"{symbol} ok"

    def statement(symbol: str) -> str:
        """Read a statement."""
        return f"{symbol} statement"

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        return {"status": "recorded"}

    model = FakeToolCallingChatModel(
        messages=iter([_calls(("lookup", {"symbol": "A"})), _calls(("lookup", {"symbol": "B"})), _calls(("statement", {"symbol": "A"})), _calls(("submit", {"answer": "x"}))])
    )
    handle = _manager().create(name="r", system_prompt="x", model=model, tools=[lookup, statement, submit], budget=ToolBudget(exempt=("submit", "statement")), final_tools=["submit"])

    result = handle.run("go", tool_budget=1)

    refused = [call for call in result.tool_calls if call.name == "lookup"][1]
    assert json.loads(refused.result) == {"error": "tool budget of 1 calls spent; lookup was not run. Finish now: call submit."}  # statement is exempt but does not finish the run
    assert [call.result for call in result.tool_calls if call.name == "statement"] == ["A statement"]  # the exempt tool still ran past the budget


def test_tool_budget_defaults_count_every_tool_and_label_nothing() -> None:
    assert ToolBudget() == ToolBudget(exempt=(), label_calls=False)


# --- opt-in "<tool> call k of N" labels on counted calls ----------------------------------------------------------


def _label_agent(*messages: AIMessage, label_calls: bool = True) -> AgentHandle:
    def quote(symbol: str) -> dict[str, Any]:
        """Quote a symbol."""
        return {"facts": {"symbol": symbol}}

    def broken() -> dict[str, str]:
        """A tool that always fails."""
        return {"error": "bad"}

    def plain() -> str:
        """A tool that answers in plain text."""
        return "just text"

    def odd() -> dict[str, str]:
        """A tool whose `instructions` is not a block."""
        return {"instructions": "read me"}

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        return {"status": "recorded"}

    model = FakeToolCallingChatModel(messages=iter(list(messages)))
    budget = ToolBudget(exempt=("submit",), label_calls=label_calls)
    return _manager().create(name="labeller", system_prompt="x", model=model, tools=[quote, broken, plain, odd, submit], budget=budget)


def _answers(result: Any, tool: str) -> list[dict[str, Any]]:
    return [json.loads(call.result) for call in result.tool_calls if call.name == tool]


def test_counted_calls_are_labelled_with_their_number_and_the_budget() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"})), _calls(("quote", {"symbol": "B"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=3)

    answers = _answers(result, "quote")
    assert [a["instructions"]["tool_call"] for a in answers] == ["quote call 1 of 3", "quote call 2 of 3"]
    assert list(answers[0]) == ["instructions", "facts"]  # the instructions come first


def test_parallel_calls_in_one_turn_are_numbered_in_request_order() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"}), ("quote", {"symbol": "B"}), ("quote", {"symbol": "C"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=7)

    labels = {a["facts"]["symbol"]: a["instructions"]["tool_call"] for a in _answers(result, "quote")}
    assert labels == {"A": "quote call 1 of 7", "B": "quote call 2 of 7", "C": "quote call 3 of 7"}


def test_an_error_answer_is_labelled_beside_the_error() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"})), _calls(("broken", {})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=2)

    assert _answers(result, "broken") == [{"instructions": {"tool_call": "broken call 2 of 2"}, "error": "bad"}]


def test_exempt_calls_are_not_labelled() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"}), ("submit", {"answer": "x"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=2)

    assert _answers(result, "submit") == [{"status": "recorded"}]
    assert _answers(result, "quote")[0]["instructions"]["tool_call"] == "quote call 1 of 2"


def test_a_refused_call_keeps_the_budget_error_and_no_label() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"})), _calls(("quote", {"symbol": "B"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=1)

    refused = _answers(result, "quote")[1]
    assert refused == {"error": "tool budget of 1 calls spent; quote was not run. Finish now: call submit."}


def test_labels_are_opt_in() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"})), AIMessage(content="done"), label_calls=False)

    result = handle.run("go", tool_budget=3)

    assert _answers(result, "quote") == [{"facts": {"symbol": "A"}}]


def test_no_label_without_a_run_budget() -> None:
    handle = _label_agent(_calls(("quote", {"symbol": "A"})), AIMessage(content="done"))

    result = handle.run("go")

    assert _answers(result, "quote") == [{"facts": {"symbol": "A"}}]


def test_an_answer_that_is_not_a_json_object_is_left_alone() -> None:
    handle = _label_agent(_calls(("plain", {})), _calls(("odd", {})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=3)

    assert [call.result for call in result.tool_calls if call.name == "plain"] == ["just text"]
    assert _answers(result, "odd") == [{"instructions": "read me"}]

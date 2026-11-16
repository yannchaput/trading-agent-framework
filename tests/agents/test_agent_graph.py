from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from trading_agent_framework.agents.graph import AgentNode, SubmitSlot, merge_dicts
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.agents.tools.text import LengthGate, TextLimit
from trading_agent_framework.utils.errors import AgentError, AgentLoopError

Script = Callable[[SubmitSlot, Mapping[str, Any] | None, str | None], Any]


class ScriptedAgent:
    """Pops the next script on each run and calls it with (slot, context, force_tool)."""

    def __init__(self, slot: SubmitSlot, scripts: list[Script]) -> None:
        self.slot = slot
        self.scripts = scripts
        self.runs: list[dict[str, Any]] = []

    def run(
        self,
        task_prompt: str,
        *,
        context: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        force_tool: str | None = None,
        tool_budget: int | None = None,
        max_steps: int | None = None,
    ) -> AgentRunResult:
        self.runs.append({"task": task_prompt, "context": context, "force_tool": force_tool})
        self.scripts.pop(0)(self.slot, context, force_tool)
        return AgentRunResult(output="", tool_calls=[])


def _submits(payload: dict[str, Any]) -> Script:
    return lambda slot, context, force_tool: slot.submit(payload)


def _nothing(slot: SubmitSlot, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
    return None


def _raises(exc: Exception) -> Script:
    def script(slot: SubmitSlot, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
        raise exc

    return script


def _node(scripts: list[Script]) -> tuple[AgentNode, ScriptedAgent]:
    slot = SubmitSlot(lambda payload, state: None)
    agent = ScriptedAgent(slot, scripts)
    node = AgentNode(
        name="n",
        agent=agent,
        slot=slot,
        submit_tool="submit_x",
        task="Do it.",
        reads=lambda state: {"seen": state["in"]},
        write=lambda value: {"out": value},
    )
    return node, agent


def test_merge_dicts_combines_parallel_writes() -> None:
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts(None, {"b": 2}) == {"b": 2}


def test_submit_slot_refuses_invalid_payload() -> None:
    slot = SubmitSlot(lambda payload, state: "bad" if payload["x"] == 0 else None)
    assert slot.submit({"x": 0}) == {"error": "bad"}
    assert slot.value is None
    assert slot.submit({"x": 1}) == {"status": "recorded"}
    assert slot.value == {"x": 1}
    slot.arm({})
    assert slot.value is None


def test_submit_slot_validator_sees_armed_state() -> None:
    slot = SubmitSlot(lambda payload, state: None if payload in state["allowed"] else "not allowed")
    slot.arm({"allowed": ["A"]})
    assert slot.submit("B") == {"error": "not allowed"}
    assert slot.submit("A") == {"status": "recorded"}


def test_node_writes_the_submission() -> None:
    node, agent = _node([_submits({"v": 1})])
    assert node({"in": 7}) == {"out": {"v": 1}}
    assert len(agent.runs) == 1
    assert agent.runs[0]["context"] == {"seen": 7}


def test_node_retries_once_with_tool_forced(caplog: pytest.LogCaptureFixture) -> None:
    node, agent = _node([_nothing, _submits({"v": 2})])
    with caplog.at_level("WARNING"):
        assert node({"in": 1}) == {"out": {"v": 2}}
    assert len(agent.runs) == 2
    assert agent.runs[0]["force_tool"] is None
    assert agent.runs[1]["force_tool"] == "submit_x"
    assert agent.runs[1]["task"].endswith("Call submit_x now.")
    assert "[n] no valid submit_x: retrying with it forced" in caplog.text


def test_node_marks_unavailable_after_failed_retry() -> None:
    node, agent = _node([_nothing, _nothing])
    assert node({"in": 1}) == {"out": None, "errors": [{"node": "n", "error": "no valid submission"}]}
    assert len(agent.runs) == 2


def test_node_retries_after_agent_error() -> None:
    node, agent = _node([_raises(AgentError("boom")), _submits({"v": 3})])
    assert node({"in": 1}) == {"out": {"v": 3}}
    assert len(agent.runs) == 2


def test_node_does_not_retry_a_loop() -> None:
    node, agent = _node([_raises(AgentLoopError("spin"))])
    result = node({"in": 1})
    assert len(agent.runs) == 1
    assert result["out"] is None
    assert result["errors"][0]["error"].startswith("loop:")


_LONG = [("a", "x" * 11, TextLimit(10, 20))]


def test_arming_a_slot_resets_its_gate() -> None:
    gate = LengthGate(2, name="submit", log_tag="t")
    slot = SubmitSlot(lambda payload, state: None, gate=gate)
    assert "error" in gate.review(_LONG)
    slot.arm({})
    assert "error" in gate.review(_LONG)
    assert slot.gate is gate
    assert SubmitSlot(lambda payload, state: None).gate is None


def test_the_forced_retry_shares_the_gate_attempts_and_the_next_run_starts_fresh() -> None:
    gate = LengthGate(2, name="submit_x", log_tag="t")
    slot = SubmitSlot(lambda payload, state: None, gate=gate)

    def review_then_submit(slot: SubmitSlot, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
        checked = gate.review(_LONG)
        if isinstance(checked, list):
            slot.submit({"text": checked[0]})

    agent = ScriptedAgent(slot, [review_then_submit] * 4)
    node = AgentNode(name="n", agent=agent, slot=slot, submit_tool="submit_x", task="Do it.", reads=lambda state: {}, write=lambda value: {"out": value})
    assert node({}) == {"out": {"text": "x" * 11}}
    assert [run["force_tool"] for run in agent.runs] == [None, "submit_x"]  # refused once, accepted on the forced retry
    assert node({}) == {"out": {"text": "x" * 11}}
    assert [run["force_tool"] for run in agent.runs] == [None, "submit_x", None, "submit_x"]  # a new run is refused again first

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from trading_agent_framework.agents.graph import AgentNode, merge_dicts
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.agents.submit import HandoffError, NodeRecorder
from trading_agent_framework.agents.tools.text import LengthGate, TextLimit
from trading_agent_framework.utils.errors import AgentError, AgentLoopError

Script = Callable[[NodeRecorder, Mapping[str, Any] | None, str | None], Any]


class ScriptedAgent:
    """Pops the next script on each run and calls it with (recorder, context, force_tool)."""

    def __init__(self, recorder: NodeRecorder, scripts: list[Script]) -> None:
        self.recorder = recorder
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
        self.scripts.pop(0)(self.recorder, context, force_tool)
        return AgentRunResult(output="", tool_calls=[])


def _submits(payload: dict[str, Any]) -> Script:
    return lambda recorder, context, force_tool: recorder.submit("submit_x", payload)


def _nothing(recorder: NodeRecorder, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
    return None


def _raises(exc: Exception) -> Script:
    def script(recorder: NodeRecorder, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
        raise exc

    return script


def _node(scripts: list[Script]) -> tuple[AgentNode, ScriptedAgent]:
    recorder = NodeRecorder("submit_x", lambda payload, state: None)
    agent = ScriptedAgent(recorder, scripts)
    node = AgentNode(
        name="n",
        agent=agent,
        recorder=recorder,
        submit_tool="submit_x",
        task="Do it.",
        reads=lambda state: {"seen": state["in"]},
        write=lambda value: {"out": value},
    )
    return node, agent


def test_merge_dicts_combines_parallel_writes() -> None:
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_dicts(None, {"b": 2}) == {"b": 2}


def _refuse_bad(payload: Any, state: Mapping[str, Any]) -> None:
    if payload["x"] == 0:
        raise HandoffError("bad")


def test_node_recorder_refuses_an_invalid_payload_and_keeps_the_first_valid_one() -> None:
    recorder = NodeRecorder("submit_x", _refuse_bad)
    recorder.arm({})
    assert recorder.submit("submit_x", {"x": 0}) == {"error": "bad"}
    assert recorder.submission is None
    assert recorder.submit("submit_x", {"x": 1}) == {"status": "recorded"}
    assert recorder.submit("submit_x", {"x": 2}) == {"error": "already recorded for this run"}  # first wins
    assert recorder.submission == {"x": 1}
    recorder.arm({})
    assert recorder.submission is None


def test_node_recorder_validator_sees_armed_state() -> None:
    def check(payload: Any, state: Mapping[str, Any]) -> None:
        if payload not in state["allowed"]:
            raise HandoffError("not allowed")

    recorder = NodeRecorder("submit_x", check)
    recorder.arm({"allowed": ["A"]})
    assert recorder.submit("submit_x", "B") == {"error": "not allowed"}
    assert recorder.submit("submit_x", "A") == {"status": "recorded"}


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


def test_arming_a_recorder_resets_its_gate() -> None:
    gate = LengthGate(2, name="submit", log_tag="t")
    recorder = NodeRecorder("submit_x", lambda payload, state: None, gate=gate)
    assert "error" in gate.review(_LONG)
    recorder.arm({})
    assert "error" in gate.review(_LONG)
    assert recorder.gate is gate
    assert NodeRecorder("submit_x", lambda payload, state: None).gate is None


def test_the_forced_retry_shares_the_gate_attempts_and_the_next_run_starts_fresh() -> None:
    gate = LengthGate(2, name="submit_x", log_tag="t")
    recorder = NodeRecorder("submit_x", lambda payload, state: None, gate=gate)

    def review_then_submit(recorder: NodeRecorder, context: Mapping[str, Any] | None, force_tool: str | None) -> None:
        checked = gate.review(_LONG)
        if isinstance(checked, list):
            recorder.submit("submit_x", {"text": checked[0]})

    agent = ScriptedAgent(recorder, [review_then_submit] * 4)
    node = AgentNode(name="n", agent=agent, recorder=recorder, submit_tool="submit_x", task="Do it.", reads=lambda state: {}, write=lambda value: {"out": value})
    assert node({}) == {"out": {"text": "x" * 11}}
    assert [run["force_tool"] for run in agent.runs] == [None, "submit_x"]  # refused once, accepted on the forced retry
    assert node({}) == {"out": {"text": "x" * 11}}
    assert [run["force_tool"] for run in agent.runs] == [None, "submit_x", None, "submit_x"]  # a new run is refused again first

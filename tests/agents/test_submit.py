from __future__ import annotations

from typing import Any

from trading_agent_framework.agents.submit import HandoffError, NodeRecorder, SubmitRecorder
from trading_agent_framework.agents.tools.text import LengthGate, TextLimit

LIMIT = TextLimit(stated=5, hard=10)


class _Toy(SubmitRecorder):
    TOOL_NAMES = {"a": "submit_a", "b": "submit_b"}
    NOT_EXPECTED_SUFFIX = " here"
    ALREADY_RECORDED = "done already"

    def _validate(self, stage: str, *args: Any) -> Any:
        if args[0] == "bad":
            raise HandoffError("bad value")
        return args[0]


def test_toy_recorder_guards_records_and_rearms() -> None:
    toy = _Toy()
    assert toy.missing_tool == ""
    assert toy.submit("a", 1) == {"error": "submit_a is not expected at this point here"}
    toy._arm("a", {})
    assert toy.missing_tool == "submit_a"
    assert toy.submit("b", 1) == {"error": "submit_b is not expected at this point here"}
    assert toy.submit("a", "bad") == {"error": "bad value"}
    assert toy.last_error == "bad value" and not toy.submitted and toy.submission is None
    assert toy.submit("a", 5) == {"status": "recorded"}
    assert toy.submitted and toy.submission == 5 and toy.last_error is None
    assert toy.submit("a", 6) == {"error": "done already"}
    assert toy.submission == 5
    toy._arm("b", {})
    assert not toy.submitted and toy.submission is None and toy.last_error is None


def _gate() -> LengthGate:
    return LengthGate(2, name="submit_a", log_tag="[t]")


class _Gated(SubmitRecorder):
    TOOL_NAMES = {"a": "submit_a", "b": "submit_b"}

    def _validate(self, stage: str, *args: Any) -> Any:
        return args[0]


def _spend(gate: LengthGate) -> None:
    assert "error" in gate.review([("x", "way too long", LIMIT)])


def test_arming_a_stage_resets_only_its_gate() -> None:
    gate_a, gate_b = _gate(), _gate()
    recorder = _Gated({"a": gate_a, "b": gate_b})
    _spend(gate_a)
    _spend(gate_b)
    recorder._arm("a", {})
    assert recorder.gate is gate_a
    assert isinstance(gate_a.review([("x", "way too long", LIMIT)]), dict)
    assert isinstance(gate_b.review([("x", "way too long", LIMIT)]), list)


def test_submit_texts_spends_no_attempt_when_the_stage_is_not_armed_or_already_recorded() -> None:
    calls: list[LengthGate] = []

    def review(gate: LengthGate) -> tuple[Any, ...]:
        calls.append(gate)
        return ("ok",)

    recorder = _Gated({"a": _gate()})
    assert recorder.submit_texts("a", review, "x") == {"error": "submit_a is not expected at this point"}
    recorder._arm("a", {})
    assert recorder.submit_texts("a", review, "x") == {"status": "recorded"}
    assert recorder.submit_texts("a", review, "x") == {"error": "already recorded for this run"}
    assert len(calls) == 1


def test_a_gate_refusal_is_returned_and_kept_in_last_error() -> None:
    recorder = _Gated({"a": _gate()})
    recorder._arm("a", {})
    assert recorder.submit_texts("a", lambda gate: {"error": "too long: x"}, "v") == {"error": "too long: x"}
    assert recorder.last_error == "too long: x" and not recorder.submitted


def test_submit_texts_records_the_reviewed_args() -> None:
    recorder = _Gated({"a": _gate()})
    recorder._arm("a", {})
    assert recorder.submit_texts("a", lambda gate: ("trimmed",), "long") == {"status": "recorded"}
    assert recorder.submission == "trimmed"


def test_submit_texts_without_a_gate_submits_the_args_unchanged() -> None:
    recorder = _Gated()
    recorder._arm("a", {})

    def review(gate: LengthGate) -> tuple[Any, ...]:
        raise AssertionError("no gate, no review")

    assert recorder.submit_texts("a", review, "same") == {"status": "recorded"}
    assert recorder.submission == "same" and recorder.gate is None


def _allowed(payload: Any, state: Any) -> None:
    if payload not in state["allowed"]:
        raise HandoffError("not allowed")


def test_node_recorder_validates_against_the_armed_state() -> None:
    recorder = NodeRecorder("submit_x", _allowed)
    recorder.arm({"allowed": [1]})
    assert recorder.submit("submit_x", 2) == {"error": "not allowed"}
    assert recorder.submit("submit_x", 1) == {"status": "recorded"}
    assert recorder.submission == 1 and recorder.missing_tool == "submit_x"
    assert recorder.tool == "submit_x"

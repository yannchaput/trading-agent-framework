from __future__ import annotations

from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdict, submit_tools
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

DEBATE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _tools() -> tuple[HandoffRecorder, dict[str, Any]]:
    recorder = HandoffRecorder(BullBearParams())
    return recorder, submit_tools(recorder)


def _bull(symbols: list[str], conviction: str = "high") -> list[dict[str, Any]]:
    return [{"symbol": s, "conviction": conviction, "argument": "strong trend"} for s in symbols]


def _bear(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "risk": "low", "concern": "none", "argument": "no red flag"} for s in symbols]


def _verdicts(keep: list[str], veto: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": "keep", "reason": "bull wins"} for s in keep] + [{"symbol": s, "verdict": "veto", "reason": "bear wins"} for s in veto or []]


# --- note ---------------------------------------------------------------------------------------


def test_a_note_for_the_armed_symbol_is_recorded_stripped() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    assert tools["submit_note"](" aaa ", "  2026-09-30: revenue up 12% y/y  ") == {"status": "recorded"}
    assert recorder.submitted and recorder.submission == Note("AAA", "2026-09-30: revenue up 12% y/y")


def test_a_note_for_another_symbol_is_refused_with_the_symbol_to_use() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    result = tools["submit_note"]("BBB", "facts")

    assert result == {"error": "this note is for AAA: submit it with symbol AAA"} and not recorder.submitted


def test_an_empty_or_long_note_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    assert "empty" in tools["submit_note"]("AAA", "   ")["error"]
    assert "under 500" in tools["submit_note"]("AAA", "x" * 501)["error"]


# --- bull and bear ------------------------------------------------------------------------------


def test_a_bull_case_covering_every_stock_once_is_recorded_with_normalised_values() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)
    cases = _bull(DEBATE)
    cases[0] = {"symbol": " aaa ", "conviction": " HIGH ", "argument": " strong trend "}

    assert tools["submit_bull_case"](cases) == {"status": "recorded"}
    assert recorder.submission[0] == BullCase("AAA", "high", "strong trend")
    assert [case.symbol for case in recorder.submission] == DEBATE


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (_bull(DEBATE[:-1]), "no case for FFF"),
        (_bull([*DEBATE, "AAA"]), "AAA appears twice"),
        (_bull([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_bull(DEBATE, conviction="huge"), "conviction for AAA must be one of low, medium, high"),
        ([{"symbol": "AAA", "conviction": "high", "argument": "x" * 301}, *_bull(DEBATE[1:])], "under 300"),
        ("not a list", "cases must be a list"),
    ],
)
def test_a_bad_bull_case_is_refused(cases: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)

    assert message in tools["submit_bull_case"](cases)["error"]
    assert recorder.last_error is not None and message in recorder.last_error


def test_a_bear_case_needs_a_known_concern() -> None:
    recorder, tools = _tools()
    recorder.expect_bear(DEBATE)
    cases = _bear(DEBATE)
    cases[1] = {"symbol": "BBB", "risk": "high", "concern": "weather", "argument": "storms"}

    assert "concern for BBB must be one of" in tools["submit_bear_case"](cases)["error"]

    cases[1]["concern"] = "Momentum_Exhaustion"
    assert tools["submit_bear_case"](cases) == {"status": "recorded"}
    assert recorder.submission[1] == BearCase("BBB", "high", "momentum_exhaustion", "storms")


# --- verdicts -----------------------------------------------------------------------------------


def test_the_judge_has_submit_verdicts_and_no_picker_tool_is_left() -> None:
    _, tools = _tools()

    assert set(tools) == {"submit_note", "submit_bull_case", "submit_bear_case", "submit_verdicts"}


def test_a_verdict_for_every_stock_is_recorded_with_its_vetoes() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=2)

    result = tools["submit_verdicts"](_verdicts(["AAA", "BBB", "CCC", "DDD"], ["EEE", "FFF"]))

    assert result == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "keep", "bull wins")
    assert recorder.submission.vetoed == ["EEE", "FFF"]


def test_keeping_every_stock_is_valid_even_with_no_veto_allowed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=0)

    assert tools["submit_verdicts"](_verdicts(DEBATE)) == {"status": "recorded"}
    assert recorder.submission.vetoed == []


def test_verdicts_and_symbols_are_normalised_like_the_debaters_cases() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=1)

    raw = [{"symbol": " aaa ", "verdict": "Veto ", "reason": "  lawsuit filed  "}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "veto", "lawsuit filed")


@pytest.mark.parametrize(
    ("verdicts", "message"),
    [
        ({"AAA": "keep"}, "verdicts must be a list of objects"),
        (_verdicts(DEBATE[:5]), "no case for FFF"),
        (_verdicts([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_verdicts([*DEBATE, "AAA"]), "AAA appears twice"),
        ([{"symbol": "AAA", "verdict": "sell", "reason": "x"}, *_verdicts(DEBATE[1:])], "the verdict for AAA must be one of keep, veto"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": " "}, *_verdicts(DEBATE[1:])], "AAA has no reason"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": "x" * 301}, *_verdicts(DEBATE[1:])], "the reason for AAA is 301 characters"),
        (_verdicts(DEBATE[:4], DEBATE[4:]), "2 vetoes: veto at most 1 stocks and keep the others"),
    ],
)
def test_bad_verdicts_are_refused(verdicts: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_vetoes=1)

    result = tools["submit_verdicts"](verdicts)

    assert message in result["error"] and not recorder.submitted


# --- the recorder -------------------------------------------------------------------------------


def test_the_first_valid_submission_is_final() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)
    tools["submit_bull_case"](_bull(DEBATE))

    assert tools["submit_bull_case"](_bull(DEBATE, conviction="low")) == {"error": "already recorded for this stage"}
    assert recorder.submission[0].conviction == "high"


def test_a_tool_called_at_the_wrong_stage_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)

    assert tools["submit_bear_case"](_bear(DEBATE)) == {"error": "submit_bear_case is not expected at this point of the review"}


def test_arming_a_stage_clears_the_previous_submission_and_error() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("BBB", "facts")
    recorder.expect_note("BBB")

    assert (recorder.submitted, recorder.submission, recorder.last_error) == (False, None, None)

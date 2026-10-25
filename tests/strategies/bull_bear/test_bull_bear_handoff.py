from __future__ import annotations

from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Verdict, submit_tools, trim_note
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

DEBATE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _tools() -> tuple[HandoffRecorder, dict[str, Any]]:
    recorder = HandoffRecorder(BullBearParams())
    return recorder, submit_tools(recorder)


def _bull(symbols: list[str], conviction: str = "high") -> list[dict[str, Any]]:
    return [{"symbol": s, "conviction": conviction, "argument": "strong trend"} for s in symbols]


def _bear(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "risk": "low", "concern": "none", "argument": "no red flag"} for s in symbols]


def _verdicts(chosen: list[str], out: list[str] | None = None) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": "in", "reason": "bull wins"} for s in chosen] + [{"symbol": s, "verdict": "out", "reason": "bear wins"} for s in out or []]


# --- note ---------------------------------------------------------------------------------------


def test_a_note_up_to_the_hard_limit_is_recorded_though_the_researcher_is_asked_for_less() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    params = BullBearParams()

    assert params.note_max_chars < params.note_hard_max_chars == 600
    assert tools["submit_note"]("AAA", "x" * 600) == {"status": "recorded"}


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


def test_an_empty_note_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert "empty" in tools["submit_note"]("AAA", "   ")["error"]


def test_a_long_note_is_refused_with_the_asked_length_and_how_to_shorten_it() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    error = tools["submit_note"]("AAA", "x" * 662)["error"]

    assert error == "the note for AAA is 662 characters; a note is at most 500: rewrite it to about 450 characters by dropping your least important facts, not by trimming words"
    assert "600" not in error  # the hard limit stays hidden from the model


def test_the_same_long_note_sent_again_is_called_out() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("AAA", "x" * 662)

    again = tools["submit_note"]("AAA", "  " + "x" * 662 + " ")["error"]  # same text once stripped
    shorter = tools["submit_note"]("AAA", "x" * 650)["error"]

    assert again.startswith("this is the same text you sent before: the note for AAA is 662 characters")
    assert not shorter.startswith("this is the same text")


def test_the_recorder_keeps_the_last_note_refused_for_length_until_the_next_stage() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert recorder.refused_note is None

    tools["submit_note"]("AAA", "x" * 700)
    tools["submit_note"]("AAA", " " + "y" * 650 + " ")
    tools["submit_note"]("BBB", "z" * 10)  # refused for its symbol: the length-refused note stays

    assert recorder.refused_note == "y" * 650
    recorder.expect_note("BBB")
    assert recorder.refused_note is None


def test_a_note_refused_for_another_reason_leaves_no_refused_note() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("BBB", "x" * 700)
    tools["submit_note"]("AAA", "  ")
    assert recorder.refused_note is None


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


def test_a_verdict_for_every_stock_is_recorded_with_the_stocks_put_in() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=4)

    result = tools["submit_verdicts"](_verdicts(["AAA", "BBB", "CCC", "DDD"], ["EEE", "FFF"]))

    assert result == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "in", "bull wins")
    assert recorder.submission.chosen == ["AAA", "BBB", "CCC", "DDD"]


def test_putting_every_stock_out_is_valid() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=4)

    assert tools["submit_verdicts"](_verdicts([], DEBATE)) == {"status": "recorded"}
    assert recorder.submission.chosen == []


def test_verdicts_and_symbols_are_normalised_like_the_debaters_cases() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6)

    raw = [{"symbol": " aaa ", "verdict": "Out ", "reason": "  lawsuit filed  "}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "out", "lawsuit filed")


@pytest.mark.parametrize(
    ("verdicts", "message"),
    [
        ({"AAA": "in"}, "verdicts must be a list of objects"),
        (_verdicts(DEBATE[:5]), "no case for FFF"),
        (_verdicts([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_verdicts([*DEBATE, "AAA"]), "AAA appears twice"),
        ([{"symbol": "AAA", "verdict": "keep", "reason": "x"}, *_verdicts([], DEBATE[1:])], "the verdict for AAA must be one of in, out"),
        ([{"symbol": "AAA", "verdict": "in", "reason": " "}, *_verdicts([], DEBATE[1:])], "AAA has no reason"),
        ([{"symbol": "AAA", "verdict": "in", "reason": "x" * 301}, *_verdicts([], DEBATE[1:])], "the reason for AAA is 301 characters"),
        (_verdicts(DEBATE[:3], DEBATE[3:]), "3 stocks in: put at most 2 in and the others out"),
    ],
)
def test_bad_verdicts_are_refused(verdicts: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=2)

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


def test_a_changed_verdict_needs_what_changed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})

    result = tools["submit_verdicts"](_verdicts(DEBATE[1:], ["AAA"]))

    assert "your previous verdict on AAA was in: to put it out, give what_changed" in result["error"]


def test_a_changed_verdict_with_what_changed_is_recorded_with_it() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in", "BBB": "in"})
    raw = [{"symbol": "AAA", "verdict": "out", "reason": "bear wins", "what_changed": "8-K: guidance cut"}, *_verdicts(DEBATE[1:])]

    assert tools["submit_verdicts"](raw) == {"status": "recorded"}
    assert recorder.submission.verdicts[0] == Verdict("AAA", "out", "bear wins", "8-K: guidance cut")
    assert recorder.submission.verdicts[1].what_changed is None


def test_an_unchanged_verdict_or_a_first_verdict_needs_no_what_changed() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})

    assert tools["submit_verdicts"](_verdicts(DEBATE)) == {"status": "recorded"}


def test_a_long_what_changed_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_verdicts(DEBATE, max_in=6, previous={"AAA": "in"})
    raw = [{"symbol": "AAA", "verdict": "out", "reason": "x", "what_changed": "y" * 301}, *_verdicts(DEBATE[1:])]

    assert "the what_changed for AAA is 301 characters" in tools["submit_verdicts"](raw)["error"]


# --- trim_note ----------------------------------------------------------------------------------


def test_a_note_within_the_limit_is_only_stripped() -> None:
    assert trim_note("  A fact. Another.  ", max_chars=600, min_chars=250) == "A fact. Another."


def test_a_long_note_keeps_the_most_whole_sentences_that_fit() -> None:
    text = "a" * 300 + ". " + "b" * 250 + "! " + "c" * 100 + "."
    assert trim_note(text, max_chars=600, min_chars=250) == "a" * 300 + ". " + "b" * 250 + "!"


def test_a_decimal_point_is_not_a_sentence_end() -> None:
    text = "Revenue " + "r" * 280 + ". Q3 EPS $1.58 vs $1.47 est " + "e" * 300 + "."
    assert trim_note(text, max_chars=600, min_chars=250) == "Revenue " + "r" * 280 + "."


def test_with_no_long_enough_sentence_the_note_is_cut_at_a_word_with_an_ellipsis() -> None:
    text = "Short. " + "word " * 200
    trimmed = trim_note(text, max_chars=600, min_chars=250)
    assert len(trimmed) <= 600 and trimmed.endswith("word…")


def test_a_note_with_no_space_is_cut_hard_with_an_ellipsis() -> None:
    assert trim_note("x" * 700, max_chars=600, min_chars=250) == "x" * 599 + "…"


@pytest.mark.parametrize("length", [601, 650, 1200])
def test_a_trimmed_note_is_always_accepted_by_the_recorder(length: int) -> None:
    text = ("Fact 2025-10-27: revenue $1.58B. " * 60)[:length]
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert tools["submit_note"]("AAA", trim_note(text, max_chars=600, min_chars=250)) == {"status": "recorded"}

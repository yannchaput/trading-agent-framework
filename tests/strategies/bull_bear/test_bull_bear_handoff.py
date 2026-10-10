from __future__ import annotations

from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, Choice, HandoffRecorder, Note, Picks, submit_tools
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

DEBATE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _tools() -> tuple[HandoffRecorder, dict[str, Any]]:
    recorder = HandoffRecorder(BullBearParams())
    return recorder, submit_tools(recorder)


def _bull(symbols: list[str], conviction: str = "high") -> list[dict[str, Any]]:
    return [{"symbol": s, "conviction": conviction, "argument": "strong trend"} for s in symbols]


def _bear(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "risk": "low", "concern": "none", "argument": "no red flag"} for s in symbols]


def _choices(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "reason": "won the debate"} for s in symbols]


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


# --- picks --------------------------------------------------------------------------------------


def test_picks_within_the_bounds_with_every_unpicked_holding_dropped_are_recorded() -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=["EEE", "FFF"])

    result = tools["submit_picks"](_choices(["AAA", "BBB", "CCC", "DDD", "EEE"]), [{"symbol": "FFF", "reason": "lost the debate"}])

    assert result == {"status": "recorded"}
    assert recorder.submission == Picks(picks=tuple(Choice(s, "won the debate") for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]), drops=(Choice("FFF", "lost the debate"),))


def test_drops_may_be_omitted_when_nothing_held_is_left_out() -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=[])

    assert tools["submit_picks"](_choices(DEBATE[:5])) == {"status": "recorded"}


@pytest.mark.parametrize(
    ("picks", "drops", "message"),
    [
        (_choices(DEBATE[:4]), [], "pick between 5 and 10 stocks, got 4"),
        (_choices([*DEBATE[:4], "ZZZ"]), [], "ZZZ is not in the debate"),
        (_choices([*DEBATE[:4], "AAA"]), [], "AAA appears twice"),
        (_choices(DEBATE[:5]), [], "FFF is held and not picked"),
        (_choices(DEBATE[:5]), _choices(["AAA", "FFF"]), "AAA cannot be dropped"),
    ],
)
def test_bad_picks_are_refused(picks: list[dict[str, Any]], drops: list[dict[str, Any]], message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=["FFF"])

    assert message in tools["submit_picks"](picks, drops)["error"]


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

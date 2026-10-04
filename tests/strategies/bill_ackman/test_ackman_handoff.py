from __future__ import annotations

import inspect
import typing
from typing import Any

import pytest
from langchain_core.tools import StructuredTool

from trading_agent_framework.strategies.bill_ackman.handoff import (
    HandoffError,
    HandoffRecorder,
    Idea,
    PortfolioPosition,
    Verdict,
    submit_tools,
    validate_portfolio,
    validate_ranking,
    validate_verdicts,
)
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams

CANDIDATES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]


def _idea(symbol: str, reason: str = "cash rich") -> dict[str, Any]:
    return {"symbol": symbol, "reason": reason}


def _ranking(raw: Any, *, candidates: list[str] = CANDIDATES, top_n: int = 5, reason_max_chars: int = 300) -> list[Idea]:
    return validate_ranking(raw, candidates=candidates, top_n=top_n, reason_max_chars=reason_max_chars)


def _portfolio(raw: Any, *, allowed: list[str] = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")) -> list[PortfolioPosition]:  # type: ignore[assignment]
    return validate_portfolio(raw, allowed=allowed, max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def _position(symbol: str, weight: Any, reason: str = "best idea") -> dict[str, Any]:
    return {"symbol": symbol, "weight": weight, "reason": reason}


# --- ranking ------------------------------------------------------------------------------------


def test_a_valid_ranking_keeps_the_order_and_normalises_symbols() -> None:
    ideas = _ranking([_idea(" bbb ", "  growing "), _idea("AAA")])

    assert ideas == [Idea("BBB", "growing"), Idea("AAA", "cash rich")]


def test_a_ranking_may_hold_the_top_n_ideas_and_no_more() -> None:
    assert len(_ranking([_idea(s) for s in CANDIDATES[:5]])) == 5
    with pytest.raises(HandoffError, match="at most 5"):
        _ranking([_idea(s) for s in CANDIDATES[:6]])


def test_the_ranking_limit_is_the_number_of_candidates_when_there_are_fewer() -> None:
    assert len(_ranking([_idea("AAA"), _idea("BBB")], candidates=["AAA", "BBB"])) == 2
    with pytest.raises(HandoffError, match="at most 2"):
        _ranking([_idea("AAA"), _idea("BBB"), _idea("CCC")], candidates=["AAA", "BBB"])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([], "empty"),
        ("AAA", "list of objects"),
        (None, "list of objects"),
        (["AAA"], "must be an object"),
        ([{"reason": "x"}], "no symbol"),
        ([{"symbol": "  ", "reason": "x"}], "no symbol"),
        ([{"symbol": 5, "reason": "x"}], "no symbol"),
        ([{"symbol": "ZZZ", "reason": "x"}], "not one of the candidates"),
        ([_idea("AAA"), _idea("aaa")], "appears twice"),
        ([{"symbol": "AAA"}], "no reason"),
        ([{"symbol": "AAA", "reason": "   "}], "no reason"),
        ([{"symbol": "AAA", "reason": "x" * 301}], "under 300"),
    ],
)
def test_an_invalid_ranking_is_refused_with_a_message_that_says_why(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        _ranking(raw)


def test_a_reason_of_exactly_the_maximum_length_is_accepted() -> None:
    assert _ranking([_idea("AAA", "x" * 300)])[0].reason == "x" * 300


def test_the_not_a_candidate_message_lists_the_candidates() -> None:
    with pytest.raises(HandoffError, match="AAA, BBB"):
        _ranking([_idea("ZZZ")], candidates=["AAA", "BBB"])


# --- verdicts -----------------------------------------------------------------------------------


def _verdict(symbol: str, verdict: Any = "survive", reason: str = "debt is fine", **extra: Any) -> dict[str, Any]:
    return {"symbol": symbol, "verdict": verdict, "reason": reason, **extra}


def test_valid_verdicts_cover_exactly_the_asked_symbols() -> None:
    verdicts = validate_verdicts([_verdict("aaa", " FAIL ", concern=" Debt "), _verdict("BBB")], expected=["AAA", "BBB"], reason_max_chars=300)

    assert verdicts == [Verdict("AAA", "fail", "debt is fine", concern="debt"), Verdict("BBB", "survive", "debt is fine")]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([_verdict("AAA")], "no verdict for BBB"),
        ([_verdict("AAA"), _verdict("BBB"), _verdict("CCC")], "CCC was not asked about"),
        ([_verdict("AAA"), _verdict("AAA"), _verdict("BBB")], "appears twice"),
        ([_verdict("AAA", "maybe"), _verdict("BBB")], "'survive' or 'fail'"),
        ([_verdict("AAA", None), _verdict("BBB")], "'survive' or 'fail'"),
        ([_verdict("AAA", reason=""), _verdict("BBB")], "no reason"),
        ([], "no verdict for AAA, BBB"),
        ("nope", "list of objects"),
    ],
)
def test_invalid_verdicts_are_refused(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        validate_verdicts(raw, expected=["AAA", "BBB"], reason_max_chars=300)


@pytest.mark.parametrize("concern", [None, "", "lawsuit", 3])
def test_a_fail_needs_a_concern_from_the_list(concern: Any) -> None:
    with pytest.raises(HandoffError, match="needs a concern: one of debt, margin, competition, management, accounting, valuation"):
        validate_verdicts([_verdict("AAA", "fail", concern=concern)], expected=["AAA"], reason_max_chars=300)


def test_a_survive_drops_any_concern() -> None:
    (verdict,) = validate_verdicts([_verdict("AAA", concern="debt")], expected=["AAA"], reason_max_chars=300)

    assert verdict.concern is None


def test_a_flip_needs_what_changed() -> None:
    with pytest.raises(HandoffError, match="AAA was 'survive' at the previous review: say in what_changed what changed since then"):
        validate_verdicts([_verdict("AAA", "fail", concern="debt")], expected=["AAA"], reason_max_chars=300, previous={"AAA": "survive"})


def test_a_flip_with_what_changed_is_accepted_and_kept() -> None:
    (verdict,) = validate_verdicts(
        [_verdict("AAA", "fail", concern="margin", what_changed=" margin fell 8 points in the new 10-K ")], expected=["AAA"], reason_max_chars=300, previous={"AAA": "survive"}
    )

    assert verdict.what_changed == "margin fell 8 points in the new 10-K"


def test_an_unchanged_verdict_or_a_new_symbol_needs_no_what_changed() -> None:
    verdicts = validate_verdicts([_verdict("AAA", "fail", concern="debt"), _verdict("BBB")], expected=["AAA", "BBB"], reason_max_chars=300, previous={"AAA": "fail"})

    assert [verdict.what_changed for verdict in verdicts] == [None, None]


def test_what_changed_is_length_capped_like_a_reason() -> None:
    with pytest.raises(HandoffError, match="what_changed for AAA is 11 characters: keep it under 10"):
        validate_verdicts([_verdict("AAA", reason="short", what_changed="x" * 11)], expected=["AAA"], reason_max_chars=10, previous={"AAA": "fail"})


# --- portfolio ----------------------------------------------------------------------------------


def test_a_valid_portfolio_is_returned_with_float_weights() -> None:
    positions = _portfolio([_position("aaa", 0.35), _position("BBB", "0.25"), _position("CCC", 0.3)])

    assert positions == [PortfolioPosition("AAA", 0.35, "best idea"), PortfolioPosition("BBB", 0.25, "best idea"), PortfolioPosition("CCC", 0.3, "best idea")]


def test_an_empty_portfolio_is_valid() -> None:
    assert _portfolio([]) == []
    assert _portfolio([], allowed=[]) == []


def test_the_weight_and_total_boundaries_are_inclusive() -> None:
    assert len(_portfolio([_position("AAA", 0.35), _position("BBB", 0.35), _position("CCC", 0.28)])) == 3  # total exactly 0.98
    assert _portfolio([_position("AAA", 0.05)])[0].weight == 0.05
    assert _portfolio([_position("AAA", 0.35)])[0].weight == 0.35


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([_position("AAA", 0.04)], "between 0.05 and 0.35"),
        ([_position("AAA", 0.36)], "between 0.05 and 0.35"),
        ([_position("AAA", -0.1)], "between 0.05 and 0.35"),
        ([_position("AAA", 0.35), _position("BBB", 0.35), _position("CCC", 0.29)], "at most 0.98"),
        ([_position(s, 0.1) for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")], "at most 5 positions"),
        ([_position("ZZZ", 0.2)], "not allowed"),
        ([_position("AAA", 0.2), _position("AAA", 0.2)], "appears twice"),
        ([_position("AAA", None)], "must be a number"),
        ([_position("AAA", "lots")], "must be a number"),
        ([_position("AAA", True)], "must be a number"),
        ([_position("AAA", float("nan"))], "finite"),
        ([_position("AAA", float("inf"))], "finite"),
        ([_position("AAA", 10**400)], "must be a number"),
        ([{"symbol": "AAA", "weight": 0.2}], "no reason"),
        ("AAA", "list of objects"),
    ],
)
def test_an_invalid_portfolio_is_refused(raw: Any, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        _portfolio(raw)


def test_with_nothing_allowed_the_refusal_says_to_submit_an_empty_list() -> None:
    with pytest.raises(HandoffError, match="empty list"):
        _portfolio([_position("AAA", 0.2)], allowed=[])


def test_a_required_holding_must_stay_in_the_portfolio() -> None:
    with pytest.raises(HandoffError, match="CCC is held and has not failed twice: keep it with a weight of at least 0.05"):
        validate_portfolio([_position("AAA", 0.3)], allowed=["AAA", "CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def test_an_empty_portfolio_is_refused_while_a_holding_is_required() -> None:
    with pytest.raises(HandoffError, match="CCC is held and has not failed twice"):
        validate_portfolio([], allowed=["CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def test_a_required_holding_may_be_shrunk_to_the_minimum_weight() -> None:
    positions = validate_portfolio(
        [_position("AAA", 0.3), _position("CCC", 0.05)], allowed=["AAA", "CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300
    )

    assert [position.symbol for position in positions] == ["AAA", "CCC"]


# --- the recorder and the tools -------------------------------------------------------------------


def _recorder() -> HandoffRecorder:
    return HandoffRecorder(AckmanParams())


def test_a_valid_submission_is_recorded_and_exposed() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)

    result = submit_tools(recorder)["submit_ranking"]([_idea("AAA")])

    assert result == {"status": "recorded"}
    assert recorder.submitted
    assert recorder.submission == [Idea("AAA", "cash rich")]
    assert recorder.last_error is None


def test_an_invalid_submission_returns_the_error_and_records_nothing() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)

    result = submit_tools(recorder)["submit_ranking"]([_idea("ZZZ")])

    assert "not one of the candidates" in result["error"]
    assert not recorder.submitted
    assert recorder.last_error == result["error"]


def test_the_model_can_correct_an_invalid_submission_in_the_same_run() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    ranking = submit_tools(recorder)["submit_ranking"]

    assert "error" in ranking([_idea("ZZZ")])
    assert ranking([_idea("AAA")]) == {"status": "recorded"}
    assert recorder.last_error is None


def test_the_first_valid_submission_is_final() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    ranking = submit_tools(recorder)["submit_ranking"]
    ranking([_idea("AAA")])

    second = ranking([_idea("BBB")])

    assert second == {"error": "already recorded for this review"}
    assert recorder.submission == [Idea("AAA", "cash rich")]


def test_an_empty_portfolio_counts_as_a_submission() -> None:
    recorder = _recorder()
    recorder.expect_portfolio([])

    assert submit_tools(recorder)["submit_portfolio"]([]) == {"status": "recorded"}
    assert recorder.submitted
    assert recorder.submission == []


def test_a_tool_called_out_of_turn_is_refused() -> None:
    recorder = _recorder()
    tools = submit_tools(recorder)

    assert "not expected" in tools["submit_ranking"]([_idea("AAA")])["error"]  # nothing armed yet
    recorder.expect_verdicts(["AAA"])
    assert "not expected" in tools["submit_portfolio"]([])["error"]  # the verdicts stage is armed, not the portfolio
    assert not recorder.submitted


def test_arming_a_stage_clears_the_previous_submission_and_error() -> None:
    recorder = _recorder()
    recorder.expect_ranking(CANDIDATES)
    tools = submit_tools(recorder)
    tools["submit_ranking"]([_idea("ZZZ")])
    tools["submit_ranking"]([_idea("AAA")])

    recorder.expect_verdicts(["AAA"])

    assert not recorder.submitted
    assert recorder.submission is None
    assert recorder.last_error is None


def test_the_recorder_checks_flips_and_required_holdings_against_what_it_was_armed_with() -> None:
    recorder = _recorder()
    tools = submit_tools(recorder)

    recorder.expect_verdicts(["AAA"], previous={"AAA": "survive"})
    assert "what_changed" in tools["submit_verdicts"]([_verdict("AAA", "fail", concern="debt")])["error"]
    assert tools["submit_verdicts"]([_verdict("AAA", "fail", concern="debt", what_changed="debt doubled")]) == {"status": "recorded"}

    recorder.expect_portfolio(["AAA", "CCC"], required=["CCC"])
    assert "CCC is held and has not failed twice" in tools["submit_portfolio"]([_position("AAA", 0.3)])["error"]
    assert tools["submit_portfolio"]([_position("AAA", 0.3), _position("CCC", 0.1)]) == {"status": "recorded"}


def test_each_stage_validates_with_the_recorders_parameters() -> None:
    recorder = HandoffRecorder(AckmanParams(max_weight=0.2, min_weight=0.05))
    recorder.expect_portfolio(["AAA"])

    result = submit_tools(recorder)["submit_portfolio"]([_position("AAA", 0.25)])

    assert "between 0.05 and 0.2" in result["error"]


def test_an_overflowing_integer_is_refused_with_a_correctable_error() -> None:
    recorder = _recorder()
    recorder.expect_portfolio(["AAA"])
    tools = submit_tools(recorder)

    result = tools["submit_portfolio"]([{"symbol": "AAA", "weight": 10**400, "reason": "x"}])

    assert "error" in result
    assert "must be a number" in result["error"]
    assert not recorder.submitted
    assert recorder.last_error is not None

    # The model can correct it in the same run
    corrected = tools["submit_portfolio"]([_position("AAA", 0.3)])

    assert corrected == {"status": "recorded"}
    assert recorder.submitted


def test_the_tools_are_named_after_their_keys_and_have_one_line_docstrings() -> None:
    tools = submit_tools(_recorder())

    assert set(tools) == {"submit_ranking", "submit_verdicts", "submit_portfolio"}
    for name, tool in tools.items():
        assert tool.__name__ == name
        assert tool.__doc__ is not None and len(tool.__doc__.strip().splitlines()) == 1


def test_the_tools_have_real_annotations_the_agent_layer_can_read() -> None:
    # Regression guard for the no-`from __future__ import annotations` rule of this module.
    for tool in submit_tools(_recorder()).values():
        for parameter in inspect.signature(tool).parameters.values():
            assert not isinstance(parameter.annotation, str)
        assert typing.get_type_hints(tool)["return"] == dict[str, Any]


@pytest.mark.parametrize(("name", "argument"), [("submit_ranking", "ideas"), ("submit_verdicts", "verdicts"), ("submit_portfolio", "positions")])
def test_langchain_builds_an_array_schema_from_each_tool(name: str, argument: str) -> None:
    tool = StructuredTool.from_function(submit_tools(_recorder())[name])

    assert list(tool.args) == [argument]
    assert tool.args[argument]["type"] == "array"
    assert tool.name == name

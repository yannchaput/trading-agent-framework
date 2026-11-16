"""Tests for the sector_pods submit schemas, validators and submit tools."""

import logging

import pytest

from trading_agent_framework.agents.submit import HandoffError, NodeRecorder
from trading_agent_framework.agents.tools.text import LengthGate
from trading_agent_framework.strategies.sector_pods.handoff import (
    PM_SUBMIT,
    POD_SUBMIT,
    RISK_SUBMIT,
    UNAVAILABLE,
    check_weights,
    pm_submit_tool,
    pod_submit_tool,
    risk_submit_tool,
    unresearched,
    validate_pm,
    validate_pod,
    validate_risk,
)
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import PODS

PARAMS = SectorPodsParams()
TECH = PODS[0]
ROW = {"return_1w_pct": 1.0}


def _soft(validator):
    """The validator as a function returning its refusal message, or None when it accepts."""

    def check(payload, state):
        try:
            validator(payload, state)
        except HandoffError as exc:
            return str(exc)
        return None

    return check


def _msg(value):
    assert value is not None
    return value


def _state(**over):
    state = {
        "facts": {"tech_pod": {"XLK": ROW, "SMH": ROW}},
        "holdings": {"XLF": 0.10},
        "pods": {"tech_pod": {"scores": [], "sector_view": "x"}, "financials_pod": UNAVAILABLE},
    }
    return {**state, **over}


def _full_state(holdings=None):
    """Every pod available with facts for all its ETFs."""
    return {
        "facts": {pod.key: {s: ROW for s in pod.sleeve} for pod in PODS},
        "holdings": holdings or {},
        "pods": {pod.key: {"scores": [], "sector_view": "x"} for pod in PODS},
    }


def _scores(*pairs, reason="r"):
    return {"scores": [{"symbol": s, "score": v, "reason": reason} for s, v in pairs], "sector_view": "view"}


def test_pod_view_must_cover_exactly_its_facts():
    check = _soft(validate_pod(TECH, PARAMS))
    assert check(_scores(("XLK", 1), ("SMH", 0)), _state()) is None
    assert "SMH" in _msg(check(_scores(("XLK", 1)), _state()))
    assert "XLE" in _msg(check(_scores(("XLK", 1), ("SMH", 0), ("XLE", 0)), _state()))
    assert "XLK" in _msg(check(_scores(("XLK", 1), ("XLK", 0), ("SMH", 0)), _state()))


@pytest.mark.parametrize("bad", [3, -3, True, 1.5])
def test_pod_score_refused(bad):
    assert _soft(validate_pod(TECH, PARAMS))(_scores(("XLK", bad), ("SMH", 0)), _state()) is not None


@pytest.mark.parametrize("good", [-2, 2])
def test_pod_score_accepted(good):
    assert _soft(validate_pod(TECH, PARAMS))(_scores(("XLK", good), ("SMH", 0)), _state()) is None


def test_reason_length_limits():
    check = _soft(validate_pod(TECH, PARAMS))
    assert check(_scores(("XLK", 1), ("SMH", 0), reason="a" * (PARAMS.reason_hard_max_chars + 1)), _state()) is not None
    assert check(_scores(("XLK", 1), ("SMH", 0), reason="a" * PARAMS.reason_hard_max_chars), _state()) is None
    assert check(_scores(("XLK", 1), ("SMH", 0), reason=" "), _state()) is not None
    assert check({**_scores(("XLK", 1), ("SMH", 0)), "sector_view": "a" * (PARAMS.summary_hard_max_chars + 1)}, _state()) is not None
    assert check({**_scores(("XLK", 1), ("SMH", 0)), "sector_view": ""}, _state()) is not None


def test_pod_payload_shape():
    check = _soft(validate_pod(TECH, PARAMS))
    assert check("nope", _state()) is not None
    assert check({"scores": "x", "sector_view": "v"}, _state()) is not None
    assert check({"scores": ["x"], "sector_view": "v"}, _state()) is not None


def _obj(symbols=("XLK",), kind="macro", severity="low", reason="r"):
    return {"objections": [{"symbols": list(symbols), "kind": kind, "severity": severity, "reason": reason}], "summary": "s"}


def test_risk_review_kinds_and_symbols():
    check = _soft(validate_risk(PARAMS))
    assert check(_obj(), _state()) is None
    assert check({"objections": [], "summary": "s"}, _state()) is None
    assert check(_obj(kind="liquidity"), _state()) is not None
    assert check(_obj(symbols=("SPY",)), _state()) is not None
    assert check(_obj(severity="medium"), _state()) is not None
    assert check(_obj(reason="a" * (PARAMS.reason_hard_max_chars + 1)), _state()) is not None
    assert check({"objections": [], "summary": ""}, _state()) is not None


def test_weights_caps():
    state = _full_state()
    assert check_weights({"XLK": 0.26}, state, PARAMS) is not None
    assert "tech_pod" in _msg(check_weights({"XLK": 0.25, "SMH": 0.20}, state, PARAMS))
    assert check_weights({"XLK": -0.01}, state, PARAMS) is not None
    assert check_weights({"SPY": 0.1}, state, PARAMS) is not None
    even = {"XLK": 0.2, "SMH": 0.2, "XLF": 0.2, "KRE": 0.2, "XLV": 0.19}
    assert check_weights(even, state, PARAMS) is not None
    assert check_weights({**even, "XLV": 0.18}, state, PARAMS) is None


def test_keep_or_trim_for_unavailable_pod():
    state = _state()
    assert check_weights({"XLF": 0.10}, state, PARAMS) is None
    assert check_weights({"XLF": 0.0}, state, PARAMS) is None
    assert "no view this week" in _msg(check_weights({"XLF": 0.11}, state, PARAMS))
    assert check_weights({"KRE": 0.05}, state, PARAMS) is not None


def test_keep_or_trim_for_symbol_without_facts():
    state = _state(holdings={"XLC": 0.08})
    assert check_weights({"XLC": 0.08}, state, PARAMS) is None
    assert check_weights({"XLC": 0.09}, state, PARAMS) is not None


def test_unresearched_matches_check_weights():
    state = _state(holdings={"XLF": 0.10, "XLC": 0.08, "XLK": 0.05, "SHV": 0.3, "ZZZ": 0.1})
    assert unresearched(state) == {"XLF": 0.10, "XLC": 0.08}


def test_pm_payload():
    check = _soft(validate_pm(PARAMS))
    state = _full_state()
    good = {"weights": [{"symbol": "XLK", "weight": 0.2, "reason": "r"}], "overrides": "", "summary": "s"}
    assert check(good, state) is None
    dup = {**good, "weights": good["weights"] * 2}
    assert "XLK" in _msg(check(dup, state))
    assert check({**good, "weights": [{"symbol": "XLK", "weight": 0.3, "reason": "r"}]}, state) is not None
    assert check({**good, "weights": [{"symbol": "XLK", "weight": "a", "reason": "r"}]}, state) is not None
    assert check({**good, "weights": [{"symbol": "XLK", "weight": True, "reason": "r"}]}, state) is not None
    assert check({**good, "summary": ""}, state) is not None
    assert check({**good, "overrides": "a" * (PARAMS.summary_hard_max_chars + 1)}, state) is not None


def _book(total):
    """Two pods at exactly 0.40 and a third taking the rest of `total`."""
    first, second, third = PODS[:3]
    return {first.sleeve[0]: 0.25, first.sleeve[1]: 0.15, second.sleeve[0]: 0.25, second.sleeve[1]: 0.15, third.sleeve[0]: round(total - 0.80, 2)}


@pytest.mark.parametrize(
    ("book", "message"),
    [
        ({"XLK": float("nan")}, "must be between"),
        ({"XLK": float("inf")}, "must be between"),
        ({"XLK": -0.01}, "must be between"),
        ({"XLK": 0.25}, None),
        ({"XLK": 0.25, "SMH": 0.15}, None),
        ({"XLK": 0.25, "SMH": 0.16}, "pod limit"),
        (_book(0.98), None),
        (_book(0.99), "the limit is"),
    ],
)
def test_check_weights_boundaries(book, message):
    result = check_weights(book, _full_state(), PARAMS)
    if message is None:
        assert result is None
    else:
        assert message in _msg(result)
    assert (_soft(validate_pm(PARAMS))({"weights": [{"symbol": s, "weight": w, "reason": "r"} for s, w in book.items()], "overrides": "", "summary": "s"}, _full_state()) is None) == (message is None)


def test_submit_tools_record_into_their_recorder():
    recorder = NodeRecorder(POD_SUBMIT, validate_pod(TECH, PARAMS))
    recorder.arm(_state())
    tool = pod_submit_tool(recorder, PARAMS)
    assert tool.__name__ == "submit_pod_view"
    result = tool(scores=_scores(("XLK", 1), ("SMH", 0))["scores"], sector_view="x")
    assert result == {"status": "recorded"}
    assert recorder.submission is not None and recorder.submission["scores"][0]["symbol"] == "XLK"
    assert "error" in tool(scores=[], sector_view="x")


def test_risk_and_pm_tools():
    risk = NodeRecorder(RISK_SUBMIT, validate_risk(PARAMS))
    risk.arm(_state())
    tool = risk_submit_tool(risk, PARAMS)
    assert tool.__name__ == "submit_risk_review"
    assert tool(objections=[], summary="s") == {"status": "recorded"}
    pm = NodeRecorder(PM_SUBMIT, validate_pm(PARAMS))
    pm.arm(_full_state())
    tool = pm_submit_tool(pm, PARAMS)
    assert tool.__name__ == "submit_target_weights"
    assert tool(weights=[], overrides="", summary="s") == {"status": "recorded"}


LONG = "First sentence here. " * 30  # 629 characters of whole sentences: over the reason hard cap, under the summary one
LONGER = "First sentence here. " * 40  # 839 characters: over both hard caps
TOO_LONG_SUFFIX = "Count them with count_characters, shorten those texts and call {tool} again with the whole submission."


def _gated(validator, tool, attempts=PARAMS.max_submit_attempts):
    return NodeRecorder(tool, validator, gate=LengthGate(attempts, name=tool, log_tag="sector_pods"))


def test_an_over_long_pod_text_is_refused_with_its_counts_then_trimmed_on_the_third_attempt(caplog: pytest.LogCaptureFixture):
    recorder = _gated(validate_pod(TECH, PARAMS), POD_SUBMIT)
    recorder.arm(_state())
    tool = pod_submit_tool(recorder, PARAMS)
    scores = [{"symbol": "XLK", "score": 1, "reason": LONG}, {"symbol": "SMH", "score": 0, "reason": "short"}]

    for _ in range(2):
        assert tool(scores=scores, sector_view=LONGER) == {
            "error": f"too long: reason for XLK has {len(LONG.strip())} characters (limit 200); sector_view has {len(LONGER.strip())} characters (limit 400). "
            + TOO_LONG_SUFFIX.format(tool=POD_SUBMIT)
        }
        assert recorder.submission is None
    with caplog.at_level(logging.WARNING):
        assert tool(scores=scores, sector_view=LONGER) == {"status": "recorded"}

    assert recorder.submission is not None
    view, reason = recorder.submission["sector_view"], recorder.submission["scores"][0]["reason"]
    assert 200 // 2 <= len(reason) <= PARAMS.reason_hard_max_chars and reason.endswith("here.")
    assert 400 // 2 <= len(view) <= PARAMS.summary_hard_max_chars and view.endswith("here.")
    assert recorder.submission["scores"][1]["reason"] == "short"
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2  # the short reason is not logged
    assert any(f"submit_pod_view: reason for XLK trimmed from {len(LONG.strip())} to {len(reason)} characters (limit 300)" in m for m in warnings)
    assert any(f"submit_pod_view: sector_view trimmed from {len(LONGER.strip())} to {len(view)} characters (limit 700)" in m for m in warnings)


def test_a_text_between_the_stated_and_hard_limits_is_recorded_whole_on_the_last_attempt():
    recorder = _gated(validate_pod(TECH, PARAMS), POD_SUBMIT, attempts=1)
    recorder.arm(_state())
    view, reason = "First sentence here. " * 22, "x" * 250  # 461 and 250 characters

    assert pod_submit_tool(recorder, PARAMS)(scores=[{"symbol": "XLK", "score": 1, "reason": reason}, {"symbol": "SMH", "score": 0, "reason": "r"}], sector_view=view) == {"status": "recorded"}

    assert recorder.submission is not None and recorder.submission["sector_view"] == view.strip() and recorder.submission["scores"][0]["reason"] == reason


def test_two_objections_on_the_same_symbols_are_reviewed_apart(caplog: pytest.LogCaptureFixture):
    recorder = _gated(validate_risk(PARAMS), RISK_SUBMIT, attempts=2)
    recorder.arm(_state())
    tool = risk_submit_tool(recorder, PARAMS)
    objections = [
        {"symbols": ["XLK", "SMH"], "kind": "crowding", "severity": "high", "reason": "short"},
        {"symbols": ["XLK", "SMH"], "kind": "macro", "severity": "low", "reason": LONG},
    ]

    refused = tool(objections=objections, summary="s")
    assert refused == {"error": f"too long: reason for XLK, SMH has {len(LONG.strip())} characters (limit 200). " + TOO_LONG_SUFFIX.format(tool=RISK_SUBMIT)}
    with caplog.at_level(logging.WARNING):
        assert tool(objections=objections, summary="s") == {"status": "recorded"}

    assert recorder.submission is not None
    first, second = (objection["reason"] for objection in recorder.submission["objections"])
    assert first == "short" and len(second) <= PARAMS.reason_hard_max_chars and second.endswith("here.")
    assert any("submit_risk_review: reason for XLK, SMH trimmed from" in record.getMessage() for record in caplog.records)


def test_a_weight_refusal_uses_no_length_attempt():
    recorder = _gated(validate_pm(PARAMS), PM_SUBMIT)
    recorder.arm(_full_state())
    tool = pm_submit_tool(recorder, PARAMS)
    long_book = [{"symbol": "XLK", "weight": 0.1, "reason": LONG}]
    over_cap = [{"symbol": "XLK", "weight": 0.9, "reason": "short"}]

    assert "too long" in tool(weights=long_book, overrides="", summary="s")["error"]  # length refusal 1
    weight_refusal = tool(weights=over_cap, overrides="", summary="s")  # short texts, refused by the validator only
    assert "error" in weight_refusal and "too long" not in weight_refusal["error"]
    assert "too long" in tool(weights=long_book, overrides="", summary="s")["error"]  # length refusal 2: the weight refusal used none
    assert tool(weights=long_book, overrides="", summary="s") == {"status": "recorded"}  # third length attempt: trimmed
    assert recorder.submission is not None and len(recorder.submission["weights"][0]["reason"]) <= PARAMS.reason_hard_max_chars


def test_a_missing_reason_still_gets_the_validators_message():
    recorder = _gated(validate_pod(TECH, PARAMS), POD_SUBMIT)
    recorder.arm(_state())

    result = pod_submit_tool(recorder, PARAMS)(scores=[{"symbol": "XLK", "score": 1, "reason": None}, {"symbol": "SMH", "score": 0, "reason": "r"}], sector_view="x")

    assert result == {"error": "XLK has no reason: give one short sentence"}


def test_without_a_gate_texts_reach_the_validator_unreviewed():
    recorder = NodeRecorder(POD_SUBMIT, validate_pod(TECH, PARAMS))
    recorder.arm(_state())

    result = pod_submit_tool(recorder, PARAMS)(scores=_scores(("XLK", 1), ("SMH", 0))["scores"], sector_view=LONGER)

    assert result == {"error": f"the sector_view for your submission is {len(LONGER.strip())} characters: keep it under {PARAMS.summary_hard_max_chars}"}


def test_a_submit_tool_still_refuses_what_is_not_text():
    recorder = NodeRecorder(RISK_SUBMIT, validate_risk(PARAMS))
    recorder.arm(_state())

    assert "error" in risk_submit_tool(recorder, PARAMS)(objections=[], summary=None)
    assert "error" in risk_submit_tool(recorder, PARAMS)(objections=["not an object"], summary="s")


def test_a_second_valid_submission_is_refused_the_first_one_wins():
    recorder = NodeRecorder(POD_SUBMIT, validate_pod(TECH, PARAMS))
    recorder.arm(_state())
    tool = pod_submit_tool(recorder, PARAMS)

    assert tool(scores=_scores(("XLK", 1), ("SMH", 0))["scores"], sector_view="first") == {"status": "recorded"}
    assert tool(scores=_scores(("XLK", 2), ("SMH", 2))["scores"], sector_view="second") == {"error": "already recorded for this run"}
    assert recorder.submission["sector_view"] == "first"


def test_a_gate_refusal_in_a_sector_pods_tool_is_the_recorders_last_error():
    recorder = _gated(validate_pod(TECH, PARAMS), POD_SUBMIT)
    recorder.arm(_state())

    pod_submit_tool(recorder, PARAMS)(scores=_scores(("XLK", 1), ("SMH", 0))["scores"], sector_view=LONGER)

    assert recorder.last_error is not None and recorder.last_error.startswith("too long: sector_view has")

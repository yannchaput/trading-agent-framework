"""Tests for the sector_pods submit schemas, validators and submit tools."""

import pytest

from trading_agent_framework.agents.graph import SubmitSlot
from trading_agent_framework.strategies.sector_pods.handoff import (
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
    check = validate_pod(TECH, PARAMS)
    assert check(_scores(("XLK", 1), ("SMH", 0)), _state()) is None
    assert "SMH" in _msg(check(_scores(("XLK", 1)), _state()))
    assert "XLE" in _msg(check(_scores(("XLK", 1), ("SMH", 0), ("XLE", 0)), _state()))
    assert "XLK" in _msg(check(_scores(("XLK", 1), ("XLK", 0), ("SMH", 0)), _state()))


@pytest.mark.parametrize("bad", [3, -3, True, 1.5])
def test_pod_score_refused(bad):
    assert validate_pod(TECH, PARAMS)(_scores(("XLK", bad), ("SMH", 0)), _state()) is not None


@pytest.mark.parametrize("good", [-2, 2])
def test_pod_score_accepted(good):
    assert validate_pod(TECH, PARAMS)(_scores(("XLK", good), ("SMH", 0)), _state()) is None


def test_reason_length_limits():
    check = validate_pod(TECH, PARAMS)
    assert check(_scores(("XLK", 1), ("SMH", 0), reason="a" * 201), _state()) is not None
    assert check(_scores(("XLK", 1), ("SMH", 0), reason="a" * 200), _state()) is None
    assert check(_scores(("XLK", 1), ("SMH", 0), reason=" "), _state()) is not None
    assert check({**_scores(("XLK", 1), ("SMH", 0)), "sector_view": "a" * 401}, _state()) is not None
    assert check({**_scores(("XLK", 1), ("SMH", 0)), "sector_view": ""}, _state()) is not None


def test_pod_payload_shape():
    check = validate_pod(TECH, PARAMS)
    assert check("nope", _state()) is not None
    assert check({"scores": "x", "sector_view": "v"}, _state()) is not None
    assert check({"scores": ["x"], "sector_view": "v"}, _state()) is not None


def _obj(symbols=("XLK",), kind="macro", severity="low", reason="r"):
    return {"objections": [{"symbols": list(symbols), "kind": kind, "severity": severity, "reason": reason}], "summary": "s"}


def test_risk_review_kinds_and_symbols():
    check = validate_risk(PARAMS)
    assert check(_obj(), _state()) is None
    assert check({"objections": [], "summary": "s"}, _state()) is None
    assert check(_obj(kind="liquidity"), _state()) is not None
    assert check(_obj(symbols=("SPY",)), _state()) is not None
    assert check(_obj(severity="medium"), _state()) is not None
    assert check(_obj(reason="a" * 201), _state()) is not None
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
    check = validate_pm(PARAMS)
    state = _full_state()
    good = {"weights": [{"symbol": "XLK", "weight": 0.2, "reason": "r"}], "overrides": "", "summary": "s"}
    assert check(good, state) is None
    dup = {**good, "weights": good["weights"] * 2}
    assert "XLK" in _msg(check(dup, state))
    assert check({**good, "weights": [{"symbol": "XLK", "weight": 0.3, "reason": "r"}]}, state) is not None
    assert check({**good, "weights": [{"symbol": "XLK", "weight": "a", "reason": "r"}]}, state) is not None
    assert check({**good, "weights": [{"symbol": "XLK", "weight": True, "reason": "r"}]}, state) is not None
    assert check({**good, "summary": ""}, state) is not None
    assert check({**good, "overrides": "a" * 401}, state) is not None


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
    assert (validate_pm(PARAMS)({"weights": [{"symbol": s, "weight": w, "reason": "r"} for s, w in book.items()], "overrides": "", "summary": "s"}, _full_state()) is None) == (message is None)


def test_submit_tools_record_into_their_slot():
    slot = SubmitSlot(validate_pod(TECH, PARAMS))
    slot.arm(_state())
    tool = pod_submit_tool(slot)
    assert tool.__name__ == "submit_pod_view"
    result = tool(scores=_scores(("XLK", 1), ("SMH", 0))["scores"], sector_view="x")
    assert result == {"status": "recorded"}
    assert slot.value is not None and slot.value["scores"][0]["symbol"] == "XLK"
    assert "error" in tool(scores=[], sector_view="x")


def test_risk_and_pm_tools():
    risk = SubmitSlot(validate_risk(PARAMS))
    risk.arm(_state())
    tool = risk_submit_tool(risk)
    assert tool.__name__ == "submit_risk_review"
    assert tool(objections=[], summary="s") == {"status": "recorded"}
    pm = SubmitSlot(validate_pm(PARAMS))
    pm.arm(_full_state())
    tool = pm_submit_tool(pm)
    assert tool.__name__ == "submit_target_weights"
    assert tool(weights=[], overrides="", summary="s") == {"status": "recorded"}

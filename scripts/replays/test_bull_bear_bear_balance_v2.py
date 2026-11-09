"""Tests of the pure parts of the v2 bear-balance replay script (no network, no model).

Run by path: `uv run pytest scripts/replays/test_bull_bear_bear_balance_v2.py` (the repo's `testpaths` stays `tests`).
"""

import importlib.util
import random
from pathlib import Path

import pytest

_PATH = Path(__file__).with_name("bull_bear_bear_balance_v2.py")
_SPEC = importlib.util.spec_from_file_location("bull_bear_bear_balance_v2", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
v2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v2)


def base_inputs() -> dict:
    """Inputs on which all six tests pass; each test below changes one thing."""
    return {
        "bear": {"A": ["high"] * 10 + ["medium"] * 45 + ["low"] * 45, "B": ["high"] * 5 + ["medium"] * 50 + ["low"] * 45},
        "veto": {"J-old": (27, 30), "J-new": (21, 30)},
        "runway_out_new": (8, 10),
        "excess_by_level_b": {"high": [-0.05] * 20, "medium": [-0.01] * 20},
        "runway_high_b": (5, 10),
        "flips": {"B_in_A_out": [0.02], "A_in_B_out": [0.01]},
    }


def run(**changes: object) -> list[dict]:
    """The six tests on `base_inputs()` with `changes` applied."""
    return v2.keep_drop_tests_v2(**{**base_inputs(), **changes})


def test_fresh_pool_removes_v1_reviews() -> None:
    """v1's sample reviews and its anti-goal-only reviews leave the pool."""
    plan = {"sample": [{"index": 0, "date": "d1"}], "anti_goal": {"extra_reviews": ["d3"]}}
    assert v2.fresh_pool(["d1", "d2", "d3", "d4"], plan) == ["d2", "d4"]


def test_full_sample_is_the_fixed_seed_draw() -> None:
    """36 reviews, the seed-20261109 draw of the sorted pool; a smaller pool is an error."""
    pool = [f"2022-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)] + [f"2023-{m:02d}-01" for m in range(1, 13)]
    pool += ["2024-01-01", "2024-02-01"]
    assert v2.full_sample(pool) == sorted(random.Random(20261109).sample(sorted(pool), 36))
    with pytest.raises(ValueError):
        v2.full_sample(pool[:35])


def test_judge_sample_keeps_only_eligible_reviews() -> None:
    """60 of the reviews with a bull medium/high and bear high stock, the seed-20261110 draw."""
    records = {f"e{i:02d}": {"bull": [{"symbol": "X", "conviction": "medium"}], "bear": [{"symbol": "X", "risk": "high"}]} for i in range(61)}
    records |= {f"n{i}": {"bull": [{"symbol": "X", "conviction": "high"}], "bear": [{"symbol": "X", "risk": "medium"}]} for i in range(5)}
    chosen = v2.judge_sample(sorted(records), records)
    assert len(chosen) == 60
    assert all(day.startswith("e") for day in chosen)
    assert chosen == sorted(random.Random(20261110).sample(sorted(d for d in records if d.startswith("e")), 60))
    with pytest.raises(ValueError):
        v2.judge_sample([f"e{i:02d}" for i in range(59)], records)


def test_judge_eligible_needs_bull_medium_or_high_and_bear_high_on_the_same_stock() -> None:
    """Bull medium/high on one stock and bear high on another is not eligible."""
    record = {
        "bull": [{"symbol": "X", "conviction": "low"}, {"symbol": "Y", "conviction": "high"}],
        "bear": [{"symbol": "X", "risk": "high"}, {"symbol": "Y", "risk": "low"}],
    }
    assert not v2.judge_eligible(record)


def test_judge_context_old_is_logged_and_new_adds_fields_only() -> None:
    """J-old equals the logged context; J-new only adds the four fields to each fact sheet, without mutating the log."""
    ctx = {
        "current_datetime": "2024-01-02T09:00:00-05:00",
        "constraints": {"max_in": 10},
        "stocks": [
            {
                "fact_sheet": {"symbol": "X", "operating_margin_pct": 5.0, "operating_margin_prior_year_pct": 20.0},
                "note": "n",
                "bull": {"conviction": "high", "argument": "a"},
                "bear": {"risk": "high", "concern": "fundamentals", "argument": "b"},
                "previous_verdict": {"verdict": "in"},
            }
        ],
    }
    fields = {"2024-01-02": {"X": {"operating_margin_change_pts": -15.0, "days_since_filed": 10, "return_since_filed_pct": -3.0}}}
    assert v2.judge_context("J-old", "2024-01-02", ctx, fields) == ctx
    new = v2.judge_context("J-new", "2024-01-02", ctx, fields)
    assert list(new["stocks"][0]["fact_sheet"]) == ["symbol", "operating_margin_pct", "operating_margin_prior_year_pct", "operating_margin_change_pts", "days_since_filed", "return_since_filed_pct"]
    assert {k: v for k, v in new["stocks"][0].items() if k != "fact_sheet"} == {k: v for k, v in ctx["stocks"][0].items() if k != "fact_sheet"}
    assert new["constraints"] == ctx["constraints"]
    assert "days_since_filed" not in ctx["stocks"][0]["fact_sheet"]


def test_valid_pairs_drop_a_review_failed_in_either_arm() -> None:
    """A review that failed in one arm is excluded from both, and reported per arm."""
    lines = [
        {"date": "d1", "arm": "J-old", "kind": "judge", "ok": True},
        {"date": "d1", "arm": "J-new", "kind": "judge", "ok": False},
        {"date": "d2", "arm": "J-old", "kind": "judge", "ok": True},
        {"date": "d2", "arm": "J-new", "kind": "judge", "ok": True},
        {"date": "d2", "arm": "A", "kind": "full", "ok": True},
    ]
    assert v2.valid_pairs(lines, ["d1", "d2", "d3"], ("J-old", "J-new"), "judge") == (["d2"], {"J-old": [], "J-new": ["d1"]})


def test_done_keys_separate_full_and_judge_lines() -> None:
    """A full line and a judge line of the same review never share a resume key."""
    assert v2.done_key({"date": "d1", "arm": "A", "kind": "full"}) != v2.done_key({"date": "d1", "arm": "J-old", "kind": "judge"})


def test_runway_group_skips_absent_runway() -> None:
    """Only a present runway under 12 months counts; 12 itself and an absent runway do not."""
    sheets = [{"symbol": "A", "runway_months": 11.9}, {"symbol": "B", "runway_months": 12}, {"symbol": "C"}]
    assert v2.runway_symbols(sheets) == {"A"}


def test_all_six_pass_on_base_inputs() -> None:
    """The base inputs pass every test and the decision is KEEP."""
    tests = run()
    assert [t["test"] for t in tests] == [1, 2, 3, 4, 5, 6]
    assert [t["name"] for t in tests] == ["the bear still argues", "no veto", "the judge still removes real risks", "high means risk", "real risks still flagged", "no harm in the flips"]
    assert all(t["pass"] for t in tests)
    assert v2.decision(tests, True) == "KEEP: all six tests pass"


def test_bear_still_argues_fails_below_4_percent() -> None:
    """A high share of 3% fails test 1."""
    assert not run(bear={"A": base_inputs()["bear"]["A"], "B": ["high"] * 3 + ["medium"] * 52 + ["low"] * 45})[0]["pass"]


def test_veto_fails_at_75_percent() -> None:
    """An out rate of 76.7% fails test 2."""
    assert not run(veto={"J-old": (27, 30), "J-new": (23, 30)})[1]["pass"]


def test_veto_needs_15_points_below_old() -> None:
    """A 10-point drop from the old judge fails test 2 even under 75%."""
    assert not run(veto={"J-old": (25, 30), "J-new": (22, 30)})[1]["pass"]


def test_veto_passes_at_exactly_15_points() -> None:
    """85% against 70% is exactly 15 points and passes despite float rounding."""
    assert run(veto={"J-old": (34, 40), "J-new": (28, 40)})[1]["pass"]


def test_veto_not_testable_under_30_stocks() -> None:
    """A group of 29 stocks cannot be tested and does not pass; the note names the minimum."""
    result = run(veto={"J-old": (26, 29), "J-new": (10, 29)})[1]
    assert result["testable"] is False
    assert result["pass"] is False
    assert "30" in result["note"]


def test_runway_out_fails_below_80_percent() -> None:
    """7 of 10 runway stocks put out fails test 3; 8 of 10 passes."""
    assert not run(runway_out_new=(7, 10))[2]["pass"]
    assert run(runway_out_new=(8, 10))[2]["pass"]


def test_runway_out_not_testable_under_10() -> None:
    """9 runway stocks cannot be tested."""
    result = run(runway_out_new=(9, 9))[2]
    assert result["testable"] is False
    assert result["pass"] is False


def test_high_means_risk_fails_when_not_worse() -> None:
    """A high group no worse than the medium group fails test 4."""
    assert not run(excess_by_level_b={"high": [-0.01] * 20, "medium": [-0.01] * 20})[3]["pass"]


def test_high_means_risk_not_testable_under_20() -> None:
    """19 high stocks cannot be tested."""
    result = run(excess_by_level_b={"high": [-0.05] * 19, "medium": [-0.01] * 20})[3]
    assert result["testable"] is False
    assert result["pass"] is False


def test_high_means_risk_ignores_missing_excess() -> None:
    """The group size is the number of excesses present; the result reports it."""
    result = run(excess_by_level_b={"high": [-0.05] * 20, "medium": [-0.01] * 20})[3]
    assert result["pass"]
    assert result["n_high"] == 20


def test_real_risks_flagged_fails_below_half() -> None:
    """4 of 10 runway stocks rated high fails test 5."""
    assert not run(runway_high_b=(4, 10))[4]["pass"]


def test_real_risks_flagged_not_testable_under_10() -> None:
    """9 runway stocks cannot be tested."""
    result = run(runway_high_b=(9, 9))[4]
    assert result["testable"] is False
    assert result["pass"] is False


def test_flips_empty_group_counts_as_zero() -> None:
    """An empty B-in group counts as 0, which beats a negative A-in mean."""
    assert run(flips={"B_in_A_out": [], "A_in_B_out": [-0.01]})[5]["pass"]


def test_decision_lists_each_failure() -> None:
    """The DROP line names each failed test with its name."""
    tests = run(runway_high_b=(4, 10), veto={"J-old": (27, 30), "J-new": (23, 30)})
    assert v2.decision(tests, True) == "DROP: test 2 (no veto) fails; test 5 (real risks still flagged) fails"


def test_decision_none_when_incomplete() -> None:
    """An incomplete replay takes no decision."""
    assert v2.decision(run(), False).startswith("NONE")

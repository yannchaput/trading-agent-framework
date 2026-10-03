from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bill_ackman.hysteresis import advance_cooldowns, apply_verdicts


def _apply(*, holdings=(), ranking=(), verdicts, fail_counts=None, forced_exit_fails: int = 2):
    return apply_verdicts(holdings=list(holdings), ranking=list(ranking), verdicts=verdicts, fail_counts=fail_counts or {}, forced_exit_fails=forced_exit_fails)


def test_a_surviving_holding_resets_its_counter() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "survive"}, fail_counts={"HLT": 1})

    assert outcome.fail_counts == {}
    assert outcome.forced_exits == [] and outcome.pending == []
    assert outcome.allowed == ["HLT"]


def test_a_first_fail_is_pending_and_the_holding_stays_allowed() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "fail"})

    assert outcome.fail_counts == {"HLT": 1}
    assert outcome.pending == ["HLT"]
    assert outcome.forced_exits == []
    assert outcome.allowed == ["HLT"]


def test_a_second_consecutive_fail_forces_the_exit_and_removes_it_from_the_allowed_set() -> None:
    outcome = _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, fail_counts={"HLT": 1})

    assert outcome.forced_exits == ["HLT"]
    assert outcome.pending == []
    assert outcome.allowed == []
    assert outcome.fail_counts == {"HLT": 2}  # kept: if the sell fails the holding is still forced out next time


def test_the_forced_exit_threshold_is_a_parameter() -> None:
    assert _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, forced_exit_fails=1).forced_exits == ["HLT"]
    assert _apply(holdings=["HLT"], verdicts={"HLT": "fail"}, fail_counts={"HLT": 2}, forced_exit_fails=4).pending == ["HLT"]


def test_a_new_candidate_that_fails_is_not_allowed_and_has_no_counter() -> None:
    outcome = _apply(ranking=["NEW"], verdicts={"NEW": "fail"})

    assert outcome.allowed == []
    assert outcome.fail_counts == {}


def test_counters_of_symbols_no_longer_held_are_dropped() -> None:
    outcome = _apply(holdings=["KO"], verdicts={"KO": "survive"}, fail_counts={"GONE": 1, "KO": 1})

    assert outcome.fail_counts == {}


def test_the_allowed_set_lists_ranked_survivors_first_then_other_survivors_then_pending_fails() -> None:
    outcome = _apply(
        holdings=["OLDSURVIVOR", "RANKEDPENDING", "OLDPENDING"],
        ranking=["B", "RANKEDPENDING", "A", "REJECTED"],
        verdicts={
            "A": "survive",
            "B": "survive",
            "REJECTED": "fail",
            "RANKEDPENDING": "fail",
            "OLDSURVIVOR": "survive",
            "OLDPENDING": "fail",
        },
    )

    assert outcome.allowed == ["B", "A", "OLDSURVIVOR", "RANKEDPENDING", "OLDPENDING"]
    assert outcome.pending == ["RANKEDPENDING", "OLDPENDING"]


def test_a_ranked_holding_that_survives_appears_once() -> None:
    outcome = _apply(holdings=["A"], ranking=["A", "B"], verdicts={"A": "survive", "B": "survive"})

    assert outcome.allowed == ["A", "B"]


def test_a_holding_without_a_verdict_is_a_pipeline_bug() -> None:
    with pytest.raises(ValueError, match="HLT"):
        _apply(holdings=["HLT"], verdicts={})


def test_a_ranked_symbol_without_a_verdict_is_a_pipeline_bug() -> None:
    with pytest.raises(ValueError, match="NEW"):
        _apply(ranking=["NEW"], verdicts={})


def test_an_unknown_verdict_value_is_refused() -> None:
    with pytest.raises(ValueError, match="maybe"):
        _apply(holdings=["HLT"], verdicts={"HLT": "maybe"})


def test_several_holdings_are_handled_independently() -> None:
    outcome = _apply(
        holdings=["A", "B", "C"],
        verdicts={"A": "survive", "B": "fail", "C": "fail"},
        fail_counts={"B": 1, "C": 0},
    )

    assert outcome.forced_exits == ["B"]
    assert outcome.pending == ["C"]
    assert outcome.fail_counts == {"B": 2, "C": 1}
    assert outcome.allowed == ["A", "C"]


def test_a_forced_exit_starts_a_cooldown() -> None:
    assert advance_cooldowns({}, forced_exits=["HLT"], reviews=4) == {"HLT": 4}


def test_each_completed_review_counts_every_cooldown_down_and_drops_it_at_zero() -> None:
    assert advance_cooldowns({"A": 3, "B": 1}, forced_exits=[], reviews=4) == {"A": 2}


def test_a_forced_exit_keeps_the_symbol_out_of_the_next_reviews_exactly() -> None:
    cooldowns = advance_cooldowns({}, forced_exits=["HLT"], reviews=4)  # forced out at review R
    for _ in range(4):  # reviews R+1 .. R+4 start with HLT still cooling down
        assert "HLT" in cooldowns
        cooldowns = advance_cooldowns(cooldowns, forced_exits=[], reviews=4)
    assert "HLT" not in cooldowns  # back at review R+5


def test_a_symbol_forced_out_again_restarts_its_cooldown() -> None:
    assert advance_cooldowns({"HLT": 2}, forced_exits=["HLT"], reviews=4) == {"HLT": 4}


def test_zero_reviews_disables_the_cooldown() -> None:
    assert advance_cooldowns({}, forced_exits=["HLT"], reviews=0) == {}


def test_the_input_cooldowns_are_not_modified() -> None:
    before = {"A": 2}

    advance_cooldowns(before, forced_exits=["B"], reviews=4)

    assert before == {"A": 2}

"""bull_bear's agent scores: forward returns of each review's candidates, then each agent's calls against them."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.batch.agent_scores import ForwardReturn, Returns, forward_returns, newey_west_t, score_agents


def _weekdays(start: date, count: int) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


SESSIONS = _weekdays(date(2026, 1, 5), 40)  # Monday 2026-01-05 onwards
TUESDAY = date(2026, 1, 6)  # SESSIONS[1]; its orders fill at SESSIONS[2]'s open


def _review(day: date, symbols: list[str], **fields: Any) -> dict[str, Any]:
    return {"date": day.isoformat(), "abandoned": False, "debate_set": [{"symbol": s, "rank": i + 1, "held": False} for i, s in enumerate(symbols)], **fields}


# --- forward returns ---------------------------------------------------------------------------------------


def test_a_forward_return_runs_from_the_open_after_the_review_to_the_open_h_sessions_later() -> None:
    entry, exit_ = SESSIONS[2], SESSIONS[7]
    opens = {"AAA": {entry: 100.0, exit_: 110.0}, "BBB": {entry: 50.0, exit_: 50.0}}

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB"])], opens, SESSIONS, horizons=(5,))

    by_symbol = {row.symbol: row for row in result.rows}
    assert by_symbol["AAA"] == ForwardReturn(TUESDAY, "AAA", 5, pytest.approx(0.10), pytest.approx(0.05))  # excess: +10% against a +5% mean
    assert by_symbol["BBB"].raw == pytest.approx(0.0) and by_symbol["BBB"].excess == pytest.approx(-0.05)
    assert result.dropped == 0


def test_a_candidate_without_an_open_is_dropped_counted_and_left_out_of_the_mean() -> None:
    entry, exit_ = SESSIONS[2], SESSIONS[7]
    opens = {"AAA": {entry: 100.0, exit_: 110.0}, "BBB": {entry: 50.0, exit_: 50.0}, "CCC": {entry: 10.0}}  # CCC has no exit open

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB", "CCC", "DDD"])], opens, SESSIONS, horizons=(5,))  # DDD: no data at all

    assert sorted(row.symbol for row in result.rows) == ["AAA", "BBB"]
    assert {row.symbol: row.excess for row in result.rows} == {"AAA": pytest.approx(0.05), "BBB": pytest.approx(-0.05)}
    assert result.dropped == 2


def test_an_exit_past_the_last_session_is_immature_and_counted_apart_from_missing_prices() -> None:
    late = SESSIONS[-3]  # entry SESSIONS[-2]; 5 sessions later is past the calendar
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}

    result = forward_returns([_review(late, ["AAA"])], opens, SESSIONS, horizons=(5, 20))

    assert result.rows == [] and result.immature == 2 and result.dropped == 0  # one candidate x two horizons


def test_a_review_before_the_calendars_first_session_is_dropped_not_given_a_later_entry() -> None:
    early = SESSIONS[0] - timedelta(days=3)  # the calendar does not show the session that followed it
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}

    result = forward_returns([_review(early, ["AAA"])], opens, SESSIONS, horizons=(5, 20))

    assert result.rows == [] and result.dropped == 2 and result.immature == 0


def test_each_horizon_gets_its_own_rows_and_its_own_candidate_mean() -> None:
    entry, exit5, exit20 = SESSIONS[2], SESSIONS[7], SESSIONS[22]
    opens = {"AAA": {entry: 100.0, exit5: 102.0, exit20: 120.0}, "BBB": {entry: 100.0, exit5: 100.0, exit20: 100.0}}

    result = forward_returns([_review(TUESDAY, ["AAA", "BBB"])], opens, SESSIONS)

    assert {(row.symbol, row.horizon): row.excess for row in result.rows} == {
        ("AAA", 5): pytest.approx(0.01),
        ("BBB", 5): pytest.approx(-0.01),
        ("AAA", 20): pytest.approx(0.10),
        ("BBB", 20): pytest.approx(-0.10),
    }


def test_abandoned_reviews_and_reviews_without_a_debate_set_give_no_returns() -> None:
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}
    reviews = [_review(TUESDAY, ["AAA"], abandoned=True), {"date": TUESDAY.isoformat(), "abandoned": False, "holdings": ["AAA"]}]

    result = forward_returns(reviews, opens, SESSIONS)

    assert result.rows == [] and result.dropped == 0


# --- the t-stat ---------------------------------------------------------------------------------------------


def test_with_no_lag_the_t_stat_is_the_textbook_one() -> None:
    assert newey_west_t([1.0, 2.0, 3.0, 4.0], lags=0) == pytest.approx(2.5 / (1.2909944 / 2))  # mean / (sample sd / root n)


def test_newey_west_adds_the_bartlett_weighted_autocovariances() -> None:
    # demeaned [-1.5, 0.5, -0.5, 1.5]: gamma0 = 1.25, gamma1 = -0.4375; 1.25 + 2 x 0.5 x -0.4375 = 0.8125, x 4/3 = 1.083333
    assert newey_west_t([1.0, 3.0, 2.0, 4.0], lags=1) == pytest.approx(2.5 / (1.0833333 / 4) ** 0.5)


def test_the_t_stat_needs_two_values_and_some_variance() -> None:
    assert newey_west_t([0.01], lags=0) is None
    assert newey_west_t([0.02, 0.02, 0.02], lags=0) is None


def test_values_equal_up_to_float_noise_have_no_variance_and_no_t_stat() -> None:
    # two reviews whose spreads are the same number, computed from prices that differ by rounding only
    assert newey_west_t([0.05, math.nextafter(0.05, 1.0), 0.05], lags=0) is None


# --- agent scores -------------------------------------------------------------------------------------------

NEXT_TUESDAY = date(2026, 1, 13)


def _returns(*rows: tuple[date, str, float], horizon: int = 5) -> Returns:
    """Rows with a given excess return (raw = excess: only the excess is scored)."""
    return Returns([ForwardReturn(day, symbol, horizon, excess, excess) for day, symbol, excess in rows], dropped=0)


def _bull(**convictions: str) -> list[dict[str, str]]:
    return [{"symbol": s, "conviction": c, "argument": "x"} for s, c in convictions.items()]


def _bear(**risks: tuple[str, str]) -> list[dict[str, str]]:
    return [{"symbol": s, "risk": risk, "concern": concern, "argument": "x"} for s, (risk, concern) in risks.items()]


def _verdicts(**verdicts: str) -> list[dict[str, Any]]:
    return [{"symbol": s, "verdict": v, "reason": "x", "what_changed": None, "previous_verdict": None} for s, v in verdicts.items()]


def test_the_bull_scores_high_against_low_conviction_review_by_review() -> None:
    reviews = [
        _review(TUESDAY, ["AAA", "BBB", "CCC"], bull=_bull(AAA="high", BBB="low", CCC="medium")),
        _review(NEXT_TUESDAY, ["AAA", "BBB", "CCC"], bull=_bull(AAA="high", BBB="low", CCC="medium")),
    ]
    returns = _returns(
        (TUESDAY, "AAA", 0.04), (TUESDAY, "BBB", -0.02), (TUESDAY, "CCC", -0.02),
        (NEXT_TUESDAY, "AAA", 0.01), (NEXT_TUESDAY, "BBB", 0.03), (NEXT_TUESDAY, "CCC", -0.04),
    )  # fmt: skip

    bull = score_agents(reviews, returns, horizons=(5,)).agents[0]

    assert bull.agent == "bull" and bull.spread_label == "high - low"
    score = bull.by_horizon[5]
    assert list(score.groups) == ["low", "medium", "high"]
    high, low, medium = score.groups["high"], score.groups["low"], score.groups["medium"]
    assert (high.n, high.mean_excess, high.hit_rate) == (2, pytest.approx(0.025), 1.0)
    assert (low.n, low.mean_excess, low.hit_rate) == (2, pytest.approx(0.005), 0.5)
    assert (medium.n, medium.mean_excess, medium.hit_rate) == (2, pytest.approx(-0.03), 0.0)
    # per review: 0.04 - (-0.02) = 0.06, then 0.01 - 0.03 = -0.02; mean 0.02, t = 0.02 / (0.0565685 / root 2) = 0.5
    assert score.spread.mean == pytest.approx(0.02) and score.spread.t == pytest.approx(0.5)
    assert (score.spread.reviews, score.spread.skipped) == (2, 0)


def test_the_bear_is_right_when_low_risk_beats_high_risk_and_its_concerns_are_broken_down() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], bear=_bear(AAA=("low", "none"), BBB=("high", "valuation"), CCC=("high", "valuation")))]
    returns = _returns((TUESDAY, "AAA", 0.03), (TUESDAY, "BBB", -0.01), (TUESDAY, "CCC", -0.02))

    bear = score_agents(reviews, returns, horizons=(5,)).agents[1]

    assert bear.agent == "bear" and bear.spread_label == "low - high"
    score = bear.by_horizon[5]
    assert score.spread.mean == pytest.approx(0.03 - (-0.015)) and score.spread.t is None  # one review: no t-stat
    assert score.concerns["valuation"].n == 2 and score.concerns["valuation"].mean_excess == pytest.approx(-0.015)
    assert score.concerns["none"].n == 1


def test_the_judge_scores_in_against_out() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], verdicts=_verdicts(AAA="in", BBB="in", CCC="out"))]
    returns = _returns((TUESDAY, "AAA", 0.02), (TUESDAY, "BBB", 0.0), (TUESDAY, "CCC", -0.02))

    judge = score_agents(reviews, returns, horizons=(5,)).agents[2]

    assert judge.agent == "judge" and judge.spread_label == "in - out"
    score = judge.by_horizon[5]
    assert (score.groups["in"].n, score.groups["out"].n) == (2, 1)
    assert score.spread.mean == pytest.approx(0.01 - (-0.02))


def test_a_picker_record_puts_its_picks_in_and_the_rest_of_its_debate_set_out() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB", "CCC"], picks=[{"symbol": "AAA", "reason": "x"}])]
    returns = _returns((TUESDAY, "AAA", 0.02), (TUESDAY, "BBB", 0.0), (TUESDAY, "CCC", -0.02))

    score = score_agents(reviews, returns, horizons=(5,)).agents[2].by_horizon[5]

    assert (score.groups["in"].n, score.groups["out"].n) == (1, 2)
    assert score.spread.mean == pytest.approx(0.02 - (-0.01))


def test_a_review_missing_one_group_is_skipped_for_the_spread_and_counted() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB"], bull=_bull(AAA="high", BBB="high"))]  # no low conviction: no spread
    returns = _returns((TUESDAY, "AAA", 0.01), (TUESDAY, "BBB", -0.01))

    score = score_agents(reviews, returns, horizons=(5,)).agents[0].by_horizon[5]

    assert score.groups["high"].n == 2
    assert (score.spread.mean, score.spread.reviews, score.spread.skipped) == (None, 0, 1)


def test_an_abandoned_review_keeps_its_verdicts_but_none_of_its_calls_count() -> None:
    abandoned = _review(
        TUESDAY,
        ["AAA", "BBB"],
        abandoned=True,
        bull=_bull(AAA="high", BBB="low"),
        bear=_bear(AAA=("low", "none"), BBB=("high", "fundamentals")),
        verdicts=_verdicts(AAA="in", BBB="out"),
    )
    returns = _returns((TUESDAY, "AAA", 0.05), (TUESDAY, "BBB", -0.05))

    scores = score_agents([abandoned], returns, horizons=(5,))

    for agent in scores.agents:
        score = agent.by_horizon[5]
        assert all(group.n == 0 for group in score.groups.values()) and score.spread.reviews == 0 and score.spread.skipped == 0


def test_a_verdict_on_a_symbol_outside_the_debate_set_is_ignored() -> None:
    reviews = [_review(TUESDAY, ["AAA", "BBB"], verdicts=_verdicts(AAA="in", BBB="out", ZZZ="in"))]
    returns = _returns((TUESDAY, "AAA", 0.01), (TUESDAY, "BBB", -0.01), (NEXT_TUESDAY, "ZZZ", 0.50))  # ZZZ only has a return for another review

    score = score_agents(reviews, returns, horizons=(5,)).agents[2].by_horizon[5]

    assert score.groups["in"].n == 1 and score.spread.mean == pytest.approx(0.02)


def test_the_run_scores_carry_both_counts_of_candidates_left_out() -> None:
    scores = score_agents([], Returns([], dropped=3, immature=5), horizons=(5,))

    assert (scores.dropped, scores.immature) == (3, 5)


def test_the_four_week_spread_uses_newey_west_with_three_lags() -> None:
    days = [TUESDAY + timedelta(weeks=week) for week in range(6)]
    excesses = [(0.03, -0.01), (0.00, 0.02), (0.04, 0.01), (-0.02, 0.01), (0.05, -0.03), (0.01, 0.00)]
    reviews = [_review(day, ["AAA", "BBB"], bull=_bull(AAA="high", BBB="low")) for day in days]
    returns = Returns([ForwardReturn(day, s, 20, x, x) for day, (a, b) in zip(days, excesses, strict=True) for s, x in (("AAA", a), ("BBB", b))], dropped=0)

    spread = score_agents(reviews, returns, horizons=(20,)).agents[0].by_horizon[20].spread

    assert spread.t == pytest.approx(newey_west_t([a - b for a, b in excesses], lags=3))
    assert spread.t != pytest.approx(newey_west_t([a - b for a, b in excesses], lags=0))

"""bull_bear's agent scores: forward returns of each review's candidates, then each agent's calls against them."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.batch.agent_scores import ForwardReturn, forward_returns


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


def test_a_review_whose_exit_is_past_the_last_session_is_dropped_and_counted() -> None:
    late = SESSIONS[-3]  # entry SESSIONS[-2]; 5 sessions later is past the calendar
    opens = {"AAA": {day: 100.0 for day in SESSIONS}}

    result = forward_returns([_review(late, ["AAA"])], opens, SESSIONS, horizons=(5, 20))

    assert result.rows == [] and result.dropped == 2  # one candidate x two horizons


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

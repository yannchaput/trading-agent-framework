from __future__ import annotations

from datetime import date

from trading_agent_framework.strategies.bull_bear.feedback import recent_outcomes
from trading_agent_framework.strategies.bull_bear.market_data import DailySeries

HISTORY = [
    {"date": "2026-09-22", "verdicts": {"AAA": {"verdict": "in", "reason": "r", "close": 100.0}, "BBB": {"verdict": "out", "reason": "lawsuit", "close": 50.0}}},
    {
        "date": "2026-09-29",
        "verdicts": {
            "AAA": {"verdict": "in", "reason": "r", "close": 110.0},
            "CCC": {"verdict": "in", "reason": "r", "close": 20.0},
            "DDD": {"verdict": "out", "reason": "dilution", "close": 10.0},
        },
    },
]


def test_each_out_verdict_is_shown_newest_first_against_that_days_in_stocks() -> None:
    now = {"AAA": 121.0, "BBB": 60.0, "CCC": 22.0, "DDD": 9.0}

    assert recent_outcomes(HISTORY, now) == [
        {"symbol": "DDD", "date": "2026-09-29", "reason": "dilution", "return_since_pct": -10.0, "in_names_return_pct": 10.0},
        {"symbol": "BBB", "date": "2026-09-22", "reason": "lawsuit", "return_since_pct": 20.0, "in_names_return_pct": 21.0},
    ]


def test_a_stock_without_a_price_now_is_left_out_and_never_estimated() -> None:
    assert [outcome["symbol"] for outcome in recent_outcomes(HISTORY, {"AAA": 121.0, "BBB": 60.0, "CCC": 22.0})] == ["BBB"]


def test_a_review_whose_in_stocks_have_no_price_shows_none_for_them() -> None:
    assert recent_outcomes(HISTORY[:1], {"BBB": 60.0})[0]["in_names_return_pct"] is None


def test_no_history_means_no_feedback() -> None:
    assert recent_outcomes([], {"AAA": 1.0}) == []


# --- the "then" close comes from today's series ------------------------------------------------------------


def _series() -> DailySeries:
    """Closes on a split-adjusted basis: 50 on Monday 09-28 (the stored 100 was the unadjusted close), 55 now."""
    return DailySeries(closes=[48.0, 50.0, 52.0, 55.0], volumes=[1.0] * 4, dates=[date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29), date(2026, 9, 30)])


def _before(symbol: str, review_date: str) -> float | None:
    return _series().close_before(date.fromisoformat(review_date)) if symbol == "AAA" else None


def test_a_split_in_the_window_does_not_show_a_fake_return() -> None:
    history = [{"date": "2026-09-29", "verdicts": {"AAA": {"verdict": "out", "reason": "r", "close": 100.0}}}]

    unadjusted = recent_outcomes(history, {"AAA": 55.0})
    adjusted = recent_outcomes(history, {"AAA": 55.0}, _before)

    assert unadjusted[0]["return_since_pct"] == -45.0
    assert adjusted[0]["return_since_pct"] == 10.0  # 55 against the 50 of the session before 09-29, not against the stored 100


def test_the_stored_close_is_the_fallback_when_the_series_has_no_earlier_session() -> None:
    history = [{"date": "2026-09-25", "verdicts": {"AAA": {"verdict": "out", "reason": "r", "close": 40.0}, "BBB": {"verdict": "out", "reason": "r", "close": 20.0}}}]

    shown = recent_outcomes(history, {"AAA": 50.0, "BBB": 25.0}, _before)

    # AAA: no session before 09-25 in the series; BBB: no series at all
    assert {outcome["symbol"]: outcome["return_since_pct"] for outcome in shown} == {"AAA": 25.0, "BBB": 25.0}


def test_the_close_before_a_date_is_strictly_before_it() -> None:
    series = _series()

    assert series.close_before(date(2026, 9, 29)) == 50.0  # the bar dated 09-29 is not used
    assert series.close_before(date(2026, 9, 30)) == 52.0
    assert series.close_before(date(2026, 9, 28)) == 48.0
    assert series.close_before(date(2026, 9, 25)) is None
    assert series.close_before(date(2026, 10, 15)) == 55.0
    assert DailySeries(closes=[1.0], volumes=[1.0]).close_before(date(2026, 9, 29)) is None


def test_a_review_dated_on_a_bar_uses_the_session_before_that_bar() -> None:
    history = [{"date": "2026-09-29", "verdicts": {"AAA": {"verdict": "out", "reason": "r", "close": 1.0}}}]
    seen: list[tuple[str, str]] = []

    def spy(symbol: str, review_date: str) -> float | None:
        seen.append((symbol, review_date))
        return _before(symbol, review_date)

    assert recent_outcomes(history, {"AAA": 55.0}, spy)[0]["return_since_pct"] == 10.0
    assert seen == [("AAA", "2026-09-29")]


def test_a_reason_is_cut_to_150_characters() -> None:
    long = "x" * 400
    short = "y" * 150
    history = [{"date": "2026-09-29", "verdicts": {"AAA": {"verdict": "out", "reason": long, "close": 10.0}, "BBB": {"verdict": "out", "reason": short, "close": 10.0}}}]

    shown = {outcome["symbol"]: outcome["reason"] for outcome in recent_outcomes(history, {"AAA": 11.0, "BBB": 11.0})}

    assert shown["AAA"] == "x" * 150 + "..."
    assert shown["BBB"] == short

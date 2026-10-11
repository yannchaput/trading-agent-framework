from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.feedback import recent_outcomes

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

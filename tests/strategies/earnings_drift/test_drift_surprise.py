from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal as D

import pytest

from trading_agent_framework.strategies.earnings_drift.surprise import Surprise, articles_for, parse_surprise, pick_surprise


@pytest.mark.parametrize(
    ("headline", "eps", "estimate", "sales", "sales_estimate"),
    [
        ("Omnicom Group Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "1.77", "1.68", "3440000000", "3360000000"),
        ("XYZ Corp Q2 Adj. EPS $(0.12) Misses $(0.05) Estimate, Sales $120.5M Beat $118M Estimate", "-0.12", "-0.05", "120500000", "118000000"),
        ("ABC Inc Q4 EPS $-0.30 Beats $-0.41 Estimate", "-0.30", "-0.41", None, None),
        ("Small Co FY25 Adj EPS $2.10 In-Line With $2.10 Estimate, Revenue $850K Miss $900K Estimate", "2.10", "2.10", "850000", "900000"),
        ("CORRECTION: Omnicom Group Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "1.77", "1.68", "3440000000", "3360000000"),
    ],
)
def test_parse_surprise_reads_the_numbers(headline: str, eps: str, estimate: str, sales: str | None, sales_estimate: str | None) -> None:
    surprise = parse_surprise(headline)
    assert surprise is not None
    assert (surprise.eps_actual, surprise.eps_estimate) == (D(eps), D(estimate))
    assert surprise.sales_actual == (D(sales) if sales else None)
    assert surprise.sales_estimate == (D(sales_estimate) if sales_estimate else None)


@pytest.mark.parametrize("headline", ["XYZ Q3 EPS $1.20 Up From $1.00 YoY", "Omnicom Group: Q3 Earnings Insights", "", "Earnings Scheduled For October 18, 2022"])
def test_headlines_without_an_estimate_are_not_surprises(headline: str) -> None:
    assert parse_surprise(headline) is None


def test_the_result_comes_from_the_numbers_not_the_verb() -> None:
    surprise = parse_surprise("Mislabelled Inc Q1 EPS $1.00 Misses $0.90 Estimate, Sales $10M Beat $12M Estimate")
    assert surprise is not None
    assert surprise.eps_beat is True and surprise.eps_result == "BEAT"
    assert surprise.sales_result == "MISS"


def test_surprise_percentages_and_results() -> None:
    beat = Surprise(D("1.52"), D("1.20"), D("110"), D("100"))
    assert beat.eps_surprise_pct == pytest.approx(0.26667, rel=1e-4)
    assert beat.sales_surprise_pct == pytest.approx(0.10)
    negative = Surprise(D("-0.30"), D("-0.41"))
    assert negative.eps_beat and negative.eps_surprise_pct == pytest.approx(0.26829, rel=1e-4)
    assert Surprise(D("0.10"), D("0")).eps_surprise_pct is None
    assert Surprise(D("2.10"), D("2.10")).eps_result == "IN-LINE"
    assert Surprise(D("1"), D("2")).eps_result == "MISS"
    assert Surprise(D("1"), D("2")).sales_result is None and Surprise(D("1"), D("2")).sales_surprise_pct is None


def _article(headline: str, created_at: str, symbols: list[str]) -> dict[str, object]:
    return {"headline": headline, "created_at": created_at, "symbols": symbols}


def test_articles_for_keeps_the_symbol_and_the_window_oldest_first() -> None:
    articles = [
        _article("late", "2026-09-01T21:00:00+00:00", ["AAA"]),
        _article("early", "2026-09-01T20:00:00+00:00", ["AAA", "BBB"]),
        _article("other symbol", "2026-09-01T20:30:00+00:00", ["BBB"]),
        _article("too old", "2026-09-01T10:00:00+00:00", ["AAA"]),
        _article("no time", "", ["AAA"]),
    ]
    kept = articles_for(articles, "aaa", start=datetime(2026, 9, 1, 18, tzinfo=UTC), end=datetime(2026, 9, 2, tzinfo=UTC))
    assert [a["headline"] for a in kept] == ["early", "late"]


def test_pick_prefers_the_latest_correction_then_the_earliest_parsable() -> None:
    original = _article("AAA Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44M Miss $3.36B Estimate", "2026-09-01T20:06:37Z", ["AAA"])
    correction = _article("CORRECTION: AAA Q3 EPS $1.77 Beats $1.68 Estimate, Sales $3.44B Beat $3.36B Estimate", "2026-09-01T20:14:28Z", ["AAA"])
    noise = _article("AAA: Q3 Earnings Insights", "2026-09-01T21:09:47Z", ["AAA"])
    picked = pick_surprise([original, correction, noise])
    assert picked is not None and picked.headline.startswith("CORRECTION:")
    assert picked.surprise.sales_actual == D("3440000000")
    assert picked.created_at == datetime(2026, 9, 1, 20, 14, 28, tzinfo=UTC)
    first = pick_surprise([original, noise])
    assert first is not None and first.headline == original["headline"]
    assert pick_surprise([noise]) is None

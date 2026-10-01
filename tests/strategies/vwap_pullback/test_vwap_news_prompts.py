from __future__ import annotations

from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines


def test_lean_headlines_keeps_the_newest_distinct_headlines() -> None:
    articles = [
        {"headline": "AAA beats estimates", "created_at": "2026-09-01T12:00:00Z", "source": "benzinga", "summary": "long text"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "", "created_at": "2026-09-01T14:00:00Z", "source": "x"},
        {"headline": "Old news", "created_at": "2026-08-30T12:00:00Z", "source": "x"},
    ]
    rows = lean_headlines(articles, 2)
    assert rows == [
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
    ]

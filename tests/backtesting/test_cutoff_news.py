"""Tests for `CutoffNewsProvider`: which source answers a window around the cutoff, and how a spanning window merges."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from trading_agent_framework.backtesting.news import GDELT_CUTOFF, CutoffNewsProvider
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.utils.errors import BrokerError

_DAY = timedelta(days=1)


def _article(created: datetime) -> dict[str, object]:
    return {"headline": created.isoformat(), "created_at": created.isoformat()}


class _Fake:
    """A provider holding fixed articles: answers like a real one (window, sort, limit) and records each call."""

    def __init__(self, articles: list[datetime]) -> None:
        self.articles = articles
        self.calls: list[dict[str, object]] = []

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
        sort: str | None = None,
    ) -> list[dict[str, object]]:
        self.calls.append({"symbols": list(symbols), "start": start, "end": end, "limit": limit, "sort": sort})
        found = [t for t in self.articles if (start is None or start <= t) and t <= end]
        return [_article(t) for t in sorted(found, reverse=sort != "asc")[:limit]]


def _router(before: _Fake, after: _Fake) -> tuple[CutoffNewsProvider, list[int]]:
    built: list[int] = []

    def factory() -> NewsProvider:
        built.append(1)
        return after

    return CutoffNewsProvider(before, factory), built


def test_a_window_before_the_cutoff_asks_only_before_and_never_builds_after() -> None:
    before, after = _Fake([GDELT_CUTOFF - 3 * _DAY]), _Fake([])
    router, built = _router(before, after)

    articles = router.get_news(["NVDA"], start=GDELT_CUTOFF - 5 * _DAY, end=GDELT_CUTOFF - _DAY, limit=5)

    assert len(articles) == 1
    assert before.calls[0]["symbols"] == ["NVDA"]
    assert before.calls[0]["limit"] == 5
    assert built == []


def test_a_window_from_the_cutoff_on_asks_only_after() -> None:
    before, after = _Fake([GDELT_CUTOFF - _DAY]), _Fake([GDELT_CUTOFF + _DAY])
    router, _ = _router(before, after)

    articles = router.get_news(start=GDELT_CUTOFF, end=GDELT_CUTOFF + 2 * _DAY)

    assert [a["created_at"] for a in articles] == [(GDELT_CUTOFF + _DAY).isoformat()]
    assert before.calls == []


def test_a_window_ending_exactly_at_the_cutoff_asks_after_for_the_cutoff_instant_only() -> None:
    on_cutoff = _Fake([GDELT_CUTOFF])
    before = _Fake([GDELT_CUTOFF - _DAY, GDELT_CUTOFF])  # holds an article AT the cutoff: it must not come from here
    router, _ = _router(before, on_cutoff)

    articles = router.get_news(start=GDELT_CUTOFF - 2 * _DAY, end=GDELT_CUTOFF)

    assert before.calls[0]["end"] == GDELT_CUTOFF - timedelta(microseconds=1)
    assert (on_cutoff.calls[0]["start"], on_cutoff.calls[0]["end"]) == (GDELT_CUTOFF, GDELT_CUTOFF)
    assert [a["created_at"] for a in articles] == [GDELT_CUTOFF.isoformat(), (GDELT_CUTOFF - _DAY).isoformat()]


def test_a_spanning_window_splits_at_the_cutoff_merges_newest_first_and_limits() -> None:
    before = _Fake([GDELT_CUTOFF - 2 * _DAY, GDELT_CUTOFF - _DAY])
    after = _Fake([GDELT_CUTOFF + _DAY, GDELT_CUTOFF + 2 * _DAY])
    router, _ = _router(before, after)

    articles = router.get_news(start=GDELT_CUTOFF - 5 * _DAY, end=GDELT_CUTOFF + 5 * _DAY, limit=3)

    assert [a["created_at"] for a in articles] == [t.isoformat() for t in (GDELT_CUTOFF + 2 * _DAY, GDELT_CUTOFF + _DAY, GDELT_CUTOFF - _DAY)]
    assert before.calls[0]["end"] == GDELT_CUTOFF - timedelta(microseconds=1)
    assert after.calls[0]["start"] == GDELT_CUTOFF


def test_a_spanning_window_with_sort_asc_merges_oldest_first() -> None:
    before = _Fake([GDELT_CUTOFF - 2 * _DAY, GDELT_CUTOFF - _DAY])
    after = _Fake([GDELT_CUTOFF + _DAY])
    router, _ = _router(before, after)

    articles = router.get_news(start=GDELT_CUTOFF - 5 * _DAY, end=GDELT_CUTOFF + 5 * _DAY, limit=2, sort="asc")

    assert [a["created_at"] for a in articles] == [t.isoformat() for t in (GDELT_CUTOFF - 2 * _DAY, GDELT_CUTOFF - _DAY)]


def test_start_none_with_end_after_the_cutoff_asks_only_after() -> None:
    before, after = _Fake([]), _Fake([GDELT_CUTOFF + _DAY])
    router, _ = _router(before, after)

    assert len(router.get_news(end=GDELT_CUTOFF + 2 * _DAY)) == 1
    assert before.calls == []


def test_the_after_factory_is_built_once() -> None:
    router, built = _router(_Fake([]), _Fake([]))

    router.get_news(start=GDELT_CUTOFF, end=GDELT_CUTOFF + _DAY)
    router.get_news(start=GDELT_CUTOFF, end=GDELT_CUTOFF + _DAY)

    assert built == [1]


def test_a_factory_broker_error_propagates() -> None:
    def factory() -> NewsProvider:
        raise BrokerError("no news source available: no keys")

    router = CutoffNewsProvider(_Fake([]), factory)

    with pytest.raises(BrokerError, match="no news source available"):
        router.get_news(end=datetime(2026, 6, 1, tzinfo=UTC))

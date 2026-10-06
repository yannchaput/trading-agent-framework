"""`FakeNewsProvider` is a test double other suites lean on: its `sort` must behave like Alpaca's (sort, then limit)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from tests.fakes import FakeNewsProvider, et

_START = et(2026, 9, 14, 9)
_END = et(2026, 9, 14, 17)


def _articles(*minutes: int) -> dict[str, list[dict[str, object]]]:
    return {"AAA": [{"headline": f"h{m}", "created_at": (_START + timedelta(minutes=m)).isoformat(), "symbols": ["AAA"]} for m in minutes]}


def _headlines(rows: list[dict[str, object]]) -> list[object]:
    return [row["headline"] for row in rows]


def test_without_a_sort_the_insertion_order_is_kept() -> None:
    provider = FakeNewsProvider(_articles(30, 10, 20))

    assert _headlines(provider.get_news(["AAA"], start=_START, end=_END, limit=10)) == ["h30", "h10", "h20"]


def test_asc_orders_oldest_first_before_the_limit() -> None:
    provider = FakeNewsProvider(_articles(30, 10, 20))

    assert _headlines(provider.get_news(["AAA"], start=_START, end=_END, limit=2, sort="asc")) == ["h10", "h20"]


def test_desc_orders_newest_first_before_the_limit() -> None:
    provider = FakeNewsProvider(_articles(30, 10, 20))

    assert _headlines(provider.get_news(["AAA"], start=_START, end=_END, limit=2, sort="desc")) == ["h30", "h20"]


def test_the_sort_is_recorded_apart_from_the_five_tuple_calls() -> None:
    provider = FakeNewsProvider(_articles(10))

    provider.get_news(["AAA"], start=_START, end=_END, limit=3, sort="asc")
    provider.get_news(["AAA"], start=_START, end=_END, limit=3)

    assert provider.sorts == ["asc", None]
    assert provider.calls == [(("AAA",), _START, _END, 3, False)] * 2


def test_an_unknown_sort_is_refused() -> None:
    with pytest.raises(ValueError, match="sort"):
        FakeNewsProvider(_articles(10)).get_news(["AAA"], end=_END, sort="newest")

# tests/strategies/earnings_drift/test_drift_events.py
from __future__ import annotations

from datetime import UTC, date, datetime

from tests.fakes import et

from trading_agent_framework.strategies.earnings_drift.events import (
    EarningsEvent,
    earnings_events,
    events_reacting_on,
    older_pages,
    reaction_date,
    release_timing,
)

# Tue 1 Sep .. Mon 7 Sep 2026 (the fake calendar ignores Labor Day): Wed 2, Thu 3, Fri 4, Mon 7.
DATES = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]


def _columns(rows: list[tuple[str, str, str, str]]) -> dict[str, list[str]]:
    """rows: (form, items, acceptanceDateTime, accessionNumber)."""
    return {
        "form": [r[0] for r in rows],
        "items": [r[1] for r in rows],
        "acceptanceDateTime": [r[2] for r in rows],
        "accessionNumber": [r[3] for r in rows],
        "primaryDocument": [f"{r[3]}.htm" for r in rows],
        "filingDate": [r[2][:10] for r in rows],
    }


def test_only_8k_filings_with_item_2_02_are_events_across_recent_and_older_pages() -> None:
    recent = {
        "filings": {
            "recent": _columns(
                [
                    ("8-K", "2.02,9.01", "2026-07-28T20:07:49.000Z", "acc-3"),
                    ("8-K", "7.01", "2026-06-01T12:00:00.000Z", "acc-x1"),
                    ("8-K/A", "2.02", "2026-07-29T12:00:00.000Z", "acc-x2"),
                    ("10-Q", "", "2026-07-30T12:00:00.000Z", "acc-x3"),
                    ("8-K", "1.01,2.03", "2026-05-01T12:00:00.000Z", "acc-x4"),
                    ("8-K", "2.02", "2026-04-28T20:08:25.000Z", "acc-2"),
                ]
            ),
            "files": [],
        }
    }
    page = _columns([("8-K", "2.02,7.01,9.01", "2026-01-20T21:01:00.000Z", "acc-1"), ("8-K", "2.02", "2026-04-28T20:08:25.000Z", "acc-2")])
    events = earnings_events("omc", [recent, page])
    assert [e.accession_number for e in events] == ["acc-1", "acc-2", "acc-3"]  # oldest first, the duplicate accession once
    assert events[0] == EarningsEvent("OMC", datetime(2026, 1, 20, 21, 1, tzinfo=UTC), "acc-1", "acc-1.htm")


def test_a_row_without_a_timestamp_is_skipped() -> None:
    payload = {"filings": {"recent": _columns([("8-K", "2.02", "", "acc-1")])}}
    assert earnings_events("AAA", [payload]) == []


def test_older_pages_are_those_reaching_since() -> None:
    payload = {
        "filings": {
            "recent": {},
            "files": [
                {"name": "CIK1-submissions-001.json", "filingFrom": "2024-01-01", "filingTo": "2026-01-31"},
                {"name": "CIK1-submissions-002.json", "filingFrom": "2010-01-01", "filingTo": "2023-12-31"},
                {"name": "bad.json"},
            ],
        }
    }
    assert older_pages(payload, date(2025, 9, 1)) == ["CIK1-submissions-001.json"]
    assert older_pages(payload, date(2023, 6, 1)) == ["CIK1-submissions-001.json", "CIK1-submissions-002.json"]


def test_reaction_date_is_the_first_session_closing_after_the_release() -> None:
    assert reaction_date(et(2026, 9, 2, 7, 0), DATES) == date(2026, 9, 2)  # before the open
    assert reaction_date(et(2026, 9, 2, 12, 0), DATES) == date(2026, 9, 2)  # during the session
    assert reaction_date(et(2026, 9, 2, 16, 0), DATES) == date(2026, 9, 3)  # at the close: next session
    assert reaction_date(et(2026, 9, 2, 16, 5), DATES) == date(2026, 9, 3)  # after the close
    assert reaction_date(et(2026, 9, 4, 16, 30), DATES) == date(2026, 9, 7)  # Friday evening: Monday
    assert reaction_date(et(2026, 9, 7, 16, 30), DATES) is None  # past the last known session


def _event(symbol: str, accepted_at: datetime, accession: str = "") -> EarningsEvent:
    return EarningsEvent(symbol, accepted_at, accession or f"{symbol}-{accepted_at.isoformat()}", "doc.htm")


def test_events_reacting_on_a_day() -> None:
    events = [
        _event("AAA", et(2026, 9, 2, 16, 5)),  # reacts Thu 3
        _event("BBB", et(2026, 9, 3, 7, 0)),  # reacts Thu 3
        _event("BBB", et(2026, 9, 3, 8, 0)),  # same symbol, same reaction: only the earliest is kept
        _event("CCC", et(2026, 9, 3, 16, 1)),  # reacts Fri 4
        _event("DDD", et(2026, 9, 2, 15, 0)),  # reacted Wed 2
    ]
    found = events_reacting_on(events, date(2026, 9, 3), DATES, now=et(2026, 9, 3, 16, 0))
    assert [(e.symbol, e.accepted_at) for e in found] == [("AAA", et(2026, 9, 2, 16, 5)), ("BBB", et(2026, 9, 3, 7, 0))]


def test_events_reacting_on_ignores_the_future_unknown_days_and_the_first_date() -> None:
    late = _event("AAA", et(2026, 9, 3, 15, 0))
    assert events_reacting_on([late], date(2026, 9, 3), DATES, now=et(2026, 9, 3, 14, 0)) == []  # not filed yet at `now`
    assert events_reacting_on([late], date(2026, 9, 5), DATES, now=et(2026, 9, 5, 16, 0)) == []  # not a trading date
    assert events_reacting_on([_event("AAA", et(2026, 9, 1, 7, 0))], date(2026, 9, 1), DATES, now=et(2026, 9, 1, 16, 0)) == []


def test_release_timing() -> None:
    assert release_timing(et(2026, 9, 2, 16, 5), date(2026, 9, 3)) == "after_close"
    assert release_timing(et(2026, 9, 3, 7, 0), date(2026, 9, 3)) == "before_open"
    assert release_timing(et(2026, 9, 3, 11, 0), date(2026, 9, 3)) == "during_session"

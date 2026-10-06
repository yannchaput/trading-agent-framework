# src/trading_agent_framework/strategies/earnings_drift/events.py
"""Earnings events from SEC submissions, and the session each one moves the stock in (spec §3.1). Pure.

An event is an 8-K carrying item 2.02 ("Results of Operations"), timed by its `acceptanceDateTime`. Its reaction
session is the first session whose close is strictly after that time. Sessions are given as trading dates (the
benchmark's daily bars) and every close is taken as 16:00 ET: early closes are a known limit (spec §11).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from trading_agent_framework.fundamentals.sec import parse_dt
from trading_agent_framework.utils.clock import MARKET_TZ

EARNINGS_ITEM = "2.02"
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


@dataclass(frozen=True, slots=True)
class EarningsEvent:
    symbol: str
    accepted_at: datetime  # aware
    accession_number: str
    primary_document: str


def _columns(payload: Mapping[str, Any]) -> Mapping[str, Sequence[Any]]:
    """The columnar filing table: `filings.recent` of a submissions payload, or the top level of an older page."""
    filings = payload.get("filings")
    if isinstance(filings, Mapping):
        recent = filings.get("recent")
        return recent if isinstance(recent, Mapping) else {}
    return payload


def _at(columns: Mapping[str, Sequence[Any]], key: str, index: int) -> Any:
    values = columns.get(key) or []
    return values[index] if index < len(values) else None


def earnings_events(symbol: str, payloads: Iterable[Mapping[str, Any]]) -> list[EarningsEvent]:
    """Every 8-K carrying item 2.02 across `payloads`, oldest first, one per accession number."""
    seen: set[str] = set()
    events: list[EarningsEvent] = []
    for payload in payloads:
        columns = _columns(payload)
        for index, form in enumerate(columns.get("form") or []):
            if str(form).upper() != "8-K":
                continue
            items = [part.strip() for part in str(_at(columns, "items", index) or "").split(",")]
            if EARNINGS_ITEM not in items:
                continue
            accepted = parse_dt(_at(columns, "acceptanceDateTime", index))
            accession = str(_at(columns, "accessionNumber", index) or "")
            if accepted is None or accepted.tzinfo is None or not accession or accession in seen:
                continue
            seen.add(accession)
            events.append(EarningsEvent(symbol.upper(), accepted, accession, str(_at(columns, "primaryDocument", index) or "")))
    return sorted(events, key=lambda event: event.accepted_at)


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def older_pages(payload: Mapping[str, Any], since: date) -> list[str]:
    """Names of the older submission pages whose `filingTo` is on or after `since`."""
    filings = payload.get("filings")
    files = filings.get("files") if isinstance(filings, Mapping) else None
    names: list[str] = []
    for entry in files or []:
        if not isinstance(entry, Mapping) or not entry.get("name"):
            continue
        filing_to = _iso_date(entry.get("filingTo"))
        if filing_to is not None and filing_to >= since:
            names.append(str(entry["name"]))
    return names


def session_close(day: date) -> datetime:
    return datetime.combine(day, SESSION_CLOSE, tzinfo=MARKET_TZ)


def reaction_date(accepted_at: datetime, trading_dates: Sequence[date]) -> date | None:
    """The first trading date whose 16:00 ET close is strictly after `accepted_at`; None past the last date."""
    for day in sorted(trading_dates):
        if session_close(day) > accepted_at:
            return day
    return None


def events_reacting_on(events: Iterable[EarningsEvent], day: date, trading_dates: Sequence[date], now: datetime) -> list[EarningsEvent]:
    """The events whose reaction date is `day` and that were filed by `now`: one per symbol, its earliest, by symbol."""
    dates = sorted(trading_dates)
    if day not in dates or dates.index(day) == 0:
        return []
    lower, upper = session_close(dates[dates.index(day) - 1]), session_close(day)
    earliest: dict[str, EarningsEvent] = {}
    for event in events:
        if lower <= event.accepted_at < upper and event.accepted_at <= now:
            kept = earliest.get(event.symbol)
            if kept is None or event.accepted_at < kept.accepted_at:
                earliest[event.symbol] = event
    return [earliest[symbol] for symbol in sorted(earliest)]


def news_window(reaction_day: date, trading_dates: Sequence[date], now: datetime) -> tuple[datetime, datetime] | None:
    """Where an earnings headline for `reaction_day` can be: the previous session's close to this session's close, never past `now`.

    Anchored on the session, not on the 8-K: SEC's acceptance time can trail the real release by hours (about 4 h
    for JPM, UNH, AAPL, OMC), so a window built around it misses a wire that went out before it. Every wire that
    moves `reaction_day` falls after the previous close (an after-close release) or before this close (a pre-open
    or intraday one). Closes are taken as 16:00 ET (early closes are a known limit, spec §11). None when
    `reaction_day` is not a trading date, has no previous one, or `now` is before the previous close (the window
    would end before it starts).
    """
    dates = sorted(trading_dates)
    if reaction_day not in dates or dates.index(reaction_day) == 0:
        return None
    start, end = session_close(dates[dates.index(reaction_day) - 1]), min(now, session_close(reaction_day))
    return (start, end) if end >= start else None


def release_timing(accepted_at: datetime, reaction_day: date) -> str:
    local = accepted_at.astimezone(MARKET_TZ)
    if local.date() < reaction_day:
        return "after_close"
    return "before_open" if local.time() < SESSION_OPEN else "during_session"

"""SEC submissions for the universe, reduced to earnings events and kept in memory (spec §3.2).

The only module of the strategy that talks to SEC EDGAR. Every payload goes through `SecEdgarClient`'s cache with
`max_age_days=0` against the strategy clock: a backtest never refetches a file fetched after its simulated date,
paper/live refetch at every load. A backtest loads once (`reload_every_cycle=False`), paper/live at every cycle.
Older pages (`filings.files`) never change and are cached for good.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol

from trading_agent_framework.fundamentals.edgar_client import SEC_DATA_BASE_URL, SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent, earnings_events, older_pages
from trading_agent_framework.utils.errors import FundamentalsError


@dataclass(frozen=True, slots=True)
class LoadReport:
    loaded: int
    failed: dict[str, str] = field(default_factory=dict)  # symbol -> error message

    def is_hollow(self, fraction: float, min_failures: int) -> bool:
        """Too many symbols failed to trust the events: at least `min_failures`, and more than `fraction` of all."""
        total = self.loaded + len(self.failed)
        return total > 0 and len(self.failed) >= min_failures and len(self.failed) > fraction * total


class EventProvider(Protocol):
    def needs_load(self) -> bool: ...

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport: ...

    def discard(self) -> None: ...

    def all_events(self) -> list[EarningsEvent]: ...


class EventSource:
    def __init__(self, client: SecEdgarClient, *, reload_every_cycle: bool) -> None:
        self._client = client
        self._reload_every_cycle = reload_every_cycle
        self._events: dict[str, list[EarningsEvent]] | None = None

    def needs_load(self) -> bool:
        return self._events is None or self._reload_every_cycle

    def load(self, symbols: Sequence[str], *, as_of: datetime, since: date) -> LoadReport:
        events: dict[str, list[EarningsEvent]] = {}
        failed: dict[str, str] = {}
        for symbol in symbols:
            try:
                events[symbol.upper()] = self._symbol_events(symbol, as_of=as_of, since=since)
            except FundamentalsError as exc:
                failed[symbol.upper()] = str(exc)
        self._events = events
        return LoadReport(loaded=len(events), failed=failed)

    def discard(self) -> None:
        self._events = None

    def all_events(self) -> list[EarningsEvent]:
        return [event for events in (self._events or {}).values() for event in events]

    def _symbol_events(self, symbol: str, *, as_of: datetime, since: date) -> list[EarningsEvent]:
        cik = self._client.ticker_to_cik(symbol)
        recent = self._client.get_submissions_payload(cik, as_of=as_of, max_age_days=0)
        pages = [self._client.get_json(f"{SEC_DATA_BASE_URL}/submissions/{name}", ("submissions", name)) for name in older_pages(recent, since)]
        return earnings_events(symbol, [recent, *pages])

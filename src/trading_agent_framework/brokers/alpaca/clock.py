"""`MarketClock` backed by Alpaca's market calendar (GET /calendar)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

from trading_agent_framework.brokers.alpaca import account
from trading_agent_framework.brokers.alpaca.orders import AlpacaTradingClient
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.errors import BrokerError


class AlpacaMarketClock(MarketClock):
    """Sessions (early closes included) come from Alpaca's calendar, fetched about ten
    trading days at a time and refetched once every cached session has closed.
    Nothing is fetched until the first `next_session()` call."""

    LOOKAHEAD_DAYS = 14

    def __init__(
        self, client: AlpacaTradingClient, now: Callable[[], datetime] | None = None
    ) -> None:
        self._client = client
        self._now = now
        self._sessions: list[MarketSession] = []

    def now(self) -> datetime:
        return self._now() if self._now is not None else super().now()

    def next_session(self) -> MarketSession:
        now = self.now()
        session = self._first_session_closing_after(now)
        if session is None:
            self._refresh(now.date())
            session = self._first_session_closing_after(now)
        if session is None:
            raise BrokerError(f"Alpaca calendar has no session after {now.isoformat()}")
        return session

    def _first_session_closing_after(self, now: datetime) -> MarketSession | None:
        return next((s for s in self._sessions if s.close > now), None)

    def _refresh(self, start: date) -> None:
        request = account.build_calendar_request(
            start, start + timedelta(days=self.LOOKAHEAD_DAYS)
        )
        try:
            responses = self._client.get_calendar(filters=request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca market calendar: {exc}") from exc
        self._sessions = account.parse_calendar(responses, self.tz)

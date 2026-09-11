from __future__ import annotations

from datetime import date, datetime

import pytest
from tests.fakes import (
    FakeTradingClient,
    et,
    make_alpaca_calendar,
    make_api_error,
    make_session,
)

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.clock import MARKET_TZ
from trading_agent_framework.errors import BrokerError


class _Now:
    """Mutable wall clock for tests."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _client_with_days(*days: str) -> FakeTradingClient:
    client = FakeTradingClient()
    client.calendar_response = [make_alpaca_calendar(day) for day in days]
    return client


def test_construction_does_not_fetch_the_calendar() -> None:
    client = _client_with_days("2026-09-14")
    AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))
    assert client.calendar_requests == []


def test_next_session_fetches_a_lookahead_window_from_today() -> None:
    client = _client_with_days("2026-09-14", "2026-09-15")
    clock = AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))

    assert clock.next_session() == make_session(date(2026, 9, 14))
    [request] = client.calendar_requests
    assert request.start == date(2026, 9, 14)
    assert request.end == date(2026, 9, 28)


def test_next_session_returns_the_open_session_and_uses_the_cache() -> None:
    client = _client_with_days("2026-09-14", "2026-09-15")
    now = _Now(et(2026, 9, 14, 7))
    clock = AlpacaMarketClock(client, now=now)
    clock.next_session()

    now.value = et(2026, 9, 14, 11)
    assert clock.next_session() == make_session(date(2026, 9, 14))
    now.value = et(2026, 9, 14, 17)
    assert clock.next_session() == make_session(date(2026, 9, 15))
    assert len(client.calendar_requests) == 1


def test_next_session_refetches_once_every_cached_session_closed() -> None:
    client = _client_with_days("2026-09-14")
    now = _Now(et(2026, 9, 14, 7))
    clock = AlpacaMarketClock(client, now=now)
    clock.next_session()

    client.calendar_response = [make_alpaca_calendar("2026-09-15")]
    now.value = et(2026, 9, 14, 17)
    assert clock.next_session() == make_session(date(2026, 9, 15))
    assert [r.start for r in client.calendar_requests] == [date(2026, 9, 14), date(2026, 9, 14)]


def test_next_session_raises_when_the_calendar_is_empty() -> None:
    clock = AlpacaMarketClock(_client_with_days(), now=_Now(et(2026, 9, 14, 7)))
    with pytest.raises(BrokerError, match="no session"):
        clock.next_session()


def test_next_session_wraps_client_errors() -> None:
    client = _client_with_days("2026-09-14")
    client.raises["get_calendar"] = make_api_error(500)
    clock = AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))
    with pytest.raises(BrokerError, match="calendar"):
        clock.next_session()


def test_default_now_uses_market_time_zone() -> None:
    clock = AlpacaMarketClock(FakeTradingClient())
    assert clock.now().tzinfo is MARKET_TZ

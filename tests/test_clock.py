from __future__ import annotations

import threading
import time as time_module
from datetime import UTC, date, datetime, timedelta

import pytest
from tests.fakes import ET, FakeClock, et, make_session, weekday_sessions

from trading_agent_framework.clock import MARKET_TZ, MarketClock, MarketSession


class _NoSessionClock(MarketClock):
    def next_session(self) -> MarketSession | None:
        return None


def test_market_session_requires_tz_aware_times() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        MarketSession(open=datetime(2026, 9, 14, 9, 30), close=et(2026, 9, 14, 16))


def test_market_session_requires_close_after_open() -> None:
    with pytest.raises(ValueError, match="after"):
        MarketSession(open=et(2026, 9, 14, 16), close=et(2026, 9, 14, 9, 30))


def test_default_now_is_aware_in_market_time_zone() -> None:
    now = _NoSessionClock().now()
    assert now.tzinfo is MARKET_TZ
    assert abs(now - datetime.now(UTC)) < timedelta(seconds=5)


def test_default_wait_returns_early_when_woken() -> None:
    wake = threading.Event()
    wake.set()
    started = time_module.monotonic()
    _NoSessionClock().wait(30, wake)
    assert time_module.monotonic() - started < 1


def test_fake_clock_wait_advances_time() -> None:
    clock = FakeClock(et(2026, 9, 14, 9, 0))
    clock.wait(90, threading.Event())
    assert clock.now() == et(2026, 9, 14, 9, 1, 30)
    assert clock.waits == [90]


def test_fake_clock_wait_does_not_advance_when_woken() -> None:
    clock = FakeClock(et(2026, 9, 14, 9, 0))
    wake = threading.Event()
    wake.set()
    clock.wait(90, wake)
    assert clock.now() == et(2026, 9, 14, 9, 0)


def test_fake_clock_next_session_skips_closed_sessions_then_ends() -> None:
    monday, tuesday = weekday_sessions(date(2026, 9, 14), 2)
    clock = FakeClock(et(2026, 9, 14, 16, 30), [monday, tuesday])
    assert clock.next_session() == tuesday
    clock.advance(timedelta(days=1).total_seconds())
    assert clock.next_session() is None


def test_fake_clock_returns_current_session_while_open() -> None:
    session = make_session(date(2026, 9, 14))
    clock = FakeClock(et(2026, 9, 14, 11, 0), [session])
    assert clock.next_session() == session


def test_weekday_sessions_skip_weekends() -> None:
    sessions = weekday_sessions(date(2026, 9, 18), 2)  # Friday
    assert [s.open.date() for s in sessions] == [date(2026, 9, 18), date(2026, 9, 21)]
    assert sessions[0].open.tzinfo is ET

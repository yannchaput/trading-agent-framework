from __future__ import annotations

import math
import threading
from datetime import date, time

from tests.fakes import et, make_session, weekday_sessions

from trading_agent_framework.backtesting.clock import BacktestClock


def test_max_wait_slice_is_infinite() -> None:
    clock = BacktestClock(start=et(2026, 1, 5), sessions=[])
    assert clock.max_wait_slice == math.inf


def test_now_starts_at_the_given_start_time() -> None:
    start = et(2026, 1, 5, 9, 30)
    clock = BacktestClock(start=start, sessions=[])
    assert clock.now() == start


def test_wait_advances_now_by_exactly_the_requested_seconds() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.wait(3600, threading.Event())
    assert clock.now() == et(2026, 1, 5, 10, 30)


def test_wait_does_nothing_when_seconds_is_not_positive() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.wait(0, threading.Event())
    assert clock.now() == et(2026, 1, 5, 9, 30)


def test_wait_does_nothing_when_wake_is_already_set() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    wake = threading.Event()
    wake.set()
    clock.wait(3600, wake)
    assert clock.now() == et(2026, 1, 5, 9, 30)


def test_wait_calls_on_advance_with_previous_and_new_now() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(
        start=et(2026, 1, 5, 9, 30), sessions=[], on_advance=lambda prev, new: calls.append((prev, new))
    )
    clock.wait(60, threading.Event())
    assert calls == [(et(2026, 1, 5, 9, 30), et(2026, 1, 5, 9, 31))]


def test_wait_does_not_call_on_advance_when_it_does_not_move_time() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(
        start=et(2026, 1, 5, 9, 30), sessions=[], on_advance=lambda prev, new: calls.append((prev, new))
    )
    wake = threading.Event()
    wake.set()
    clock.wait(60, wake)
    assert calls == []


def test_on_advance_can_be_set_after_construction() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.on_advance = lambda prev, new: calls.append((prev, new))
    clock.wait(1, threading.Event())
    assert len(calls) == 1


def test_next_session_returns_the_first_session_closing_after_now() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 3)
    clock = BacktestClock(start=et(2026, 1, 5, 12, 0), sessions=sessions)
    assert clock.next_session() == sessions[0]


def test_next_session_returns_none_past_the_last_session() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 1)
    clock = BacktestClock(start=et(2026, 1, 5, 20, 0), sessions=sessions)  # after the one session closed
    assert clock.next_session() is None


def test_next_session_skips_sessions_already_closed() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 3)
    clock = BacktestClock(start=sessions[0].close, sessions=sessions)
    assert clock.next_session() == sessions[1]


def test_next_session_respects_early_closes() -> None:
    early = make_session(date(2026, 11, 27), open_at=time(9, 30), close_at=time(13, 0))
    clock = BacktestClock(start=et(2026, 11, 27, 12, 0), sessions=[early])
    assert clock.next_session() == early

"""`BacktestBroker.get_bars(..., include_after_hours=False)` on minute bars: regular-session rows only (final review I4)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
from tests.fakes import FrameDataSource, et, make_session, minute_ohlc

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset

AAA = Asset("AAA")
SESSIONS = [make_session(date(2026, 9, 1)), make_session(date(2026, 9, 2))]
ROW = (10.0, 10.5, 9.5, 10.0, 100.0)


def _broker(now) -> BacktestBroker:
    # Close-stamped minute bars: day 1 15:56-16:05 (the last five after hours), day 2 09:26-09:35 (the first five pre-market;
    # the 09:30 bar is 09:29-09:30, still pre-market).
    frame = pd.concat([minute_ohlc(et(2026, 9, 1, 15, 56), [ROW] * 10), minute_ohlc(et(2026, 9, 2, 9, 26), [ROW] * 10)])
    frames = {("AAA", "minute"): frame, ("AAA", "day"): frame}
    clock = BacktestClock(start=now, sessions=SESSIONS)
    return BacktestBroker("vwap", data_source=FrameDataSource(frames, sessions=SESSIONS), clock=clock, budget=Decimal(1000), timestep="minute")


def _times(broker: BacktestBroker, length: int, timestep: str = "minute", **kwargs) -> list:
    return [ts.to_pydatetime() for ts in broker.get_bars([AAA], length, timestep, **kwargs)[AAA].df.index]


def test_regular_hours_minute_bars_skip_extended_hours_and_still_fill_the_length() -> None:
    broker = _broker(et(2026, 9, 2, 9, 35))
    expected = [et(2026, 9, 1, 15, 58), et(2026, 9, 1, 15, 59), et(2026, 9, 1, 16, 0)] + [et(2026, 9, 2, 9, m) for m in range(31, 36)]
    assert _times(broker, 8, include_after_hours=False) == expected


def test_extended_hours_are_kept_by_default_and_for_daily_bars() -> None:
    broker = _broker(et(2026, 9, 2, 9, 35))
    assert _times(broker, 8) == [et(2026, 9, 2, 9, m) for m in range(28, 36)]
    assert len(_times(broker, 8, "day", include_after_hours=False)) == 8  # daily bars are not filtered


def test_regular_hours_bars_never_reach_past_the_clock() -> None:
    broker = _broker(et(2026, 9, 2, 9, 32))
    assert _times(broker, 3, include_after_hours=False) == [et(2026, 9, 1, 16, 0), et(2026, 9, 2, 9, 31), et(2026, 9, 2, 9, 32)]


def test_regular_hours_returns_what_there_is_when_history_runs_out() -> None:
    broker = _broker(et(2026, 9, 2, 9, 35))
    assert len(_times(broker, 50, include_after_hours=False)) == 10

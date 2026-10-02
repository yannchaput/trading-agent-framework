"""The market regime in a backtest: `run_backtesting` widens the warmup so the regime exists from the first
session, and a session's value never sees a bar that closes after the refresh."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock

from trading_agent_framework.backtesting.warmup import warmup_calendar_days
from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession

ET = ZoneInfo("America/New_York")
SPY = Asset("SPY")
FEES = TradingFeeFactory(BrokerKind.ALPACA)


class IdleStrategy(Strategy):
    sleeptime = "1D"


def _sessions(first_day: date, count: int) -> list[MarketSession]:
    sessions: list[MarketSession] = []
    day = first_day
    while len(sessions) < count:
        if day.weekday() < 5:
            sessions.append(MarketSession(open=datetime.combine(day, time(9, 30), tzinfo=ET), close=datetime.combine(day, time(16, 0), tzinfo=ET)))
        day += timedelta(days=1)
    return sessions


def _bars(sessions: list[MarketSession], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"),
    )


def _strategy(tmp_path: Path, start: datetime) -> Strategy:
    return IdleStrategy(FakeBroker(FakeClock(start), "idle"), project_root=tmp_path)


@pytest.mark.parametrize(("requested", "expected"), [(0, 273), (10, 273), (400, 400)])
def test_run_backtesting_widens_the_warmup_to_the_regime_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: int, expected: int) -> None:
    captured: dict[str, object] = {}
    windows: list[tuple[datetime, datetime]] = []
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kwargs: captured.update(kwargs))
    start = datetime(2026, 1, 5, 8, 30, tzinfo=ET)
    end = datetime(2026, 1, 9, tzinfo=ET)
    source = FakeBacktestDataSource()

    _strategy(tmp_path, start).run_backtesting(
        start=start, end=end, fees=FEES, warmup_trading_days=requested,
        data_source=lambda window_start, window_end: windows.append((window_start, window_end)) or source,
    )

    assert captured["warmup_trading_days"] == expected
    assert captured["data_source"] is source
    assert windows == [(start - timedelta(days=warmup_calendar_days(expected)), end)]


def test_a_daily_backtest_has_a_regime_from_its_first_session_and_never_looks_ahead(tmp_path: Path) -> None:
    sessions = _sessions(date(2025, 1, 6), 303)  # 300 sessions of history, then a 3-session backtest
    closes = [100 * 1.001**i for i in range(303)]
    closes[-1] = 1.0  # the last session CLOSES in a crash: invisible to that morning's refresh
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(SPY, _bars(sessions, closes))
    start = sessions[300].open - timedelta(hours=1)
    end = sessions[302].close

    result = _strategy(tmp_path, start).run_backtesting(
        start=start, end=end, budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day", fees=FEES,
    )

    assert source.load_windows == [(start - timedelta(days=warmup_calendar_days(273)), end)]
    lines = pd.read_parquet(result.run_dir / "indicators.parquet")
    regime = lines[lines["name"] == "Regime"].sort_values("datetime")
    assert [float(value) for value in regime["value"]] == [1.0, 1.0, 1.0]
    assert set(regime["plot_name"]) == {"Regime"}
    stamps = [pd.Timestamp(stamp).tz_convert(ET) for stamp in pd.to_datetime(regime["datetime"], utc=True)]
    assert stamps == [pd.Timestamp(session.open - timedelta(minutes=60)) for session in sessions[300:]]

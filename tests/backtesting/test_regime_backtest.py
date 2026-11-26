"""The market regime in a backtest: `run_backtesting` keeps its daily warmup floor, and the regime (read from Yahoo
through a `RegimeSource`) exists from the first session without ever seeing a close of the session itself.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeRegimeSource

from trading_agent_framework.backtesting.warmup import warmup_calendar_days
from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
from trading_agent_framework.core.regime import RegimeParameters
from trading_agent_framework.core.regime_data import RegimeSeries
from trading_agent_framework.core.strategy import DAILY_WARMUP_FLOOR, Strategy
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


@pytest.mark.parametrize(("requested", "expected"), [(0, 283), (10, 283), (400, 400)])
def test_a_daily_run_widens_the_warmup_to_the_regime_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: int, expected: int) -> None:
    captured: dict[str, object] = {}
    windows: list[tuple[datetime, datetime]] = []
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kwargs: captured.update(kwargs))
    start = datetime(2026, 1, 5, 8, 30, tzinfo=ET)
    end = datetime(2026, 1, 9, tzinfo=ET)
    source = FakeBacktestDataSource()

    _strategy(tmp_path, start).run_backtesting(
        start=start,
        end=end,
        fees=FEES,
        warmup_trading_days=requested,
        timestep="day",
        data_source=lambda window_start, window_end: windows.append((window_start, window_end)) or source,
    )

    assert captured["warmup_trading_days"] == expected
    assert captured["timestep"] == "day"
    assert captured["data_source"] is source
    assert windows == [(start - timedelta(days=warmup_calendar_days(expected)), end)]
    assert source.load_calls == []  # run_backtest's own eager load covers the benchmark's daily frame


@pytest.mark.parametrize("requested", [0, 10, 400])
@pytest.mark.parametrize("as_instance", [False, True])
def test_a_minute_run_keeps_its_warmup_and_preloads_only_its_benchmarks_daily_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: int, as_instance: bool) -> None:
    captured: dict[str, object] = {}
    windows: list[tuple[datetime, datetime]] = []
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kwargs: captured.update(kwargs))
    start = datetime(2026, 1, 5, 8, 30, tzinfo=ET)
    end = datetime(2026, 1, 9, tzinfo=ET)
    source = FakeBacktestDataSource()
    data_source = source if as_instance else (lambda window_start, window_end: windows.append((window_start, window_end)) or source)

    # The runner reads the daily closes of the benchmark this run reports against, so that one is preloaded.
    _strategy(tmp_path, start).run_backtesting(
        start=start,
        end=end,
        fees=FEES,
        warmup_trading_days=requested,
        timestep="minute",
        benchmark="QQQ",
        data_source=data_source,
    )

    assert captured["warmup_trading_days"] == requested  # not raised for the regime
    assert captured["benchmark"] == "QQQ"
    assert captured["data_source"] is source
    if not as_instance:
        assert windows == [(start - timedelta(days=warmup_calendar_days(requested)), end)]  # constructor window not widened
    assert source.load_calls == [(Asset("QQQ"),)]
    assert source.load_timesteps == ["day"]
    assert source.load_windows == [(start - timedelta(days=warmup_calendar_days(DAILY_WARMUP_FLOOR)), end)]


def test_a_daily_backtest_has_a_regime_from_its_first_session_and_never_looks_ahead(tmp_path: Path) -> None:
    history = RegimeParameters().min_bars
    sessions = _sessions(date(2023, 1, 2), history + 3)  # 503 sessions of history, then a 3-session backtest
    closes = [100 * 1.001**i for i in range(len(sessions))]
    closes[-1] = 1.0  # the last session CLOSES in a crash: invisible to that morning's refresh
    regime_source = FakeRegimeSource(RegimeSeries(tuple(s.open.date() for s in sessions), tuple(closes), tuple([15.0] * len(sessions)), tuple([18.0] * len(sessions))))
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(SPY, _bars(sessions, closes))
    start = sessions[history].open - timedelta(hours=1)
    end = sessions[-1].close
    strategy = _strategy(tmp_path, start)
    strategy.regime_source = regime_source

    result = strategy.run_backtesting(
        start=start,
        end=end,
        budget=Decimal(10000),
        data_source=source,
        benchmark="SPY",
        timestep="day",
        fees=FEES,
    )

    assert source.load_windows == [(start - timedelta(days=warmup_calendar_days(DAILY_WARMUP_FLOOR)), end)]
    assert [call[0] for call in regime_source.calls] == [s.open.date() for s in sessions[history:]]
    lines = pd.read_parquet(result.run_dir / "indicators.parquet")
    regime = lines[lines["name"] == "Regime"].sort_values("datetime")
    assert [float(value) for value in regime["value"]] == [1.0, 1.0, 1.0]
    assert set(regime["plot_name"]) == {"Regime"}
    stamps = [pd.Timestamp(stamp).tz_convert(ET) for stamp in pd.to_datetime(regime["datetime"], utc=True)]
    assert stamps == [pd.Timestamp(session.open - timedelta(minutes=60)) for session in sessions[history:]]

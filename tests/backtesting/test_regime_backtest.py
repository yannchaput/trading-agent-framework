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
from trading_agent_framework.core.strategy import REGIME_WARMUP_SLACK, Strategy
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


# NYSE full-day closures 2024-2026 (Jan 9 2025: the national day of mourning for President Carter).
NYSE_HOLIDAYS = frozenset(
    date(*ymd)
    for ymd in [
        (2024, 1, 1), (2024, 1, 15), (2024, 2, 19), (2024, 3, 29), (2024, 5, 27), (2024, 6, 19), (2024, 7, 4), (2024, 9, 2), (2024, 11, 28), (2024, 12, 25),
        (2025, 1, 1), (2025, 1, 9), (2025, 1, 20), (2025, 2, 17), (2025, 4, 18), (2025, 5, 26), (2025, 6, 19), (2025, 7, 4), (2025, 9, 1), (2025, 11, 27), (2025, 12, 25),
        (2026, 1, 1), (2026, 1, 19), (2026, 2, 16), (2026, 4, 3), (2026, 5, 25), (2026, 6, 19), (2026, 7, 3), (2026, 9, 7), (2026, 11, 26), (2026, 12, 25),
    ]
)


def _sessions_between(first: date, stop: date) -> int:
    """Real NYSE sessions in `[first, stop)`: weekdays that are not a full-day holiday."""
    days = (first + timedelta(days=offset) for offset in range((stop - first).days))
    return sum(1 for day in days if day.weekday() < 5 and day not in NYSE_HOLIDAYS)


def test_the_regime_warmup_covers_min_bars_sessions_for_every_2025_start_date() -> None:
    """`warmup_calendar_days`' fixed holiday buffer under-covers a 13-month span: without the slack, 2025-07-07 got 270."""
    min_bars = Strategy.regime_params.min_bars
    window = warmup_calendar_days(min_bars + REGIME_WARMUP_SLACK)
    starts = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(365)]
    short = {
        start: count
        for start in starts
        if start.weekday() < 5 and start not in NYSE_HOLIDAYS and (count := _sessions_between(start - timedelta(days=window), start)) < min_bars
    }
    assert short == {}
    # The unslacked window does fall short somewhere (the reviewer's 2025-07-07 included), so the test has teeth.
    assert _sessions_between(date(2025, 7, 7) - timedelta(days=warmup_calendar_days(min_bars)), date(2025, 7, 7)) < min_bars


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
        start=start, end=end, fees=FEES, warmup_trading_days=requested, timestep="day",
        data_source=lambda window_start, window_end: windows.append((window_start, window_end)) or source,
    )

    assert captured["warmup_trading_days"] == expected
    assert captured["timestep"] == "day"
    assert captured["data_source"] is source
    assert windows == [(start - timedelta(days=warmup_calendar_days(expected)), end)]
    assert source.load_calls == []  # run_backtest's own eager load covers the benchmark's daily frame


@pytest.mark.parametrize("requested", [0, 10, 400])
@pytest.mark.parametrize("as_instance", [False, True])
def test_a_minute_run_keeps_its_warmup_and_preloads_only_the_benchmarks_daily_regime_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: int, as_instance: bool
) -> None:
    captured: dict[str, object] = {}
    windows: list[tuple[datetime, datetime]] = []
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kwargs: captured.update(kwargs))
    start = datetime(2026, 1, 5, 8, 30, tzinfo=ET)
    end = datetime(2026, 1, 9, tzinfo=ET)
    source = FakeBacktestDataSource()
    data_source = source if as_instance else (lambda window_start, window_end: windows.append((window_start, window_end)) or source)

    # The runner's benchmark argument stays as given; the regime reads `strategy.benchmark_symbol` (SPY).
    _strategy(tmp_path, start).run_backtesting(
        start=start, end=end, fees=FEES, warmup_trading_days=requested, timestep="minute", benchmark="QQQ", data_source=data_source,
    )

    assert captured["warmup_trading_days"] == requested  # not raised for the regime
    assert captured["benchmark"] == "QQQ"
    assert captured["data_source"] is source
    if not as_instance:
        assert windows == [(start - timedelta(days=warmup_calendar_days(requested)), end)]  # constructor window not widened
    assert source.load_calls == [(SPY,)]
    assert source.load_timesteps == ["day"]
    assert source.load_windows == [(start - timedelta(days=warmup_calendar_days(273 + REGIME_WARMUP_SLACK)), end)]


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

    assert source.load_windows == [(start - timedelta(days=warmup_calendar_days(273 + REGIME_WARMUP_SLACK)), end)]
    lines = pd.read_parquet(result.run_dir / "indicators.parquet")
    regime = lines[lines["name"] == "Regime"].sort_values("datetime")
    assert [float(value) for value in regime["value"]] == [1.0, 1.0, 1.0]
    assert set(regime["plot_name"]) == {"Regime"}
    stamps = [pd.Timestamp(stamp).tz_convert(ET) for stamp in pd.to_datetime(regime["datetime"], utc=True)]
    assert stamps == [pd.Timestamp(session.open - timedelta(minutes=60)) for session in sessions[300:]]

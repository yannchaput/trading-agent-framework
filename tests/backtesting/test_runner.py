"""`run_backtest` orchestration: builds the simulated clock/broker, runs the real
`StrategyExecutor` unmodified, writes the full report, and returns a `BacktestResult`.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock

from trading_agent_framework.backtesting.runner import BacktestResult, run_backtest
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession

ET = ZoneInfo("America/New_York")
AAPL = Asset("AAPL")
SPY = Asset("SPY")


def _sessions(first_day: date, count: int) -> list[MarketSession]:
    sessions: list[MarketSession] = []
    day = first_day
    while len(sessions) < count:
        if day.weekday() < 5:
            sessions.append(MarketSession(
                open=datetime.combine(day, time(9, 30), tzinfo=ET),
                close=datetime.combine(day, time(16, 0), tzinfo=ET),
            ))
        day += timedelta(days=1)
    return sessions


def _bars(sessions: list[MarketSession], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
         "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"),
    )


def _placeholder_strategy(strategy_cls: type[Strategy], tmp_path: Path, start: datetime) -> Strategy:
    """A `Strategy` built against a throwaway `FakeBroker`/`FakeClock` pair purely to
    satisfy `Strategy.__init__`'s required `broker` argument (it reads `broker.clock`
    when no `clock=` is given). `run_backtest` overwrites both before `executor.run()`
    is ever called, so neither the fake broker nor its clock is ever actually used to
    trade -- mirrors `tests/core/test_runners.py`'s use of `FakeBroker` as a placeholder.
    """
    return strategy_cls(FakeBroker(FakeClock(start), "buyonce"), project_root=tmp_path)


class BuyOnceStrategy(Strategy):
    sleeptime = "1D"

    def on_trading_iteration(self) -> None:
        if self.first_iteration:
            self.submit_order(self.create_order(AAPL, 5, "buy"))


def test_run_backtest_writes_every_expected_file(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 4)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0, 152.0, 153.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 402.0, 401.0, 405.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    assert isinstance(result, BacktestResult)
    for filename in ("settings.json", "metrics.json", "equity.parquet", "trades.parquet", "indicators.parquet"):
        assert (result.run_dir / filename).is_file()

    settings = json.loads((result.run_dir / "settings.json").read_text())
    assert settings["mode"] == "backtesting"
    assert settings["benchmark_symbol"] == "SPY"
    assert settings["backtesting_data_sources"] == "fake"

    equity = pd.read_parquet(result.run_dir / "equity.parquet")
    assert "benchmark_close" in equity.columns
    assert equity["benchmark_close"].notna().any()

    trades = pd.read_parquet(result.run_dir / "trades.parquet")
    assert len(trades) == 1  # the one order fills once

    assert "sharpe_strategy" in result.metrics


def test_run_backtest_rebinds_the_strategys_broker_and_clock(tmp_path: Path) -> None:
    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.backtesting.clock import BacktestClock

    sessions = _sessions(date(2026, 1, 5), 2)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, start)

    run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    assert isinstance(strategy.broker, BacktestBroker)
    assert isinstance(strategy.clock, BacktestClock)


class RecordingStrategy(Strategy):
    """Mirrors `test_no_look_ahead.py`'s guardian strategy: records what it observed
    on every iteration so the test can assert no future bar ever leaked through. Also
    buys once (like `test_no_look_ahead.py`'s `OrderPlacingStrategy`) purely so the
    portfolio's equity curve isn't perfectly flat -- a flat curve has zero variance,
    which makes `compute_metrics`'s benchmark-correlation math divide by zero (a
    pre-existing, harmless-but-noisy edge case in `metrics.py`'s `_relative_stats`,
    unrelated to what this test is checking) and is not representative of a real run."""

    sleeptime = "1D"

    def initialize(self) -> None:
        self.vars.observations = []

    def on_trading_iteration(self) -> None:
        now = self.clock.now()
        bars = self.get_historical_prices(AAPL, 10, "day")
        latest_bar_close = (
            bars.df.index[-1].to_pydatetime() if bars is not None and not bars.df.empty else None
        )
        self.vars.observations.append({"now": now, "latest_bar_close": latest_bar_close})
        if self.first_iteration:
            self.submit_order(self.create_order(AAPL, 5, "buy"))


def test_run_backtest_never_lets_the_strategy_observe_an_unclosed_bar(tmp_path: Path) -> None:
    """The exact no-look-ahead guarantee from Task 9's guardian test, exercised through
    `run_backtest`'s own clock/broker wiring rather than by hand -- this is what proves
    `run_backtest` reproduces that wiring correctly instead of trusting it by inspection.
    """
    sessions = _sessions(date(2026, 1, 5), 5)
    closes = [150.0, 151.0, 149.0, 152.0, 153.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, closes))
    source.set_bars(SPY, _bars(sessions, closes))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(RecordingStrategy, tmp_path, start)

    run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    observations = strategy.vars.observations
    assert len(observations) == 5  # one iteration per session (sleeptime="1D")
    for i, obs in enumerate(observations):
        if obs["latest_bar_close"] is None:
            assert i == 0  # only the very first session has no prior closed bar at all
            continue
        assert obs["latest_bar_close"] <= obs["now"]
        assert obs["latest_bar_close"] < sessions[i].close  # never its own, still-forming bar


def test_run_backtest_computes_returns_at_session_cadence_not_raw_sample_cadence(tmp_path: Path) -> None:
    """Critical review finding on Task 15: `_run()` used to build `portfolio_returns`
    from every raw `EquitySample` in `broker.ledger.equity` -- one sample per clock
    advance (pre-open/open/pre-close/close under `Strategy`'s DEFAULT timing), not one
    per trading session. `benchmark_returns` has always been one row per session (built
    from `data_source.bars`), so the mismatch diluted every annualised strategy metric
    (`cagr_strategy` understated ~3.5x in the reviewer's repro over 4 sessions/15
    samples) and misaligned every strategy-vs-benchmark stat.

    This test does NOT override `minutes_before_opening`/`minutes_before_closing`/
    `minutes_after_closing` -- `BuyOnceStrategy` uses `Strategy`'s defaults (60/1/0),
    which is what actually produces the oversampling; overriding them to 0 would hide
    the bug rather than exercise it.
    """
    sessions = _sessions(date(2026, 3, 2), 4)  # Mon-Thu, 4 consecutive trading days
    aapl_closes = [150.0, 151.0, 152.0, 154.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, aapl_closes))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0, 399.0, 403.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    broker = strategy.broker
    raw_samples = broker.ledger.equity
    # The bug's premise, verified rather than assumed: default timing genuinely
    # produces more raw equity samples than trading sessions (>= 1 per session for
    # pre-open/open/pre-close/close, vs. exactly `len(sessions)` sessions).
    assert len(raw_samples) > len(sessions)

    fills = broker.ledger.fills
    assert len(fills) == 1  # the one order fills once
    fill_time, fill_price = fills[0].time, fills[0].price
    cash_after_fill = Decimal(10000) - 5 * fill_price

    # Ground truth built from the broker's *actual recorded fill* (not by re-deriving
    # runner.py's own dedup algorithm): before the fill, equity is just the untouched
    # budget; from the fill's session onward, it's settled cash plus the 5-share
    # position revalued at that session's own closing price.
    expected_equity = [
        10000.0
        if session.close < fill_time
        else float(cash_after_fill + 5 * Decimal(str(close)))
        for session, close in zip(sessions, aapl_closes, strict=True)
    ]
    expected_returns = pd.Series(
        expected_equity, index=pd.DatetimeIndex([s.close for s in sessions])
    ).pct_change().dropna()
    assert len(expected_returns) == len(sessions) - 1  # one return per day-over-day session gap

    expected_total_return = float((1 + expected_returns).prod() - 1)
    expected_cagr = (1 + expected_total_return) ** (252 / len(expected_returns)) - 1

    assert result.metrics["total_return_strategy"] == pytest.approx(expected_total_return, rel=1e-9)
    assert result.metrics["cagr_strategy"] == pytest.approx(expected_cagr, rel=1e-6)

    # And explicitly not what the pre-fix code would have computed straight from the
    # raw, oversampled ledger samples -- proves this isn't accidentally the same number.
    old_index = [s.time for s in raw_samples]
    old_values = [float(s.portfolio_value) for s in raw_samples]
    old_returns = pd.Series(old_values, index=old_index).pct_change().dropna()
    old_total_return = float((1 + old_returns).prod() - 1)
    old_cagr = (1 + old_total_return) ** (252 / len(old_returns)) - 1
    assert result.metrics["cagr_strategy"] != pytest.approx(old_cagr, rel=1e-3)

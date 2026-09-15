"""`run_backtest` orchestration: builds the simulated clock/broker, runs the real
`StrategyExecutor` unmodified, writes the full report, and returns a `BacktestResult`.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock

from trading_agent_framework.backtesting.runner import BacktestResult, run_backtest
from trading_agent_framework.backtesting.warmup import warmup_calendar_days
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestError

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
    # Not `.notna().any()`: that passed at 6 non-null rows out of 23 while the column
    # was almost entirely broken (see the dedicated tests below).
    assert equity["benchmark_close"].notna().sum() == len(sessions)

    trades = pd.read_parquet(result.run_dir / "trades.parquet")
    assert len(trades) == 1  # the one order fills once

    assert "sharpe_strategy" in result.metrics


@pytest.mark.parametrize(
    ("naive_start", "naive_end", "expected"),
    [(True, True, "start, end"), (True, False, "start"), (False, True, "end")],
)
def test_run_backtest_rejects_naive_datetimes_with_a_clear_error(
    tmp_path: Path, naive_start: bool, naive_end: bool, expected: str
) -> None:
    """Important whole-branch review finding: the README's own quickstart used
    `datetime.now()`, and the only symptom was `BacktestError: backtest run failed:
    can't compare offset-naive and offset-aware datetimes` raised from deep inside
    `BacktestClock.next_session()` -- nothing naming `start`/`end` or saying what to do.
    `MarketSession` requires tz-aware datetimes, so this is validated up front.
    """
    sessions = _sessions(date(2026, 1, 5), 2)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0]))

    aware_start = sessions[0].open - timedelta(hours=1)
    aware_end = sessions[-1].close
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, aware_start)

    with pytest.raises(BacktestError, match="timezone-aware") as excinfo:
        run_backtest(
            strategy,
            start=aware_start.replace(tzinfo=None) if naive_start else aware_start,
            end=aware_end.replace(tzinfo=None) if naive_end else aware_end,
            budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
            commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
        )

    # The message names the offending argument(s) -- the whole point of validating here.
    assert expected in str(excinfo.value)
    # ...and it is the up-front check, not the old obscure downstream comparison.
    assert "offset-naive" not in str(excinfo.value)
    assert not source.load_calls  # rejected before any data was even requested


def test_run_backtest_accepts_aware_datetimes_in_any_timezone(tmp_path: Path) -> None:
    """The validation is "aware", not "in the market's timezone" -- a UTC bound (what
    `smoke_backtest.py`-style code and most callers produce) must still work."""
    sessions = _sessions(date(2026, 1, 5), 2)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0]))

    start = (sessions[0].open - timedelta(hours=1)).astimezone(UTC)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close.astimezone(UTC),
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )
    assert (result.run_dir / "metrics.json").is_file()


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


def _equity_parquet_for(tmp_path: Path, strategy_cls: type[Strategy], session_count: int = 6):
    """Run a real `run_backtest` over `session_count` sessions and return
    `(equity_dataframe, sessions)`."""
    sessions = _sessions(date(2026, 3, 2), session_count)
    closes = [150.0 + i for i in range(session_count)]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, closes))
    source.set_bars(SPY, _bars(sessions, [400.0 + (i % 3) for i in range(session_count)]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(strategy_cls, tmp_path, start)
    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )
    # Premise of the whole finding, verified rather than assumed: the raw ledger really
    # is oversampled relative to sessions, so a session-cadence parquet is a real
    # reduction and not an accident of this fixture.
    assert len(strategy.broker.ledger.equity) > len(sessions)
    return pd.read_parquet(result.run_dir / "equity.parquet"), sessions


def _assert_session_cadence_equity(equity: pd.DataFrame, sessions: list[MarketSession]) -> None:
    # One row per trading session -- NOT one per clock advance.
    assert len(equity) == len(sessions)
    assert list(equity.index) == [pd.Timestamp(s.close) for s in sessions]
    # Every session's row carries a benchmark close, and every session-over-session
    # gap therefore yields a real benchmark return (only the first row's pct_change
    # is legitimately NaN).
    assert equity["benchmark_close"].notna().sum() == len(sessions)
    assert equity["benchmark_return"].notna().sum() == len(sessions) - 1
    assert equity["return"].notna().sum() == len(sessions) - 1


def test_run_backtest_writes_a_session_cadence_equity_parquet_with_a_fully_joined_benchmark(
    tmp_path: Path,
) -> None:
    """Critical whole-branch review finding. `equity.parquet` never received the
    session-cadence fix `metrics.json`'s `portfolio_returns` got: `write_equity` was
    handed the RAW ledger (every clock advance) and joined the benchmark by exact
    timestamp equality against those raw instants. Reviewer's repro over 6 sessions,
    default timing: 23 rows, `benchmark_close` non-null on 6 of them (the ones that
    happened to land exactly on 16:00), `benchmark_return` non-null on NONE (isolated
    non-null values separated by holes -> pct_change is NaN everywhere).

    Uses `Strategy`'s DEFAULT timing deliberately -- that is what produces the
    oversampling; zeroing the minutes_* knobs would hide the bug.
    """
    equity, sessions = _equity_parquet_for(tmp_path, BuyOnceStrategy)
    _assert_session_cadence_equity(equity, sessions)


class LateCloseBuyOnceStrategy(BuyOnceStrategy):
    """Same as `BuyOnceStrategy`, but with a non-default, publicly-overridable
    `minutes_after_closing` -- the exact knob round 2 of this review finding is about.
    """

    minutes_after_closing = 5


def test_run_backtest_session_cadence_survives_nonzero_minutes_after_closing(tmp_path: Path) -> None:
    """Round 2 of the Critical review finding on Task 15. Round 1's fix
    (`_session_equity_series`) bucketed equity samples with a `sample.time <=
    session.close` cutoff. `StrategyExecutor._run_session` makes a final
    `wait_until(session.close + minutes_after_closing)` call per session -- the one
    that produces that session's TRUE, fully-settled post-close equity sample -- and
    when `minutes_after_closing > 0` that sample's `time` is `> session.close`, so the
    old cutoff wrongly bucketed it into the *next* session instead: every session's
    recorded value lagged by one, and the last session's post-close sample was dropped
    off the end entirely (no session exists after the last one to catch it).

    This reuses the exact same 4-session/1-fill fixture as
    `test_run_backtest_computes_returns_at_session_cadence_not_raw_sample_cadence`,
    with `minutes_after_closing=5` instead of the default 0. The state at each
    session's close (cash + position revalued at that session's own closing price)
    does not depend on `minutes_after_closing` -- nothing trades between `close` and
    `close + minutes_after_closing` -- so the CORRECT expected equity/returns are
    identical to that default-timing test's. Only a buggy, shifted bucketing rule
    would produce a different number here, which is exactly what round 1's fix did.
    """
    sessions = _sessions(date(2026, 3, 2), 4)  # Mon-Thu, 4 consecutive trading days
    aapl_closes = [150.0, 151.0, 152.0, 154.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, aapl_closes))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0, 399.0, 403.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(LateCloseBuyOnceStrategy, tmp_path, start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    broker = strategy.broker
    raw_samples = broker.ledger.equity
    assert len(raw_samples) > len(sessions)  # oversampling still occurs, as before

    fills = broker.ledger.fills
    assert len(fills) == 1
    fill_time, fill_price = fills[0].time, fills[0].price
    cash_after_fill = Decimal(10000) - 5 * fill_price

    # Same ground-truth formula as the default-timing test: correct per-session
    # equity, built from the broker's actual recorded fill, does not depend on
    # `minutes_after_closing`.
    expected_equity = [
        10000.0
        if session.close < fill_time
        else float(cash_after_fill + 5 * Decimal(str(close)))
        for session, close in zip(sessions, aapl_closes, strict=True)
    ]
    assert expected_equity == [10000.0, 10000.0, pytest.approx(10005.0), pytest.approx(10015.0)]

    expected_returns = pd.Series(
        expected_equity, index=pd.DatetimeIndex([s.close for s in sessions])
    ).pct_change().dropna()
    expected_total_return = float((1 + expected_returns).prod() - 1)
    expected_cagr = (1 + expected_total_return) ** (252 / len(expected_returns)) - 1

    assert result.metrics["total_return_strategy"] == pytest.approx(expected_total_return, rel=1e-9)
    assert result.metrics["cagr_strategy"] == pytest.approx(expected_cagr, rel=1e-6)

    # And explicitly not what the round-1-only fix (cutoff at `session.close`, not
    # `session[i+1].open`) would have computed -- proves the round-2 fix actually
    # changed the number for this configuration, not that a plausible number happened
    # to come out.
    round1_index: list[datetime] = []
    round1_values: list[float] = []
    sample_index = 0
    last_value: float | None = None
    for session in sessions:
        while sample_index < len(raw_samples) and raw_samples[sample_index].time <= session.close:
            last_value = float(raw_samples[sample_index].portfolio_value)
            sample_index += 1
        if last_value is not None:
            round1_index.append(session.close)
            round1_values.append(last_value)
    round1_returns = pd.Series(round1_values, index=pd.DatetimeIndex(round1_index)).pct_change().dropna()
    round1_total_return = float((1 + round1_returns).prod() - 1)
    round1_cagr = (1 + round1_total_return) ** (252 / len(round1_returns)) - 1
    assert result.metrics["cagr_strategy"] != pytest.approx(round1_cagr, rel=1e-3)


def test_equity_parquet_session_cadence_survives_nonzero_minutes_after_closing(
    tmp_path: Path,
) -> None:
    """`equity.parquet`'s half of the round-2 `minutes_after_closing` failure mode,
    mirroring `test_run_backtest_session_cadence_survives_nonzero_minutes_after_closing`
    for `metrics.json`. With `minutes_after_closing=5` no clock advance lands exactly on
    16:00 any more, so the old raw-ledger exact-timestamp benchmark join matched
    *nothing*: `benchmark_close` was null on ALL 23 rows in the reviewer's repro (down
    from a lucky 6 under default timing). The session-close re-stamping makes the join
    independent of that knob entirely.
    """
    equity, sessions = _equity_parquet_for(tmp_path, LateCloseBuyOnceStrategy)
    _assert_session_cadence_equity(equity, sessions)


def test_run_backtest_widens_only_the_eager_benchmark_load_by_warmup(tmp_path: Path) -> None:
    """`warmup_trading_days` widens the eager benchmark `load()` call's start bound by
    exactly `warmup_calendar_days`'s output, so warm-up history reaches the benchmark
    asset even though it is only lazily fetched (via `get_historical_prices`) for other
    tickers. The end bound and the exact call count are untouched.
    """
    sessions = _sessions(date(2026, 1, 5), 4)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0, 152.0, 153.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 402.0, 401.0, 405.0]))

    start = sessions[0].open - timedelta(hours=1)
    end = sessions[-1].close
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path, start)

    run_backtest(
        strategy, start=start, end=end,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
        warmup_trading_days=10,
    )

    expected_start = start - timedelta(days=warmup_calendar_days(10))
    assert source.load_windows == [(expected_start, end)]


def test_run_backtest_warmup_never_becomes_an_extra_simulated_session(tmp_path: Path) -> None:
    """Widening the eager benchmark load must never widen `sessions()`/the clock: the
    number of simulated session rows in `equity.parquet` is identical whether or not
    `warmup_trading_days` is passed.
    """

    def _run(warmup_trading_days: int, subdir: str) -> int:
        sessions = _sessions(date(2026, 1, 5), 4)
        source = FakeBacktestDataSource()
        source.set_sessions(sessions)
        source.set_bars(AAPL, _bars(sessions, [150.0, 151.0, 152.0, 153.0]))
        source.set_bars(SPY, _bars(sessions, [400.0, 402.0, 401.0, 405.0]))

        start = sessions[0].open - timedelta(hours=1)
        strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path / subdir, start)

        result = run_backtest(
            strategy, start=start, end=sessions[-1].close,
            budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
            commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
            warmup_trading_days=warmup_trading_days,
        )
        return len(pd.read_parquet(result.run_dir / "equity.parquet"))

    assert _run(0, "no_warmup") == _run(10, "with_warmup")


def test_run_backtest_records_warmup_trading_days_in_settings(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 2)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path / "explicit", start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
        warmup_trading_days=10,
    )
    settings = json.loads((result.run_dir / "settings.json").read_text())
    assert settings["warmup_trading_days"] == 10


def test_run_backtest_defaults_warmup_trading_days_to_zero_in_settings(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 2)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 401.0]))

    start = sessions[0].open - timedelta(hours=1)
    strategy = _placeholder_strategy(BuyOnceStrategy, tmp_path / "default", start)

    result = run_backtest(
        strategy, start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )
    settings = json.loads((result.run_dir / "settings.json").read_text())
    assert settings["warmup_trading_days"] == 0

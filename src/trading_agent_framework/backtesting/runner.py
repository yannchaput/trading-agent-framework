"""Orchestrates one backtest run end to end: builds the simulated clock/broker, runs
the strategy through the (unmodified) executor, computes metrics, and writes the
report. The only module that wires `Strategy` to the backtesting subsystem --
`Strategy.run_backtesting()` (a later task) is a thin wrapper around `run_backtest`.

Clock-driving invariant (carried forward from Task 8's review of
`BacktestBroker._process_pending`, which only reasons about the single latest bar per
`on_advance` call): this module drives nothing itself -- it hands the *real*,
unmodified `StrategyExecutor` the simulated clock/broker and lets it run its normal
session loop. That loop only ever crosses a bar close one at a time for
`timestep="day"`, because `StrategyExecutor._run_sessions` processes trading sessions
strictly one at a time (`_next_session()` returns the single next session, and its
full lifecycle -- including the `wait_until(session.close + minutes_after_closing)`
call where the day's bar actually closes -- runs to completion before the next
session is even looked up). `tests/backtesting/test_no_look_ahead.py` (Task 9) and
this module's own `test_run_backtest_never_lets_the_strategy_observe_an_unclosed_bar`
verify that directly. This is NOT verified for `timestep="minute"` combined with a
`sleeptime` coarser than one minute (e.g. a session-based sleeptime, or a multi-minute
interval): such a combination lets a single `clock.wait()` jump span more than one
minute-bar close, which `_process_pending` cannot see past the latest of. Fixing that
would mean changing `BacktestBroker`, not this orchestration module -- until then,
keep `sleeptime` at least as fine-grained as `timestep` when using minute bars.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework import __version__
from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.backtesting.data.base import FULL_HISTORY
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BacktestError
from trading_agent_framework.utils.log import setup_strategy_logging

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.data.base import BacktestDataSource
    from trading_agent_framework.core.strategy import Strategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """The outcome of one `run_backtest` call."""

    run_dir: Path
    settings: dict[str, Any]
    metrics: dict[str, Any]


def run_backtest(
    strategy: Strategy,
    *,
    start: datetime,
    end: datetime,
    budget: Decimal,
    data_source: BacktestDataSource,
    benchmark: str,
    timestep: str,
    commission: Decimal,
    slippage: Decimal,
    risk_free_rate: float,
) -> BacktestResult:
    """Run `strategy` through a full simulated `[start, end]` backtest.

    Builds a `BacktestClock`/`BacktestBroker` pair wired exactly as Task 9's
    guardian test proved is no-look-ahead-safe (`clock.on_advance = broker.on_advance`
    set after both are constructed, breaking their circular dependency), rebinds
    `strategy.broker`/`strategy.clock` to them, then runs the real, unmodified
    `StrategyExecutor` through its normal lifecycle -- this function contains no
    backtest-only trading logic of its own. The benchmark series is fetched once,
    right after the run, and handed to `report.write_equity` so Alpha/Beta/
    correlation stay reproducible without a later live network call.

    `data_source` must already be constructed with a window covering `[start,
    end]` (Task 12's review finding): this function always calls
    `data_source.load(...)`/`.sessions(...)` with the exact `start`/`end` given
    here, but a source whose own fetch window is fixed narrower at its own
    construction (e.g. `AlpacaBacktestData`, `YahooBacktestData`) will silently
    return an incomplete calendar/bars near the edges -- sources expose no public
    window accessor, so that mismatch cannot be detected from here. Construct
    `data_source` with the same `start`/`end` passed to this function to avoid it.

    Never raises a raw exception: anything other than an existing `BacktestError`
    is wrapped in one.
    """
    try:
        return _run(
            strategy,
            start=start,
            end=end,
            budget=budget,
            data_source=data_source,
            benchmark=benchmark,
            timestep=timestep,
            commission=commission,
            slippage=slippage,
            risk_free_rate=risk_free_rate,
        )
    except BacktestError:
        raise
    except Exception as exc:
        raise BacktestError(f"backtest run failed: {exc}") from exc


def _run(
    strategy: Strategy,
    *,
    start: datetime,
    end: datetime,
    budget: Decimal,
    data_source: BacktestDataSource,
    benchmark: str,
    timestep: str,
    commission: Decimal,
    slippage: Decimal,
    risk_free_rate: float,
) -> BacktestResult:
    import pandas as pd

    from trading_agent_framework.backtesting import metrics as metrics_module
    from trading_agent_framework.backtesting import report

    # Captured before any rebinding below: `strategy.name` reads `strategy.broker
    # .strategy_name`, and the placeholder broker the caller constructed the
    # strategy with exists only to carry that name (see docstring/tests).
    name = strategy.name

    benchmark_asset = Asset(benchmark)
    data_source.load([benchmark_asset], start, end, timestep)
    sessions = data_source.sessions(start, end)

    clock = BacktestClock(start=start, sessions=sessions)
    broker = BacktestBroker(
        name, data_source=data_source, clock=clock, budget=budget,
        timestep=timestep, commission=commission, slippage=slippage,
    )
    clock.on_advance = broker.on_advance

    strategy.trading_mode = TradingMode.BACKTESTING
    strategy.broker = broker
    strategy.clock = clock

    log_file = setup_strategy_logging(name, TradingMode.BACKTESTING, project_root=strategy.project_root)
    run_dir = log_file.parent

    started = time.monotonic()
    strategy.executor.run()
    elapsed = time.monotonic() - started

    benchmark_bars = data_source.bars(benchmark_asset, end, FULL_HISTORY, timestep)
    benchmark_series = (
        pd.Series(benchmark_bars.df["close"].to_numpy(), index=benchmark_bars.df.index)
        if benchmark_bars is not None
        else None
    )
    benchmark_by_time = (
        {ts: Decimal(str(v)) for ts, v in benchmark_series.items()} if benchmark_series is not None else None
    )

    report.write_equity(run_dir, broker.ledger, benchmark_by_time)
    report.write_trades(run_dir, broker.ledger)
    report.write_indicators(run_dir, broker.ledger)

    equity_index = [sample.time for sample in broker.ledger.equity]
    equity_values = [float(sample.portfolio_value) for sample in broker.ledger.equity]
    portfolio_returns = pd.Series(equity_values, index=equity_index).pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna() if benchmark_series is not None else None

    computed_metrics = metrics_module.compute_metrics(
        portfolio_returns, benchmark_returns, timestep=timestep, risk_free_rate=risk_free_rate
    )
    report.write_metrics(run_dir, computed_metrics)

    settings = {
        "name": name,
        "mode": TradingMode.BACKTESTING.value,
        "run_ts": run_dir.name,
        "backtesting_start": start.isoformat(),
        "backtesting_end": end.isoformat(),
        "budget": float(budget),
        "risk_free_rate": risk_free_rate,
        "backtesting_data_sources": data_source.name,
        "backtest_time_seconds": elapsed,
        "timestep": timestep,
        "sleeptime": strategy.sleeptime,
        "commission": float(commission),
        "slippage": float(slippage),
        "benchmark_symbol": benchmark,
        "framework_version": __version__,
        "parameters": dict(strategy.parameters),
    }
    report.write_settings(run_dir, settings)

    return BacktestResult(run_dir=run_dir, settings=settings, metrics=computed_metrics)

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

import dataclasses
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from trading_agent_framework import __version__
from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.backtesting.data.base import FULL_HISTORY
from trading_agent_framework.backtesting.warmup import warmup_calendar_days
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BacktestError
from trading_agent_framework.utils.log import setup_strategy_logging

if TYPE_CHECKING:
    import pandas as pd

    from trading_agent_framework.backtesting.data.base import BacktestDataSource
    from trading_agent_framework.backtesting.ledger import EquitySample
    from trading_agent_framework.brokers.news import NewsProvider
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.utils.clock import MarketSession


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
    warmup_trading_days: int = 0,
    preload_assets: Sequence[Asset] = (),
    news_source: NewsProvider | None = None,
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

    `start` and `end` must be timezone-aware. Every `MarketSession` this subsystem
    produces is tz-aware (`MarketSession.__post_init__` enforces it), so a naive
    `start` blows up deep inside `BacktestClock.next_session()`'s
    `session.close > self._now` comparison with a bare "can't compare offset-naive and
    offset-aware datetimes" -- a message that says nothing about which argument was at
    fault. Validated up front instead (Important whole-branch review finding: the
    README's own quickstart example used `datetime.now()` and hit exactly that).

    `warmup_trading_days` (default `0`, preserving every existing caller) widens ONLY
    the one eager `data_source.load(...)` call's start bound, by
    `warmup.warmup_calendar_days(warmup_trading_days)` calendar days, so warm-up
    history for the benchmark asset (and any `preload_assets`) is available to a
    strategy's indicators from the very first simulated session (see `_run`'s
    docstring for why the benchmark specifically needs this). It never affects
    `sessions()`/the `BacktestClock`: the simulated `[start, end]` window a strategy
    actually trades through is unchanged, so warm-up history never becomes an extra
    simulated trading session.

    `preload_assets` are batched into that same eager `load(...)` call alongside the
    benchmark, so a strategy with a large universe (e.g. cross_momentum) gets one
    combined fetch instead of a separate preload plus the runner's own benchmark
    load. Pass the benchmark asset in `preload_assets` too if it's part of your
    universe and you want it fetched only once -- `YahooBacktestData.load()` skips
    already-cached assets, but the ABC contract doesn't require every source to.

    `news_source` is handed to the `BacktestBroker` (default: an Alpaca provider built lazily from the env credentials).

    Never raises a raw exception: anything other than an existing `BacktestError`
    is wrapped in one -- except `warmup_trading_days` itself, which is validated
    up front (see below) so a bad value always raises the same `ValueError`
    regardless of which `data_source` the caller passed in.
    """
    _require_aware(start=start, end=end)
    # Computed once, here, before the try/except below: `warmup_calendar_days`
    # raises `ValueError` for a negative `warmup_trading_days`, and that must
    # propagate as-is rather than get wrapped into a `BacktestError` by the
    # catch-all below -- a caller reaching this through the default data source
    # never entered a try/except at all, so wrapping only when an explicit
    # `data_source` happens to be passed would make the same bad input raise two
    # different exception types depending on an unrelated caller choice. The
    # result is threaded into `_run` so it isn't recomputed there.
    warmup_days = warmup_calendar_days(warmup_trading_days)
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
            warmup_trading_days=warmup_trading_days,
            warmup_days=warmup_days,
            preload_assets=preload_assets,
            news_source=news_source,
        )
    except BacktestError:
        raise
    except Exception as exc:
        raise BacktestError(f"backtest run failed: {exc}") from exc


def _require_aware(**moments: datetime) -> None:
    """Reject naive `start`/`end` with a message that names the offending argument and
    shows the fix, rather than letting a downstream comparison fail obscurely."""
    naive = [name for name, value in moments.items() if value.tzinfo is None or value.tzinfo.utcoffset(value) is None]
    if naive:
        raise BacktestError(
            f"run_backtest needs timezone-aware datetimes; {', '.join(naive)} "
            f"{'is' if len(naive) == 1 else 'are'} naive. Trading sessions carry a "
            "market timezone, so a naive bound cannot be compared against them. Use "
            'e.g. `from zoneinfo import ZoneInfo; ET = ZoneInfo("America/New_York"); '
            "datetime.now(ET)`."
        )


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
    warmup_trading_days: int = 0,
    warmup_days: int = 0,
    preload_assets: Sequence[Asset] = (),
    news_source: NewsProvider | None = None,
) -> BacktestResult:
    """The unvalidated body of `run_backtest` -- see that function's docstring for the
    full contract, including `warmup_trading_days`'s scope: it widens only the eager
    `data_source.load(...)` call below (benchmark plus `preload_assets`), never
    `sessions()`/the clock.

    `warmup_days` is `warmup.warmup_calendar_days(warmup_trading_days)`, already
    computed (and validated) once by `run_backtest` -- `warmup_trading_days` itself
    is only carried through here to be recorded verbatim in `settings.json`.
    """
    import pandas as pd

    from trading_agent_framework.backtesting import metrics as metrics_module
    from trading_agent_framework.backtesting import report

    # Captured before any rebinding below: `strategy.name` reads `strategy.broker
    # .strategy_name`, and the placeholder broker the caller constructed the
    # strategy with exists only to carry that name (see docstring/tests).
    name = strategy.name

    # Configured before the eager data_source.load() below (not after
    # strategy/broker/clock wiring, where it used to sit): a WARNING logged during
    # that load -- e.g. YahooBacktestData reporting a ticker with no data -- would
    # otherwise fire before any handler is attached to the package logger, fall
    # through to `logging.lastResort`'s raw, unformatted stderr dump, and never
    # reach the run's log file at all.
    log_file = setup_strategy_logging(name, TradingMode.BACKTESTING, project_root=strategy.project_root)
    run_dir = log_file.parent

    benchmark_asset = Asset(benchmark)
    # Widened ONLY for this one eager load() call -- see warmup.py and the
    # module docstring above for why the benchmark specifically needs this
    # (CrossMomentumStrategy's benchmark is "SPY", which it also reads directly
    # for its risk overlay). `preload_assets` rides along in the same batched
    # call rather than a separate one the caller would otherwise have to make.
    warmup_start = start - timedelta(days=warmup_days)
    assets_to_load = [benchmark_asset, *(a for a in preload_assets if a != benchmark_asset)]
    data_source.load(assets_to_load, warmup_start, end, timestep)
    # sessions()/BacktestClock stay on the exact, narrow [start, end] -- warm-up
    # history must never become an extra simulated trading session.
    sessions = data_source.sessions(start, end)

    clock = BacktestClock(start=start, sessions=sessions)
    broker = BacktestBroker(
        name,
        data_source=data_source,
        clock=clock,
        budget=budget,
        timestep=timestep,
        commission=commission,
        slippage=slippage,
        news_source=news_source,
    )
    clock.on_advance = broker.on_advance

    strategy.trading_mode = TradingMode.BACKTESTING
    strategy.broker = broker
    strategy.clock = clock

    started = time.monotonic()
    strategy.executor.run()
    elapsed = time.monotonic() - started

    benchmark_bars = data_source.bars(benchmark_asset, end, FULL_HISTORY, timestep)
    benchmark_series = pd.Series(benchmark_bars.df["close"].to_numpy(), index=benchmark_bars.df.index) if benchmark_bars is not None else None
    benchmark_by_time: dict[datetime, Decimal] | None = {cast(datetime, ts): Decimal(str(v)) for ts, v in benchmark_series.items()} if benchmark_series is not None else None

    # ONE session-reduced equity series, shared by equity.parquet and metrics.json.
    # They used to be built from different things -- metrics from this reduction,
    # equity.parquet straight from the raw (oversampled) ledger -- which made the
    # dashboard's plotted return series disagree with the metrics computed beside it
    # and broke equity.parquet's benchmark join outright (see `report.write_equity`).
    session_equity_samples = _session_equity_samples(broker.ledger.equity, sessions)

    report.write_equity(run_dir, session_equity_samples, benchmark_by_time)
    report.write_trades(run_dir, broker.ledger)
    report.write_indicators(run_dir, broker.ledger)

    session_equity = _session_equity_series(session_equity_samples)
    portfolio_returns = session_equity.pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna() if benchmark_series is not None else None

    if benchmark_returns is not None:
        # `portfolio_returns` and `benchmark_returns` are both meant to be one row per
        # trading session at this point (see `_session_equity_samples`'s docstring), but
        # they come from two independent sources (the ledger vs. `data_source.bars`) --
        # a silent index mismatch here would misalign `compute_metrics`'s alpha/beta/
        # correlation/information-ratio math without either series looking obviously
        # wrong on its own. Align explicitly so any such mismatch degrades to "fewer
        # overlapping rows" instead of "wrong rows paired together".
        portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join="inner")

    computed_metrics = metrics_module.compute_metrics(portfolio_returns, benchmark_returns, timestep=timestep, risk_free_rate=risk_free_rate)
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
        "warmup_trading_days": warmup_trading_days,
        "benchmark_symbol": benchmark,
        "framework_version": __version__,
        "parameters": dict(strategy.parameters),
    }
    report.write_settings(run_dir, settings)

    return BacktestResult(run_dir=run_dir, settings=settings, metrics=computed_metrics)


def _session_equity_series(session_samples: Sequence[EquitySample]) -> pd.Series:
    """The `portfolio_value` curve of `_session_equity_samples`'s output, as a float
    `pd.Series` indexed by session close -- the shape `compute_metrics` wants.

    Kept deliberately thin: the reduction itself lives in `_session_equity_samples` so
    that `report.write_equity` and `compute_metrics` consume the *same* rows rather
    than two independently-derived series (the defect this split fixes).
    """
    import pandas as pd

    return pd.Series(
        [float(s.portfolio_value) for s in session_samples],
        index=pd.DatetimeIndex([s.time for s in session_samples], name="timestamp"),
    )


def _session_equity_samples(equity_samples: list[EquitySample], sessions: list[MarketSession]) -> list[EquitySample]:
    """Reduce `equity_samples` down to exactly one sample per trading session.

    Each surviving sample is re-stamped with its session's `close`, so the result is
    indexed by *session identity*: that is what lets `report.write_equity` join the
    (also one-row-per-session, bar-close-indexed) benchmark series by timestamp, and
    what puts `equity.parquet`'s `return` column on the same cadence as
    `metrics.json`'s `portfolio_returns`.

    `BacktestBroker.on_advance` samples equity on every clock advance, not just at
    session boundaries: under `Strategy`'s default timing (`minutes_before_opening=60`,
    `minutes_before_closing=1`, `minutes_after_closing=0`), `StrategyExecutor` makes
    several distinct `wait_until()` calls per session (pre-open, session-open,
    pre-close, close), each producing its own `EquitySample`. Feeding all of those into
    `portfolio_returns` (as an earlier version of this module did) oversamples the
    strategy side relative to `benchmark_returns` -- which is built from
    `data_source.bars(...)` and is naturally one row per session -- and silently
    understates every annualised metric `compute_metrics` derives from `portfolio_
    returns` (Critical review finding on Task 15: `cagr_strategy` off by ~3.5x in the
    reviewer's repro).

    A session's bucket is every remaining sample with `time` strictly before the
    NEXT session's `open` (not a cutoff at this session's own `close`): the final
    `wait_until(session.close + minutes_after_closing)` `StrategyExecutor` makes per
    session (`_run_session` in `core/executor.py`) produces that session's true
    post-close equity sample, and when `minutes_after_closing > 0` that sample's
    `time` is `> session.close` -- so a `time <= session.close` cutoff would wrongly
    shove it into the *next* session's bucket, lagging every session's recorded value
    by one and dropping the last session's post-close sample off the end entirely
    (round 2 of this same review finding: `cagr_strategy` silently wrong for any
    non-default `minutes_after_closing`, with no error). Using the next session's
    `open` as the boundary is a safe, slightly-conservative choice: the executor's own
    pre-open wait for session i+1 lands at `session[i+1].open - minutes_before_opening`,
    which is always earlier than `session[i+1].open` itself, so nothing belonging to
    session i+1 can have a `time` before that boundary. The LAST session has no next
    session to bound it, so its bucket is simply everything not yet claimed -- with no
    upper bound -- which correctly captures its post-close sample regardless of
    `minutes_after_closing`.

    Within each bucket, keeps the LAST sample: the fully-settled end-of-session
    equity, reflecting both that session's own fills and its bar's freshly-closed
    price (earlier same-session samples still see the *prior* session's closing price
    -- see `BacktestBroker._positions_value`/`_latest_bar`). `equity_samples` and
    `sessions` are both expected in chronological order, which is an existing
    invariant of `Ledger.equity` (append-only, written by a clock that only ever
    advances) and `BacktestDataSource.sessions()`.
    """
    reduced: list[EquitySample] = []
    sample_index = 0
    last_sample: EquitySample | None = None
    session_count = len(sessions)
    for i, session in enumerate(sessions):
        if i + 1 < session_count:
            boundary = sessions[i + 1].open
            while sample_index < len(equity_samples) and equity_samples[sample_index].time < boundary:
                last_sample = equity_samples[sample_index]
                sample_index += 1
        else:
            # Last session: no next session to bound it -- claim every remaining
            # sample, however late, so its true post-close equity isn't dropped.
            while sample_index < len(equity_samples):
                last_sample = equity_samples[sample_index]
                sample_index += 1
        if last_sample is not None:
            reduced.append(dataclasses.replace(last_sample, time=session.close))
    return reduced

# cross_momentum rebalance timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cross_momentum's Tuesday decision is the same whatever time it runs: it reads completed sessions only,
its risk overlay aligns series by date, it sizes orders at the last trade, and it runs at a fixed market time
(12:00 ET) set in `parameters.py`.

**Architecture:** A new optional `Strategy.iteration_start_time` moves each session's first
`on_trading_iteration` in the executor (default `None` = the open, unchanged for every other strategy).
cross_momentum sets it from `parameters["rebalance_time"]`, drops today's (partial) daily bar before every
computation, passes date-indexed close Series to a rewritten `_build_aligned_returns` that inner-joins on dates,
and fetches a last-trade price per target for sizing.

**Tech Stack:** Python 3.14, pandas, numpy, pytest, `uv`, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-10-06-cross-momentum-rebalance-timing-design.md`

## Global Constraints

- Run everything through `uv` (`uv run pytest`, `uv run ruff check`, `uv run pyright`); never pip.
- The automated tests never touch the network; use hand-written fakes (`tests/fakes.py`, the per-file
  `FakeStrategy` classes), not `MagicMock`.
- No `time.sleep` / `datetime.now` in strategy or executor code: time comes from `strategy.clock` /
  `self.get_datetime()`.
- Market time is `MARKET_TZ` (`trading_agent_framework.utils.clock`, America/New_York).
- Default behaviour of every other strategy is unchanged: `iteration_start_time = None` must reproduce today's
  executor exactly (all existing `tests/core/test_executor.py` tests stay green unmodified).
- `parameters.py`: `"rebalance_time": "12:00"` (ET, `HH:MM`).
- Out of scope: the momentum score, filter/overlay/breadth thresholds, `risk_diagnostics.py`, other strategies,
  a once-per-week rebalance guard.
- Commit after every task; each commit message ends with the `Co-Authored-By:` trailer of the model executing
  the task. Line length and style follow the surrounding code (ruff config in `pyproject.toml`).

## Review Focus

- A restart after the close on a Tuesday (bot down 12:00-15:59): no iteration, no rebalance that week, no crash
  (existing executor rule; Task 1 test `test_a_start_after_the_closing_window_runs_no_iteration`).
- An early-close session (13:00) with `iteration_start_time = 12:00`: the iteration still runs at 12:00 (Task 1
  test `test_an_early_close_session_still_iterates_at_the_start_time`).
- A daily frame with two bars on the same market date (e.g. a data source restamping a bar): `close_series`
  keeps one value per date instead of making `pd.concat` raise on a duplicate index (Task 2 test
  `test_close_series_keeps_one_value_per_date`).
- Target series that share no date with SPY (stale or broken data): the overlay returns NORMAL with
  `observations == 0`, never raises (Task 2 test `test_series_sharing_no_date_give_normal_with_zero_observations`).
- A `rebalance_time` given with a UTC offset (`"12:00+02:00"`) or as prose (`"noon"`): rejected at
  construction with `ConfigurationError` (Task 5 tests).

---

## File map

| File | Change |
|---|---|
| `src/trading_agent_framework/core/strategy.py` | `iteration_start_time` class attribute |
| `src/trading_agent_framework/core/executor.py` | `_trade` starts at `_iteration_start(session)` |
| `src/trading_agent_framework/strategies/cross_momentum/portfolio_risk_overlay.py` | date-joined `_build_aligned_returns`, Series interface, `observations` metric |
| `src/trading_agent_framework/strategies/cross_momentum/utils.py` | `close_series`, `completed_bars`, `parse_rebalance_time` |
| `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` | `_risk_exposure` (extracted), completed bars in the scan, live sizing prices, `iteration_start_time` from parameters |
| `src/trading_agent_framework/strategies/cross_momentum/parameters.py` | `"rebalance_time": "12:00"` |
| `tests/core/test_executor.py` | start-time tests |
| `tests/strategies/test_cross_momentum_risk_overlay.py` | new: overlay alignment, `close_series`, `_risk_exposure` |
| `tests/strategies/test_cross_momentum_indicators.py` | completed bars in the scan |
| `tests/strategies/test_cross_momentum_breadth.py` | `_indicator` helper gains `close_series` |
| `tests/strategies/test_cross_momentum_rebalance.py` | sizing at the last trade |
| `tests/strategies/test_cross_momentum_timing.py` | new: `completed_bars`, `parse_rebalance_time`, strategy wiring |
| `CLAUDE.md` | executor note, cross_momentum gotcha |

---

### Task 1: Executor `iteration_start_time`

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py` (imports line 12, timing attributes lines 84-90)
- Modify: `src/trading_agent_framework/core/executor.py` (imports lines 19-26, `_trade` lines 175-199)
- Test: `tests/core/test_executor.py`

**Interfaces:**
- Produces: `Strategy.iteration_start_time: time | None = None` (market time). `StrategyExecutor._iteration_start(session: MarketSession) -> datetime`.

- [ ] **Step 1: Write the failing tests**

In `tests/core/test_executor.py`, change the import line `from datetime import date, datetime, timedelta` to
`from datetime import date, datetime, time, timedelta`, then append:

```python
# --- iteration_start_time ---------------------------------------------------------------


class Noon(Recorder):
    sleeptime = "1D"
    iteration_start_time = time(12, 0)


def test_iteration_start_time_moves_the_iteration_and_leaves_the_other_hooks_alone() -> None:
    strategy = _run(Noon, sessions=2)
    assert strategy.times("on_trading_iteration") == [et(2026, 9, 14, 12), et(2026, 9, 15, 12)]
    assert strategy.times("before_market_opens") == [et(2026, 9, 14, 8, 30), et(2026, 9, 15, 8, 30)]
    assert strategy.times("before_starting_trading") == [et(2026, 9, 14, 9, 30), et(2026, 9, 15, 9, 30)]
    assert strategy.times("before_market_closes") == [et(2026, 9, 14, 15, 59), et(2026, 9, 15, 15, 59)]


def test_a_start_before_the_iteration_start_time_waits_for_it() -> None:
    assert _run(Noon, start=et(2026, 9, 14, 10, 15)).times("on_trading_iteration") == [et(2026, 9, 14, 12)]


def test_a_start_after_the_iteration_start_time_iterates_at_once() -> None:
    # 2026-10-06: the bot restarted at 15:36 ET on a rebalance day; it must still run that session.
    assert _run(Noon, start=et(2026, 9, 14, 15, 36)).times("on_trading_iteration") == [et(2026, 9, 14, 15, 36)]


def test_a_start_after_the_closing_window_runs_no_iteration() -> None:
    strategy = _run(Noon, start=et(2026, 9, 14, 16, 30), sessions=1)
    assert strategy.times("on_trading_iteration") == []


def test_an_iteration_start_time_before_the_open_changes_nothing() -> None:
    class Early(Recorder):
        sleeptime = "1D"
        iteration_start_time = time(8, 0)

    assert _run(Early).times("on_trading_iteration") == [et(2026, 9, 14, 9, 30)]


def test_an_interval_sleeptime_starts_its_grid_at_the_iteration_start_time() -> None:
    class FromNoon(Recorder):
        sleeptime = "2H"
        iteration_start_time = time(12, 0)

    assert _run(FromNoon).times("on_trading_iteration") == [et(2026, 9, 14, 12), et(2026, 9, 14, 14)]


def test_an_early_close_session_still_iterates_at_the_start_time() -> None:
    session = replace(weekday_sessions(MONDAY, 1)[0], close=et(2026, 9, 14, 13))
    strategy = Noon(FakeBroker(FakeClock(et(2026, 9, 14, 7), [session])))
    strategy.executor.run()
    assert strategy.times("on_trading_iteration") == [et(2026, 9, 14, 12)]


def test_an_iteration_start_time_in_the_closing_window_skips_the_iterations(caplog: pytest.LogCaptureFixture) -> None:
    class Late(Recorder):
        sleeptime = "1D"
        iteration_start_time = time(15, 59)

    caplog.set_level(logging.INFO)
    strategy = _run(Late)
    assert strategy.times("on_trading_iteration") == []
    assert strategy.times("before_market_closes") == [et(2026, 9, 14, 15, 59)]
    assert any("iteration_start_time" in record.getMessage() for record in caplog.records)
```

(`replace`, `FakeBroker`, `FakeClock`, `weekday_sessions`, `MONDAY`, `logging` and `pytest` are already imported
at the top of this file. `MarketSession` is a frozen dataclass, so `dataclasses.replace` works on it.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_executor.py -k "iteration_start_time or start_time or closing_window or early_close" -v`
Expected: the tests asserting 12:00 / 14:00 iterations FAIL (iterations land at 09:30 / 10:15), the
closing-window test FAILS (an iteration at 09:30); `test_a_start_after_the_iteration_start_time_iterates_at_once`,
`test_a_start_after_the_closing_window_runs_no_iteration` and
`test_an_iteration_start_time_before_the_open_changes_nothing` PASS already (existing behaviour, kept as
guards).

- [ ] **Step 3: Implement**

`src/trading_agent_framework/core/strategy.py`: change `from datetime import datetime, timedelta` to
`from datetime import datetime, time, timedelta`, and below `minutes_after_closing: int = 0` add:

```python
    # First on_trading_iteration() of each session at this market time (America/New_York); None = at the open.
    # The other hooks keep their times; a start at or after the closing window skips that session's iterations.
    iteration_start_time: time | None = None
```

`src/trading_agent_framework/core/executor.py`: change `from trading_agent_framework.utils.clock import MarketSession`
to `from trading_agent_framework.utils.clock import MARKET_TZ, MarketSession`. Replace the head of `_trade`:

```python
    def _trade(self, session: MarketSession) -> None:
        strategy = self.strategy
        stop_at = session.close - timedelta(minutes=strategy.minutes_before_closing)
        tick = max(session.open, self._now())
```

with:

```python
    def _trade(self, session: MarketSession) -> None:
        strategy = self.strategy
        stop_at = session.close - timedelta(minutes=strategy.minutes_before_closing)
        start = self._iteration_start(session)
        if start >= stop_at:
            logger.info(
                "No iteration for %s this session: iteration_start_time %s is not before %s",
                strategy.name,
                start.isoformat(),
                stop_at.isoformat(),
            )
            self.wait_until(stop_at)
            return
        tick = max(start, self._now())
```

and add this method right after `_trade`:

```python
    def _iteration_start(self, session: MarketSession) -> datetime:
        """The session's first iteration time: the open, or `iteration_start_time` (market time) when later."""
        start_time = self.strategy.iteration_start_time
        if start_time is None:
            return session.open
        session_date = session.open.astimezone(MARKET_TZ).date()
        return max(session.open, datetime.combine(session_date, start_time, tzinfo=MARKET_TZ))
```

In the module docstring, change `on_trading_iteration every \`sleeptime\`` to
`on_trading_iteration every \`sleeptime\` (from \`iteration_start_time\` when set)`.

- [ ] **Step 4: Run the executor tests**

Run: `uv run pytest tests/core/ -q`
Expected: all PASS (new and existing).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py src/trading_agent_framework/core/executor.py tests/core/test_executor.py
git commit -m "Task 1: Strategy.iteration_start_time moves each session's first iteration"
```

---

### Task 2: Risk overlay aligned by session date

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/portfolio_risk_overlay.py` (`_build_aligned_returns` lines 78-146, `compute_risk_overlay` lines 243-317)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (new `close_series`)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`_compute_indicators_for_ticker` return dict, `compute_target_portfolio` line 330, `on_trading_iteration` Step 3 lines 578-603, new `_risk_exposure`)
- Create: `tests/strategies/test_cross_momentum_risk_overlay.py`
- Modify: `tests/strategies/test_cross_momentum_breadth.py` (`_indicator` helper)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `close_series(df: pd.DataFrame) -> pd.Series` in `strategies/cross_momentum/utils.py`: closes indexed by
    `datetime.date` (market time), one value per date (last kept), oldest first.
  - `compute_risk_overlay(closes_map: dict[str, pd.Series], target_weights: dict[str, float], benchmark_closes: pd.Series | None, min_obs: int = 40) -> tuple[str, float, dict]`;
    the metrics dict gains `"observations": int`.
  - `CrossMomentumStrategy._risk_exposure(self, target: list[dict]) -> float`.
  - Indicator entries gain `"close_series": pd.Series`; `self.vars.target_closes: dict[str, pd.Series]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_cross_momentum_risk_overlay.py`:

```python
"""cross_momentum's portfolio risk overlay aligns series by session date, never by position.

On 2026-10-06 about half of the target series ended a session before SPY's; aligned from the end by position,
every return was paired with the wrong day, and the overlay read NORMAL (beta 1.35) instead of CRITICAL (2.8).
"""

from datetime import datetime, time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
from trading_agent_framework.strategies.cross_momentum.portfolio_risk_overlay import (
    RiskState,
    classify_risk_state,
    compute_risk_overlay,
)
from trading_agent_framework.strategies.cross_momentum.utils import close_series
from trading_agent_framework.utils.clock import MARKET_TZ

DATES = [d.date() for d in pd.bdate_range("2026-01-02", periods=120)]


def _market(n_stocks: int = 10, beta: float = 2.5, seed: int = 0) -> tuple[dict[str, pd.Series], pd.Series]:
    """Stocks moving `beta` times SPY plus noise, one close per date in DATES."""
    rng = np.random.default_rng(seed)
    spy_returns = rng.normal(0.0, 0.01, len(DATES) - 1)
    spy = pd.Series(100 * np.cumprod(np.r_[1.0, 1 + spy_returns]), index=DATES)
    stocks = {}
    for i in range(n_stocks):
        returns = beta * spy_returns + rng.normal(0.0, 0.01, len(spy_returns))
        stocks[f"S{i}"] = pd.Series(50 * np.cumprod(np.r_[1.0, 1 + returns]), index=DATES)
    return stocks, spy


def _equal(stocks: dict[str, pd.Series]) -> dict[str, float]:
    return {symbol: 1 / len(stocks) for symbol in stocks}


def test_series_one_day_short_are_aligned_by_date_not_by_position():
    stocks, spy = _market()
    short = {symbol: (series.iloc[:-1] if i % 2 else series) for i, (symbol, series) in enumerate(stocks.items())}

    _, _, full = compute_risk_overlay(stocks, _equal(stocks), spy)
    state, exposure, metrics = compute_risk_overlay(short, _equal(short), spy)

    assert metrics["beta_63d"] == pytest.approx(full["beta_63d"], abs=0.05)
    assert metrics["corr_20d"] == pytest.approx(full["corr_20d"], abs=0.1)
    assert (state, exposure) == ("critical", 0.4)


def test_complete_series_give_one_return_per_date():
    stocks, spy = _market()

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), spy)

    assert metrics["observations"] == len(DATES) - 1


def test_a_date_missing_from_one_series_is_dropped_for_all():
    stocks, spy = _market()
    stocks["S0"] = stocks["S0"].drop(DATES[60])

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), spy)

    assert metrics["observations"] == len(DATES) - 2


def test_fewer_than_min_obs_aligned_returns_is_normal():
    stocks, spy = _market()
    recent = {symbol: series.iloc[-30:] for symbol, series in stocks.items()}

    state, exposure, metrics = compute_risk_overlay(recent, _equal(recent), spy)

    assert (state, exposure) == ("normal", 1.0)
    assert metrics["beta_63d"] is None
    assert metrics["observations"] == 29


def test_series_sharing_no_date_give_normal_with_zero_observations():
    stocks, spy = _market()
    stale = {symbol: series.iloc[:50] for symbol, series in stocks.items()}

    state, exposure, metrics = compute_risk_overlay(stale, _equal(stale), spy.iloc[60:])

    assert (state, exposure) == ("normal", 1.0)
    assert metrics["observations"] == 0


def test_without_a_benchmark_vol_and_corr_are_still_computed():
    stocks, _ = _market()

    _, _, metrics = compute_risk_overlay(stocks, _equal(stocks), None)

    assert metrics["beta_63d"] is None
    assert metrics["vol_20d"] > 0
    assert metrics["corr_20d"] > 0.5


@pytest.mark.parametrize(
    ("beta", "vol", "corr", "expected"),
    [
        (2.0, None, None, RiskState.CRITICAL),
        (1.99, None, None, RiskState.ELEVATED),
        (1.7, None, None, RiskState.ELEVATED),
        (1.69, None, None, RiskState.NORMAL),
        (None, 0.35, 0.35, RiskState.CRITICAL),
        (None, 0.30, 0.30, RiskState.ELEVATED),
        (None, 0.29, 0.35, RiskState.NORMAL),
    ],
)
def test_the_classification_thresholds_are_unchanged(beta, vol, corr, expected):
    assert classify_risk_state(beta, vol, corr) is expected


# --- close_series -----------------------------------------------------------------------


def _frame(dates, closes, hour: int) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime.combine(d, time(hour), tzinfo=MARKET_TZ) for d in dates])
    return pd.DataFrame({"close": closes}, index=index)


@pytest.mark.parametrize("hour", [0, 16])  # live Alpaca bars are stamped at midnight, backtest bars at the close
def test_close_series_is_indexed_by_market_date(hour):
    series = close_series(_frame(DATES[:3], [1.0, 2.0, 3.0], hour))

    assert list(series.index) == DATES[:3]
    assert list(series) == [1.0, 2.0, 3.0]


def test_close_series_keeps_one_value_per_date():
    frame = pd.concat([_frame(DATES[:2], [1.0, 2.0], 0), _frame(DATES[1:2], [2.5], 16)]).sort_index()

    series = close_series(frame)

    assert list(series.index) == DATES[:2]
    assert list(series) == [1.0, 2.5]


# --- CrossMomentumStrategy._risk_exposure ---------------------------------------------------


class FakeStrategy:
    """Just enough of `Strategy` for `_risk_exposure`."""

    def __init__(self, target_closes, spy):
        self.vars = SimpleNamespace(target_closes=target_closes, alpaca_rate_limiter=SimpleNamespace(wait=lambda: None))
        self._spy = spy
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def get_historical_prices(self, ticker, length, timestep):
        if self._spy is None:
            return None
        return Bars(Asset("SPY"), "day", _frame(list(self._spy.index), list(self._spy), 0))

    def log_info(self, message, *args, **kwargs):
        self.infos.append(message)

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)


def _target(stocks):
    return [{"symbol": symbol, "target_weight": 1 / len(stocks)} for symbol in stocks]


def test_risk_exposure_aligns_by_date_and_names_the_series_a_session_short():
    stocks, spy = _market()
    short = {symbol: (series.iloc[:-1] if i % 2 else series) for i, (symbol, series) in enumerate(stocks.items())}
    fake = FakeStrategy(short, spy)

    exposure = CrossMomentumStrategy._risk_exposure(fake, _target(short))

    assert exposure == 0.4
    assert len(fake.warnings) == 1
    assert "S1" in fake.warnings[0] and "S0" not in fake.warnings[0]
    assert any("CRITICAL" in line and "obs=" in line for line in fake.infos)


def test_risk_exposure_without_spy_data_is_neutral():
    stocks, _ = _market()
    fake = FakeStrategy(stocks, None)

    assert CrossMomentumStrategy._risk_exposure(fake, _target(stocks)) == 1.0
    assert any("SPY" in message for message in fake.warnings)


def test_risk_exposure_without_targets_is_neutral():
    fake = FakeStrategy({}, None)

    assert CrossMomentumStrategy._risk_exposure(fake, []) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_risk_overlay.py -v`
Expected: collection ERROR (`ImportError: cannot import name 'close_series'`). After Step 3's `close_series`
alone, the alignment test still FAILS (positional beta about half), the `observations` tests FAIL with
`KeyError: 'observations'`, and the `_risk_exposure` tests FAIL with `AttributeError`.

- [ ] **Step 3: Implement `close_series`**

In `src/trading_agent_framework/strategies/cross_momentum/utils.py`, add to the imports
`from trading_agent_framework.utils.clock import MARKET_TZ`, and add after `compute_atr_from_df`:

```python
def close_series(df: pd.DataFrame) -> pd.Series:
    """Closes indexed by session date in market time, one per date (the last), oldest first.

    Live Alpaca daily bars are stamped at midnight market time and backtest bars at the close; both map to the
    same date, so series from either source can be joined on it.
    """
    dates = pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ).date
    series = pd.Series(df["close"].to_numpy(dtype=float), index=pd.Index(dates))
    return series[~series.index.duplicated(keep="last")].sort_index()
```

- [ ] **Step 4: Implement the date-aligned overlay**

In `portfolio_risk_overlay.py`, replace `_build_aligned_returns` entirely with:

```python
_BENCHMARK_COLUMN = "__benchmark__"  # never a ticker


def _build_aligned_returns(
    closes_map: dict[str, pd.Series],
    weights_dict: dict[str, float],
    benchmark_closes: pd.Series | None,
) -> tuple[pd.DataFrame | None, pd.Series | None, np.ndarray | None]:
    """Build daily returns aligned by session date.

    The target series (and the benchmark's, if any) are inner-joined on their dates and rows with a gap are
    dropped BEFORE computing simple daily returns, so every return spans the same two sessions for every symbol
    and for the benchmark. A date missing from one series is dropped for all. Weights are renormalized over the
    symbols that made it into the returns frame.

    Args:
        closes_map: {symbol: closes indexed by session date} for each target stock.
        weights_dict: {symbol: target_weight} — raw weights (may not sum to 1).
        benchmark_closes: Benchmark (SPY) closes indexed by session date, or None.

    Returns:
        (returns_df, benchmark_returns, aligned_weights), all indexed by the later session's date; all three are
        None when fewer than two common dates remain.
    """
    if not closes_map:
        return None, None, None

    columns = dict(closes_map)
    if benchmark_closes is not None:
        columns[_BENCHMARK_COLUMN] = benchmark_closes
    prices = pd.concat(columns, axis=1, join="inner").sort_index().dropna()
    if len(prices) < 2:
        return None, None, None

    returns = prices.pct_change().iloc[1:]
    benchmark_returns = returns.pop(_BENCHMARK_COLUMN) if benchmark_closes is not None else None
    if returns.shape[1] < 1:
        return None, None, None

    cols = returns.columns.tolist()
    raw = np.array([weights_dict.get(sym, 0.0) for sym in cols], dtype=float)
    total = np.sum(raw)
    if total <= 0:
        return None, None, None
    return returns, benchmark_returns, raw / total
```

In `compute_risk_overlay`: change the annotations to `closes_map: dict[str, pd.Series]` and
`benchmark_closes: pd.Series | None`; in the docstring replace "price series" / "List of SPY close prices"
with "closes indexed by session date" / "SPY closes indexed by session date"; then make the metrics carry the
number of aligned returns. Right after the `_build_aligned_returns(...)` call add:

```python
    observations = 0 if returns_df is None else len(returns_df)
```

add `"observations": observations,` to the early-return metrics dict (the `min_obs` gate) and to the final
`metrics` dict. Nothing else in the module changes (`_compute_portfolio_beta` already joins on the index,
which is now dates).

- [ ] **Step 5: Wire the strategy**

In `agent_cross_momentum.py`, add `close_series` to the `from .utils import (...)` list. In
`_compute_indicators_for_ticker`, add to the returned dict (after `"closes": closes,`):

```python
            "close_series": close_series(df),
```

In `compute_target_portfolio`, change

```python
        self.vars.target_closes = {entry["symbol"]: entry["closes"] for entry in selected}
```

to

```python
        self.vars.target_closes = {entry["symbol"]: entry["close_series"] for entry in selected}
```

Add this method after `_breadth_exposure`:

```python
    def _risk_exposure(self, target: list[dict]) -> float:
        """Exposure multiplier from the portfolio risk overlay: beta/vol/corr of the target stocks against SPY.

        1.0 without targets or SPY data. The overlay aligns the series by session date; a target whose last bar
        is dated differently from SPY's is still logged, because on 2026-10-06 half of them were a session short
        for a reason never established.
        """
        if not target or not self.vars.target_closes:
            return 1.0
        self.vars.alpaca_rate_limiter.wait()
        spy_bars = self.get_historical_prices("SPY", length=300, timestep="day")
        if spy_bars is None or spy_bars.empty:
            self.log_warning("Risk overlay: SPY data unavailable — defaulting to NORMAL (100% exposure)")
            return 1.0
        spy = close_series(spy_bars.pandas_df)
        spy_last = spy.index[-1]
        off_date = {symbol: series.index[-1] for symbol, series in self.vars.target_closes.items() if len(series) and series.index[-1] != spy_last}
        if off_date:
            listed = ", ".join(f"{symbol} {day}" for symbol, day in sorted(off_date.items()))
            self.log_warning(f"Risk overlay: last bar date differs from SPY's {spy_last}: {listed}")
        target_weights = {entry["symbol"]: entry["target_weight"] for entry in target}
        risk_state, risk_exposure, metrics = compute_risk_overlay(self.vars.target_closes, target_weights, spy)
        self.log_info(
            f"Risk overlay: {risk_state.upper()} (beta={metrics.get('beta_63d')}, vol={metrics.get('vol_20d')}, corr={metrics.get('corr_20d')}, obs={metrics.get('observations')}, exposure={risk_exposure:.0%})"
        )
        return risk_exposure
```

In `on_trading_iteration`, replace the whole Step 3 block (from the `# Step 3: Portfolio Risk Overlay` comment
through the `self.log_info(f"Risk overlay: ...")` call, i.e. the lines defining `risk_exposure = 1.0`,
`risk_metrics = {}` and the `if target and self.vars.target_closes:` block) with:

```python
        # Step 3: Portfolio Risk Overlay — beta/vol/corr of today's target stocks against SPY, aligned by date.
        risk_exposure = self._risk_exposure(target)
```

In `tests/strategies/test_cross_momentum_breadth.py`, the `_indicator` helper must carry the new key: add
`"close_series": pd.Series(closes),` after `"closes": closes,`, and add `import pandas as pd` after
`import pytest` (the file does not import pandas yet).

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/strategies/ -q -k cross_momentum`
Expected: all PASS.

- [ ] **Step 7: Lint and type-check the touched files**

Run: `uv run ruff check src/trading_agent_framework/strategies/cross_momentum tests/strategies && uv run pyright src/trading_agent_framework/strategies/cross_momentum`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum tests/strategies/test_cross_momentum_risk_overlay.py tests/strategies/test_cross_momentum_breadth.py
git commit -m "Task 2: cross_momentum risk overlay aligns series by session date"
```

---

### Task 3: Completed sessions only

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (new `completed_bars`)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`_compute_indicators_for_ticker`, `_risk_exposure`, new `_market_date`, module constant)
- Create: `tests/strategies/test_cross_momentum_timing.py`
- Modify: `tests/strategies/test_cross_momentum_indicators.py`
- Modify: `tests/strategies/test_cross_momentum_risk_overlay.py` (the `_risk_exposure` fake)

**Interfaces:**
- Consumes: `close_series`, `_risk_exposure` (Task 2).
- Produces: `completed_bars(df: pd.DataFrame, today: date) -> pd.DataFrame`; `CrossMomentumStrategy._market_date(self) -> date`; module constant `_HISTORY_BARS = 300`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_cross_momentum_timing.py`:

```python
"""cross_momentum decides on completed sessions only: a bar dated today is partial while the session is open."""

from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent_framework.strategies.cross_momentum.utils import completed_bars
from trading_agent_framework.utils.clock import MARKET_TZ

MONDAY, TUESDAY = date(2026, 10, 5), date(2026, 10, 6)


def _frame(dates, hour: int) -> pd.DataFrame:
    index = pd.DatetimeIndex([datetime.combine(d, time(hour), tzinfo=MARKET_TZ) for d in dates])
    return pd.DataFrame({"close": [float(i) for i in range(len(dates))]}, index=index)


@pytest.mark.parametrize("hour", [0, 16])  # live Alpaca bars are stamped at midnight, backtest bars at the close
def test_completed_bars_drops_the_bar_dated_today(hour):
    df = completed_bars(_frame([date(2026, 10, 2), MONDAY, TUESDAY], hour), TUESDAY)

    assert [ts.date() for ts in df.index] == [date(2026, 10, 2), MONDAY]


def test_completed_bars_without_a_bar_for_today_changes_nothing():
    frame = _frame([date(2026, 10, 2), MONDAY], 16)

    pd.testing.assert_frame_equal(completed_bars(frame, TUESDAY), frame)
```

In `tests/strategies/test_cross_momentum_indicators.py`, add the imports
`from datetime import date, datetime, time`, `import pandas as pd`,
`from trading_agent_framework.entities.asset import Asset`,
`from trading_agent_framework.entities.bars import Bars`,
`from trading_agent_framework.utils.clock import MARKET_TZ`; extend `FakeStrategy` so it records the requested
length and answers `get_datetime`:

```python
    def __init__(self, *, bars=None, raises=None, now=None):
        self.parameters = {"min_trading_days": 250, "skip_days": 21, "volatility_window": 20}
        self.vars = SimpleNamespace(alpaca_rate_limiter=SimpleNamespace(wait=lambda: None))
        self._bars = bars
        self._raises = raises
        self._now = now or datetime(2026, 10, 6, 12, 0, tzinfo=MARKET_TZ)
        self.requested_lengths: list[int] = []
        self.warnings: list[str] = []

    def get_historical_prices(self, ticker, length, timestep):
        self.requested_lengths.append(length)
        if self._raises is not None:
            raise self._raises
        return self._bars

    def get_datetime(self):
        return self._now

    _market_date = CrossMomentumStrategy._market_date
```

and append:

```python
def _daily_bars(sessions: int, last: date) -> Bars:
    """`sessions` midnight-stamped daily bars ending on `last`; the last one is a -50% crash."""
    dates = [d.date() for d in pd.bdate_range(end=last, periods=sessions)]
    closes = [100.0 + i for i in range(sessions)]
    closes[-1] = closes[-2] * 0.5
    index = pd.DatetimeIndex([datetime.combine(d, time(0), tzinfo=MARKET_TZ) for d in dates])
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1e6] * sessions}, index=index)
    return Bars(Asset("AAA"), "day", frame)


def test_the_scan_ignores_todays_partial_bar():
    bars = _daily_bars(301, date(2026, 10, 6))  # 300 completed sessions + Tuesday's crash, still forming
    fake = FakeStrategy(bars=bars)

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "AAA")

    assert fake.requested_lengths == [301]
    assert result is not None
    assert result["trading_days"] == 300
    assert result["price"] == bars.df["close"].iloc[-2]  # Monday's close, not the crash
    assert result["close_series"].index[-1] == date(2026, 10, 5)


def test_the_scan_keeps_300_bars_when_today_has_no_bar_yet():
    fake = FakeStrategy(bars=_daily_bars(301, date(2026, 10, 5)))  # a bar per session through Monday

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "AAA")

    assert result is not None
    assert result["trading_days"] == 300
    assert result["close_series"].index[-1] == date(2026, 10, 5)
```

In `tests/strategies/test_cross_momentum_risk_overlay.py`, give the `_risk_exposure` fake a clock and the
helper, and add a test that SPY's partial bar is dropped. Add `timedelta` to the `datetime` import
(`from datetime import datetime, time, timedelta`). Change `FakeStrategy.__init__`'s signature to
`def __init__(self, target_closes, spy, now=None):` and add as its last line:

```python
        # Default: the day after the last date in DATES, so no test bar counts as today's partial bar.
        self._now = now or datetime.combine(DATES[-1] + timedelta(days=1), time(12), tzinfo=MARKET_TZ)
```

Add to the class body:

```python
    _market_date = CrossMomentumStrategy._market_date

    def get_datetime(self):
        return self._now
```

then append:

```python
def test_risk_exposure_drops_spys_partial_bar_for_today():
    stocks, spy = _market()
    completed = {symbol: series.iloc[:-1] for symbol, series in stocks.items()}  # every target through yesterday
    today_noon = datetime.combine(DATES[-1], time(12), tzinfo=MARKET_TZ)
    fake = FakeStrategy(completed, spy, now=today_noon)  # SPY also carries today's (partial) bar

    CrossMomentumStrategy._risk_exposure(fake, _target(completed))

    assert fake.warnings == []  # SPY's last date is yesterday too once today's bar is gone
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_timing.py tests/strategies/test_cross_momentum_indicators.py tests/strategies/test_cross_momentum_risk_overlay.py -v`
Expected: `ImportError: cannot import name 'completed_bars'`; then (after `completed_bars` exists) the scan tests
FAIL (`requested_lengths == [300]`, `price` is the crash), the fakes FAIL on `_market_date` (AttributeError),
and `test_risk_exposure_drops_spys_partial_bar_for_today` FAILS with one warning.

- [ ] **Step 3: Implement `completed_bars`**

In `utils.py`, add `date` to the `from datetime import ...` line (`from datetime import UTC, date, datetime`),
and after `close_series`:

```python
def completed_bars(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """`df` without its bars dated `today` (market time): while a session is open its daily bar is partial.

    The strategy only iterates between its start time and the close, so a bar dated today is always partial
    there. In backtests the data gate already hides it and this is a no-op.
    """
    dates = pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ).date
    return df[dates != today]
```

- [ ] **Step 4: Use it in the scan and for SPY**

In `agent_cross_momentum.py`: add `completed_bars` to the `from .utils import (...)` list and `date` to
`from datetime import datetime` (`from datetime import date, datetime`). Below `_REBALANCE_BAND = 0.20` add:

```python
# Daily bars behind every computation (scores, filters, breadth, risk overlay), all from completed sessions.
_HISTORY_BARS = 300
```

Add this method before `_compute_indicators_for_ticker`:

```python
    def _market_date(self) -> date:
        """Today's date in market time: the date of the session whose daily bar is still forming."""
        return self.get_datetime().astimezone(MARKET_TZ).date()
```

In `_compute_indicators_for_ticker`, change `bars = self.get_historical_prices(ticker, length=300, timestep="day")`
to `length=_HISTORY_BARS + 1`, and right after the `isinstance(df, pd.DataFrame)` check insert:

```python
        # Completed sessions only: today's bar is partial, and it made the same Tuesday's decision depend on the
        # hour it ran (2026-10-06). One extra bar is fetched so the window stays _HISTORY_BARS long.
        df = completed_bars(df, self._market_date()).tail(_HISTORY_BARS)
        if df.empty:
            return None
```

In `_risk_exposure`, change the SPY fetch to `length=_HISTORY_BARS + 1` and
`spy = close_series(spy_bars.pandas_df)` to:

```python
        spy = close_series(completed_bars(spy_bars.pandas_df, self._market_date()).tail(_HISTORY_BARS))
        if spy.empty:
            self.log_warning("Risk overlay: no completed SPY session — defaulting to NORMAL (100% exposure)")
            return 1.0
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/strategies/ -q -k cross_momentum`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum tests/strategies/test_cross_momentum_timing.py tests/strategies/test_cross_momentum_indicators.py tests/strategies/test_cross_momentum_risk_overlay.py
git commit -m "Task 3: cross_momentum scores, filters and the overlay read completed sessions only"
```

---

### Task 4: Size orders at the last trade

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`rebalance`)
- Test: `tests/strategies/test_cross_momentum_rebalance.py`

**Interfaces:**
- Consumes: `_price_or_zero(symbol) -> float` (existing).
- Produces: no new names; `rebalance()` sizing semantics change.

- [ ] **Step 1: Write the failing tests**

Append to `tests/strategies/test_cross_momentum_rebalance.py`:

```python
def test_buys_are_sized_at_the_last_trade_not_the_completed_close():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 50.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})  # closed at 100, trades at 50

    assert _orders(fake, "AAA", "buy") == [6.0]


def test_trims_are_sized_at_the_last_trade():
    fake = FakeStrategy(cash=0.0, positions=[_held("AAA", 10.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.4, 80.0, 1)], {"AAA": 1})  # closed at 80, trades at 100

    assert _orders(fake, "AAA", "sell") == [6.0]  # worth 1000 against a 400 target


def test_a_target_without_a_price_is_neither_bought_nor_trimmed():
    fake = FakeStrategy(cash=1000.0, last_prices={"BBB": 50.0, "SHV": 50.0}, price_errors={"AAA"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1), _target("BBB", 0.3, 50.0, 2)], {"AAA": 1, "BBB": 2})

    assert _orders(fake, "AAA", "buy") == []
    assert _orders(fake, "BBB", "buy") == [6.0]
    assert any("AAA" in message and "no buy or trim" in message for message in fake.warnings)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -v`
Expected: the three new tests FAIL (quantities `[3.0]` and `[5.0]`; AAA bought / no "no buy or trim"
warning); every existing test PASSES.

- [ ] **Step 3: Implement**

In `rebalance()`, right after `current_positions = self.get_positions()`, add:

```python
        # Size at the last trade: an entry's "price" is the last COMPLETED close, a session old by now.
        prices = {entry["symbol"]: self._price_or_zero(entry["symbol"]) for entry in target}
        for symbol, price in prices.items():
            if price <= 0:
                self.log_warning(f"No price for {symbol}: no buy or trim this week")
```

In Phase 1's `if symbol in target_symbols:` branch, replace

```python
                entry = target_by_symbol[symbol]
                target_value = portfolio_value * entry["target_weight"]
                current_value = float(pos.quantity) * entry["price"]
```

with

```python
                entry = target_by_symbol[symbol]
                price = prices[symbol]
                if price <= 0:
                    continue
                target_value = portfolio_value * entry["target_weight"]
                current_value = float(pos.quantity) * price
```

and in the rest of that branch replace every `entry["price"]` with `price` (the `trim_qty` division, the
`Trimming` log line, and `estimated_sell_proceeds += trim_qty * price`).

In Phase 2's loop, after `symbol = entry["symbol"]` add:

```python
            price = prices[symbol]
            if price <= 0:
                continue
```

and replace every `entry["price"]` in the loop body with `price` (`current_value`, `raw_qty`, `cost`, the
reduced `quantity`, the `Holding` and `Buying` log lines).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/strategies/test_cross_momentum_rebalance.py
git commit -m "Task 4: cross_momentum sizes trims and buys at the last trade"
```

---

### Task 5: `rebalance_time` parameter and docs

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/parameters.py` (rebalance schedule block)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (new `parse_rebalance_time`)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`__init__`, module docstring)
- Modify: `CLAUDE.md`
- Test: `tests/strategies/test_cross_momentum_timing.py`

**Interfaces:**
- Consumes: `Strategy.iteration_start_time` (Task 1).
- Produces: `parse_rebalance_time(value: str) -> datetime.time` (naive, market time); raises `ConfigurationError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/strategies/test_cross_momentum_timing.py` (add the imports
`from tests.fakes import FakeBroker, FakeClock, et`,
`from trading_agent_framework.config import TradingMode`,
`from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy`,
`from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG`,
`from trading_agent_framework.strategies.cross_momentum.utils import parse_rebalance_time`,
`from trading_agent_framework.utils.errors import ConfigurationError`):

```python
def test_the_default_rebalance_time_is_noon():
    assert CONFIG["rebalance_time"] == "12:00"


def test_parse_rebalance_time_reads_hh_mm():
    assert parse_rebalance_time("12:00") == time(12, 0)
    assert parse_rebalance_time("10:30") == time(10, 30)


@pytest.mark.parametrize("value", ["noon", "25:00", "12:00+02:00", ""])
def test_parse_rebalance_time_rejects_anything_else(value):
    with pytest.raises(ConfigurationError, match="rebalance_time"):
        parse_rebalance_time(value)


def _strategy(**parameters):
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 7), []))
    return CrossMomentumStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], parameters=parameters)


def test_the_strategy_iterates_at_its_rebalance_time():
    assert _strategy().iteration_start_time == time(12, 0)
    assert _strategy(rebalance_time="10:30").iteration_start_time == time(10, 30)


def test_a_malformed_rebalance_time_fails_at_construction():
    with pytest.raises(ConfigurationError, match="rebalance_time"):
        _strategy(rebalance_time="noon")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_timing.py -v`
Expected: `ImportError: cannot import name 'parse_rebalance_time'`; after Step 3's helper, the CONFIG and
strategy tests FAIL (`KeyError: 'rebalance_time'`, `iteration_start_time is None`).

- [ ] **Step 3: Implement**

`parameters.py`, in the "Rebalance schedule" block after `"day_of_week": 1,`:

```python
    # Market time (ET, HH:MM) of the daily iteration, so of the weekly rebalance: past the opening's wide spreads,
    # well before the close. The decision reads completed sessions only, so the hour changes execution, not signals.
    "rebalance_time": "12:00",
```

`utils.py`: change `from datetime import UTC, date, datetime` to `from datetime import UTC, date, datetime, time`,
add `from trading_agent_framework.utils.errors import ConfigurationError`, and add:

```python
def parse_rebalance_time(value: str) -> time:
    """`HH:MM` in market time (America/New_York): when the executor runs each session's iteration."""
    try:
        parsed = time.fromisoformat(value)
    except (TypeError, ValueError):
        raise ConfigurationError(f"rebalance_time must be HH:MM in market time, got {value!r}") from None
    if parsed.tzinfo is not None:
        raise ConfigurationError(f"rebalance_time is market time and takes no UTC offset, got {value!r}")
    return parsed
```

`agent_cross_momentum.py`: add `parse_rebalance_time` to the `from .utils import (...)` list; in `__init__`,
right after `self.parameters = {**CONFIG, **self.parameters}`:

```python
        # The executor runs each session's iteration, so the Tuesday rebalance, at this market time
        self.iteration_start_time = parse_rebalance_time(self.parameters["rebalance_time"])
```

In the module docstring, change line 13's `4. Weekly (Friday): filter, score, rank, select top N, weight by inverse vol`
to `4. Weekly (Tuesday, at rebalance_time): filter, score, rank, select top N on completed sessions, weight by inverse vol`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/ -q -k cross_momentum`
Expected: all PASS.

- [ ] **Step 5: Update `CLAUDE.md`**

In the `core/` bullet of the Architecture section, after `` `timing.py` (pure `sleeptime` parsing and tick maths), ``
insert: `` `Strategy.iteration_start_time` (a market `time`, default `None` = the open) moves each session's first `on_trading_iteration` -- the other hooks keep their times, a start in the closing window skips that session's iterations, a mid-session start after it iterates at once -- ``.

In "Key patterns / gotchas", after the "vwap_pullback backtests charge 5 bps" bullet, add:

```markdown
- **cross_momentum decides on completed sessions and rebalances at 12:00 ET.** Live daily bars include the forming session's bar, so a Tuesday decision used to depend on the hour it ran: on 2026-10-06 the 09:42 ET run read the risk overlay NORMAL and a 15:36 restart read CRITICAL. Every computation (scores, filters, breadth, the overlay's SPY series) now drops the bar dated today (`utils.completed_bars`, a no-op in backtests where the data gate already hides it) and keeps 300 completed bars; trims and buys are sized at the last trade (`get_last_price`), not at the completed close. The overlay joins its series on session DATE (`utils.close_series`; live bars are stamped at midnight, backtest bars at the close): it used to align them by position, and half the series ending a session early halved beta and correlation. The cause of that shift was never found; a warning names any target whose last bar date differs from SPY's. `parameters["rebalance_time"]` ("12:00", ET) sets `iteration_start_time`, so the daily equity sample is taken at 12:00 too. A restart on a Tuesday after 12:00 rebalances again (deliberately unguarded).
```

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum tests/strategies/test_cross_momentum_timing.py CLAUDE.md
git commit -m "Task 5: cross_momentum rebalance_time parameter (12:00 ET) drives iteration_start_time"
```

---

### Task 6: Verification

**Files:**
- Create (scratch, not committed): `<scratchpad>/replay_overlay.py`

**Interfaces:**
- Consumes: everything above. Produces nothing in the repo.

- [ ] **Step 1: Full suite, lint, types**

Run: `uv run pytest -q && uv run ruff check && uv run pyright`
Expected: all tests PASS, `All checks passed!`, `0 errors`. Report any failure by name; do not continue past a
red suite.

- [ ] **Step 2: Backtest comparison**

Run (in the background; it takes a while): `uv run agent cross_momentum backtesting`
Then compare the new run's `metrics.json` with the reference run's:

```bash
uv run python - <<'EOF'
import json, pathlib
runs = sorted(pathlib.Path("logs/cross_momentum/backtesting").iterdir())
ref = json.loads(pathlib.Path("logs/cross_momentum/backtesting/2026-09-29_151101_backtesting/metrics.json").read_text())
new = json.loads((runs[-1] / "metrics.json").read_text())
print("new run:", runs[-1].name)
for key in sorted(set(ref) & set(new)):
    if isinstance(ref[key], (int, float)) and isinstance(new[key], (int, float)):
        print(f"{key:40s} {ref[key]:>14.4f} {new[key]:>14.4f}")
EOF
```

Expected: total return, CAGR, Sharpe, max drawdown and trade count close to the reference (backtests already
decided on the previous close and filled at the next open; only the date join can move them). A material
difference is investigated as a bug before completion, not explained away.

- [ ] **Step 3: Real-data replay of the 2026-10-06 failure (read-only)**

(The spec's "09:42 against 15:48" replay cannot be rebuilt: Alpaca serves a bar's current content, not its
09:42 content, and `completed_bars` drops that bar in both cases by construction. This step replays the failure
mode that was actually measured, half the series a session short, on real data.)

Write `<scratchpad>/replay_overlay.py` (scratchpad directory, never the repo) and run it with
`uv run python -I <scratchpad>/replay_overlay.py`:

```python
"""Read-only: the 2026-10-06 targets on real IEX bars, with half the series a session short, through the new code."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client
from trading_agent_framework.config.env import AlpacaCredentials, load_strategy_env
from trading_agent_framework.strategies.cross_momentum.portfolio_risk_overlay import compute_risk_overlay
from trading_agent_framework.strategies.cross_momentum.utils import close_series, completed_bars

ET = ZoneInfo("America/New_York")
load_strategy_env("cross_momentum", "live")
client = build_stock_data_client(AlpacaCredentials.for_data())
SYMS = "SNDK MU MRNA LITE RVMD DELL WDC STX INTC AAOI MRVL NBIS TSEM DOCN AMD HPE LRCX TER SMTC TWST".split()
now = datetime.now(ET)
df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=[*SYMS, "SPY"], timeframe=TimeFrame.Day, start=now - timedelta(days=450), end=now, feed=DataFeed.IEX, adjustment=Adjustment.ALL)).df
series = {s: close_series(completed_bars(df.loc[s].tz_convert(ET), now.date()).tail(300)) for s in [*SYMS, "SPY"]}
weights = {s: 1 / len(SYMS) for s in SYMS}
full = compute_risk_overlay({s: series[s] for s in SYMS}, weights, series["SPY"])
short = compute_risk_overlay({s: (series[s].iloc[:-1] if i % 2 else series[s]) for i, s in enumerate(SYMS)}, weights, series["SPY"])
print("aligned:              ", full[0], full[2])
print("half a session short: ", short[0], short[2])
```

Expected: both lines show the same state, with beta and correlation within a few hundredths (the old code read
NORMAL on the second line). Report both lines.

- [ ] **Step 4: Report**

Summarize: suite/lint/pyright results, the metrics table from Step 2 with any material difference called out,
the two replay lines from Step 3, and the rollout reminder from the spec (restart the live bot; a Tuesday
restart after 12:00 ET rebalances at once; equity sampled at 12:00 from then on).

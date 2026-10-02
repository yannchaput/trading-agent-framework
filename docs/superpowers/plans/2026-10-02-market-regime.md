# Market Regime Estimate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every strategy computes a daily market regime (`1` bullish, `0` neutral, `-1` bearish) by default, logs and charts it, and the dashboard shows it under "Trade Activity"; the ADX / RSI / VIX charting of vwap_pullback is removed.

**Architecture:** A pure classifier in `core/regime.py` (trend from SMA50/SMA200 of the benchmark's daily closes, capped at neutral when 20-day realized volatility is abnormally high). `Strategy._refresh_regime()` reads the bars through the existing broker facade, stores `strategy.regime`, logs it and calls `add_line("Regime", ...)`. The executor calls it once per session before `before_market_opens`. `run_backtesting` widens the warmup so the regime exists from the first session. `trades_chart` renders the `"Regime"` pane as a dedicated step-line row with colored bands.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pandas (bars), plotly (dashboard). No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-02-market-regime-design.md`

## Global Constraints

- Regime values are ints: `1` bullish, `0` neutral, `-1` bearish. `None` means "not computable yet".
- Defaults: `sma_fast=50`, `sma_slow=200`, `vol_window=20`, `vol_lookback=252`, `vol_percentile=0.80`; `min_bars = max(sma_slow, vol_window + vol_lookback) + 1` (273).
- Cadence: once per session, whatever the `sleeptime`.
- Information only: no strategy trades on the regime; cross_momentum's breadth overlay and the scorecard's manual Regime tag are untouched.
- `core/regime.py` is pure: no I/O, no pandas, no `Strategy` import.
- No wall-clock time (`datetime.now`, `time.sleep`) anywhere; time comes from `strategy.clock`.
- A regime failure never skips a session, never ends a run and never calls `on_bot_crash`.
- The indicator line and its pane are both named exactly `"Regime"` (`add_line("Regime", value, plot_name="Regime")`).
- Dashboard charts use `CHART_TEMPLATE` (dark); never `plotly_white`.
- Tests never touch the network and use the hand-written fakes in `tests/fakes.py` / `tests/backtesting/fakes.py`, not `MagicMock`.
- Commands: `uv run pytest`, `uv run ruff check`. Python 3.14, package manager `uv`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Repo cadence: one commit per task, then a review-fix commit if the review finds something.

## Review Focus

1. **A benchmark whose volatility is perfectly flat** (constant daily return, or a synthetic test series): it must not read as permanently "stressed" because of float noise. Pinned in Task 1 (`test_a_flat_volatility_series_is_not_stressed`).
2. **A benchmark with too little history, or no bars at all** (new listing, a data source built without warmup): `strategy.regime` stays `None`, one warning for the whole run, the session runs normally. Pinned in Task 2.
3. **A corrupt bar** (zero, negative or NaN close): the refresh warns and keeps the previous value instead of raising a maths error into the executor. Pinned in Tasks 1 and 2.
4. **A day-timestep backtest started with `warmup_trading_days=0`**: the runner's eager benchmark load must be widened too, or the data source caches a short frame and the regime is missing for the first ~270 sessions. Also: a bar that closes after the refresh must not change that session's value. Pinned in Task 3 (end-to-end test).
5. **A live run started mid-session, and an unexpected exception inside the refresh**: the regime is still refreshed (the `before_market_opens` hook is skipped in that case), and the session's iterations still run. Pinned in Task 3 (executor tests).

Also covered in Task 5: a run directory with no `"Regime"` pane (old runs) renders exactly as before, and a regime series with a single point does not break the chart.

## File Structure

| File | Responsibility |
|---|---|
| `src/trading_agent_framework/core/regime.py` (create) | Pure: `RegimeParameters`, `RegimeReading`, `classify_regime`, the line name and labels. |
| `src/trading_agent_framework/core/strategy.py` (modify) | `regime_params`, `regime`, `_refresh_regime()`, warmup widening in `run_backtesting`. |
| `src/trading_agent_framework/core/executor.py` (modify) | One guarded `_refresh_regime()` call per session. |
| `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` (modify) | Remove ADX / RSI / VIX charting. |
| `src/trading_agent_framework/agents/tools/vix.py` (delete) | No remaining user. |
| `src/trading_agent_framework/dashboard/components/charts.py` (modify) | Regime row and bands in `trades_chart`. |
| `tests/core/test_regime.py` (create) | Classifier tests. |
| `tests/core/test_strategy_regime.py` (create) | `Strategy._refresh_regime` tests. |
| `tests/core/test_executor.py` (modify) | Executor calls the refresh once per session. |
| `tests/backtesting/test_regime_backtest.py` (create) | Warmup widening and end-to-end regime in a backtest. |
| `tests/strategies/vwap_pullback/test_vwap_strategy.py` (modify) | Drop charting tests, add "charts nothing of its own". |
| `tests/agents/tools/test_vix.py` (delete) | Tests the deleted module. |
| `tests/dashboard/test_trades_chart.py`, `tests/dashboard/test_reader.py` (modify) | Regime row tests; sample panes renamed. |
| `CLAUDE.md`, `TODO.md`, the spec (modify) | Docs. |

---

### Task 1: Pure regime classifier

**Files:**
- Create: `src/trading_agent_framework/core/regime.py`
- Create: `tests/core/test_regime.py`
- Modify: `docs/superpowers/specs/2026-10-02-market-regime-design.md` (one sentence, step 6)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `REGIME_LINE: str = "Regime"`
  - `REGIME_LABELS: dict[int, str] = {1: "bullish", 0: "neutral", -1: "bearish"}`
  - `RegimeParameters(sma_fast=50, sma_slow=200, vol_window=20, vol_lookback=252, vol_percentile=0.80)`, frozen, with property `min_bars -> int`. Invalid values raise `ValueError`.
  - `RegimeReading(regime: int, close: float, sma_fast: float, sma_slow: float, vol: float, vol_threshold: float, stressed: bool)`, frozen.
  - `classify_regime(closes: Sequence[float], params: RegimeParameters = RegimeParameters()) -> RegimeReading | None`. `None` with fewer than `params.min_bars` closes; `ValueError` on a non-finite or non-positive close.

The tests use small windows (`sma_fast=3, sma_slow=5, vol_window=2, vol_lookback=4`, so `min_bars == 7`) with hand-checked series:

| Series | Close vs SMA5 | SMA3 vs SMA5 | Latest vol | Expected |
|---|---|---|---|---|
| `[1,2,3,4,5,6,7]` | 7 > 5 | 6 > 5 | lowest of the window | `+1` |
| `[7,6,5,4,3,2,1]` | 1 < 3 | 2 < 3 | n/a | `-1` |
| `[5,5,10,10,1,1,8]` | 8 > 6 | 3.33 < 6 | n/a | `0` |
| `[10,10.1,10.2,10.3,10.4,10.5,13]` | 13 > 10.88 | 11.3 > 10.88 | spike, highest | `0` (capped) |
| `[10,9.9,9.8,9.7,9.6,9.5,7]` | 7 < 9.12 | 8.7 < 9.12 | spike, highest | `-1` (vol never lifts a bearish read) |

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_regime.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.core.regime import REGIME_LABELS, REGIME_LINE, RegimeParameters, classify_regime

SMALL = RegimeParameters(sma_fast=3, sma_slow=5, vol_window=2, vol_lookback=4)


def test_defaults_need_273_daily_closes() -> None:
    params = RegimeParameters()
    assert (params.sma_fast, params.sma_slow, params.vol_window, params.vol_lookback, params.vol_percentile) == (50, 200, 20, 252, 0.80)
    assert params.min_bars == 273
    assert SMALL.min_bars == 7


def test_the_line_name_and_labels() -> None:
    assert REGIME_LINE == "Regime"
    assert REGIME_LABELS == {1: "bullish", 0: "neutral", -1: "bearish"}


def test_an_uptrend_with_calm_volatility_is_bullish() -> None:
    reading = classify_regime([1, 2, 3, 4, 5, 6, 7], SMALL)
    assert reading is not None
    assert reading.regime == 1 and not reading.stressed
    assert (reading.close, reading.sma_fast, reading.sma_slow) == (7, 6, 5)


def test_a_downtrend_is_bearish() -> None:
    reading = classify_regime([7, 6, 5, 4, 3, 2, 1], SMALL)
    assert reading is not None and reading.regime == -1


def test_a_close_above_the_slow_average_with_the_fast_average_below_is_neutral() -> None:
    reading = classify_regime([5, 5, 10, 10, 1, 1, 8], SMALL)
    assert reading is not None and reading.regime == 0


def test_a_volatility_spike_caps_a_bullish_trend_at_neutral() -> None:
    reading = classify_regime([10, 10.1, 10.2, 10.3, 10.4, 10.5, 13], SMALL)
    assert reading is not None
    assert reading.stressed and reading.vol > reading.vol_threshold
    assert reading.regime == 0


def test_a_volatility_spike_leaves_a_bearish_trend_bearish() -> None:
    reading = classify_regime([10, 9.9, 9.8, 9.7, 9.6, 9.5, 7], SMALL)
    assert reading is not None and reading.stressed
    assert reading.regime == -1


def test_a_flat_volatility_series_is_not_stressed() -> None:
    # Constant daily return: every rolling volatility is equal up to float noise, which must not read as a spike.
    reading = classify_regime([100 * 1.01**i for i in range(7)], SMALL)
    assert reading is not None
    assert not reading.stressed and reading.regime == 1


def test_only_the_last_min_bars_matter_for_the_volatility_window() -> None:
    # A huge move long before the lookback window must not change the reading.
    calm = [100 * 1.01**i for i in range(7)]
    assert classify_regime([1.0, 500.0, *calm], SMALL) == classify_regime(calm, SMALL)


def test_too_few_closes_give_no_reading() -> None:
    assert classify_regime([1, 2, 3, 4, 5, 6], SMALL) is None
    assert classify_regime([], SMALL) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_close_that_is_not_finite_and_positive_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        classify_regime([1, 2, 3, bad, 5, 6, 7], SMALL)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sma_fast": 0},
        {"sma_fast": 200, "sma_slow": 200},
        {"vol_window": 1},
        {"vol_lookback": 0},
        {"vol_percentile": 0.0},
        {"vol_percentile": 1.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RegimeParameters(**kwargs)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_regime.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.core.regime'`.

- [ ] **Step 3: Write the implementation**

Create `src/trading_agent_framework/core/regime.py`:

```python
"""Pure market-regime estimate from a benchmark's daily closes (no I/O, no pandas, no `Strategy`).

Trend: the close and the fast SMA against the slow SMA. Volatility: when the latest realized volatility is
above its own high percentile over the lookback, a bullish read is capped at neutral. Volatility never
turns a read bearish. `Strategy._refresh_regime` feeds it and logs the result; nothing trades on it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

REGIME_LINE = "Regime"  # the `add_line` name and pane; the dashboard's `charts.REGIME_PANE` must match
REGIME_LABELS = {1: "bullish", 0: "neutral", -1: "bearish"}

_TRADING_DAYS_PER_YEAR = 252
_VOL_EPSILON = 1e-12  # float noise between equal volatilities must not read as a spike


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    sma_fast: int = 50
    sma_slow: int = 200
    vol_window: int = 20
    vol_lookback: int = 252
    vol_percentile: float = 0.80

    def __post_init__(self) -> None:
        if not 1 <= self.sma_fast < self.sma_slow:
            raise ValueError(f"need 1 <= sma_fast < sma_slow, got {self.sma_fast} and {self.sma_slow}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be at least 2, got {self.vol_window}")
        if self.vol_lookback < 1:
            raise ValueError(f"vol_lookback must be at least 1, got {self.vol_lookback}")
        if not 0 < self.vol_percentile < 1:
            raise ValueError(f"vol_percentile must be strictly between 0 and 1, got {self.vol_percentile}")

    @property
    def min_bars(self) -> int:
        """Daily closes `classify_regime` needs (and the only ones it reads)."""
        return max(self.sma_slow, self.vol_window + self.vol_lookback) + 1


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """The regime (1 bullish, 0 neutral, -1 bearish) and the figures behind it."""

    regime: int
    close: float
    sma_fast: float
    sma_slow: float
    vol: float
    vol_threshold: float
    stressed: bool


def classify_regime(closes: Sequence[float], params: RegimeParameters = RegimeParameters()) -> RegimeReading | None:
    """The regime at the last of `closes` (oldest first); None with fewer than `params.min_bars` closes."""
    if len(closes) < params.min_bars:
        return None
    window = [float(close) for close in closes[-params.min_bars :]]
    if any(not math.isfinite(close) or close <= 0 for close in window):
        raise ValueError("closes must be finite and positive")
    close = window[-1]
    sma_fast = statistics.fmean(window[-params.sma_fast :])
    sma_slow = statistics.fmean(window[-params.sma_slow :])
    if close > sma_slow and sma_fast > sma_slow:
        trend = 1
    elif close < sma_slow and sma_fast < sma_slow:
        trend = -1
    else:
        trend = 0
    vol, vol_threshold = _volatility(window, params)
    stressed = vol - vol_threshold > _VOL_EPSILON
    return RegimeReading(
        regime=0 if trend == 1 and stressed else trend,
        close=close,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        vol=vol,
        vol_threshold=vol_threshold,
        stressed=stressed,
    )


def _volatility(closes: Sequence[float], params: RegimeParameters) -> tuple[float, float]:
    """The latest annualized realized volatility, and its `vol_percentile` over the last `vol_lookback` values."""
    returns = [math.log(current / previous) for previous, current in zip(closes, closes[1:])]
    annualize = math.sqrt(_TRADING_DAYS_PER_YEAR)
    vols = [statistics.stdev(returns[end - params.vol_window : end]) * annualize for end in range(params.vol_window, len(returns) + 1)]
    return vols[-1], _percentile(vols[-params.vol_lookback :], params.vol_percentile)


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile (numpy's default method), on plain floats."""
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/test_regime.py -q`
Expected: all pass.

- [ ] **Step 5: Lint**

Run: `uv run ruff check src/trading_agent_framework/core/regime.py tests/core/test_regime.py`
Expected: `All checks passed!`

- [ ] **Step 6: Align the spec with the strict comparison**

The spec says a bullish read is capped when the vol is "at or above" its percentile. With "at or above", a perfectly flat volatility series is always stressed (every value equals the percentile). The code uses strictly above. In `docs/superpowers/specs/2026-10-02-market-regime-design.md`, section "1. `core/regime.py` (pure)", replace:

```
If the latest vol is at or above its own
    `vol_percentile` over the trailing `vol_lookback` days, a `+1` is capped to
    `0`.
```

with:

```
If the latest vol is strictly above its own
    `vol_percentile` over the trailing `vol_lookback` days (the latest value
    included, float noise ignored), a `+1` is capped to `0`.
```

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/core/regime.py tests/core/test_regime.py docs/superpowers/specs/2026-10-02-market-regime-design.md
git commit -m "$(cat <<'EOF'
feat: pure market regime classifier (Task 1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `Strategy.regime` and `_refresh_regime`

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py` (imports at lines 18-41, class attributes near line 95, `__init__` near line 123, new method after `add_line`, which ends at line 297)
- Create: `tests/core/test_strategy_regime.py`

**Interfaces:**
- Consumes (Task 1): `REGIME_LABELS`, `REGIME_LINE`, `RegimeParameters`, `classify_regime` from `trading_agent_framework.core.regime`.
- Produces:
  - `Strategy.regime_params: RegimeParameters` (class attribute, overridable).
  - `strategy.regime: int | None` (instance attribute, `None` until computed).
  - `Strategy._refresh_regime() -> None`. Reads `regime_params.min_bars` daily bars of `benchmark_symbol`, sets `regime`, logs, calls `add_line(REGIME_LINE, regime, color="#d1d4dc", plot_name=REGIME_LINE)`. Never raises `BrokerError`, `BacktestError` or `ValueError`.

Notes for the implementer:
- `tests/fakes.py` `FakeBroker.get_bars` returns `bar_frames[symbol].iloc[-length:]` and records `bars_calls` as `(symbols, length, timestep, include_after_hours)`. Setting `broker.market_data_error = BrokerError(...)` makes it raise.
- `Strategy.add_line` is already a no-op outside backtesting and with a non-`BacktestBroker`, so the tests replace it with a recorder.
- `BacktestError` lives in `trading_agent_framework.utils.errors` (`BacktestDataError` is its subclass).

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_strategy_regime.py`:

```python
from __future__ import annotations

import logging

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.regime import RegimeParameters
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import BacktestDataError, BrokerError

UPTREND = [100 * 1.001**i for i in range(300)]
DOWNTREND = [100 * 0.999**i for i in range(300)]


def _strategy(closes: list[float] | None = None, cls: type[Strategy] = Strategy) -> tuple[Strategy, FakeBroker, list[tuple]]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 8, 30)))
    if closes is not None:
        broker.bar_frames["SPY"] = make_bars_frame(closes, start=et(2025, 1, 6))
    strategy = cls(broker, mode=TradingMode.BACKTESTING)
    lines: list[tuple] = []
    strategy.add_line = lambda name, value, **kw: lines.append((name, value, kw["plot_name"]))  # ty: ignore[invalid-assignment]
    return strategy, broker, lines


def test_the_regime_is_unknown_before_the_first_refresh() -> None:
    strategy, _, _ = _strategy(UPTREND)
    assert strategy.regime is None


def test_a_refresh_sets_logs_and_charts_the_regime(caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    with caplog.at_level(logging.INFO):
        strategy._refresh_regime()
    assert strategy.regime == 1
    assert lines == [("Regime", 1, "Regime")]
    assert broker.bars_calls == [(("SPY",), 273, "day", True)]
    assert "Market regime +1 (bullish)" in caplog.text


def test_a_downtrend_is_logged_as_minus_one() -> None:
    strategy, _, lines = _strategy(DOWNTREND)
    strategy._refresh_regime()
    assert strategy.regime == -1
    assert lines == [("Regime", -1, "Regime")]


def test_the_benchmark_and_the_parameters_come_from_the_class() -> None:
    class Small(Strategy):
        benchmark_symbol = "QQQ"
        regime_params = RegimeParameters(sma_fast=3, sma_slow=5, vol_window=2, vol_lookback=4)

    strategy, broker, _ = _strategy(cls=Small)
    broker.bar_frames["QQQ"] = make_bars_frame([7, 6, 5, 4, 3, 2, 1], start=et(2026, 9, 1))
    strategy._refresh_regime()
    assert strategy.regime == -1
    assert broker.bars_calls == [(("QQQ",), 7, "day", True)]


@pytest.mark.parametrize("error", [BrokerError("data down"), BacktestDataError("no session")])
def test_a_data_failure_keeps_the_previous_regime(error: Exception, caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    strategy._refresh_regime()
    broker.market_data_error = error  # ty: ignore[invalid-assignment]
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime == 1
    assert len(lines) == 1  # nothing charted for the failed refresh
    assert "Market regime not refreshed" in caplog.text


@pytest.mark.parametrize("closes", [None, UPTREND[:100]])
def test_too_little_history_leaves_the_regime_unknown_and_warns_once(closes: list[float] | None, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, lines = _strategy(closes)
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
        strategy._refresh_regime()
    assert strategy.regime is None and lines == []
    assert caplog.text.count("Market regime unavailable") == 1


def test_a_corrupt_close_keeps_the_previous_regime(caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, lines = _strategy(UPTREND)
    strategy._refresh_regime()
    broker.bar_frames["SPY"] = make_bars_frame([*UPTREND[:-1], 0.0], start=et(2025, 1, 6))
    with caplog.at_level(logging.WARNING):
        strategy._refresh_regime()
    assert strategy.regime == 1 and len(lines) == 1
    assert "Market regime not refreshed" in caplog.text


def test_outside_backtesting_the_regime_is_set_and_nothing_is_charted() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 8, 30)))
    broker.bar_frames["SPY"] = make_bars_frame(UPTREND, start=et(2025, 1, 6))
    strategy = Strategy(broker, mode=TradingMode.PAPER)
    strategy._refresh_regime()  # the real add_line: a no-op outside backtesting
    assert strategy.regime == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_strategy_regime.py -q`
Expected: FAIL, `AttributeError: 'Strategy' object has no attribute 'regime'` (and `_refresh_regime`).

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/core/strategy.py`:

1. Add the import next to the other `core` imports (after `from trading_agent_framework.core.indicators import Indicators`):

```python
from trading_agent_framework.core.regime import REGIME_LABELS, REGIME_LINE, RegimeParameters, classify_regime
```

2. Add `BacktestError` to the errors import:

```python
from trading_agent_framework.utils.errors import BacktestError, BrokerError, ConfigurationError, LLMStatsError, OrderValidationError
```

3. Add the class attribute after the `agent_telemetry: bool = False` line:

```python
    # Market regime (1 bullish, 0 neutral, -1 bearish), estimated once per session from `benchmark_symbol`'s daily
    # bars (`core/regime.py`). Information only: logged, and charted in backtests. Override to change the windows.
    regime_params: RegimeParameters = RegimeParameters()
```

4. In `__init__`, after `self.run_id: str | None = None`:

```python
        # Latest market regime; None until `_refresh_regime` could compute one (the executor calls it every session).
        self.regime: int | None = None
        self._regime_history_warned = False
```

5. Add the method right after `add_line` (before `wait_for_order_execution`):

```python
    def _refresh_regime(self) -> None:
        """Estimate the market regime from the benchmark's daily bars; the executor calls this once per session.

        Information only: a log line, and an `add_line` point in backtests. Any failure keeps the previous
        value, because a regime estimate must never cost a session.
        """
        params = self.regime_params
        try:
            bars = self.get_historical_prices(self.benchmark_symbol, params.min_bars, "day")
            reading = None if bars is None else classify_regime(bars.df["close"].tolist(), params)
        except (BrokerError, BacktestError, ValueError) as exc:
            self.log_warning(f"Market regime not refreshed (still {self.regime}): {exc}")
            return
        if reading is None:
            if not self._regime_history_warned:
                self._regime_history_warned = True
                self.log_warning(f"Market regime unavailable: fewer than {params.min_bars} daily bars of {self.benchmark_symbol}")
            return
        self.regime = reading.regime
        self.log_info(
            f"Market regime {reading.regime:+d} ({REGIME_LABELS[reading.regime]}): {self.benchmark_symbol} close {reading.close:.2f}, "
            f"SMA{params.sma_fast} {reading.sma_fast:.2f}, SMA{params.sma_slow} {reading.sma_slow:.2f}, "
            f"vol {reading.vol:.1%} vs cap {reading.vol_threshold:.1%}{' (stressed)' if reading.stressed else ''}"
        )
        self.add_line(REGIME_LINE, reading.regime, color="#d1d4dc", plot_name=REGIME_LINE)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/test_strategy_regime.py tests/core/test_regime.py -q`
Expected: all pass.

- [ ] **Step 5: Run the core tests and lint**

Run: `uv run pytest tests/core -q && uv run ruff check src/trading_agent_framework/core tests/core`
Expected: all pass, `All checks passed!`. Nothing calls `_refresh_regime` yet, so no other test changes.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py tests/core/test_strategy_regime.py
git commit -m "$(cat <<'EOF'
feat: Strategy computes and logs the market regime (Task 2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Executor refresh once per session, and backtest warmup

**Files:**
- Modify: `src/trading_agent_framework/core/executor.py` (`_run_session`, lines 145-170; module docstring, lines 1-9)
- Modify: `src/trading_agent_framework/core/strategy.py` (`run_backtesting`, lines 481-578)
- Modify: `tests/core/test_executor.py` (append)
- Create: `tests/backtesting/test_regime_backtest.py`

**Interfaces:**
- Consumes (Task 2): `Strategy._refresh_regime()`, `Strategy.regime_params.min_bars`.
- Produces: `StrategyExecutor._refresh_regime()` (private, called in `_run_session`); `Strategy.run_backtesting` passes `warmup_trading_days = max(requested, regime_params.min_bars)` to both the data-source construction and `run_backtest`.

Why the runner's warmup must be widened, not only the data source's window: `run_backtest` eagerly loads the benchmark over `[start - warmup, end]`. With a warmup of 0, a data source that caches one frame per asset (Yahoo, and Alpaca for day-timestep runs) keeps that short frame, and the regime's later request for 273 daily bars is served from it.

- [ ] **Step 1: Write the failing executor tests**

Append to `tests/core/test_executor.py`:

```python
# --- market regime refresh -------------------------------------------------------------


class RegimeRecorder(Recorder):
    """Records the framework's regime refresh like a hook, to check where it sits in the session."""

    def _refresh_regime(self) -> None:
        self._record("refresh_regime")


def test_the_regime_is_refreshed_once_per_session_before_before_market_opens() -> None:
    strategy = _run(RegimeRecorder, sessions=2)
    assert strategy.times("refresh_regime") == [et(2026, 9, 14, 8, 30), et(2026, 9, 15, 8, 30)]
    assert strategy.hooks()[:3] == ["initialize", "refresh_regime", "before_market_opens"]


def test_a_mid_session_start_still_refreshes_the_regime() -> None:
    strategy = _run(RegimeRecorder, start=et(2026, 9, 14, 10, 15))
    assert "before_market_opens" not in strategy.hooks()
    assert strategy.times("refresh_regime") == [et(2026, 9, 14, 10, 15)]
    assert strategy.hooks()[:3] == ["initialize", "refresh_regime", "before_starting_trading"]


def test_a_failing_regime_refresh_never_costs_the_session(caplog: pytest.LogCaptureFixture) -> None:
    class Broken(Recorder):
        def _refresh_regime(self) -> None:
            raise RuntimeError("regime maths exploded")

    with caplog.at_level(logging.ERROR):
        strategy = _run(Broken)
    assert strategy.errors == []  # not a strategy hook: on_bot_crash is not called
    assert len(strategy.times("on_trading_iteration")) == 4
    assert "Market regime refresh failed" in caplog.text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/core/test_executor.py -q -k regime`
Expected: 3 failures (`refresh_regime` never recorded; `"Market regime refresh failed"` not logged).

- [ ] **Step 3: Implement the executor call**

In `src/trading_agent_framework/core/executor.py`, in `_run_session`, add the call after the first stop check:

```python
        started_before_open = self._now() < session.open
        self.wait_until(session.open - timedelta(minutes=strategy.minutes_before_opening))
        if self._stop.is_set():
            return
        self._refresh_regime()
        if started_before_open:
```

Add the helper in the `# --- helpers` section, before `_call_hook`:

```python
    def _refresh_regime(self) -> None:
        """Once per session, also on a mid-session start. A framework step, not a strategy hook: a failure is
        logged and the session carries on, without `on_bot_crash`."""
        try:
            self.strategy._refresh_regime()
        except Exception:
            logger.exception("Market regime refresh failed for strategy %s", self.strategy.name)
```

Update the module docstring's lifecycle sentence:

```python
`StrategyExecutor.run()` walks the market calendar one session at a time:
market regime refresh -> before_market_opens -> before_starting_trading ->
on_trading_iteration every `sleeptime` -> before_market_closes ->
after_market_closes. Every wait goes
```

- [ ] **Step 4: Run the executor tests**

Run: `uv run pytest tests/core/test_executor.py -q`
Expected: all pass, including the three new ones.

- [ ] **Step 5: Write the failing backtest tests**

Create `tests/backtesting/test_regime_backtest.py`:

```python
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
```

- [ ] **Step 6: Run them to verify they fail**

Run: `uv run pytest tests/backtesting/test_regime_backtest.py -q`
Expected: the `(0, 273)` and `(10, 273)` cases fail (`0 != 273`, `10 != 273`); the end-to-end test fails on `load_windows` (not widened) and an empty `Regime` series. The `(400, 400)` case passes already.

- [ ] **Step 7: Implement the warmup widening**

In `src/trading_agent_framework/core/strategy.py`, `run_backtesting`:

Replace

```python
        warmup_start = resolved_start - timedelta(days=warmup_calendar_days(warmup_trading_days))
```

with

```python
        # The regime needs `regime_params.min_bars` daily bars of the benchmark before the first session; without
        # them the eager benchmark load caches a frame too short for it (see `_refresh_regime`).
        resolved_warmup = max(warmup_trading_days, self.regime_params.min_bars)
        warmup_start = resolved_start - timedelta(days=warmup_calendar_days(resolved_warmup))
```

and in the `run_backtest(...)` call replace `warmup_trading_days=warmup_trading_days,` with:

```python
            warmup_trading_days=resolved_warmup,
```

In the docstring, replace the first sentence of the `warmup_trading_days:` entry

```
            warmup_trading_days: extra trading days of history to make available before
                `start` (default 0, i.e. no widening) so a strategy's indicators aren't
                starved near `backtesting_start`. Computed once, here, via
```

with

```
            warmup_trading_days: extra trading days of history to make available before
                `start` so a strategy's indicators aren't starved near `backtesting_start`.
                Raised to at least `regime_params.min_bars` (273 by default), which the
                market regime needs from the first session. Computed once, here, via
```

- [ ] **Step 8: Run the new tests**

Run: `uv run pytest tests/backtesting/test_regime_backtest.py -q`
Expected: all 4 pass.

- [ ] **Step 9: Run the full suite and fix what the new per-session bars request breaks**

Run: `uv run pytest -q`

Every executor run now makes one extra `get_bars([benchmark], 273, "day")` call per session, and `Strategy.run_backtesting` passes a warmup of at least 273. Expected failure shapes, and the fix for each (do not weaken an assertion beyond this):

- A test asserting an exact list of bars requests (`FakeBroker.bars_calls`, `FakeBacktestDataSource.bars_calls`): exclude the regime's request, which is the only one with `length == Strategy.regime_params.min_bars` and timestep `"day"` on the benchmark. Example:

```python
regime_bars = Strategy.regime_params.min_bars
calls = [call for call in broker.bars_calls if call[1] != regime_bars]
```

- A test asserting the warmup that `Strategy.run_backtesting` hands to `run_backtest` or to a data-source callable with a value below 273. Two are known, both in `tests/core/test_runners.py`: `test_run_backtesting_widens_default_yahoo_source_for_warmup` and `test_run_backtesting_forwards_warmup_trading_days_with_explicit_data_source` request `warmup_trading_days=10` and assert `10`. They test that an explicit value is forwarded, so change the requested value and every expected `10` to `400` (above the floor) in both; the 273 floor itself is covered by `tests/backtesting/test_regime_backtest.py`. Tests that monkeypatch `Strategy.run_backtesting` itself (ackman, news_binary) are unaffected.
- A test asserting that a run logs no warning: the regime warns once per run when the fake benchmark has fewer than 273 bars. Assert on the specific warning the test cares about instead of on an empty log.

If a failure has a different shape, stop and report it instead of adapting the test.

Expected after the fixes: full suite passes.

- [ ] **Step 10: Lint**

Run: `uv run ruff check`
Expected: `All checks passed!`

- [ ] **Step 11: Commit**

```bash
git add -A src/trading_agent_framework/core tests
git commit -m "$(cat <<'EOF'
feat: executor refreshes the regime each session; backtests warm it up (Task 3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Remove ADX / RSI / VIX from vwap_pullback

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py` (imports lines 13-33, `__init__` lines 69-70, tick line 111, methods lines 141-181)
- Delete: `src/trading_agent_framework/agents/tools/vix.py`
- Delete: `tests/agents/tools/test_vix.py`
- Modify: `tests/strategies/vwap_pullback/test_vwap_strategy.py` (lines 133-173)

**Interfaces:**
- Consumes: nothing from earlier tasks (the regime line now comes from the base class).
- Produces: `VwapPullbackStrategy` no longer has `_chart_indicators`, `_vix_previous_close`, `_vix`, `_vix_failed`. `strategy.indicators.adx` / `.rsi` and `agents/tools/indicators.py` are untouched.

- [ ] **Step 1: Replace the charting tests with the new failing test**

In `tests/strategies/vwap_pullback/test_vwap_strategy.py`, delete everything from `class _FakeIndicators:` to the end of `test_nothing_is_charted_outside_backtesting` (the class, `_charting`, and the three tests `test_a_tick_charts_adx_rsi_and_the_previous_vix_close`, `test_a_value_that_is_not_ready_is_skipped_not_charted`, `test_nothing_is_charted_outside_backtesting`). In their place add:

```python
def test_a_backtest_tick_charts_no_indicator_of_its_own(tmp_path: Path) -> None:
    # The market regime line comes from the base Strategy (once per session); the tick itself charts nothing.
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.scanner.prepare_session = lambda: _session()
    steps = _record_steps(strategy)
    lines: list[str] = []
    strategy.add_line = lambda name, value, **kw: lines.append(name)
    strategy.on_trading_iteration()
    assert steps == ["reconcile", "scan", "enter"]
    assert lines == []
```

- [ ] **Step 2: Do NOT run the new test yet**

Against the old code this test would reach `_vix_previous_close`, which builds a real `VixSeries` and calls Yahoo. The suite never touches the network, so the red run is skipped for this one test. Instead confirm the old tests are gone and the file still collects:

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_strategy.py -q --collect-only`
Expected: collection succeeds, `test_a_backtest_tick_charts_no_indicator_of_its_own` is listed, no `adx` / `vix` test remains.

- [ ] **Step 3: Remove the charting code**

In `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py`:

1. Delete the import lines:

```python
from trading_agent_framework.agents.tools.vix import VixSeries
```

```python
from trading_agent_framework.core.indicators import IndicatorRow
```

2. In `__init__`, delete:

```python
        self._vix: VixSeries | None = None  # backtest chart only, loaded on the first tick that needs it
        self._vix_failed = False
```

3. In `on_trading_iteration`, delete the last line of the tick:

```python
        self._chart_indicators(state)
```

4. Delete the whole `_chart_indicators` method and the whole `_vix_previous_close` method (from `def _chart_indicators(self, state: SessionState) -> None:` through `return self._vix.previous_close(day)`). Keep the `# --- backtesting ---` section comment above `_trade_log_path`.

5. Delete the two files:

```bash
git rm src/trading_agent_framework/agents/tools/vix.py tests/agents/tools/test_vix.py
```

- [ ] **Step 4: Remove the imports that became unused**

Run: `uv run ruff check --fix src/trading_agent_framework/strategies/vwap_pullback tests/strategies/vwap_pullback`
Expected: ruff removes any now-unused import (`date` in the strategy module if nothing else uses it) and ends with `All checks passed!`. `timedelta`, `BacktestError` and `BrokerError` stay: `_preload` and `_ensure_session` use them.

- [ ] **Step 5: Check nothing else references the removed code**

Run: `grep -rniE "VixSeries|agents\.tools\.vix|_chart_indicators|_vix" src tests scripts`
Expected: no output.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback tests/agents -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -A src/trading_agent_framework/strategies/vwap_pullback src/trading_agent_framework/agents/tools tests/strategies/vwap_pullback tests/agents/tools
git commit -m "$(cat <<'EOF'
refactor: vwap_pullback no longer charts ADX, RSI and VIX (Task 4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Regime row and bands on the Trade Activity chart

**Files:**
- Modify: `src/trading_agent_framework/dashboard/components/charts.py` (constants after `MODEL_COLORS` near line 28; `trades_chart`, lines 619-719)
- Modify: `tests/dashboard/test_trades_chart.py`
- Modify: `tests/dashboard/test_reader.py` (lines 561-577, sample names only)

**Interfaces:**
- Consumes: `reader.load_indicator_lines(ref)` output, unchanged: `{plot_name: [ {name, color, dash, times, values} ]}`, with `times` tz-aware UTC timestamps and `values` floats. The regime arrives as the pane `"Regime"` holding one series.
- Produces: `trades_chart(trades_data, title="Trade Activity", indicators=None)`, same signature. New module constants `REGIME_PANE = "Regime"`, `REGIME_AXIS_LABELS = {1: "Bullish", 0: "Neutral", -1: "Bearish"}`, `REGIME_BAND_COLORS = {1: "#16a34a", 0: "#6b7280", -1: "#dc2626"}`.

Layout rules:
- The regime row is always row 2, directly under the portfolio chart; other panes follow from row 3.
- `shared_xaxes=True` already matches every row's x-axis to row 1: that is the "same scale".
- One band per run of equal consecutive values, drawn twice: faint on the portfolio row, stronger on the regime row. A run ends where the next run starts; the last run ends at the latest time on the chart.

- [ ] **Step 1: Write the failing tests**

Replace the content of `tests/dashboard/test_trades_chart.py` with:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trading_agent_framework.dashboard.components.charts import REGIME_BAND_COLORS, trades_chart

T0 = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
TRADES = {
    "values": [{"time": T0, "portfolio_value": 10000.0}, {"time": T0 + timedelta(days=5), "portfolio_value": 10100.0}],
    "trades": [{"time": T0, "side": "buy", "symbol": "AAA", "qty": 1.0, "price": 10.0, "cost": 0.0, "portfolio_value": 10000.0}],
}


def _line(name: str) -> dict:
    return {"name": name, "color": "#ffffff", "dash": "solid", "times": [T0], "values": [1.0]}


def _regime(values: list[int]) -> dict:
    times = [T0 + timedelta(days=i) for i in range(len(values))]
    return {"Regime": [{"name": "Regime", "color": "#d1d4dc", "dash": "solid", "times": times, "values": [float(v) for v in values]}]}


def test_without_indicators_the_chart_is_a_single_pane() -> None:
    fig = trades_chart(TRADES)
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy"}
    assert len(fig._grid_ref) == 1
    assert not fig.layout.shapes


def test_each_indicator_pane_gets_its_own_sub_row_under_the_portfolio_chart() -> None:
    fig = trades_chart(TRADES, indicators={"Breadth": [_line("Breadth")], "Averages": [_line("Fast"), _line("Slow")]})
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy", "Breadth", "Fast", "Slow"}
    assert len(fig._grid_ref) == 3  # portfolio + two panes
    rows = {t.name: t.yaxis for t in fig.data}
    assert rows["Fast"] == rows["Slow"] != rows["Breadth"] != rows["Portfolio Value"]
    assert not fig.layout.shapes  # no regime, no bands


def test_the_regime_row_sits_right_under_the_portfolio_chart_on_the_same_time_axis() -> None:
    fig = trades_chart(TRADES, indicators={"Other": [_line("Other")], **_regime([1, 1, 0, -1])})
    rows = {t.name: t.yaxis for t in fig.data}
    assert (rows["Portfolio Value"], rows["Regime"], rows["Other"]) == ("y", "y2", "y3")
    assert fig.layout.xaxis2.matches == "x" and fig.layout.xaxis3.matches == "x"
    assert tuple(fig.layout.yaxis2.range) == (-1.3, 1.3)
    assert tuple(fig.layout.yaxis2.tickvals) == (-1, 0, 1)
    assert tuple(fig.layout.yaxis2.ticktext) == ("Bearish", "Neutral", "Bullish")
    regime = next(t for t in fig.data if t.name == "Regime")
    assert regime.line.shape == "hv" and list(regime.y) == [1, 1, 0, -1]
    portfolio, regime_row = fig.layout.yaxis.domain, fig.layout.yaxis2.domain
    assert (regime_row[1] - regime_row[0]) < (portfolio[1] - portfolio[0])


def test_each_regime_run_is_shaded_on_both_rows() -> None:
    fig = trades_chart(TRADES, indicators=_regime([1, 1, 0, -1]))  # three runs: bullish, neutral, bearish
    on_portfolio = [s for s in fig.layout.shapes if s.xref == "x"]
    on_regime = [s for s in fig.layout.shapes if s.xref == "x2"]
    expected = [REGIME_BAND_COLORS[1], REGIME_BAND_COLORS[0], REGIME_BAND_COLORS[-1]]
    assert [s.fillcolor for s in on_portfolio] == expected
    assert [s.fillcolor for s in on_regime] == expected
    assert all(s.opacity < 0.2 for s in on_portfolio) and all(s.opacity > s2.opacity for s, s2 in zip(on_regime, on_portfolio))
    # a run ends where the next one starts; the last one runs to the end of the chart (the last portfolio value)
    assert [(s.x0, s.x1) for s in on_portfolio] == [
        (T0, T0 + timedelta(days=2)),
        (T0 + timedelta(days=2), T0 + timedelta(days=3)),
        (T0 + timedelta(days=3), T0 + timedelta(days=5)),
    ]


def test_a_single_regime_point_makes_one_band_per_row() -> None:
    fig = trades_chart(TRADES, indicators=_regime([-1]))
    assert len(fig.layout.shapes) == 2
    assert {s.fillcolor for s in fig.layout.shapes} == {REGIME_BAND_COLORS[-1]}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/dashboard/test_trades_chart.py -q`
Expected: collection error, `ImportError: cannot import name 'REGIME_BAND_COLORS'`.

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/dashboard/components/charts.py`:

1. After the `MODEL_COLORS = [...]` line, add:

```python
# The market regime line every strategy charts (`core.regime.REGIME_LINE`): its own row on the Trades chart.
REGIME_PANE = "Regime"
REGIME_AXIS_LABELS = {1: "Bullish", 0: "Neutral", -1: "Bearish"}
REGIME_BAND_COLORS = {1: "#16a34a", 0: "#6b7280", -1: "#dc2626"}
_REGIME_LINE_COLOR = "#d1d4dc"
_REGIME_BAND_OPACITY = {1: 0.10, 2: 0.30}  # by row: faint behind the portfolio, stronger on the regime row
```

2. Add this helper above `trades_chart`:

```python
def _regime_runs(times: Sequence[Any], values: Sequence[float], end: Any) -> list[tuple[Any, Any, int]]:
    """`(start, end, regime)` for each run of equal consecutive values; the last run ends at `end`."""
    runs: list[list[Any]] = []
    for time, value in zip(times, (int(round(v)) for v in values)):
        if runs and runs[-1][2] == value:
            continue
        if runs:
            runs[-1][1] = time
        runs.append([time, end, value])
    return [(start, stop, value) for start, stop, value in runs]
```

3. In `trades_chart`, update the docstring's `indicators` entry:

```python
        indicators: Optional ``{plot_name: [series]}`` from :func:`load_indicator_lines`; each pane gets its own
            sub-row under the portfolio chart (they live on other scales), sharing its time axis. The
            ``"Regime"`` pane (market regime, -1/0/1) is drawn right under the portfolio chart as a step line,
            with one colored band per regime run on both rows.
```

4. Replace the block that builds the subplots:

```python
    panes = list((indicators or {}).items())
    fig = make_subplots(
        rows=1 + len(panes), cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, *[0.5 / len(panes)] * len(panes)] if panes else [1.0],
    )
```

with:

```python
    indicator_panes = dict(indicators or {})
    regime_series = indicator_panes.pop(REGIME_PANE, None)
    regime = regime_series[0] if regime_series else None
    panes = list(indicator_panes.items())
    row_heights = [1.0]
    if regime is not None or panes:
        row_heights = [0.5, *([0.12] if regime is not None else []), *[0.5 / len(panes)] * len(panes)]
    first_pane_row = 3 if regime is not None else 2
    fig = make_subplots(rows=len(row_heights), cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights)
```

5. Replace the indicator-pane loop header

```python
    for row, (pane, series) in enumerate(panes, start=2):
```

with

```python
    for row, (pane, series) in enumerate(panes, start=first_pane_row):
```

6. Right before the `# ── Indicator panes` comment, add the regime row:

```python
    # ── Market regime: a step line on its own row, and one band per run on both rows ──
    if regime is not None:
        regime_values = [int(round(v)) for v in regime["values"]]
        fig.add_trace(
            go.Scatter(
                x=regime["times"],
                y=regime_values,
                mode="lines",
                name=REGIME_PANE,
                line=dict(color=_REGIME_LINE_COLOR, width=1.5, shape="hv"),
                text=[REGIME_AXIS_LABELS.get(v, str(v)) for v in regime_values],
                hovertemplate="%{text}<extra>Regime</extra>",
                showlegend=False,
            ),
            row=2, col=1,
        )
        chart_end = max(pd.to_datetime([regime["times"][-1], *[v["time"] for v in values], *[t["time"] for t in trades]], utc=True))
        for start, stop, value in _regime_runs(regime["times"], regime["values"], chart_end):
            for row, opacity in _REGIME_BAND_OPACITY.items():
                fig.add_vrect(
                    x0=start, x1=stop, fillcolor=REGIME_BAND_COLORS.get(value, REGIME_BAND_COLORS[0]),
                    opacity=opacity, line_width=0, layer="below", row=row, col=1,
                )
        fig.update_yaxes(
            range=[-1.3, 1.3], tickvals=[-1, 0, 1], ticktext=[REGIME_AXIS_LABELS[v] for v in (-1, 0, 1)],
            title_text=REGIME_PANE, row=2, col=1,
        )
```

7. Replace the two layout lines that depend on the row count:

```python
    fig.update_xaxes(title_text="Date", row=1 + len(panes), col=1)
```

with

```python
    fig.update_xaxes(title_text="Date", row=len(row_heights), col=1)
```

and

```python
        height=450 + 200 * len(panes),
```

with

```python
        height=450 + 200 * len(panes) + (140 if regime is not None else 0),
```

- [ ] **Step 4: Run the chart tests**

Run: `uv run pytest tests/dashboard/test_trades_chart.py -q`
Expected: all pass.

If `test_each_regime_run_is_shaded_on_both_rows` fails only on the `(s.x0, s.x1)` comparison because plotly stored the bounds as `pandas.Timestamp`, keep the implementation and compare through `pd.Timestamp(...)` on both sides in the test:

```python
    import pandas as pd

    assert [(pd.Timestamp(s.x0), pd.Timestamp(s.x1)) for s in on_portfolio] == [
        (pd.Timestamp(T0), pd.Timestamp(T0 + timedelta(days=2))),
        (pd.Timestamp(T0 + timedelta(days=2)), pd.Timestamp(T0 + timedelta(days=3))),
        (pd.Timestamp(T0 + timedelta(days=3)), pd.Timestamp(T0 + timedelta(days=5))),
    ]
```

- [ ] **Step 5: Rename the ADX / RSI samples in the reader test**

`reader.load_indicator_lines` is generic and unchanged; only the sample names go. In `tests/dashboard/test_reader.py`, in `test_load_indicator_lines_groups_series_by_pane_and_drops_nothing_else`, replace the body from `for minutes, rsi in` to the last assertion with:

```python
    for minutes, value in ((5, "55"), (0, "50")):  # recorded out of order on purpose
        ledger.record_line(IndicatorLine(NOW + timedelta(minutes=minutes), "Fast", Decimal(value), "#a78bfa", "dashed", "Averages"))
    ledger.record_line(IndicatorLine(NOW, "Slow", Decimal("22"), "#f59e0b", "solid", "Averages"))
    ledger.record_line(IndicatorLine(NOW, "Regime", Decimal("-1"), "#d1d4dc", "solid", "Regime"))
    ledger.record_line(IndicatorLine(NOW, "SMA", Decimal("9"), None, "solid", "default_plot"))
    report.write_indicators(run_dir, ledger)

    panes = load_indicator_lines(_ref(run_dir))

    assert panes is not None and list(panes) == ["Averages", "Regime", "Indicators"]
    fast = next(line for line in panes["Averages"] if line["name"] == "Fast")
    assert fast["values"] == [50.0, 55.0] and fast["dash"] == "dash" and fast["color"] == "#a78bfa"
    assert panes["Regime"][0]["values"] == [-1.0]  # the regime line reaches the chart as its own pane
    assert panes["Indicators"][0]["color"]  # a line without a colour gets a palette one
```

- [ ] **Step 6: Run the dashboard tests and lint**

Run: `uv run pytest tests/dashboard -q && uv run ruff check src/trading_agent_framework/dashboard tests/dashboard`
Expected: all pass (including `test_dark_theme.py`), `All checks passed!`.

- [ ] **Step 7: Look at it once in the real dashboard**

Run a short backtest that trades, then open the dashboard:

```bash
uv run agent cross_momentum backtesting
uv run dashboard
```

Open the latest run, tab "Trades". Check: the regime row is directly under the portfolio chart, zooming one zooms the other, the bands line up with the step changes, the labels read Bearish / Neutral / Bullish, and no ADX / RSI / VIX pane appears. If the backtest cannot run in this environment (no data credentials), say so in the task report instead of claiming the check.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/dashboard/components/charts.py tests/dashboard/test_trades_chart.py tests/dashboard/test_reader.py
git commit -m "$(cat <<'EOF'
feat: dashboard shows the market regime under Trade Activity (Task 5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Documentation and final verification

**Files:**
- Modify: `CLAUDE.md` (architecture `core/` line; one new gotcha; vwap_pullback mention if it names the removed charts)
- Modify: `TODO.md` (line 32)

**Interfaces:**
- Consumes: the behavior built in Tasks 1-5.
- Produces: documentation only.

- [ ] **Step 1: Update the `core/` architecture line in `CLAUDE.md`**

Replace

```
`indicators.py` (`strategy.indicators.<pandas-ta name>(asset, ...)`, no cache).
```

with

```
`indicators.py` (`strategy.indicators.<pandas-ta name>(asset, ...)`, no cache), `regime.py` (**pure** market regime classifier: `RegimeParameters`, `classify_regime`).
```

- [ ] **Step 2: Add the gotcha to `CLAUDE.md`**

In "Key patterns / gotchas", add after the "No `time.sleep` / `datetime.now` in strategy or executor code" entry:

```
- **Every strategy gets a market regime by default, and nothing trades on it.** The executor calls `Strategy._refresh_regime()` once per session (before `before_market_opens`, also on a mid-session start, whatever the `sleeptime`): `core/regime.classify_regime` reads `regime_params.min_bars` (273) daily closes of `benchmark_symbol` and gives `strategy.regime` = 1 bullish / 0 neutral / -1 bearish (`None` until there is enough history). Trend is the close and SMA50 against SMA200; a 20-day realized volatility strictly above its own 80th percentile over 252 days caps a bullish read at neutral and never makes one bearish. It is logged in every mode and charted with `add_line("Regime", ...)` in backtests only (the dashboard's Trades chart draws that pane as its own row with bands: `charts.REGIME_PANE` must equal `regime.REGIME_LINE`). A failure keeps the previous value and never costs a session, and it is not a hook (no `on_bot_crash`). `Strategy.run_backtesting` raises `warmup_trading_days` to at least `regime_params.min_bars`: the runner's eager benchmark load would otherwise cache a frame too short for it. Using the regime to size or gate a strategy is a new design decision, not a default.
```

- [ ] **Step 3: Check `CLAUDE.md` and `README.md` for stale mentions**

Run: `grep -niE "\b(adx|vix)\b|VixSeries|_chart_indicators" CLAUDE.md README.md`
Expected: no output. If a line mentions the removed charting, delete that sentence.

- [ ] **Step 4: Strike the TODO item**

In `TODO.md`, replace

```
* REplace VIX, ADX etc with a "regime" indicator: 1 =bullish, neutral =0, bearish = -1
```

with

```
* ~~REplace VIX, ADX etc with a "regime" indicator: 1 =bullish, neutral =0, bearish = -1~~
```

- [ ] **Step 5: Final verification**

Run: `uv run pytest -q && uv run ruff check`
Expected: the full suite passes and `All checks passed!`. Report the actual pass count from the output.

Run: `grep -rniE "\b(adx|vix)\b" src tests --include=*.py | grep -v "indicators"`
Expected: no output (only the generic indicator tool and its tests may still name `rsi`).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md TODO.md README.md
git commit -m "$(cat <<'EOF'
docs: market regime in CLAUDE.md, TODO item done (Task 6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

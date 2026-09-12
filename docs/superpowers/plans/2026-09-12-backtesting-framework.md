# Backtesting Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the third trading mode, `backtesting`: an event-driven simulation that reuses the existing `StrategyExecutor` unchanged, fills orders against historical bars with a no-look-ahead guarantee, and reports performance (via vectorbt) in a format the to-be-migrated dashboard can read.

**Architecture:** A new `backtesting/` package provides a `BacktestClock`/`BacktestBroker` pair that plug into the existing `Strategy`/`StrategyExecutor` exactly where `AlpacaMarketClock`/`AlpacaBroker` do today, so strategy code is unmodified between paper and backtesting. A provider-agnostic `BacktestDataSource` ABC (Yahoo default, Alpaca also shipped) is the single no-look-ahead chokepoint. Fills are pure functions over OHLC bars (next-bar-open); a `Ledger` records fills/equity/indicator lines in `Decimal`; `metrics.py`/`report.py` convert to float64 only at the very end, writing parquet + JSON to `logs/<strategy>/backtesting/<ts>_backtesting/`.

**Tech Stack:** Python 3.14, pandas 3.0, vectorbt 1.1.0, pyarrow (parquet), yfinance (optional extra), existing stdlib `decimal`/`dataclasses`.

**Spec:** `docs/superpowers/specs/2026-09-12-backtesting-framework-design.md`

## Global Constraints

- Python `>=3.14`; `vectorbt` pinned to a version supporting `pandas>=3.0.3,numpy>=2.4.6` (1.1.0 verified compatible).
- Money is `Decimal` everywhere in `backtesting/broker.py`, `backtesting/fills.py`, `backtesting/ledger.py`. `backtesting/metrics.py` and `backtesting/report.py` are the one new float64 boundary (CLAUDE.md amendment, Task 22) — nowhere else in the new package converts to float.
- `orders.py`-style purity: `backtesting/fills.py` and `backtesting/data/base.py` are pure — no I/O, no state, no client instances.
- No raw SDK/pydantic exception may escape a public method; wrap in `BacktestError`/`BacktestDataError`/`OrderValidationError` (existing type, reused).
- `vectorbt`+`numba` and `yfinance` are imported **only inside method bodies**, never at module level. `backtesting/__init__.py` uses the lazy `__getattr__` pattern from `brokers/__init__.py` verbatim.
- No `time.sleep` / `datetime.now()` anywhere in the new package; all time comes from the injected `MarketClock`.
- Tests never touch the network. Hand-written fakes only (see `tests/fakes.py`, `tests/backtesting/fakes.py`) — never `MagicMock`.
- `BacktestDataSource.bars()` returns rows indexed by bar **close** time (not open/start) — this is the enforcement point for the no-look-ahead rule (`visible(bar) ⟺ bar_end <= cutoff` becomes `index <= cutoff`). This is specific to this subsystem and intentionally differs from the live Alpaca `Bars` convention (indexed at bar open) used by `brokers/alpaca/market_data.py`.
- All new source files start `from __future__ import annotations`, matching the rest of the codebase.
- Follow `ruff` config already in `pyproject.toml` (line-length 200, `E,F,I,UP,B`).

## Planning Notes (read before implementing)

Two points where this plan makes an implementation-level decision the spec left open, flagged here rather than silently:

1. **vectorbt's role is narrower than `Portfolio.from_orders` reconstructing the equity curve.** The spec's decision table says vectorbt "rebuilds equity/cash/positions from the recorded fill log." In practice, `BacktestBroker` already computes exact, `Decimal`-precise equity/cash/positions every clock advance (that's the ledger's whole job — see `backtesting/ledger.py`), and `Portfolio.from_orders` reconstructing a *second*, independent equity series from raw fills for a multi-asset strategy is fragile (it wants one price timeline; a multi-symbol portfolio needs careful column alignment) and would risk disagreeing with the broker's own numbers. This plan uses the **ledger's own equity samples** as the source of `equity.parquet`, and vectorbt purely for the *statistics* (`returns.vbt.returns(...)` accessor) computed from that equity series' returns — which is what the spec's `metrics.py` module description already says ("vectorbt reporting... the returns accessor supplies Sharpe, Sortino, ..."). This is a narrowing, not a contradiction: vectorbt still computes every requested statistic.
2. **`BacktestDataSource.load()` is a pre-warming optimisation, not a precondition.** The spec's ABC sketch implies `load()` is called once per run for the whole universe, but a strategy's asset universe is often decided dynamically inside `on_trading_iteration()` — the runner cannot know it in advance. `bars()` on both concrete sources therefore fetches lazily (and caches in memory) the first time it's asked about an asset it hasn't seen, using a `start`/`end` window fixed at the source's construction. `load()` remains useful for a known, fixed asset (the runner uses it once for the benchmark).

---

### Task 1: Clock-provided `max_wait_slice`

**Files:**
- Modify: `src/trading_agent_framework/utils/clock.py`
- Modify: `src/trading_agent_framework/core/executor.py:35,104`
- Test: `tests/test_clock.py`, `tests/core/test_executor.py`

**Interfaces:**
- Produces: `MarketClock.max_wait_slice: float` (class attribute, default `60.0`), read by `executor.py`'s wait loop.

- [ ] **Step 1: Write the failing test for the new attribute**

Add to `tests/test_clock.py`:

```python
def test_market_clock_default_max_wait_slice_is_60_seconds() -> None:
    from trading_agent_framework.utils.clock import MarketClock

    class ConcreteClock(MarketClock):
        def next_session(self):
            return None

    assert ConcreteClock().max_wait_slice == 60.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_clock.py::test_market_clock_default_max_wait_slice_is_60_seconds -v`
Expected: FAIL — `AttributeError: max_wait_slice`

- [ ] **Step 3: Add the attribute to `MarketClock`**

In `src/trading_agent_framework/utils/clock.py`, inside `class MarketClock(ABC):` right after `tz: ZoneInfo = MARKET_TZ`:

```python
    # Correction cadence for an open-ended wait, e.g. after a suspended laptop.
    # A simulated clock (BacktestClock) overrides this to math.inf: nothing needs
    # correcting when time never actually passes.
    max_wait_slice: float = 60.0
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/test_clock.py::test_market_clock_default_max_wait_slice_is_60_seconds -v`
Expected: PASS

- [ ] **Step 5: Write the failing executor test for the wiring**

Add to `tests/core/test_executor.py` (use the existing `FakeClock`/`FakeBroker` fixtures already imported there):

```python
def test_wait_until_uses_the_clocks_max_wait_slice() -> None:
    clock = FakeClock(et(2026, 1, 5, 9, 0))
    clock.max_wait_slice = 10.0  # ty: ignore[invalid-assignment]
    strategy = Hello(FakeBroker(clock))
    executor = strategy.executor

    executor.wait_until(clock.now() + timedelta(seconds=25))

    # 10s, 10s, 5s -- never a bare 60s slice from the old module constant.
    assert clock.waits == [10.0, 10.0, 5.0]
```

Check the exact fixture names already in `tests/core/test_executor.py` (likely `Hello`, `FakeClock`, `FakeBroker`, `et`, `timedelta` imports) and reuse them rather than reinventing; adjust the assertion to whatever helper strategy class that file already defines for a no-op `on_trading_iteration`.

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/core/test_executor.py::test_wait_until_uses_the_clocks_max_wait_slice -v`
Expected: FAIL — waits recorded as `[25.0]` or similar (still using the old `MAX_WAIT_SLICE_SECONDS` constant, unaffected by the fake clock's override since it's currently ignored).

- [ ] **Step 7: Wire the executor to the clock's attribute**

In `src/trading_agent_framework/core/executor.py`:

Delete the module constant:
```python
MAX_WAIT_SLICE_SECONDS = 60.0
```

In `wait_until`, change:
```python
            clock.wait(min(remaining, MAX_WAIT_SLICE_SECONDS), self._wake)
```
to:
```python
            clock.wait(min(remaining, clock.max_wait_slice), self._wake)
```

- [ ] **Step 8: Run both tests, then the full executor/clock suites**

Run: `uv run pytest tests/core/test_executor.py tests/test_clock.py -v`
Expected: PASS, including every pre-existing test in both files (live behaviour is unchanged since `MarketClock.max_wait_slice` defaults to `60.0`).

- [ ] **Step 9: Run the full test suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/trading_agent_framework/utils/clock.py src/trading_agent_framework/core/executor.py tests/test_clock.py tests/core/test_executor.py
git commit -m "Move executor's wait-slice constant onto MarketClock

Replaces the module-level MAX_WAIT_SLICE_SECONDS with MarketClock.max_wait_slice
(default 60.0, live behaviour unchanged), so a future BacktestClock can override
it to math.inf and skip ~2M no-op wait slices in a 10-year backtest."
```

---

### Task 2: New error types for the backtesting subsystem

**Files:**
- Modify: `src/trading_agent_framework/utils/errors.py`
- Test: `tests/test_errors.py` (create if it does not already exist — check first: `ls tests/test_errors.py` or `grep -rl "class TestErrors\|TradingFrameworkError" tests/`)

**Interfaces:**
- Produces: `BacktestError(TradingFrameworkError)`, `BacktestDataError(BacktestError)` — used by every later `backtesting/` task.

- [ ] **Step 1: Check for an existing errors test file**

Run: `find tests -iname "*error*"`

If one exists, add the new tests there instead of creating `tests/test_errors.py`; otherwise create it fresh with the structure below.

- [ ] **Step 2: Write the failing tests**

```python
from __future__ import annotations

from trading_agent_framework.utils.errors import (
    BacktestDataError,
    BacktestError,
    TradingFrameworkError,
)


def test_backtest_error_is_a_trading_framework_error() -> None:
    assert issubclass(BacktestError, TradingFrameworkError)


def test_backtest_data_error_is_a_backtest_error() -> None:
    assert issubclass(BacktestDataError, BacktestError)
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_errors.py -v` (or wherever Step 1 placed it)
Expected: FAIL — `ImportError: cannot import name 'BacktestError'`

- [ ] **Step 4: Add the error types**

In `src/trading_agent_framework/utils/errors.py`, after `AgentError`:

```python
class BacktestError(TradingFrameworkError):
    """Raised when the backtesting simulation cannot proceed (never a raw exception)."""


class BacktestDataError(BacktestError):
    """Raised when a BacktestDataSource cannot fetch, cache, or parse historical data."""
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/utils/errors.py tests/test_errors.py
git commit -m "Add BacktestError and BacktestDataError"
```

---

### Task 3: `BacktestDataSource` ABC and the shared test fake

**Files:**
- Create: `src/trading_agent_framework/backtesting/__init__.py` (empty package marker only for now — Task 16 fills in the lazy exports)
- Create: `src/trading_agent_framework/backtesting/data/__init__.py` (empty)
- Create: `src/trading_agent_framework/backtesting/data/base.py`
- Create: `tests/backtesting/__init__.py` (empty)
- Create: `tests/backtesting/fakes.py`
- Test: `tests/backtesting/data/test_base.py`

**Interfaces:**
- Produces: `BacktestDataSource` (ABC: `load`, `bars`, `sessions`), `FULL_HISTORY = 10_000_000` — both imported by every later `backtesting/` task. `FakeBacktestDataSource` — imported by every later test task.

- [ ] **Step 1: Create the empty package markers**

```bash
mkdir -p src/trading_agent_framework/backtesting/data tests/backtesting/data
touch src/trading_agent_framework/backtesting/__init__.py
touch src/trading_agent_framework/backtesting/data/__init__.py
touch tests/backtesting/__init__.py
touch tests/backtesting/data/__init__.py
```

- [ ] **Step 2: Write the failing test**

`tests/backtesting/data/test_base.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.backtesting.data.base import BacktestDataSource


def test_backtest_data_source_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BacktestDataSource()  # ty: ignore[abstract-class-instantiated]
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/backtesting/data/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_agent_framework.backtesting.data.base'`

- [ ] **Step 4: Write `backtesting/data/base.py`**

```python
"""Provider-agnostic backtest data source: the seam a future IBKR implementation
plugs into, and the only thing `BacktestBroker` depends on for prices and sessions.

`bars()` is the no-look-ahead chokepoint (design spec, section 5): its `cutoff`
parameter is always `clock.now()`, and every returned row's index value is that
bar's CLOSE timestamp (not its open/start -- a deliberate convention specific to
this subsystem, distinct from the live Alpaca `Bars` convention used by
`brokers/alpaca/market_data.py`, which indexes at bar open). A row at index `t`
means "this bar was fully formed and knowable as of `t`"; filtering `index <=
cutoff` is then sufficient and correct.

Implementations fetch lazily: `bars()` must return data for any asset even if
`load()` was never called for it -- `load()` is a pre-warming optimisation (e.g.
the runner uses it once for the benchmark), not a precondition. This lets a
strategy trade an asset picked dynamically inside `on_trading_iteration()`
without the runner knowing the universe in advance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession

FULL_HISTORY = 10_000_000  # "return every bar you have" sentinel for bars(..., length=...)


class BacktestDataSource(ABC):
    """Bars and sessions for a backtest run. Never imports a broker; `BacktestBroker`
    depends on this, never the reverse."""

    name: ClassVar[str]

    @abstractmethod
    def load(
        self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str
    ) -> None:
        """Pre-fetch and cache `assets` for [start, end]. Optional to call; `bars()`
        fetches lazily for any asset it hasn't seen."""

    @abstractmethod
    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        """The last `length` bars closed at or before `cutoff`, oldest first, indexed
        by bar CLOSE time. None if the asset has no data at all."""

    @abstractmethod
    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        """Trading sessions in [start, end], early closes included where known."""
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/backtesting/data/test_base.py -v`
Expected: PASS

- [ ] **Step 6: Write `tests/backtesting/fakes.py`**

This is a test-only fixture, not a TDD deliverable itself, but write a quick smoke test for it (Step 7) since every later task depends on it behaving correctly.

```python
"""In-memory BacktestDataSource fake shared by every backtesting test module."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession


class FakeBacktestDataSource(BacktestDataSource):
    """Bars set directly by the test via `set_bars`; must already be indexed by bar
    CLOSE timestamp (tz-aware), matching the real contract in `data/base.py`."""

    name = "fake"

    def __init__(self) -> None:
        self._frames: dict[Asset, pd.DataFrame] = {}
        self._sessions: list[MarketSession] = []
        self.load_calls: list[tuple[Asset, ...]] = []
        self.bars_calls: list[tuple[Asset, datetime, int, str]] = []

    def set_bars(self, asset: Asset, df: pd.DataFrame) -> None:
        self._frames[asset] = df.sort_index()

    def set_sessions(self, sessions: list[MarketSession]) -> None:
        self._sessions = sessions

    def load(self, assets, start, end, timestep) -> None:
        self.load_calls.append(tuple(assets))

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        self.bars_calls.append((asset, cutoff, length, timestep))
        df = self._frames.get(asset)
        if df is None:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return [s for s in self._sessions if s.open >= start and s.close <= end]


def make_close_indexed_frame(
    closes: list[float], *, start: datetime, freq: str = "1D"
) -> pd.DataFrame:
    """An OHLCV frame indexed by bar CLOSE (see `data/base.py`'s convention).
    high = close + 1, low = close - 1, matching `tests/fakes.py:make_bars_frame`'s shape."""
    index = pd.date_range(start, periods=len(closes), freq=freq, name="timestamp")
    close = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=index,
    )
```

- [ ] **Step 7: Write and run a smoke test for the fake**

Add to `tests/backtesting/data/test_base.py`:

```python
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame
from trading_agent_framework.entities.asset import Asset


def test_fake_data_source_honours_the_cutoff_and_length_contract() -> None:
    source = FakeBacktestDataSource()
    asset = Asset("AAPL")
    df = make_close_indexed_frame([100.0, 101.0, 102.0], start=datetime(2026, 1, 5, 16, tzinfo=UTC))
    source.set_bars(asset, df)

    # Cutoff before the first bar closes: nothing visible.
    assert source.bars(asset, df.index[0] - pd.Timedelta(seconds=1), 5, "day") is None

    # Cutoff exactly at the first bar's close: exactly that one bar.
    result = source.bars(asset, df.index[0], 5, "day")
    assert result is not None
    assert len(result.df) == 1
    assert result.df["close"].iloc[-1] == 100.0

    # Cutoff after all three: all three, oldest first, length respected.
    result = source.bars(asset, df.index[-1], 2, "day")
    assert result is not None
    assert list(result.df["close"]) == [101.0, 102.0]
```

Add the needed imports (`from datetime import UTC, datetime`, `import pandas as pd`) at the top of the test file.

Run: `uv run pytest tests/backtesting/data/test_base.py -v`
Expected: PASS

- [ ] **Step 8: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/backtesting tests/backtesting
git commit -m "Add BacktestDataSource ABC and the shared test fake

The no-look-ahead chokepoint: bars() is indexed by bar CLOSE time, so
'index <= cutoff' is the whole visibility rule (design spec, section 5)."
```

---

### Task 4: `backtesting/fills.py` — pure OHLC fill rules

**Files:**
- Create: `src/trading_agent_framework/backtesting/fills.py`
- Test: `tests/backtesting/test_fills.py`

**Interfaces:**
- Consumes: `OrderSide`, `OrderType` from `trading_agent_framework.entities.enums` (Task-independent, already exist).
- Produces: `Bar` (frozen dataclass: `open, high, low, close: Decimal`), `FillResult` (frozen dataclass: `price: Decimal`), `evaluate_fill(*, order_type, side, bar, limit_price=None, stop_price=None, stop_limit_price=None) -> FillResult | None`, `apply_commission_and_slippage(price, side, *, commission, slippage) -> tuple[Decimal, Decimal]` — all consumed by `backtesting/broker.py` (Tasks 7-8).

- [ ] **Step 1: Write the failing tests (table-driven)**

`tests/backtesting/test_fills.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.backtesting.fills import Bar, apply_commission_and_slippage, evaluate_fill
from trading_agent_framework.entities.enums import OrderSide, OrderType

D = Decimal
BAR = Bar(open=D(100), high=D(105), low=D(95), close=D(102))


def test_market_order_always_fills_at_open() -> None:
    result = evaluate_fill(order_type=OrderType.MARKET, side=OrderSide.BUY, bar=BAR)
    assert result is not None
    assert result.price == D(100)


@pytest.mark.parametrize(
    "side,limit_price,expected",
    [
        (OrderSide.BUY, D(96), D(96)),  # touches low (95), doesn't gap through open (100) -> limit price
        (OrderSide.BUY, D(101), D(100)),  # open already satisfies (100 <= 101) -> gapped, better price
        (OrderSide.SELL, D(104), D(104)),  # touches high (105), doesn't gap through open -> limit price
        (OrderSide.SELL, D(99), D(100)),  # open already satisfies (100 >= 99) -> gapped, better price
    ],
)
def test_limit_fills_when_touched_pessimistically(
    side: OrderSide, limit_price: Decimal, expected: Decimal
) -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=side, bar=BAR, limit_price=limit_price)
    assert result is not None
    assert result.price == expected


def test_buy_limit_does_not_fill_when_low_never_touches_it() -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.BUY, bar=BAR, limit_price=D(90))
    assert result is None


def test_sell_limit_does_not_fill_when_high_never_touches_it() -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.SELL, bar=BAR, limit_price=D(110))
    assert result is None


@pytest.mark.parametrize(
    "side,stop_price,expected",
    [
        (OrderSide.BUY, D(103), D(103)),  # touches high (105), open (100) below stop -> stop price
        (OrderSide.BUY, D(99), D(100)),  # open already above stop -> gapped, worse price (pessimistic)
        (OrderSide.SELL, D(97), D(97)),  # touches low (95), open above stop -> stop price
        (OrderSide.SELL, D(101), D(100)),  # open already below stop -> gapped, worse price
    ],
)
def test_stop_triggers_pessimistically(side: OrderSide, stop_price: Decimal, expected: Decimal) -> None:
    result = evaluate_fill(order_type=OrderType.STOP, side=side, bar=BAR, stop_price=stop_price)
    assert result is not None
    assert result.price == expected


def test_buy_stop_does_not_trigger_when_high_never_reaches_it() -> None:
    result = evaluate_fill(order_type=OrderType.STOP, side=OrderSide.BUY, bar=BAR, stop_price=D(110))
    assert result is None


def test_stop_limit_buy_needs_both_the_stop_trigger_and_the_limit_touch() -> None:
    # Stop at 103 (triggers, high=105), limit at 96 (also touched, low=95): fills.
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.BUY, bar=BAR,
        stop_price=D(103), stop_limit_price=D(96),
    )
    assert result is not None
    assert result.price == D(96)

    # Stop triggers (105 >= 103) but the limit (99) is never touched by the low (95 <= 99 is TRUE,
    # so use a limit that is NOT touched: low=95 means anything >= 95 IS touched; pick a limit
    # below the low to prove the "not touched" branch).
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.BUY, bar=BAR,
        stop_price=D(103), stop_limit_price=D(90),
    )
    assert result is None  # low (95) never reaches down to 90


def test_stop_limit_sell_needs_both_the_stop_trigger_and_the_limit_touch() -> None:
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.SELL, bar=BAR,
        stop_price=D(97), stop_limit_price=D(104),
    )
    assert result is not None
    assert result.price == D(104)

    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.SELL, bar=BAR,
        stop_price=D(97), stop_limit_price=D(110),
    )
    assert result is None  # high (105) never reaches up to 110


def test_trailing_stop_is_not_supported() -> None:
    with pytest.raises(ValueError, match="TRAIL"):
        evaluate_fill(order_type=OrderType.TRAIL, side=OrderSide.BUY, bar=BAR)


def test_limit_order_without_a_limit_price_raises() -> None:
    with pytest.raises(ValueError, match="limit_price"):
        evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.BUY, bar=BAR)


@pytest.mark.parametrize(
    "side,commission,slippage,expected_price,expected_commission",
    [
        (OrderSide.BUY, D(0), D(0), D(100), D(0)),
        (OrderSide.BUY, D("0.001"), D(0), D(100), D("0.1")),  # 10bps of 100
        (OrderSide.BUY, D(0), D("0.01"), D(101), D(0)),  # buys pay more
        (OrderSide.SELL, D(0), D("0.01"), D(99), D(0)),  # sells receive less
    ],
)
def test_commission_and_slippage(
    side: OrderSide, commission: Decimal, slippage: Decimal,
    expected_price: Decimal, expected_commission: Decimal,
) -> None:
    price, commission_per_share = apply_commission_and_slippage(
        D(100), side, commission=commission, slippage=slippage
    )
    assert price == expected_price
    assert commission_per_share == expected_commission
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_fills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_agent_framework.backtesting.fills'`

- [ ] **Step 3: Write `backtesting/fills.py`**

```python
"""Pure OHLC fill rules for the backtest broker: no I/O, no state, no client instances.

Fills evaluate one already-closed bar against one order's request. `BacktestBroker`
only ever calls this against a bar that closed strictly after the order was submitted
(next-bar-open) -- this module has no notion of "which bar" beyond the one it's given.

Fill rules, all "ties resolved pessimistically" (design spec, section 2): when the
bar's range only *touches* the trigger price, the trader gets exactly that price, not
a better one; when the bar gaps clean through it, the trader gets the real, unavoidable
open price -- which is worse for them in every case below (that's what makes it
pessimistic rather than optimistic).

- MARKET: always fills, at the bar's open.
- LIMIT buy: fills iff low <= limit_price, at min(open, limit_price).
- LIMIT sell: fills iff high >= limit_price, at max(open, limit_price).
- STOP buy: triggers iff high >= stop_price, at max(open, stop_price).
- STOP sell: triggers iff low <= stop_price, at min(open, stop_price).
- STOP_LIMIT buy: triggers iff high >= stop_price AND low <= stop_limit_price,
  at min(open, stop_limit_price).
- STOP_LIMIT sell: triggers iff low <= stop_price AND high >= stop_limit_price,
  at max(open, stop_limit_price).
- TRAIL: not supported (needs a trailing reference price tracked across bars,
  which is out of scope -- design spec, section 1.3); raises ValueError.

Commission is a fraction of trade notional (e.g. Decimal("0.001") = 10bps), returned
as a per-share rate -- multiply by fill quantity for the total dollar cost. Slippage
is a fraction applied against the trader: buys pay price * (1 + slippage), sells
receive price * (1 - slippage).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class Bar:
    """One already-closed bar's OHLC, as Decimal."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True, slots=True)
class FillResult:
    """Where an order filled, before commission/slippage."""

    price: Decimal


def evaluate_fill(
    *,
    order_type: OrderType,
    side: OrderSide,
    bar: Bar,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
    stop_limit_price: Decimal | None = None,
) -> FillResult | None:
    """The raw fill price for one order against one bar, or None if it doesn't fill this bar."""
    if order_type is OrderType.MARKET:
        return FillResult(price=bar.open)
    if order_type is OrderType.LIMIT:
        return _limit_fill(side, bar, _require(limit_price, "limit_price"))
    if order_type is OrderType.STOP:
        return _stop_fill(side, bar, _require(stop_price, "stop_price"))
    if order_type is OrderType.STOP_LIMIT:
        return _stop_limit_fill(
            side, bar,
            _require(stop_price, "stop_price"),
            _require(stop_limit_price, "stop_limit_price"),
        )
    raise ValueError(f"backtesting does not support order_type={order_type} (TRAIL)")


def _require(value: Decimal | None, name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{name} is required for this order_type")
    return value


def _limit_fill(side: OrderSide, bar: Bar, limit_price: Decimal) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.low > limit_price:
            return None
        return FillResult(price=min(bar.open, limit_price))
    if bar.high < limit_price:
        return None
    return FillResult(price=max(bar.open, limit_price))


def _stop_fill(side: OrderSide, bar: Bar, stop_price: Decimal) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.high < stop_price:
            return None
        return FillResult(price=max(bar.open, stop_price))
    if bar.low > stop_price:
        return None
    return FillResult(price=min(bar.open, stop_price))


def _stop_limit_fill(
    side: OrderSide, bar: Bar, stop_price: Decimal, stop_limit_price: Decimal
) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.high < stop_price or bar.low > stop_limit_price:
            return None
        return FillResult(price=min(bar.open, stop_limit_price))
    if bar.low > stop_price or bar.high < stop_limit_price:
        return None
    return FillResult(price=max(bar.open, stop_limit_price))


def apply_commission_and_slippage(
    price: Decimal, side: OrderSide, *, commission: Decimal, slippage: Decimal
) -> tuple[Decimal, Decimal]:
    """Return (execution_price, commission_per_share). The caller multiplies
    commission_per_share by the fill quantity for the total dollar cost."""
    execution_price = price * (1 + slippage) if side is OrderSide.BUY else price * (1 - slippage)
    commission_per_share = execution_price * commission
    return execution_price, commission_per_share
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_fills.py -v`
Expected: PASS (all parametrized cases)

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/fills.py tests/backtesting/test_fills.py
git commit -m "Add pure OHLC fill rules (backtesting/fills.py)

Market/limit/stop/stop-limit against a single closed bar, ties resolved
pessimistically. No I/O, no state -- BacktestBroker (later tasks) is the
only caller."
```

---

### Task 5: `backtesting/ledger.py` — fills, equity, indicator lines

**Files:**
- Create: `src/trading_agent_framework/backtesting/ledger.py`
- Test: `tests/backtesting/test_ledger.py`

**Interfaces:**
- Consumes: nothing new (stdlib + `entities.enums`).
- Produces: `FillRecord`, `EquitySample`, `IndicatorLine` (frozen dataclasses), `Ledger` (`.fills: list[FillRecord]`, `.equity: list[EquitySample]`, `.lines: list[IndicatorLine]`, `.record_fill/.record_equity/.record_line`) — consumed by `backtesting/broker.py` (Tasks 7-8), `backtesting/report.py` (Task 14), `Strategy.add_line` (Task 17).

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_ledger.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)


def test_ledger_starts_empty() -> None:
    ledger = Ledger()
    assert ledger.fills == []
    assert ledger.equity == []
    assert ledger.lines == []


def test_ledger_records_a_fill() -> None:
    ledger = Ledger()
    record = FillRecord(
        time=NOW, identifier="abc", symbol="AAPL", side=OrderSide.BUY,
        order_type=OrderType.MARKET, quantity=Decimal(10), filled_quantity=Decimal(10),
        price=Decimal("150.00"), trade_cost=Decimal("0.15"), trade_slippage=Decimal("0.05"),
    )

    ledger.record_fill(record)

    assert ledger.fills == [record]
    assert ledger.fills[0].status == "fill"  # default


def test_ledger_records_an_equity_sample() -> None:
    ledger = Ledger()
    sample = EquitySample(
        time=NOW, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)
    )

    ledger.record_equity(sample)

    assert ledger.equity == [sample]


def test_ledger_records_an_indicator_line() -> None:
    ledger = Ledger()
    line = IndicatorLine(
        time=NOW, name="sma_200", value=Decimal("148.5"), color=None,
        style="solid", plot_name="default_plot",
    )

    ledger.record_line(line)

    assert ledger.lines == [line]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/ledger.py`**

```python
"""Records what happened during a backtest run: fills, per-bar equity samples, and
strategy-added indicator lines. Decimal throughout -- the codebase's third float
boundary begins at `backtesting/metrics.py` and `backtesting/report.py`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class FillRecord:
    time: datetime
    identifier: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal
    price: Decimal
    trade_cost: Decimal
    trade_slippage: Decimal
    status: str = "fill"
    event_kind: str = "trade"


@dataclass(frozen=True, slots=True)
class EquitySample:
    time: datetime
    portfolio_value: Decimal
    cash: Decimal
    positions_value: Decimal


@dataclass(frozen=True, slots=True)
class IndicatorLine:
    time: datetime
    name: str
    value: Decimal
    color: str | None
    style: str
    plot_name: str


class Ledger:
    """Accumulates fills, equity samples and indicator lines in memory during a run."""

    def __init__(self) -> None:
        self.fills: list[FillRecord] = []
        self.equity: list[EquitySample] = []
        self.lines: list[IndicatorLine] = []

    def record_fill(self, record: FillRecord) -> None:
        self.fills.append(record)

    def record_equity(self, sample: EquitySample) -> None:
        self.equity.append(sample)

    def record_line(self, line: IndicatorLine) -> None:
        self.lines.append(line)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_ledger.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/ledger.py tests/backtesting/test_ledger.py
git commit -m "Add Ledger: fills, equity samples, indicator lines (Decimal)"
```

---

### Task 6: `backtesting/clock.py` — `BacktestClock`

**Files:**
- Create: `src/trading_agent_framework/backtesting/clock.py`
- Test: `tests/backtesting/test_clock.py`

**Interfaces:**
- Consumes: `MarketClock`, `MarketSession` from `trading_agent_framework.utils.clock` (Task 1's `max_wait_slice` attribute).
- Produces: `BacktestClock(start, sessions, on_advance=None)` — `.now()`, `.wait(seconds, wake)`, `.next_session()`, `.max_wait_slice == math.inf`, public `.on_advance: Callable[[datetime, datetime], None] | None` settable after construction. Consumed by `backtesting/broker.py` (Task 7) and `backtesting/runner.py` (Task 15).

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/test_clock.py`:

```python
from __future__ import annotations

import math
import threading
from datetime import date, timedelta

from tests.fakes import et, make_session, weekday_sessions
from trading_agent_framework.backtesting.clock import BacktestClock


def test_max_wait_slice_is_infinite() -> None:
    clock = BacktestClock(start=et(2026, 1, 5), sessions=[])
    assert clock.max_wait_slice == math.inf


def test_now_starts_at_the_given_start_time() -> None:
    start = et(2026, 1, 5, 9, 30)
    clock = BacktestClock(start=start, sessions=[])
    assert clock.now() == start


def test_wait_advances_now_by_exactly_the_requested_seconds() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.wait(3600, threading.Event())
    assert clock.now() == et(2026, 1, 5, 10, 30)


def test_wait_does_nothing_when_seconds_is_not_positive() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.wait(0, threading.Event())
    assert clock.now() == et(2026, 1, 5, 9, 30)


def test_wait_does_nothing_when_wake_is_already_set() -> None:
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    wake = threading.Event()
    wake.set()
    clock.wait(3600, wake)
    assert clock.now() == et(2026, 1, 5, 9, 30)


def test_wait_calls_on_advance_with_previous_and_new_now() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(
        start=et(2026, 1, 5, 9, 30), sessions=[], on_advance=lambda prev, new: calls.append((prev, new))
    )
    clock.wait(60, threading.Event())
    assert calls == [(et(2026, 1, 5, 9, 30), et(2026, 1, 5, 9, 31))]


def test_wait_does_not_call_on_advance_when_it_does_not_move_time() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(
        start=et(2026, 1, 5, 9, 30), sessions=[], on_advance=lambda prev, new: calls.append((prev, new))
    )
    wake = threading.Event()
    wake.set()
    clock.wait(60, wake)
    assert calls == []


def test_on_advance_can_be_set_after_construction() -> None:
    calls: list[tuple] = []
    clock = BacktestClock(start=et(2026, 1, 5, 9, 30), sessions=[])
    clock.on_advance = lambda prev, new: calls.append((prev, new))
    clock.wait(1, threading.Event())
    assert len(calls) == 1


def test_next_session_returns_the_first_session_closing_after_now() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 3)
    clock = BacktestClock(start=et(2026, 1, 5, 12, 0), sessions=sessions)
    assert clock.next_session() == sessions[0]


def test_next_session_returns_none_past_the_last_session() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 1)
    clock = BacktestClock(start=et(2026, 1, 5, 20, 0), sessions=sessions)  # after the one session closed
    assert clock.next_session() is None


def test_next_session_skips_sessions_already_closed() -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 3)
    clock = BacktestClock(start=sessions[0].close, sessions=sessions)
    assert clock.next_session() == sessions[1]


def test_next_session_respects_early_closes() -> None:
    early = make_session(date(2026, 11, 27), open_at=et(2026, 1, 1, 9, 30).time(), close_at=et(2026, 1, 1, 13, 0).time())
    clock = BacktestClock(start=et(2026, 11, 27, 12, 0), sessions=[early])
    assert clock.next_session() == early
```

Check `tests/fakes.py` for the exact signature of `make_session`/`weekday_sessions`/`et` before writing this (already read in this plan's research phase: `et(year, month, day, hour=0, minute=0, second=0) -> datetime`; `make_session(day: date, open_at: time = time(9,30), close_at: time = time(16,0)) -> MarketSession`; `weekday_sessions(first_day: date, count: int) -> list[MarketSession]`) and adjust the early-close test's `open_at`/`close_at` arguments to `time` objects, not `datetime`s, e.g. `time(9, 30)` / `time(13, 0)` — fix the import accordingly (`from datetime import time`).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_clock.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/clock.py`**

```python
"""`MarketClock` driving simulated time from injected sessions.

`wait()` jumps straight to the target time -- there is no real sleeping, so
`max_wait_slice` is infinite (Task 1's seam). Every successful jump calls
`on_advance(previous_now, new_now)` so `BacktestBroker` can process fills and
sample equity; `on_advance` is a plain settable attribute (not a constructor-only
argument) because the broker that needs to receive it is constructed *with* this
clock, creating a circular dependency the runner breaks by wiring it after both exist.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from trading_agent_framework.utils.clock import MarketClock, MarketSession


class BacktestClock(MarketClock):
    max_wait_slice: float = math.inf

    def __init__(
        self,
        start: datetime,
        sessions: Sequence[MarketSession],
        on_advance: Callable[[datetime, datetime], None] | None = None,
    ) -> None:
        self._now = start
        self._sessions = list(sessions)
        self.on_advance = on_advance

    def now(self) -> datetime:
        return self._now

    def wait(self, seconds: float, wake: threading.Event) -> None:
        if seconds <= 0 or wake.is_set():
            return
        previous = self._now
        self._now = previous + timedelta(seconds=seconds)
        if self.on_advance is not None:
            self.on_advance(previous, self._now)

    def next_session(self) -> MarketSession | None:
        return next((s for s in self._sessions if s.close > self._now), None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_clock.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/clock.py tests/backtesting/test_clock.py
git commit -m "Add BacktestClock: simulated time with an on_advance hook"
```

---

### Task 7: `backtesting/broker.py` part 1 — order lifecycle, account, market data reads

**Files:**
- Create: `src/trading_agent_framework/backtesting/broker.py`
- Test: `tests/backtesting/test_broker_orders.py`

**Interfaces:**
- Consumes: `BacktestDataSource`, `FULL_HISTORY` (Task 3); `Ledger` (Task 5); `MarketClock` (existing); `Broker` ABC (existing, `brokers/base.py`); `OrderTracker` (existing).
- Produces: `BacktestBroker(strategy_name, *, data_source, clock, budget, timestep="day", commission=Decimal(0), slippage=Decimal(0), tracker=None)` implementing every `Broker` abstract method. `.ledger: Ledger` (public, read by `report.py` in Task 14 and `Strategy.add_line` in Task 17). Order submission in this task tracks orders as `NEW` and queues them in `._pending`, but does not yet fill them — Task 8 adds `process_pending`/`on_advance`.

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/test_broker_orders.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

AAPL = Asset("AAPL")
NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)


def _broker(budget: Decimal = Decimal(10000)) -> BacktestBroker:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=NOW)
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=NOW, sessions=[])
    return BacktestBroker("momentum", data_source=source, clock=clock, budget=budget)


def test_submitting_a_market_order_tracks_it_as_new_and_pending() -> None:
    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))

    result = broker.submit_order(order)

    assert result.status is OrderStatus.NEW
    assert result.client_order_id == f"momentum:{result.identifier}"
    assert result in broker.tracker.new.snapshot()


def test_submitting_a_notional_order_raises() -> None:
    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, notional=Decimal(1000))
    with pytest.raises(OrderValidationError, match="notional"):
        broker.submit_order(order)


def test_cancel_order_removes_it_from_pending_and_the_tracker() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    broker.cancel_order(order)

    assert order.status is OrderStatus.CANCELED
    assert order not in broker.tracker.new.snapshot()
    assert order in broker.tracker.canceled.snapshot()


def test_modify_order_updates_the_pending_orders_prices() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(
            strategy_name="momentum", asset=AAPL, side=OrderSide.BUY,
            quantity=Decimal(10), limit_price=Decimal(140),
        )
    )

    replacement = broker.modify_order(order, limit_price=Decimal(145))

    assert replacement.limit_price == Decimal(145)


def test_modify_order_on_an_unpending_order_raises() -> None:
    from trading_agent_framework.utils.errors import BacktestError

    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    with pytest.raises(BacktestError):
        broker.modify_order(order, limit_price=Decimal(100))


def test_sync_open_orders_returns_nothing_for_a_fresh_backtest() -> None:
    assert _broker().sync_open_orders() == []


def test_pull_orders_and_pull_order_read_from_the_tracker() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    assert broker.pull_order(order.identifier) == order
    assert order in broker.pull_orders()
    assert broker.pull_order("does-not-exist") is None


def test_get_last_price_returns_the_latest_visible_close() -> None:
    broker = _broker()
    assert broker.get_last_price(AAPL) == Decimal("150.0")


def test_get_last_price_returns_none_for_an_asset_with_no_data() -> None:
    broker = _broker()
    assert broker.get_last_price(Asset("MISSING")) is None


def test_get_quote_synthesises_bid_and_ask_from_the_last_close() -> None:
    broker = _broker()
    quote = broker.get_quote(AAPL)
    assert quote is not None
    assert quote.bid == quote.ask == Decimal("150.0")


def test_get_bars_delegates_to_the_data_source() -> None:
    broker = _broker()
    result = broker.get_bars([AAPL], 2, "day")
    assert AAPL in result
    assert len(result[AAPL].df) <= 2


def test_get_account_reports_cash_only_when_no_positions_are_held() -> None:
    broker = _broker(budget=Decimal(5000))
    account = broker.get_account()
    assert account.cash == Decimal(5000)
    assert account.portfolio_value == Decimal(5000)


def test_pull_positions_is_empty_for_a_fresh_backtest() -> None:
    assert _broker().pull_positions() == []


def test_close_position_with_no_position_returns_none() -> None:
    assert _broker().close_position(AAPL) is None


def test_start_and_stop_stream_are_no_ops() -> None:
    broker = _broker()
    broker.start_stream()
    broker.stop_stream()  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_broker_orders.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/broker.py` (order lifecycle + market data reads)**

```python
"""`Broker` implementation simulating fills against a `BacktestDataSource`, with no
network calls. Cash and positions are Decimal-exact. Order submission here tracks the
order and queues it for the next bar; `on_advance`/`process_pending` (this module,
added alongside the fill engine) are what actually fill it -- see the module's second
half below the "--- fills" marker.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from trading_agent_framework.backtesting import fills
from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.backtesting.ledger import Ledger
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BacktestError, OrderValidationError

logger = logging.getLogger(__name__)


@dataclass
class _PendingOrder:
    order: Order
    asset: Asset
    last_evaluated: datetime


class BacktestBroker(Broker):
    """`Broker` implementation backed by simulated fills against a `BacktestDataSource`."""

    name: ClassVar[str] = "backtest"

    def __init__(
        self,
        strategy_name: str,
        *,
        data_source: BacktestDataSource,
        clock: MarketClock,
        budget: Decimal,
        timestep: str = "day",
        commission: Decimal = Decimal(0),
        slippage: Decimal = Decimal(0),
        tracker: OrderTracker | None = None,
    ) -> None:
        super().__init__(strategy_name, tracker, clock=clock, is_paper=True)
        self._data_source = data_source
        self._timestep = timestep
        self._commission = commission
        self._slippage = slippage
        self._cash = budget
        self._positions: dict[Asset, Position] = {}
        self._pending: dict[str, _PendingOrder] = {}
        self.ledger = Ledger()

    # --- Broker ABC: orders ------------------------------------------------------------

    def _conform_order(self, order: Order) -> Order:
        return order  # no broker-specific tick rounding to simulate

    def _submit_order(self, order: Order) -> Order:
        if order.notional is not None:
            raise OrderValidationError(
                "backtesting only supports quantity-based orders, not notional orders"
            )
        if not order.client_order_id:
            order.client_order_id = f"{self.strategy_name}:{order.identifier}"
        self.tracker.track_unprocessed(order)
        self.tracker.process_trade_event(order, OrderEvent.NEW)
        self._pending[order.identifier] = _PendingOrder(
            order=order, asset=order.asset, last_evaluated=self.clock.now()
        )
        return order

    def cancel_order(self, order: Order) -> None:
        self._pending.pop(order.identifier, None)
        self.tracker.process_trade_event(order, OrderEvent.CANCELED)

    def pull_order(self, identifier: str) -> Order | None:
        return self.tracker.get_tracked_order(identifier)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        return self.tracker.get_all_tracked_orders()[:limit]

    def pull_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> AccountBalances:
        portfolio_value = self._portfolio_value(self.clock.now())
        return AccountBalances(
            cash=self._cash, portfolio_value=portfolio_value, buying_power=self._cash
        )

    def modify_order(
        self, order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None
    ) -> Order:
        if order.identifier not in self._pending:
            raise BacktestError(f"order {order.identifier} is not pending; cannot modify")
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        return order

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = self._positions.get(asset)
        if position is None:
            return None
        quantity = (position.quantity * fraction).copy_abs()
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        order = Order(strategy_name=self.strategy_name, asset=asset, side=side, quantity=quantity)
        return self.submit_order(order)

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            for pending in list(self._pending.values()):
                self.cancel_order(pending.order)
        closed: list[Order] = []
        for asset in list(self._positions):
            order = self.close_position(asset)
            if order is not None:
                closed.append(order)
        return closed

    def sync_open_orders(self) -> list[Order]:
        return []  # a fresh backtest has no prior state to adopt

    # --- Broker ABC: market data --------------------------------------------------------

    def get_last_price(self, asset: Asset) -> Decimal | None:
        bar = self._latest_bar(asset, self.clock.now())
        return bar.close if bar is not None else None

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return {asset: self.get_last_price(asset) for asset in assets}

    def get_quote(self, asset: Asset) -> Quote | None:
        bar = self._latest_bar(asset, self.clock.now())
        if bar is None:
            return None
        return Quote(
            asset=asset, bid=bar.close, ask=bar.close, bid_size=None, ask_size=None,
            timestamp=self.clock.now(),
        )

    def get_bars(
        self, assets: Sequence[Asset], length: int, timestep: str = "day", *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        result: dict[Asset, Bars] = {}
        for asset in assets:
            bars = self._data_source.bars(asset, self.clock.now(), length, timestep)
            if bars is not None:
                result[asset] = bars
        return result

    def start_stream(self) -> None:
        pass

    def stop_stream(self, timeout: float = 5.0) -> None:
        pass

    # --- shared bar lookup -----------------------------------------------------------------

    def _latest_bar_with_time(
        self, asset: Asset, cutoff: datetime
    ) -> tuple[fills.Bar, datetime] | None:
        bars = self._data_source.bars(asset, cutoff, 1, self._timestep)
        if bars is None or bars.df.empty:
            return None
        row = bars.df.iloc[-1]
        bar = fills.Bar(
            open=Decimal(str(row["open"])), high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])), close=Decimal(str(row["close"])),
        )
        return bar, bars.df.index[-1].to_pydatetime()

    def _latest_bar(self, asset: Asset, cutoff: datetime) -> fills.Bar | None:
        found = self._latest_bar_with_time(asset, cutoff)
        return None if found is None else found[0]

    def _portfolio_value(self, cutoff: datetime) -> Decimal:
        return self._cash + self._positions_value(cutoff)

    def _positions_value(self, cutoff: datetime) -> Decimal:
        total = Decimal(0)
        for asset, position in self._positions.items():
            bar = self._latest_bar(asset, cutoff)
            price = bar.close if bar is not None else (position.avg_fill_price or Decimal(0))
            signed_qty = position.quantity if position.side is PositionSide.LONG else -position.quantity
            total += signed_qty * price
        return total
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_broker_orders.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/broker.py tests/backtesting/test_broker_orders.py
git commit -m "Add BacktestBroker: order lifecycle, account, market data reads

Orders submit, track as NEW, and queue for the next bar; fills themselves
land in the next task alongside on_advance/process_pending."
```

---

### Task 8: `backtesting/broker.py` part 2 — fills, `on_advance`, equity sampling

**Files:**
- Modify: `src/trading_agent_framework/backtesting/broker.py` (add methods to `BacktestBroker`; no existing method from Task 7 changes)
- Test: `tests/backtesting/test_broker_fills.py`

**Interfaces:**
- Consumes: `fills.evaluate_fill`, `fills.apply_commission_and_slippage` (Task 4); `Ledger.record_fill/.record_equity` (Task 5).
- Produces: `BacktestBroker.on_advance(previous_now, new_now)` — the callback `backtesting/runner.py` (Task 15) wires to `BacktestClock.on_advance` (Task 6).

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/test_broker_fills.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, PositionSide
from trading_agent_framework.entities.order import Order

AAPL = Asset("AAPL")
DAY1 = datetime(2026, 1, 5, 16, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
DAY3 = DAY2 + timedelta(days=1)


def _broker_with_two_bars(budget: Decimal = Decimal(10000)) -> BacktestBroker:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=budget)
    clock.on_advance = broker.on_advance
    return broker, clock, source


def test_market_order_fills_on_the_next_bar_not_the_submission_bar() -> None:
    broker, clock, _ = _broker_with_two_bars()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    # Advancing time without a new bar closing yet: still pending (no bar past DAY1 exists yet
    # at exactly DAY1's cutoff -- the fake's only bar closed *at* DAY1, same as last_evaluated).
    broker.on_advance(clock.now(), clock.now())
    assert order.status is OrderStatus.NEW

    # Simulate the clock reaching DAY2, where the second bar has closed.
    clock._now = DAY2  # test-only direct time jump; production code goes through clock.wait()
    broker.on_advance(DAY1, DAY2)

    assert order.status is OrderStatus.FILL
    assert order.avg_fill_price == Decimal("151.0")  # bar 2's open == its close in this fixture
    assert order.identifier not in broker._pending


def test_fill_updates_cash_and_creates_a_long_position() -> None:
    broker, clock, _ = _broker_with_two_bars(budget=Decimal(10000))
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert broker._cash == Decimal(10000) - Decimal(10) * Decimal("151.0")
    position = broker.pull_positions()[0]
    assert position.asset == AAPL
    assert position.quantity == Decimal(10)
    assert position.side is PositionSide.LONG


def test_fill_records_a_fill_in_the_ledger() -> None:
    broker, clock, _ = _broker_with_two_bars()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert len(broker.ledger.fills) == 1
    record = broker.ledger.fills[0]
    assert record.identifier == order.identifier
    assert record.symbol == "AAPL"
    assert record.filled_quantity == Decimal(10)
    assert record.price == Decimal("151.0")


def test_commission_and_slippage_reduce_cash_beyond_the_raw_notional() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([100.0, 100.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker(
        "momentum", data_source=source, clock=clock, budget=Decimal(10000),
        commission=Decimal("0.01"), slippage=Decimal("0.01"),
    )
    clock.on_advance = broker.on_advance
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    # execution_price = 100 * 1.01 = 101; commission = 101 * 0.01 = 1.01/share
    expected_cash = Decimal(10000) - (Decimal(10) * Decimal("101.00") + Decimal(10) * Decimal("1.0100"))
    assert broker._cash == expected_cash


def test_unfilled_limit_order_stays_pending_and_is_retried_next_bar() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 200.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance
    order = broker.submit_order(
        Order(
            strategy_name="momentum", asset=AAPL, side=OrderSide.BUY,
            quantity=Decimal(10), limit_price=Decimal(90),  # never touched by this fixture's bars
        )
    )

    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert order.status is OrderStatus.NEW
    assert order.identifier in broker._pending

    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)
    assert order.status is OrderStatus.NEW  # still never touched -- stays pending, doesn't error


def test_on_advance_samples_equity_after_processing_fills() -> None:
    broker, clock, _ = _broker_with_two_bars(budget=Decimal(10000))
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert len(broker.ledger.equity) == 1
    sample = broker.ledger.equity[0]
    assert sample.time == DAY2
    assert sample.positions_value == Decimal(10) * Decimal("151.0")
    assert sample.cash == broker._cash
    assert sample.portfolio_value == broker._cash + sample.positions_value


def test_selling_closes_the_position_when_quantity_returns_to_zero() -> None:
    broker, clock, _ = _broker_with_two_bars(budget=Decimal(10000))
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert len(broker.pull_positions()) == 1

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(10)))
    # Advance once more: the sell was submitted at DAY2, needs a bar closing after DAY2.
    # The two-bar fixture has no bar after DAY2, so extend it in this test:
```

Note for the implementer: the last test above (`test_selling_closes_the_position_when_quantity_returns_to_zero`) needs a **three-bar** fixture (so the sell submitted after the first fill has a further bar to fill against), not the two-bar `_broker_with_two_bars` helper. Rewrite it using a local three-bar setup mirroring `test_unfilled_limit_order_stays_pending_and_is_retried_next_bar`'s pattern:

```python
def test_selling_closes_the_position_when_quantity_returns_to_zero() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert len(broker.pull_positions()) == 1

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(10)))
    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)

    assert broker.pull_positions() == []
```

Replace the earlier, incomplete draft of this test with this version before running the suite.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_broker_fills.py -v`
Expected: FAIL — `AttributeError: 'BacktestBroker' object has no attribute 'on_advance'`

- [ ] **Step 3: Add the fill engine to `backtesting/broker.py`**

Add these methods to the `BacktestBroker` class, after `_positions_value` (the last method from Task 7). Also add `from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord` to the existing `from trading_agent_framework.backtesting.ledger import Ledger` import line (change it to import all three names).

```python
    # --- fills: called by BacktestClock.on_advance --------------------------------------

    def on_advance(self, previous_now: datetime, new_now: datetime) -> None:
        """Registered as `BacktestClock.on_advance`: process fills, then sample equity."""
        self._process_pending(new_now)
        self._sample_equity(new_now)

    def _process_pending(self, cutoff: datetime) -> None:
        for identifier in list(self._pending):
            pending = self._pending[identifier]
            found = self._latest_bar_with_time(pending.asset, cutoff)
            if found is None:
                continue
            bar, bar_time = found
            if bar_time <= pending.last_evaluated:
                continue  # no new bar has closed for this asset since we last checked
            pending.last_evaluated = bar_time
            order = pending.order
            try:
                result = fills.evaluate_fill(
                    order_type=order.order_type, side=order.side, bar=bar,
                    limit_price=order.limit_price, stop_price=order.stop_price,
                    stop_limit_price=order.stop_limit_price,
                )
            except ValueError as exc:
                order.set_error(exc)
                self.tracker.process_trade_event(order, OrderEvent.ERROR)
                del self._pending[identifier]
                continue
            if result is None:
                continue  # still doesn't touch the trigger; retried on the next bar
            self._fill(order, result.price, bar_time)
            del self._pending[identifier]

    def _fill(self, order: Order, raw_price: Decimal, bar_time: datetime) -> None:
        assert order.quantity is not None  # notional orders are rejected at submission
        execution_price, commission_per_share = fills.apply_commission_and_slippage(
            raw_price, order.side, commission=self._commission, slippage=self._slippage
        )
        quantity = order.quantity
        commission_cost = commission_per_share * quantity
        notional = execution_price * quantity
        if order.side is OrderSide.BUY:
            self._cash -= notional + commission_cost
        else:
            self._cash += notional - commission_cost
        self._apply_to_position(order.asset, order.side, quantity, execution_price)
        self.ledger.record_fill(FillRecord(
            time=bar_time, identifier=order.identifier, symbol=order.asset.symbol,
            side=order.side, order_type=order.order_type, quantity=order.quantity,
            filled_quantity=quantity, price=execution_price, trade_cost=commission_cost,
            trade_slippage=(execution_price - raw_price).copy_abs(),
        ))
        self.tracker.process_trade_event(
            order, OrderEvent.FILLED, price=execution_price, filled_quantity=quantity
        )

    def _apply_to_position(self, asset: Asset, side: OrderSide, quantity: Decimal, price: Decimal) -> None:
        existing = self._positions.get(asset)
        signed = quantity if side is OrderSide.BUY else -quantity
        new_quantity = signed if existing is None else (
            existing.quantity if existing.side is PositionSide.LONG else -existing.quantity
        ) + signed
        if new_quantity == 0:
            self._positions.pop(asset, None)
            return
        self._positions[asset] = Position(
            strategy_name=self.strategy_name, asset=asset, quantity=new_quantity.copy_abs(),
            side=PositionSide.LONG if new_quantity > 0 else PositionSide.SHORT,
            avg_fill_price=price,
        )

    def _sample_equity(self, cutoff: datetime) -> None:
        positions_value = self._positions_value(cutoff)
        self.ledger.record_equity(EquitySample(
            time=cutoff, portfolio_value=self._cash + positions_value,
            cash=self._cash, positions_value=positions_value,
        ))
```

Note the fix in `_apply_to_position` versus a naive draft: an existing position's *signed* quantity must be reconstructed from its `(quantity, side)` pair (Position stores an unsigned magnitude, per `entities/position.py`) before adding the new fill's signed delta — otherwise a SELL that flips a long position short would double-subtract.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_broker_fills.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/broker.py tests/backtesting/test_broker_fills.py
git commit -m "Add BacktestBroker's fill engine: on_advance, process_pending, equity sampling

Orders fill against the first bar that closes strictly after submission (or
later, if the limit/stop condition isn't touched yet); every advance also
records a Decimal-exact equity sample in the ledger."
```

---

### Task 9: No-look-ahead guardian test + full simulated session test

**Files:**
- Create: `tests/backtesting/test_no_look_ahead.py`

**Interfaces:**
- Consumes: `BacktestClock` (Task 6), `BacktestBroker` (Tasks 7-8), `FakeBacktestDataSource` (Task 3), `Strategy`/`StrategyExecutor` (existing, unmodified).
- Produces: nothing new — this is the integration test that protects the whole premise (design spec, section 8), wiring the pieces exactly as `backtesting/runner.py` will in Task 15, but by hand, so it can land before the runner exists.

This task has no separable "make it pass" implementation step: every module it exercises already exists from Tasks 3-8. It is still written test-first in spirit — run it immediately after writing to confirm today's code already satisfies the no-look-ahead guarantee, which is the point of the test.

- [ ] **Step 1: Write the tests**

`tests/backtesting/test_no_look_ahead.py`:

```python
"""Integration test: the no-look-ahead guarantee across a full simulated multi-session
run. This is the test that protects the whole premise of the backtesting subsystem
(design spec, section 8): every price a strategy observes during on_trading_iteration()
must come from a bar that had already closed by the time of that observation, and the
bar for the CURRENT, still-forming session must never be visible.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tests.backtesting.fakes import FakeBacktestDataSource

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession

ET = ZoneInfo("America/New_York")
AAPL = Asset("AAPL")


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


def _close_indexed_bars(sessions: list[MarketSession], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
            "close": closes, "volume": [1000.0] * len(closes),
        },
        index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"),
    )


class RecordingStrategy(Strategy):
    sleeptime = "1D"

    def initialize(self) -> None:
        self.vars.observations = []

    def on_trading_iteration(self) -> None:
        now = self.clock.now()
        bars = self.get_historical_prices(AAPL, 10, "day")
        last_price = self.get_last_price(AAPL)
        latest_bar_close = (
            bars.df.index[-1].to_pydatetime() if bars is not None and not bars.df.empty else None
        )
        self.vars.observations.append(
            {"now": now, "latest_bar_close": latest_bar_close, "last_price": last_price}
        )


def test_strategy_never_observes_a_bar_that_has_not_closed_yet(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 5)
    closes = [150.0, 151.0, 149.0, 152.0, 153.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _close_indexed_bars(sessions, closes))

    clock = BacktestClock(start=sessions[0].open - timedelta(hours=1), sessions=sessions)
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    strategy = RecordingStrategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    strategy.executor.run()

    observations = strategy.vars.observations
    assert len(observations) == 5  # one iteration per session (sleeptime="1D")

    for i, obs in enumerate(observations):
        if obs["latest_bar_close"] is None:
            assert i == 0  # only the very first session has no prior closed bar at all
            continue
        # The chokepoint itself: bar_end <= cutoff, always, for every observation.
        assert obs["latest_bar_close"] <= obs["now"]
        # The sharper claim: session i's own iteration never sees session i's own bar --
        # only a strictly earlier session's.
        assert obs["latest_bar_close"] < sessions[i].close

    # Concretely: session 3's (index 2) iteration sees session 2's (index 1) close.
    assert observations[2]["last_price"] == Decimal(str(closes[1]))
    assert observations[2]["latest_bar_close"] == sessions[1].close


class OrderPlacingStrategy(Strategy):
    sleeptime = "1D"

    def initialize(self) -> None:
        self.vars.hooks_called = []
        self.vars.fill_args = None

    def before_market_opens(self) -> None:
        self.vars.hooks_called.append("before_market_opens")

    def before_starting_trading(self) -> None:
        self.vars.hooks_called.append("before_starting_trading")

    def on_trading_iteration(self) -> None:
        self.vars.hooks_called.append("on_trading_iteration")
        if self.first_iteration:
            self.submit_order(self.create_order(AAPL, 5, "buy"))

    def on_filled_order(self, position, order, price, quantity, multiplier) -> None:
        self.vars.fill_args = (order.identifier, price, quantity)

    def before_market_closes(self) -> None:
        self.vars.hooks_called.append("before_market_closes")

    def after_market_closes(self) -> None:
        self.vars.hooks_called.append("after_market_closes")


def test_full_simulated_session_dispatches_hooks_in_order_and_fills_next_bar(tmp_path: Path) -> None:
    sessions = _sessions(date(2026, 1, 5), 3)
    closes = [150.0, 151.0, 152.0]
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _close_indexed_bars(sessions, closes))

    clock = BacktestClock(start=sessions[0].open - timedelta(hours=1), sessions=sessions)
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    strategy = OrderPlacingStrategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    strategy.executor.run()

    assert strategy.vars.hooks_called[:3] == [
        "before_market_opens", "before_starting_trading", "on_trading_iteration",
    ]
    assert strategy.vars.hooks_called.count("before_market_opens") == 3
    assert strategy.vars.hooks_called.count("on_trading_iteration") == 3
    assert strategy.vars.hooks_called.count("before_market_closes") == 3
    assert strategy.vars.hooks_called.count("after_market_closes") == 3

    # Submitted during session 1's iteration; fills against session 2's bar, and
    # on_filled_order (dispatched on the executor thread, per executor.py's design)
    # has fired with that fill's data by the time the run ends.
    assert strategy.vars.fill_args is not None
    identifier, price, quantity = strategy.vars.fill_args
    assert quantity == Decimal(5)
    assert price == Decimal("151.0")
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/backtesting/test_no_look_ahead.py -v`
Expected: PASS. If either test fails, do not patch the test to make it pass — this test is the oracle; a failure here means a real bug in Tasks 3-8 (most likely in `_process_pending`'s `bar_time <= pending.last_evaluated` guard, or in a data source's bar-close indexing). Fix the implementation, not the assertions.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/backtesting/test_no_look_ahead.py
git commit -m "Add the no-look-ahead guardian test and a full simulated-session test

Proves bar_end <= now for every observation across a 5-session run, and that
a strategy never sees its own session's still-forming bar; separately proves
hook dispatch order and that a submitted order fills on the following bar
with on_filled_order firing before the run ends."
```

---

### Task 10: `backtesting/data/cache.py` — `CachedDataSource`

**Files:**
- Create: `src/trading_agent_framework/backtesting/data/cache.py`
- Test: `tests/backtesting/data/test_cache.py`

**Interfaces:**
- Consumes: `BacktestDataSource`, `FULL_HISTORY` (Task 3); `FakeBacktestDataSource` (Task 3) as the wrapped `inner` in tests.
- Produces: `CachedDataSource(inner, cache_dir)` implementing `BacktestDataSource`. Consumed by `Strategy.run_backtesting`'s default construction is NOT this task's job (the default is Yahoo, wrapped in a cache, wired in Task 17) — this task only builds and tests the wrapper itself.

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/data/test_cache.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.data.cache import CachedDataSource
from trading_agent_framework.entities.asset import Asset

AAPL = Asset("AAPL")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def test_first_load_fetches_from_the_inner_source_and_writes_a_parquet_file(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0, 102.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")

    files = list((tmp_path / "fake").glob("AAPL_day_*.parquet"))
    assert len(files) == 1
    assert len(inner.load_calls) == 1


def test_second_load_of_the_same_window_does_not_touch_the_inner_source_again(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")
    cache.load([AAPL], START, END, "day")

    assert len(inner.load_calls) == 1  # second call was a cache hit


def test_bars_reads_from_the_cached_parquet_file(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0, 101.0, 102.0], start=START))
    cache = CachedDataSource(inner, tmp_path)
    cache.load([AAPL], START, END, "day")

    result = cache.bars(AAPL, END, 2, "day")

    assert result is not None
    assert list(result.df["close"]) == [101.0, 102.0]


def test_bars_without_a_prior_load_falls_through_to_the_inner_source(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    result = cache.bars(AAPL, START, 1, "day")

    assert result is not None
    assert len(inner.bars_calls) == 1


def test_cache_writes_a_meta_json_sidecar_with_provenance(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    inner.set_bars(AAPL, make_close_indexed_frame([100.0], start=START))
    cache = CachedDataSource(inner, tmp_path)

    cache.load([AAPL], START, END, "day")

    [meta_path] = list((tmp_path / "fake").glob("*.meta.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["provider"] == "fake"
    assert meta["symbol"] == "AAPL"
    assert meta["rows"] == 1
    assert "fetched_at" in meta


def test_sessions_delegates_to_the_inner_source(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    cache = CachedDataSource(inner, tmp_path)
    assert cache.sessions(START, END) == inner.sessions(START, END)


def test_cache_name_matches_the_inner_sources_name(tmp_path: Path) -> None:
    inner = FakeBacktestDataSource()
    cache = CachedDataSource(inner, tmp_path)
    assert cache.name == "fake"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/data/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/data/cache.py`**

```python
"""Read-through parquet cache wrapping any `BacktestDataSource`. First `load()` fetches
and writes; a later `load()` for the same window is network-free -- the case that
matters most when iterating on an agent prompt against a fixed backtest period.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from trading_agent_framework.backtesting.data.base import FULL_HISTORY, BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession


class CachedDataSource(BacktestDataSource):
    """Wraps `inner`, caching each asset's fetched window at
    `<cache_dir>/<inner.name>/<symbol>_<timestep>_<start>_<end>.parquet` plus a
    `.meta.json` sidecar recording provenance (provider, symbol, fetch time, row count)
    -- `inner` may revise its data over time (e.g. Yahoo's retroactive dividend
    adjustments), so a cached file is a frozen, dated snapshot on purpose.
    """

    def __init__(self, inner: BacktestDataSource, cache_dir: Path) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir) / inner.name
        self.name = inner.name

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._ensure_cached(asset, start, end, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        # bars() alone carries no [start, end] window to cache against -- only load()
        # does -- so a cache miss here falls straight through to the inner source.
        path = self._latest_cache_path(asset, timestep)
        if path is None:
            return self._inner.bars(asset, cutoff, length, timestep)
        return self._read(path, asset, timestep, cutoff, length)

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return self._inner.sessions(start, end)

    def _cache_path(self, asset: Asset, start: datetime, end: datetime, timestep: str) -> Path:
        stamp = f"{asset.symbol}_{timestep}_{start.date()}_{end.date()}"
        return self._cache_dir / f"{stamp}.parquet"

    def _latest_cache_path(self, asset: Asset, timestep: str) -> Path | None:
        if not self._cache_dir.is_dir():
            return None
        matches = sorted(self._cache_dir.glob(f"{asset.symbol}_{timestep}_*.parquet"))
        return matches[-1] if matches else None

    def _ensure_cached(self, asset: Asset, start: datetime, end: datetime, timestep: str) -> None:
        path = self._cache_path(asset, start, end, timestep)
        if path.exists():
            return
        self._inner.load([asset], start, end, timestep)
        bars = self._inner.bars(asset, end, FULL_HISTORY, timestep)
        if bars is None:
            return
        self._write(path, bars, asset)

    def _write(self, path: Path, bars: Bars, asset: Asset) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        bars.df.to_parquet(path)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "provider": self.name, "symbol": asset.symbol,
                    "fetched_at": datetime.now().isoformat(), "rows": len(bars.df),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _read(
        self, path: Path, asset: Asset, timestep: str, cutoff: datetime, length: int
    ) -> Bars | None:
        import pandas as pd

        df = pd.read_parquet(path)
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))
```

Note: `path.with_suffix(".meta.json")` on a path ending in `.parquet` produces `..._2026-01-01_2026-01-10.meta.json` (it replaces only the final `.parquet` suffix, not a double extension) — this is correct pathlib behaviour, not a bug; confirm with a quick `python -c "from pathlib import Path; print(Path('a.parquet').with_suffix('.meta.json'))"` if unsure before moving on.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/data/test_cache.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/data/cache.py tests/backtesting/data/test_cache.py
git commit -m "Add CachedDataSource: read-through parquet cache for any BacktestDataSource"
```

---

### Task 11: `backtesting/data/yahoo.py` — `YahooBacktestData` (default provider)

**Files:**
- Create: `src/trading_agent_framework/backtesting/data/yahoo.py`
- Test: `tests/backtesting/data/test_yahoo.py`

**Interfaces:**
- Consumes: `BacktestDataSource` (Task 3); `BacktestDataError` (Task 2).
- Produces: `YahooBacktestData(start, end, download=None)` implementing `BacktestDataSource` (`timestep="day"` only); `parse_yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame` (pure). `yfinance` is imported only inside `_real_download`, never at module level — never touched by this task's tests, which always inject `download`.

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/data/test_yahoo.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData, parse_yahoo_frame
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BacktestDataError

AAPL = Asset("AAPL")
ET = ZoneInfo("America/New_York")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def _raw_yahoo_frame() -> pd.DataFrame:
    """Shaped like a real single-ticker yfinance.download(...) result: capitalised
    columns, a naive DatetimeIndex of session dates."""
    index = pd.date_range("2026-01-05", periods=3, freq="B")
    return pd.DataFrame(
        {
            "Open": [150.0, 151.0, 152.0], "High": [151.0, 152.0, 153.0],
            "Low": [149.0, 150.0, 151.0], "Close": [150.5, 151.5, 152.5],
            "Volume": [1000.0, 1100.0, 1200.0],
        },
        index=index,
    )


def test_parse_yahoo_frame_lowercases_columns_and_indexes_by_session_close() -> None:
    df = parse_yahoo_frame(_raw_yahoo_frame())

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 150.5
    assert df.index[0] == datetime(2026, 1, 5, 16, 0, tzinfo=ET)
    assert df.index[0].tzinfo is not None


def test_parse_yahoo_frame_handles_an_empty_frame() -> None:
    assert parse_yahoo_frame(pd.DataFrame()).empty


def test_parse_yahoo_frame_drops_a_multiindex_ticker_level() -> None:
    raw = _raw_yahoo_frame()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
    df = parse_yahoo_frame(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_load_and_bars_use_the_injected_download_function() -> None:
    calls: list[tuple] = []

    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        calls.append((symbol, kwargs))
        return _raw_yahoo_frame()

    source = YahooBacktestData(START, END, download=fake_download)
    source.load([AAPL], START, END, "day")

    assert len(calls) == 1
    assert calls[0][0] == "AAPL"

    result = source.bars(AAPL, END, 2, "day")
    assert result is not None
    assert list(result.df["close"]) == [151.5, 152.5]


def test_bars_fetches_lazily_when_load_was_never_called() -> None:
    def fake_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        return _raw_yahoo_frame()

    source = YahooBacktestData(START, END, download=fake_download)
    result = source.bars(AAPL, END, 1, "day")
    assert result is not None


def test_a_download_failure_raises_backtest_data_error() -> None:
    def failing_download(symbol: str, **kwargs: object) -> pd.DataFrame:
        raise RuntimeError("network is down")

    source = YahooBacktestData(START, END, download=failing_download)
    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_minute_timestep_is_not_supported() -> None:
    source = YahooBacktestData(START, END, download=lambda *a, **k: _raw_yahoo_frame())
    with pytest.raises(BacktestDataError, match="day"):
        source.bars(AAPL, END, 1, "minute")


def test_sessions_are_one_per_weekday_930_to_1600_et() -> None:
    source = YahooBacktestData(START, END)
    sessions = source.sessions(datetime(2026, 1, 5, tzinfo=ET), datetime(2026, 1, 9, tzinfo=ET))
    assert len(sessions) == 5  # Mon-Fri
    assert sessions[0].open == datetime(2026, 1, 5, 9, 30, tzinfo=ET)
    assert sessions[0].close == datetime(2026, 1, 5, 16, 0, tzinfo=ET)


def test_name_is_yahoo() -> None:
    assert YahooBacktestData(START, END).name == "yahoo"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/data/test_yahoo.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/data/yahoo.py`**

```python
"""`BacktestDataSource` backed by yfinance daily OHLCV -- the default provider
(design spec, section 4.2): consolidated tape, official closing-auction closes,
decades of free history. `yfinance` is imported lazily so a strategy that never
backtests with Yahoo data never pays for its 12 transitive dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

if TYPE_CHECKING:
    import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

DownloadFn = Callable[..., "pd.DataFrame"]


class YahooBacktestData(BacktestDataSource):
    """Daily OHLCV from Yahoo Finance. Only `timestep="day"` is supported (design
    spec, section 4.1 -- Yahoo has no usable minute history)."""

    name = "yahoo"

    def __init__(
        self, start: datetime, end: datetime, *, download: DownloadFn | None = None
    ) -> None:
        self._start = start
        self._end = end
        self._download = download  # injected in tests; real yfinance.download otherwise
        self._frames: dict[Asset, pd.DataFrame] = {}

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._fetch(asset, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        if timestep != "day":
            raise BacktestDataError(f"YahooBacktestData only supports timestep='day', got {timestep!r}")
        df = self._frames.get(asset)
        if df is None:
            df = self._fetch(asset, timestep)
        if df is None or df.empty:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        """One 9:30-16:00 ET session per weekday in [start, end]. Half-days are not
        modelled (design spec, section 4.1) -- irrelevant for daily-bar strategies."""
        sessions: list[MarketSession] = []
        day = start.astimezone(MARKET_TZ).date()
        last = end.astimezone(MARKET_TZ).date()
        while day <= last:
            if day.weekday() < 5:
                sessions.append(
                    MarketSession(
                        open=datetime.combine(day, SESSION_OPEN, tzinfo=MARKET_TZ),
                        close=datetime.combine(day, SESSION_CLOSE, tzinfo=MARKET_TZ),
                    )
                )
            day += timedelta(days=1)
        return sessions

    def _fetch(self, asset: Asset, timestep: str) -> pd.DataFrame | None:
        download = self._download if self._download is not None else self._real_download()
        try:
            raw = download(
                asset.symbol,
                start=self._start.date().isoformat(),
                end=(self._end.date() + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise BacktestDataError(f"failed to fetch Yahoo data for {asset.symbol}: {exc}") from exc
        df = parse_yahoo_frame(raw)
        self._frames[asset] = df
        return df

    def _real_download(self) -> DownloadFn:
        import yfinance as yf

        return yf.download


def parse_yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure: normalise a yfinance download into `Bars.df` shape, indexed by bar CLOSE
    (each daily row's session date at 16:00 ET -- yfinance's own index is that
    session's date, naive)."""
    import pandas as pd

    if raw.empty:
        return raw
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    close_index = [datetime.combine(ts.date(), SESSION_CLOSE, tzinfo=MARKET_TZ) for ts in df.index]
    df.index = pd.DatetimeIndex(close_index, name="timestamp")
    return df.sort_index()
```

Remove the now-unused `date` import if `ruff` flags it (only `datetime`/`time`/`timedelta`/`ZoneInfo` are actually referenced in the final version above).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/data/test_yahoo.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/data/yahoo.py tests/backtesting/data/test_yahoo.py
git commit -m "Add YahooBacktestData: the default backtest data source

Daily OHLCV, official closing-auction closes, lazy yfinance import. bars()
fetches on demand so a strategy's dynamic asset universe never needs to be
known up front."
```

---

### Task 12: `backtesting/data/alpaca.py` — `AlpacaBacktestData`

**Files:**
- Create: `src/trading_agent_framework/backtesting/data/alpaca.py`
- Test: `tests/backtesting/data/test_alpaca.py`

**Interfaces:**
- Consumes: `BacktestDataSource`, `FULL_HISTORY` (Task 3); `BacktestDataError` (Task 2); existing `brokers.alpaca.account`/`brokers.alpaca.market_data` pure translation functions (`build_calendar_request`, `parse_calendar`, `build_bars_request`, `parse_bars`, `MARKET_TZ`); existing `tests/fakes.py`'s `FakeTradingClient`, `FakeStockHistoricalDataClient`, `bar_payload`, `make_alpaca_calendar`.
- Produces: `AlpacaBacktestData(client, trading_client, start, end)` implementing `BacktestDataSource`; `reindex_to_bar_close(df, timestep, sessions) -> pd.DataFrame` (pure).

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/data/test_alpaca.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from tests.fakes import FakeStockHistoricalDataClient, FakeTradingClient, bar_payload, make_alpaca_calendar

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData, reindex_to_bar_close
from trading_agent_framework.brokers.alpaca.market_data import MARKET_TZ
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

AAPL = Asset("AAPL")
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 10, tzinfo=UTC)


def _sessions() -> list[MarketSession]:
    return [
        MarketSession(
            open=datetime(2026, 1, 5, 9, 30, tzinfo=MARKET_TZ),
            close=datetime(2026, 1, 5, 16, 0, tzinfo=MARKET_TZ),
        ),
        MarketSession(
            open=datetime(2026, 1, 6, 9, 30, tzinfo=MARKET_TZ),
            close=datetime(2026, 1, 6, 13, 0, tzinfo=MARKET_TZ),  # early close
        ),
    ]


def test_reindex_to_bar_close_maps_daily_bars_to_their_sessions_close() -> None:
    df = pd.DataFrame(
        {"open": [150.0, 151.0], "high": [151.0, 152.0], "low": [149.0, 150.0],
         "close": [150.5, 151.5], "volume": [1000.0, 1100.0]},
        index=pd.DatetimeIndex(
            [datetime(2026, 1, 5, tzinfo=MARKET_TZ), datetime(2026, 1, 6, tzinfo=MARKET_TZ)]
        ),
    )
    result = reindex_to_bar_close(df, "day", _sessions())
    assert list(result.index) == [_sessions()[0].close, _sessions()[1].close]  # early close respected


def test_reindex_to_bar_close_shifts_minute_bars_by_one_minute() -> None:
    df = pd.DataFrame(
        {"open": [150.0], "high": [151.0], "low": [149.0], "close": [150.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([datetime(2026, 1, 5, 9, 30, tzinfo=MARKET_TZ)]),
    )
    result = reindex_to_bar_close(df, "minute", [])
    assert result.index[0] == datetime(2026, 1, 5, 9, 31, tzinfo=MARKET_TZ)


def test_reindex_to_bar_close_handles_an_empty_frame() -> None:
    assert reindex_to_bar_close(pd.DataFrame(), "day", []).empty


def test_bars_fetches_and_reindexes_via_the_injected_clients() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [
        make_alpaca_calendar("2026-01-05"), make_alpaca_calendar("2026-01-06"),
    ]
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [
        bar_payload("2026-01-05T00:00:00Z", 150.0), bar_payload("2026-01-06T00:00:00Z", 151.0),
    ]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    result = source.bars(AAPL, END, 2, "day")

    assert result is not None
    assert list(result.df["close"]) == [150.0, 151.0]
    # Reindexed to each session's 16:00 ET close, not midnight.
    assert result.df.index[0].hour == 16


def test_bars_returns_none_for_an_asset_with_no_data() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    assert source.bars(Asset("MISSING"), END, 1, "day") is None


def test_a_bars_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-05")]
    data_client = FakeStockHistoricalDataClient()
    data_client.raises["get_stock_bars"] = RuntimeError("API down")
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    with pytest.raises(BacktestDataError, match="AAPL"):
        source.bars(AAPL, END, 1, "day")


def test_a_calendar_fetch_failure_raises_backtest_data_error() -> None:
    trading_client = FakeTradingClient()
    trading_client.raises["get_calendar"] = RuntimeError("API down")
    data_client = FakeStockHistoricalDataClient()
    data_client.bars["AAPL"] = [bar_payload("2026-01-05T00:00:00Z", 150.0)]
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    with pytest.raises(BacktestDataError, match="calendar"):
        source.bars(AAPL, END, 1, "day")


def test_sessions_are_exact_including_early_closes() -> None:
    trading_client = FakeTradingClient()
    trading_client.calendar_response = [make_alpaca_calendar("2026-01-06", close_at="13:00")]
    data_client = FakeStockHistoricalDataClient()
    source = AlpacaBacktestData(data_client, trading_client, START, END)

    [session] = source.sessions(START, END)
    assert session.close.hour == 13


def test_name_is_alpaca() -> None:
    source = AlpacaBacktestData(FakeStockHistoricalDataClient(), FakeTradingClient(), START, END)
    assert source.name == "alpaca"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/data/test_alpaca.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/data/alpaca.py`**

```python
"""`BacktestDataSource` backed by Alpaca's IEX feed, reusing the existing pure
`brokers/alpaca/market_data.py` translation and `brokers/alpaca/account.py`'s
calendar parsing. Exact sessions (early closes included) -- the choice when feed
parity with paper/live matters more than Yahoo's decades of free history (design
spec, section 4.2).

Bars are re-indexed from Alpaca's bar-START convention to this subsystem's
bar-CLOSE convention (see `data/base.py`): a daily bar's index becomes its
session's actual close (handling early closes correctly); a minute bar's index
becomes `timestamp + 1 minute` (Alpaca minute bars are exactly 1-minute windows
starting at `timestamp`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from trading_agent_framework.backtesting.data.base import FULL_HISTORY, BacktestDataSource
from trading_agent_framework.brokers.alpaca import account, market_data
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

if TYPE_CHECKING:
    import pandas as pd
    from alpaca.trading.requests import GetCalendarRequest


class AlpacaTradingCalendarClient(Protocol):
    """The one trading-client method this module needs -- narrower than the full
    `orders.AlpacaTradingClient` Protocol since this module never submits orders."""

    def get_calendar(self, filters: GetCalendarRequest) -> list[object]: ...


class AlpacaBacktestData(BacktestDataSource):
    """Exact Alpaca calendar sessions; IEX daily/minute bars, fetched lazily per asset
    over the fixed `[start, end]` window given at construction."""

    name = "alpaca"

    def __init__(
        self,
        client: market_data.AlpacaStockDataClient,
        trading_client: AlpacaTradingCalendarClient,
        start: datetime,
        end: datetime,
    ) -> None:
        self._client = client
        self._trading_client = trading_client
        self._start = start
        self._end = end
        self._frames: dict[Asset, pd.DataFrame] = {}
        self._sessions_cache: list[MarketSession] | None = None

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._fetch(asset, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        df = self._frames.get(asset)
        if df is None:
            df = self._fetch(asset, timestep)
        if df is None or df.empty:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        if self._sessions_cache is None:
            request = account.build_calendar_request(self._start.date(), self._end.date())
            try:
                days = self._trading_client.get_calendar(filters=request)
            except Exception as exc:
                raise BacktestDataError(f"failed to fetch the Alpaca calendar: {exc}") from exc
            self._sessions_cache = account.parse_calendar(days, market_data.MARKET_TZ)
        return [s for s in self._sessions_cache if s.open >= start and s.close <= end]

    def _fetch(self, asset: Asset, timestep: str) -> pd.DataFrame | None:
        import pandas as pd

        request = market_data.build_bars_request([asset], timestep, self._start, self._end)
        try:
            barset = self._client.get_stock_bars(request)
        except Exception as exc:
            raise BacktestDataError(f"failed to fetch Alpaca bars for {asset.symbol}: {exc}") from exc
        parsed = market_data.parse_bars(barset, [asset], timestep, FULL_HISTORY)
        source_bars = parsed.get(asset)
        if source_bars is None:
            self._frames[asset] = pd.DataFrame()
            return self._frames[asset]
        sessions = self.sessions(self._start, self._end)
        df = reindex_to_bar_close(source_bars.df, timestep, sessions)
        self._frames[asset] = df
        return df


def reindex_to_bar_close(
    df: pd.DataFrame, timestep: str, sessions: Sequence[MarketSession]
) -> pd.DataFrame:
    """Pure: shift Alpaca's bar-START index to bar-CLOSE."""
    import pandas as pd

    if df.empty:
        return df
    if timestep == "minute":
        df = df.copy()
        df.index = df.index + timedelta(minutes=1)
        return df.sort_index()
    close_by_date = {s.open.astimezone(market_data.MARKET_TZ).date(): s.close for s in sessions}
    new_index = [close_by_date.get(ts.date(), ts) for ts in df.index]
    df = df.copy()
    df.index = pd.DatetimeIndex(new_index, name="timestamp")
    return df.sort_index()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/data/test_alpaca.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/data/alpaca.py tests/backtesting/data/test_alpaca.py
git commit -m "Add AlpacaBacktestData: exact-calendar IEX provider

Reuses the existing pure Alpaca market_data/account translation; reindexes
bars from Alpaca's bar-START convention to this subsystem's bar-CLOSE
convention, early closes included."
```

---

### Task 13: `backtesting/metrics.py` — vectorbt performance metrics

**Files:**
- Create: `src/trading_agent_framework/backtesting/metrics.py`
- Test: `tests/backtesting/test_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier backtesting tasks (takes plain pandas Series).
- Produces: `compute_metrics(returns, benchmark_returns, *, timestep, risk_free_rate) -> dict[str, Any]` — a flat dict keyed exactly by the dashboard's `MetricSet` field names plus `raw`. Consumed by `backtesting/runner.py` (Task 15).

**Implementation note before starting:** this task calls `vectorbt`'s `returns` accessor for Sharpe/Sortino/Calmar/Omega/max-drawdown/annualised-return/annualised-volatility. The exact accessor method names below are believed correct for vectorbt 1.x but have not been executed against the installed package as part of writing this plan. If any call raises `AttributeError`, run `python -c "import pandas as pd; help(pd.Series.vbt.returns)"` (after `import vectorbt` has registered the accessor) to find the actual name and fix the call site — the golden-value test in Step 1 is the correctness oracle either way, not the specific method names below.

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_metrics.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent_framework.backtesting.metrics import compute_metrics


def _returns() -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02, -0.03, 0.015, 0.005], index=dates)


def test_compute_metrics_sharpe_matches_the_standard_annualised_formula() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    expected_sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert result["sharpe_strategy"] == pytest.approx(expected_sharpe, rel=1e-3)


def test_compute_metrics_max_drawdown_matches_the_cumulative_curve_formula() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    curve = (1 + returns).cumprod()
    expected_max_dd = float((curve / curve.cummax() - 1).min())
    assert result["max_drawdown_strategy"] == pytest.approx(expected_max_dd, rel=1e-6)


def test_compute_metrics_total_return_matches_the_compounded_return() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)

    expected_total = float((1 + returns).prod() - 1)
    assert result["total_return_strategy"] == pytest.approx(expected_total, rel=1e-6)


def test_compute_metrics_without_a_benchmark_omits_relative_fields_but_not_the_others() -> None:
    result = compute_metrics(_returns(), None, timestep="day", risk_free_rate=0.0)
    assert "sharpe_strategy" in result
    assert "beta" not in result  # no benchmark given -> no relative stats block


def test_compute_metrics_with_a_benchmark_computes_beta_alpha_and_correlation() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    strategy = pd.Series([0.02, -0.01, 0.03, 0.0, 0.02, -0.02, 0.03, -0.01, 0.02, 0.01], index=dates)
    benchmark = pd.Series([0.01, -0.005, 0.015, 0.0, 0.01, -0.01, 0.015, -0.005, 0.01, 0.005], index=dates)

    result = compute_metrics(strategy, benchmark, timestep="day", risk_free_rate=0.0)

    expected_beta = float(np.cov(strategy, benchmark)[0, 1] / np.var(benchmark))
    assert result["beta"] == pytest.approx(expected_beta, rel=1e-3)
    assert result["correlation"] == pytest.approx(float(np.corrcoef(strategy, benchmark)[0, 1]), rel=1e-3)
    assert "sharpe_benchmark" in result


def test_compute_metrics_skew_and_kurtosis_match_pandas() -> None:
    returns = _returns()
    result = compute_metrics(returns, None, timestep="day", risk_free_rate=0.0)
    assert result["skew_strategy"] == pytest.approx(float(returns.skew()), rel=1e-6)
    assert result["kurtosis_strategy"] == pytest.approx(float(returns.kurt()), rel=1e-6)


def test_compute_metrics_includes_summary_tables_in_raw() -> None:
    result = compute_metrics(_returns(), None, timestep="day", risk_free_rate=0.0)
    assert "summary_tables" in result["raw"]
    assert "eoy_returns_vs_benchmark" in result["raw"]["summary_tables"]
    assert "drawdowns" in result["raw"]["summary_tables"]
    assert result["raw"]["summary_tables"]["eoy_returns_vs_benchmark"][0]["year"] == 2024


def test_compute_metrics_never_raises_on_a_constant_return_series() -> None:
    """A flat return series has zero std/variance -- must degrade gracefully, not NaN-crash."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    flat = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0], index=dates)
    result = compute_metrics(flat, flat, timestep="day", risk_free_rate=0.0)
    assert result["beta"] == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/metrics.py`**

```python
"""vectorbt-based performance metrics: the dashboard's full `MetricSet` (design spec,
section 6.5) computed from a returns series. The only module importing `vectorbt`, and
only inside `compute_metrics` -- importing `backtesting.metrics` at module level must
never pull vectorbt in.

vectorbt's `returns` accessor computes Sharpe/Sortino/Calmar/Omega/max-drawdown/
annualised-return/annualised-volatility -- the ratios it is built for. Alpha/Beta/
correlation/R^2/Treynor/information-ratio/skew/kurtosis/win-rates/recovery-factor are
plain closed-form statistics (linear regression, pandas' own `.skew()`/`.kurt()`),
computed directly rather than guessed through an uncertain vectorbt method name.
`raw.summary_tables` carries the yearly-returns and drawdown tables `metrics.json`
promises the dashboard (design spec, section 6.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

PERIODS_PER_YEAR = {"day": 252, "minute": 252 * 390}


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None,
    *,
    timestep: str,
    risk_free_rate: float,
) -> dict[str, Any]:
    """`returns`/`benchmark_returns`: daily fractional returns, float64, no NaNs (the
    caller drops the first row of a pct_change() series). Returns a flat dict keyed
    exactly by the dashboard's `MetricSet` field names, plus `raw.summary_tables`."""
    import vectorbt as vbt  # noqa: F401 -- registers the .vbt accessor on pd.Series

    periods = PERIODS_PER_YEAR.get(timestep, 252)
    accessor = returns.vbt.returns(freq="D")

    metrics: dict[str, Any] = {
        "total_return_strategy": float(accessor.total()),
        "cagr_strategy": float(accessor.annualized()),
        "sharpe_strategy": float(accessor.sharpe_ratio(risk_free=risk_free_rate)),
        "sortino_strategy": float(accessor.sortino_ratio()),
        "calmar_strategy": float(accessor.calmar_ratio()),
        "omega_strategy": float(accessor.omega_ratio()),
        "max_drawdown_strategy": float(accessor.max_drawdown()),
        "volatility_strategy": float(accessor.annualized_volatility()),
        "skew_strategy": float(returns.skew()),
        "kurtosis_strategy": float(returns.kurt()),
        "win_days_pct_strategy": float((returns > 0).mean()),
    }
    metrics.update(_drawdown_stats(returns, suffix="strategy"))
    metrics.update(_monthly_win_pct(returns, suffix="strategy"))

    if benchmark_returns is not None and not benchmark_returns.empty:
        bm_accessor = benchmark_returns.vbt.returns(freq="D")
        metrics.update({
            "total_return_benchmark": float(bm_accessor.total()),
            "cagr_benchmark": float(bm_accessor.annualized()),
            "sharpe_benchmark": float(bm_accessor.sharpe_ratio(risk_free=risk_free_rate)),
            "sortino_benchmark": float(bm_accessor.sortino_ratio()),
            "calmar_benchmark": float(bm_accessor.calmar_ratio()),
            "omega_benchmark": float(bm_accessor.omega_ratio()),
            "max_drawdown_benchmark": float(bm_accessor.max_drawdown()),
            "volatility_benchmark": float(bm_accessor.annualized_volatility()),
            "skew_benchmark": float(benchmark_returns.skew()),
            "kurtosis_benchmark": float(benchmark_returns.kurt()),
            "win_days_pct_benchmark": float((benchmark_returns > 0).mean()),
        })
        metrics.update(_drawdown_stats(benchmark_returns, suffix="benchmark"))
        metrics.update(_monthly_win_pct(benchmark_returns, suffix="benchmark"))
        metrics.update(_relative_stats(returns, benchmark_returns, periods, risk_free_rate))

    metrics["raw"] = {
        "summary_tables": {
            "eoy_returns_vs_benchmark": _yearly_table(returns, benchmark_returns),
            "drawdowns": _drawdown_table(returns),
        }
    }
    return metrics


def _drawdown_stats(returns: pd.Series, *, suffix: str) -> dict[str, float]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    underwater = drawdown < 0
    longest = _longest_run(underwater)
    avg_dd = float(drawdown[underwater].mean()) if underwater.any() else 0.0
    max_dd = float(drawdown.min())
    recovery_factor = float(curve.iloc[-1] - 1) / abs(max_dd) if max_dd != 0 else 0.0
    return {
        f"longest_dd_days_{suffix}": float(longest),
        f"avg_drawdown_{suffix}": avg_dd,
        f"recovery_factor_{suffix}": recovery_factor,
    }


def _longest_run(mask: pd.Series) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _monthly_win_pct(returns: pd.Series, *, suffix: str) -> dict[str, float]:
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {f"win_month_pct_{suffix}": float((monthly > 0).mean()) if not monthly.empty else 0.0}


def _relative_stats(
    returns: pd.Series, benchmark_returns: pd.Series, periods: int, risk_free_rate: float
) -> dict[str, float]:
    import numpy as np

    aligned = returns.align(benchmark_returns, join="inner")
    strat, bench = aligned[0].to_numpy(), aligned[1].to_numpy()
    if len(strat) < 2 or np.std(bench) == 0:
        return {
            "beta": 0.0, "alpha": 0.0, "correlation": 0.0, "r_squared_strategy": 0.0,
            "r_squared_benchmark": 0.0, "treynor_ratio": 0.0,
            "information_ratio_strategy": 0.0, "information_ratio_benchmark": 0.0,
        }
    covariance = np.cov(strat, bench)[0, 1]
    beta = covariance / np.var(bench)
    daily_rf = risk_free_rate / periods
    alpha_daily = (np.mean(strat) - daily_rf) - beta * (np.mean(bench) - daily_rf)
    correlation = float(np.corrcoef(strat, bench)[0, 1])
    r_squared = correlation**2
    excess = strat - bench
    tracking_error = np.std(excess, ddof=1)
    information_ratio = (
        float(np.mean(excess) / tracking_error * np.sqrt(periods)) if tracking_error else 0.0
    )
    mean_excess_return = np.mean(strat) - daily_rf
    treynor = float(mean_excess_return * periods / beta) if beta != 0 else 0.0
    return {
        "beta": float(beta), "alpha": float(alpha_daily * periods), "correlation": correlation,
        "r_squared_strategy": float(r_squared), "r_squared_benchmark": float(r_squared),
        "treynor_ratio": treynor,
        "information_ratio_strategy": information_ratio,
        "information_ratio_benchmark": information_ratio,
    }


def _yearly_table(returns: pd.Series, benchmark_returns: pd.Series | None) -> list[dict[str, Any]]:
    years = sorted({ts.year for ts in returns.index})
    rows: list[dict[str, Any]] = []
    for year in years:
        mask = returns.index.year == year
        strat_ret = float((1 + returns[mask]).prod() - 1)
        bench_ret = None
        won = False
        if benchmark_returns is not None:
            b_year = benchmark_returns[benchmark_returns.index.year == year]
            if not b_year.empty:
                bench_ret = float((1 + b_year).prod() - 1)
                won = strat_ret > bench_ret
        rows.append({
            "year": year, "strategy": round(strat_ret, 6),
            "benchmark": round(bench_ret, 6) if bench_ret is not None else None, "won": won,
        })
    return rows


def _drawdown_table(returns: pd.Series) -> list[dict[str, Any]]:
    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    underwater = drawdown < 0
    rows: list[dict[str, Any]] = []
    start = None
    for time, is_under in underwater.items():
        if is_under and start is None:
            start = time
        elif not is_under and start is not None:
            window = drawdown[start:time]
            rows.append({
                "start": str(start.date()), "end": str(time.date()),
                "max_drawdown": round(float(window.min()), 6), "days": int(len(window)),
            })
            start = None
    if start is not None:
        window = drawdown[start:]
        rows.append({
            "start": str(start.date()), "end": str(drawdown.index[-1].date()),
            "max_drawdown": round(float(window.min()), 6), "days": int(len(window)),
        })
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_metrics.py -v`

If a `vectorbt` accessor call raises `AttributeError`, apply the fix described in this task's implementation note above (find the real method name and adjust) rather than changing the test's expected formula, unless investigation shows vectorbt's own convention genuinely differs (e.g. a different risk-free adjustment) — in that case adjust the test's expected formula to match vectorbt's documented convention and note the discrepancy in the commit message.

Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/metrics.py tests/backtesting/test_metrics.py
git commit -m "Add vectorbt-based performance metrics (backtesting/metrics.py)

Full dashboard MetricSet: vectorbt's returns accessor for Sharpe/Sortino/
Calmar/Omega/max-DD/annualised return & volatility; plain closed-form stats
for alpha/beta/correlation/R^2/Treynor/information-ratio/skew/kurtosis/
win-rates/recovery-factor. The float64 boundary (CLAUDE.md amendment,
Task 22) begins here."
```

---

### Task 14: `backtesting/report.py` — write the run to disk + dashboard contract test

**Files:**
- Create: `src/trading_agent_framework/backtesting/report.py`
- Create: `tests/backtesting/dashboard_contract.py` (test fixture, not shipped code)
- Test: `tests/backtesting/test_report.py`

**Interfaces:**
- Consumes: `Ledger`, `FillRecord`, `EquitySample`, `IndicatorLine` (Task 5).
- Produces: `write_settings(run_dir, settings) -> Path`, `write_metrics(run_dir, metrics) -> Path`, `write_equity(run_dir, ledger, benchmark=None) -> Path`, `write_trades(run_dir, ledger) -> Path`, `write_indicators(run_dir, ledger) -> Path` — all consumed by `backtesting/runner.py` (Task 15).

`tests/backtesting/dashboard_contract.py` is a **test-only, dependency-free** mirror of the field *names* the dashboard's `Settings`/`MetricSet` Pydantic models require — not a copy of the models themselves (this project has no `pydantic` dependency, and adding one solely for a test fixture would be scope creep). It exists so a future edit to `report.py`'s output shape gets caught here, in this repo, rather than only discovered when the separate dashboard-migration task runs the real dashboard against real output. Source of truth: `/home/yann/projets/lumibot-trading-agent/src/lumibot_trading_agent/dashboard/models.py`, read during the design spec's research (design spec, section 6.4) — if that file's fields change before the dashboard migration task lands, update this fixture to match.

- [ ] **Step 1: Write the dashboard contract fixture**

`tests/backtesting/dashboard_contract.py`:

```python
"""Test-only mirror of the dashboard's Settings/MetricSet field names -- see the
docstring in tests/backtesting/test_report.py's Task description for why this exists
and how to keep it in sync. Source: lumibot_trading_agent/dashboard/models.py.
"""

from __future__ import annotations

REQUIRED_SETTINGS_FIELDS = frozenset({
    "name", "backtesting_start", "backtesting_end", "budget", "risk_free_rate",
    "backtesting_data_sources", "backtest_time_seconds", "parameters",
})

METRIC_SET_FIELDS = frozenset({
    "total_return_strategy", "total_return_benchmark", "cagr_strategy", "cagr_benchmark",
    "sharpe_strategy", "sharpe_benchmark", "sortino_strategy", "sortino_benchmark",
    "calmar_strategy", "calmar_benchmark", "omega_strategy", "omega_benchmark",
    "max_drawdown_strategy", "max_drawdown_benchmark", "volatility_strategy",
    "volatility_benchmark", "beta", "alpha", "correlation", "treynor_ratio",
    "information_ratio_strategy", "information_ratio_benchmark", "r_squared_strategy",
    "r_squared_benchmark", "skew_strategy", "skew_benchmark", "kurtosis_strategy",
    "kurtosis_benchmark", "win_days_pct_strategy", "win_days_pct_benchmark",
    "win_month_pct_strategy", "win_month_pct_benchmark", "longest_dd_days_strategy",
    "longest_dd_days_benchmark", "avg_drawdown_strategy", "avg_drawdown_benchmark",
    "recovery_factor_strategy", "recovery_factor_benchmark", "raw",
})
```

- [ ] **Step 2: Write the failing tests**

`tests/backtesting/test_report.py`:

```python
from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS, REQUIRED_SETTINGS_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.backtesting.ledger import EquitySample, FillRecord, IndicatorLine, Ledger
from trading_agent_framework.entities.enums import OrderSide, OrderType

NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)
LATER = datetime(2026, 1, 6, 16, tzinfo=UTC)


def _ledger() -> Ledger:
    ledger = Ledger()
    ledger.record_equity(EquitySample(
        time=NOW, portfolio_value=Decimal(10000), cash=Decimal(10000), positions_value=Decimal(0)
    ))
    ledger.record_equity(EquitySample(
        time=LATER, portfolio_value=Decimal(10500), cash=Decimal(500), positions_value=Decimal(10000)
    ))
    ledger.record_fill(FillRecord(
        time=LATER, identifier="abc", symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
        quantity=Decimal(10), filled_quantity=Decimal(10), price=Decimal("1000"),
        trade_cost=Decimal("1.0"), trade_slippage=Decimal("0.5"),
    ))
    ledger.record_line(IndicatorLine(
        time=NOW, name="sma_200", value=Decimal("148.5"), color=None, style="solid", plot_name="default_plot"
    ))
    return ledger


def test_write_settings_round_trips_through_json(tmp_path: Path) -> None:
    settings = {
        "name": "momentum", "backtesting_start": NOW.isoformat(), "backtesting_end": LATER.isoformat(),
        "budget": 10000.0, "risk_free_rate": 0.03, "backtesting_data_sources": "yahoo",
        "backtest_time_seconds": 1.5, "parameters": {"lookback": 20},
    }
    path = report.write_settings(tmp_path, settings)
    loaded = json.loads(path.read_text())
    assert REQUIRED_SETTINGS_FIELDS <= loaded.keys()
    assert loaded["name"] == "momentum"


def test_write_metrics_writes_exactly_the_metric_set_field_names(tmp_path: Path) -> None:
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [], "drawdowns": []}}

    path = report.write_metrics(tmp_path, metrics)
    loaded = json.loads(path.read_text())

    # metrics.json must contain ONLY MetricSet field names -- MetricSet (unlike
    # Settings) has no extra="allow", so a stray key would be silently dropped by
    # the real dashboard's model_validate() rather than raising, which is worse.
    assert set(loaded.keys()) == METRIC_SET_FIELDS


def test_write_equity_produces_a_parquet_file_with_the_expected_columns(tmp_path: Path) -> None:
    path = report.write_equity(tmp_path, _ledger())
    df = pd.read_parquet(path)
    for column in ("portfolio_value", "cash", "positions_value", "return"):
        assert column in df.columns
    assert df["portfolio_value"].iloc[0] == 10000.0
    assert df["portfolio_value"].iloc[1] == 10500.0
    assert df["return"].iloc[1] == pytest_approx(0.05)


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-6)


def test_write_equity_joins_the_benchmark_series_by_timestamp(tmp_path: Path) -> None:
    benchmark = {NOW: Decimal("400.0"), LATER: Decimal("404.0")}
    path = report.write_equity(tmp_path, _ledger(), benchmark)
    df = pd.read_parquet(path)
    assert list(df["benchmark_close"]) == [400.0, 404.0]
    assert df["benchmark_return"].iloc[1] == pytest_approx(0.01)


def test_write_trades_produces_a_parquet_file_with_the_dashboards_expected_columns(tmp_path: Path) -> None:
    path = report.write_trades(tmp_path, _ledger())
    df = pd.read_parquet(path)
    for column in (
        "time", "symbol", "side", "status", "order_type", "quantity", "filled_quantity",
        "price", "trade_cost", "trade_slippage", "identifier", "event_kind",
    ):
        assert column in df.columns
    assert df["status"].iloc[0] == "fill"
    assert df["side"].iloc[0] == "buy"


def test_write_indicators_produces_a_parquet_file_with_the_dashboards_expected_columns(tmp_path: Path) -> None:
    path = report.write_indicators(tmp_path, _ledger())
    df = pd.read_parquet(path)
    for column in ("datetime", "name", "value", "color", "style", "plot_name"):
        assert column in df.columns
    assert df["name"].iloc[0] == "sma_200"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Write `backtesting/report.py`**

```python
"""Serialises a finished backtest run to disk: settings.json, metrics.json, and the
three parquet time series (equity/trades/indicators). description.json is never
written here -- it is the dashboard's own file, created only when a user adds a
description through the dashboard UI.

This is the codebase's second half of the third float boundary (design spec,
section 7.2, alongside metrics.py): every value here is converted from the ledger's
Decimal to float only at the point it's about to leave the process.
"""

from __future__ import annotations

import json
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.ledger import Ledger


def _float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def write_settings(run_dir: Path, settings: dict[str, Any]) -> Path:
    path = run_dir / "settings.json"
    path.write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def write_equity(
    run_dir: Path, ledger: Ledger, benchmark: dict[datetime, Decimal] | None = None
) -> Path:
    import pandas as pd

    rows = [
        {
            "datetime": sample.time,
            "portfolio_value": _float(sample.portfolio_value),
            "cash": _float(sample.cash),
            "positions_value": _float(sample.positions_value),
            "benchmark_close": _float(benchmark.get(sample.time)) if benchmark else None,
        }
        for sample in ledger.equity
    ]
    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df["return"] = df["portfolio_value"].pct_change()
    df["benchmark_return"] = (
        df["benchmark_close"].pct_change() if benchmark else pd.Series(dtype="float64")
    )
    path = run_dir / "equity.parquet"
    df.to_parquet(path)
    return path


def write_trades(run_dir: Path, ledger: Ledger) -> Path:
    import pandas as pd

    rows = [
        {
            "time": f.time, "symbol": f.symbol, "side": f.side.value, "status": f.status,
            "order_type": f.order_type.value, "quantity": _float(f.quantity),
            "filled_quantity": _float(f.filled_quantity), "price": _float(f.price),
            "trade_cost": _float(f.trade_cost), "trade_slippage": _float(f.trade_slippage),
            "identifier": f.identifier, "event_kind": f.event_kind,
        }
        for f in ledger.fills
    ]
    df = pd.DataFrame(rows)
    path = run_dir / "trades.parquet"
    df.to_parquet(path)
    return path


def write_indicators(run_dir: Path, ledger: Ledger) -> Path:
    import pandas as pd

    rows = [
        {
            "datetime": line.time, "name": line.name, "value": _float(line.value),
            "color": line.color, "style": line.style, "plot_name": line.plot_name,
        }
        for line in ledger.lines
    ]
    df = pd.DataFrame(rows)
    path = run_dir / "indicators.parquet"
    df.to_parquet(path)
    return path
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_report.py -v`
Expected: PASS

Clean up the test file's inline `pytest_approx` helper if it turns out simpler to just `import pytest` at the top and call `pytest.approx(...)` directly in each test — the inline-function version above works but is an unnecessary indirection; prefer the direct import.

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/backtesting/report.py tests/backtesting/test_report.py tests/backtesting/dashboard_contract.py
git commit -m "Add backtesting/report.py: write settings/metrics/equity/trades/indicators

metrics.json is asserted to contain exactly the dashboard's MetricSet field
names via a dependency-free local contract fixture (no pydantic added to
this project just for the test)."
```

---

### Task 15: `backtesting/runner.py` — orchestration

**Files:**
- Create: `src/trading_agent_framework/backtesting/runner.py`
- Test: `tests/backtesting/test_runner.py`

**Interfaces:**
- Consumes: `BacktestBroker` (Tasks 7-8), `BacktestClock` (Task 6), `BacktestDataSource`/`FULL_HISTORY` (Task 3), `metrics.compute_metrics` (Task 13), `report.write_*` (Task 14), `Strategy` (existing, unmodified by this task), `setup_strategy_logging` (existing).
- Produces: `BacktestResult(run_dir: Path, settings: dict, metrics: dict)` (frozen dataclass), `run_backtest(strategy, *, start, end, budget, data_source, benchmark, timestep, commission, slippage, risk_free_rate) -> BacktestResult` — consumed by `Strategy.run_backtesting()` (Task 17).

- [ ] **Step 1: Write the failing test**

`tests/backtesting/test_runner.py`:

```python
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from tests.backtesting.fakes import FakeBacktestDataSource

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

    strategy = BuyOnceStrategy(broker=None, project_root=tmp_path)  # ty: ignore[invalid-argument-type]

    result = run_backtest(
        strategy, start=sessions[0].open - timedelta(hours=1), end=sessions[-1].close,
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

    strategy = BuyOnceStrategy(broker=None, project_root=tmp_path)  # ty: ignore[invalid-argument-type]
    run_backtest(
        strategy, start=sessions[0].open - timedelta(hours=1), end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0,
    )

    assert isinstance(strategy.broker, BacktestBroker)
    assert isinstance(strategy.clock, BacktestClock)
```

Check `Strategy.__init__`'s signature before finalizing this test: it does not currently accept `broker=None` (the parameter is required, positional, typed `Broker`). Passing `None` works at runtime (Python doesn't enforce the type hint), which is why the `# ty: ignore[invalid-argument-type]` comment is needed — `run_backtest` immediately overwrites `strategy.broker` before `executor.run()` is ever called, so the placeholder `None` is never actually used as a broker. This mirrors how `tests/core/test_runners.py` constructs strategies against a `FakeBroker` purely to satisfy the constructor and then may reassign; confirm this pattern still type-checks/runs cleanly, and if `ty`/`pyright` in this project's CI is strict about it, construct a trivial placeholder instead — check `tests/fakes.py` for whether a `None`-clock/broker pattern already exists elsewhere before inventing a new one.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write `backtesting/runner.py`**

```python
"""Orchestrates one backtest run end to end: builds the simulated clock/broker, runs
the strategy through the (unmodified) executor, computes metrics, and writes the
report. The only module that wires `Strategy` to the backtesting subsystem --
`Strategy.run_backtesting()` (a later task) is a thin wrapper around `run_backtest`.
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
from trading_agent_framework.backtesting.data.base import FULL_HISTORY, BacktestDataSource
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.log import setup_strategy_logging

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
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
    import pandas as pd

    from trading_agent_framework.backtesting import metrics as metrics_module
    from trading_agent_framework.backtesting import report

    benchmark_asset = Asset(benchmark)
    data_source.load([benchmark_asset], start, end, timestep)
    sessions = data_source.sessions(start, end)

    clock = BacktestClock(start=start, sessions=sessions)
    broker = BacktestBroker(
        strategy.name, data_source=data_source, clock=clock, budget=budget,
        timestep=timestep, commission=commission, slippage=slippage,
    )
    clock.on_advance = broker.on_advance

    strategy.broker = broker
    strategy.clock = clock
    strategy.trading_mode = TradingMode.BACKTESTING

    log_file = setup_strategy_logging(
        strategy.name, TradingMode.BACKTESTING, project_root=strategy.project_root
    )
    run_dir = log_file.parent

    started = time.monotonic()
    strategy.executor.run()
    elapsed = time.monotonic() - started

    benchmark_bars = data_source.bars(benchmark_asset, end, FULL_HISTORY, timestep)
    benchmark_series = (
        pd.Series(benchmark_bars.df["close"].to_numpy(), index=benchmark_bars.df.index)
        if benchmark_bars is not None else None
    )
    benchmark_by_time = (
        {ts: Decimal(str(v)) for ts, v in benchmark_series.items()}
        if benchmark_series is not None else None
    )

    report.write_equity(run_dir, broker.ledger, benchmark_by_time)
    report.write_trades(run_dir, broker.ledger)
    report.write_indicators(run_dir, broker.ledger)

    equity_index = [s.time for s in broker.ledger.equity]
    equity_values = [float(s.portfolio_value) for s in broker.ledger.equity]
    portfolio_returns = pd.Series(equity_values, index=equity_index).pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna() if benchmark_series is not None else None

    computed_metrics = metrics_module.compute_metrics(
        portfolio_returns, benchmark_returns, timestep=timestep, risk_free_rate=risk_free_rate
    )
    report.write_metrics(run_dir, computed_metrics)

    settings = {
        "name": strategy.name, "mode": "backtesting", "run_ts": run_dir.name,
        "backtesting_start": start.isoformat(), "backtesting_end": end.isoformat(),
        "budget": float(budget), "risk_free_rate": risk_free_rate,
        "backtesting_data_sources": data_source.name, "backtest_time_seconds": elapsed,
        "timestep": timestep, "sleeptime": strategy.sleeptime,
        "commission": float(commission), "slippage": float(slippage),
        "benchmark_symbol": benchmark, "framework_version": __version__,
        "parameters": dict(strategy.parameters),
    }
    report.write_settings(run_dir, settings)

    return BacktestResult(run_dir=run_dir, settings=settings, metrics=computed_metrics)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_runner.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/runner.py tests/backtesting/test_runner.py
git commit -m "Add backtesting/runner.py: end-to-end run orchestration"
```

---

### Task 16: `backtesting/__init__.py` — lazy exports

**Files:**
- Modify: `src/trading_agent_framework/backtesting/__init__.py` (currently empty, from Task 3)
- Test: `tests/backtesting/test_lazy_imports.py`

**Interfaces:**
- Produces: `trading_agent_framework.backtesting.{BacktestDataSource, BacktestBroker, BacktestClock, CachedDataSource, YahooBacktestData, AlpacaBacktestData, run_backtest, BacktestResult}` — the package's public surface. Consumed by nothing further in this plan (Task 17 imports submodules directly, deferred inside method bodies, matching `agents/manager.py`'s pattern) but this is the surface a strategy author or a later task reaches for.

- [ ] **Step 1: Write the failing tests**

`tests/backtesting/test_lazy_imports.py` (mirrors `tests/brokers/test_lazy_imports.py`'s structure exactly):

```python
from __future__ import annotations

import subprocess
import sys

import pytest

from trading_agent_framework import backtesting
from trading_agent_framework.backtesting.data import base as base_module


def test_backtest_data_source_is_eagerly_reexported() -> None:
    assert backtesting.BacktestDataSource is base_module.BacktestDataSource


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        backtesting.does_not_exist  # noqa: B018


def test_lazy_attributes_resolve_to_the_real_classes_in_process() -> None:
    from trading_agent_framework.backtesting.broker import BacktestBroker as direct_broker
    from trading_agent_framework.backtesting.clock import BacktestClock as direct_clock
    from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData as direct_alpaca
    from trading_agent_framework.backtesting.data.cache import CachedDataSource as direct_cache
    from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData as direct_yahoo
    from trading_agent_framework.backtesting.runner import BacktestResult as direct_result
    from trading_agent_framework.backtesting.runner import run_backtest as direct_run

    assert backtesting.BacktestBroker is direct_broker
    assert backtesting.BacktestClock is direct_clock
    assert backtesting.AlpacaBacktestData is direct_alpaca
    assert backtesting.CachedDataSource is direct_cache
    assert backtesting.YahooBacktestData is direct_yahoo
    assert backtesting.BacktestResult is direct_result
    assert backtesting.run_backtest is direct_run
    # Second access hits the globals() cache set by __getattr__, not the _LAZY branch again.
    assert backtesting.BacktestBroker is direct_broker


def test_dir_includes_lazy_and_eager_names() -> None:
    names = dir(backtesting)
    for name in (
        "BacktestDataSource", "BacktestBroker", "BacktestClock", "CachedDataSource",
        "YahooBacktestData", "AlpacaBacktestData", "run_backtest", "BacktestResult",
    ):
        assert name in names


def test_importing_backtesting_package_does_not_import_vectorbt_or_yfinance() -> None:
    """Must run in a subprocess -- see tests/brokers/test_lazy_imports.py's identical
    reasoning: other test modules have already imported these by the time this test
    runs in-process, so only a fresh interpreter makes the assertion meaningful."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import trading_agent_framework.backtesting\n"
            "import sys\n"
            "assert 'vectorbt' not in sys.modules\n"
            "assert 'numba' not in sys.modules\n"
            "assert 'yfinance' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_run_backtest_lazily_imports_vectorbt_is_not_true_until_called() -> None:
    """run_backtest itself is light to import (metrics.py's vectorbt import is deferred
    further, inside compute_metrics) -- confirm importing the function doesn't pull
    vectorbt in either, only actually calling compute_metrics does (not exercised here)."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from trading_agent_framework.backtesting import run_backtest\n"
            "import sys\n"
            "assert 'vectorbt' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_yahoo_backtest_data_does_not_import_yfinance_until_used() -> None:
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from trading_agent_framework.backtesting import YahooBacktestData\n"
            "import sys\n"
            "assert 'yfinance' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/backtesting/test_lazy_imports.py -v`
Expected: FAIL — `AttributeError: module 'trading_agent_framework.backtesting' has no attribute 'BacktestDataSource'` (the package is currently empty)

- [ ] **Step 3: Write `backtesting/__init__.py`**

```python
"""Backtesting subsystem: the third trading mode.

`vectorbt`+`numba` (backtesting/metrics.py) and `yfinance`
(backtesting/data/yahoo.py) are heavy and imported only inside method bodies
there; importing this package alone must not pull them in -- only actually
calling into a codepath that needs them does. Mirrors `brokers/__init__.py`.
"""

from __future__ import annotations

from trading_agent_framework.backtesting.data.base import BacktestDataSource

__all__ = [
    "AlpacaBacktestData",
    "BacktestBroker",
    "BacktestClock",
    "BacktestDataSource",
    "BacktestResult",
    "CachedDataSource",
    "YahooBacktestData",
    "run_backtest",
]

_LAZY = {
    "BacktestBroker": (".broker", "BacktestBroker"),
    "BacktestClock": (".clock", "BacktestClock"),
    "CachedDataSource": (".data.cache", "CachedDataSource"),
    "YahooBacktestData": (".data.yahoo", "YahooBacktestData"),
    "AlpacaBacktestData": (".data.alpaca", "AlpacaBacktestData"),
    "run_backtest": (".runner", "run_backtest"),
    "BacktestResult": (".runner", "BacktestResult"),
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        module_path, attr = _LAZY[name]
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + __all__
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/backtesting/test_lazy_imports.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/backtesting/__init__.py tests/backtesting/test_lazy_imports.py
git commit -m "Add lazy re-exports for the backtesting package

Importing trading_agent_framework.backtesting never pulls in vectorbt,
numba or yfinance -- only touching the lazy-exported name that needs one
does, mirroring brokers/__init__.py."
```

---

### Task 17: `Strategy.add_line` and backtest config class attributes

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py`
- Test: `tests/core/test_strategy.py`

**Interfaces:**
- Consumes: `IndicatorLine`, `Ledger` (Task 5); `BacktestBroker` (Tasks 7-8) — both imported inside the method body, never at module level (`core/strategy.py` is imported by every strategy, including ones that never backtest).
- Produces: `Strategy.add_line(name, value, *, color=None, style="solid", plot_name="default_plot") -> None`; class attributes `backtesting_start: datetime | None = None`, `backtesting_end: datetime | None = None`, `budget: Decimal = Decimal("10000")`, `benchmark_symbol: str = "SPY"` — consumed by `Strategy.run_backtesting()` (Task 18).

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_strategy.py`, reusing that file's existing `_broker(**kwargs) -> FakeBroker` helper (defined at the top of the file as `FakeBroker(FakeClock(_START), **kwargs)`) for the no-op case, and building a `BacktestBroker` directly for the recording case:

```python
def test_add_line_is_a_no_op_outside_backtesting() -> None:
    strategy = Strategy(_broker())  # mode defaults to PAPER

    strategy.add_line("sma_200", 148.5)  # must not raise, must not touch anything

    assert strategy.trading_mode is TradingMode.PAPER  # sanity: no exception occurred


def test_add_line_records_an_indicator_line_in_backtesting() -> None:
    from tests.backtesting.fakes import FakeBacktestDataSource
    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.backtesting.clock import BacktestClock

    source = FakeBacktestDataSource()
    clock = BacktestClock(start=_START, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING)

    strategy.add_line("sma_200", 148.5, color="red", style="dashed", plot_name="overlay")

    [line] = broker.ledger.lines
    assert line.name == "sma_200"
    assert line.value == Decimal("148.5")
    assert line.color == "red"
    assert line.style == "dashed"
    assert line.plot_name == "overlay"
    assert line.time == clock.now()


def test_backtest_class_attribute_defaults() -> None:
    assert Strategy.backtesting_start is None
    assert Strategy.backtesting_end is None
    assert Strategy.budget == Decimal("10000")
    assert Strategy.benchmark_symbol == "SPY"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/test_strategy.py -v -k add_line`
Expected: FAIL — `AttributeError: 'Strategy' object has no attribute 'add_line'`

- [ ] **Step 3: Add `TYPE_CHECKING` import and the class attributes**

In `src/trading_agent_framework/core/strategy.py`, change:

```python
from typing import Any
```
to:
```python
from typing import TYPE_CHECKING, Any
```

Add after the existing imports, before `logger = logging.getLogger(__name__)`:

```python
if TYPE_CHECKING:
    from trading_agent_framework.backtesting.data.base import BacktestDataSource
    from trading_agent_framework.backtesting.runner import BacktestResult
```

Add to the `class Strategy:` body, alongside the existing `sleeptime`/`minutes_before_opening`/etc. class attributes:

```python
    # backtesting defaults (Strategy.run_backtesting()), overridable per call
    backtesting_start: datetime | None = None
    backtesting_end: datetime | None = None
    budget: Decimal = Decimal("10000")
    benchmark_symbol: str = "SPY"
```

- [ ] **Step 4: Add `add_line`**

Add this method to `Strategy`, in the "control" section near `sleep`/`stop`:

```python
    def add_line(
        self, name: str, value: Number, *, color: str | None = None,
        style: str = "solid", plot_name: str = "default_plot",
    ) -> None:
        """Record a charted value at the current simulated time (lumibot-compatible
        signature). No-op outside backtesting."""
        if not self.is_backtesting:
            return
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.ledger import IndicatorLine

        if not isinstance(self.broker, BacktestBroker):
            return
        self.broker.ledger.record_line(
            IndicatorLine(
                time=self.clock.now(), name=name, value=_to_decimal(value),
                color=color, style=style, plot_name=plot_name,
            )
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/core/test_strategy.py -v -k add_line`
Expected: PASS

- [ ] **Step 6: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py tests/core/test_strategy.py
git commit -m "Add Strategy.add_line and backtest config class attributes

No-op outside backtesting; records into the BacktestBroker's ledger inside
it. backtesting/broker.py and backtesting/ledger.py are imported only
inside the method body -- live/paper strategies never pay for them."
```

---

### Task 18: `Strategy.run_backtesting()` — the public entry point

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py`
- Modify: `tests/core/test_runners.py` (remove the now-obsolete "not implemented" test; add the end-to-end test)

**Interfaces:**
- Consumes: `run_backtest`, `BacktestResult` (Task 15); `YahooBacktestData` (Task 11) as the default `data_source`; `BacktestDataSource` (Task 3, `TYPE_CHECKING`-only, added in Task 17).
- Produces: `Strategy.run_backtesting(*, start=None, end=None, budget=None, data_source=None, benchmark=None, timestep="day", commission=Decimal(0), slippage=Decimal(0), risk_free_rate=0.0) -> BacktestResult` — this is the method every backtesting strategy actually calls; every earlier task exists to make this one call correct.

- [ ] **Step 1: Write the failing tests**

In `tests/core/test_runners.py`, **delete** `test_run_backtesting_is_not_implemented_yet` (the method it tests no longer raises) and add:

```python
def test_run_backtesting_requires_start_and_end(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    with pytest.raises(ConfigurationError, match="start/end"):
        strategy.run_backtesting()


def test_run_backtesting_runs_end_to_end_via_the_public_api(tmp_path: Path) -> None:
    from datetime import time, timedelta
    from zoneinfo import ZoneInfo

    import pandas as pd
    from tests.backtesting.fakes import FakeBacktestDataSource

    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.utils.clock import MarketSession

    et_tz = ZoneInfo("America/New_York")

    def _sessions(first_day: date, count: int) -> list[MarketSession]:
        result: list[MarketSession] = []
        day = first_day
        while len(result) < count:
            if day.weekday() < 5:
                result.append(MarketSession(
                    open=datetime.combine(day, time(9, 30), tzinfo=et_tz),
                    close=datetime.combine(day, time(16, 0), tzinfo=et_tz),
                ))
            day += timedelta(days=1)
        return result

    sessions = _sessions(date(2026, 1, 5), 3)
    closes = [150.0, 151.0, 152.0]
    df = pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
         "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([s.close for s in sessions], name="timestamp"),
    )
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(Asset("AAPL"), df)
    source.set_bars(Asset("SPY"), df)

    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    result = strategy.run_backtesting(
        start=sessions[0].open - timedelta(hours=1), end=sessions[-1].close,
        data_source=source, benchmark="SPY",
    )

    assert result.run_dir.is_dir()
    assert (result.run_dir / "metrics.json").is_file()
    assert (result.run_dir / "settings.json").is_file()
    assert isinstance(strategy.broker, BacktestBroker)  # rebound by run_backtesting
```

Add `from datetime import date` to the file's existing `from datetime import date` import line if it isn't already imported that way (the file already imports `date` per its Step 1 usage in `_strategy`, per this plan's earlier reading of `tests/core/test_runners.py:1-20`).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/test_runners.py -v -k run_backtesting`
Expected: FAIL — the "requires start/end" test fails with `NotImplementedError` instead of `ConfigurationError` (today's behaviour); the end-to-end test fails the same way.

- [ ] **Step 3: Replace `run_backtesting`'s body**

In `src/trading_agent_framework/core/strategy.py`, replace:

```python
    def run_backtesting(self) -> None:
        raise NotImplementedError(
            "backtesting is not implemented yet; it ships with the backtesting subproject"
        )
```

with:

```python
    def run_backtesting(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        budget: Number | None = None,
        data_source: BacktestDataSource | None = None,
        benchmark: str | None = None,
        timestep: str = "day",
        commission: Number = Decimal(0),
        slippage: Number = Decimal(0),
        risk_free_rate: float = 0.0,
    ) -> BacktestResult:
        """Run this strategy against simulated time and simulated fills.

        `start`/`end`/`budget`/`benchmark` fall back to the `backtesting_start`/
        `backtesting_end`/`budget`/`benchmark_symbol` class attributes when omitted.
        `data_source` defaults to a Yahoo daily source over [start, end] (no on-disk
        cache by default -- wrap it in `backtesting.CachedDataSource` for repeat runs).
        """
        from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
        from trading_agent_framework.backtesting.runner import run_backtest

        resolved_start = start if start is not None else self.backtesting_start
        resolved_end = end if end is not None else self.backtesting_end
        if resolved_start is None or resolved_end is None:
            raise ConfigurationError(
                "run_backtesting needs start/end, either as arguments or as "
                "backtesting_start/backtesting_end class attributes"
            )
        resolved_budget = _to_decimal(budget) if budget is not None else self.budget
        resolved_source = (
            data_source if data_source is not None
            else YahooBacktestData(resolved_start, resolved_end)
        )
        return run_backtest(
            self, start=resolved_start, end=resolved_end, budget=resolved_budget,
            data_source=resolved_source, benchmark=benchmark or self.benchmark_symbol,
            timestep=timestep, commission=_to_decimal(commission), slippage=_to_decimal(slippage),
            risk_free_rate=risk_free_rate,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/test_runners.py -v`
Expected: PASS, including every other pre-existing test in the file.

- [ ] **Step 5: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS — every module built across Tasks 1-17 now has an end-to-end caller.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py tests/core/test_runners.py
git commit -m "Wire Strategy.run_backtesting() to the backtesting subsystem

The public entry point: builds a Yahoo-backed cache-free default data
source, delegates to backtesting.runner.run_backtest, returns a
BacktestResult. Replaces the NotImplementedError stub."
```

---

### Task 19: `pyproject.toml` — dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `vectorbt`, `pyarrow` as main dependencies; `yfinance` under a `backtesting-yahoo` optional-dependencies extra (design spec, section 7.1). Nothing later depends on this task's *code* (there is none) — every earlier task that runs `uv run pytest` against `vectorbt`/pyarrow-touching code (Tasks 13, 14, 15, 16, 18) implicitly needs this done first in practice, even though it's placed here in the plan for a clean, reviewable diff. **Do this task before Task 13**, not after — see the note below.

**Sequencing note:** unlike every other task in this plan, this one has no failing test of its own to drive it — it is infrastructure. Apply it **immediately before starting Task 13** (the first task that imports `vectorbt`), not at the end of the plan; it is placed here in the document only to keep the task list in the same order as the spec's own section numbering. A subagent-driven or sequential executor should treat this as "Task 12.5" in actual execution order.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, change:

```toml
dependencies = [
    "alpaca-py>=0.44.0,<0.45",
    "langchain>=1.0,<2.0",
    "langchain-openai>=1.0",
    "pandas-ta-classic>=0.6.52,<0.7",
    "python-dotenv>=1.2",
]
```

to:

```toml
dependencies = [
    "alpaca-py>=0.44.0,<0.45",
    "langchain>=1.0,<2.0",
    "langchain-openai>=1.0",
    "pandas-ta-classic>=0.6.52,<0.7",
    "python-dotenv>=1.2",
    "vectorbt>=1.1.0,<2.0",
    "pyarrow>=15.0.0",
]

[project.optional-dependencies]
backtesting-yahoo = ["yfinance>=0.2.61"]
```

- [ ] **Step 2: Sync and verify the install**

Run: `uv sync`
Expected: resolves cleanly (design spec, section 1.2 already verified `vectorbt` 1.1.0 against this project's pinned `pandas`/`numpy`/Python 3.14).

Run: `uv run python -c "import vectorbt, pyarrow; print(vectorbt.__version__, pyarrow.__version__)"`
Expected: prints both version strings without error.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS (this is where any task from 13 onward that was blocked on missing `vectorbt`/`pyarrow` now actually runs green for the first time, if this task was applied at the point the sequencing note describes).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add vectorbt and pyarrow dependencies; yfinance as an optional extra

vectorbt/pyarrow are main dependencies (every backtest run needs them for
reporting); yfinance is optional (backtesting-yahoo extra) since it pulls
12 transitive dependencies and is only needed by the default data source."
```

---

### Task 20: `scripts/tests/smoke_backtest.py` — manual real-data smoke test

**Files:**
- Create: `scripts/tests/smoke_backtest.py`

**Interfaces:**
- Consumes: `Strategy.run_backtesting()` (Task 18), the real `YahooBacktestData` default. No test suite consumes this file — like the other three scripts in `scripts/tests/`, it is excluded from `tests/` and run by hand (it touches the real network).

- [ ] **Step 1: Write the script**

`scripts/tests/smoke_backtest.py`:

```python
"""Manual smoke test: run a trivial buy-and-hold strategy through
Strategy.run_backtesting() against real Yahoo Finance data, over a short recent
window, and print the resulting metrics and output files.

Excluded from `tests/` on purpose (see scripts/tests/smoke_alpaca_data.py and its
siblings for the same convention) -- this hits the real network and is meant to be
run by hand, not in CI.

Run: uv run python scripts/tests/smoke_backtest.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset

ET = ZoneInfo("America/New_York")
AAPL = Asset("AAPL")


class BuyAndHold(Strategy):
    sleeptime = "1D"
    benchmark_symbol = "SPY"

    def on_trading_iteration(self) -> None:
        if self.first_iteration:
            price = self.get_last_price(AAPL)
            if price is not None:
                quantity = (self.get_cash() * Decimal("0.9") / price).quantize(Decimal(1))
                if quantity > 0:
                    self.submit_order(self.create_order(AAPL, quantity, "buy"))


def main() -> None:
    end = datetime.now(ET)
    start = end - timedelta(days=120)

    # A Strategy needs a broker to construct, but run_backtesting() rebinds it before
    # the first bar is ever touched -- any Broker instance works as a placeholder.
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
    from trading_agent_framework.config.env import load_strategy_env, AlpacaCredentials

    load_strategy_env("smoke_backtest", "paper")
    placeholder_broker = AlpacaBroker.from_credentials(
        "smoke_backtest", AlpacaCredentials.from_env(), with_stream=False
    )

    strategy = BuyAndHold(placeholder_broker, mode=TradingMode.BACKTESTING)
    result = strategy.run_backtesting(start=start, end=end)

    print(f"Run directory: {result.run_dir}")
    print(f"Sharpe (strategy): {result.metrics['sharpe_strategy']:.3f}")
    print(f"Max drawdown (strategy): {result.metrics['max_drawdown_strategy']:.3%}")
    print(f"Total return (strategy): {result.metrics['total_return_strategy']:.3%}")
    for filename in ("settings.json", "metrics.json", "equity.parquet", "trades.parquet", "indicators.parquet"):
        path = result.run_dir / filename
        print(f"  {filename}: {'OK' if path.is_file() else 'MISSING'}")


if __name__ == "__main__":
    main()
```

Check the exact import path/signature of `load_strategy_env`/`AlpacaCredentials`/`AlpacaBroker.from_credentials` against the other three scripts in `scripts/tests/` (e.g. `scripts/tests/smoke_alpaca_data.py`) before finalizing — reuse whatever bootstrap pattern they already establish rather than diverging, since it needs valid paper-trading credentials in `env/.env.smoke_backtest.paper` (or the shared `env/.env` fallback) to construct the placeholder broker.

- [ ] **Step 2: Run it by hand once, to confirm it works**

Run: `uv run python scripts/tests/smoke_backtest.py`
Expected: prints a run directory and three real metric values; exits 0. (Requires real Yahoo network access and valid Alpaca paper credentials for the placeholder broker; if paper credentials aren't available in this environment, skip actually running it and note that in the commit message — the file itself, being excluded from `tests/`, is still a valid deliverable either way.)

- [ ] **Step 3: Commit**

```bash
git add scripts/tests/smoke_backtest.py
git commit -m "Add scripts/tests/smoke_backtest.py: manual real-Yahoo-data smoke test"
```

---

### Task 21: `README.md` — backtesting section

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation only).

Note before starting: the current `README.md` is entirely about env-file management (naming convention, resolution order, getting started) — `CLAUDE.md` points at it specifically for that. This task adds a clearly separated new top-level section at the end rather than interleaving with the env-file content, so the file's existing scope stays legible.

- [ ] **Step 1: Append the section**

Add to the end of `README.md`:

```markdown

## Backtesting

`Strategy.run_backtesting(start=..., end=...)` runs a strategy against simulated time
and simulated fills — no network calls to a broker, and (with the default data
source) no Alpaca account needed at all.

```python
from datetime import datetime, timedelta

result = my_strategy.run_backtesting(
    start=datetime.now() - timedelta(days=365),
    end=datetime.now(),
)
print(result.metrics["sharpe_strategy"], result.run_dir)
```

- **Data source**: defaults to `YahooBacktestData` — free daily OHLCV, official
  closing-auction closes, no API key. Pass `data_source=AlpacaBacktestData(...)` for
  feed parity with paper/live trading, or wrap either in
  `backtesting.CachedDataSource(source, cache_dir)` for network-free reruns against
  the same window (useful when iterating on an agent prompt against a fixed period).
  `YahooBacktestData` needs the `backtesting-yahoo` extra: `uv sync --extra backtesting-yahoo`.
- **Fill model**: orders fill against the *next* bar's open (never the bar they were
  submitted on), so a strategy can't trade a price it has already observed. Limit and
  stop orders fill only when the bar's range actually touches the trigger price.
- **Output**: `logs/<strategy>/backtesting/<timestamp>_backtesting/` — `settings.json`,
  `metrics.json` (Sharpe, Sortino, Calmar, max drawdown, and the rest of the standard
  tearsheet), and three parquet files (`equity.parquet`, `trades.parquet`,
  `indicators.parquet`) shaped for the strategy dashboard.
- **Cache**: `CachedDataSource` writes to `cache/backtesting/<provider>/` by default;
  delete that directory to force a full refetch.
- **No look-ahead, structurally**: every price the strategy can see is gated by
  `clock.now()` — a bar is visible only after it has *closed*. See
  `docs/superpowers/specs/2026-09-12-backtesting-framework-design.md` for the full
  design and its guarantees.
```

- [ ] **Step 2: Proofread against the actual public API**

Confirm every name mentioned (`run_backtesting`, `result.metrics`, `result.run_dir`, `YahooBacktestData`, `AlpacaBacktestData`, `CachedDataSource`, the `backtesting-yahoo` extra) matches exactly what Tasks 11, 12, 10, 18 and 19 actually produced — fix any drift before committing.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Document backtesting in README.md"
```

---

### Task 22: `CLAUDE.md` — three amendments (design spec, section 7.2)

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none (documentation only). **Read the file fresh before editing** — the anchor text below is quoted from this plan's own research phase and must be confirmed against the actual current file (it may have moved or been touched by an unrelated change since); do not apply these edits blind.

- [ ] **Step 1: Amend the money-boundary rule**

Find the bullet beginning `- **Money is \`Decimal\`** everywhere except two deliberate float boundaries...` in the "Key patterns / gotchas" section. Replace it with:

```markdown
- **Money is `Decimal`** everywhere except three deliberate float boundaries: `orders.py`
  (Alpaca's SDK wants floats for some request fields), `Bars.df` (float64 OHLCV for
  indicator maths, built only in `market_data._bars_frame`), and the backtesting
  ledger-to-reporting seam (`backtesting/metrics.py` and `backtesting/report.py` --
  vectorbt and JSON/parquet output both need float64; everything upstream of that seam,
  including `backtesting/broker.py`, `backtesting/fills.py` and `backtesting/ledger.py`,
  stays exact `Decimal`). Don't add a fourth.
```

- [ ] **Step 2: Add the no-look-ahead contract**

In the same "Key patterns / gotchas" section, add a new bullet near the existing **Market data:** bullet:

```markdown
- **No data tool may use wall-clock time.** A backtest's no-look-ahead guarantee
  (`backtesting/data/base.py`) holds only as far as the chokepoint it controls --
  `BacktestDataSource.bars()`. Any other source of "now" inside a data-fetching tool
  (news, SEC fundamentals, a future screener) must take its cutoff from
  `strategy.clock.now()`, exactly as `MemoryStore` already takes `now=self.clock.now`,
  or it silently leaks future information into a backtest.
```

- [ ] **Step 3: Add the architecture entry**

In the `## Architecture` section's module list, add a new bullet after the existing `agents/` bullet (before the `log.py` bullet):

```markdown
- `backtesting/` -- the third trading mode. `clock.py` (`BacktestClock`, simulated time),
  `broker.py` (`BacktestBroker`, simulated fills via `fills.py`'s pure OHLC rules),
  `ledger.py` (fills/equity/indicator lines, `Decimal`), `data/` (`BacktestDataSource`
  ABC, `CachedDataSource`, `YahooBacktestData` default, `AlpacaBacktestData`),
  `metrics.py` (vectorbt reporting) and `report.py` (writes the run to
  `logs/<strategy>/backtesting/<ts>_backtesting/`) are the pure-to-I/O layers;
  `runner.py` orchestrates a run and `Strategy.run_backtesting()` is the public entry
  point. The executor itself needed no changes -- `MarketClock.max_wait_slice`
  (`60.0` live, `math.inf` simulated) is the only seam it exposed.
```

- [ ] **Step 4: Proofread**

Read the whole `CLAUDE.md` back and confirm no duplicate bullets, no contradiction with the two edits, and that the amended money-boundary bullet reads coherently in context.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "Document the backtesting subsystem in CLAUDE.md

Three amendments: the third Decimal/float boundary, the no-look-ahead
contract for future data tools, and the backtesting/ architecture entry."
```

---

### Task 23: `TODO.md` — close out the backtesting item

**Files:**
- Modify: `TODO.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Tick the completed item and add follow-ups**

In `TODO.md`'s `## MIGRATION` section, change:

```markdown
* Backtesting (vectorbt)
```

to:

```markdown
* ~~Backtesting (vectorbt)~~
```

matching the file's existing convention for completed items (`~~Alpaca broker~~`, `~~Memory~~`, etc., already in the file).

Leave `* Cache management` and `* dashboard` exactly as they are (both still open; the backtesting cache built in this plan is deliberately scoped to what `CachedDataSource` needs, not the broader TTL/purge/remote-cache item).

- [ ] **Step 2: Add the two follow-ups this plan identified but deliberately deferred**

In the `## General` section, add two new bullets (matching the file's existing terse, imperative style):

```markdown
* Cache LLM calls during backtesting so reruns against the same period are deterministic and free (design spec section 1.3 -- currently every backtest rerun calls the live LLM again)
* Add agent call telemetry (token counts, latency, cache hits) to backtesting settings.json, matching what the old lumibot dashboard's Parameters tab showed (design spec section 1.3)
```

- [ ] **Step 3: Commit**

```bash
git add TODO.md
git commit -m "Update TODO.md: backtesting done, note two deferred follow-ups"
```

---

## Self-Review

**1. Spec coverage** — every numbered section of `docs/superpowers/specs/2026-09-12-backtesting-framework-design.md` maps to a task:

| Spec section | Task(s) |
|---|---|
| §3.3(a) `max_wait_slice` | 1 |
| §3.1 `utils/errors.py` additions | 2 |
| §4.1 `BacktestDataSource` ABC | 3 |
| §2 fill model | 4 |
| §3.1 `ledger.py` | 5 |
| §3.1 `clock.py` | 6 |
| §6.1 `BacktestBroker` | 7, 8 |
| §5 no-look-ahead rule (the guarantee itself) | 9 (proves it), 3/7/8/10/11/12 (implement it) |
| §4.3 cache | 10 |
| §4.1/§4.2 Yahoo (default) | 11 |
| §4.1/§4.2 Alpaca (exact sessions) | 12 |
| §6.5 metrics | 13 |
| §6.4 output contract | 14 |
| §3.3(b)/§6.2 orchestration | 15, 18 |
| §7.1 lazy imports | 16 (package), 11/13 (deferred imports already inline in their own tasks) |
| §6.3 `add_line` | 17 |
| §6.2 public API | 18 |
| §7.1 dependencies | 19 |
| §8 testing (fills/gate/no-look-ahead/cache/broker/clock/executor/metrics/report/lazy-imports/smoke) | 4, 3, 9, 10, 7-8, 6, 9, 13, 14, 16, 20 |
| §7.2 CLAUDE.md amendments | 22 |
| §9 documentation | 20 (implicitly, as the manual verification path), 21, 22, 23 |

No gaps found.

**2. Placeholder scan** — searched for "TBD", "TODO", "implement later", "handle edge cases", "similar to Task N" without code. None found; every step either has real code or (Tasks 19-23, which are non-code infrastructure/docs tasks) a fully written artifact to add. The two places that read as softer instructions —
Task 13's vectorbt-method-name verification note, and Task 20's "if paper credentials aren't available, skip and note it" — are not placeholders: both give concrete, actionable fallback behaviour with a stated reason, not an unresolved "figure it out."

**3. Type consistency** — cross-checked signatures used across task boundaries:
- `BacktestDataSource.bars(asset, cutoff, length, timestep) -> Bars | None` — identical in Tasks 3, 7, 10, 11, 12.
- `Ledger.record_fill/record_equity/record_line` and `FillRecord`/`EquitySample`/`IndicatorLine` field names — identical across Tasks 5, 8, 14, 17.
- `fills.evaluate_fill(*, order_type, side, bar, limit_price=, stop_price=, stop_limit_price=)` and `fills.Bar`/`fills.FillResult` — identical across Tasks 4 and 8.
- `BacktestClock(start, sessions, on_advance=None)` with public `.on_advance` — identical across Tasks 6, 8, 9, 15.
- `BacktestBroker(strategy_name, *, data_source, clock, budget, timestep=, commission=, slippage=, tracker=None)` and `.ledger`/`.on_advance` — identical across Tasks 7, 8, 9, 15, 17.
- `run_backtest(strategy, *, start, end, budget, data_source, benchmark, timestep, commission, slippage, risk_free_rate) -> BacktestResult` — identical across Tasks 15, 16, 18.
- `compute_metrics(returns, benchmark_returns, *, timestep, risk_free_rate) -> dict` — identical across Tasks 13, 15.
- `write_settings/write_metrics/write_equity/write_trades/write_indicators` signatures — identical across Tasks 14, 15.
- `Strategy.run_backtesting(*, start=, end=, budget=, data_source=, benchmark=, timestep=, commission=, slippage=, risk_free_rate=) -> BacktestResult` — matches the spec's §6.2 signature exactly and matches Task 15's `run_backtest` parameter names one-to-one (only the object being called differs: instance method vs. module function).

No drift found.


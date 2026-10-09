# bull_bear Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A weekly (Tuesday 12:00 ET) strategy where a researcher, a bull, a bear and a judge debate the top 15 of cross_momentum's momentum ranking, the judge picks 5–10 winners, and code sizes them by inverse volatility and rebalances.

**Architecture:** First move the code bull_bear shares with cross_momentum and bill_ackman into a new `strategies/common/` package (momentum scoring, rebalancer, target portfolio, session helpers, Yahoo daily bars, sector provider), without changing either strategy's behaviour. Then build `strategies/bull_bear/` on the bill_ackman pattern: pure modules (parameters, sizing, debate set, fact sheet, hand-off validators), a `ReviewPipeline` that runs the agents through submit tools validated by a `HandoffRecorder`, a state file and a `reviews.jsonl`, and code (the moved `Rebalancer`) placing every order.

**Tech Stack:** Python 3.14, uv, pytest, ruff, pyright, LangChain agents via the repo's `AgentManager`, pandas (bars only), yfinance (lazy, via the moved modules).

**Spec:** `docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md`

## Global Constraints

- Python 3.14 (`.python-version`); package manager `uv` (`uv run pytest`, `uv run ruff check`, `uv run pyright`).
- Ruff line length is 200: one-line signatures and dict literals are normal in this repo.
- The test suite never touches the network; tests use hand-written fakes (`tests/fakes.py`), never `MagicMock`.
- No `time.sleep` / `datetime.now` in strategy code: time comes from `strategy.clock.now()`, waits from `strategy.sleep()`.
- `handoff.py` has NO `from __future__ import annotations` (the agent layer reads real annotations to build tool schemas). Every other new module starts with `from __future__ import annotations`.
- No agent has an order tool; code places every order through `Rebalancer`.
- Money: the moved `Rebalancer` keeps its existing float arithmetic (unchanged code); add no new float money boundary elsewhere.
- No re-export shims: modules moved to `strategies/common/` are imported from there everywhere.
- Momentum parameters are read from cross_momentum's `CONFIG` (`strategies/cross_momentum/parameters.py`), never duplicated.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

**Plan-level decision (not in the spec):** `completed_bars` and `parse_rebalance_time` (with their constants) also move to `strategies/common/sessions.py` in Task 3, so bull_bear never imports from `cross_momentum/utils.py`.

## Review Focus

1. **Debate set smaller than `min_picks`** (a market where few stocks pass the filters): the review must be abandoned at stage `data` with no order, not reach a judge who cannot satisfy the pick count. → test in Task 9.
2. **An abandoned review on a Tuesday, then a restart the same day**: `last_completed_review` must not be saved, so the review runs again. → test in Task 9.
3. **Symbols the model writes in lower case or with spaces** (`" s00 "`): normalised to `S00`, not refused. → test in Task 6.
4. **The researcher submits a note for another symbol than the one it was given**: refused with the symbol it must use. → test in Task 6.
5. **Holdings include SHV and a position outside the universe**: SHV is never a forced exit (the rebalancer's `holdings()` excludes it); the outside position is forced out as `unranked`. → test in Task 5 (debate set) and Task 9 (pipeline, `OLD`).

---

## File Structure

**Created**
- `src/trading_agent_framework/strategies/common/__init__.py` — package docstring only.
- `src/trading_agent_framework/strategies/common/scoring.py` — pure momentum scoring (moved + `momentum_inputs`, `score_stock`, `rank`).
- `src/trading_agent_framework/strategies/common/rebalancer.py` — moved from bill_ackman, typed against `RebalanceParams`.
- `src/trading_agent_framework/strategies/common/portfolio.py` — moved from bill_ackman.
- `src/trading_agent_framework/strategies/common/sessions.py` — `completed_bars`, `parse_rebalance_time` (moved from cross_momentum/utils).
- `src/trading_agent_framework/strategies/common/yahoo_daily_bars.py` — moved from cross_momentum.
- `src/trading_agent_framework/strategies/common/sector_provider.py` — moved from cross_momentum.
- `src/trading_agent_framework/strategies/bull_bear/__init__.py`
- `src/trading_agent_framework/strategies/bull_bear/parameters.py` — `BullBearParams`.
- `src/trading_agent_framework/strategies/bull_bear/sizing.py` — `capped_inverse_volatility`.
- `src/trading_agent_framework/strategies/bull_bear/debate_set.py` — `build_debate_set`.
- `src/trading_agent_framework/strategies/bull_bear/fact_sheet.py` — `fact_sheet_row`.
- `src/trading_agent_framework/strategies/bull_bear/handoff.py` — validators, `HandoffRecorder`, `submit_tools`.
- `src/trading_agent_framework/strategies/bull_bear/state.py` — `StateStore`, `ReviewLog`.
- `src/trading_agent_framework/strategies/bull_bear/market_data.py` — `YahooBars`, `GateBars`.
- `src/trading_agent_framework/strategies/bull_bear/prompts.py`
- `src/trading_agent_framework/strategies/bull_bear/pipeline.py` — `ReviewPipeline`.
- `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` — `BullBearStrategy`.
- Tests: `tests/strategies/common/test_common_scoring.py`, `test_common_rebalancer.py` (moved), `test_common_portfolio.py` (moved); `tests/strategies/bull_bear/test_bull_bear_{params,sizing,debate_set,fact_sheet,handoff,state,market_data,prompts,pipeline,strategy,backtest}.py`.

**Modified**
- `strategies/cross_momentum/agent_cross_momentum.py`, `utils.py`, `log_diagnostics.py` — imports from `common/`.
- `strategies/bill_ackman/agent_bill_ackman.py`, `pipeline.py` — imports from `common/`.
- `utils/strategy_factory.py` — `Strategies.BULL_BEAR`.
- `tests/strategies/test_cross_momentum_indicators.py` (pin test), `test_cross_momentum_timing.py`, `test_cross_momentum_yahoo_bars.py`, `bill_ackman/test_ackman_pipeline.py` — imports.
- `CLAUDE.md`, `README.md`.

---

### Task 1: Shared momentum scoring in `strategies/common/scoring.py`

**Files:**
- Create: `src/trading_agent_framework/strategies/common/__init__.py`
- Create: `src/trading_agent_framework/strategies/common/scoring.py`
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (delete `compute_return_from_prices`, `annualized_volatility`, `momentum_score`, `apply_filters`, lines ~68-134)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (imports lines 51-75; `_compute_indicators_for_ticker` lines ~306-346)
- Test: `tests/strategies/test_cross_momentum_indicators.py` (add a pin test)
- Test: `tests/strategies/common/test_common_scoring.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (all in `trading_agent_framework.strategies.common.scoring`):
  - `compute_return_from_prices(closes: Sequence[float], lookback_days: int, skip_days: int = 0) -> float | None`
  - `annualized_volatility(closes: Sequence[float], window: int) -> float | None`
  - `momentum_score(ret_12_1m: float, ret_6_1m: float, ret_3m: float, w_12m: float, w_6m: float, w_3m: float) -> float`
  - `apply_filters(price: float, avg_dollar_volume: float, volatility: float | None, trading_days: int, params: Mapping[str, Any]) -> bool`
  - `MomentumInputs(price, ret_12_1m, ret_6_1m, ret_3m, volatility: float | None, avg_dollar_volume, trading_days: int)` (frozen dataclass)
  - `momentum_inputs(closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumInputs | None`
  - `MomentumRow(symbol: str, inputs: MomentumInputs, volatility: float, score: float, closes: tuple[float, ...])` (frozen)
  - `score_stock(symbol: str, closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumRow | None`
  - `RankedRow(rank: int, row: MomentumRow)` (frozen)
  - `rank(rows: Iterable[MomentumRow]) -> list[RankedRow]`

- [ ] **Step 1: Write the pin test against today's cross_momentum (it must PASS before any change)**

Append to `tests/strategies/test_cross_momentum_indicators.py` (add `import math`, `import statistics` and `import pytest` to its imports):

```python
def test_the_scan_computes_cross_momentums_inputs_on_completed_sessions():
    """Pins the momentum inputs before they move to strategies/common/scoring.py: 300 completed closes 100..399."""
    fake = FakeStrategy(bars=_daily_bars(301, date(2026, 10, 6)))  # Tuesday's crash is still forming and is dropped

    result = CrossMomentumStrategy._compute_indicators_for_ticker(fake, "AAA")

    assert result is not None
    assert result["ret_12_1m"] == pytest.approx((378 - 126) / 126)  # closes[-22] over closes[-274]
    assert result["ret_6_1m"] == pytest.approx((378 - 252) / 252)  # closes[-22] over closes[-148]
    assert result["ret_3m"] == pytest.approx((399 - 336) / 336)  # closes[-1] over closes[-64]
    assert result["avg_dollar_volume"] == pytest.approx(1e6 * 399)
    closes = [100.0 + i for i in range(300)]
    logs = [math.log(closes[i] / closes[i - 1]) for i in range(-20, 0)]
    assert result["volatility"] == pytest.approx(statistics.stdev(logs) * math.sqrt(252))
```

- [ ] **Step 2: Run it: it must pass on the current code**

Run: `uv run pytest tests/strategies/test_cross_momentum_indicators.py -v`
Expected: all PASS (this is a characterisation test; if it fails, the arithmetic in the comment is wrong — fix the test, not the code).

- [ ] **Step 3: Write the failing tests for the new module**

Create `tests/strategies/common/test_common_scoring.py`:

```python
from __future__ import annotations

import math
import statistics

import pytest

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, momentum_inputs, rank, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG

CLOSES = [100.0 + i for i in range(300)]
VOLUMES = [1e6] * 300


def test_momentum_inputs_are_cross_momentums_returns_volatility_and_dollar_volume() -> None:
    inputs = momentum_inputs(CLOSES, VOLUMES, CONFIG)

    assert inputs is not None
    assert (inputs.price, inputs.trading_days) == (399.0, 300)
    assert inputs.ret_12_1m == pytest.approx(2.0)
    assert inputs.ret_6_1m == pytest.approx(0.5)
    assert inputs.ret_3m == pytest.approx(0.1875)
    assert inputs.avg_dollar_volume == pytest.approx(399e6)
    logs = [math.log(CLOSES[i] / CLOSES[i - 1]) for i in range(-20, 0)]
    assert inputs.volatility == pytest.approx(statistics.stdev(logs) * math.sqrt(252))


def test_a_history_shorter_than_min_trading_days_has_no_inputs() -> None:
    assert momentum_inputs(CLOSES[:249], VOLUMES[:249], CONFIG) is None


def test_a_history_too_short_for_the_12_month_return_has_no_inputs() -> None:
    assert momentum_inputs(CLOSES[:273], VOLUMES[:273], CONFIG) is None  # needs 252 + 21 + 1 = 274 closes


def test_score_stock_weights_the_returns_like_cross_momentum() -> None:
    row = score_stock("AAA", CLOSES, VOLUMES, CONFIG)

    assert isinstance(row, MomentumRow)
    assert row.symbol == "AAA"
    assert row.score == pytest.approx(0.5 * 2.0 + 0.3 * 0.5 + 0.2 * 0.1875)
    assert row.volatility == row.inputs.volatility
    assert row.closes == tuple(CLOSES)


def test_score_stock_applies_cross_momentums_filters() -> None:
    cheap = [c / 100 for c in CLOSES]  # last close 3.99, under min_price 10
    thin = [1_000.0] * 300  # 399 x 1,000 = 0.4M a day, under min_dollar_volume 20M

    assert score_stock("CHEAP", cheap, VOLUMES, CONFIG) is None
    assert score_stock("THIN", CLOSES, thin, CONFIG) is None


def test_score_stock_needs_a_volatility_above_zero() -> None:
    flat = [100.0] * 300  # every log return is exactly 0: zero volatility (a geometric series leaves float noise)

    assert score_stock("FLAT", flat, VOLUMES, CONFIG) is None


def test_rank_orders_by_score_best_first_from_1() -> None:
    slow = score_stock("SLOW", [100.0 + 0.5 * i for i in range(300)], VOLUMES, CONFIG)
    fast = score_stock("FAST", CLOSES, VOLUMES, CONFIG)
    assert slow is not None and fast is not None

    ranked = rank([slow, fast])

    assert ranked == [RankedRow(1, fast), RankedRow(2, slow)]


def test_rank_of_nothing_is_empty() -> None:
    assert rank([]) == []
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/strategies/common/test_common_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.strategies.common'`.

- [ ] **Step 5: Create the package and the module**

`src/trading_agent_framework/strategies/common/__init__.py`:

```python
"""Code shared by several strategies: momentum scoring, the rebalancer, session helpers and data sources."""
```

`src/trading_agent_framework/strategies/common/scoring.py` (the four helpers are moved verbatim from `cross_momentum/utils.py`, only their annotations widen to `Sequence`/`Mapping`):

```python
"""Momentum scoring shared by cross_momentum and bull_bear (pure: no I/O, no pandas).

Moved from cross_momentum (`utils.py` and `CrossMomentumStrategy._compute_indicators_for_ticker`) so that bull_bear
shortlists on cross_momentum's own ranking by construction. `params` is cross_momentum's `CONFIG`, or any mapping
with the keys read here (`min_trading_days`, `skip_days`, `volatility_window`, the filter thresholds, the weights).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_YEAR, _HALF_YEAR, _QUARTER = 252, 126, 63
_DOLLAR_VOLUME_DAYS = 20


def compute_return_from_prices(
    closes: Sequence[float],
    lookback_days: int,
    skip_days: int = 0,
) -> float | None:
    """Compute percentage return over a lookback window, optionally skipping recent days.

    For "12-1 month momentum": lookback_days=252, skip_days=21.
    Returns (price[-skip_days-1] - price[-lookback_days-skip_days-1]) / price[-lookback_days-skip_days-1].
    """
    needed = lookback_days + skip_days + 1
    if len(closes) < needed:
        return None
    start_price = closes[-needed]
    end_price = closes[-skip_days - 1]
    if start_price <= 0:
        return None
    return (end_price - start_price) / start_price


def annualized_volatility(closes: Sequence[float], window: int) -> float | None:
    """Annualized volatility from daily log returns over the most recent `window` days."""
    if len(closes) < window + 1:
        return None
    recent = closes[-window - 1 :]
    log_returns = []
    for i in range(1, len(recent)):
        if recent[i - 1] <= 0 or recent[i] <= 0:
            return None
        log_returns.append(math.log(recent[i] / recent[i - 1]))
    if len(log_returns) < 2:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252)


def momentum_score(
    ret_12_1m: float,
    ret_6_1m: float,
    ret_3m: float,
    w_12m: float,
    w_6m: float,
    w_3m: float,
) -> float:
    """Weighted momentum score. Returns a single float (higher = stronger momentum)."""
    return w_12m * ret_12_1m + w_6m * ret_6_1m + w_3m * ret_3m


def apply_filters(
    price: float,
    avg_dollar_volume: float,
    volatility: float | None,
    trading_days: int,
    params: Mapping[str, Any],
) -> bool:
    """Check tradability filters. Returns True if the stock passes all gates."""
    if price < params["min_price"]:
        return False
    if avg_dollar_volume < params["min_dollar_volume"]:
        return False
    if volatility is not None and volatility > params["max_volatility"]:
        return False
    if trading_days < params["min_trading_days"]:
        return False
    return True


@dataclass(frozen=True, slots=True)
class MomentumInputs:
    price: float  # the last completed close
    ret_12_1m: float
    ret_6_1m: float
    ret_3m: float
    volatility: float | None  # annualized, over `volatility_window` sessions; None when it cannot be computed
    avg_dollar_volume: float  # the last 20 sessions' average volume times the last close
    trading_days: int


def momentum_inputs(closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumInputs | None:
    """The returns, volatility and liquidity cross_momentum scores on, from completed sessions, oldest first.

    None when the history is shorter than `min_trading_days` or too short for one of the returns.
    """
    trading_days = len(closes)
    if trading_days < params["min_trading_days"]:
        return None
    skip = params["skip_days"]
    ret_12_1m = compute_return_from_prices(closes, _YEAR, skip)
    ret_6_1m = compute_return_from_prices(closes, _HALF_YEAR, skip)
    ret_3m = compute_return_from_prices(closes, _QUARTER, 0)
    if ret_12_1m is None or ret_6_1m is None or ret_3m is None:
        return None
    price = closes[-1]
    recent = volumes[-_DOLLAR_VOLUME_DAYS:] if len(volumes) >= _DOLLAR_VOLUME_DAYS else volumes
    avg_volume = sum(recent) / max(len(recent), 1)
    return MomentumInputs(
        price=price,
        ret_12_1m=ret_12_1m,
        ret_6_1m=ret_6_1m,
        ret_3m=ret_3m,
        volatility=annualized_volatility(closes, params["volatility_window"]),
        avg_dollar_volume=avg_volume * price,
        trading_days=trading_days,
    )


@dataclass(frozen=True, slots=True)
class MomentumRow:
    symbol: str
    inputs: MomentumInputs
    volatility: float  # `inputs.volatility`, known to be finite and above zero
    score: float
    closes: tuple[float, ...]  # the completed closes scored, oldest first (the fact sheet reads them)


def score_stock(symbol: str, closes: Sequence[float], volumes: Sequence[float], params: Mapping[str, Any]) -> MomentumRow | None:
    """A stock's momentum row, or None when it has no inputs, fails cross_momentum's filters, or has no volatility.

    cross_momentum ranks a stock whose volatility cannot be computed (non-positive closes); bull_bear sizes by
    volatility, so `score_stock` leaves such a stock out. It cannot happen with positive prices.
    """
    inputs = momentum_inputs(closes, volumes, params)
    if inputs is None or inputs.volatility is None or not (math.isfinite(inputs.volatility) and inputs.volatility > 0):
        return None
    if not apply_filters(inputs.price, inputs.avg_dollar_volume, inputs.volatility, inputs.trading_days, params):
        return None
    score = momentum_score(inputs.ret_12_1m, inputs.ret_6_1m, inputs.ret_3m, params["w_12m"], params["w_6m"], params["w_3m"])
    return MomentumRow(symbol=symbol, inputs=inputs, volatility=inputs.volatility, score=score, closes=tuple(closes))


@dataclass(frozen=True, slots=True)
class RankedRow:
    rank: int  # 1 = the strongest momentum
    row: MomentumRow


def rank(rows: Iterable[MomentumRow]) -> list[RankedRow]:
    """Rows sorted by score, best first, ranked from 1 (a stable sort: equal scores keep their input order)."""
    ordered = sorted(rows, key=lambda row: row.score, reverse=True)
    return [RankedRow(index, row) for index, row in enumerate(ordered, start=1)]
```

- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/strategies/common/test_common_scoring.py -v`
Expected: PASS.

- [ ] **Step 7: Point cross_momentum at the shared module**

In `cross_momentum/utils.py`, delete the four functions `compute_return_from_prices`, `annualized_volatility`, `momentum_score` and `apply_filters` (the block between `parse_rebalance_time` and `inverse_volatility_weights`).

In `cross_momentum/agent_cross_momentum.py`, remove `annualized_volatility`, `apply_filters`, `compute_return_from_prices` and `momentum_score` from the `from .utils import (...)` list, and add next to the other absolute imports:

```python
from trading_agent_framework.strategies.common.scoring import apply_filters, momentum_inputs, momentum_score
```

Replace the body of `_compute_indicators_for_ticker` from `closes = df["close"].tolist()` to the end of the method with:

```python
        closes = df["close"].tolist()
        inputs = momentum_inputs(closes, df["volume"].tolist(), self.parameters)
        if inputs is None:
            return None

        atr = compute_atr_from_df(df) if inputs.trading_days >= 15 else None

        return {
            "symbol": ticker,
            "price": inputs.price,
            "ret_12_1m": inputs.ret_12_1m,
            "ret_6_1m": inputs.ret_6_1m,
            "ret_3m": inputs.ret_3m,
            "volatility": inputs.volatility,
            "avg_dollar_volume": inputs.avg_dollar_volume,
            "trading_days": inputs.trading_days,
            "atr": atr,
            "closes": closes,
            "close_series": close_series(df),
        }
```

`_process_ticker` keeps calling `apply_filters` and `momentum_score`, now from `common.scoring`.

- [ ] **Step 8: Run the cross_momentum tests, the new tests and the linters**

Run: `uv run pytest tests/strategies/ -k "cross_momentum or common" -q && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/common src/trading_agent_framework/strategies/cross_momentum`
Expected: all PASS; if ruff reports `math` (or another import) unused in `cross_momentum/utils.py`, remove that import and run again.

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/strategies/common src/trading_agent_framework/strategies/cross_momentum tests/strategies/common tests/strategies/test_cross_momentum_indicators.py
git commit -m "refactor(common): move momentum scoring to strategies/common/scoring.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Move the rebalancer and target portfolio to `strategies/common/`

**Files:**
- Move: `src/trading_agent_framework/strategies/bill_ackman/rebalancer.py` → `src/trading_agent_framework/strategies/common/rebalancer.py`
- Move: `src/trading_agent_framework/strategies/bill_ackman/portfolio.py` → `src/trading_agent_framework/strategies/common/portfolio.py`
- Move: `tests/strategies/bill_ackman/test_ackman_rebalancer.py` → `tests/strategies/common/test_common_rebalancer.py`
- Move: `tests/strategies/bill_ackman/test_ackman_portfolio.py` → `tests/strategies/common/test_common_portfolio.py`
- Modify: `src/trading_agent_framework/strategies/bill_ackman/pipeline.py:22,24`, `agent_bill_ackman.py:27`, `tests/strategies/bill_ackman/test_ackman_pipeline.py:25`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `trading_agent_framework.strategies.common.rebalancer.RebalanceParams` (Protocol with read-only `parking_symbol: str`, `rebalance_band: float`, `min_trade_pct: float`, `cash_buffer: float`)
  - `Rebalancer(strategy: Strategy, params: RebalanceParams)` with `holdings() -> list[str]`, `current_weights() -> dict[str, float]`, `rebalance(target: TargetPortfolio, forced_exits: Collection[str] = ()) -> list[PlacedOrder]`, attribute `placed: list[PlacedOrder]`
  - `PlacedOrder(symbol: str, side: str, quantity: float)`
  - `trading_agent_framework.strategies.common.portfolio.TargetPortfolio(weights: dict[str, float], parking_weight: float)` and `target_portfolio(weights: Mapping[str, float], *, cash_buffer: float) -> TargetPortfolio`

- [ ] **Step 1: Move the files with git**

```bash
mkdir -p tests/strategies/common
git mv src/trading_agent_framework/strategies/bill_ackman/rebalancer.py src/trading_agent_framework/strategies/common/rebalancer.py
git mv src/trading_agent_framework/strategies/bill_ackman/portfolio.py src/trading_agent_framework/strategies/common/portfolio.py
git mv tests/strategies/bill_ackman/test_ackman_rebalancer.py tests/strategies/common/test_common_rebalancer.py
git mv tests/strategies/bill_ackman/test_ackman_portfolio.py tests/strategies/common/test_common_portfolio.py
```

- [ ] **Step 2: Fix every import**

Replace, in every file found by `grep -rln "bill_ackman.rebalancer\|bill_ackman.portfolio" src tests`:
- `trading_agent_framework.strategies.bill_ackman.rebalancer` → `trading_agent_framework.strategies.common.rebalancer`
- `trading_agent_framework.strategies.bill_ackman.portfolio` → `trading_agent_framework.strategies.common.portfolio`

(Files: `common/rebalancer.py` itself, `bill_ackman/pipeline.py`, `bill_ackman/agent_bill_ackman.py`, `tests/strategies/common/test_common_rebalancer.py`, `tests/strategies/common/test_common_portfolio.py`, `tests/strategies/bill_ackman/test_ackman_pipeline.py`.)

- [ ] **Step 3: Type the rebalancer against a protocol and state the cash invariant**

In `common/rebalancer.py`:
- replace `from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams` with nothing, and add `Protocol` to the `typing` import;
- add after the imports:

```python
class RebalanceParams(Protocol):
    """What the rebalancer reads from a strategy's parameters (`AckmanParams`, `BullBearParams`)."""

    @property
    def parking_symbol(self) -> str: ...

    @property
    def rebalance_band(self) -> float: ...

    @property
    def min_trade_pct(self) -> float: ...

    @property
    def cash_buffer(self) -> float: ...
```

- change `def __init__(self, strategy: Strategy, params: AckmanParams) -> None:` to `def __init__(self, strategy: Strategy, params: RebalanceParams) -> None:`;
- replace the docstring of `rebalance` with:

```python
        """Trade the book toward `target`; returns the orders that were accepted, in submission order.

        Cash invariant (live fills must never be counted twice, cf. congress_trades 275c2f3): the account is read
        once, BEFORE any order and before positions and open orders, and `cash` is never read again in this call.
        Sell proceeds are added to that pre-sell snapshot, so a live sell filled within seconds counts once. Buying
        power is read again after the sells, but only inside `min()`. A buy refused for buying power replaces
        `available` with the broker's own figure. Earlier open orders count only their unfilled part. Reading the
        account after positions/open orders would double-count an older sell that fills in between.
        """
```

- [ ] **Step 4: Run the moved and dependent tests**

Run: `uv run pytest tests/strategies/common tests/strategies/bill_ackman -q`
Expected: PASS (behaviour unchanged).

- [ ] **Step 5: Write the two failing-on-regression tests for live fills**

Append to `tests/strategies/common/test_common_rebalancer.py` (add `from trading_agent_framework.entities.enums import OrderStatus` to the imports; `OrderSide`, `Order`, `AccountBalances`, `Decimal` are already imported):

```python
class _LiveFillBroker(FakeBroker):
    """Fills every order at submission, at its last price, and moves cash at once: Alpaca with a market order.

    Buying power is 4x cash (a margin account), so `min(buying_power, ...)` cannot hide a double count.
    """

    def _submit_order(self, order: Order) -> Order:
        self.submitted.append(order)
        cost = self.last_prices[order.asset.symbol] * order.quantity
        cash = self.account.cash - cost if order.side is OrderSide.BUY else self.account.cash + cost
        self.account = AccountBalances(cash=cash, portfolio_value=self.account.portfolio_value, buying_power=4 * cash)
        order.status, order.filled_quantity = OrderStatus.FILL, order.quantity
        return order  # never tracked as active: it is already filled


def _live_book(tmp_path: Path, *, positions: dict[str, float], prices: dict[str, float], cash: float) -> tuple[_LiveFillBroker, Rebalancer]:
    broker = _LiveFillBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG) for symbol, quantity in positions.items()]
    broker.last_prices = {symbol: Decimal(str(price)) for symbol, price in prices.items()}
    # Portfolio value 10,000 while only cash + A is visible: the money, not the targets, binds the buys.
    broker.account = AccountBalances(cash=Decimal(str(cash)), portfolio_value=Decimal(10_000), buying_power=Decimal(str(4 * cash)))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return broker, Rebalancer(strategy, AckmanParams())


def test_a_live_filled_sell_funds_the_buys_once(tmp_path: Path) -> None:
    """Sell A (5,000) then buy B and C: at most pre-sell cash 1,000 + 5,000 - the 200 reserve = 5,800 is spent."""
    broker, rebalancer = _live_book(tmp_path, positions={"A": 100}, prices={"A": 50, "B": 25, "C": 20, "SHV": 100}, cash=1_000)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.49, "C": 0.49}, cash_buffer=0.02))

    # B wants 4,900 (196 shares), C gets the 900 left (45 shares); re-reading the filled cash would have bought far more C
    assert placed == [PlacedOrder("A", "sell", 100.0), PlacedOrder("B", "buy", 196.0), PlacedOrder("C", "buy", 45.0)]
    assert broker.account.cash >= Decimal(200)


def test_a_live_filled_buy_is_not_deducted_twice_from_the_next_buy(tmp_path: Path) -> None:
    """Two buys from 5,000 cash: B 2,400 then C 2,400; C must not be sized on (cash after B) - B's cost again."""
    broker, rebalancer = _live_book(tmp_path, positions={}, prices={"B": 25, "C": 20, "SHV": 100}, cash=5_000)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.24, "C": 0.24}, cash_buffer=0.02))

    assert placed[:2] == [PlacedOrder("B", "buy", 96.0), PlacedOrder("C", "buy", 120.0)]
    assert broker.account.cash >= Decimal(0)
```

- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/strategies/common/test_common_rebalancer.py -v -k live`
Expected: PASS (the current code already honours the invariant; these tests pin it). If either fails, first rule out the fake: the order must NOT be tracked (`tracker.track_unprocessed` would leave a filled order in the active buckets and the rebalancer would count it as in flight), and `Strategy.submit_order` must return without waiting. If the fake is right and a test still fails, stop and report: the moved code double-counts and the spec's claim is wrong.

- [ ] **Step 7: Full suite and linters**

Run: `uv run pytest -q && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/common src/trading_agent_framework/strategies/bill_ackman`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A src/trading_agent_framework/strategies tests/strategies
git commit -m "refactor(common): move the rebalancer and target portfolio to strategies/common

Typed against a RebalanceParams protocol; the cash invariant is stated and pinned
by two live-fill tests.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Move session helpers, Yahoo daily bars and the sector provider to `strategies/common/`

**Files:**
- Create: `src/trading_agent_framework/strategies/common/sessions.py` (moved from `cross_momentum/utils.py`)
- Move: `cross_momentum/yahoo_daily_bars.py` → `common/yahoo_daily_bars.py`
- Move: `cross_momentum/sector_provider.py` → `common/sector_provider.py`
- Modify: `cross_momentum/utils.py`, `cross_momentum/agent_cross_momentum.py`, `cross_momentum/log_diagnostics.py:33`
- Modify: `tests/strategies/test_cross_momentum_timing.py:12`, `tests/strategies/test_cross_momentum_yahoo_bars.py:25`

**Interfaces:**
- Produces:
  - `trading_agent_framework.strategies.common.sessions.completed_bars(df: pd.DataFrame, today: date) -> pd.DataFrame`
  - `trading_agent_framework.strategies.common.sessions.parse_rebalance_time(value: str) -> time` (raises `ConfigurationError`)
  - `trading_agent_framework.strategies.common.yahoo_daily_bars.YahooDailyBars` with `bars(symbols: Sequence[str], today: date) -> dict[str, Bars]` (raises `YahooDataError`)
  - `trading_agent_framework.strategies.common.sector_provider.SectorProvider` with `get_sector(symbol: str) -> str` (`"UNKNOWN"` on failure)

- [ ] **Step 1: Move the two modules with git**

```bash
git mv src/trading_agent_framework/strategies/cross_momentum/yahoo_daily_bars.py src/trading_agent_framework/strategies/common/yahoo_daily_bars.py
git mv src/trading_agent_framework/strategies/cross_momentum/sector_provider.py src/trading_agent_framework/strategies/common/sector_provider.py
```

Edit the first line of the moved `yahoo_daily_bars.py` docstring from `"""Yahoo as the source of cross_momentum's daily bars in paper/live.` to `"""Yahoo as the source of daily bars in paper/live (cross_momentum, bull_bear).`

- [ ] **Step 2: Create `common/sessions.py` by moving `completed_bars` and `parse_rebalance_time`**

Cut from `cross_momentum/utils.py`: the constants `_REBALANCE_EARLIEST`, `_REBALANCE_LATEST`, `_REBALANCE_TIME_PATTERN` (with the comment above them), `parse_rebalance_time`, and `completed_bars`. Create:

```python
"""Session helpers shared by the daily strategies (pure): today's partial bar, and the iteration's market time."""

from __future__ import annotations

import re
from datetime import date, time

import pandas as pd

from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

# <paste the three _REBALANCE_* constants with their comment, unchanged>


# <paste parse_rebalance_time, unchanged>


# <paste completed_bars, unchanged>
```

(The angle-bracket lines are paste markers for the code cut above, not placeholders: the functions move byte for byte.)

- [ ] **Step 3: Fix the imports**

- `cross_momentum/agent_cross_momentum.py`: remove `completed_bars` and `parse_rebalance_time` from the `from .utils import (...)` list; replace `from .yahoo_daily_bars import YahooDailyBars` with `from trading_agent_framework.strategies.common.yahoo_daily_bars import YahooDailyBars`; add `from trading_agent_framework.strategies.common.sessions import completed_bars, parse_rebalance_time`.
- `cross_momentum/log_diagnostics.py`: `from .sector_provider import SectorProvider` → `from trading_agent_framework.strategies.common.sector_provider import SectorProvider`.
- `tests/strategies/test_cross_momentum_timing.py`: `from trading_agent_framework.strategies.cross_momentum.utils import completed_bars, parse_rebalance_time` → `from trading_agent_framework.strategies.common.sessions import completed_bars, parse_rebalance_time`.
- `tests/strategies/test_cross_momentum_yahoo_bars.py`: `...cross_momentum.yahoo_daily_bars import YahooDailyBars` → `...common.yahoo_daily_bars import YahooDailyBars`.
- Then run `grep -rn "cross_momentum.yahoo_daily_bars\|cross_momentum.sector_provider\|cross_momentum\.utils\.completed_bars\|cross_momentum\.utils\.parse_rebalance_time" src tests scripts` and fix every remaining hit (including monkeypatch target strings) the same way.

- [ ] **Step 4: Run tests and linters**

Run: `uv run pytest -q && uv run ruff check && uv run pyright src/trading_agent_framework/strategies`
Expected: PASS; if ruff reports `re`, `time` or `ConfigurationError` unused in `cross_momentum/utils.py`, remove them and run again.

- [ ] **Step 5: Commit**

```bash
git add -A src/trading_agent_framework/strategies tests/strategies
git commit -m "refactor(common): move session helpers, Yahoo daily bars and the sector provider to strategies/common

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `BullBearParams` and inverse-volatility sizing

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/parameters.py`
- Create: `src/trading_agent_framework/strategies/bull_bear/sizing.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `tests/strategies/bull_bear/test_bull_bear_sizing.py`

(`bull_bear/__init__.py` is created in Task 10, with the strategy; until then the package is a namespace package, which pytest and pyright handle.)

**Interfaces:**
- Consumes: `common.sessions.parse_rebalance_time`.
- Produces:
  - `BullBearParams` (frozen dataclass; fields and defaults in Step 3) with property `investable -> float` (= `1 - cash_buffer`). Satisfies `RebalanceParams`.
  - `capped_inverse_volatility(volatilities: Mapping[str, float], *, total: float, min_weight: float, max_weight: float) -> dict[str, float]` (keys in input order; raises `ValueError`)

- [ ] **Step 1: Write the failing tests**

`tests/strategies/bull_bear/test_bull_bear_params.py`:

```python
from __future__ import annotations

import re

import pytest

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams


def test_the_defaults_are_the_specs_and_valid() -> None:
    params = BullBearParams()

    assert (params.shortlist_size, params.retention_rank, params.min_picks, params.max_picks) == (15, 35, 5, 10)
    assert (params.min_weight, params.max_weight, params.cash_buffer, params.rebalance_band) == (0.04, 0.20, 0.02, 0.03)
    assert params.investable == pytest.approx(0.98)
    assert (params.rebalance_time, params.rebalance_weekday, params.parking_symbol) == ("12:00", 1, "SHV")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"min_weight": 0.03}, "min_weight must be above rebalance_band"),
        ({"min_weight": 0.1}, "max_picks x min_weight must fit"),
        ({"min_picks": 4}, "min_picks x max_weight must reach"),
        ({"retention_rank": 15}, "retention_rank must be above shortlist_size"),
        ({"max_picks": 16}, "1 <= min_picks <= max_picks <= shortlist_size"),
        ({"rebalance_time": "9:00"}, "rebalance_time"),
        ({"rebalance_weekday": 5}, "rebalance_weekday must be a weekday"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
        ({"parking_symbol": " "}, "parking_symbol must not be blank"),
        ({"min_yahoo_coverage": 0.0}, "min_yahoo_coverage must be in (0, 1]"),
        ({"yahoo_retry_delays": (60.0, -1.0)}, "yahoo_retry_delays"),
        ({"research_tool_budget": 0}, "research_tool_budget must be at least 1"),
    ],
)
def test_an_invalid_value_is_refused_with_a_reason(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):  # messages contain "(0, 1]"
        BullBearParams(**overrides)
```

`tests/strategies/bull_bear/test_bull_bear_sizing.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.sizing import capped_inverse_volatility


def _size(volatilities: dict[str, float]) -> dict[str, float]:
    return capped_inverse_volatility(volatilities, total=0.98, min_weight=0.04, max_weight=0.20)


def test_equal_volatilities_share_the_investable_money_equally() -> None:
    weights = _size({symbol: 0.3 for symbol in "ABCDE"})

    assert weights == pytest.approx({symbol: 0.196 for symbol in "ABCDE"})


def test_a_very_calm_stock_is_capped_and_the_rest_share_the_remainder() -> None:
    weights = _size({"A": 0.05, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5})

    assert weights["A"] == 0.20
    assert [weights[s] for s in "BCDE"] == pytest.approx([0.195] * 4)


def test_a_very_volatile_stock_is_floored() -> None:
    volatilities = {f"S{i}": 0.2 for i in range(9)} | {"WILD": 5.0}

    weights = _size(volatilities)

    assert weights["WILD"] == 0.04
    assert [weights[f"S{i}"] for i in range(9)] == pytest.approx([0.94 / 9] * 9)


def test_the_weights_sum_to_the_total_within_the_bounds_and_keep_the_input_order() -> None:
    volatilities = {"Z": 0.9, "A": 0.15, "M": 0.4, "B": 0.25, "Q": 0.6, "C": 0.33, "D": 0.21}

    weights = _size(volatilities)

    assert list(weights) == list(volatilities)  # the rebalancer buys in this order
    assert sum(weights.values()) == pytest.approx(0.98, abs=1e-9)
    assert all(0.04 <= weight <= 0.20 for weight in weights.values())
    assert weights["A"] == weights["D"] == 0.20  # both capped once the excess is redistributed
    assert weights["D"] > weights["B"] > weights["C"] > weights["M"] > weights["Q"] > weights["Z"]  # calmer gets more


@pytest.mark.parametrize("volatility", [0.0, -0.1, float("nan"), float("inf")])
def test_a_volatility_that_is_not_finite_and_positive_is_a_programming_error(volatility: float) -> None:
    with pytest.raises(ValueError, match="volatility of B"):
        _size({"A": 0.2, "B": volatility, "C": 0.2, "D": 0.2, "E": 0.2})


def test_bounds_that_cannot_hold_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot sum to"):
        _size({"A": 0.2, "B": 0.2, "C": 0.2})  # 3 x 0.20 < 0.98


def test_no_stock_is_refused() -> None:
    with pytest.raises(ValueError, match="no stock"):
        _size({})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear -v`
Expected: FAIL with `ModuleNotFoundError` for `bull_bear.parameters` / `bull_bear.sizing`.

- [ ] **Step 3: Write `parameters.py`**

```python
"""`BullBearParams`: every threshold of the bull_bear strategy in one frozen dataclass.

The momentum parameters (score weights, skip, filters) are NOT here: they are cross_momentum's `CONFIG`, so the
shortlist is cross_momentum's ranking by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.utils.errors import ConfigurationError

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class BullBearParams:
    shortlist_size: int = 15  # top-ranked stocks debated
    retention_rank: int = 35  # a holding ranked worse (or unranked) is a forced exit
    min_picks: int = 5  # the judge's pick count, inclusive bounds
    max_picks: int = 10
    min_weight: float = 0.04  # per-stock weight bounds, fractions of portfolio value
    max_weight: float = 0.20
    cash_buffer: float = 0.02  # never invested (fees, fill drift)
    rebalance_band: float = 0.03  # drift that triggers a trade
    min_trade_pct: float = 0.005  # smallest order
    parking_symbol: str = "SHV"  # where unspent money goes
    research_tool_budget: int = 3  # tool calls per researcher run; submit_note is exempt
    note_max_chars: int = 500
    argument_max_chars: int = 300
    reason_max_chars: int = 300
    max_consecutive_abandoned: int = 3  # abandoned reviews in a row that abort a backtest
    agent_temperature: float | None = 0.3  # all four agents; None leaves the LLM server's default
    rebalance_time: str = "12:00"  # market time (ET) of the weekly review
    rebalance_weekday: int = 1  # 0 = Monday ... 4 = Friday; 1 = Tuesday, as cross_momentum
    yahoo_retry_delays: tuple[float, ...] = (60.0, 180.0)  # seconds before each new attempt at an unusable Yahoo batch
    min_yahoo_coverage: float = 0.5  # share of the universe a Yahoo batch must cover

    def __post_init__(self) -> None:
        floats = (self.min_weight, self.max_weight, self.cash_buffer, self.rebalance_band, self.min_trade_pct, self.min_yahoo_coverage)
        investable = 1 - self.cash_buffer
        try:
            parse_rebalance_time(self.rebalance_time)
            time_problem = ""
        except ConfigurationError as exc:
            time_problem = str(exc)
        problems = {
            "weights, cash_buffer, rebalance_band, min_trade_pct and min_yahoo_coverage must be finite": not all(math.isfinite(value) for value in floats),
            "shortlist_size must be at least 1": self.shortlist_size < 1,
            "retention_rank must be above shortlist_size": self.retention_rank <= self.shortlist_size,
            "1 <= min_picks <= max_picks <= shortlist_size": not 1 <= self.min_picks <= self.max_picks <= self.shortlist_size,
            "min_weight must be above 0 and at most max_weight, which is at most 1": not 0 < self.min_weight <= self.max_weight <= 1,
            "min_weight must be above rebalance_band (a new pick below the band would never be bought)": self.min_weight <= self.rebalance_band,
            "cash_buffer must be in [0, 1)": not 0 <= self.cash_buffer < 1,
            "rebalance_band must be in (0, 1)": not 0 < self.rebalance_band < 1,
            "min_trade_pct must be in [0, 1)": not 0 <= self.min_trade_pct < 1,
            "max_picks x min_weight must fit in 1 - cash_buffer": self.max_picks * self.min_weight > investable + _EPS,
            "min_picks x max_weight must reach 1 - cash_buffer (the book is always fully invested)": self.min_picks * self.max_weight < investable - _EPS,
            "parking_symbol must not be blank": not self.parking_symbol.strip(),
            "research_tool_budget must be at least 1": self.research_tool_budget < 1,
            "note_max_chars, argument_max_chars and reason_max_chars must be at least 1": min(self.note_max_chars, self.argument_max_chars, self.reason_max_chars) < 1,
            "max_consecutive_abandoned must be at least 1": self.max_consecutive_abandoned < 1,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
            f"rebalance_time: {time_problem}": bool(time_problem),
            "rebalance_weekday must be a weekday (0 = Monday ... 4 = Friday)": not 0 <= self.rebalance_weekday <= 4,
            "yahoo_retry_delays must be finite and at least 0": not all(math.isfinite(delay) and delay >= 0 for delay in self.yahoo_retry_delays),
            "min_yahoo_coverage must be in (0, 1]": not 0 < self.min_yahoo_coverage <= 1,
        }
        failed = [message for message, bad in problems.items() if bad]
        if failed:
            raise ValueError("BullBearParams: " + "; ".join(failed))

    @property
    def investable(self) -> float:
        """The share of portfolio value the stock weights sum to: everything but the cash buffer."""
        return 1 - self.cash_buffer
```

- [ ] **Step 4: Write `sizing.py`**

```python
"""Target weights from the judge's picks (pure): inverse volatility, bounded per stock, summing to what is investable.

cross_momentum's `inverse_volatility_weights` renormalises after capping, so with few names it breaks the cap; this
one finds the single scale at which the clipped weights sum to the total, so both bounds hold exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

_ITERATIONS = 200  # bisection steps: the scale is exact to float precision long before
_EPS = 1e-9


def capped_inverse_volatility(volatilities: Mapping[str, float], *, total: float, min_weight: float, max_weight: float) -> dict[str, float]:
    """Each weight is `clip(scale / volatility, min_weight, max_weight)`, with the one `scale` that makes them sum to `total`.

    The clipped sum only grows with `scale`, so bisection finds it; the clip is the last step, so both bounds hold
    exactly. Keys keep the input order (the rebalancer buys in that order). Raises `ValueError` for no stock, a
    volatility that is not finite and above zero, or bounds that cannot hold (`n x min_weight > total` or
    `n x max_weight < total`).
    """
    if not volatilities:
        raise ValueError("no stock to size")
    for symbol, volatility in volatilities.items():
        if not (math.isfinite(volatility) and volatility > 0):
            raise ValueError(f"the volatility of {symbol} must be finite and above zero, got {volatility}")
    count = len(volatilities)
    if count * min_weight > total + _EPS or count * max_weight < total - _EPS:
        raise ValueError(f"{count} stocks cannot sum to {total} with weights in [{min_weight}, {max_weight}]")
    raw = {symbol: 1.0 / volatility for symbol, volatility in volatilities.items()}

    def weights(scale: float) -> dict[str, float]:
        return {symbol: min(max(scale * value, min_weight), max_weight) for symbol, value in raw.items()}

    low, high = 0.0, max_weight / min(raw.values())  # at `high` every weight is at its cap
    for _ in range(_ITERATIONS):
        middle = (low + high) / 2
        if sum(weights(middle).values()) < total:
            low = middle
        else:
            high = middle
    return weights(high)
```

- [ ] **Step 5: Run the tests and linters**

Run: `uv run pytest tests/strategies/bull_bear -v && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): parameters and capped inverse-volatility sizing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The debate set and the fact sheet

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/debate_set.py`
- Create: `src/trading_agent_framework/strategies/bull_bear/fact_sheet.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_debate_set.py`, `tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`

**Interfaces:**
- Consumes: `common.scoring.MomentumRow`, `RankedRow`, `score_stock`, `compute_return_from_prices`.
- Produces:
  - `DebateStock(rank: int, row: MomentumRow, held: bool)` with property `symbol -> str`
  - `ForcedExit(symbol: str, reason: str)` — reason `"unranked"` or `"rank N"`
  - `DebateSet(stocks: tuple[DebateStock, ...], forced_exits: tuple[ForcedExit, ...])` with properties `symbols -> list[str]`, `held -> list[str]` and method `stock(symbol: str) -> DebateStock`
  - `build_debate_set(ranked: Sequence[RankedRow], holdings: Collection[str], *, shortlist_size: int, retention_rank: int) -> DebateSet`
  - `fact_sheet_row(stock: DebateStock, *, sector: str, weight: float) -> dict[str, Any]` with keys `symbol, momentum_rank, momentum_score, return_12m_skip_1m_pct, return_6m_skip_1m_pct, return_3m_pct, return_1m_pct, volatility_pct, drawdown_from_52w_high_pct, vs_sma200_pct, sector, held, weight_pct`

- [ ] **Step 1: Write the failing tests**

`tests/strategies/bull_bear/test_bull_bear_debate_set.py`:

```python
from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.debate_set import ForcedExit, build_debate_set
from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow, score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _row(symbol: str) -> MomentumRow:
    row = score_stock(symbol, [100.0 + i for i in range(300)], [1e6] * 300, CONFIG)
    assert row is not None
    return row


def _ranked(count: int) -> list[RankedRow]:
    return [RankedRow(rank, _row(f"S{rank:02d}")) for rank in range(1, count + 1)]


def test_the_top_15_are_debated_in_rank_order() -> None:
    debate = build_debate_set(_ranked(40), [], shortlist_size=15, retention_rank=35)

    assert debate.symbols == [f"S{rank:02d}" for rank in range(1, 16)]
    assert debate.held == [] and debate.forced_exits == ()


def test_a_holding_ranked_16_to_35_joins_the_debate_tagged_held() -> None:
    debate = build_debate_set(_ranked(40), ["S20", "S35", "S03"], shortlist_size=15, retention_rank=35)

    assert debate.symbols == [*(f"S{rank:02d}" for rank in range(1, 16)), "S20", "S35"]
    assert debate.held == ["S03", "S20", "S35"]
    assert debate.stock("S20").rank == 20 and debate.stock("S20").held
    assert not debate.stock("S01").held


def test_a_holding_ranked_below_the_retention_rank_or_unranked_is_forced_out() -> None:
    debate = build_debate_set(_ranked(40), ["S36", "GONE", "S02"], shortlist_size=15, retention_rank=35)

    assert debate.forced_exits == (ForcedExit("GONE", "unranked"), ForcedExit("S36", "rank 36"))
    assert "S36" not in debate.symbols and "GONE" not in debate.symbols


def test_a_holding_inside_the_shortlist_is_debated_once_and_never_forced_out() -> None:
    """SHV never reaches here: the pipeline passes `Rebalancer.holdings()`, which excludes the parking instrument."""
    debate = build_debate_set(_ranked(20), ["S05"], shortlist_size=15, retention_rank=35)

    assert debate.symbols.count("S05") == 1 and debate.stock("S05").held
    assert debate.forced_exits == ()
```

`tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.common.scoring import score_stock
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG


def _stock(closes: list[float], *, held: bool = False) -> DebateStock:
    row = score_stock("AAA", closes, [1e6] * len(closes), CONFIG)
    assert row is not None
    return DebateStock(rank=3, row=row, held=held)


def test_a_row_reports_the_momentum_facts_in_rounded_percentages() -> None:
    sheet = fact_sheet_row(_stock([100.0 + i for i in range(300)], held=True), sector="Technology", weight=0.1234)

    assert sheet["symbol"] == "AAA" and sheet["momentum_rank"] == 3
    assert sheet["return_12m_skip_1m_pct"] == 200.0
    assert sheet["return_6m_skip_1m_pct"] == 50.0
    assert sheet["return_3m_pct"] == 18.8  # 0.1875
    assert sheet["return_1m_pct"] == 5.6  # (399 - 378) / 378
    assert sheet["drawdown_from_52w_high_pct"] == 0.0  # the last close is the high
    assert sheet["vs_sma200_pct"] == 33.2  # 399 / mean(200..399) - 1
    assert sheet["momentum_score"] == pytest.approx(1.188, abs=1e-3)
    assert (sheet["sector"], sheet["held"], sheet["weight_pct"]) == ("Technology", True, 12.3)
    assert isinstance(sheet["volatility_pct"], float)


def test_the_drawdown_is_measured_from_the_52_week_high() -> None:
    # The last close falls 10% from 398 (a 50% fall would push the 20-day volatility over max_volatility and filter the stock out)
    closes = [100.0 + i for i in range(299)] + [398.0 * 0.9]

    sheet = fact_sheet_row(_stock(closes), sector="UNKNOWN", weight=0.0)

    assert sheet["drawdown_from_52w_high_pct"] == pytest.approx(-10.0, abs=0.05)
    assert sheet["weight_pct"] == 0.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_debate_set.py tests/strategies/bull_bear/test_bull_bear_fact_sheet.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `debate_set.py`**

```python
"""The stocks the agents debate this week, and the holdings code sells without asking them (pure).

The top `shortlist_size` by momentum rank, plus every holding still ranked within `retention_rank` (cross_momentum's
own sell threshold). A holding ranked worse, or unranked (no data, failed a filter, left the universe), is a forced
exit: it never reaches the agents.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from trading_agent_framework.strategies.common.scoring import MomentumRow, RankedRow


@dataclass(frozen=True, slots=True)
class DebateStock:
    rank: int
    row: MomentumRow
    held: bool

    @property
    def symbol(self) -> str:
        return self.row.symbol


@dataclass(frozen=True, slots=True)
class ForcedExit:
    symbol: str
    reason: str  # "unranked" or "rank N"


@dataclass(frozen=True, slots=True)
class DebateSet:
    stocks: tuple[DebateStock, ...]  # in rank order
    forced_exits: tuple[ForcedExit, ...]  # by symbol

    @property
    def symbols(self) -> list[str]:
        return [stock.symbol for stock in self.stocks]

    @property
    def held(self) -> list[str]:
        """The held stocks in the debate, sorted (the judge must keep or drop each)."""
        return sorted(stock.symbol for stock in self.stocks if stock.held)

    def stock(self, symbol: str) -> DebateStock:
        return next(stock for stock in self.stocks if stock.symbol == symbol)


def build_debate_set(ranked: Sequence[RankedRow], holdings: Collection[str], *, shortlist_size: int, retention_rank: int) -> DebateSet:
    """`holdings` are the stocks held, the parking instrument excluded (`Rebalancer.holdings()`)."""
    held = set(holdings)
    rank_of = {entry.row.symbol: entry.rank for entry in ranked}
    stocks = tuple(
        DebateStock(entry.rank, entry.row, entry.row.symbol in held)
        for entry in ranked
        if entry.rank <= shortlist_size or (entry.row.symbol in held and entry.rank <= retention_rank)
    )
    forced = tuple(
        ForcedExit(symbol, "unranked" if symbol not in rank_of else f"rank {rank_of[symbol]}")
        for symbol in sorted(held)
        if symbol not in rank_of or rank_of[symbol] > retention_rank
    )
    return DebateSet(stocks=stocks, forced_exits=forced)
```

- [ ] **Step 4: Write `fact_sheet.py`**

```python
"""The fact sheet: one row per debate-set stock, computed by code from its momentum row (pure).

The agents read these numbers and must not recompute them; percentages are rounded to one decimal so a 25-stock
prompt stays short.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from trading_agent_framework.strategies.bull_bear.debate_set import DebateStock
from trading_agent_framework.strategies.common.scoring import compute_return_from_prices

_YEAR = 252
_SMA = 200
_MONTH = 21


def _pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100, 1)


def drawdown_from_high(closes: Sequence[float]) -> float | None:
    """The last close against the highest close of the last 252 sessions (0 at a high, negative below it)."""
    window = closes[-_YEAR:]
    high = max(window) if window else 0.0
    return None if high <= 0 else closes[-1] / high - 1


def versus_sma200(closes: Sequence[float]) -> float | None:
    """The last close against its 200-session simple average; None with fewer closes."""
    if len(closes) < _SMA:
        return None
    sma = sum(closes[-_SMA:]) / _SMA
    return None if sma <= 0 else closes[-1] / sma - 1


def fact_sheet_row(stock: DebateStock, *, sector: str, weight: float) -> dict[str, Any]:
    """`weight` is the current share of portfolio value (open orders counted), 0 when not held."""
    row = stock.row
    inputs = row.inputs
    return {
        "symbol": row.symbol,
        "momentum_rank": stock.rank,
        "momentum_score": round(row.score, 3),
        "return_12m_skip_1m_pct": _pct(inputs.ret_12_1m),
        "return_6m_skip_1m_pct": _pct(inputs.ret_6_1m),
        "return_3m_pct": _pct(inputs.ret_3m),
        "return_1m_pct": _pct(compute_return_from_prices(row.closes, _MONTH, 0)),
        "volatility_pct": _pct(row.volatility),
        "drawdown_from_52w_high_pct": _pct(drawdown_from_high(row.closes)),
        "vs_sma200_pct": _pct(versus_sma200(row.closes)),
        "sector": sector,
        "held": stock.held,
        "weight_pct": _pct(weight),
    }
```

- [ ] **Step 5: Run the tests and linters**

Run: `uv run pytest tests/strategies/bull_bear -v && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): debate set (top 15 plus holdings ranked to 35) and fact sheet

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Hand-offs — validators, recorder and submit tools

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/handoff.py` (NO `from __future__ import annotations`)
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`

**Interfaces:**
- Consumes: `BullBearParams`.
- Produces:
  - constants `LEVELS = ("low", "medium", "high")`, `CONCERNS = ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none")`, stage names `NOTE, BULL, BEAR, PICKS`
  - `HandoffError(ValueError)`
  - dataclasses `Note(symbol, note)`, `BullCase(symbol, conviction, argument)`, `BearCase(symbol, risk, concern, argument)`, `Choice(symbol, reason)`, `Picks(picks: tuple[Choice, ...], drops: tuple[Choice, ...])`
  - `HandoffRecorder(params)` with `expect_note(symbol)`, `expect_bull(symbols)`, `expect_bear(symbols)`, `expect_picks(allowed, held)`, properties `submitted -> bool`, `submission -> Any` (a `Note`, `list[BullCase]`, `list[BearCase]` or `Picks`), attribute `last_error: str | None`, `submit(stage, *args) -> dict[str, Any]`
  - `submit_tools(recorder) -> dict[str, Callable[..., dict[str, Any]]]` keyed `submit_note`, `submit_bull_case`, `submit_bear_case`, `submit_picks`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from typing import Any

import pytest

from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, Choice, HandoffRecorder, Note, Picks, submit_tools
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

DEBATE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


def _tools() -> tuple[HandoffRecorder, dict[str, Any]]:
    recorder = HandoffRecorder(BullBearParams())
    return recorder, submit_tools(recorder)


def _bull(symbols: list[str], conviction: str = "high") -> list[dict[str, Any]]:
    return [{"symbol": s, "conviction": conviction, "argument": "strong trend"} for s in symbols]


def _bear(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "risk": "low", "concern": "none", "argument": "no red flag"} for s in symbols]


def _choices(symbols: list[str]) -> list[dict[str, Any]]:
    return [{"symbol": s, "reason": "won the debate"} for s in symbols]


# --- note ---------------------------------------------------------------------------------------


def test_a_note_for_the_armed_symbol_is_recorded_stripped() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    assert tools["submit_note"](" aaa ", "  2026-09-30: revenue up 12% y/y  ") == {"status": "recorded"}
    assert recorder.submitted and recorder.submission == Note("AAA", "2026-09-30: revenue up 12% y/y")


def test_a_note_for_another_symbol_is_refused_with_the_symbol_to_use() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    result = tools["submit_note"]("BBB", "facts")

    assert result == {"error": "this note is for AAA: submit it with symbol AAA"} and not recorder.submitted


def test_an_empty_or_long_note_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    assert "empty" in tools["submit_note"]("AAA", "   ")["error"]
    assert "under 500" in tools["submit_note"]("AAA", "x" * 501)["error"]


# --- bull and bear ------------------------------------------------------------------------------


def test_a_bull_case_covering_every_stock_once_is_recorded_with_normalised_values() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)
    cases = _bull(DEBATE)
    cases[0] = {"symbol": " aaa ", "conviction": " HIGH ", "argument": " strong trend "}

    assert tools["submit_bull_case"](cases) == {"status": "recorded"}
    assert recorder.submission[0] == BullCase("AAA", "high", "strong trend")
    assert [case.symbol for case in recorder.submission] == DEBATE


@pytest.mark.parametrize(
    ("cases", "message"),
    [
        (_bull(DEBATE[:-1]), "no case for FFF"),
        (_bull([*DEBATE, "AAA"]), "AAA appears twice"),
        (_bull([*DEBATE, "ZZZ"]), "ZZZ is not in the debate"),
        (_bull(DEBATE, conviction="huge"), "conviction for AAA must be one of low, medium, high"),
        ([{"symbol": "AAA", "conviction": "high", "argument": "x" * 301}, *_bull(DEBATE[1:])], "under 300"),
        ("not a list", "cases must be a list"),
    ],
)
def test_a_bad_bull_case_is_refused(cases: Any, message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)

    assert message in tools["submit_bull_case"](cases)["error"]
    assert recorder.last_error is not None and message in recorder.last_error


def test_a_bear_case_needs_a_known_concern() -> None:
    recorder, tools = _tools()
    recorder.expect_bear(DEBATE)
    cases = _bear(DEBATE)
    cases[1] = {"symbol": "BBB", "risk": "high", "concern": "weather", "argument": "storms"}

    assert "concern for BBB must be one of" in tools["submit_bear_case"](cases)["error"]

    cases[1]["concern"] = "Momentum_Exhaustion"
    assert tools["submit_bear_case"](cases) == {"status": "recorded"}
    assert recorder.submission[1] == BearCase("BBB", "high", "momentum_exhaustion", "storms")


# --- picks --------------------------------------------------------------------------------------


def test_picks_within_the_bounds_with_every_unpicked_holding_dropped_are_recorded() -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=["EEE", "FFF"])

    result = tools["submit_picks"](_choices(["AAA", "BBB", "CCC", "DDD", "EEE"]), [{"symbol": "FFF", "reason": "lost the debate"}])

    assert result == {"status": "recorded"}
    assert recorder.submission == Picks(picks=tuple(Choice(s, "won the debate") for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]), drops=(Choice("FFF", "lost the debate"),))


def test_drops_may_be_omitted_when_nothing_held_is_left_out() -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=[])

    assert tools["submit_picks"](_choices(DEBATE[:5])) == {"status": "recorded"}


@pytest.mark.parametrize(
    ("picks", "drops", "message"),
    [
        (_choices(DEBATE[:4]), [], "pick between 5 and 10 stocks, got 4"),
        (_choices([*DEBATE[:4], "ZZZ"]), [], "ZZZ is not in the debate"),
        (_choices([*DEBATE[:4], "AAA"]), [], "AAA appears twice"),
        (_choices(DEBATE[:5]), [], "FFF is held and not picked"),
        (_choices(DEBATE[:5]), _choices(["AAA", "FFF"]), "AAA cannot be dropped"),
    ],
)
def test_bad_picks_are_refused(picks: list[dict[str, Any]], drops: list[dict[str, Any]], message: str) -> None:
    recorder, tools = _tools()
    recorder.expect_picks(DEBATE, held=["FFF"])

    assert message in tools["submit_picks"](picks, drops)["error"]


# --- the recorder -------------------------------------------------------------------------------


def test_the_first_valid_submission_is_final() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)
    tools["submit_bull_case"](_bull(DEBATE))

    assert tools["submit_bull_case"](_bull(DEBATE, conviction="low")) == {"error": "already recorded for this stage"}
    assert recorder.submission[0].conviction == "high"


def test_a_tool_called_at_the_wrong_stage_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_bull(DEBATE)

    assert tools["submit_bear_case"](_bear(DEBATE)) == {"error": "submit_bear_case is not expected at this point of the review"}


def test_arming_a_stage_clears_the_previous_submission_and_error() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("BBB", "facts")
    recorder.expect_note("BBB")

    assert (recorder.submitted, recorder.submission, recorder.last_error) == (False, None, None)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `handoff.py`**

```python
"""The hand-off between the four agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool. A tool validates its argument against what the pipeline armed
the recorder with for this stage, records the first valid submission, and otherwise returns `{"error": ...}` so the
model can correct itself in the same run. Free text from an agent is never trusted.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema
from the function's real annotations (as `bill_ackman/handoff.py` and `memory/tools.py` do).
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams

LEVELS = ("low", "medium", "high")
CONCERNS = ("valuation", "momentum_exhaustion", "earnings", "fundamentals", "news_event", "sector", "none")

NOTE, BULL, BEAR, PICKS = "note", "bull", "bear", "picks"
_TOOL_NAMES = {NOTE: "submit_note", BULL: "submit_bull_case", BEAR: "submit_bear_case", PICKS: "submit_picks"}


class HandoffError(ValueError):
    """A submission that breaks a rule; the message tells the model what is wrong and what is allowed."""


@dataclass(frozen=True, slots=True)
class Note:
    symbol: str
    note: str


@dataclass(frozen=True, slots=True)
class BullCase:
    symbol: str
    conviction: str  # one of LEVELS
    argument: str


@dataclass(frozen=True, slots=True)
class BearCase:
    symbol: str
    risk: str  # one of LEVELS
    concern: str  # one of CONCERNS
    argument: str


@dataclass(frozen=True, slots=True)
class Choice:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class Picks:
    picks: tuple[Choice, ...]  # in the judge's order: the buy order
    drops: tuple[Choice, ...]  # the held debate stocks not picked


# --- validation (pure) --------------------------------------------------------------------------


def _items(raw: Any, what: str) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list | tuple):
        raise HandoffError(f"{what} must be a list of objects")
    items = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise HandoffError(f"{what} item {index} must be an object, got {type(item).__name__}")
        items.append(item)
    return items


def _symbol(item: Mapping[str, Any], index: int) -> str:
    value = item.get("symbol")
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"item {index} has no symbol")
    return value.strip().upper()


def _text(item: Mapping[str, Any], key: str, symbol: str, max_chars: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{symbol} has no {key}: give one short sentence")
    text = value.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the {key} for {symbol} is {len(text)} characters: keep it under {max_chars}")
    return text


def _one_of(item: Mapping[str, Any], key: str, symbol: str, allowed: Sequence[str]) -> str:
    value = item.get(key)
    choice = value.strip().lower() if isinstance(value, str) else None
    if choice is None or choice not in allowed:
        raise HandoffError(f"the {key} for {symbol} must be one of {', '.join(allowed)}")
    return choice


def _covering(raw: Any, expected: Sequence[str]) -> list[tuple[str, Mapping[str, Any]]]:
    """(symbol, item) for exactly one item per symbol in `expected`, in submission order."""
    items = _items(raw, "cases")
    wanted = set(expected)
    seen: set[str] = set()
    covered = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in wanted:
            raise HandoffError(f"{symbol} is not in the debate: give cases only for {', '.join(expected)}")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        covered.append((symbol, item))
    missing = [symbol for symbol in expected if symbol not in seen]
    if missing:
        raise HandoffError(f"no case for {', '.join(missing)}: give exactly one case per stock")
    return covered


def validate_note(symbol: Any, note: Any, *, expected: str, max_chars: int) -> Note:
    if not isinstance(symbol, str) or symbol.strip().upper() != expected:
        raise HandoffError(f"this note is for {expected}: submit it with symbol {expected}")
    if not isinstance(note, str) or not note.strip():
        raise HandoffError(f"the note for {expected} is empty: write the dated facts you found, or say that you found none")
    text = note.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the note for {expected} is {len(text)} characters: keep it under {max_chars}")
    return Note(expected, text)


def validate_bull_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BullCase]:
    return [BullCase(symbol, _one_of(item, "conviction", symbol, LEVELS), _text(item, "argument", symbol, max_chars)) for symbol, item in _covering(raw, expected)]


def validate_bear_cases(raw: Any, *, expected: Sequence[str], max_chars: int) -> list[BearCase]:
    return [
        BearCase(symbol, _one_of(item, "risk", symbol, LEVELS), _one_of(item, "concern", symbol, CONCERNS), _text(item, "argument", symbol, max_chars))
        for symbol, item in _covering(raw, expected)
    ]


def validate_picks(picks_raw: Any, drops_raw: Any, *, allowed: Sequence[str], held: Collection[str], min_picks: int, max_picks: int, max_chars: int) -> Picks:
    """`min_picks` to `max_picks` unique stocks from `allowed`; `drops` names exactly the `held` stocks not picked."""
    items = _items(picks_raw, "picks")
    if not min_picks <= len(items) <= max_picks:
        raise HandoffError(f"pick between {min_picks} and {max_picks} stocks, got {len(items)}")
    allowed_set = set(allowed)
    picked: set[str] = set()
    picks = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in allowed_set:
            raise HandoffError(f"{symbol} is not in the debate (choose from {', '.join(allowed)})")
        if symbol in picked:
            raise HandoffError(f"{symbol} appears twice")
        picked.add(symbol)
        picks.append(Choice(symbol, _text(item, "reason", symbol, max_chars)))
    must_drop = [symbol for symbol in held if symbol not in picked]
    dropped: set[str] = set()
    drops = []
    for index, item in enumerate(_items([] if drops_raw is None else drops_raw, "drops"), start=1):
        symbol = _symbol(item, index)
        if symbol not in must_drop:
            raise HandoffError(f"{symbol} cannot be dropped: drops lists only the held stocks you did not pick ({', '.join(must_drop) or 'none'})")
        if symbol in dropped:
            raise HandoffError(f"{symbol} appears twice")
        dropped.add(symbol)
        drops.append(Choice(symbol, _text(item, "reason", symbol, max_chars)))
    missing = [symbol for symbol in must_drop if symbol not in dropped]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        raise HandoffError(f"{', '.join(missing)} {verb} held and not picked: list each in drops with a reason")
    return Picks(picks=tuple(picks), drops=tuple(drops))


# --- the recorder ---------------------------------------------------------------------------------


class HandoffRecorder:
    """Holds what the current stage must validate against, and the first valid submission for it."""

    def __init__(self, params: BullBearParams) -> None:
        self._params = params
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def expect_note(self, symbol: str) -> None:
        self._arm(NOTE, {"expected": symbol})

    def expect_bull(self, symbols: Sequence[str]) -> None:
        self._arm(BULL, {"expected": list(symbols)})

    def expect_bear(self, symbols: Sequence[str]) -> None:
        self._arm(BEAR, {"expected": list(symbols)})

    def expect_picks(self, allowed: Sequence[str], held: Sequence[str]) -> None:
        self._arm(PICKS, {"allowed": list(allowed), "held": list(held)})

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission (`Note`, `list[BullCase]`, `list[BearCase]` or `Picks`), else None."""
        return self._submission

    def submit(self, stage: str, *args: Any) -> dict[str, Any]:
        """What a submit tool returns: `{"status": "recorded"}` or `{"error": ...}` (recorded in `last_error`)."""
        if self._stage != stage:
            return self._fail(f"{_TOOL_NAMES[stage]} is not expected at this point of the review")
        if self._submitted:
            return self._fail("already recorded for this stage")
        params, context = self._params, self._context
        try:
            if stage == NOTE:
                value: Any = validate_note(args[0], args[1], expected=context["expected"], max_chars=params.note_max_chars)
            elif stage == BULL:
                value = validate_bull_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
            elif stage == BEAR:
                value = validate_bear_cases(args[0], expected=context["expected"], max_chars=params.argument_max_chars)
            else:
                value = validate_picks(
                    args[0],
                    args[1],
                    allowed=context["allowed"],
                    held=context["held"],
                    min_picks=params.min_picks,
                    max_picks=params.max_picks,
                    max_chars=params.reason_max_chars,
                )
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}


# --- the tools ------------------------------------------------------------------------------------


def submit_tools(recorder: HandoffRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The four submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_note(symbol: str, note: str) -> dict[str, Any]:
        """Submit your note on the stock: its symbol and the dated facts you found (no opinion)."""
        return recorder.submit(NOTE, symbol, note)

    def submit_bull_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one case per stock: objects with symbol, conviction (low, medium or high) and argument."""
        return recorder.submit(BULL, cases)

    def submit_bear_case(cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one case per stock: objects with symbol, risk (low, medium or high), concern and argument."""
        return recorder.submit(BEAR, cases)

    def submit_picks(picks: list[dict[str, Any]], drops: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Submit the stocks that win the debate (objects with symbol and reason) and, for each held stock you do not pick, a drop (symbol and reason)."""
        return recorder.submit(PICKS, picks, drops)

    return {"submit_note": submit_note, "submit_bull_case": submit_bull_case, "submit_bear_case": submit_bear_case, "submit_picks": submit_picks}
```

- [ ] **Step 4: Run the tests and linters**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -v && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/handoff.py tests/strategies/bull_bear/test_bull_bear_handoff.py
git commit -m "feat(bull_bear): submit tools, validators and the hand-off recorder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: State file and review log

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/state.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_state.py`

**Interfaces:**
- Produces:
  - `STATE_VERSION = 1`
  - `state_path(project_root: Path, mode: TradingMode) -> Path` → `<root>/data/bull_bear_state_<mode>.json`
  - `BullBearState(last_completed_review: str | None = None, abandoned_streak: int = 0, last_picks: list[dict[str, str]] = [])` (frozen; each pick `{"symbol", "reason", "date"}`)
  - `StateStore(path)` with `load() -> BullBearState`, `save(state) -> None`, `wipe() -> None` (never raise on I/O)
  - `ReviewLog(path: Path | None)` with `append(record: dict[str, Any]) -> None`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bull_bear.state import STATE_VERSION, BullBearState, ReviewLog, StateStore, state_path


def test_the_state_file_is_per_mode_under_data(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "bull_bear_state_paper.json"


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == BullBearState()


def test_a_saved_state_loads_back(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")
    state = BullBearState(last_completed_review="2026-10-06", abandoned_streak=0, last_picks=[{"symbol": "AAA", "reason": "won", "date": "2026-10-06"}])

    store.save(state)

    assert store.load() == state
    assert json.loads((tmp_path / "data" / "state.json").read_text())["version"] == STATE_VERSION


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 99, "abandoned_streak": 1}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": -1}),
        json.dumps({"version": STATE_VERSION, "last_picks": [{"symbol": "AAA"}]}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": True}),
    ],
)
def test_an_unreadable_or_invalid_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content)

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "starting from an empty state" in caplog.text


def test_wipe_deletes_the_file_and_is_safe_without_one(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(BullBearState(abandoned_streak=2))

    store.wipe()
    store.wipe()

    assert store.load() == BullBearState()


def test_the_review_log_appends_one_json_line_per_review(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "run" / "reviews.jsonl")

    log.append({"date": "2026-10-06", "abandoned": False})
    log.append({"date": "2026-10-13", "abandoned": True})

    lines = (tmp_path / "run" / "reviews.jsonl").read_text().splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-10-06", "2026-10-13"]


def test_a_review_log_without_a_path_does_nothing(tmp_path: Path) -> None:
    ReviewLog(None).append({"date": "2026-10-06"})

    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_state.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `state.py`**

```python
"""What bull_bear remembers between reviews, and the per-review log.

`StateStore` keeps the date of the last completed review (a completed Tuesday is not reviewed again after a
restart), the abandoned-review streak and the last picks, in one small JSON file per mode, written atomically. A
missing or invalid file is an empty state. `ReviewLog` appends one JSON line per review to the run directory.
Neither ever raises on an I/O problem: bookkeeping must not stop a review.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "BullBearState")

STATE_VERSION = 1
_PICK_KEYS = {"symbol", "reason", "date"}


def state_path(project_root: Path, mode: TradingMode) -> Path:
    """`<project_root>/data/bull_bear_state_<mode>.json`."""
    return project_root / "data" / f"bull_bear_state_{mode.value}.json"


@dataclass(frozen=True, slots=True)
class BullBearState:
    last_completed_review: str | None = None  # ISO market date of the last completed review
    abandoned_streak: int = 0  # abandoned reviews in a row
    last_picks: list[dict[str, str]] = field(default_factory=list)  # [{symbol, reason, date}] of the last completed review


def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last = raw.get("last_completed_review")
    streak = raw.get("abandoned_streak", 0)
    picks = raw.get("last_picks", [])
    return (
        (last is None or isinstance(last, str))
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
        and isinstance(picks, list)
        and all(isinstance(pick, dict) and set(pick) == _PICK_KEYS and all(isinstance(value, str) for value in pick.values()) for pick in picks)
    )


class StateStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> BullBearState:
        """The saved state; an empty one when the file is missing, unreadable, or not a valid state (warned about)."""
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return BullBearState()
        except (OSError, ValueError) as exc:
            logger.log_warning(f"state file {self._path} is unreadable, starting from an empty state: {exc}")
            return BullBearState()
        if not _valid(raw):
            logger.log_warning(f"state file {self._path} is not a valid state, starting from an empty state")
            return BullBearState()
        return BullBearState(
            last_completed_review=raw.get("last_completed_review"),
            abandoned_streak=raw.get("abandoned_streak", 0),
            last_picks=[dict(pick) for pick in raw.get("last_picks", [])],
        )

    def save(self, state: BullBearState) -> None:
        """Write the state atomically (temporary file in the same directory, then replace); an I/O error is logged."""
        temporary: str | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"version": STATE_VERSION, **asdict(state)}, handle)
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.log_warning(f"state could not be saved to {self._path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def wipe(self) -> None:
        """Delete the state file (a backtest starts clean); safe when there is none."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.log_warning(f"state file {self._path} could not be deleted: {exc}")


class ReviewLog:
    """`reviews.jsonl`: one JSON line per review. With no path (a strategy run outside a runner) it does nothing."""

    def __init__(self, path: Path | None) -> None:
        self._path = path

    def append(self, record: dict[str, Any]) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            logger.log_warning(f"review log {self._path} could not be written: {exc}")
```

- [ ] **Step 4: Run the tests and linters**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_state.py -v && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS. If the warning assertion fails because `ColorLogger` does not propagate to `caplog`, check how `tests/strategies/bill_ackman/test_ackman_state.py` asserts its warnings and do the same.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/state.py tests/strategies/bull_bear/test_bull_bear_state.py
git commit -m "feat(bull_bear): state file and review log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Daily bars for a review — Yahoo batch (paper/live) and the gate (backtests)

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/market_data.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_market_data.py`

**Interfaces:**
- Consumes: `common.sessions.completed_bars`, `entities.bars.Bars`, `YahooDataError`, `BrokerError`.
- Produces:
  - `HISTORY_BARS = 300`
  - `DataUnavailable(Exception)`
  - `DailySeries(closes: list[float], volumes: list[float])` (frozen)
  - `DailyBars` Protocol: `load(universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]`
  - `BarsSource` Protocol: `bars(symbols: Sequence[str], today: date) -> dict[str, Bars]` (satisfied by `YahooDailyBars`)
  - `YahooBars(source: BarsSource, *, today: Callable[[], date], sleep: Callable[[float], None], warn: Callable[[str], None], retry_delays: Sequence[float], min_coverage: float)`
  - `GateBars(fetch: Callable[[str, int], Bars | None], *, today: Callable[[], date], warn: Callable[[str], None])`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.bull_bear.market_data import DataUnavailable, GateBars, YahooBars
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError, YahooDataError

TODAY = date(2026, 10, 6)


def _bars(symbol: str, sessions: int = 302, last: date = TODAY) -> Bars:
    """Daily bars stamped at the 16:00 close, closes 100, 101, ...; the last one is dated `last` (today: still forming)."""
    dates = [d.date() for d in pd.bdate_range(end=last, periods=sessions)]
    closes = [100.0 + i for i in range(sessions)]
    index = pd.DatetimeIndex([datetime.combine(d, time(16), tzinfo=MARKET_TZ) for d in dates])
    frame = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1e6] * sessions}, index=index)
    return Bars(Asset(symbol), "day", frame)


class FakeSource:
    def __init__(self, answers: list[dict[str, Bars] | Exception]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[str], date]] = []

    def bars(self, symbols, today):  # noqa: ANN001, ANN201
        self.calls.append((list(symbols), today))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _yahoo(source: FakeSource, sleeps: list[float], warnings: list[str]) -> YahooBars:
    return YahooBars(source, today=lambda: TODAY, sleep=sleeps.append, warn=warnings.append, retry_delays=(60.0, 180.0), min_coverage=0.5)


def test_a_usable_batch_gives_300_completed_sessions_per_universe_symbol() -> None:
    source = FakeSource([{"AAA": _bars("AAA"), "BBB": _bars("BBB"), "XXX": _bars("XXX")}])

    series = _yahoo(source, [], []).load(["AAA", "BBB"], held=[])

    assert set(series) == {"AAA", "BBB"}  # a symbol outside the universe is ignored
    assert len(series["AAA"].closes) == 300
    assert series["AAA"].closes[-1] == 400.0  # today's 401 is dropped
    assert series["AAA"].closes[0] == 101.0
    assert series["AAA"].volumes == [1e6] * 300
    assert source.calls == [(["AAA", "BBB"], TODAY)]


def test_a_batch_covering_too_little_of_the_universe_is_retried_then_unavailable() -> None:
    sleeps: list[float] = []
    warnings: list[str] = []
    one = {"AAA": _bars("AAA")}
    source = FakeSource([one, one, one])

    with pytest.raises(DataUnavailable, match="Yahoo covers only 1/4 symbols"):
        _yahoo(source, sleeps, warnings).load(["AAA", "BBB", "CCC", "DDD"], held=[])

    assert sleeps == [60.0, 180.0]
    assert len(warnings) == 2 and "retry in 60s" in warnings[0]


def test_a_failed_download_is_retried_and_a_later_success_is_used() -> None:
    sleeps: list[float] = []
    source = FakeSource([YahooDataError("429"), {"AAA": _bars("AAA")}])

    series = _yahoo(source, sleeps, []).load(["AAA"], held=[])

    assert set(series) == {"AAA"} and sleeps == [60.0]


def test_a_batch_missing_a_held_universe_stock_is_unusable() -> None:
    half = {"AAA": _bars("AAA")}
    source = FakeSource([half, half, half])

    with pytest.raises(DataUnavailable, match="no bars for held BBB"):
        _yahoo(source, [], []).load(["AAA", "BBB"], held=["BBB", "OUTSIDE"])  # OUTSIDE is not in the universe: not required


def test_the_gate_reads_each_symbol_skips_failures_and_drops_todays_bar() -> None:
    warnings: list[str] = []
    answers = {"AAA": _bars("AAA"), "NONE": None}
    requested: list[tuple[str, int]] = []

    def fetch(symbol: str, length: int) -> Bars | None:
        requested.append((symbol, length))
        if symbol == "BAD":
            raise BrokerError("invalid symbol")
        return answers[symbol]

    series = GateBars(fetch, today=lambda: TODAY, warn=warnings.append).load(["AAA", "BAD", "NONE"], held=[])

    assert set(series) == {"AAA"} and series["AAA"].closes[-1] == 400.0
    assert requested == [("AAA", 301), ("BAD", 301), ("NONE", 301)]
    assert any("BAD" in warning for warning in warnings)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_market_data.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `market_data.py`**

```python
"""Where a review's daily bars come from: one Yahoo batch in paper/live, the backtest gate otherwise.

Paper/live read Yahoo for the same reason as cross_momentum: Alpaca's IEX bars carry ~5% of consolidated volume, so
the $20M dollar-volume filter would empty the ranking. Both sources hand the pipeline completed sessions only (the
bar dated today is partial while the session is open), at most `HISTORY_BARS` of them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.common.sessions import completed_bars
from trading_agent_framework.utils.errors import BrokerError, YahooDataError

HISTORY_BARS = 300  # completed sessions scored: cross_momentum's own window


class DataUnavailable(Exception):
    """No usable daily bars for this review; the message says why."""


@dataclass(frozen=True, slots=True)
class DailySeries:
    closes: list[float]  # oldest first
    volumes: list[float]


class DailyBars(Protocol):
    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]: ...


class BarsSource(Protocol):
    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]: ...


def _series(bars: Bars, today: date) -> DailySeries | None:
    frame = completed_bars(bars.df, today).tail(HISTORY_BARS)
    if frame.empty:
        return None
    return DailySeries(closes=[float(value) for value in frame["close"]], volumes=[float(value) for value in frame["volume"]])


class YahooBars:
    """Paper/live: one batched Yahoo download per review, fetched again after each delay while it is unusable.

    Unusable when the download fails, covers less than `min_coverage` of the universe, or misses a held universe
    stock (it would be forced out as unranked on missing data). After the last attempt: `DataUnavailable`.
    """

    def __init__(
        self,
        source: BarsSource,
        *,
        today: Callable[[], date],
        sleep: Callable[[float], None],
        warn: Callable[[str], None],
        retry_delays: Sequence[float],
        min_coverage: float,
    ) -> None:
        self._source = source
        self._today = today
        self._sleep = sleep
        self._warn = warn
        self._retry_delays = tuple(retry_delays)
        self._min_coverage = min_coverage

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        problem = "no attempt made"
        for delay in (*self._retry_delays, None):
            bars, found = self._attempt(universe, held)
            if found is None:
                today = self._today()
                in_universe = set(universe)
                result = {}
                for symbol, symbol_bars in bars.items():
                    series = _series(symbol_bars, today) if symbol in in_universe else None
                    if series is not None:
                        result[symbol] = series
                return result
            problem = found
            if delay is not None:
                self._warn(f"{problem}: retry in {delay:.0f}s")
                self._sleep(delay)
        raise DataUnavailable(problem)

    def _attempt(self, universe: Sequence[str], held: Collection[str]) -> tuple[dict[str, Bars], str | None]:
        try:
            bars = self._source.bars(list(universe), self._today())
        except YahooDataError as exc:
            return {}, f"Yahoo lookup failed ({exc})"
        covered = sum(1 for symbol in universe if symbol in bars)
        if covered < len(universe) * self._min_coverage:
            return {}, f"Yahoo covers only {covered}/{len(universe)} symbols"
        missing_held = sorted((set(held) & set(universe)) - bars.keys())
        if missing_held:
            return {}, f"Yahoo has no bars for held {', '.join(missing_held)}"
        return bars, None


class GateBars:
    """Backtests: each symbol through `Strategy.get_historical_prices`, the no-look-ahead gate.

    A `BrokerError` for one symbol skips it (as cross_momentum does); a `BacktestError` propagates and abandons the
    review.
    """

    def __init__(self, fetch: Callable[[str, int], Bars | None], *, today: Callable[[], date], warn: Callable[[str], None]) -> None:
        self._fetch = fetch
        self._today = today
        self._warn = warn

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        today = self._today()
        result: dict[str, DailySeries] = {}
        for symbol in universe:
            try:
                bars = self._fetch(symbol, HISTORY_BARS + 1)  # one extra: today's bar, if any, is dropped
            except BrokerError as exc:
                self._warn(f"Skipping {symbol}: failed to fetch bars ({exc})")
                continue
            if bars is None or bars.empty:
                continue
            series = _series(bars, today)
            if series is not None:
                result[symbol] = series
        return result
```

- [ ] **Step 4: Run the tests and linters**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_market_data.py -v && uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/market_data.py tests/strategies/bull_bear/test_bull_bear_market_data.py
git commit -m "feat(bull_bear): daily bars from one Yahoo batch (paper/live) or the backtest gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Prompts and the review pipeline

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/prompts.py`
- Create: `src/trading_agent_framework/strategies/bull_bear/pipeline.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8: `score_stock`, `rank`, `build_debate_set`, `fact_sheet_row`, `HandoffRecorder`, `capped_inverse_volatility`, `target_portfolio`, `Rebalancer`, `StateStore`, `BullBearState`, `ReviewLog`, `DailyBars`, `DataUnavailable`, `BullBearParams`.
- Produces:
  - `prompts`: `RESEARCHER_SYSTEM`, `BULL_SYSTEM`, `BEAR_SYSTEM`, `JUDGE_SYSTEM`, `researcher_task(symbol: str, max_chars: int) -> str`, `BULL_TASK`, `BEAR_TASK`, `JUDGE_TASK`, `retry_prompt(tool: str, error: str) -> str`, `UNAVAILABLE_NOTE = "research unavailable"`
  - `pipeline.ReviewOutcome(completed: bool, abandoned_streak: int)`
  - `pipeline.ReviewPipeline(*, strategy, params, agents, recorder, state, review_log, rebalancer, universe, bars, sector_of, momentum)` with `run() -> ReviewOutcome` and `completed_today() -> bool`
  - agent names used: `"researcher"`, `"bull"`, `"bear"`, `"judge"`
  - judge context keys: `current_datetime`, `stocks` (each `{fact_sheet, note, bull: {conviction, argument}, bear: {risk, concern, argument}}`), `held`, `forced_exits`, `constraints`

- [ ] **Step 1: Write the failing prompt tests**

`tests/strategies/bull_bear/test_bull_bear_prompts.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.strategies.bull_bear import prompts


@pytest.mark.parametrize(
    ("system", "tool"),
    [
        (prompts.RESEARCHER_SYSTEM, "submit_note"),
        (prompts.BULL_SYSTEM, "submit_bull_case"),
        (prompts.BEAR_SYSTEM, "submit_bear_case"),
        (prompts.JUDGE_SYSTEM, "submit_picks"),
    ],
)
def test_every_system_prompt_requires_english_and_names_its_submit_tool(system: str, tool: str) -> None:
    assert "English" in system and tool in system


@pytest.mark.parametrize("system", [prompts.RESEARCHER_SYSTEM, prompts.BULL_SYSTEM, prompts.BEAR_SYSTEM, prompts.JUDGE_SYSTEM])
def test_no_prompt_states_the_review_cadence(system: str) -> None:
    assert not any(word in system.lower() for word in ("weekly", "every week", "tuesday", "daily", "every day"))


def test_only_the_bear_prompt_lists_the_concerns() -> None:
    assert "momentum_exhaustion" in prompts.BEAR_SYSTEM and "momentum_exhaustion" not in prompts.BULL_SYSTEM


def test_the_researcher_task_names_the_symbol_and_the_note_limit() -> None:
    task = prompts.researcher_task("AAA", 500)

    assert "AAA" in task and "500" in task


def test_the_retry_prompt_quotes_the_error_and_the_tool() -> None:
    text = prompts.retry_prompt("submit_picks", "pick between 5 and 10 stocks, got 4")

    assert "submit_picks" in text and "got 4" in text
```

- [ ] **Step 2: Write `prompts.py`**

```python
"""The four system prompts and task prompts (English only, written for a local model).

Each starts from the lumibot example's one-sentence role and adds what a local model needs: how to read the fact
sheet, the scale it rates on, the language rule, and the contract of its single submit tool. No agent is given an
order tool; code sizes and places every order. No prompt states the review cadence.
"""

from __future__ import annotations

_ENGLISH = "Write everything in English: your notes, arguments, reasons and every tool argument, even if a source you read drifts into another language."

_FACT_SHEET = (
    "Each stock comes with a fact sheet computed by code; do not recompute its numbers. momentum_rank is the stock's rank "
    "by momentum across the whole universe (1 is the strongest) and momentum_score the weighted 12-, 6- and 3-month return "
    "behind it. return_12m_skip_1m_pct and return_6m_skip_1m_pct leave out the last month; return_3m_pct and return_1m_pct "
    "do not. volatility_pct is the annualised volatility of the last 20 sessions. drawdown_from_52w_high_pct is how far "
    "the last close is below its one-year high. vs_sma200_pct is how far it is above (or below) its 200-day average. "
    "held says whether the portfolio owns it, and weight_pct its current share of the portfolio."
)

_SUBMIT_RULE = "If the tool returns an error, read it and call it again with a corrected argument. Once it returns status recorded, reply with one line and call no other tool."

RESEARCHER_SYSTEM = (
    "You are the researcher of a long-only stock portfolio that buys strong momentum stocks. You receive one stock at a "
    "time and gather the facts a bull and a bear will argue from. Do not trade and do not give an opinion.\n\n"
    f"{_FACT_SHEET}\n"
    "Look for what the fact sheet cannot show: recent company news (results, guidance, deals, lawsuits, management "
    "changes, analyst actions) and the trend of revenue, operating income and debt in the latest filings. Your tool calls "
    "are limited: spend them on what matters most for this stock. Write facts only, each with its date, and no "
    "recommendation. Do not repeat the fact sheet's numbers. If you find nothing useful, say so in one line.\n\n"
    f"{_ENGLISH}\n\n"
    f"End your run by calling submit_note exactly once, with the stock's symbol and your note. {_SUBMIT_RULE}"
)

BULL_SYSTEM = (
    "You are the bull. Argue for buying the strongest stocks. Do not trade.\n\n"
    f"{_FACT_SHEET}\n"
    "Each stock also has the researcher's note of dated facts. For every stock in stocks, rate your conviction that it "
    "is worth owning now (low, medium or high) and give one argument grounded in its fact sheet or its note. Be "
    "selective: high is for the stocks with the strongest and most durable case, not for every stock with a good trend.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_bull_case exactly once, with one object per stock: the symbol, the conviction and "
    f"one short argument. Give a case for every stock in stocks and no other. {_SUBMIT_RULE}"
)

BEAR_SYSTEM = (
    "You are the bear. Argue the biggest risk in each stock. Do not trade.\n\n"
    f"{_FACT_SHEET}\n"
    "Each stock also has the researcher's note of dated facts. For every stock in stocks, rate its risk (low, medium or "
    "high), name the concern behind it and give one argument grounded in its fact sheet or its note. The concern is one "
    "of valuation, momentum_exhaustion, earnings, fundamentals, news_event, sector or none. A strong trend is not a risk "
    "by itself: use momentum_exhaustion only when the fact sheet shows a stretched move (for example a one-month return "
    "far above its three-month pace, or a price far above its 200-day average). Use none with low when you find no real "
    "risk. Rate each stock on its own facts; do not give every stock the same rating.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_bear_case exactly once, with one object per stock: the symbol, the risk, the concern "
    f"and one short argument. Give a case for every stock in stocks and no other. {_SUBMIT_RULE}"
)

JUDGE_SYSTEM = (
    "You are the judge. Weigh the debate between the bull and the bear and pick the stocks that win it. Do not trade: "
    "code sizes the positions and places the orders.\n\n"
    f"{_FACT_SHEET}\n"
    "Each stock has the researcher's note, the bull's conviction and argument, and the bear's risk, concern and argument. "
    "A stock wins when the bull's case is stronger than the bear's risk. Pick between the minimum and maximum number of "
    "stocks given in the constraints, best first. The stocks in held are owned now: keep one unless the bear's case beats "
    "the bull's, since every change costs fees; for each held stock you do not pick, give a drop with its reason. Stocks "
    "in forced_exits are sold by code and are not in the debate.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_picks exactly once: picks is a list of objects with the symbol and one short reason, "
    f"best first; drops lists each held stock you did not pick, with the symbol and one short reason. {_SUBMIT_RULE}"
)

UNAVAILABLE_NOTE = "research unavailable"


def researcher_task(symbol: str, max_chars: int) -> str:
    return f"Research {symbol}: its fact sheet is in the context. Then submit your note on {symbol} (at most {max_chars} characters). The current datetime is in the context."


BULL_TASK = "Make the bull case for every stock in the context and submit it. The current datetime is in the context."

BEAR_TASK = "Make the bear case for every stock in the context and submit it. The current datetime is in the context."

JUDGE_TASK = "Judge the debate over the stocks in the context and submit your picks and drops. The current datetime is in the context."


def retry_prompt(tool: str, error: str) -> str:
    """The corrective turn after a run that never made a valid submit call."""
    return f"Your previous run ended without a valid {tool} call. The last problem was: {error}\nCall {tool} now, once, with a valid argument, then stop. Do not do any more research."
```

- [ ] **Step 3: Run the prompt tests**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -v`
Expected: PASS.

- [ ] **Step 4: Write the failing pipeline tests**

`tests/strategies/bull_bear/test_bull_bear_pipeline.py`:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bull_bear.market_data import DailySeries, DataUnavailable
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewOutcome, ReviewPipeline
from trading_agent_framework.strategies.bull_bear.prompts import UNAVAILABLE_NOTE
from trading_agent_framework.strategies.bull_bear.state import BullBearState, ReviewLog, StateStore
from trading_agent_framework.strategies.common.rebalancer import PlacedOrder, Rebalancer
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.errors import AgentError, BrokerError, ConfigurationError

UNIVERSE = [f"S{i:02d}" for i in range(20)]  # S00 has the strongest momentum, S19 the weakest

Script = Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]


def _series(growth: float, sessions: int = 300) -> DailySeries:
    """A trend with a +-1% zigzag, so the volatility is above zero; 50 x 1e6 shares clears the dollar-volume filter."""
    closes = [50.0 * (1 + growth) ** i * (1.01 if i % 2 else 0.99) for i in range(sessions)]
    return DailySeries(closes=closes, volumes=[1e6] * sessions)


def _default_series() -> dict[str, DailySeries]:
    return {symbol: _series(0.004 - index * 0.0001) for index, symbol in enumerate(UNIVERSE)}


# --- fakes -------------------------------------------------------------------------------------------


class FakeBars:
    def __init__(self, series: dict[str, DailySeries]) -> None:
        self.series = series
        self.error: DataUnavailable | None = None
        self.calls: list[tuple[list[str], list[str]]] = []

    def load(self, universe, held):  # noqa: ANN001, ANN201
        self.calls.append((list(universe), sorted(held)))
        if self.error is not None:
            raise self.error
        return {symbol: series for symbol, series in self.series.items() if symbol in universe}


class FakeAgent:
    """Runs `script` on every call, or the next of `overrides` first; a script calls the real submit tools."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self.overrides: list[Script] = []
        self.calls: list[dict[str, Any]] = []
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {}

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool, "tool_budget": tool_budget})
        step = self.overrides.pop(0) if self.overrides else self.script
        step(self.tools, context)
        return AgentRunResult(output="done", tool_calls=[])


def notes(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_note"](ctx["fact_sheet"]["symbol"], "2026-09-30: quarterly revenue up 12% y/y")


def bull_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bull_case"]([{"symbol": s["fact_sheet"]["symbol"], "conviction": "high", "argument": "strong trend"} for s in ctx["stocks"]])


def bear_cases(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_bear_case"]([{"symbol": s["fact_sheet"]["symbol"], "risk": "low", "concern": "none", "argument": "no red flag"} for s in ctx["stocks"]])


def picks_first(count: int) -> Script:
    def step(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        picked = [s["fact_sheet"]["symbol"] for s in ctx["stocks"]][:count]
        drops = [{"symbol": symbol, "reason": "lost the debate"} for symbol in ctx["held"] if symbol not in picked]
        tools["submit_picks"]([{"symbol": symbol, "reason": "won the debate"} for symbol in picked], drops)

    return step


def does_nothing(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    return None


@dataclass
class Harness:
    pipeline: ReviewPipeline
    broker: FakeBroker
    bars: FakeBars
    agents: dict[str, FakeAgent]
    rebalancer: Rebalancer
    store: StateStore
    log_path: Path
    orders: list[tuple[str, str, float]] = field(default_factory=list)

    def run(self) -> ReviewOutcome:
        self.broker.submitted.clear()
        outcome = self.pipeline.run()
        self.orders = [(order.asset.symbol, order.side.value, float(order.quantity)) for order in self.broker.submitted]
        return outcome

    def log_lines(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()] if self.log_path.exists() else []


def _harness(tmp_path: Path, *, held: dict[str, float] | None = None, series: dict[str, DailySeries] | None = None, params: BullBearParams | None = None, state: BullBearState | None = None) -> Harness:
    """Tuesday 2026-10-06 12:00, portfolio value 10,000; every stock costs 50, SHV 100; `held` maps a symbol to shares."""
    params = params or BullBearParams()
    held = held or {}
    broker = FakeBroker(FakeClock(et(2026, 10, 6, 12)), strategy_name="bull_bear")
    broker.last_prices = {symbol: Decimal(50) for symbol in [*UNIVERSE, "OLD"]} | {"SHV": Decimal(100)}
    broker.positions = [Position(strategy_name="bull_bear", asset=Asset(symbol), quantity=Decimal(str(shares)), side=PositionSide.LONG) for symbol, shares in held.items()]
    invested = sum(shares * 50 for shares in held.values())
    broker.account = AccountBalances(cash=Decimal(str(10_000 - invested)), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    store = StateStore(tmp_path / "data" / "state.json")
    if state is not None:
        store.save(state)
    recorder = HandoffRecorder(params)
    agents = {"researcher": FakeAgent(notes), "bull": FakeAgent(bull_cases), "bear": FakeAgent(bear_cases), "judge": FakeAgent(picks_first(5))}
    for agent in agents.values():
        agent.tools = submit_tools(recorder)
    bars = FakeBars(series if series is not None else _default_series())
    rebalancer = Rebalancer(strategy, params)
    log_path = tmp_path / "logs" / "reviews.jsonl"
    pipeline = ReviewPipeline(
        strategy=strategy,
        params=params,
        agents=agents,
        recorder=recorder,
        state=store,
        review_log=ReviewLog(log_path),
        rebalancer=rebalancer,
        universe=UNIVERSE,
        bars=bars,
        sector_of=lambda symbol: "Technology",
        momentum=CONFIG,
    )
    return Harness(pipeline, broker, bars, agents, rebalancer, store, log_path)


# --- the happy path ---------------------------------------------------------------------------------


def test_a_review_debates_the_top_15_and_buys_the_judges_picks(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=True, abandoned_streak=0)
    research = h.agents["researcher"].calls
    assert [call["context"]["fact_sheet"]["symbol"] for call in research] == UNIVERSE[:15]
    assert all(call["tool_budget"] == 3 for call in research)
    bull_context = h.agents["bull"].calls[0]["context"]
    assert [stock["fact_sheet"]["symbol"] for stock in bull_context["stocks"]] == UNIVERSE[:15]
    assert bull_context["stocks"][0]["note"] == "2026-09-30: quarterly revenue up 12% y/y"
    assert h.agents["bear"].calls[0]["context"] == bull_context  # the same evidence, and not the bull's case
    assert [order[:2] for order in h.orders] == [(symbol, "buy") for symbol in UNIVERSE[:5]]  # in the judge's order
    state = h.store.load()
    assert state.last_completed_review == "2026-10-06" and state.abandoned_streak == 0
    assert [pick["symbol"] for pick in state.last_picks] == UNIVERSE[:5]


def test_the_review_log_records_the_whole_review_as_one_line(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    h.run()

    (line,) = h.log_lines()
    assert line["date"] == "2026-10-06" and line["abandoned"] is False
    assert [stock["symbol"] for stock in line["debate_set"]] == UNIVERSE[:15]
    assert line["debate_set"][0] == {"symbol": "S00", "rank": 1, "held": False}
    assert line["forced_exits"] == []
    assert line["notes"] == {"written": 15, "failed": []}
    assert line["bull_conviction"] == {"high": 15} and line["bear_risk"] == {"low": 15}
    assert [pick["symbol"] for pick in line["picks"]] == UNIVERSE[:5] and line["drops"] == []
    stock_weights = {symbol: weight for symbol, weight in line["targets"].items() if symbol != "SHV"}
    assert sum(stock_weights.values()) == pytest.approx(0.98)
    assert all(0.04 <= weight <= 0.20 for weight in stock_weights.values())
    assert [order["symbol"] for order in line["orders"]] == UNIVERSE[:5]


def test_holdings_in_the_retention_band_are_debated_and_the_others_are_forced_out(tmp_path: Path) -> None:
    h = _harness(tmp_path, held={"S16": 20, "S19": 20, "OLD": 20}, params=BullBearParams(retention_rank=18))

    outcome = h.run()

    assert outcome.completed
    (line,) = h.log_lines()
    assert [stock["symbol"] for stock in line["debate_set"]] == [*UNIVERSE[:15], "S16"]
    assert line["debate_set"][-1] == {"symbol": "S16", "rank": 17, "held": True}
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}, {"symbol": "S19", "reason": "rank 20"}]
    assert h.agents["judge"].calls[0]["context"]["held"] == ["S16"]
    assert line["drops"] == [{"symbol": "S16", "reason": "lost the debate"}]
    assert {("OLD", "sell", 20.0), ("S19", "sell", 20.0), ("S16", "sell", 20.0)} <= set(h.orders)
    assert h.bars.calls == [(UNIVERSE, ["OLD", "S16", "S19"])]


# --- research failures --------------------------------------------------------------------------------


def test_a_stock_whose_research_fails_gets_research_unavailable_and_the_review_goes_on(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    def flaky(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        if ctx["fact_sheet"]["symbol"] == "S03":
            raise AgentError("timeout")
        notes(tools, ctx)

    h.agents["researcher"].script = flaky

    outcome = h.run()

    assert outcome.completed
    assert len(h.agents["researcher"].calls) == 16  # S03 once more with a forced submit_note
    stock = next(s for s in h.agents["bull"].calls[0]["context"]["stocks"] if s["fact_sheet"]["symbol"] == "S03")
    assert stock["note"] == UNAVAILABLE_NOTE
    assert h.log_lines()[0]["notes"] == {"written": 14, "failed": ["S03"]}


def test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state(tmp_path: Path) -> None:
    h = _harness(tmp_path)

    def misconfigured(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
        raise ConfigurationError("SEC_EDGAR_USER_AGENT is not set")

    h.agents["researcher"].script = misconfigured

    with pytest.raises(ConfigurationError):
        h.run()
    assert h.store.load() == BullBearState() and h.orders == []


# --- abandoned reviews --------------------------------------------------------------------------------


@pytest.mark.parametrize(("agent", "tool"), [("bull", "submit_bull_case"), ("bear", "submit_bear_case"), ("judge", "submit_picks")])
def test_a_debater_without_a_valid_submission_abandons_the_review_with_no_order(tmp_path: Path, agent: str, tool: str) -> None:
    h = _harness(tmp_path, held={"OLD": 20})
    h.agents[agent].script = does_nothing

    outcome = h.run()

    assert outcome == ReviewOutcome(completed=False, abandoned_streak=1)
    assert h.orders == []  # not even the forced exit of OLD
    assert len(h.agents[agent].calls) == 2 and h.agents[agent].calls[1]["force_tool"] == tool
    state = h.store.load()
    assert (state.last_completed_review, state.abandoned_streak) == (None, 1)
    (line,) = h.log_lines()
    assert line["abandoned"] is True and line["stage"] == agent and tool in line["error"]
    assert line["forced_exits"] == [{"symbol": "OLD", "reason": "unranked"}]  # what was known before the stage failed


def test_a_corrected_submission_on_the_forced_retry_completes_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["bear"].overrides = [lambda tools, ctx: tools["submit_bear_case"]([{"symbol": "S00", "risk": "high", "concern": "valuation", "argument": "too expensive"}])]

    outcome = h.run()

    assert outcome.completed
    retry = h.agents["bear"].calls[1]
    assert retry["force_tool"] == "submit_bear_case" and "no case for S01" in retry["task"]


def test_unusable_data_abandons_the_review_before_any_agent_runs(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.bars.error = DataUnavailable("Yahoo covers only 3/20 symbols")

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == [] and h.orders == []
    (line,) = h.log_lines()
    assert (line["stage"], line["error"]) == ("data", "Yahoo covers only 3/20 symbols")


def test_a_debate_set_smaller_than_min_picks_abandons_the_review(tmp_path: Path) -> None:
    h = _harness(tmp_path, series={symbol: series for symbol, series in _default_series().items() if symbol in UNIVERSE[:4]})

    outcome = h.run()

    assert not outcome.completed and h.agents["researcher"].calls == []
    (line,) = h.log_lines()
    assert line["stage"] == "data" and "fewer than min_picks (5)" in line["error"]


def test_a_broker_error_before_the_rebalance_abandons_at_the_broker_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)

    def broken() -> list[str]:
        raise BrokerError("positions unavailable")

    monkeypatch.setattr(h.rebalancer, "holdings", broken)

    outcome = h.run()

    assert not outcome.completed and h.log_lines()[0]["stage"] == "broker"


def test_a_broker_error_during_the_rebalance_logs_the_orders_already_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)

    def half_way(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        h.rebalancer.placed = [PlacedOrder("S00", "buy", 10.0)]
        raise BrokerError("connection reset")

    monkeypatch.setattr(h.rebalancer, "rebalance", half_way)

    outcome = h.run()

    assert not outcome.completed
    (line,) = h.log_lines()
    assert line["stage"] == "execution" and line["orders"] == [{"symbol": "S00", "side": "buy", "quantity": 10.0}]


# --- the streak and the same-day rule -------------------------------------------------------------------


def test_a_completed_review_resets_the_streak_and_marks_the_day_done(tmp_path: Path) -> None:
    h = _harness(tmp_path, state=BullBearState(abandoned_streak=2))

    assert not h.pipeline.completed_today()
    h.run()

    assert h.store.load().abandoned_streak == 0 and h.pipeline.completed_today()


def test_an_abandoned_review_leaves_the_day_open_so_a_restart_reviews_again(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["judge"].overrides = [does_nothing, does_nothing]

    assert not h.run().completed
    assert not h.pipeline.completed_today()

    assert h.run() == ReviewOutcome(completed=True, abandoned_streak=0)
    assert len(h.agents["judge"].calls) == 3
```

- [ ] **Step 5: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: ... bull_bear.pipeline`.

- [ ] **Step 6: Write `pipeline.py`**

```python
"""One review: data → debate set → researcher (one run per stock) → bull → bear → judge → sizing → rebalance.

The agents decide which stocks win; this module validates (through the `HandoffRecorder`), sizes, remembers
(`StateStore`) and calls the `Rebalancer`. A debater stage that never yields a valid submission abandons the review:
no order at all, forced exits included. One stock's failed research only gives it the note "research unavailable".
Free text from an agent is logged and otherwise ignored. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md.
"""

from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.strategies.bull_bear.debate_set import build_debate_set
from trading_agent_framework.strategies.bull_bear.fact_sheet import fact_sheet_row
from trading_agent_framework.strategies.bull_bear.handoff import BearCase, BullCase, HandoffRecorder, Note, Picks
from trading_agent_framework.strategies.bull_bear.market_data import DailyBars, DataUnavailable
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.prompts import BEAR_TASK, BULL_TASK, JUDGE_TASK, UNAVAILABLE_NOTE, researcher_task, retry_prompt
from trading_agent_framework.strategies.bull_bear.sizing import capped_inverse_volatility
from trading_agent_framework.strategies.bull_bear.state import BullBearState, ReviewLog, StateStore
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.scoring import MomentumRow, rank, score_stock
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class AgentLike(Protocol):
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...


class AgentLookup(Protocol):
    """What the pipeline needs from the agents: one by name (`AgentManager`, or a plain dict in tests)."""

    def __getitem__(self, name: str, /) -> AgentLike: ...


class ReviewAbandoned(Exception):
    """A stage could not produce what the review needs; the review ends with nothing traded."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    completed: bool
    abandoned_streak: int  # abandoned reviews in a row after this one (0 after a completed review)


class ReviewPipeline:
    def __init__(
        self,
        *,
        strategy: Strategy,
        params: BullBearParams,
        agents: AgentLookup,
        recorder: HandoffRecorder,
        state: StateStore,
        review_log: ReviewLog,
        rebalancer: Rebalancer,
        universe: Sequence[str],
        bars: DailyBars,
        sector_of: Callable[[str], str],
        momentum: Mapping[str, Any],
    ) -> None:
        self._strategy = strategy
        self._params = params
        self._agents = agents
        self._recorder = recorder
        self._state = state
        self._log = review_log
        self._rebalancer = rebalancer
        self._universe = list(universe)
        self._bars = bars
        self._sector_of = sector_of
        self._momentum = momentum  # cross_momentum's CONFIG: the score weights, the skip and the filters
        self._stage = "setup"  # "setup" until the rebalancer starts, then "execution": what a broker failure is attributed to

    def _now(self) -> datetime:
        return self._strategy.clock.now().astimezone(MARKET_TZ)

    def completed_today(self) -> bool:
        """Whether today's review (market date) already completed: a restart then does not run it again."""
        return self._state.load().last_completed_review == self._now().date().isoformat()

    # --- one review --------------------------------------------------------------------------------

    def run(self) -> ReviewOutcome:
        state = self._state.load()
        now = self._now()
        record: dict[str, Any] = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
        self._stage = "setup"
        extra: dict[str, Any] = {}
        try:
            self._review(now, record)
        except ReviewAbandoned as exc:
            stage, error = exc.stage, exc
        except (BrokerError, BacktestError) as exc:
            # A broker or data failure anywhere in the review (ConfigurationError is not one: it propagates).
            stage, error = ("execution" if self._stage == "execution" else "broker"), exc
            if stage == "execution":
                extra["orders"] = [asdict(order) for order in self._rebalancer.placed]  # sent before the failure
        else:
            return ReviewOutcome(completed=True, abandoned_streak=0)
        streak = state.abandoned_streak + 1
        self._strategy.log_error(f"[bull_bear] review abandoned at the {stage} stage (streak {streak}): {error}")
        self._state.save(replace(state, abandoned_streak=streak))
        self._log.append({**record, "abandoned": True, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
        return ReviewOutcome(completed=False, abandoned_streak=streak)

    def _review(self, now: datetime, record: dict[str, Any]) -> None:
        params, strategy = self._params, self._strategy
        today = now.date().isoformat()
        moment = now.isoformat()

        # 1. Data and ranking.
        holdings = self._rebalancer.holdings()
        try:
            series = self._bars.load(self._universe, holdings)
        except DataUnavailable as exc:
            raise ReviewAbandoned("data", str(exc)) from exc
        rows: list[MomentumRow] = []
        for symbol in self._universe:
            if symbol in series:
                row = score_stock(symbol, series[symbol].closes, series[symbol].volumes, self._momentum)
                if row is not None:
                    rows.append(row)
        ranked = rank(rows)

        # 2. The debate set, and the holdings code sells without asking the agents.
        debate = build_debate_set(ranked, holdings, shortlist_size=params.shortlist_size, retention_rank=params.retention_rank)
        symbols = debate.symbols
        record.update(
            ranked=len(ranked),
            debate_set=[{"symbol": stock.symbol, "rank": stock.rank, "held": stock.held} for stock in debate.stocks],
            forced_exits=[asdict(exit) for exit in debate.forced_exits],
        )
        strategy.log_info(
            f"[bull_bear] {len(ranked)} stocks ranked; debate: {', '.join(f'{s.symbol} (#{s.rank}{", held" if s.held else ""})' for s in debate.stocks)}; "
            f"forced exits: {', '.join(f'{e.symbol} ({e.reason})' for e in debate.forced_exits) or 'none'}"
        )
        if len(symbols) < params.min_picks:
            raise ReviewAbandoned("data", f"only {len(symbols)} stocks in the debate set, fewer than min_picks ({params.min_picks})")

        # 3. Fact sheet.
        weights = self._rebalancer.current_weights()
        sheets = {stock.symbol: fact_sheet_row(stock, sector=self._sector_of(stock.symbol), weight=weights.get(stock.symbol, 0.0)) for stock in debate.stocks}

        # 4. Researcher: one short run per stock, so one failure costs one note, not the review.
        notes: dict[str, str] = {}
        for symbol in symbols:
            self._recorder.expect_note(symbol)
            try:
                self._run_stage("researcher", "submit_note", researcher_task(symbol, params.note_max_chars), {"current_datetime": moment, "fact_sheet": sheets[symbol]}, tool_budget=params.research_tool_budget)
            except ReviewAbandoned as exc:
                strategy.log_warning(f"[researcher] no note for {symbol}: {exc}")
                notes[symbol] = UNAVAILABLE_NOTE
                continue
            note: Note = self._recorder.submission
            notes[symbol] = note.note
        failed = [symbol for symbol in symbols if notes[symbol] == UNAVAILABLE_NOTE]
        record["notes"] = {"written": len(symbols) - len(failed), "failed": failed}
        evidence = [{"fact_sheet": sheets[symbol], "note": notes[symbol]} for symbol in symbols]

        # 5. Bull, then bear: the same evidence; neither sees the other's case.
        self._recorder.expect_bull(symbols)
        self._run_stage("bull", "submit_bull_case", BULL_TASK, {"current_datetime": moment, "stocks": evidence})
        bull: dict[str, BullCase] = {case.symbol: case for case in self._recorder.submission}
        self._recorder.expect_bear(symbols)
        self._run_stage("bear", "submit_bear_case", BEAR_TASK, {"current_datetime": moment, "stocks": evidence})
        bear: dict[str, BearCase] = {case.symbol: case for case in self._recorder.submission}
        record.update(bull_conviction=dict(Counter(case.conviction for case in bull.values())), bear_risk=dict(Counter(case.risk for case in bear.values())))
        for symbol in symbols:
            strategy.log_info(f"[debate]   {symbol}: bull {bull[symbol].conviction} ({bull[symbol].argument}); bear {bear[symbol].risk}/{bear[symbol].concern} ({bear[symbol].argument})")

        # 6. Judge.
        held = debate.held
        self._recorder.expect_picks(symbols, held=held)
        context = {
            "current_datetime": moment,
            "stocks": [
                {
                    "fact_sheet": sheets[symbol],
                    "note": notes[symbol],
                    "bull": {"conviction": bull[symbol].conviction, "argument": bull[symbol].argument},
                    "bear": {"risk": bear[symbol].risk, "concern": bear[symbol].concern, "argument": bear[symbol].argument},
                }
                for symbol in symbols
            ],
            "held": held,
            "forced_exits": [asdict(exit) for exit in debate.forced_exits],
            "constraints": {"min_picks": params.min_picks, "max_picks": params.max_picks},
        }
        self._run_stage("judge", "submit_picks", JUDGE_TASK, context)
        picks: Picks = self._recorder.submission
        strategy.log_info(f"[judge] picks: {', '.join(f'{p.symbol} ({p.reason})' for p in picks.picks)}; drops: {', '.join(f'{d.symbol} ({d.reason})' for d in picks.drops) or 'none'}")

        # 7. Sizing and execution.
        volatilities = {pick.symbol: debate.stock(pick.symbol).row.volatility for pick in picks.picks}
        target_weights = capped_inverse_volatility(volatilities, total=params.investable, min_weight=params.min_weight, max_weight=params.max_weight)
        target = target_portfolio(target_weights, cash_buffer=params.cash_buffer)
        self._stage = "execution"
        orders = self._rebalancer.rebalance(target, [exit.symbol for exit in debate.forced_exits])

        # 8. Remember and log.
        self._state.save(BullBearState(last_completed_review=today, abandoned_streak=0, last_picks=[{"symbol": pick.symbol, "reason": pick.reason, "date": today} for pick in picks.picks]))
        self._log.append(
            {
                **record,
                "bull": [asdict(case) for case in bull.values()],
                "bear": [asdict(case) for case in bear.values()],
                "picks": [asdict(pick) for pick in picks.picks],
                "drops": [asdict(drop) for drop in picks.drops],
                "targets": {**target.weights, params.parking_symbol: target.parking_weight},
                "orders": [asdict(order) for order in orders],
            }
        )

    # --- helpers -----------------------------------------------------------------------------------

    def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any], tool_budget: int | None = None) -> None:
        """Run an agent until it makes a valid `tool` call: once, then once more with a forced call; else abandon."""
        agent = self._agents[agent_name]
        run_id = uuid.uuid4().hex
        prompt, force_tool, error = task, None, ""
        for _ in range(2):
            self._recorder.last_error, error = None, ""  # an error from the first attempt must not be reported for the second
            try:
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool, tool_budget=tool_budget)
                self._strategy.log_info(f"[{agent_name}] {result.output}")
                for i, tool_call in enumerate(result.tool_calls):
                    self._strategy.log_debug(f"[{agent_name}] tool_call_{i}: {tool_call}")
            except AgentError as exc:
                error = str(exc)
                self._strategy.log_error(f"[{agent_name}] run failed: {exc}")
            if self._recorder.submitted:
                return
            error = self._recorder.last_error or error or f"the agent ended without calling {tool}"
            prompt, force_tool = retry_prompt(tool, error), tool
        raise ReviewAbandoned(agent_name, error)
```

Note: the f-string in the `log_info` of step 2 nests double quotes inside a single-quoted f-string, which is valid on Python 3.12+ (the repo is on 3.14).

- [ ] **Step 7: Run the pipeline tests**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_pipeline.py -v`
Expected: PASS. If `test_a_corrected_submission_on_the_forced_retry_completes_the_review` fails on `"no case for S01"`, print `retry["task"]` and compare with the `_covering` message: the retry prompt quotes `recorder.last_error`.

- [ ] **Step 8: Linters**

Run: `uv run ruff check && uv run pyright src/trading_agent_framework/strategies/bull_bear`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "feat(bull_bear): prompts and the review pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `BullBearStrategy`, registration and an end-to-end backtest

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py`
- Create: `src/trading_agent_framework/strategies/bull_bear/__init__.py`
- Modify: `src/trading_agent_framework/utils/strategy_factory.py` (enum, builder, map)
- Test: `tests/strategies/bull_bear/test_bull_bear_strategy.py`, `tests/strategies/bull_bear/test_bull_bear_backtest.py`

**Interfaces:**
- Consumes: `ReviewPipeline`, `ReviewOutcome`, `YahooBars`, `GateBars`, `HandoffRecorder`, `submit_tools`, the four system prompts, `StateStore`, `ReviewLog`, `state_path`, `Rebalancer`, `SectorProvider`, `YahooDailyBars`, `parse_rebalance_time`, cross_momentum `CONFIG`, `agents.tools.only`, `news_tools`, `fundamentals_tools`.
- Produces:
  - `BullBearStrategy(broker, *, mode=TradingMode.PAPER, universe: Sequence[str], settings: BullBearParams | None = None, bars_source: BarsSource | None = None, sector_of: Callable[[str], str] | None = None, **kwargs)` with attributes `universe: list[str]`, `settings`, `pipeline: ReviewPipeline | None`
  - `Strategies.BULL_BEAR` (value `"bull_bear"`)

- [ ] **Step 1: Write the failing strategy tests**

`tests/strategies/bull_bear/test_bull_bear_strategy.py`:

```python
from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.bull_bear import BullBearStrategy
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewOutcome
from trading_agent_framework.strategies.bull_bear.state import BullBearState, StateStore, state_path
from trading_agent_framework.utils.errors import FatalStrategyError
from trading_agent_framework.utils.strategy_factory import Strategies

UNIVERSE = ["AAA", "BBB", "CCC", "SHV"]
TUESDAY, MONDAY = et(2026, 10, 6, 12), et(2026, 10, 5, 12)


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()

    def __getitem__(self, name: str) -> object:
        return object()


class _NoBars:
    def bars(self, symbols, today):  # noqa: ANN001, ANN201
        raise AssertionError("not called in these tests")


class _FakePipeline:
    def __init__(self, outcomes: list[ReviewOutcome], *, completed: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.completed = completed
        self.runs = 0

    def completed_today(self) -> bool:
        return self.completed

    def run(self) -> ReviewOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, now: datetime = TUESDAY, **kwargs: Any) -> tuple[BullBearStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(now), strategy_name="bull_bear")
    strategy = BullBearStrategy(broker, mode=mode, universe=UNIVERSE, project_root=tmp_path, bars_source=_NoBars(), sector_of=lambda symbol: "Technology", **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


def test_it_is_registered_as_bull_bear() -> None:
    assert Strategies("bull_bear") is Strategies.BULL_BEAR


def test_it_iterates_every_session_at_noon_and_never_scores_the_parking_symbol(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)

    assert strategy.sleeptime == "1D" and strategy.iteration_start_time == time(12, 0)
    assert strategy.universe == ["AAA", "BBB", "CCC"]


def test_initialize_creates_the_four_agents_with_only_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    tools = {created["name"]: _tool_names(created) for created in agents.created}
    assert tools == {
        "researcher": {"search_news", "get_income_statement", "get_balance_sheet", "submit_note"},
        "bull": {"submit_bull_case"},
        "bear": {"submit_bear_case"},
        "judge": {"submit_picks"},
    }


def test_only_the_researcher_has_a_budget_exempt_tool_and_every_agent_uses_the_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(agent_temperature=0.1))

    strategy.initialize()

    assert {created["name"]: created.get("exempt_tools") for created in agents.created} == {"researcher": ["submit_note"], "bull": None, "bear": None, "judge": None}
    assert {created["temperature"] for created in agents.created} == {0.1}


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, expected in [(TradingMode.BACKTESTING, BullBearState()), (TradingMode.PAPER, BullBearState(abandoned_streak=2))]:
        store = StateStore(state_path(tmp_path, mode))
        store.save(BullBearState(abandoned_streak=2))
        strategy, _ = _strategy(tmp_path, mode=mode)

        strategy.initialize()

        assert store.load() == expected


def test_a_missing_sec_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy, _ = _strategy(tmp_path)

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_the_review_runs_on_tuesdays_only(tmp_path: Path) -> None:
    monday, _ = _strategy(tmp_path, now=MONDAY)
    monday.pipeline = _FakePipeline([ReviewOutcome(True, 0)])  # type: ignore[assignment]
    tuesday, _ = _strategy(tmp_path, now=TUESDAY)
    tuesday.pipeline = _FakePipeline([ReviewOutcome(True, 0)])  # type: ignore[assignment]

    monday.on_trading_iteration()
    tuesday.on_trading_iteration()

    assert (monday.pipeline.runs, tuesday.pipeline.runs) == (0, 1)  # type: ignore[union-attr]


def test_a_review_already_completed_today_is_not_run_again(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.pipeline = _FakePipeline([], completed=True)  # type: ignore[assignment]

    strategy.on_trading_iteration()

    assert strategy.pipeline.runs == 0  # type: ignore[union-attr]


def test_a_backtest_aborts_after_three_abandoned_reviews_in_a_row(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 3)])  # type: ignore[assignment]

    with pytest.raises(FatalStrategyError, match="3 reviews abandoned in a row"):
        strategy.on_trading_iteration()


def test_paper_never_aborts_on_abandoned_reviews(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 5)])  # type: ignore[assignment]

    strategy.on_trading_iteration()  # logs and carries on


def test_run_backtesting_passes_the_class_window_the_daily_yahoo_source_and_the_preloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["start"] == BullBearStrategy.parameters["backtesting_start"]
    assert captured["data_source"] is YahooBacktestData and captured["timestep"] == "day"
    assert {asset.symbol for asset in captured["preload_assets"]} == {"AAA", "BBB", "CCC", "SHV", "SPY"}
    assert captured["budget"] == Decimal(10000) and captured["warmup_trading_days"] == 300
```

- [ ] **Step 2: Write the failing backtest test**

`tests/strategies/bull_bear/test_bull_bear_backtest.py`:

```python
"""End to end: a backtest over two Tuesdays with fake agents (they call the real submit tools) and fake daily bars."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bull_bear import BullBearStrategy

PRIOR = weekday_sessions(date(2025, 7, 21), 300)  # history before the window: 300 completed sessions to score on
SESSIONS = weekday_sessions(date(2026, 9, 14), 7)  # Monday 14th to Tuesday 22nd: two Tuesdays
UNIVERSE = [f"S{i}" for i in range(6)]


def _bars(growth: float, base: float = 50.0) -> pd.DataFrame:
    sessions = [*PRIOR, *SESSIONS]
    closes = [base * (1 + growth) ** i * (1.01 if i % 2 else 0.99) for i in range(len(sessions))]
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.005 for c in closes], "low": [c * 0.995 for c in closes], "close": closes, "volume": [1e6] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in sessions], name="timestamp"),
    )


class _Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _research(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_note"](context["fact_sheet"]["symbol"], "no news found")


def _bull(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_bull_case"]([{"symbol": s["fact_sheet"]["symbol"], "conviction": "high", "argument": "trend"} for s in context["stocks"]])


def _bear(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_bear_case"]([{"symbol": s["fact_sheet"]["symbol"], "risk": "low", "concern": "none", "argument": "none"} for s in context["stocks"]])


def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    picked = [s["fact_sheet"]["symbol"] for s in context["stocks"]][:5]
    tools["submit_picks"]([{"symbol": s, "reason": "won"} for s in picked], [{"symbol": s, "reason": "lost"} for s in context["held"] if s not in picked])


class _Manager:
    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": _research, "bull": _bull, "bear": _bear, "judge": _judge}

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, manager: _Manager) -> BullBearStrategy:
    source = FakeBacktestDataSource()
    source.set_sessions(SESSIONS)
    for index, symbol in enumerate(UNIVERSE):
        source.set_bars(Asset(symbol), _bars(0.003 - index * 0.0002))
    source.set_bars(Asset("SHV"), _bars(0.0, base=100.0))
    source.set_bars(Asset("SPY"), _bars(0.001, base=400.0))
    strategy = BullBearStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="bull_bear"),
        mode=TradingMode.BACKTESTING,
        universe=UNIVERSE,
        project_root=tmp_path,
        sector_of=lambda symbol: "Technology",
    )
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=source, budget=Decimal(10000), benchmark="SPY")
    return strategy


def _review_lines(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("reviews.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_backtest_reviews_on_the_two_tuesdays_and_buys_the_judges_picks(tmp_path: Path) -> None:
    manager = _Manager()

    _run(tmp_path, manager)

    lines = _review_lines(tmp_path)
    assert [line["date"] for line in lines] == ["2026-09-15", "2026-09-22"]
    assert not any(line["abandoned"] for line in lines)
    assert [stock["symbol"] for stock in lines[0]["debate_set"]] == UNIVERSE  # ranked by momentum, S0 first
    assert {order["symbol"] for order in lines[0]["orders"]} == set(UNIVERSE[:5])
    assert manager.handles["researcher"].runs == 2 * len(UNIVERSE)
    assert all(manager.handles[name].runs == 2 for name in ("bull", "bear", "judge"))
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_strategy.py tests/strategies/bull_bear/test_bull_bear_backtest.py -v`
Expected: FAIL with `ImportError: cannot import name 'BullBearStrategy'`.

- [ ] **Step 4: Write `agent_bull_bear.py`**

```python
"""BullBearStrategy: a researcher, a bull, a bear and a judge debate the top of cross_momentum's ranking.

Port of lumibot's "Bull vs Bear AI Stock Trading Bot", hardened for a local model: code ranks the universe with
cross_momentum's own score and filters, builds the fact sheets, sizes the judge's picks by inverse volatility and
places every order (`ReviewPipeline`, `Rebalancer`). No agent has an order tool. One review a week, on Tuesday at
12:00 ET, as cross_momentum. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.agents.tools import only
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bull_bear.market_data import BarsSource, DailyBars, GateBars, YahooBars
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewPipeline
from trading_agent_framework.strategies.bull_bear.prompts import BEAR_SYSTEM, BULL_SYSTEM, JUDGE_SYSTEM, RESEARCHER_SYSTEM
from trading_agent_framework.strategies.bull_bear.state import ReviewLog, StateStore, state_path
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.sector_provider import SectorProvider
from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.strategies.common.yahoo_daily_bars import YahooDailyBars
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError

_RESEARCH_TOOLS = {"search_news", "get_income_statement", "get_balance_sheet"}


class BullBearStrategy(Strategy):
    """One review a week (see `ReviewPipeline`); every other session's iteration returns at once."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.HALF_YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.HALF_YEAR)[1],
        "benchmark_symbol": "SPY",
        # 300 completed sessions before the first review: the 12-1 month return needs 274 (as cross_momentum)
        "warmup_trading_days": 300,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: BullBearParams | None = None,
        bars_source: BarsSource | None = None,
        sector_of: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or BullBearParams()
        self.iteration_start_time = parse_rebalance_time(self.settings.rebalance_time)
        self.universe = [symbol for symbol in universe if symbol != self.settings.parking_symbol]  # SHV is parking, never a stock
        # Paper/live read Yahoo's daily bars (Alpaca's IEX volume would empty the dollar-volume filter); a backtest reads the gate.
        if bars_source is None and self.trading_mode is not TradingMode.BACKTESTING:
            bars_source = YahooDailyBars()
        self._bars_source = bars_source
        self._sector_of = sector_of  # injected in tests; yfinance's sectors otherwise
        self.pipeline: ReviewPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the four agents and the pipeline (once per run)."""
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's streak and date must not leak into the next; paper and live state is never wiped
        try:
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools

            temperature = self.settings.agent_temperature
            research = only([*news_tools(self), *fundamentals_tools(self)], _RESEARCH_TOOLS)
            self.agents.create(name="researcher", system_prompt=RESEARCHER_SYSTEM, tools=[*research, submit["submit_note"]], temperature=temperature, exempt_tools=["submit_note"])
            # The bull, the bear and the judge argue from the same evidence pack: no research tool, only their submit tool.
            self.agents.create(name="bull", system_prompt=BULL_SYSTEM, tools=[submit["submit_bull_case"]], temperature=temperature)
            self.agents.create(name="bear", system_prompt=BEAR_SYSTEM, tools=[submit["submit_bear_case"]], temperature=temperature)
            self.agents.create(name="judge", system_prompt=JUDGE_SYSTEM, tools=[submit["submit_picks"]], temperature=temperature)
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every week (SEC_EDGAR_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = ReviewPipeline(
            strategy=self,
            params=self.settings,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            review_log=ReviewLog(self._review_log_path()),
            rebalancer=Rebalancer(self, self.settings),
            universe=self.universe,
            bars=self._daily_bars(),
            sector_of=self._sector_of or SectorProvider().get_sector,
            momentum=CONFIG,
        )
        self.log_info(f"BullBearStrategy initialized: {len(self.universe)} symbols, review on weekday {self.settings.rebalance_weekday} at {self.iteration_start_time}")

    def on_trading_iteration(self) -> None:
        """The weekly review. A backtest aborts when too many reviews in a row were abandoned (a dead LLM or data source)."""
        assert self.pipeline is not None, "initialize() has not run"
        today = self._market_date()
        if today.weekday() != self.settings.rebalance_weekday:
            return
        if self.pipeline.completed_today():
            self.log_info(f"[bull_bear] the review of {today} already completed: not run again")
            return
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    def _market_date(self) -> date:
        return self.clock.now().astimezone(MARKET_TZ).date()

    def _daily_bars(self) -> DailyBars:
        if self._bars_source is None:
            return GateBars(lambda symbol, length: self.get_historical_prices(symbol, length, "day"), today=self._market_date, warn=self.log_warning)
        return YahooBars(
            self._bars_source,
            today=self._market_date,
            sleep=self.sleep,
            warn=self.log_warning,
            retry_delays=self.settings.yahoo_retry_delays,
            min_coverage=self.settings.min_yahoo_coverage,
        )

    def _review_log_path(self) -> Path | None:
        """`reviews.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "reviews.jsonl"

    # --- backtesting ------------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window on Yahoo daily bars, the universe preloaded."""
        symbols = list(dict.fromkeys([*self.universe, self.settings.parking_symbol, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],  # one Yahoo download, not 1,200 per review
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})
```

`src/trading_agent_framework/strategies/bull_bear/__init__.py`:

```python
"""The bull vs bear strategy: a researcher, a bull, a bear and a judge debate the top of cross_momentum's ranking."""

from trading_agent_framework.strategies.bull_bear.agent_bull_bear import BullBearStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["BullBearStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")
```

- [ ] **Step 5: Register the strategy**

In `utils/strategy_factory.py`:
- add `from trading_agent_framework.strategies.bull_bear import BullBearStrategy` next to the other strategy imports;
- add `BULL_BEAR = auto()` to `Strategies` (alphabetically, after `BILL_ACKMAN`);
- add the builder after `_build_bill_ackman`:

```python
def _build_bull_bear(broker: Broker, mode: TradingMode) -> Strategy | None:
    return _build_with_universe(BullBearStrategy, broker, mode)
```

- add `Strategies.BULL_BEAR: _build_bull_bear,` to `_STRATEGY_BUILDERS`.

- [ ] **Step 6: Run the strategy and backtest tests**

Run: `uv run pytest tests/strategies/bull_bear tests/test_strategy_factory.py -v`
Expected: PASS. If the backtest test fails on `debate_set` order or the Tuesday dates, print `lines` and check that `weekday_sessions(date(2026, 9, 14), 7)` ends on Tuesday 22nd and that `PRIOR` ends before Monday 14th.

- [ ] **Step 7: Full suite and linters**

Run: `uv run pytest -q && uv run ruff check && uv run pyright`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear src/trading_agent_framework/utils/strategy_factory.py tests/strategies/bull_bear
git commit -m "feat(bull_bear): BullBearStrategy, registration and an end-to-end backtest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Documentation

**Files:**
- Modify: `CLAUDE.md` (Commands, Architecture: `strategies/` bullet, new `strategies/common/` bullet, cross_momentum and bill_ackman mentions of moved modules)
- Modify: `README.md` (env var paragraph at line ~75, new strategy section after `bill_ackman`'s at line ~286)

- [ ] **Step 1: Update `CLAUDE.md`**

1. In **Commands**, no new command is needed (`uv run agent <strategy_name> <mode>` already covers it).
2. Add a new Architecture bullet right after the `strategies/` bullet:

```markdown
- `strategies/common/` -- code shared between strategies, imported from here (no re-export shims): `scoring.py` (**pure** momentum scoring moved from cross_momentum: `momentum_inputs`, `apply_filters`, `momentum_score`, `score_stock` (`None` when filtered out or without a volatility), `rank`), `rebalancer.py` (`Rebalancer`, moved from bill_ackman, typed against the `RebalanceParams` protocol; reads the account ONCE before any order and never reads `cash` again, so a live sell filled within seconds is counted once -- pinned by two live-fill tests), `portfolio.py` (`TargetPortfolio`), `sessions.py` (`completed_bars`, `parse_rebalance_time`), `yahoo_daily_bars.py` (`YahooDailyBars`), `sector_provider.py` (`SectorProvider`, yfinance, not point-in-time).
```

3. In the `strategies/` bullet, append after the `congress_trades/` description:

```markdown
`bull_bear/` (`BullBearStrategy`, registered as `"bull_bear"`): lumibot's bull vs bear debate over cross_momentum's ranking, once a week (Tuesday 12:00 ET, `sleeptime = "1D"`, other sessions return at once; a Tuesday whose review completed is not run again after a restart, unlike cross_momentum). Code ranks the universe with `common/scoring.py` and cross_momentum's own `CONFIG` (never duplicated), debates the top 15 plus holdings ranked 16–35 (`debate_set.py`; a holding ranked worse or unranked is a forced exit, sold by code), and builds the fact sheet (`fact_sheet.py`). Four agents hand over through submit tools validated by `HandoffRecorder` (`handoff.py`, no `from __future__ import annotations`): a researcher run PER STOCK (news + SEC income statement and balance sheet, `tool_budget` 3, a failed run gives the note "research unavailable"), then the bull and the bear, independently and on the same evidence, with no research tool (every stock rated low/medium/high; the bear names a concern), then the judge (5–10 picks; `drops` must name exactly the held stocks not picked). Code sizes the picks by inverse volatility within [4%, 20%] summing to 98% (`sizing.capped_inverse_volatility`, a bisection, so both bounds hold) and rebalances through `common.Rebalancer`. No overlays. A debater with no valid submission after the forced retry, unusable data (`market_data.YahooBars`: one Yahoo batch in paper/live, retried after 60 s and 180 s; `GateBars` in backtests), a debate set under `min_picks`, or a broker error abandons the review with NO order, forced exits included; a backtest raises `FatalStrategyError` at 3 in a row. State in `data/bull_bear_state_<mode>.json` (wiped in backtests), `reviews.jsonl` in the run directory.
```

4. In the bill_ackman text, replace "`Rebalancer` is the only order code" with "`Rebalancer` (`strategies/common/rebalancer.py`) is the only order code".
5. In the cross_momentum Yahoo gotcha, replace `` `yahoo_daily_bars.YahooDailyBars` `` with `` `strategies/common/yahoo_daily_bars.YahooDailyBars` ``.

- [ ] **Step 2: Update `README.md`**

1. In the `SEC_EDGAR_USER_AGENT` paragraph (line ~75), change "(`bill_ackman`)" to "(`bill_ackman`, `bull_bear`)".
2. After the `bill_ackman` section (it ends with "The first run downloads about 5 GB of SEC data once."), insert:

```markdown
### 📈 `bull_bear` — Bull vs bear debate over the momentum ranking (researcher, bull, bear, judge)

| Field | Value |
| --- | --- |
| **File** | `strategies/bull_bear/agent_bull_bear.py` (`BullBearStrategy`) |
| **Model** | from `LLM_MODEL` in the env file (one model for the four agents) |
| **Agents** | `researcher` (one run per stock), `bull`, `bear`, `judge` |
| **Tools** | researcher: news + SEC income statement and balance sheet; bull, bear, judge: none. Each agent ends with one submit tool (`submit_note`, `submit_bull_case`, `submit_bear_case`, `submit_picks`); none can place an order |
| **Asset universe** | the `cross_momentum` universe file, cut to the top 15 by cross_momentum's momentum score (plus holdings still ranked 16–35), plus SHV |
| **Agent frequency** | once a week, Tuesday 12:00 ET |
| **Trading modes** | backtest, paper, live |
| **Benchmark** | SPY |
| **Env file** | `env/.env.bull_bear.<mode>`: `LLM_*`, `SEC_EDGAR_USER_AGENT`, `ALPACA_NEWS_*` (the researcher's news tool), plus the broker keys in paper/live |

A port of lumibot's "Bull vs Bear AI Stock Trading Bot": instead of a fixed list of 13 large caps, the agents debate the 15 strongest stocks of the cross_momentum ranking, computed by the same code. A researcher writes one note of dated facts per stock, the bull and the bear rate every stock from that same evidence, and the judge picks 5 to 10 winners. Code sizes them by inverse volatility (4% to 20% each, 98% invested), sells holdings that fell below rank 35 and the judge's drops, and places every order. Run `uv run agent bull_bear backtesting` (default window `PredefinedWindow.HALF_YEAR`; each review runs 15 to 25 researcher calls, so start short); each run writes `reviews.jsonl` (one line per weekly review) next to its report.
```

- [ ] **Step 3: Check the docs still match the code**

Run: `grep -n "bull_bear\|strategies/common" CLAUDE.md README.md` and read each hit against Tasks 1–10 (names, numbers, defaults).
Expected: every name exists in the code (`grep -rn "<name>" src` for any you are unsure of).

- [ ] **Step 4: Final full verification**

Run: `uv run pytest -q && uv run ruff check && uv run pyright`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs(bull_bear): CLAUDE.md and README for strategies/common and bull_bear

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

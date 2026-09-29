# cross_momentum Breadth Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a market-breadth exposure leg to cross_momentum: cut exposure in steps (100% / 70% / 40%) when fewer scored stocks trade above their own 100-day SMA, with a 5-point buffer before re-risking.

**Architecture:** Three pure functions in `utils.py` (breadth share, hysteresis step, step exposure). `compute_target_portfolio` computes breadth from the closes it already holds; a small `_breadth_exposure()` method turns it into a multiplier and keeps the step in `self.vars`; `on_trading_iteration` adds it as a third leg in `min(risk, vol, breadth)`. Trim and SHV parking (already on `main`) turn a cut into real de-risking.

**Tech Stack:** Python 3.14, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-29-cross-momentum-breadth-overlay-design.md`

## Global Constraints

- Branch: `feature/cross-momentum-breadth-overlay`. Stage explicit paths only.
- Config, exactly: `{"enabled": True, "sma_window": 100, "min_stocks": 50, "thresholds": (0.50, 0.30), "exposures": (1.0, 0.7, 0.4), "hysteresis": 0.05}`.
- Raw step: breadth ≥ 0.50 → 0; 0.30 ≤ breadth < 0.50 → 1; breadth < 0.30 → 2. Cutting is immediate; moving to a less defensive step needs the step's threshold plus `hysteresis`.
- Combination: `final_exposure = min(risk_exposure, vol_exposure, breadth exposure)`.
- Stock selection, weighting, trim and SHV parking are untouched.
- Tests never touch the network; hand-written fakes, no `MagicMock`.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; subjects use `Task N: ...`.

## Review Focus

1. **A stock with fewer closes than `sma_window`** — expect it excluded from numerator and denominator, not counted as "below" → pinned in Task 1.
2. **A non-finite close (NaN) in a stock's window** — expect the stock excluded, never silently counted below its SMA → pinned in Task 1.
3. **A close exactly equal to its SMA (flat stock)** — expect "not above" (strict comparison) → pinned in Task 1.
4. **No stock passes the filters this week (early return) after an earlier reading** — expect `breadth` reset to `None`, never a stale value feeding the leg → pinned in Task 2.
5. **Overlay disabled or breadth `None`** — expect exposure 1.0 and the stored step untouched → pinned in Task 2.

---

### Task 1: Pure breadth functions and config

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py` (add three functions)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/parameters.py` (`breadth_overlay` block)
- Test: `tests/strategies/test_cross_momentum_breadth.py` (create)

**Interfaces:**
- Produces (all in `utils.py`):
  - `breadth_share(closes_by_symbol: dict[str, list[float]], sma_window: int, min_stocks: int) -> float | None`
  - `next_breadth_step(breadth: float, previous_step: int | None, thresholds: tuple[float, ...], hysteresis: float) -> int`
  - `breadth_exposure(step: int, exposures: tuple[float, ...]) -> float`
  - `CONFIG["breadth_overlay"]` with the keys listed in Global Constraints.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_cross_momentum_breadth.py`:

```python
"""Breadth overlay for cross_momentum: pure breadth/step/exposure functions."""

import pytest

from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.strategies.cross_momentum.utils import (
    breadth_exposure,
    breadth_share,
    next_breadth_step,
)

THRESHOLDS = (0.50, 0.30)
RISING = [float(i) for i in range(1, 101)]  # last close 100 > SMA 50.5
FALLING = list(reversed(RISING))  # last close 1 < SMA 50.5


def test_config_carries_the_agreed_breadth_settings():
    assert CONFIG["breadth_overlay"] == {
        "enabled": True,
        "sma_window": 100,
        "min_stocks": 50,
        "thresholds": (0.50, 0.30),
        "exposures": (1.0, 0.7, 0.4),
        "hysteresis": 0.05,
    }


def test_breadth_share_is_the_fraction_above_their_sma():
    closes = {"A": RISING, "B": RISING, "C": RISING, "D": FALLING}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.75)


def test_a_stock_with_too_little_history_is_excluded_from_both_sides():
    closes = {"A": RISING, "B": FALLING, "SHORT": RISING[:50]}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.5)


def test_a_non_finite_close_excludes_the_stock():
    closes = {"A": RISING, "NAN": [*RISING[:-1], float("nan")]}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(1.0)


def test_a_flat_stock_is_not_above_its_sma():
    closes = {"FLAT": [10.0] * 100, "UP": RISING}

    assert breadth_share(closes, sma_window=100, min_stocks=1) == pytest.approx(0.5)


def test_fewer_valid_stocks_than_min_stocks_gives_none():
    closes = {"A": RISING, "B": FALLING, "SHORT": RISING[:50]}

    assert breadth_share(closes, sma_window=100, min_stocks=3) is None
    assert breadth_share({}, sma_window=100, min_stocks=1) is None


@pytest.mark.parametrize(
    ("breadth", "expected"),
    [(0.60, 0), (0.50, 0), (0.49, 1), (0.30, 1), (0.29, 2), (0.0, 2)],
)
def test_the_raw_step_without_history(breadth, expected):
    assert next_breadth_step(breadth, None, THRESHOLDS, 0.05) == expected


def test_cutting_is_immediate():
    assert next_breadth_step(0.25, 0, THRESHOLDS, 0.05) == 2
    assert next_breadth_step(0.29, 1, THRESHOLDS, 0.05) == 2
    assert next_breadth_step(0.45, 0, THRESHOLDS, 0.05) == 1


@pytest.mark.parametrize(
    ("previous", "breadth", "expected"),
    [
        (2, 0.33, 2),  # 0.30 + 0.05 not cleared
        (2, 0.35, 1),
        (1, 0.52, 1),  # 0.50 + 0.05 not cleared
        (1, 0.55, 0),
    ],
)
def test_re_risking_needs_the_buffer(previous, breadth, expected):
    assert next_breadth_step(breadth, previous, THRESHOLDS, 0.05) == expected


def test_a_multi_step_jump_lands_on_the_least_defensive_step_whose_bound_is_met():
    assert next_breadth_step(0.52, 2, THRESHOLDS, 0.05) == 1
    assert next_breadth_step(0.60, 2, THRESHOLDS, 0.05) == 0


def test_breadth_exposure_maps_each_step():
    assert [breadth_exposure(step, (1.0, 0.7, 0.4)) for step in (0, 1, 2)] == [1.0, 0.7, 0.4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_breadth.py -q`
Expected: collection error — `ImportError: cannot import name 'breadth_exposure'`.

- [ ] **Step 3: Implement**

In `parameters.py`, replace the existing block

```python
    "breadth_overlay": {
        "enabled": True,
        "sma_window": 100,
    },
```

with:

```python
    # ── Breadth overlay (market regime) ─────────
    # Share of the scored stocks closing above their own sma_window-day SMA. thresholds are the breadth levels
    # below which the strategy enters step 1 / step 2; exposures[step] scales the stock weights. Cutting is
    # immediate; re-risking to a less defensive step needs that step's threshold plus `hysteresis`.
    # Fewer than min_stocks valid stocks leaves the leg neutral.
    "breadth_overlay": {
        "enabled": True,
        "sma_window": 100,
        "min_stocks": 50,
        "thresholds": (0.50, 0.30),
        "exposures": (1.0, 0.7, 0.4),
        "hysteresis": 0.05,
    },
```

In `utils.py`, add after `compute_volatility_exposure`:

```python
def breadth_share(
    closes_by_symbol: dict[str, list[float]],
    sma_window: int,
    min_stocks: int,
) -> float | None:
    """Share of stocks whose latest close is above their own `sma_window`-day simple moving average.

    A stock with fewer than `sma_window` closes, or a non-finite last close or SMA, is left out of both the
    numerator and the denominator. A close equal to its SMA does not count as above. Returns None when fewer
    than `min_stocks` stocks are valid, so a thin day can't trigger an exposure cut.
    """
    above = 0
    valid = 0
    for closes in closes_by_symbol.values():
        if len(closes) < sma_window:
            continue
        sma = sum(closes[-sma_window:]) / sma_window
        if not (math.isfinite(sma) and math.isfinite(closes[-1])):
            continue
        valid += 1
        if closes[-1] > sma:
            above += 1
    if valid < min_stocks:
        return None
    return above / valid


def next_breadth_step(
    breadth: float,
    previous_step: int | None,
    thresholds: tuple[float, ...],
    hysteresis: float,
) -> int:
    """Breadth step (0 = full exposure, higher = more defensive) with hysteresis on the way back up.

    `thresholds` runs from the least to the most defensive boundary, e.g. (0.50, 0.30): breadth below 0.50
    is at least step 1, below 0.30 step 2. A cut to a more defensive step is immediate; moving to a less
    defensive one needs breadth above that step's threshold plus `hysteresis`, and when breadth jumps several
    steps the result is the least defensive step whose bound is met.
    """
    raw = sum(1 for threshold in thresholds if breadth < threshold)
    if previous_step is None or raw >= previous_step:
        return raw
    step = previous_step
    while step > raw and breadth >= thresholds[step - 1] + hysteresis:
        step -= 1
    return step


def breadth_exposure(step: int, exposures: tuple[float, ...]) -> float:
    """Exposure multiplier for a breadth step."""
    return exposures[step]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies -q && uv run ruff check src/trading_agent_framework/strategies/cross_momentum tests/strategies`
Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/utils.py src/trading_agent_framework/strategies/cross_momentum/parameters.py tests/strategies/test_cross_momentum_breadth.py
git commit -m "Task 1: pure breadth functions -- share above SMA, hysteresis step, step exposure -- and config

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Wire breadth into the strategy

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (imports, module and class docstrings, `__init__`, `compute_target_portfolio`, `_breadth_exposure` new, `on_trading_iteration`)
- Test: `tests/strategies/test_cross_momentum_breadth.py` (append)

**Interfaces:**
- Consumes: `breadth_share`, `next_breadth_step`, `breadth_exposure`, `CONFIG["breadth_overlay"]` (Task 1).
- Produces: `self.vars.breadth: float | None`, `self.vars.breadth_step: int | None`; `CrossMomentumStrategy._breadth_exposure(self) -> float`.

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/test_cross_momentum_breadth.py`, add to the imports at the top:

```python
from types import SimpleNamespace

from trading_agent_framework.strategies.cross_momentum.agent_cross_momentum import CrossMomentumStrategy
```

Append:

```python
class FakeBreadthStrategy:
    """Just enough of `Strategy` for `_breadth_exposure`."""

    _breadth_exposure = CrossMomentumStrategy._breadth_exposure

    def __init__(self, *, breadth, step=None, enabled=True):
        self.parameters = {"breadth_overlay": {**CONFIG["breadth_overlay"], "enabled": enabled}}
        self.vars = SimpleNamespace(breadth=breadth, breadth_step=step)
        self.infos: list[str] = []

    def log_info(self, message, *args, **kwargs):
        self.infos.append(message)


def test_a_breadth_of_forty_percent_caps_exposure_at_seventy_percent_and_logs_it():
    fake = FakeBreadthStrategy(breadth=0.40)

    assert fake._breadth_exposure() == pytest.approx(0.7)
    assert fake.vars.breadth_step == 1
    assert any("40%" in message and "step 1" in message for message in fake.infos)


def test_no_reading_leaves_exposure_at_one_and_the_step_untouched():
    fake = FakeBreadthStrategy(breadth=None, step=2)

    assert fake._breadth_exposure() == 1.0
    assert fake.vars.breadth_step == 2


def test_a_disabled_overlay_ignores_breadth():
    fake = FakeBreadthStrategy(breadth=0.10, step=None, enabled=False)

    assert fake._breadth_exposure() == 1.0
    assert fake.vars.breadth_step is None


def test_the_stored_step_carries_hysteresis_across_consecutive_calls():
    fake = FakeBreadthStrategy(breadth=0.25)
    assert fake._breadth_exposure() == pytest.approx(0.4)

    fake.vars.breadth = 0.33  # above 0.30 but below 0.35: stay defensive
    assert fake._breadth_exposure() == pytest.approx(0.4)

    fake.vars.breadth = 0.36
    assert fake._breadth_exposure() == pytest.approx(0.7)


class FakeScoringStrategy:
    """Just enough of `Strategy` for `compute_target_portfolio`; each ticker's indicators are canned."""

    compute_target_portfolio = CrossMomentumStrategy.compute_target_portfolio

    def __init__(self, indicators):
        self.parameters = dict(CONFIG)
        self.vars = SimpleNamespace(universe=list(indicators), target_closes={}, breadth=None, breadth_step=None)
        self._indicators = indicators
        self.errors: list[str] = []

    def _compute_indicators_for_ticker(self, ticker):
        return self._indicators[ticker]

    def log_info(self, *args, **kwargs): ...

    def log_error(self, message, *args, **kwargs):
        self.errors.append(message)


def _indicator(symbol, closes):
    return {
        "symbol": symbol,
        "price": 50.0,
        "avg_dollar_volume": 50_000_000.0,
        "volatility": 0.3,
        "trading_days": 300,
        "ret_12_1m": 0.2,
        "ret_6_1m": 0.1,
        "ret_3m": 0.05,
        "atr": None,
        "closes": closes,
    }


def test_compute_target_portfolio_sets_breadth_from_the_scored_stocks():
    rising = [float(i) for i in range(1, 301)]
    indicators = {f"UP{i}": _indicator(f"UP{i}", rising) for i in range(45)}
    indicators |= {f"DN{i}": _indicator(f"DN{i}", list(reversed(rising))) for i in range(15)}
    fake = FakeScoringStrategy(indicators)

    fake.compute_target_portfolio()

    assert fake.vars.breadth == pytest.approx(0.75)


def test_a_week_with_no_scored_stocks_resets_breadth_to_none():
    fake = FakeScoringStrategy({"AAA": None, "BBB": None})
    fake.vars.breadth = 0.40  # last week's reading must not survive

    target, ranks = fake.compute_target_portfolio()

    assert (target, ranks) == ([], {})
    assert fake.vars.breadth is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_breadth.py -q`
Expected: the four `_breadth_exposure` tests fail with `AttributeError: type object 'CrossMomentumStrategy' has no attribute '_breadth_exposure'` at import of the fake; because that class attribute assignment fails at import, the whole module errors. Expected either way: FAIL/ERROR on `_breadth_exposure`.

- [ ] **Step 3: Implement**

In `agent_cross_momentum.py`:

1. Module docstring: replace the block from `  5. Portfolio risk overlay` through `  7. Combined via min(risk_exposure, vol_exposure) — most conservative wins` with:

```
  5. Portfolio risk overlay (beta/vol/corr) — first exposure leg
  6. Breadth overlay (share of scored stocks above their 100d SMA, stepped, with hysteresis) — second leg
  7. Fast/slow volatility targeting (equity-curve-based) — third exposure leg
  8. Combined via min(risk_exposure, breadth_exposure, vol_exposure) — most conservative wins
```

and renumber the two following lines to `  9. Hysteresis: ...` and `  10. Park the de-risked capital ...`.

2. Class docstring: replace `with the portfolio risk overlay via min() — the most conservative leg` with `with the portfolio risk overlay and the market-breadth overlay via min() — the most conservative leg`.

3. Imports: add `breadth_exposure,`, `breadth_share,` and `next_breadth_step,` to the `from .utils import (...)` block in alphabetical order (`breadth_*` after `apply_filters`; `next_breadth_step` after `momentum_score`).

4. In `__init__`, after `self.vars.target_closes = {}` add:

```python
        # Breadth overlay state: this rebalance's reading and the step it put the book on (None until first read)
        self.vars.breadth = None
        self.vars.breadth_step = None
```

5. In `compute_target_portfolio`, right after the first `self.log_info(f"Computing target portfolio ...")` add:

```python
        self.vars.breadth = None  # no stale reading if this run returns early
```

and right after `all_ranks = {entry["symbol"]: entry["rank"] for entry in scored}` add:

```python
        breadth_config = self.parameters["breadth_overlay"]
        self.vars.breadth = breadth_share(
            {entry["symbol"]: entry["closes"] for entry in scored},
            breadth_config["sma_window"],
            breadth_config["min_stocks"],
        )
```

6. Add this method right before `_price_or_zero`:

```python
    def _breadth_exposure(self) -> float:
        """Exposure multiplier from market breadth (share of scored stocks above their SMA), with hysteresis.

        1.0 when the overlay is disabled or there is no reading; otherwise the current step's multiplier. The
        step is kept on `self.vars` so a breadth hovering at a threshold doesn't flip the book week to week.
        """
        config = self.parameters["breadth_overlay"]
        breadth = self.vars.breadth
        if not config["enabled"] or breadth is None:
            return 1.0
        step = next_breadth_step(breadth, self.vars.breadth_step, config["thresholds"], config["hysteresis"])
        self.vars.breadth_step = step
        exposure = breadth_exposure(step, config["exposures"])
        self.log_info(f"Breadth: {breadth:.0%} of stocks above their {config['sma_window']}d SMA -> step {step} (exposure {exposure:.0%})")
        return exposure
```

7. In `on_trading_iteration`, insert before `        # Step 4: Fast/Slow Volatility Targeting — compute realized vol`:

```python
        # Step 4: Breadth overlay — market regime from the share of scored stocks above their SMA
        breadth_leg = self._breadth_exposure()

```

then rename the following comments: `# Step 4: Fast/Slow` → `# Step 5: Fast/Slow`, `# Step 5: Combine` → `# Step 6: Combine`, `# Step 6: Scale` → `# Step 7: Scale`, `# Step 7: Store` → `# Step 8: Store`, `# Step 8: Rebalance` → `# Step 9: Rebalance`. Replace the combine block:

```python
        final_exposure = min(risk_exposure, vol_exposure)
        if final_exposure < 1.0:
            self.log_info(f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, vol={vol_exposure:.0%})")
```

with:

```python
        final_exposure = min(risk_exposure, vol_exposure, breadth_leg)
        if final_exposure < 1.0:
            self.log_info(f"Combined exposure: {final_exposure:.0%} (risk={risk_exposure:.0%}, breadth={breadth_leg:.0%}, vol={vol_exposure:.0%})")
```

(The local is `breadth_leg`, not `breadth_exposure`, so it doesn't shadow the imported function.)

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/strategies/test_cross_momentum_breadth.py
git commit -m "Task 2: feed market breadth into the exposure legs as min(risk, breadth, vol)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## After the plan

The user runs `uv run agent cross_momentum backtesting` and compares `metrics.json` (CAGR, Sharpe, Sortino, beta,
correlation, max drawdown) plus trade count and fees against
`logs/cross_momentum/backtesting/2026-09-28_120209_backtesting`. The backtest log shows one
`Breadth: NN% of stocks above their 100d SMA -> step S (exposure E%)` line per rebalance, which tells how often
each step was active.

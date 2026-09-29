# cross_momentum: breadth overlay as a market-regime filter

Date: 2026-09-29 · Branch: `feature/cross-momentum-breadth-overlay`

## Problem

The −35% max drawdown came from momentum-crash reversals: broad market weakness followed by a violent bounce
in the beaten-down names momentum is short of. Nothing in the strategy reads market breadth.
`CONFIG["breadth_overlay"] = {"enabled": True, "sma_window": 100}` already exists in `parameters.py` but no
code reads it.

Earlier attempts (memory): beta-adjusted ranking and a 25% sector cap degraded the metrics and were
discarded; SHV parking (on `main`) backtested ≈ baseline. This change leaves stock selection and weighting
untouched and only adds an exposure leg, the kind of change that has not hurt so far.

## Goal

Cut exposure when fewer stocks are in uptrends, and put the freed capital in SHV (already how an exposure cut
works on `main`: the strategy trims held stocks to the scaled target and parks the rest). This mainly lowers
beta and drawdown in bear markets; it is not expected to lower overall correlation. Judged by the user's
backtest against `logs/cross_momentum/backtesting/2026-09-28_120209_backtesting` (CAGR 0.32, Sharpe 1.10,
Sortino 1.54, beta 1.07, correlation 0.65, max drawdown −0.35), including trade count and fees.

Replaces behaviour outright when `enabled` (the config already has the flag). Unchanged: universe, filters,
ranking, weighting, risk overlay, vol targeting, trim, SHV parking, weekly Tuesday rebalance.

## 1. Signal

Breadth = share of the **scored** stocks (those that passed the filters in `compute_target_portfolio`) whose
latest close is above their own `sma_window`-day simple moving average.

- Computed from the `closes` already attached to every scored entry; no extra data calls. Each scored stock has
  ≥ 250 closes, so the 100-day SMA is always defined.
- With fewer than `min_stocks` scored stocks the result is `None` and the leg stays at 1.0, so a thin universe
  day cannot trigger a cut.
- Stored as `self.vars.breadth` by `compute_target_portfolio` (set to `None` on its early returns).

## 2. Step mapping with hysteresis

Steps: 0 = full exposure, 1 = reduced, 2 = defensive. `thresholds = (0.50, 0.30)` are the breadth levels below
which the strategy enters step 1 and step 2; `exposures = (1.0, 0.7, 0.4)` are the multipliers per step.

- **Raw step:** breadth ≥ 0.50 → 0; 0.30 ≤ breadth < 0.50 → 1; breadth < 0.30 → 2.
- **Cutting is immediate:** if the raw step is more defensive than the current step, move to the raw step.
- **Re-risking needs a buffer:** to move to a less defensive step, breadth must clear that step's threshold plus
  `hysteresis` (0.05): from step 2 into step 1 needs breadth ≥ 0.35; into step 0 needs ≥ 0.55. When breadth
  jumps several steps, land on the least defensive step whose bound is met (e.g. 0.20 → 0.52 lands on step 1;
  0.20 → 0.60 lands on step 0).
- The current step lives in `self.vars.breadth_step` and is **persisted**, so a crash or a stop/start in
  paper or live resumes on the same step instead of losing the hysteresis (see section 3b). `None` means "no
  history" and the first reading applies directly.
- `breadth is None` → step 0 exposure 1.0 and the stored step is left unchanged.

## 3. Pure functions (`utils.py`)

```python
def breadth_share(closes_by_symbol: dict[str, list[float]], sma_window: int, min_stocks: int) -> float | None
def next_breadth_step(breadth: float, previous_step: int | None, thresholds: tuple[float, ...], hysteresis: float) -> int
def breadth_exposure(step: int, exposures: tuple[float, ...]) -> float
```

- `breadth_share`: counts stocks with `len(closes) >= sma_window` and `closes[-1] > mean(closes[-sma_window:])`;
  the denominator is the number of stocks with enough closes; `None` if that number is below `min_stocks`.
- `next_breadth_step`: as in section 2 (`thresholds` is ordered from the least to the most defensive boundary).
- `breadth_exposure`: `exposures[step]`.

## 3b. Persistence of the step

Same pattern as the equity history (`load_equity_history` / `save_equity_history`): one small JSON file per
trading mode, `data/cross_momentum_breadth_{mode}.json`, holding
`{"date": "YYYY-MM-DD", "step": 1, "breadth": 0.42}`.

```python
def load_breadth_step(path: Path, n_steps: int) -> int | None
def save_breadth_step(path: Path, step: int, breadth: float, today_str: str) -> None
```

- `load_breadth_step` returns `None` for a missing file, unreadable or corrupt JSON, a non-dict payload, or a
  `step` that is not an `int` (a `bool` does not count) in `0 <= step < n_steps`. `None` means the first
  reading applies directly.
- `save_breadth_step` writes atomically (temp file + `os.replace`, parent directory created). A disk error only
  logs a warning; it never raises into the trading loop.
- `initialize()` loads the step into `self.vars.breadth_step` (alongside the equity history).
- `_breadth_exposure()` saves the step every time it applies a reading, stamped with the strategy clock's date
  (`self.get_datetime()`, never wall-clock time).
- `run_backtesting()` deletes the file first, like the equity history, so a backtest starts clean and never
  inherits or leaves state for paper/live (their files are separate by mode).
- The step is stored as-is even if the bot was down for weeks: the next reading cuts immediately if it is
  more defensive, and re-risks only past the buffer, exactly as if the bot had kept running.

## 4. Wiring in `on_trading_iteration`

- New step between the risk overlay and vol targeting (renumbering the comments):
  `breadth = self.vars.breadth`; if `enabled` and `breadth is not None`, compute the step, store it, and set
  `breadth_exposure`; otherwise `breadth_exposure = 1.0`. One log line:
  `Breadth: 42% of stocks above their 100d SMA -> step 1 (exposure 70%)`.
- `final_exposure = min(risk_exposure, vol_exposure, breadth_exposure)`; the combined-exposure log line gains
  the third leg.
- Everything downstream (scaling target weights, trim, SHV parking) is unchanged.

## 5. Config (`parameters.py`)

```python
"breadth_overlay": {
    "enabled": True,
    "sma_window": 100,
    "min_stocks": 50,
    "thresholds": (0.50, 0.30),
    "exposures": (1.0, 0.7, 0.4),
    "hysteresis": 0.05,
},
```

## Testing (TDD, no network)

- `breadth_share`: 3 of 4 stocks above their SMA → 0.75; a stock with too few closes is excluded from both
  numerator and denominator; fewer than `min_stocks` valid stocks → `None`.
- `next_breadth_step`: raw boundaries (0.50 → step 0, 0.49 → 1, 0.30 → 1, 0.29 → 2); immediate cut
  (step 0, breadth 0.25 → 2); buffer (step 2, 0.33 → 2; 0.35 → 1; step 1, 0.52 → 1; 0.55 → 0); multi-step
  jumps (0.20 → 0.52 → 1, 0.20 → 0.60 → 0); `previous_step None` → raw step.
- `breadth_exposure`: maps 0/1/2 to 1.0/0.7/0.4.
- Persistence with a temp path: save then load round-trips the step; a missing file, corrupt JSON, a non-dict
  payload, an out-of-range step, a float step and a boolean step all load as `None`; a failing write (parent
  path is a file) does not raise.
- Restart continuity: after a reading of 0.25 (step 2) is applied and saved, a fresh strategy that loads the
  file in `initialize()` treats 0.33 as still step 2 (hysteresis survives) and 0.36 as step 1.
- Strategy wiring with the existing fake-strategy style: a breadth of 0.4 caps `final_exposure` at 0.7 when the
  other legs are 1.0; a `None` breadth leaves it at 1.0; `enabled: False` ignores breadth; the stored step
  drives hysteresis across two consecutive calls.

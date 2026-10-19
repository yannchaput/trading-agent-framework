# cross_momentum: trend-filtered parking sleeve (SHV/GLD/IEF) and a 5-weekday A/B protocol

Date: 2026-10-07

## Problem

The exposure legs (risk overlay, breadth, fast/slow vol targeting) do their job: the strategy's beta is 0.76 in
high-volatility markets and 1.41 in calm ones, for the same ~31% annual return in both, and the full-period beta
(0.86) is driven by the volatile periods where the legs cut. Raising exposure would raise beta about one for one,
and in calm markets the book is already ~94% in stocks.

What the legs take out of stocks earns the T-bill rate. Over run `2026-10-07_214307` (2016-01 → 2026-09-23,
Tuesday rebalance) the book averaged 85% stocks, 12% SHV, 2% cash. That 12% is where CAGR can grow without
adding beta: assets with near-zero beta to SPY and a positive return, held only while they trend.

A second problem decides how any change is judged: the same code with a Wednesday rebalance
(`2026-10-07_074747`) got CAGR 29.6% vs 32.2% on Tuesday (alpha 0.153 vs 0.175, beta 0.88 vs 0.86). A one-run
A/B cannot see an effect smaller than that spread.

## Goal

1. Replace SHV-only parking with a trend-filtered sleeve: half GLD, half IEF, each held only while its last
   completed close is above its 200-day SMA, the rest in SHV.
2. Judge it with a 5-weekday protocol: the baseline and the candidate each run once per rebalance weekday
   (Monday to Friday) and are compared on the means.

**Success bar (KEEP), all of:**

- mean CAGR of the candidate > mean CAGR of the baseline;
- mean alpha ≥ baseline's;
- mean beta ≤ baseline's;
- mean max drawdown no worse than the baseline's by more than 1 percentage point (`mdd_c >= mdd_b - 0.01`, max
  drawdown being negative);
- the candidate's CAGR beats the baseline's on at least 3 of the 5 days.

Anything else is REJECT: the branch is deleted unmerged and the result is recorded in memory, as for the earlier
cross_momentum experiments.

The sleeve **replaces** SHV-only parking outright (no enable flag). Unchanged: universe, filters, ranking,
weights, the three exposure legs and their `min()`, rank-35 hysteresis, trims, `cash_buffer_pct`, `min_trade_pct`,
`_REBALANCE_BAND`, rebalance day and time.

## Part 1 — the sleeve

### 1.1 Config (`parameters.py`)

```python
"parking": {
    "symbol": "SHV",                    # fallback: holds every share whose trend is off
    "trend_assets": ("GLD", "IEF"),     # each gets 1/len(trend_assets) of the sleeve while trending
    "trend_sma_window": 200,
    "min_trade_pct": 0.01,
},
```

### 1.2 Signal (pure, `utils.py`)

```python
def sleeve_weights(
    closes_by_asset: dict[str, list[float]],
    trend_assets: tuple[str, ...],
    sma_window: int,
    fallback: str,
) -> dict[str, float]:
```

- Returns a weight for `fallback` and for every trend asset; the weights sum to 1.0.
- Each trend asset gets `1 / len(trend_assets)` when its last close is **strictly** above the simple average of
  its last `sma_window` closes; otherwise its share is added to `fallback` (and it gets 0.0).
- A trend asset missing from `closes_by_asset`, with fewer than `sma_window` closes, or with a non-finite last
  close or SMA is treated as off.
- No hysteresis: the weekly cadence, the ±20% band and `min_trade_pct` already damp a close hovering at its SMA.

### 1.3 Data

On rebalance days only, inside `rebalance()` before Phase 1, for each trend asset:
`self.vars.alpaca_rate_limiter.wait()`, then
`get_historical_prices(asset, length=_HISTORY_BARS + 1, timestep="day")`, then
`completed_bars(df, self._market_date()).tail(_HISTORY_BARS)` (300 bars ≥ the 200 the SMA needs). The closes go
to `sleeve_weights`.

A `BrokerError`, `None` or empty bars for an asset: log a warning, leave it out of `closes_by_asset` (so its half
goes to SHV), and carry on with the rebalance.

### 1.4 Rebalance (`agent_cross_momentum.py`)

- `parking_target` (the total sleeve value) keeps its formula:
  `max(0, pv * (1 - cash_buffer_pct) - Σ stock target values - hysteresis_value)`.
- Each sleeve symbol's target is `parking_target * weight[symbol]`; its current value is its held quantity times
  `_price_or_zero(symbol)`.
- The sleeve symbols are `(parking.symbol, *parking.trend_assets)`.
- **Phase 1 (sells):** the exit/hysteresis loop skips every sleeve symbol (today it skips only SHV). Then, for
  each sleeve symbol, the current SHV rule applies: if `value > target * (1 + _REBALANCE_BAND)` and the excess is at
  least `min_trade_pct * pv`, sell the excess (`fractional_qty(excess / price)`), or the exact held quantity when
  the target is 0. Proceeds add to `estimated_sell_proceeds`.
- **Phase 2 (stock buys):** unchanged.
- **Phase 3 (sleeve buys):** trend assets first, in `trend_assets` order, then SHV. For each: if
  `value < target * (1 - _REBALANCE_BAND)`, buy `fractional_qty(min(target - value, available_cash) / price)` when
  that dollar amount is at least `min_trade_pct * pv`, and deduct the cost from `available_cash`.
- A sleeve symbol whose price is ≤ 0: warning, no order for that symbol this week; the others proceed.
- Order failures keep the existing pattern (log, carry on; a failed sell adds no proceeds).
- `__init__` removes the sleeve symbols from `self.vars.universe`, so a sleeve asset can never be scored, ranked
  or bought as a stock.

Logs: one line per rebalance,
`Sleeve: GLD on (close 245.10 > SMA200 231.04), IEF off (94.20 <= 95.01) -> SHV 50% / GLD 50% / IEF 0%`, then
one `Parking: <symbol> target $X (current $Y)` line per sleeve symbol (replacing today's single SHV line), plus the
existing per-order lines.

### 1.5 Wiring

- `_backtest_preload_assets()` adds the trend assets to the universe and SHV.
- Module and class docstrings: point 10 becomes "park the de-risked capital in a trend-filtered SHV/GLD/IEF
  sleeve". README's cross_momentum section gets one sentence on the sleeve.
- `CLAUDE.md`'s cross_momentum gotcha is not affected; no new gotcha.

Known and accepted:

- With `BROKER=ibkr` on an EU retail account, PRIIPs blocks GLD and IEF as it blocks SHV: their orders fail and
  are logged, and the capital stays in cash.
- The risk diagnostics report GLD and IEF as "unknown sector" positions, like SHV.
- A paper/live account that holds SHV today moves part of it into GLD/IEF on the first rebalance after deployment.
- 2016–2026 was a strong decade for gold; the trend filter limits, but does not remove, that hindsight.

### 1.6 Tests (TDD)

`tests/strategies/test_cross_momentum_sleeve.py` (pure):

1. Both trending → GLD 0.5, IEF 0.5, SHV 0.0.
2. One trending → that asset 0.5, SHV 0.5, the other 0.0.
3. None trending → SHV 1.0.
4. Close equal to its SMA counts as off.
5. Short history, a missing asset, and a NaN close each count as off.
6. Weights always sum to 1.0.

`tests/strategies/test_cross_momentum_rebalance.py` (hand-written `FakeStrategy`):

7. Per-asset targets: idle cash with both trends on is split between GLD and IEF (≈ half the parking target each).
8. A sleeve symbol (GLD, IEF, SHV) is never sold as unranked.
9. A trend that turns off sells that asset's whole holding and the proceeds go to SHV.
10. Buy order: GLD and IEF orders are submitted before the SHV order, and the sleeve buys fit `available_cash`.
11. A missing price for one sleeve asset places no order for it; the other sleeve and stock orders go through.
12. A bars fetch failure for a trend asset sends its half to SHV and logs a warning.
13. The existing parking, trim and cash-buffer tests still pass (adapted to the per-symbol targets where needed).

## Part 2 — the 5-weekday protocol

### 2.1 Why sequential

Each backtest deletes and rewrites `data/cross_momentum_ptf_history_backtesting.json` and
`data/cross_momentum_breadth_backtesting.json`; two concurrent runs would corrupt each other's vol and breadth
state. One run takes ~13 min (782 s for `2026-10-07_214307`), so a five-day set takes about 65 min.

### 2.2 Pure module: `strategies/cross_momentum/weekday_protocol.py`

- `Manifest` (dataclass): `label`, `slippage`, `window` (`start`, `end` as ISO strings), `universe_sha256`, and
  `runs: dict[int, RunEntry]` keyed by `day_of_week`; `RunEntry` holds `run_dir`, `commit`, `dirty`.
  `load_manifest(path)` / `save_manifest(path, manifest)` (JSON).
- `days_to_run(manifest, days) -> list[int]`: the requested days not already in the manifest (resume).
- `RunMetrics` (dataclass): `cagr`, `alpha`, `beta`, `max_drawdown`, `sharpe`, read from a run's `metrics.json`
  keys `cagr_strategy`, `alpha`, `beta`, `max_drawdown_strategy`, `sharpe_strategy`.
- `compare(baseline, candidate, metrics_by_day_b, metrics_by_day_c) -> Comparison`:
  - raises `ValueError` when the two manifests differ in window, slippage, day set or universe hash;
  - `Comparison` holds the per-day rows, the means of each metric for both sets and their deltas, the CAGR win
    count, the list of warnings (a set mixing commits, or any run with `dirty=True`), the list of failed criteria
    (empty means KEEP) and `verdict` (`"KEEP"` / `"REJECT"`), applying the success bar above.
- `render(comparison) -> str`: the table and verdict as plain text.

No I/O beyond the manifest JSON helpers and a `read_run_metrics(run_dir)` helper; no strategy import.

### 2.3 CLI: `scripts/experiments/cross_momentum_weekdays.py`

Not part of the automated suite (like `scripts/tests/`).

- `run --label <name> [--slippage 0] [--days 0,1,2,3,4]`
  - loads the env as `main.py` does (`load_strategy_env("cross_momentum", "backtesting", project_root)`), builds a
    `PlaceholderBroker("cross_momentum")` and the universe with `load_cross_momentum_universe()`;
  - creates or loads `logs/cross_momentum/experiments/<label>.json`; on load, refuses to continue if the stored
    slippage, window or universe hash differ from the current ones;
  - for each day in `days_to_run(...)`, sequentially:
    `CrossMomentumStrategy(broker=..., mode=TradingMode.BACKTESTING, universe=..., parameters={"day_of_week": d})`
    then `.run_backtesting(slippage=Decimal(slippage))`, then records `run_dir`, `git rev-parse HEAD` and
    `git status --porcelain` non-empty as `dirty`, and saves the manifest after every run.
- `compare <baseline manifest path> <candidate manifest path>`: loads both manifests and each run's metrics,
  prints `render(compare(...))`, exits 0 on KEEP and 1 on REJECT. It takes paths, not labels, because the
  baseline set is recorded under `main`'s `logs/` and the candidate set under the worktree's.

`run_dir` is stored as an absolute path, so a manifest can be read from either checkout.

`universe_sha256` is the sha256 of `data/universe/us_stock_universe.json`; the window comes from the strategy's
`backtesting_start` / `backtesting_end` parameters.

### 2.4 Tests (TDD, `tests/strategies/test_cross_momentum_weekday_protocol.py`)

1. KEEP when every criterion holds.
2. Each criterion fails alone, both at its threshold and just past it: mean CAGR equal → REJECT; mean alpha just
   below → REJECT, equal → pass; mean beta just above → REJECT, equal → pass; max drawdown exactly 1 pt worse →
   pass, just past → REJECT; 2 wins of 5 → REJECT, 3 → pass.
3. Refusals: different window, slippage, day set or universe hash raise `ValueError`.
4. Warnings: mixed commits and a dirty run are listed, and do not change the verdict.
5. `days_to_run` skips recorded days and keeps the requested order.
6. Manifest JSON round trip.

## Order of work

1. Part 2 (protocol module, tests, CLI) lands on `main`.
2. The `shv-baseline` set runs from `main` (background, ~65 min).
3. Part 1 is built in a worktree; the universe file is copied into the worktree's `data/universe/` (gitignored),
   and the hash check catches a mismatch.
4. The `trend-sleeve` set runs from the worktree, then `compare` is given the two manifest paths.
5. KEEP → merge. REJECT → delete the branch unmerged and record the result in memory.

Out of scope: slippage sensitivity (the CLI's `--slippage` makes a 5 bps pair of sets possible later), the
hysteresis positions escaping the exposure legs, and the `cash_buffer_pct` level.

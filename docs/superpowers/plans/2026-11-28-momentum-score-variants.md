# Momentum score variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test four published momentum score variants against today's score on the rank-only book, development 2005-2015 then a one-shot holdout 2016-2026, rule A (higher CAGR, Sharpe no lower).

**Architecture:** One research script, `scripts/bull_bear/score_variants.py`, importing the simulator, data loader and helpers of `scripts/bull_bear/rank_only_stress.py` (a sibling import: the script's folder is on `sys.path` when run). Pure score and book functions checked by `self_check()`, as in the stress script. No `src/` change.

**Tech Stack:** Python 3.14, pandas, the repo's `strategies/common/scoring.py`.

**Spec:** `docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md`

## Global Constraints

- Variants: `today`, `vol_scaled`, `residual`, `high_52w`, `r3_skip`, defined exactly as in the spec table; no tuning.
- Every variant keeps today's filters through `score_stock` on the last `CLOSES` (300) sessions, with `max_volatility` exempt for holdings; the variant only changes the sort key.
- Book (step 1): top 10, 9.8% each, weekly (`rebalance_days`), rest in SHV, 5 bps a side: `rank_only_stress.simulate` unchanged.
- Development 2005-01-03 to 2015-12-31; holdout 2016-01-04 to 2026-11-24; each window starts a fresh book at 1.0.
- Pass on a window: CAGR > `today`'s and Sharpe >= `today`'s (daily mean / std x sqrt(252), no risk-free rate). Holdout runs only for development passers, in the same invocation. Kept only if it passes both.
- Step 2 (Task 3) runs only if a variant passed both windows.
- Cache: read `cache/stress_2005/` only; write nothing under `data/` or `src/`.
- No change to the rule after a result is seen.

## Review Focus

- A stock with fewer than 757 sessions: `residual` returns None and the stock is not ranked by that variant (never scored on a short window).
- A flat or perfectly market-explained stock (residual std 0) or a non-finite score: None, never `inf`/NaN sorted to the top.
- `high_52w` ties at exactly 1.0 (many stocks at their high): broken by today's score, deterministic.
- A stock session SPY has no bar for (or the reverse): SPY aligned on the stock's own dates, missing dates dropped from both before sampling months.
- Look-ahead: every score reads sessions strictly before the trade day (reuse `ranked`'s slicing rule from `rank_only_stress.ranked_symbols`).

---

### Task 1: Score variants and book rule (pure)

**Files:**
- Create: `scripts/bull_bear/score_variants.py`

**Interfaces:**
- Consumes: `trading_agent_framework.strategies.common.scoring` (`MomentumRow`, `annualized_volatility`, `compute_return_from_prices`, `momentum_score`), `CONFIG` from cross_momentum parameters.
- Produces:
  - `VARIANTS: tuple[str, ...] = ("today", "vol_scaled", "residual", "high_52w", "r3_skip")`
  - `variant_score(name: str, row: MomentumRow, closes: list[float], spy: list[float]) -> float | None`: `closes` and `spy` are date-aligned, oldest first, sessions strictly before the trade day (up to 800); `row` is `score_stock` on the last 300 of them. Returns None for a non-finite result.
    - `today`: `row.score`. `vol_scaled`: `row.score / annualized_volatility(closes, 126)`. `high_52w`: `closes[-1] / max(closes[-252:])`. `r3_skip`: `momentum_score(row.inputs.ret_12_1m, row.inputs.ret_6_1m, compute_return_from_prices(closes, 63, 21), CONFIG weights)`.
    - `residual`: needs >= 757 closes. Sample `closes[-757::21]` and `spy[-757::21]` (37 points -> 36 simple returns each), OLS of stock on SPY returns, residuals `e`; score = `sum(e[-12:-1]) / stdev(e[-12:-1])` (ddof 1).
  - `sort_key(variant: float, today: float) -> tuple[float, float]`: rank best first on the variant, ties by today's score.
  - `book_picks(ranked: list[str], held: set[str], cooldown: dict[str, int], top_n: int = 10, keep_rank: int = 35) -> list[str]` (used by Task 3): holdings ranked <= 35 kept, then best-ranked names not held and not in `cooldown` fill up to 10; best first.
  - `advance_cooldown(cooldown: dict[str, int], sold: set[str], reviews: int = 4) -> dict[str, int]`: decrement every count, drop zeros, then set each sold name to `reviews`.
  - `self_check() -> None`

- [ ] **Step 1: Write `self_check()` with hand-worked asserts**, stubs raising `NotImplementedError`:
  - `vol_scaled` on a series with known 126-day vol equals `row.score / vol` (build `row` with `score_stock` on a synthetic 300-session series).
  - `high_52w([..., 10, 12, 9, 11])` with max 12 over the window = `11 / 12`.
  - `r3_skip` equals today's blend with R(3) = `closes[-22] / closes[-85] - 1`.
  - `residual`: a stock = 2 x SPY monthly returns exactly -> std 0 -> None; a stock with 756 closes -> None; a stock = SPY + a fixed +1% per month on months 2-12 and noise elsewhere -> positive score.
  - Non-finite: `vol_scaled` with zero volatility -> None.
  - `sorted([...], key=sort_key)` puts two `high_52w` = 1.0 names in today's-score order.
  - `book_picks`: held `X` at rank 30 kept, held `Y` at rank 40 dropped, `Z` in cooldown skipped, fill to 10 from the top; `advance_cooldown({"Z": 1}, {"Y"})` == `{"Y": 4}`.
- [ ] **Step 2:** Run `uv run python scripts/bull_bear/score_variants.py --self-check`. Expected: `NotImplementedError`.
- [ ] **Step 3:** Implement the functions above.
- [ ] **Step 4:** Rerun. Expected `self-check ok`, exit 0. `uv run ruff check scripts/bull_bear && uv run ruff format --check scripts/bull_bear` exit 0; `uv run pyright scripts/bull_bear/score_variants.py` 0 errors.
- [ ] **Step 5:** On a new branch `research/momentum-score-variants` (never `main`): `git add -f docs/superpowers/specs/2026-11-28-momentum-score-variants-design.md docs/superpowers/plans/2026-11-28-momentum-score-variants.md scripts/bull_bear/score_variants.py`, commit `research(bull_bear): momentum score variants, pure scores and self-check`.

### Task 2: Ranking, both windows, report, verdict

**Files:**
- Modify: `scripts/bull_bear/score_variants.py`

**Interfaces:**
- Consumes: Task 1; from `rank_only_stress`: `load_data`, `simulate`, `rebalance_days`, `max_drawdown`, `month_universe`, `SLEEVE`, `CLOSES`, `CONFIG`, `TOP_N`, `WEIGHTS[0]`, `WINDOWS`.
- Produces:
  - `ranked(day, held, listed, frames, sessions, variant) -> list[str]`: same slicing and held exemption as `ranked_symbols`, but reads up to 800 sessions before `day`, scores `today` with `score_stock` on the last 300, then sorts the survivors by `sort_key(variant_score(...), row.score)`, dropping None.
  - `window_stats(equity: pd.Series) -> dict[str, float]`: `cagr`, `sharpe`, `max_dd`, the worst fall inside each `WINDOWS` entry overlapping the window.
  - `passes(stats: dict, base: dict) -> bool`: `stats["cagr"] > base["cagr"] and stats["sharpe"] >= base["sharpe"]`.
  - `main()`: development run of all five variants, table, passers; holdout run of `today` + passers only, table; final verdict line per variant (`KEPT`, `FAILED development`, `FAILED holdout`). Also prints per variant: mean ranked count, turnover (mean share of the 10 names replaced per rebalance). Saves `cache/stress_2005/score_variants_equity.csv`.

- [ ] **Step 1:** Add to `self_check()`: `passes` on (CAGR higher, Sharpe equal) -> True, (CAGR higher, Sharpe lower) -> False, (CAGR equal) -> False; `ranked` on the stress script's synthetic `UP`/`FLAT`/`HALT` frames (plus a SPY frame) with `today` returns `["UP", "FLAT"]` (same as `ranked_symbols`) and a crash bar on the trade day is not read.
- [ ] **Step 2:** Run `--self-check`. Expected: the new asserts fail.
- [ ] **Step 3:** Implement. Cache picks per (variant, day) as the stress script does.
- [ ] **Step 4:** Rerun `--self-check` (ok), lint/format/pyright as in Task 1.
- [ ] **Step 5:** Run `uv run python scripts/bull_bear/score_variants.py` once (background; minutes to tens of minutes). Expected: both tables and one verdict line per variant. The `today` development row must match the stress book's 9.8% curve restricted to 2005-2015 started at 1.0 within 1 point of CAGR (sanity check; if not, stop and report).
- [ ] **Step 6:** Append `## Results` to the spec (tables, verdicts, caveats). Write the replay record `docs/superpowers/replays/2026-11-28-momentum-score-variants.md`, a memory note, and update TODO Bull Bear item 6 only if the user asks (never touch TODO.md unasked). Commit `research(bull_bear): momentum score variants results`.

### Task 3 (only if a variant is KEPT): bull_bear's real book rules

**Files:**
- Modify: `scripts/bull_bear/score_variants.py`

**Interfaces:**
- Consumes: Task 1 `book_picks`, `advance_cooldown`; Task 2 `ranked`, `window_stats`, `passes`.
- Produces: `main(--book-rules)`: `today` and each KEPT variant on both windows with picks = `book_picks(ranked(...), held, cooldown)`; cooldown advanced each rebalance with the names sold; same pass rule, verdict line per variant.

- [ ] **Step 1:** Add to `self_check()`: a 3-rebalance synthetic run where a name sold at rebalance 1 cannot be bought at rebalances 2-5 and can at 6.
- [ ] **Step 2:** Run `--self-check`. Expected: fails.
- [ ] **Step 3:** Implement the `--book-rules` path (the cooldown state lives in the `picks` closure, keyed per variant).
- [ ] **Step 4:** `--self-check` ok, lint/format/pyright.
- [ ] **Step 5:** Run `--book-rules` once. Append the results to the spec and the replay record. Commit.

### Final passes (after the last task run)

- [ ] `/simplify`, code-simplifier, stage, `/ponytail:ponytail-review` on the staged diff; fix every "Must fix".
- [ ] Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest --cov`, `uv run interrogate -v`; report the numbers (known on `main`: 2 news-default test failures and an `interrogate` crash, per the stress plan; re-check whether they still occur).
- [ ] No push, PR or merge without the user's go-ahead.

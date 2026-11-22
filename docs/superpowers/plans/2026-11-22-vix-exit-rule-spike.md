# VIX exit-rule spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One throwaway script that scores four fast exits after a VIX-alone switch on money (dev 2016-2024), then
tests the best one once on 2008-2015.

**Architecture:** `scripts/regime/vix_exit_spike.py`, modes `check` (assert self-check on synthetic series), `dev`,
`test`. One state-machine function holds entry / price exit / calm exit / re-arm / failed rally; exit signals are
precomputed boolean series (E1, E3, E4) or a rebound threshold (E2, needs the episode low).

**Tech Stack:** Python 3.14, pandas, numpy, yfinance (already used by the sibling spikes). Run with
`uv run --with jumpmodels` because `vix_jump_spike.py` imports `jumpmodels` at module level.

**Spec:** `docs/superpowers/specs/2026-11-22-vix-exit-rule-spike-design.md`

## Global Constraints

- Every value used before the open of session t comes from closes up to t-1 (signals shifted by one session).
- Entry trigger: VIX/VIX3M >= 0.95 or VIXY 5-session return >= +10%; VIXY NaN (before 2011-01-04) = false.
- VIX calm exit counted exactly as `run_switch()` (`scripts/news/early_regime_spike.py:175`): counter reset on the
  switch session, 3 consecutive defensive sessions with VIX/VIX3M (t-1) < 0.90.
- Windows: dev 2016-01-04..2024-12-31; test 2008-01-02..2015-12-31 run once; 2025-2026 never used.
- 10 bp per switch at the open (`portfolio()` from `vix_jump_spike.py`, unchanged).
- No parameter or variant change after the first `dev` run. No new dependency, no code in `src/`, no tests in `tests/`.

## Review Focus

- VIXY NaN before 2011: comparison must be false, not an error or NaN-propagating state (test window).
- Failed-rally low must include the close of the exit morning (the last session's close), per the spec.
- E2 low starts at the close before the episode's first session (`spy_prev` on the first defensive session).
- Entry and price exit on consecutive mornings: an exit fires at the earliest on the morning after the switch session.
- `test` must refuse a second run (`cache/regime/vix_exit_test.done`) and refuse when no variant passed dev.

---

### Task 1: State machine and its self-check

**Files:**
- Create: `scripts/regime/vix_exit_spike.py`

**Interfaces:**
- Produces: `run_exit(trigger: pd.Series, calm: pd.Series, spy_prev: pd.Series, exit_sig: pd.Series | None = None, rebound: float | None = None) -> tuple[pd.Series, list[date]]`
  returns (defensive flag per session, sessions where a **price** exit fired). All inputs indexed by session t, values
  known before its open (`spy_prev[t]` = SPY close of t-1). Exactly one of `exit_sig` / `rebound` is given.
- Produces: `check() -> None` (asserts; prints `self-check OK`).

- [ ] **Step 1: Write `check()` with these synthetic cases** (10-20 sessions each, `pd.Series` on a range of dates):
  1. trigger True on s1..s6, `exit_sig` True on s3 -> defensive s1..s2, normal s3..s6 (disarmed, trigger still on); price exits == [s3].
  2. same, trigger False on s7, True on s8 -> defensive again from s8 (re-armed on s7).
  3. disarmed after case 1, `spy_prev` on s5 below the episode's low -> defensive from s5 (failed rally).
  4. trigger True on s1 only, calm True on s1..s4 -> defensive s1..s3, normal s4 (calm on the switch session not counted; matches `run_switch(None, trigger, calm)`).
  5. disarmed, on s5 trigger False **and** `spy_prev` below the low -> failed rally fires on s5 (`armed` updated after steps 1-2).
  6. `rebound=0.04`: `spy_prev` 100 on s1 (switch), 97 on s2, 100.9 on s3 (exit, 97 x 1.04 = 100.88) -> normal from s3.
  7. trigger built from a VIXY series with NaN -> no exception, and `trigger` there is False.
- [ ] **Step 2:** `uv run --with jumpmodels python scripts/regime/vix_exit_spike.py check` -> fails (`run_exit` missing).
- [ ] **Step 3: Implement `run_exit`** as one loop over sessions, the spec's order: (1) if normal: switch if `armed` and trigger, else if not `armed` and `spy_prev[t] < prev_low`; (2) if defensive since t-1: price exit (`exit_sig[t]`, or `spy_prev[t] >= (1 + rebound) * episode_low`) or calm counter reaches 3; (3) update `armed`. `episode_low` = min of `spy_prev` from the first defensive session through the current morning; on exit it becomes `prev_low` (so it includes the exit morning's `spy_prev`). Calm counter: reset on the switch session, `+1` when defensive and calm, else 0.
- [ ] **Step 4:** run `check` -> `self-check OK`.
- [ ] **Step 5:** `uv run ruff check scripts/regime/vix_exit_spike.py && uv run ruff format --check scripts/regime/vix_exit_spike.py` -> exit 0.

### Task 2: Signals and the `dev` run

**Files:**
- Modify: `scripts/regime/vix_exit_spike.py`

**Interfaces:**
- Consumes: `run_exit`; from `vix_jump_spike.py`: `open_close() -> DataFrame`, `portfolio(state, px) -> Series`, `stats(r, years) -> str`; from `leaders_crack_spike.py`: `falls(spy) -> list[(high, low, depth)]`, `evaluate(name, state, spy, fall_list, years) -> (median_dd, false_per_year)`, `run_switch`.
- Produces: `market() -> DataFrame` (SPY open/high/low/close, `vix`, `vix3m`, `VIXY`, from 2007-06-01, date index);
  `inputs(m) -> (trigger, calm, spy_prev)` shifted one session; `exit_signals(m) -> dict[str, Series | float]` with keys
  `E1 price reclaim`, `E2 rebound off the low` (value 0.04), `E3 volatility compression`, `E4 DMI`;
  `run_window(mode: str) -> dict[str, dict]` (per variant: state, price exits, sharpe, total, mdd); `choose(results) -> str | None`.

- [ ] **Step 1: Signals** (row t from closes up to t-1): E1 `close > close.shift(1).rolling(5).max()`; E3 `logret.rolling(5).std() < logret.rolling(20).std()` and `close > close.rolling(5).mean()`; E4 Wilder +DI(14) / -DI(14) on high, low, close (first smoothed value = simple mean of the first 14), `+DI > -DI`; all `.shift(1)`.
- [ ] **Step 2: `dev` output**, VIX alone + E1..E4 on the dev window: (a) money table (total, CAGR, max DD, Sharpe, Calmar, switches/yr, defensive share) with buy and hold, via `portfolio` / `stats`; (b) `evaluate()` entry table per variant; (c) per fall defensive at its trough: sessions trough -> back to SPY, SPY % from trough then; (d) whipsaws per variant (spec definition); (e) pass bar per variant: Sharpe > VIX alone, total > VIX alone, max DD >= -20% -> PASS/FAIL; chosen = highest Sharpe among PASS (tie: Calmar), or `none -> verdict NO`.
- [ ] **Step 3:** `uv run --with jumpmodels python scripts/regime/vix_exit_spike.py dev`; check that the VIX-alone row matches the spec of the jump spike (dev: +118.8%, Sharpe 1.04, max DD -14.2%). A mismatch is a bug to find before reading the variants.
- [ ] **Step 4:** ruff check + format check -> exit 0; `/ponytail:ponytail-review` on the script; fix Must fix; commit
  `research: VIX exit-rule spike script (check, dev)`.

### Task 3: Record dev, then the single `test` run

**Files:**
- Modify: `scripts/regime/vix_exit_spike.py`, `docs/superpowers/specs/2026-11-22-vix-exit-rule-spike-design.md`

- [ ] **Step 1:** write the dev results and the verdict into the spec (`## Results`), status line updated. Commit.
- [ ] **Step 2: `test` mode**: exits with a message if `cache/regime/vix_exit_test.done` exists, or if `choose()` on dev returns None; otherwise runs the 2008-01-02..2015-12-31 window for the chosen variant, VIX alone and buy and hold (same outputs as dev), prints PASS/FAIL (Sharpe and total above VIX alone), then writes the marker.
- [ ] **Step 3:** only if dev chose a variant: **ask the user before running `test`** (one shot). Then run it.
- [ ] **Step 4:** record the test result in the spec; update the memory note `vix-jump-model-tested.md` or add a new one; gate (ruff) + review; commit.

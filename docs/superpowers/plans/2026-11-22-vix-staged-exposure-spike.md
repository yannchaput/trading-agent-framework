# VIX staged-exposure spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One throwaway script that scores partial SPY exposure after a VIX switch (B thirds, C half until confirmed)
against a fixed-blend line (A) on dev 2016-2024, then the better one once on 2008-2015.

**Architecture:** `scripts/regime/vix_staged_spike.py`, modes `check`, `dev`, `test`. One state machine returns the SPY
exposure per session for B (and C when a 60-session-high series is passed); one portfolio function prices any exposure
series. Inputs come from `vix_exit_spike.py`, so the entry and signals are the ones already reviewed.

**Tech Stack:** Python 3.14, pandas, numpy, yfinance; `uv run --with jumpmodels` (sibling imports need it).

**Spec:** `docs/superpowers/specs/2026-11-22-vix-staged-exposure-spike-design.md`

## Global Constraints

- Row t uses closes up to t-1 (inputs from `vix_exit_spike.py` are already lagged; the 60-session high is `close.rolling(60).max().shift(1)`).
- Tranches 1/3 each; C floor 0.5; confirmation = trigger on t and t-1, or `spy_prev <= 0.97 x high60`; cost 0.0010 x |change in e|.
- A points w = 0, 0.25, 0.5, 0.75, 1; pass = total above the A line at the same max DD, and above VIX alone.
- Windows: dev 2016-01-04..2024-12-31; test 2008-01-02..2015-12-31 once (guard `cache/regime/vix_staged_test.done`); 2025-2026 never.
- No parameter change after the first `dev` run. No new dependency, no `src/` code, no `tests/` files.

## Review Focus

- Calm exit and new low on the same morning: the calm exit wins (e = 1), so B with no E1/E3 equals VIX alone.
- A tranche sent out by a new low returns on the next morning its signal is true (level, not crossing).
- C confirmed on the switch morning itself when SPY is already 3% below its 60-session high (e = 0, not 0.5).
- Exposure portfolio at e in {0, 1} must equal `portfolio()` to the last digit, including the first session.
- A line when the variant's max DD is better than VIX alone's or worse than buy and hold's (spec: compare with VIX alone / fail).

---

### Task 1: State machine, exposure portfolio, self-check

**Files:** Create `scripts/regime/vix_staged_spike.py`

**Interfaces:**
- Consumes (from `vix_exit_spike.py`): `market() -> (m, px)`, `entry_trigger(m) -> Series[bool]`, `exit_signals(m) -> dict`, `metrics(r, years) -> dict`, `run_switch`, `falls`; (from `vix_jump_spike.py`) `portfolio(state, px)`.
- Produces: `run_staged(trigger, calm, spy_prev, e1, e3, high60: pd.Series | None = None) -> pd.Series` (SPY exposure in [0, 1] per session; C when `high60` is given);
  `portfolio_exposure(e: pd.Series, px: pd.DataFrame) -> pd.Series` (daily returns).

- [ ] **Step 1: `check()` cases** (12 synthetic sessions, numbered s1..s12, flat `spy_prev` 100 unless stated):
  1. E1, E3 never true -> `1 - run_staged(...)` equals `run_switch(None, trigger, calm)` as 0/1.
  2. trigger s1; E1 true from s3, E3 true from s5; calm s7..s9 -> e: s1-s2 = 0, s3-s4 = 1/3, s5-s8 = 2/3, s9 on = 1.
  3. as 2 but `spy_prev` 99 on s6 (new low) -> e s6 = 0, s7 = 2/3 again (E1 and E3 still true: level rule).
  4. new low on s9 and calm reaching 3 on s9 -> e s9 = 1.
  5. C: trigger s1 only, `high60` 100 -> e s1 = 0.5 (unconfirmed); trigger s1 and s2 -> e s2 = 0; trigger s1 with `spy_prev` s1 = 96.9 -> e s1 = 0.
  6. `portfolio_exposure` on a random synthetic px with e in {0, 1} equals `portfolio(e == 0, px)`; a 0 -> 0.5 change costs 5 bp.
- [ ] **Step 2:** `uv run --with jumpmodels python scripts/regime/vix_staged_spike.py check` -> fails (`run_staged` missing).
- [ ] **Step 3: Implement** `run_staged` as one loop in the spec's order (new low / tranches / calm exit checked every morning; C floor applied last), and `portfolio_exposure` with the spec's formula.
- [ ] **Step 4:** `check` -> `self-check OK`; ruff check + format check -> exit 0; commit.

### Task 2: `dev` run

**Files:** Modify `scripts/regime/vix_staged_spike.py`

**Interfaces:**
- Produces: `exposures(m, days) -> dict[str, Series]` (A w=0..1 as `A 0%`..`A 100%`, `B thirds`, `C half until confirmed`);
  `a_line(points: list[tuple[float, float]], mdd: float) -> float | None` (None = worse than buy and hold);
  `report(mode) -> dict` (prints and returns per-variant metrics).

- [ ] **Step 1:** table per variant: total, CAGR, max DD, Sharpe, average exposure, turnover/yr; `A 100%` must print +118.8% / Sharpe 1.04 / -14.2% on dev (stop and debug if not).
- [ ] **Step 2:** leak split per variant (lost log return vs SHV x (1 - e): false / fall / after trough, spec definitions).
- [ ] **Step 3:** pass bar for B and C (`total > a_line(...)` and `total > VIX alone`), chosen = higher total or `none -> verdict NO`.
- [ ] **Step 4:** `uv run --with jumpmodels python scripts/regime/vix_staged_spike.py dev`; ruff; `/ponytail:ponytail-review`; fix Must fix; commit.

### Task 3: Record, then the single `test` run

- [ ] **Step 1:** dev results and verdict into the spec; commit.
- [ ] **Step 2:** only if a variant was chosen: `test` mode (guard file; refuses if `report("dev")` chooses none; same outputs and bar on 2008-2015). **Ask the user before running it.**
- [ ] **Step 3:** record the test result in the spec; memory note; gate (ruff) + review; commit.

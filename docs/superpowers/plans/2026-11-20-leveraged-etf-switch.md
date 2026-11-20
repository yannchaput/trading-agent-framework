# Leveraged ETF Switch Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One research script that simulates UPRO and TQQQ from 1993, runs rules R2 (graded regime), R3 (volatility-scaled) and R4 (200-day trend) against baselines B1-B3, prints the spec's tables and the pass bar, and a committed record of the single run.

**Architecture:** `scripts/research/leveraged_switch.py`, untracked (research code is not kept, as in the earlier studies). Pure functions over pandas Series indexed by session date: data -> synthetic legs -> per-rule daily target weights -> one engine (lag, band, cost) -> metrics. The production regime is imported from `core/regime.py`, never changed. An assert-based `self_check()` runs before every run; the repository test suite is not touched.

**Tech Stack:** Python 3.14, pandas, numpy, yfinance (Yahoo), `trading_agent_framework.core.regime`, `trading_agent_framework.core.regime_scorecard.zigzag`.

**Spec:** `docs/superpowers/specs/2026-11-20-leveraged-etf-switch-design.md`

## Global Constraints

- Window 1993-01-04 to the last session downloaded; sub-windows 1993-01-04..2015-12-31 and 2016-01-04..end; warm-up data before 1993.
- Synthetic 3x per session: `3 r - 2 rf - ER / 252`; ER 0.0091 (UPRO), 0.0084 (TQQQ); `rf = ^IRX / 100 / 252`.
- Underlyings: `^SP500TR` (UPRO), `^NDX` (TQQQ). Regime inputs: `^GSPC`, `^VIX`, `^VIX3M` (1e9 before VIX3M exists).
- R3 target vol 0.30, `sigma20` = std of the last 20 daily underlying returns x sqrt(252); band 0.10 (R3 only).
- R4: 200-session SMA of `^GSPC` (UPRO) / `^NDX` (TQQQ).
- Lag: a target computed from closes up to session t is held from the close of t+1 (it earns session t+2's return first).
- Cost 0.0010 x traded fraction of the account, one way, on every change.
- Bar: Calmar > B3's on full and both sub-windows, and worst calendar year > B1's. One run; nothing changed after it.
- Output saved under `logs/research/leveraged_switch/`.

## Review Focus

- A Yahoo series with gaps (holidays differ between `^IRX`, `^VIX` and the indices): align on the underlying's sessions, forward-fill `^IRX`/`^VIX`/`^VIX3M` at most 5 sessions, and fail loudly if a gap remains in the window.
- Sessions with no regime reading (the first `min_bars`): the window must start after them; assert R2 has a target on every window session.
- Sub-window metrics must start fresh (wealth 1 at the sub-window's first session), not read off the full-window curve.
- A fall still open at the end of the data (zigzag records it): its 63-session rebound window runs past the data; skip falls with fewer than 63 sessions after the trough.
- Worst calendar year for the first and last (partial) years: computed on the sessions present, labelled as partial in the year table.

---

### Task 1: Data and synthetic legs

**Files:**
- Create: `scripts/research/leveraged_switch.py`

**Interfaces:**
- Produces: `load_data() -> pd.DataFrame` (index: session dates, columns `spx_tr, ndx, gspc, vix, vix3m, rf`, rf already per session); `legs(underlying: pd.Series, rf: pd.Series, er: float) -> pd.DataFrame` (columns `x3, x1, cash`: daily simple returns); `validate(synthetic_x3: pd.Series, ticker: str) -> str` (one line: CAGR synthetic vs real and daily-return correlation over the real ETF's life).

- [ ] **Step 1: Write `self_check()` with the synthetic-leg case**

```python
r = pd.Series([0.01, -0.02], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
rf = pd.Series([0.0001, 0.0001], index=r.index)
x = legs(r, rf, er=0.0091)
assert abs(x["x3"].iloc[0] - (0.03 - 0.0002 - 0.0091 / 252)) < 1e-12
assert x["x1"].equals(r) and x["cash"].equals(rf)
```

- [ ] **Step 2: Run it, expect a NameError for `legs`**

Run: `uv run python scripts/research/leveraged_switch.py --self-check`

- [ ] **Step 3: Implement `load_data`, `legs`, `validate`**

One `yfinance.download` of `^SP500TR ^NDX ^GSPC ^VIX ^VIX3M ^IRX` from 1985-01-01, `auto_adjust=False`, `Close`; align on `^SP500TR`'s sessions (Review Focus 1); `vix3m` = 1e9 where missing before its first value. `validate` downloads UPRO/TQQQ `Adj Close`.

- [ ] **Step 4: Run `--self-check`, expect `self-check ok`**

### Task 2: Rule targets

**Files:**
- Modify: `scripts/research/leveraged_switch.py`

**Interfaces:**
- Consumes: `load_data()`, `legs()` from Task 1.
- Produces: each returns `pd.DataFrame` of target weights (columns `x3, x1, cash`, rows summing to 1) indexed by the session whose close they use:
  `r2_graded(data: pd.DataFrame) -> pd.DataFrame`, `r3_vol(underlying_close: pd.Series) -> pd.DataFrame`, `r4_trend(signal_close: pd.Series) -> pd.DataFrame` (both take close levels, not returns), `b1(index) / b2(index) / b3_monthly(index) -> pd.DataFrame` (B3: 1/3 x3, 2/3 x1, with a `rebalance` bool column True on each month's first session).

- [ ] **Step 1: Add self-check cases**

```python
# R4: above the 200-day mean -> all 3x, below -> all cash
up = pd.Series(np.r_[np.ones(199), 2.0], index=pd.bdate_range("2000-01-03", periods=200))
assert r4_trend(up).iloc[-1].to_dict() == {"x3": 1.0, "x1": 0.0, "cash": 0.0}
# R3: 20-day sigma of 0.20 annualised -> 0.30 / 0.60 = 0.5 in 3x
closes = (1 + pd.Series(np.tile([0.2, -0.2], 15) / np.sqrt(252))).cumprod()  # daily returns of +-0.2/sqrt(252)
w = r3_vol(closes)
assert abs(w["x3"].iloc[-1] - 0.5) < 0.02
```
R2: assert the mapping 1 -> (1, 0, 0), 0 -> (0, 1, 0), -1 -> (0, 0, 1) on a stub reading dict.

- [ ] **Step 2: Run `--self-check`, expect failures for the missing functions**

- [ ] **Step 3: Implement the six functions**

R2 calls `regime_by_session(sessions, gspc, vix, vix3m)` once; the reading for session t+1 (closes strictly before t+1) is the target indexed at t. R3/R4 use pandas rolling windows (20 and 200). Rows before a rule's warm-up are NaN.

- [ ] **Step 4: Run `--self-check`, expect `self-check ok`**

### Task 3: Engine and metrics

**Files:**
- Modify: `scripts/research/leveraged_switch.py`

**Interfaces:**
- Consumes: targets from Task 2, legs from Task 1.
- Produces: `simulate(targets: pd.DataFrame, legs: pd.DataFrame, start: date, end: date, band: float = 0.0) -> pd.Series` (daily wealth from 1.0 at `start`); `metrics(wealth: pd.Series) -> dict` with keys `multiple, cagr, max_dd, calmar, worst_year, losing_years, changes`; `rebound(wealth, b1_wealth, underlying_close, start, end) -> tuple[float, float, int]` (median 63-session return of the rule, of B1, number of falls).

- [ ] **Step 1: Add self-check cases**

```python
# lag: a target set from the close of day 0 earns day 2's return first
# (legs x3 = +10% every day, target flips from cash to x3 at index 0)
# -> the trade at the close of day 1 costs 0.001: wealth[1] == 0.999, wealth[2] == 0.999 * 1.10
# Calmar: wealth 1 -> 2 over 252 sessions with a 50% drawdown -> cagr 1.0, max_dd 0.5, calmar 2.0
# band: an R3 target moving 0.05 does not trade; 0.12 does
```

- [ ] **Step 2: Run `--self-check`, expect failures**

- [ ] **Step 3: Implement `simulate`, `metrics`, `rebound`**

Weights drift with each leg's return between trades; a trade happens when the lagged target differs from the held weights (any amount for R2/R4/B1/B2, on `rebalance` days for B3, by `band` or more for R3); cost = 0.001 x half the sum of absolute weight changes (the one-way traded fraction: all-cash to all-3x is 1.0), charged at the trade's close. `rebound` uses `zigzag(sessions, closes, 0.15)` on the underlying within the window, skipping falls with fewer than 63 sessions after the trough.

- [ ] **Step 4: Run `--self-check`, expect `self-check ok`**

### Task 4: Run once and record

**Files:**
- Modify: `scripts/research/leveraged_switch.py` (`main`)
- Create: `docs/superpowers/replays/2026-11-20-leveraged-etf-switch.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Implement `main()`**

For UPRO (`spx_tr`, ER 0.0091, R4 on `gspc`) and TQQQ (`ndx`, ER 0.0084, R4 on `ndx`): the validation line, then per window (full, 1993-2015, 2016-end) one row per B1, B2, B3, R2, R3, R4 with the metrics, share of sessions with some 3x and the rebound pair; then a per-year return table on the full window; then `PASS`/`FAIL` per rule per instrument with the failing condition named. Print and write `report.txt`; write `wealth.csv`.

- [ ] **Step 2: Run once**

Run: `uv run python scripts/research/leveraged_switch.py`
Expected: `self-check ok`, two validation lines, the tables, six PASS/FAIL lines. Nothing in the rules changes after this run.

- [ ] **Step 3: Write the record**

Same shape as `2026-11-20-ms-tvtp-exit.md`: question, fixed method (link the spec), validation lines, tables as printed, PASS/FAIL, conclusion, limits (pre-2006 regime handicap, NDX without dividends, synthetic financing without spread, 2016-2026 already seen).

- [ ] **Step 4: Gate and commit the record only**

Run: `uv run ruff check && uv run pytest -q` (expect clean and all pass), `/ponytail:ponytail-review` on the staged record, then:

```bash
git add docs/superpowers/replays/2026-11-20-leveraged-etf-switch.md
git commit -m "docs: leveraged ETF switch research record (code not kept)"
```

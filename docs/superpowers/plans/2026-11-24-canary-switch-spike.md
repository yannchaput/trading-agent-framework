# Canary Switch Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One research script that replays Keller's BAA canary switch (bull mode = bull_bear, bear mode = the BAA
defensive basket) on bull_bear's saved daily equity and on SPY 2008-2026, and prints a pre-registered GO / NO-GO.

**Architecture:** One file, `scripts/regime/canary_switch_spike.py`, modes `check` (assert self-check on synthetic
data, no network) and `run`, built like `scripts/regime/vix_staged_spike.py` and `scripts/earnings/short_side_spike.py`
(module constants, plain functions, `check()` of numbered asserts, `main()` on `sys.argv[1]`). The rule maths are pure
functions on price frames; one `replay` function serves the gated account and every reference row.

**Tech Stack:** Python 3.14, uv, pandas, yfinance (installed).

**Spec:** `docs/superpowers/specs/2026-11-24-canary-switch-spike-design.md`

## Global Constraints

- No change under `src/` or `tests/`; no new dependency; no LLM run. Script only, plus a replay record.
- Worktree `.worktrees/bearish-regime`; git-ignored inputs read from the main checkout:
  `MAIN = Path(git rev-parse --path-format=absolute --git-common-dir).parent`; cache under `MAIN/cache/canary_switch/`.
- bull_bear runs: `MAIN/logs/bull_bear/backtesting/2026-11-05_012455_backtesting/equity.parquet` ("5Y") and
  `.../2026-11-09_224910_backtesting/equity.parquet` ("COVID"), column `portfolio_value`.
- Canaries `SPY, VWO, VEA, BND`; defensive `TIP, DBC, BIL, IEF, TLT, LQD, BND`; reference `SHV`. Yahoo
  `auto_adjust=True` closes everywhere.
- Fast momentum `(12 r1 + 4 r3 + 2 r6 + 1 r12) / 19`; bull only when all four canaries are **> 0** (0 is bear).
- Slow momentum `p0 / mean(p0 and the last 12 month-end closes) - 1`; top 3 at 1/3; a pick below BIL becomes BIL.
- Decision at a month-end close `t`, effective at the close of the next session `t+1`.
- Costs: **10 bps** on the bull_bear side of a trade (5 bps when the bull leg is SPY), **5 bps** on the ETF side.
- Bar: on both windows max drawdown smaller than bull_bear's **and** gain >= 0.85 x bull_bear's gain; and the SPY
  check's compounded return over bear-mode sessions 2008-2026 >= 0. GO needs all.
- Lint: `uv run ruff check` and `uv run ruff format --check` clean on the script and the Markdown.

## Review Focus

1. A decision date with fewer than 12 earlier month-ends or a NaN close for a needed ETF: `decide` raises
   `ValueError`, never a silent bull or bear. Test in Task 1 (assert 6).
2. A decision whose effective session falls after the window's last session: ignored; the latest decision before the
   window start sets the opening holdings. Test in Task 2 (assert 10).
3. BIL in the top 3 and also a replacement: weights still sum to 1 (BIL 2/3). Test in Task 1 (assert 4).
4. A canary momentum of exactly 0: bear. Test in Task 1 (assert 3).
5. bull_bear's first sessions in cash (flat curve) and the first session of a replay: equity equals the start value,
   no NaN. Test in Task 2 (assert 9).

---

### Task 1: The rule maths, with the self-check

**Files:**
- Create: `scripts/regime/canary_switch_spike.py`

**Interfaces:**
- Produces:
  - Constants `MAIN: Path`, `CACHE = MAIN / "cache" / "canary_switch"`, `CANARIES = ("SPY", "VWO", "VEA", "BND")`,
    `DEFENSIVE = ("TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND")`, `BULL = "BULL"` (the bull leg's key),
    `STOCK_COST = 0.001`, `ETF_COST = 0.0005`, `SHARE = 0.85`.
  - `month_ends(days: list[date]) -> list[date]`: the last session of each calendar month in `days`.
  - `fast_momentum(p0: float, past: list[float]) -> float`: `past` = month-end closes before the decision, oldest
    first; `rN = p0 / past[-N] - 1`.
  - `slow_momentum(p0: float, past: list[float]) -> float`: `p0 / mean([p0] + past[-12:]) - 1`.
  - `is_bull(fast: dict[str, float]) -> bool`: every canary strictly above 0.
  - `defensive_weights(slow: dict[str, float]) -> dict[str, float]`: top 3 of `DEFENSIVE` by slow momentum (ties by
    symbol), each pick scoring below `slow["BIL"]` replaced by BIL, 1/3 each, BIL's weights summed.
  - `decide(px: pd.DataFrame, day: date, ends: list[date]) -> dict[str, float]`: the target on `day`: `{BULL: 1.0}` in
    bull mode, else `defensive_weights`. `px` = daily closes (date index, one column per ETF); `p0 = px.loc[day]`,
    `past` = closes on the last 12 month-ends of `ends` strictly before `day` (both scores use these 12 plus
    `p0`). Raises `ValueError` with fewer than 12 earlier month-ends or a NaN among the needed closes.

- [ ] **Step 1: Write the failing self-check**

```python
# 1. month_ends: Jan 30 (Fri) and Feb 27 (Fri) 2026 out of a run of sessions ending Mar 2
# 2. fast_momentum: past = 12 month-ends at 100, p0 = 110 -> 0.10 (every rN is 0.10); p0 = 100 -> 0.0
#    past rising 100..111 (12 values), p0 = 112: (12*(112/111-1) + 4*(112/109-1) + 2*(112/106-1) + (112/100-1)) / 19
# 3. is_bull: all four 0.01 -> True; one exactly 0.0 -> False; one -0.01 -> False
# 4. defensive_weights: TLT 0.05, IEF 0.04, TIP 0.03, BIL 0.01, others < 0 -> TLT, IEF, TIP at 1/3;
#    TLT 0.05, BIL 0.01, IEF 0.00, others < 0 -> TLT 1/3, BIL 2/3 (IEF below BIL is replaced; BIL itself in the top 3)
#    weights always sum to 1
# 5. slow_momentum: past = 12 values at 100, p0 = 113 -> 113 / ((113 + 1200) / 13) - 1
# 6. decide raises ValueError with 11 earlier month-ends, and with a NaN close for VEA; with every canary rising it
#    returns {BULL: 1.0}; with VWO falling it returns the defensive weights
print("self-check OK")
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python scripts/regime/canary_switch_spike.py check`
Expected: FAIL (`NameError` on the first missing function).

- [ ] **Step 3: Implement the constants and the functions in "Produces"**

- [ ] **Step 4: Run the self-check, then lint**

Run: `uv run python scripts/regime/canary_switch_spike.py check && uv run ruff check scripts/regime/canary_switch_spike.py
&& uv run ruff format --check scripts/regime/canary_switch_spike.py`
Expected: `self-check OK`, ruff exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/regime/canary_switch_spike.py
git commit -m "research: canary switch spike, BAA rule maths"
```

---

### Task 2: Replay, reference rows, the bar and `run` mode

**Files:**
- Modify: `scripts/regime/canary_switch_spike.py`

**Interfaces:**
- Consumes: Task 1's `month_ends`, `decide`, constants.
- Produces:
  - `effective_targets(px: pd.DataFrame, decision_days: list[date], sessions: list[date]) -> dict[date, dict[str,
    float]]`: each decision `decide(px, t, ends)` keyed by the first session after `t` in `sessions`; a decision with
    no later session is dropped; the latest decision on or before `sessions[0]` is keyed by `sessions[0]` (opening
    holdings). `ends = month_ends(list(px.index))`.
  - `replay(bull: pd.Series, px: pd.DataFrame, sessions: list[date], targets: dict[date, dict[str, float]],
    bull_cost: float = STOCK_COST) -> pd.DataFrame`: columns `equity` (starts at `bull.iloc[0]`) and `bear` (True on
    sessions whose holdings came from a target without `BULL`). Per session: holdings grow by each asset's close to
    close return (`BULL` from `bull`, ETFs from `px`), then a target keyed by that session rebalances them, paying
    `bull_cost x |change of BULL| + ETF_COST x sum |change of each ETF|`. The opening holdings pay `ETF_COST` on their
    ETF part only.
  - `max_drawdown(equity: pd.Series) -> float` (negative) and `gain(equity: pd.Series) -> float` (`last / first - 1`).
  - `bear_return(rows: pd.DataFrame) -> float`: compounded daily return of `equity` over the sessions with `bear`
    True.
  - `passes(gated: dict[str, tuple[float, float]], alone: dict[str, tuple[float, float]], bear_ret: float) -> bool`:
    `(gain, max_dd)` per window; the spec's three conditions.
  - `report() -> bool`: loads the data (Yahoo ETFs from 2006-01-01 cached to `CACHE / "etfs.pkl"`; the two equity
    files, index converted to session dates), prints every row and returns the verdict.

- [ ] **Step 1: Add failing asserts to `check()`**

```python
# 7. replay, bull mode only, bull curve 100 -> 110 -> 121: equity 100, 110, 121, never "bear"
# 8. replay, a switch at session 2 from BULL to {"TLT": 1.0} with TLT flat: cost 0.001 x 110 + 0.0005 x 110 taken at
#    session 2; equity flat afterwards while bull keeps rising; "bear" True from session 2
# 9. replay with a flat bull curve (cash days) and no ETF in the opening target: equity == start on every session, no NaN
# 10. effective_targets: a month-end decision on the window's last session is dropped; one before sessions[0] keys
#     sessions[0]; a decision at t keys the next session after t
# 11. max_drawdown([100, 120, 90, 130]) == -0.25; gain == 0.30; bear_return over two bear sessions of +10% and -10% == -0.01
# 12. passes: both windows smaller drawdown and gain >= 0.85 x, bear_ret 0.0 -> True; gain 0.84 x on one window -> False;
#     drawdown equal (not smaller) on one window -> False; bear_ret -0.001 -> False
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run python scripts/regime/canary_switch_spike.py check`
Expected: FAIL on assert 7.

- [ ] **Step 3: Implement the functions in "Produces" and `run` in `main()`**

`report` prints, per bull_bear window: the monthly gated account (the bar's cell), the Tuesday-check row (information;
decision days = Tuesdays of the ETF calendar), bull_bear alone, 70% bull_bear / 30% SHV (target `{BULL: 0.7, "SHV":
0.3}` at every month-end), SPY buy and hold; gain and max drawdown for each; months in bear mode; each bear-mode
stretch (start, end, gated return, bull_bear return). Then the SPY check from the first month-end where `decide`
succeeds to the last ETF session: canary on SPY (`bull` = SPY closes, `bull_cost = ETF_COST`), SPY, 60% SPY / 40% IEF
(monthly), the bear-mode return overall and split at 2022-08-01 (the paper's out-of-sample start). Last: the three
conditions with PASS / FAIL and `GO` or `NO`.

- [ ] **Step 4: Run the self-check and lint**

Run: `uv run python scripts/regime/canary_switch_spike.py check && uv run ruff check scripts/regime/canary_switch_spike.py
&& uv run ruff format --check scripts/regime/canary_switch_spike.py`
Expected: `self-check OK`, ruff exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/regime/canary_switch_spike.py
git commit -m "research: canary switch spike, replay, references and run mode"
```

---

### Task 3: The run and the record

**Files:**
- Create: `docs/superpowers/replays/2026-11-24-canary-switch.md`

- [ ] **Step 1: Run once**

Run: `uv run python scripts/regime/canary_switch_spike.py run > logs/canary_switch_run.txt 2>&1; tail -80
logs/canary_switch_run.txt`
Expected: both window tables, the SPY check, the three conditions and `GO` or `NO`.

- [ ] **Step 2: Write the record and commit it**

The record: the question, the fixed bar, the data and its biases (bull_bear curves flattered by universe and window
end; two index falls; no held-out), every table of the run, the switch dates around each fall, the verdict.

```bash
git add docs/superpowers/replays/2026-11-24-canary-switch.md
git commit -m "research: canary switch spike, result"
```

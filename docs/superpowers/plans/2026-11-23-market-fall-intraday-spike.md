# Market-fall intraday spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One research script that says whether holding SH / SDS / SPXU from open to close on `in_market_fall` days
makes money on 2016-2024, then scores the chosen setting once on 2025-2026.

**Architecture:** A single file, `scripts/regime/fall_intraday_spike.py`, in the style of the sibling
`scripts/regime/*_spike.py` files: modes `check` (assert-based self-check on synthetic data), `dev`, `test` (one shot,
guard file). Daily signal from Yahoo closes through `utils.helpers.in_market_fall`; minute bars from Alpaca, cached.

**Tech Stack:** Python 3.14, `uv`, pandas, yfinance, alpaca-py (`StockHistoricalDataClient`), python-dotenv; all
already project dependencies.

**Spec:** `docs/superpowers/specs/2026-11-23-market-fall-intraday-spike-design.md`

## Global Constraints

- No change under `src/` or `tests/`; no new dependency. Script only, plus a replay record.
- Signal for session `d` = `in_market_fall(vix[:d], vix3m[:d], vixy[:d])` on Yahoo `^VIX`, `^VIX3M`, `VIXY` closes from
  2008-01-02, **excluding `d`** (the full prefix each day: exact, a few seconds).
- Minute bars: Alpaca SIP feed, `Adjustment.ALL`, credentials `ALPACA_DATA_API_KEY/SECRET` loaded with
  `config.env.load_strategy_env("market_fall", "backtesting")` (falls back to `env/.env`). Cache:
  `cache/fall_intraday/<SYMBOL>_<YEAR>.parquet`.
- Regular session only, converted to `America/New_York`; entry = first bar's open, exit = **last bar of the session**
  (13:00 early closes included).
- Account 5,000, fractional shares (6 decimals), costs 3 bps per side, trailing stop rules exactly as the spec.
- Dev 2016-01-04 to 2024-12-31; held-out 2025-01-02 to the last complete session, run once, guard file
  `cache/fall_intraday/test.done`.
- Bar, grid and choice rule: copied from the spec, not changed after the first run.
- `uv run ruff check scripts/regime/fall_intraday_spike.py` and `uv run ruff format --check` on it exit 0.
- Before each commit, run `/ponytail:ponytail-review` on the staged changes and fix every "Must fix".

## Review Focus

- Early-close sessions (13:00): exit on that session's last bar, never skip the day or read the next day's bars.
- A fall day with no bars for the ETF (data hole): the day is skipped (no trade, equity flat) and counted in the report,
  never filled from another day.
- DST and UTC timestamps: a bar at 13:30 UTC is 09:30 New York in summer only; filter in New York time.
- The trailing stop's own bar: a bar's high must not raise the stop it is tested against in that same bar.
- A gap through the stop (bar opens below it) fills at the bar's open, not at the stop price.

---

### Task 1: Data, signal and trade model with a self-check

**Files:**
- Create: `scripts/regime/fall_intraday_spike.py`

**Interfaces:**
- Produces:
  - `fall_days(closes: pd.DataFrame) -> pd.Series` — `closes` has columns `vix, vix3m, vixy`, date index; returns a bool
    series on the same index, row `d` from rows before `d` (first row False).
  - `daily_closes() -> pd.DataFrame` — Yahoo `^VIX, ^VIX3M, VIXY` closes from 2008-01-02, renamed `vix, vix3m, vixy`.
  - `minute_bars(symbol: str, start: date, end: date) -> pd.DataFrame` — columns `open, high, low, close`, New York
    tz-aware index, regular session only; cached per symbol-year.
  - `inverse_of(bars: pd.DataFrame) -> pd.DataFrame` — the synthetic -1 x SPY: per session, each price `p` becomes
    `2 - p / session_open` (high and low swap).
  - `day_return(bars: pd.DataFrame, stop: float | None, cost: float = 0.0003) -> tuple[float, str]` — one session's
    bars of one instrument; returns the multiplier on the money traded (costs included) and `"close"` or `"stop"`.

- [ ] **Step 1: Write `check()` with these assertions (the self-check, run before any code exists)**

```python
def check() -> None:
    # 1. no look-ahead: changing day k's closes never changes day k's verdict
    closes = synthetic_closes(300)  # random walk, vix/vix3m near 1, a stressed stretch at 150-160
    base = fall_days(closes)
    for k in range(200, 300, 9):
        moved = closes.copy()
        moved.iloc[k] *= [3.0, 1.0, 2.0]
        assert fall_days(moved).iloc[k] == base.iloc[k]
    assert base.iloc[151:160].any()  # the stressed stretch is seen the day after
    # 2. no stop: open 100 -> close 90, cost 3 bps each side
    bars = frame([(100, 101, 99, 100), (100, 100, 90, 90)])
    mult, exit_ = day_return(bars, None)
    assert exit_ == "close" and close_to(mult, 0.9, costs=True)
    # 3. stop 2%: high 105 on bar 1, bar 2 low 102 -> stop at 105 * 0.98 = 102.9, filled at 102.9
    bars = frame([(100, 105, 100, 104), (104, 104, 102, 103), (103, 120, 103, 120)])
    assert day_return(bars, 0.02)[1] == "stop" and close_to(day_return(bars, 0.02)[0], 102.9 / 100, costs=True)
    # 4. a bar's own high does not raise its stop: bar 2 high 110 then low 104 -> not stopped (stop is 98 from bar 1)
    bars = frame([(100, 100, 99, 100), (100, 110, 104, 105)])
    assert day_return(bars, 0.02)[1] == "close"
    # 5. gap through: stop 98, next bar opens 95 -> filled at 95
    bars = frame([(100, 100, 99, 99), (95, 96, 94, 95)])
    assert close_to(day_return(bars, 0.02)[0], 0.95, costs=True)
    # 6. inverse: SPY 100 -> 99 gives 1 -> 1.01
    assert close_to(inverse_of(frame([(100, 100, 99, 99)])).iloc[-1]["close"], 1.01)
    print("self-check OK")
```

`synthetic_closes`, `frame` (rows `(open, high, low, close)` on one New York session, 1-minute index) and the
`close_to` helper (applies the 3 bps both sides when `costs=True`) live in the script next to `check()`.

- [ ] **Step 2: Run it and see it fail**

Run: `uv run python scripts/regime/fall_intraday_spike.py check`
Expected: `NameError` on `fall_days`.

- [ ] **Step 3: Implement the five functions above**

`fall_days`: loop over rows, `in_market_fall(c.vix[:i], c.vix3m[:i], c.vixy[:i])` for `i >= 1`; VIXY NaN before 2011
is passed through (the function reads NaN as missing). `minute_bars`: one `StockBarsRequest` per uncached year
(`TimeFrame.Minute`, `feed=DataFeed.SIP`, `adjustment=Adjustment.ALL`), convert to New York, keep
`09:30 <= t < 16:00`, write parquet. `day_return`: stop level from the highest high of the *earlier* bars (seeded with
the entry open); on each bar, if `open <= level` fill at open, elif `low <= level` fill at level; else exit at the last
bar's close; multiplier `exit * (1 - cost) / (entry * (1 + cost))`.

- [ ] **Step 4: Run the self-check and lint**

Run: `uv run python scripts/regime/fall_intraday_spike.py check && uv run ruff check scripts/regime/fall_intraday_spike.py && uv run ruff format --check scripts/regime/fall_intraday_spike.py`
Expected: `self-check OK`, both ruff commands exit 0.

- [ ] **Step 5: Smoke the real data**

Run: `uv run python -c` importing the script, `minute_bars("SH", date(2016,1,4), date(2016,1,8))`
Expected: 5 sessions, first bar 09:30 New York, last 15:59; also check `SDS` and `SPXU` have bars on 2016-01-04
(the spec left this unverified). If a symbol has no 2016 bars, stop and report to the user.

- [ ] **Step 6: Commit (spec, plan, script)**

```bash
git add docs/superpowers/specs/2026-11-23-market-fall-intraday-spike-design.md docs/superpowers/plans/2026-11-23-market-fall-intraday-spike.md scripts/regime/fall_intraday_spike.py
git commit -m "research: market-fall intraday spike, spec, plan, data and trade model"
```

### Task 2: Account replay, report and the dev / test modes

**Files:**
- Modify: `scripts/regime/fall_intraday_spike.py`

**Interfaces:**
- Consumes: Task 1's five functions.
- Produces:
  - `replay(bars: dict[date, pd.DataFrame], days: Iterable[date], stop: float | None, fraction: float) -> pd.DataFrame`
    — one row per session in the window: `equity` (start 5,000), `traded` (bool), `exit` (`close`/`stop`/`""`),
    `trade_return`; on a traded day `equity *= 1 + fraction * (mult - 1)`; a day in `days` with no bars is untraded
    and counted as `missing`.
  - `summary(rows: pd.DataFrame) -> dict` — total, CAGR, per calendar year, max drawdown, trades, win rate, mean trade,
    stops vs closes, missing days.
  - `report(window: str, only: set[str] | None = None) -> str | None` — prints the tables, returns the chosen cell name.

- [ ] **Step 1: Add to `check()`**

```python
    # 7. replay: two traded days at 1.1 and 0.9, fraction 0.5 -> 5000 * 1.05 * 0.95
    rows = replay(fake_bars({d1: 1.1, d2: 0.9}), [d1, d2], None, 0.5)
    assert close_to(rows.equity.iloc[-1], 5000 * 1.05 * 0.95, costs=True)
    # 8. a fall day without bars is untraded and counted
    assert summary(replay(fake_bars({d1: 1.1}), [d1, d2], None, 1.0))["missing"] == 1
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run python scripts/regime/fall_intraday_spike.py check`
Expected: `NameError` on `replay`.

- [ ] **Step 3: Implement `replay`, `summary`, `report` and `main`**

`report(window)`: cells = {SH, SDS, SPXU, -1xSPY} x stop {None, 0.02, 0.04}, each run on fall days and on every
session, with fraction 1.0 (unconstrained) and 0.5 (cash, conservative). Print: the **sign gate** (mean SPY
open-to-close on fall days, short side after costs) first; one row per cell (fall days / every session /
non-fall sessions); per-year table; SPY buy-and-hold over the same sessions; the hedge row "2,500 SPY buy-and-hold +
2,500 chosen cell" (total and max drawdown against SPY alone). Then apply the spec's bar and choice rule verbatim and
print `chosen: <cell>` or `NO: <reason>`. `main`: `check` | `dev` | `test`; `test` refuses if the guard file exists,
re-derives the choice from `dev`, writes the guard, then reports the held-out window for the chosen cell only.

- [ ] **Step 4: Run the self-check and lint**

Run: `uv run python scripts/regime/fall_intraday_spike.py check && uv run ruff check scripts/regime/fall_intraday_spike.py && uv run ruff format --check scripts/regime/fall_intraday_spike.py`
Expected: `self-check OK`, ruff exit 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/regime/fall_intraday_spike.py
git commit -m "research: market-fall intraday spike, replay and report"
```

### Task 3: Dev run, written choice, one held-out run, record

**Files:**
- Create: `docs/superpowers/replays/2026-11-23-market-fall-intraday.md`

- [ ] **Step 1: Run dev**

Run: `uv run python scripts/regime/fall_intraday_spike.py dev | tee logs/fall_intraday_dev.txt`
Expected: the tables and either `chosen: <cell>` or `NO: <reason>`.

- [ ] **Step 2: Write the record's dev section and commit it before any held-out run**

The record: question, the fixed bar, the dev tables (sign gate, every cell, reference rows, per year), the verdict.
If NO: say so, stop here, Task 3 is done after the commit.

```bash
git add docs/superpowers/replays/2026-11-23-market-fall-intraday.md
git commit -m "research: market-fall intraday spike, dev result (before the held-out run)"
```

- [ ] **Step 3: Run the held-out window once**

Run: `uv run python scripts/regime/fall_intraday_spike.py test | tee logs/fall_intraday_test.txt`
Expected: the chosen cell on 2025-01-02 to the last session, `GO` or `NO-GO`; a second run refuses.

- [ ] **Step 4: Add the held-out section to the record and commit**

```bash
git add docs/superpowers/replays/2026-11-23-market-fall-intraday.md
git commit -m "research: market-fall intraday spike, held-out result"
```

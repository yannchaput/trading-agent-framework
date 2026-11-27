# bull_bear rank-only stress test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the max drawdown of bull_bear's rank-only book over 2005-2026 and apply the user's 40% limit.

**Architecture:** One research script, `scripts/bull_bear/rank_only_stress.py`: pure simulation functions checked by `self_check()`, then data loading through the repo's universe-history builder (separate cache), references and a report. No `src/` change.

**Tech Stack:** Python 3.14, pandas, yfinance (through repo helpers), requests.

**Spec:** `docs/superpowers/specs/2026-11-27-bull-bear-rank-only-stress-test-design.md`

## Global Constraints

- Period: first rebalance in 2005-01, end 2026-11-24; equity marked at each session's close.
- Book: top 10 by rank, 9.8% each, reset every week, no cooldown, sector cap or overlay.
- Rest in SHV; cash at 0% before SHV has a price. 5 bps per side on every dollar traded, SHV included.
- Verdict at 9.8% only: max drawdown <= 40% -> "not rejected"; worse -> "rank-only at this size is rejected for real money". 8.4% and 7.0% weights (84% / 70% stocks) are report-only.
- History file and bars go in `cache/stress_2005/`; nothing under `data/universe/` or `cache/universe_history/` is written.
- If `ensure_universe_history` raises `UniverseHistoryError` for more than 1% absent symbols, stop and report; do not raise the limit.
- No change to the rule after a result is seen.

## Review Focus

- A held stock whose prices stop: sold at its last close, never valued at NaN or zero.
- A stock with no bar on the ranking session (halted, delisted, not yet listed): not ranked on stale closes.
- Look-ahead: ranking uses closes strictly before the trade session.
- Yahoo bad prints over 20 years (a non-positive price, a many-fold one-day jump) driving the book's return or drawdown: reported, not silently kept or removed.
- A holiday Tuesday: the rebalance moves to the next session of the same week, once.

---

### Task 1: Simulation core

**Files:**
- Create: `scripts/bull_bear/rank_only_stress.py`

**Interfaces:**
- Produces:
  - `rebalance_days(sessions: list[date]) -> list[date]`: per ISO week, the first session with weekday >= 1.
  - `simulate(prices: dict[str, pd.DataFrame], sessions: list[date], picks: Callable[[date, set[str]], list[str]], weight: float, cost: float = 0.0005, sleeve: str = "SHV") -> pd.Series`: equity (start 1.0) at each session close. `prices[symbol]` has `open` and `close` indexed by `date`. `picks(trade_day, held)` is called on rebalance days only and returns at most 10 symbols.
  - `max_drawdown(equity: pd.Series) -> tuple[float, date, date]`: (negative fraction, peak date, trough date).

`simulate`: hold shares. On a rebalance day value every holding at that day's open, else its last close; set each pick to `weight * value`; put `1 - weight * len(picks)` in the sleeve if it has an open that day, else cash; subtract `cost * sum(abs(dollar change))` from the sleeve, or from cash without one. A pick with no open that day is left out. A holding with no bar is valued at its last close.

- [ ] **Step 1: Write `self_check()` first** (with stubs that raise `NotImplementedError`)

```python
def self_check():
    days = [date(2024, 1, d) for d in (2, 3, 4, 8, 10, 11)]  # Tue, Wed, Thu, Mon, Wed (holiday Tue), Thu
    assert rebalance_days(days) == [date(2024, 1, 2), date(2024, 1, 10)]
    frame = lambda rows: pd.DataFrame(rows, columns=["day", "open", "close"]).set_index("day")
    dead = {"A": frame([(days[0], 10.0, 10.0), (days[1], 10.0, 8.0)])}  # prices stop after day 2
    equity = simulate(dead, days, lambda day, held: ["A"] if day == days[0] else [], weight=0.5, cost=0.0)
    assert [round(v, 6) for v in equity] == [1.0, 0.9, 0.9, 0.9, 0.9, 0.9]  # sold at its last close
    costly = simulate(dead, days[:2], lambda day, held: ["A"], weight=0.5, cost=0.001)
    assert round(costly.iloc[0], 6) == 0.9995
    seen = []
    simulate(dead, days, lambda day, held: seen.append((day, held)) or ["A"], weight=0.5)
    assert seen[1] == (days[4], {"A"})
    dd, peak, trough = max_drawdown(pd.Series([1.0, 2.0, 1.0, 1.5], index=days[:4]))
    assert (dd, peak, trough) == (-0.5, days[1], days[2])
```

- [ ] **Step 2:** Run `uv run python scripts/bull_bear/rank_only_stress.py --self-check`. Expected: `NotImplementedError`.
- [ ] **Step 3:** Implement the three functions.
- [ ] **Step 4:** Rerun. Expected: `self-check ok`, exit 0. `uv run ruff check && uv run ruff format --check` exit 0; `uv run pyright scripts/bull_bear/rank_only_stress.py` 0 errors.
- [ ] **Step 5:** Commit `research(bull_bear): rank-only stress test simulation core`.

### Task 2: Data, ranking, references, report and run

**Files:**
- Modify: `scripts/bull_bear/rank_only_stress.py`
- Modify: the spec (append `## Results`)

**Interfaces:**
- Consumes: Task 1's functions.
- Repo calls:
  - `ensure_universe_history(date(2003, 10, 1), date(2026, 11, 24), path=Path("cache/stress_2005/history.json"), cache_dir=Path("cache/stress_2005")) -> UniverseHistory`, after copying `cache/universe_history/pool.json` into `cache/stress_2005/` if absent.
  - bars: `parse_yahoo_frame(pd.read_parquet(cache/stress_2005/bars/<SYM>.parquet))` -> adjusted `open`, `close`, `volume`, plus `traded_close`, `traded_volume`; index re-keyed to `date`. Sessions are SPY's.
  - SHV is not in the pool: download it with `yfinance_download()` the same way the builder downloads a symbol (`auto_adjust=False, actions=True`).
  - `month_universe(history, day, exclude={"SHV"})`; `score_stock(symbol, closes, volumes, params, traded_closes=..., traded_volumes=...)` and `rank`, with `CONFIG` from `strategies/cross_momentum/parameters.py`, and for a held symbol `{**CONFIG, "max_volatility": math.inf}`.

- [ ] **Step 1: Ranking function** `top10(day: date, held: set[str]) -> list[str]`: symbols = the month list only; as in bull_bear, a held symbol that left it is unranked and so sold. A symbol is ranked only if it has a bar on the session before `day`; pass its last 300 sessions strictly before `day`. Add to `self_check()` an assert on a two-symbol fixture that a bar dated `day` is not read and a symbol with no previous-session bar is not ranked. Run: fails first, then passes.
- [ ] **Step 2: References.** SPY total return (adjusted close). Equal-weighted universe: each session, the mean close-to-close return of the month list's symbols that have both closes. French: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_Daily_CSV.zip`, into `cache/stress_2005/french/`, read the CSV from the zip in memory (data only, never executed); take the `Hi PRIOR` column of the value-weighted and equal-weighted daily tables, percent to fraction, drop -99.99; report its last date. If the URL fails, stop and report.
- [ ] **Step 3: Report.** For the three weights and each reference: CAGR, max drawdown with dates, and max drawdown inside 2008-01..2009-12, 2015-01..2016-12, 2018-10..2018-12, 2020, 2022 (peak and trough both inside the window). Per year: mean number of ranked stocks, and share of that year's listed symbols with bars. Data check: the 20 largest one-day returns (either sign) of held stocks, with symbol and date, and the count of non-positive prices. Save the three equity curves to `cache/stress_2005/equity.csv`. Print the verdict line from the 9.8% run.
- [ ] **Step 4: Run** `uv run python scripts/bull_bear/rank_only_stress.py` in the background (the first build downloads about 6,660 symbols and can take over an hour). Check the data check list: look up any many-fold one-day move before trusting the result; if one is a bad print that moves max drawdown, report both numbers and ask.
- [ ] **Step 5: Cross-check** the 2020-01..2021-05 drawdown at 9.8% against the earlier rebuilt -46.4% and state the difference; they are different books, so a gap is not a failure.
- [ ] **Step 6:** Append `## Results` to the spec: the tables, the verdict, the caveats. Final passes per CLAUDE.md (`/simplify`, code-simplifier, `ponytail-review` on the staged diff), then the gate (`ruff check`, `ruff format --check`, `pyright`, `pytest --cov`; 2 news-default test failures and the `interrogate` crash already happen on `main`).
- [ ] **Step 7:** Commit `research(bull_bear): rank-only stress test results` (`git add -f` for `docs/`). No push.

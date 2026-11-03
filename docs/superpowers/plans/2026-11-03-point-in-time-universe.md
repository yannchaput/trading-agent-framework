# A universe as it stood on each date -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cross_momentum's and bull_bear's backtests trade, on each date, the monthly top-1,200 universe the live batch would have built then, from Yahoo data only, built automatically by `batch-universe --history` as a backtest pre-step.

**Architecture:** a pure module `strategies/common/universe_history.py` holds the `UniverseHistory` object (month → symbols) and the monthly ranking. The live batch gains a history mode: the pool without the $2B pre-filter, cached Yahoo bars, `ensure_universe_history(...)`. Both strategies' `run_backtesting` call it first, preload the union and read the month's list at each review.

**Tech Stack:** Python 3.14, uv, pandas, yfinance 1.7.0, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-03-point-in-time-universe-design.md`

## Global Constraints

- A month's list uses no bar dated on or after that month's first session.
- Screens: traded price (`traded_close`) of at least `CONFIG["min_price"]` (10.0), 63-session average traded dollar volume (`traded_close x traded_volume`) of at least `CONFIG["min_dollar_volume"]` (20_000_000), top 1,200 by that average. No market-cap screen.
- Pool: the live batch's sources (NASDAQ screener, iShares IWB/IWM, Vanguard VTI) WITHOUT the market-cap and price pre-filter, common stocks only (`quoteType == "EQUITY"`).
- The history is the ONLY universe for these two strategies' backtests. A history that cannot cover the window stops the backtest with `UniverseHistoryError` (a `BacktestError`), before it starts.
- Yahoo only, through `backtesting/data/yahoo.yfinance_download`, `extract_ticker_frame`, `parse_yahoo_frame` (`auto_adjust=False, actions=True`).
- Paper/live, the other three strategies' backtests, and `batch-universe` without `--history` are unchanged.
- Files: history `data/universe/us_stock_universe_history.json`; bar cache `cache/universe_history/`.
- The automated suite never touches the network (injected fetchers/downloads). Real Yahoo only in `scripts/tests/`.
- Paths are relative to the repo root.

## Review Focus

1. **The month's first session itself is excluded from its own ranking** (a volume spike on that day must not count). Test in Task 1.
2. **A month without 63 sessions of history before it is an error, never a shorter average.** Test in Task 1. Task 3 downloads enough lookback to avoid it.
3. **A failing source when building the pool must stop the build.** The live batch only logs and skips a failing iShares/Vanguard source. In history mode a missing source would quietly shrink the pool, so it raises. Test in Task 2.
4. **The review date picks the right month:** 2019-03-12 → "2019-03". A date outside the history raises; it never falls back to another month. Tests in Tasks 1 and 4.
5. **Backtest construction must not need today's `us_stock_universe.json`** for these two strategies (a fresh clone has only the history). Tests in Tasks 4 and 5.

---

### Task 1: `UniverseHistory` and the monthly ranking (pure)

**Files:**
- Create: `src/trading_agent_framework/strategies/common/universe_history.py`
- Modify: `src/trading_agent_framework/utils/errors.py` (`UniverseHistoryError(BacktestError)`)
- Test: `tests/strategies/common/test_universe_history.py`

**Interfaces:**
- Produces:
  - `UniverseHistory(built: date, pool_date: date, top_n: int, min_price: float, min_dollar_volume: float, lookback: int, months: dict[str, tuple[str, ...]])` (frozen dataclass);
  - `.on(day: date) -> list[str]`, the list of `day`'s month, raising `UniverseHistoryError` if the month is absent;
  - `.covers(start: date, end: date) -> bool`, true when every month from start to end is present;
  - `.union(start: date, end: date) -> list[str]`, sorted;
  - `.to_json() -> dict` / `UniverseHistory.from_json(data: dict)`;
  - `save(path: Path)` / `UniverseHistory.load(path: Path) -> UniverseHistory | None` (None if the file is missing).
- Produces: `monthly_lists(frames: Mapping[str, pd.DataFrame], sessions: Sequence[date], months: Sequence[str], *, min_price: float, min_dollar_volume: float, top_n: int, lookback: int = 63) -> dict[str, tuple[str, ...]]`.
  - `frames` are `parse_yahoo_frame` outputs, indexed at the session close, with `traded_close`/`traded_volume`.
  - `sessions` is the market calendar.
  - A month's first session is the first `sessions` date in that month. The month must have at least `lookback` sessions before it, else `UniverseHistoryError`.
  - A symbol is eligible when it has at least `lookback` rows dated before that session. Its average is the mean of `traded_close * traded_volume` over its last `lookback` such rows. Its price is the last such `traded_close`.
  - The kept symbols are sorted by average, descending, ties broken by symbol.

- [ ] **Step 1: Write the failing tests** (hand-built frames, `lookback=3`; sessions are the weekdays of 2024-01-29 .. 2024-02-09; February's first session is 2024-02-01, preceded by 01-29, 01-30, 01-31)

- `test_a_month_ranks_by_traded_dollar_volume_before_its_first_session`: four symbols, each with traded close and volume constant on 01-29..01-31:
  - A: 20 × 2e6, $40M;
  - B: 5 × 1e7, $50M but under $10;
  - C: 50 × 1e5, $5M, under $20M;
  - D: 100 × 1e6, $100M.

  With `min_price=10, min_dollar_volume=20e6`: `top_n=2` gives `{"2024-02": ("D", "A")}`, and `top_n=1` gives `("D",)`.
- `test_the_first_session_of_the_month_is_excluded`: same, plus E at 50 × 1 (about $0) on 01-29..01-31 and 50 × 1e9 on 02-01. E is not in the February list.
- `test_a_month_without_enough_prior_sessions_is_an_error`: `months=["2024-01"]` with sessions starting 2024-01-29 raises `UniverseHistoryError`.
- `test_a_symbol_with_fewer_rows_than_the_lookback_is_not_eligible`: F has only 01-30 and 01-31 rows at $1B a day, and is absent.
- `test_on_picks_the_month_of_the_day_and_never_another`: `.on(date(2024, 2, 20))` returns February's list, and `.on(date(2024, 3, 1))` raises `UniverseHistoryError`.
- `test_covers_and_union`: months "2024-02" and "2024-03":
  - `covers(2024-02-05, 2024-03-28)` is True;
  - `covers(2024-01-31, 2024-02-05)` is False;
  - `union` is the sorted set of both months.
- `test_json_round_trip`: `from_json(h.to_json()) == h`; `save` then `load` gives the same object; `load` of a missing path returns None.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/common/test_universe_history.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** both in the new module, with no I/O beyond `save`/`load`.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/common/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/common/universe_history.py src/trading_agent_framework/utils/errors.py tests/strategies/common/test_universe_history.py
git commit -m "feat(universe): Task 1 - UniverseHistory and the monthly dollar-volume ranking"
```

### Task 2: The history pool

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/batch_us_stock_universe.py`
- Test: `tests/strategies/test_batch_us_stock_universe.py`

**Interfaces:**
- Produces: `parse_nasdaq_symbols(rows: list[dict]) -> list[str]`, every normalized symbol with no market-cap or price pre-filter, deduplicated and sorted.
- Produces: `get_history_pool(*, fetch_nasdaq_rows=..., fetch_ishares_csv=_fetch_ishares_csv, fetch_vanguard_holdings=_fetch_vanguard_holdings, is_common_stock=...) -> list[str]`:
  - the union of the NASDAQ screener rows and the IWB/IWM/VTI holdings;
  - filtered by `is_common_stock(symbol) -> bool`, which reuses the live batch's yfinance `info` fetch and rate-limit gate and is true when `quoteType == "EQUITY"`;
  - sorted.
  - Raises `UniverseHistoryError` if any source fails or returns nothing, or if the pool comes out empty.
- The live batch's `get_ticker_universe`, `get_extra_universe_symbols` and `main` keep their behaviour (failing sources still only skipped there).

- [ ] **Step 1: Write the failing tests** (fakes for every fetcher)

- `test_history_pool_keeps_small_and_cheap_listings`: NASDAQ rows include a $500M-cap stock at $4. It is in the pool, while `parse_nasdaq_rows` (live) still drops it.
- `test_history_pool_unions_the_sources_and_keeps_common_stocks_only`:
  - NASDAQ gives A and B;
  - iShares gives B and C;
  - Vanguard gives D;
  - `is_common_stock` is false for C.

  The pool is `["A", "B", "D"]`.
- `test_a_failing_source_stops_the_history_pool`: each of the NASDAQ, iShares and Vanguard fakes raising in turn makes `get_history_pool` raise `UniverseHistoryError` naming the source.
- `test_the_live_batch_still_skips_a_failing_etf_source`: the existing `get_extra_universe_symbols` behaviour, kept as a regression pin.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/test_batch_us_stock_universe.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** Factor the NASDAQ HTTP call out of `get_ticker_universe` into `_fetch_nasdaq_rows()` so both modes share it. The history path makes the common-stock check a separate small function, not a full `_screen_single_ticker` run.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/batch_us_stock_universe.py tests/strategies/test_batch_us_stock_universe.py
git commit -m "feat(universe): Task 2 - the history pool without the market-cap pre-filter"
```

### Task 3: Build, cache and ensure the history; `batch-universe --history`

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/batch_us_stock_universe.py`
- Test: `tests/strategies/test_batch_us_stock_universe.py`

**Interfaces:**
- Consumes: `UniverseHistory`, `monthly_lists`, `UniverseHistoryError` (Task 1); `get_history_pool` (Task 2).
- Produces: `ensure_universe_history(start: date, end: date, *, path: Path = DEFAULT_OUTPUT_DIR / "us_stock_universe_history.json", cache_dir: Path = PROJECT_ROOT / "cache" / "universe_history", today: Callable[[], date] = date.today, download=None, pool=None) -> UniverseHistory`.
  - **Covered:** if the history file at `path` already covers every month from start to end, return it without any download.
  - **Otherwise build or extend it:**
    - reuse the saved pool when its date is on or after the last month needed; else call `get_history_pool()`, or the injected `pool()`;
    - download the pool's daily bars in batches over [first month's first day minus 120 calendar days, end], with `auto_adjust=False, actions=True` through `yfinance_download`, plus SPY for the sessions;
    - read and write a per-batch parquet cache in `cache_dir`, downloading only what isn't cached;
    - parse each with `extract_ticker_frame` and `parse_yahoo_frame`, and compute the missing months with `monthly_lists(top_n=1200, CONFIG thresholds)`;
    - merge with the existing months and `save`.
  - **Errors:** any failure raises `UniverseHistoryError`, wrapping a plain exception. That covers a download error, the pool, a month that comes out empty, and missing SPY sessions.
- Produces: CLI `uv run batch-universe --history --start YYYY-MM --end YYYY-MM`, which calls `ensure_universe_history` and prints the months and sizes. Without `--history`, `main` is unchanged.

- [ ] **Step 1: Write the failing tests** (fake `download` returning batched `auto_adjust=False` frames with `Adj Close`/`Stock Splits`; fake `pool`; `tmp_path` for `path` and `cache_dir`)

- `test_a_covering_history_is_reused_without_downloading`: a saved history covering 2024-02..2024-03 makes `ensure_universe_history(2024-02-05, 2024-03-28)` return it, with the fake download never called.
- `test_a_missing_history_is_built_and_saved`: the file is created, its months cover the window, and each month's list matches `monthly_lists` on the same fake data.
- `test_a_history_is_extended_with_only_the_missing_months`: an existing history has 2024-02, the request is up to 2024-03, and 2024-02's list is unchanged.
- `test_cached_bars_are_not_downloaded_again`: a second build with the same pool makes no download call for cached symbols.
- `test_a_download_failure_raises_universe_history_error` and `test_an_empty_month_raises_universe_history_error`.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/test_batch_us_stock_universe.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** with the batch's rate-limit gate for downloads, and add the CLI flags.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/batch_us_stock_universe.py tests/strategies/test_batch_us_stock_universe.py
git commit -m "feat(universe): Task 3 - build, cache and ensure the universe history; batch-universe --history"
```

### Task 4: cross_momentum backtests use the history

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`run_backtesting`, the rebalance path, `_backtest_preload_assets`)
- Modify: `src/trading_agent_framework/utils/strategy_factory.py` (backtesting mode builds cross_momentum without today's universe file)
- Test: `tests/strategies/test_cross_momentum_*.py` (a new `test_cross_momentum_universe_history.py` is fine), `tests/utils/` or wherever the factory is tested

**Interfaces:**
- Consumes: `ensure_universe_history` (Task 3) and `UniverseHistory.on/union` (Task 1).
- `run_backtesting(**overrides)` first calls `ensure_universe_history(start, end)` on the window from `parameters` or overrides; a `UniverseHistoryError` propagates and the backtest never starts. It then stores the history (e.g. `self.vars.universe_history`), sets `self.parameters["universe"] = {"mode": "point_in_time", "history_built": ..., "pool_date": ..., "top_n": 1200}` so `settings.json` records it, and preloads the union of the window's lists plus the sleeve and benchmark.
- On a rebalance day in a backtest, the scan's universe is `history.on(today)` minus the sleeve symbols. Paper/live keep `vars.universe` from the file.

- [ ] **Step 1: Write the failing tests** (inject a fake `ensure_universe_history` via monkeypatch; a hand-built `UniverseHistory`)

- `test_backtest_scans_the_month_list_of_each_review_date`: with history months "2024-02" = (A, B) and "2024-03" = (B, C), the scan on 2024-02-13 reads A and B, and on 2024-03-12 reads B and C (spy on `get_historical_prices` or `compute_target_portfolio`'s symbols).
- `test_backtest_preloads_the_union_plus_sleeve_and_benchmark`.
- `test_backtest_records_the_universe_in_parameters`.
- `test_a_failing_history_stops_the_backtest_before_it_starts`: the fake raises `UniverseHistoryError`, so `run_backtesting` raises it and the runner is never called.
- `test_backtest_mode_builds_cross_momentum_without_the_universe_file` (factory; universe file absent). Paper mode still returns None with the existing warning.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/ tests/utils/ -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ tests/utils/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py src/trading_agent_framework/utils/strategy_factory.py tests/
git commit -m "feat(cross_momentum): Task 4 - backtests trade the universe as it stood on each date"
```

### Task 5: bull_bear backtests use the history

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` (`run_backtesting`, pipeline wiring), `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (`_universe` read per review)
- Modify: `src/trading_agent_framework/utils/strategy_factory.py` (backtesting mode builds bull_bear without today's file)
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py`, `tests/strategies/bull_bear/test_bull_bear_*.py`

**Interfaces:**
- Consumes: as Task 4.
- The pipeline gets its universe per review: a `Callable[[date], Sequence[str]]` (paper/live: a constant list from the file; backtest: `history.on`). The parking symbol is excluded as today.
- `run_backtesting`: same pre-step, parameters record and union preload as Task 4 (plus SHV and the benchmark).

- [ ] **Step 1: Write the failing tests**

- `test_a_backtest_review_ranks_the_month_list_of_its_date`: two reviews in different months rank different symbol sets (spy on the bars loader's `universe` argument).
- `test_a_holding_outside_the_month_list_is_dropped_and_sold`: a held stock absent from the new month's list is not ranked and is sold by the existing rule.
- `test_bull_bear_backtest_preloads_the_union_and_records_the_universe`.
- `test_a_failing_history_stops_the_bull_bear_backtest`.
- `test_backtest_mode_builds_bull_bear_without_the_universe_file`.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/bull_bear/ tests/utils/ -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest -q` then `uv run ruff check`
Expected: full suite PASS, "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/ src/trading_agent_framework/utils/strategy_factory.py tests/
git commit -m "feat(bull_bear): Task 5 - backtests trade the universe as it stood on each date"
```

### Task 6: Dashboard line, docs, smoke script

**Files:**
- Modify: `src/trading_agent_framework/dashboard/components/metric_cards.py` and the settings model in `src/trading_agent_framework/dashboard/models.py` / `reader.py` (read `parameters.universe`); `src/trading_agent_framework/dashboard/_pages/detail.py`
- Modify: `CLAUDE.md` (Commands: `uv run batch-universe --history --start YYYY-MM --end YYYY-MM`), `src/trading_agent_framework/backtesting/CLAUDE.md`, `src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md`, `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`, `src/trading_agent_framework/strategies/common/CLAUDE.md`, and one line each in `bill_ackman`, `vwap_pullback`, `earnings_drift` CLAUDE.md
- Create: `scripts/tests/smoke_universe_history.py`
- Test: `tests/dashboard/` (the existing settings/metric-card tests)

- [ ] **Step 1: Write the failing dashboard test.** For a run whose `parameters.universe.mode` is `point_in_time`, the run detail's metric card shows `**Universe:** as it stood on each date (delisted stocks missing)`. For a run without it, it shows `**Universe:** today's list`.
- [ ] **Step 2: Run, expect failure.** `uv run pytest tests/dashboard/ -q`.
- [ ] **Step 3: Implement** the line, then the docs. Say which universe each mode uses, describe the pre-step and its error, and state that the residual bias is delisted stocks.
- [ ] **Step 4: Write and run the smoke script against real Yahoo:** `uv run python scripts/tests/smoke_universe_history.py`, using a temp history path and the real cache dir. It prints PASS/FAIL per check and exits non-zero on any FAIL:
  - NVDA is in the 2017-06 list;
  - every 2017-06 symbol had a traded close of at least $10 on 2017-05-31;
  - the list has at most 1,200 symbols;
  - at least one 2017-06 symbol is absent from the latest month's list.

  Expected: PASS. The first run may take tens of minutes or more (pool check and bar download).
- [ ] **Step 5: Final check and commit**

Run: `uv run pytest -q && uv run ruff check`
Expected: PASS, "All checks passed!".

```bash
git add src/trading_agent_framework/dashboard/ CLAUDE.md src/trading_agent_framework/backtesting/CLAUDE.md src/trading_agent_framework/strategies/ scripts/tests/smoke_universe_history.py tests/dashboard/
git commit -m "docs: Task 6 - universe-history dashboard line, docs and smoke test"
```

## After the tasks (by hand)

Rerun a cross_momentum backtest on its usual window and compare it with the last run on today's universe. The audit expects roughly half the profit to go.

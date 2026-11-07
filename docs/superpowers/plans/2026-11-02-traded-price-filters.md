# Momentum filters on traded prices -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** cross_momentum's and bull_bear's `min_price` and `min_dollar_volume` filters compare against the price and dollar volume a stock actually traded at on each date, so long backtests stop dropping stocks that later split or paid dividends.

**Architecture:** One Yahoo download per batch switches to `auto_adjust=False, actions=True`. `parse_yahoo_frame` rebuilds today's adjusted OHLCV with yfinance's own arithmetic and adds `traded_close`/`traded_volume` (Yahoo's split-adjusted figures with every later split undone). The scoring computes the filter inputs from those columns when a frame has both, and exactly as today otherwise.

**Tech Stack:** Python 3.14, uv, pandas, yfinance 1.7.0, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-02-traded-price-filters-design.md`

## Global Constraints

- Returns, volatility, ranking maths, the risk overlay, the sleeve and fills keep reading today's adjusted `open/high/low/close/volume`, value for value.
- One Yahoo download per batch, as today. No second call, no per-symbol `Ticker(...).splits`.
- A frame with only one of `traded_close`/`traded_volume` is treated as having neither.
- The automated suite never touches the network (injected `download=` fakes). Real Yahoo only in `scripts/tests/`.
- `Bars.df` columns stay float64.
- Untouched: earnings_drift, vwap_pullback, bull_bear's fact sheet, `batch_us_stock_universe.py`.
- Paths below are relative to the repo root; source lives under `src/trading_agent_framework/`.

## Review Focus

1. **A split ex-date row with NaN prices.** yfinance can emit one in a multi-ticker frame. The split factor must be computed before NaN rows are dropped, or every row before it loses that split. Test in Task 1.
2. **"No split" is 0.0, not 1.0, in `Stock Splits`.** Multiplying by it zeroes every price. A reverse split is a ratio below 1. Test in Task 1.
3. **A split after the backtest window.** It must still scale the rows inside the window, and no row after the window's end may be visible through `bars()`. Test in Task 2.
4. **Series alignment after `completed_bars`/`tail`.** The traded lists must be cut from the same rows as `closes`, or the filter reads another day's price. Tests in Task 5.
5. **Old-shape frames (no `Adj Close`).** Every existing fake and any `auto_adjust=True` caller must parse exactly as today, with no traded columns. Test in Task 1.

---

### Task 1: Traded columns in `parse_yahoo_frame`

**Files:**
- Modify: `src/trading_agent_framework/backtesting/data/yahoo.py` (`parse_yahoo_frame`, new helpers)
- Test: `tests/backtesting/data/test_yahoo.py`

**Interfaces:**
- Produces: `TRADED_COLUMNS: tuple[str, str] = ("traded_close", "traded_volume")` (module constant).
- Produces: `parse_yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame`. With an `Adj Close` column (the `auto_adjust=False` shape): columns `open, high, low, close, volume, traded_close, traded_volume`. Without one (today's `auto_adjust=True` shape): `open, high, low, close, volume`, unchanged behaviour.
- Produces: `traded_series(frame: pd.DataFrame) -> tuple[list[float], list[float]] | None`. Both traded columns as float lists, or None unless both are present.

- [ ] **Step 1: Write the failing tests**

Add a fixture with a 2:1 split on the third row:

```python
def _raw_unadjusted_frame() -> pd.DataFrame:
    """An auto_adjust=False, actions=True download: Close split-adjusted only, Adj Close also dividend-adjusted."""
    index = pd.date_range("2026-01-05", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [49.0, 50.0, 51.0, 52.0, 53.0],
            "High": [51.0, 52.0, 53.0, 54.0, 55.0],
            "Low": [48.0, 49.0, 50.0, 51.0, 52.0],
            "Close": [50.0, 51.0, 52.0, 53.0, 54.0],
            "Adj Close": [45.0, 45.9, 46.8, 53.0, 54.0],
            "Volume": [2000.0, 2000.0, 1000.0, 1000.0, 1000.0],
            "Dividends": [0.0, 0.0, 0.0, 0.5, 0.0],
            "Stock Splits": [0.0, 0.0, 2.0, 0.0, 0.0],
        },
        index=index,
    )
```

Tests (literal expectations):

- `test_traded_close_undoes_every_later_split`: `list(df["traded_close"]) == [100.0, 102.0, 52.0, 53.0, 54.0]`, `list(df["traded_volume"]) == [1000.0] * 5`, and `list(df.columns) == ["open", "high", "low", "close", "volume", "traded_close", "traded_volume"]`.
- `test_adjusted_columns_follow_yfinances_auto_adjust`: `close` equals `Adj Close`; `open/high/low` equal `Open/High/Low * Adj Close / Close` row for row (`pytest.approx`, computed in the test from the fixture literals); `volume` equals `Volume`.
- `test_a_reverse_split_and_two_splits_multiply`: splits `[0, 0.1, 0, 3.0, 0]` on a 5-row frame with `Close` all 10.0 give `traded_close == [3.0, 30.0, 30.0, 10.0, 10.0]` (row 0: x0.1 x3, rows 1-2: x3, the row of the ratio itself excluded).
- `test_a_split_on_a_nan_row_still_scales_the_rows_before_it`: the fixture with the split row's `Open/High/Low/Close/Adj Close` set to NaN gives 4 rows and `traded_close == [100.0, 102.0, 53.0, 54.0]`.
- `test_an_adjusted_shape_frame_parses_as_today`: `parse_yahoo_frame(_raw_yahoo_frame())` has no traded columns, and `traded_series` of it is None.
- `test_traded_series_needs_both_columns`: the parsed fixture gives `([100.0, 102.0, 52.0, 53.0, 54.0], [1000.0] * 5)`; the same frame without `traded_volume` gives None.

- [ ] **Step 2: Run them, expect failures**

Run: `uv run pytest tests/backtesting/data/test_yahoo.py -q`
Expected: the six new tests FAIL (missing columns / missing names); the existing ones pass.

- [ ] **Step 3: Implement**

In `parse_yahoo_frame`, when `Adj Close` is a column: compute the split factor on the RAW frame, before the NaN drop. The factor for a row is the product of `Stock Splits` over rows strictly after it, with 0.0 read as 1.0 (`splits.replace(0.0, 1.0)`, then a reversed cumulative product shifted by one row). Then build the adjusted columns with yfinance's arithmetic (`ratio = Adj Close / Close`), set `traded_close = Close * factor` and `traded_volume = Volume / factor`, drop NaN rows on the five OHLCV columns as today, and keep only the seven columns. Missing `Stock Splits` means a factor of 1.0. The old-shape path is unchanged. Update the docstring.

- [ ] **Step 4: Run, expect all green**

Run: `uv run pytest tests/backtesting/data/test_yahoo.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/backtesting/data/yahoo.py tests/backtesting/data/test_yahoo.py
git commit -m "feat(backtesting): Task 1 - traded close and volume from Yahoo's split history"
```

### Task 2: `YahooBacktestData` downloads through today and trims to the window

**Files:**
- Modify: `src/trading_agent_framework/backtesting/data/yahoo.py` (`YahooBacktestData.__init__`, `load`, `_fetch`)
- Test: `tests/backtesting/data/test_yahoo.py`

**Interfaces:**
- Consumes: `parse_yahoo_frame` (Task 1).
- Produces: `YahooBacktestData(start, end, *, download=None, calendar_symbol="SPY", today: Callable[[], date] = date.today)`. `today` is the download date only; it is never a cutoff.

- [ ] **Step 1: Write the failing tests**

- `test_downloads_ask_for_unadjusted_prices_and_splits`: after `load(...)` and a lazy `bars(...)` on another asset, both captured calls have `auto_adjust is False` and `actions is True`.
- Update `test_load_honors_its_own_start_and_end_arguments`: with `today=lambda: date(2026, 3, 1)`, `calls[0]["start"] == other_start.date().isoformat()` and `calls[0]["end"] == "2026-03-02"`.
- `test_a_split_after_the_window_scales_the_window_and_stays_invisible`: the fake returns `_raw_unadjusted_frame()` reindexed to the sessions 2026-01-07..2026-01-13, so its split row is 2026-01-09; the source's window and `load`'s end are 2026-01-08. `bars(AAPL, far_future_cutoff, 10, "day")` returns only rows dated on or before 2026-01-08, and their `traded_close` are twice their `Close`.
- `test_the_lazy_fetch_trims_to_the_constructor_window`: the same check through `_fetch` (no `load` call).

- [ ] **Step 2: Run them, expect failures**

Run: `uv run pytest tests/backtesting/data/test_yahoo.py -q`
Expected: the new and updated tests FAIL.

- [ ] **Step 3: Implement**

Both download calls pass `auto_adjust=False, actions=True` and `end=(self._today() + timedelta(days=1)).isoformat()`. After parsing, drop rows whose session date is after the requested `end`'s date (`load`'s `end`; the constructor's `end` in `_fetch`). The comparison is on the session DATE, matching backtesting's inclusive-end rule. Default `today=date.today`, with a comment: the download date is what Yahoo adjusted to, never a decision cutoff.

- [ ] **Step 4: Run, expect all green**

Run: `uv run pytest tests/backtesting/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/backtesting/data/yahoo.py tests/backtesting/data/test_yahoo.py
git commit -m "feat(backtesting): Task 2 - Yahoo backtests download through today, trimmed to the window"
```

### Task 3: Paper/live Yahoo bars ask for the same shape

**Files:**
- Modify: `src/trading_agent_framework/strategies/common/yahoo_daily_bars.py` (`YahooDailyBars.bars`, module docstring)
- Test: `tests/strategies/test_cross_momentum_yahoo_bars.py`

**Interfaces:**
- Consumes: `parse_yahoo_frame` (Task 1). The download already ends today (`end = today + 1`).
- Produces: `YahooDailyBars.bars(...)` frames carrying `traded_close`/`traded_volume` when Yahoo returns the unadjusted shape.

- [ ] **Step 1: Update the failing test**

In the existing download-arguments test, replace `assert kwargs["auto_adjust"] is True` with `assert kwargs["auto_adjust"] is False` and `assert kwargs["actions"] is True`. Add `test_paper_bars_carry_the_traded_columns`: a fake returning `_raw_unadjusted_frame()`'s shape (rebuilt in this test file, a batched `pd.concat({"AAA": raw}, axis=1)`) gives bars whose `df` has both traded columns.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/test_cross_momentum_yahoo_bars.py -q`
Expected: the two tests FAIL.

- [ ] **Step 3: Implement**

Pass `auto_adjust=False, actions=True` to the download; update the docstring ("split- and dividend-adjusted" becomes "rebuilt to split- and dividend-adjusted, plus the traded close and volume").

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/common/yahoo_daily_bars.py tests/strategies/test_cross_momentum_yahoo_bars.py
git commit -m "feat(strategies): Task 3 - paper/live Yahoo bars carry the traded columns"
```

### Task 4: Scoring reads the traded series for the filters

**Files:**
- Modify: `src/trading_agent_framework/strategies/common/scoring.py` (`MomentumInputs` comments, `momentum_inputs`, `score_stock`)
- Test: `tests/strategies/common/test_common_scoring.py`

**Interfaces:**
- Produces: `momentum_inputs(closes, volumes, params, *, traded_closes: Sequence[float] | None = None, traded_volumes: Sequence[float] | None = None) -> MomentumInputs | None`.
- Produces: `score_stock(symbol, closes, volumes, params, *, traded_closes=None, traded_volumes=None) -> MomentumRow | None`, forwarding both.
- `MomentumInputs.price` and `.avg_dollar_volume` keep their names. They become the traded figures when both traded sequences are given (their only readers are the filters; the fact sheet reads returns and closes).

- [ ] **Step 1: Write the failing tests**

With `CLOSES`/`VOLUMES` from the file and `TRADED = [c * 10 for c in CLOSES]`, `TRADED_VOLUMES = [v / 10 for v in VOLUMES]`:

- `test_the_filter_inputs_come_from_the_traded_series`: `momentum_inputs(CLOSES, VOLUMES, CONFIG, traded_closes=TRADED, traded_volumes=TRADED_VOLUMES)` has `price == 3990.0`, `avg_dollar_volume == pytest.approx(399e6)` (1e5 x 3990), and the same `ret_12_1m`, `ret_6_1m`, `ret_3m` and `volatility` as the call without traded series.
- `test_a_stock_under_min_price_on_adjusted_prices_passes_on_its_traded_price`: adjusted closes `[c / 100 for c in CLOSES]` (last 3.99) with `VOLUMES`, traded series `CLOSES` and `VOLUMES` (last 399, $399M a day): `score_stock` returns None without the traded series and a row with them.
- `test_a_dividend_payer_passes_the_dollar_volume_filter_on_traded_prices`: adjusted closes that put `avg_dollar_volume` just under `CONFIG["min_dollar_volume"]` and traded closes 40% higher: None without, a row with.
- `test_one_traded_series_alone_is_ignored`: passing only `traded_closes` gives the same inputs as passing neither.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/common/test_common_scoring.py -q`
Expected: the four tests FAIL (unexpected keyword argument).

- [ ] **Step 3: Implement**

When both traded sequences are given, `price` is `traded_closes[-1]` and `avg_dollar_volume` is the mean of the last 20 `traded_volumes` times that price. Otherwise today's computation. Everything else stays on `closes`.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/common/scoring.py tests/strategies/common/test_common_scoring.py
git commit -m "feat(scoring): Task 4 - price and dollar-volume filters read the traded series"
```

### Task 5: cross_momentum and bull_bear pass the traded columns through

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`_compute_indicators_for_ticker`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/market_data.py` (`DailySeries`, `_series`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (the two `score_stock` calls)
- Test: `tests/strategies/test_cross_momentum_indicators.py`, `tests/strategies/bull_bear/test_bull_bear_market_data.py`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Consumes: `traded_series(frame)` (Task 1), `momentum_inputs`/`score_stock` keywords (Task 4).
- Produces: `DailySeries(closes: list[float], volumes: list[float], traded_closes: list[float] | None = None, traded_volumes: list[float] | None = None)`.

- [ ] **Step 1: Write the failing tests**

- cross_momentum: `test_the_scan_filters_on_the_traded_close`: bars whose adjusted `close` is under $10 but whose `traded_close` is above $10 (plus `traded_volume`) give a result whose `price` is the traded close of the last COMPLETED session, not today's partial bar (`_daily_bars`-style helper with a bar dated today).
- cross_momentum: `test_bars_without_traded_columns_scan_as_before`: the existing `_daily_bars` frame gives `price` equal to the last completed `close`.
- bull_bear market data: `test_the_series_carries_the_traded_columns_cut_like_the_closes`: a frame with today's bar and 305 completed sessions gives `len(traded_closes) == len(closes) == 300`, with the last values from the last completed session.
- bull_bear market data: `test_a_frame_without_traded_columns_gives_none`: both traded fields are None.
- bull_bear pipeline: `test_a_stock_under_10_dollars_adjusted_is_ranked_on_its_traded_price`: a `DailySeries` with adjusted closes under $10 and traded closes above it is ranked.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/test_cross_momentum_indicators.py tests/strategies/bull_bear/ -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

Both strategies call `traded_series(frame)` on the frame AFTER `completed_bars(...).tail(...)`, so the lists come from the same rows as `closes`, and pass the result as `traded_closes=`/`traded_volumes=` (None, None when it returns None).

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest -q` then `uv run ruff check`
Expected: full suite PASS, ruff "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py src/trading_agent_framework/strategies/bull_bear/market_data.py src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/test_cross_momentum_indicators.py tests/strategies/bull_bear/test_bull_bear_market_data.py tests/strategies/bull_bear/test_bull_bear_pipeline.py
git commit -m "feat(strategies): Task 5 - cross_momentum and bull_bear filter on traded prices"
```

### Task 6: Smoke script against real Yahoo, and docs

**Files:**
- Create: `scripts/tests/smoke_traded_prices.py`
- Modify: `CLAUDE.md` (Commands list), `src/trading_agent_framework/backtesting/CLAUDE.md`, `src/trading_agent_framework/strategies/common/CLAUDE.md`, `src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md`

**Interfaces:**
- Consumes: `YahooBacktestData`, `parse_yahoo_frame`, `yfinance_download` (Tasks 1-2).

- [ ] **Step 1: Write the script**

It prints PASS/FAIL per check and exits non-zero on any FAIL (same style as the other `scripts/tests/smoke_*.py`):

1. A `YahooBacktestData` window of 2015-01-01 to 2015-01-31: NVDA's `traded_close` on the first session is between $15 and $25, and its `close` is under $1.
2. For AAPL, MSFT, NVDA and KO over 2023: the rebuilt `open/high/low/close/volume` equal a plain `auto_adjust=True` download's to 1e-9 relative.
3. KO over 2012: its mean traded dollar volume (traded close x traded volume) is above its adjusted one (close x volume).

- [ ] **Step 2: Run it**

Run: `uv run python scripts/tests/smoke_traded_prices.py`
Expected: three PASS lines, exit 0. If check 2 fails, stop: the rebuild does not match yfinance and the anti-goal is broken.

- [ ] **Step 3: Update the docs**

- Root `CLAUDE.md` Commands: add `uv run python scripts/tests/smoke_traded_prices.py    # manual smoke test: Yahoo traded close/volume and the rebuilt adjusted bars (read-only)`.
- `backtesting/CLAUDE.md`: Yahoo downloads `auto_adjust=False, actions=True` through the download date (Yahoo adjusts to it) and trims to the window. The adjusted columns are rebuilt with yfinance's own arithmetic; `traded_close`/`traded_volume` undo every later split. The download date is never a cutoff.
- `strategies/common/CLAUDE.md` (`scoring.py`): `price`/`avg_dollar_volume` come from the traded series when a frame has both columns, returns and volatility from the adjusted closes; why (NVDA under $10 until 2020, dividend payers less liquid).
- `strategies/cross_momentum/CLAUDE.md`: one line pointing to the above.

- [ ] **Step 4: Final check and commit**

Run: `uv run pytest -q && uv run ruff check`
Expected: PASS, "All checks passed!".

```bash
git add scripts/tests/smoke_traded_prices.py CLAUDE.md src/trading_agent_framework/backtesting/CLAUDE.md src/trading_agent_framework/strategies/common/CLAUDE.md src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md
git commit -m "docs: Task 6 - traded-price smoke test and filter docs"
```

## After the tasks (by hand, not part of the branch's checks)

A cross_momentum backtest from 2011 against the same window on the previous code. Expected: the two runs diverge only on dates when a stock's filter result changed (NVDA eligible from the start), with the same returns for the same holdings. The run takes hours and is the user's to launch.

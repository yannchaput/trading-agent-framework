# VIX Market Regime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SMA50/SMA200 market regime with a four-condition SPY/VIX stress score that labels SPY sell-offs while they happen, wire it into every strategy from Yahoo data, and rewrite the `Regime` line of every existing backtest run.

**Architecture:** `core/regime.py` stays pure: a replayed state machine over the last 503 aligned SPY/VIX/VIX3M closes. `core/regime_data.py` downloads and aligns those three series from Yahoo, independent of any strategy's own data source. `Strategy._refresh_regime` joins the two. A read-only scorecard and a one-off run recompute share one `regime` CLI.

**Tech Stack:** Python 3.14, uv, yfinance (via `backtesting/data/yahoo.py`'s download and parse helpers), pandas/pyarrow (recompute only), pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-vix-market-regime-design.md`

## Global Constraints

- The regime reads Yahoo `SPY`, `^VIX`, `^VIX3M` in every mode for every strategy; it never reads `get_historical_prices` and never follows `benchmark_symbol`.
- Formula constants (do not tune): SMA window 50; 52-week-high window 252; VIX median window 252; VIX spike ratio 1.5; off-high 0.05; bearish at score >= 2; bounce exit 0.08; neutral hold 3 sessions; replay 252 sessions.
- A session's regime uses only closes dated strictly before that session's date (no look-ahead; the download date is never a cutoff).
- `core/regime.py` stays pure: no I/O, no pandas, no `Strategy` import.
- A failed or short refresh keeps the previous regime with a WARNING; never `on_bot_crash`, never a lost session.
- Nothing trades on the regime. `strategy.regime` keeps the values 1 / 0 / -1 / None.
- Tests never touch the network (hand-written fakes, `tests/fakes.py`). Real-Yahoo checks go in `scripts/tests/`.
- Quality gate on touched code: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (no new errors), `uv run pytest` (all pass), `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%). Google-style docstrings on every new public module, class, function and method. No `# noqa` / `# type: ignore` / `# pragma: no cover` without asking.
- After each task's tests are green: run the `code-simplifier:code-simplifier` agent on that task's diff (low effort), then re-run the tests.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Commit message prefix follows the repo: `feat:` / `fix:` / `docs:` / `test:`, and "Task N" in the subject.

## Review Focus

1. **The bar dated today.** A paper/live refresh runs before the open, or mid-session on a late start; Yahoo may already hold a partial bar dated today. It must never reach the formula. Pinned in Task 2: `test_closes_before_excludes_the_session_itself`.
2. **A VIX/VIX3M gap.** Yahoo sometimes lacks an index close that SPY has. Gaps of up to 5 sessions forward-fill. A longer gap inside the requested window raises `RegimeDataError`, so the previous regime is kept. Pinned in Task 2: `test_a_gap_longer_than_five_sessions_fails_the_slice`.
3. **A run without `equity.parquet`.** A running or crashed backtest has none. The recompute skips it with a reason and touches nothing. Pinned in Task 5: `test_a_run_without_equity_is_skipped_untouched`.
4. **Recompute run twice.** The second run must leave the first backup intact and give byte-identical Regime rows. Pinned in Task 5: `test_recompute_is_idempotent_and_keeps_the_first_backup`.
5. **Sessions before enough history.** A session with fewer than 503 prior aligned closes gets no reading. The strategy warns once; the recompute writes no row for it. Pinned in Task 1 (`test_too_few_closes_give_no_reading`) and Task 5 (`test_sessions_without_a_reading_get_no_row`).

---

### Task 1: The pure classifier

**Files:**
- Modify: `src/trading_agent_framework/core/regime.py` (rewrite the body; keep `REGIME_LINE` and `REGIME_LABELS`; delete `_volatility` and `_percentile`)
- Test: `tests/core/test_regime.py` (rewrite)

**Interfaces:**
- Produces:
  - `RegimeParameters(sma_window=50, high_window=252, vix_median_window=252, vix_spike_ratio=1.5, off_high=0.05, bear_score=2, bounce=0.08, neutral_hold=3, replay=252)`: frozen, slots. Properties `lookback -> int` (max of the three windows, 252) and `min_bars -> int` (`lookback + replay - 1`, 503). `__post_init__` raises `ValueError` for any window < 1, `vix_spike_ratio <= 1`, `off_high` or `bounce` outside (0, 1), `bear_score` outside 1..4, `neutral_hold < 1` or `replay < 1`.
  - `RegimeReading`: frozen, slots. Fields: `regime: int`, `score: int`, `spy: float`, `sma: float`, `high: float`, `vix: float`, `vix3m: float`, `vix_median: float`, `inverted: bool`, `vix_spike: bool`, `below_sma: bool`, `off_high: bool`, `episode_low: float | None` (set only while bearish), `blocked: bool` (the re-entry block is active).
  - `regime_path(spy: Sequence[float], vix: Sequence[float], vix3m: Sequence[float], params: RegimeParameters = RegimeParameters(), *, first: int) -> list[RegimeReading]`: the readings after closes `first .. len-1`, with the state machine starting fresh at `first`. Raises `ValueError` if the three lengths differ, if `first < params.lookback - 1`, or if any close is non-finite or <= 0.
  - `classify_regime(spy, vix, vix3m, params=RegimeParameters()) -> RegimeReading | None`: `None` with fewer than `min_bars` closes. Otherwise `regime_path` over the last `min_bars` closes with `first = lookback - 1`, last reading.
  - `regime_by_session(sessions: Sequence[date], spy, vix, vix3m, params=RegimeParameters()) -> dict[date, RegimeReading]`: for each index `j >= min_bars`, `sessions[j] -> classify_regime(spy[:j], vix[:j], vix3m[:j], params)`. Closes strictly before the session; no reading for the first `min_bars` sessions.

- [ ] **Step 1: Write the failing tests.** Use a small parameter set, `SMALL = RegimeParameters(sma_window=2, high_window=3, vix_median_window=3, replay=6, neutral_hold=1)` (`min_bars == 8`), unless the test says otherwise. Build the inputs per test so each assertion below holds. The tests:
  - `test_defaults` asserts the nine default values above, `RegimeParameters().lookback == 252` and `.min_bars == 503`.
  - `test_the_line_name_and_labels` is unchanged from today.
  - `test_the_four_conditions_score_one_point_each` checks one case per condition at the last index: VIX >= VIX3M gives `inverted`; VIX >= 1.5 x median gives `vix_spike`; SPY < SMA gives `below_sma`; SPY <= 0.95 x high gives `off_high`. Each gives `score == 1` and `regime == 0`.
  - `test_two_points_are_bearish_and_record_the_episode_low` asserts `regime == -1` and that `episode_low` equals the lowest SPY close since the first bearish step.
  - `test_an_eight_percent_bounce_ends_the_episode_even_with_the_score_still_high` asserts that after SPY closes >= 1.08 x the episode low with score still >= 2, the reading has `regime == 0` and `blocked is True`.
  - `test_the_block_holds_until_a_new_low_or_a_zero_score` covers both clearing paths: a close below the episode low makes the next score-2 step bearish again; a score-0 step followed by a score-2 step is bearish.
  - `test_a_score_below_two_ends_the_episode` asserts that a bearish step followed by a score-1 step reads `regime == 0` and `blocked is False`.
  - `test_the_neutral_hold_delays_bullish_neutral_changes_only` uses `RegimeParameters(..., neutral_hold=3)`. A bullish run followed by two score-1 steps stays 1; the third reads 0. A score-2 step is -1 at once, and leaving bearish to score 0 reads 1 at once.
  - `test_the_replay_ignores_closes_before_min_bars` asserts that prefixing extra, wildly different closes does not change `classify_regime`.
  - `test_regime_by_session_uses_closes_strictly_before_the_session`: with 10 sessions, the keys are `sessions[8:]`. The value for `sessions[9]` equals `classify_regime(spy[:9], ...)`. Changing `spy[9]` leaves it unchanged.
  - `test_too_few_closes_give_no_reading`: `classify_regime` with `min_bars - 1` closes returns `None`.
  - `test_mismatched_lengths_and_bad_closes_are_rejected` (parametrised over `0`, `-1`, `nan`, `inf` in each series, and over unequal lengths) expects `ValueError`.
  - `test_invalid_parameters_are_rejected` is parametrised, one case per rule in `__post_init__`.

- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/core/test_regime.py -v`. Expected: FAIL (ImportError on `regime_path` / `regime_by_session`, old fields).

- [ ] **Step 3: Implement in `core/regime.py`.** At index `i`, every indicator includes close `i`: SMA = mean of the last `sma_window` SPY closes; high = max of the last `high_window`; median = `statistics.median` of the last `vix_median_window` VIX closes. Use slices and C builtins (`statistics.fmean`, `max`, `statistics.median`), not Python loops over windows: the recompute calls `classify_regime` about 4,000 times. The state step, in this order:

```python
# block clears first, then the bear update, then entry (an exit step never re-enters)
if blocked_at is not None and (spy_i < blocked_at or score == 0):
    blocked_at = None
if bear:
    low = min(low, spy_i)
    if spy_i >= low * (1 + p.bounce):
        bear, blocked_at = False, low
    elif score < p.bear_score:
        bear = False
elif score >= p.bear_score and blocked_at is None:
    bear, low = True, spy_i
if bear:
    shown = -1
else:
    raw = 0 if score >= 1 else 1
    if shown == -1 or shown is None:  # leaving bearish, or the replay's first step: no delay
        shown, pending, count = raw, raw, 0
    elif raw != shown:
        count = count + 1 if raw == pending else 1
        pending = raw
        if count >= p.neutral_hold:
            shown, count = raw, 0
    else:
        count = 0
```

  The module docstring states the formula, the replay, and that nothing trades on it.

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/core/test_regime.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.** `git add src/trading_agent_framework/core/regime.py tests/core/test_regime.py && git commit -m "feat: Task 1 -- SPY/VIX stress-score regime classifier with bottom exit"`.
  Other modules still import the old fields until Task 3, so the full suite is red between these commits. Run only this file here.

---

### Task 2: Yahoo SPY/VIX/VIX3M series

**Files:**
- Create: `src/trading_agent_framework/core/regime_data.py`
- Modify: `src/trading_agent_framework/utils/errors.py` (add `class RegimeDataError(TradingFrameworkError)`)
- Test: `tests/core/test_regime_data.py`
- Create: `scripts/tests/smoke_regime_data.py` (manual, read-only)

**Interfaces:**
- Consumes: `extract_ticker_frame`, `parse_yahoo_frame`, `yfinance_download`, `MARKET_TZ` from `backtesting/data/yahoo.py`.
- Produces:
  - `RegimeSeries`: frozen dataclass. Fields: `sessions: tuple[date, ...]`, `spy: tuple[float, ...]`, `vix: tuple[float, ...]`, `vix3m: tuple[float, ...]` (VIX values may be `nan` where unfilled). `__len__`. Method `before(session: date, length: int) -> RegimeSeries`: the last `length` rows dated strictly before `session`. It raises `RegimeDataError` if a `nan` remains in the returned rows.
  - `RegimeSource(Protocol)`: `closes_before(session: date, length: int) -> RegimeSeries`.
  - `YahooRegimeData(*, download: DownloadFn | None = None, today: Callable[[], date] = date.today, max_fill: int = 5)`, with:
    - `series(start: date) -> RegimeSeries`: aligned on SPY's session dates from `start` through the download date;
    - `closes_before(session: date, length: int) -> RegimeSeries`: `series(session - timedelta(days=ceil(length * 1.5) + 10)).before(session, length)`.
  - Module constant `SYMBOLS = ("SPY", "^VIX", "^VIX3M")`.

- [ ] **Step 1: Write the failing tests.** Use a fake `download(symbols, **kwargs)` that returns a `group_by="ticker"` MultiIndex frame with `Open/High/Low/Close/Adj Close/Volume/Stock Splits` columns and records its calls. The tests:
  - `test_one_batched_download_of_the_three_symbols`: `series(start)` calls download once with `["SPY", "^VIX", "^VIX3M"]`, `auto_adjust=False`, `actions=True`, `group_by="ticker"`, `end = today + 1 day`.
  - `test_spy_closes_are_the_adjusted_closes`: `spy` equals the fake's `Adj Close`.
  - `test_the_series_is_aligned_on_spy_sessions_and_forward_fills_up_to_five`: with VIX missing on 3 of SPY's dates, those dates carry the previous VIX close.
  - `test_a_gap_longer_than_five_sessions_fails_the_slice`: a 6-session VIX gap yields `nan` in the series, and `before()` over it raises `RegimeDataError`.
  - `test_closes_before_excludes_the_session_itself`: the fake holds a row dated `session`; `closes_before(session, 3).sessions[-1] < session`.
  - `test_one_download_per_day_is_reused`: two `closes_before` calls on the same `today` with an earlier or equal start trigger one download. A new `today` triggers a second.
  - `test_a_failed_or_empty_download_raises_regime_data_error`: an exception from the fake raises `RegimeDataError`, and so does an empty frame.
  - `test_unsorted_or_duplicate_rows_are_normalised`: sessions come back strictly increasing.

- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/core/test_regime_data.py -v`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement `core/regime_data.py` and `RegimeDataError`.** Parse each symbol with `parse_yahoo_frame(extract_ticker_frame(raw, symbol))` and take `close`. Index by `.index.date`, `~index.duplicated(keep="last")`, sort. Reindex VIX and VIX3M on SPY's dates with `ffill(limit=max_fill)`. Cache `(downloaded_on, start, series)` and re-download when `today() != downloaded_on` or the requested start is earlier. The module docstring states that the download date is never a cutoff, only `before()` is.

- [ ] **Step 4: Run the tests to verify they pass.** Run `uv run pytest tests/core/test_regime_data.py -v`. Expected: PASS.

- [ ] **Step 5: Write `scripts/tests/smoke_regime_data.py`.** It prints the last 5 aligned rows of `YahooRegimeData().closes_before(date.today(), 503)`, the length (expect 503), and `classify_regime(...)` for today. Run it once by hand: `uv run python scripts/tests/smoke_regime_data.py`. Expected: 503 rows, the last dated before today, and a reading.

- [ ] **Step 6: Commit.** `git commit -m "feat: Task 2 -- Yahoo SPY/VIX/VIX3M series for the regime"`.

---

### Task 3: Wire the regime into `Strategy` (every strategy, every mode)

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py:26`, `:53-56`, `:115-117`, `:155-157`, `:354-381`, `:667-706`
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py:297-330` (delete `_refresh_regime` and its now-unused imports)
- Modify: `tests/fakes.py` (add `FakeRegimeSource`), `tests/conftest.py` (autouse fixture)
- Modify tests: `tests/core/test_strategy_regime.py` (rewrite), `tests/backtesting/test_regime_backtest.py`, `tests/strategies/test_backtest_defaults.py:103-104`, `tests/strategies/test_cross_momentum_yahoo_bars.py:487-540` (delete the four regime tests)
- Modify docs: `src/trading_agent_framework/core/CLAUDE.md` (regime gotcha), root `CLAUDE.md` (regime bullet), `src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md` (drop the "market regime is fetched the same way" sentence)

**Interfaces:**
- Consumes: `classify_regime`, `RegimeParameters`, `REGIME_LABELS`, `REGIME_LINE` (Task 1); `RegimeSource`, `RegimeSeries`, `YahooRegimeData` (Task 2); `RegimeDataError`.
- Produces:
  - `Strategy.regime_source: RegimeSource`, set in `__init__` to `YahooRegimeData()` (no network until the first refresh); assignable in tests.
  - `DAILY_WARMUP_FLOOR = 283` in `core/strategy.py`, replacing `REGIME_WARMUP_SLACK`. `run_backtesting` keeps today's behaviour exactly: a daily run's warmup is raised to at least 283 sessions, and other timesteps preload the benchmark's daily frame over `warmup_calendar_days(283)`. The regime no longer needs either. The comment says they are kept because strategies (for example earnings_drift's recorded 283) and the runner's daily benchmark read rely on them; removing them is a separate decision.
  - `tests.fakes.FakeRegimeSource(series: RegimeSeries | None = None, error: Exception | None = None)`: `closes_before` returns `series.before(...)`, an empty series by default, or raises `error`; it records `calls: list[tuple[date, int]]`.

- [ ] **Step 1: Write the failing tests.**
  - `tests/conftest.py`: an autouse fixture monkeypatches `trading_agent_framework.core.strategy.YahooRegimeData` to `lambda: FakeRegimeSource()`, so no test reaches Yahoo. The default empty series reproduces today's "unavailable, warned once" path.
  - `tests/core/test_strategy_regime.py` (rewrite), using a `FakeRegimeSource` over a 503-row synthetic series. Each test:
    - `test_the_regime_is_unknown_before_the_first_refresh`
    - `test_a_refresh_reads_503_closes_strictly_before_the_session_date`: `source.calls == [(date(2026, 9, 14), 503)]` for clock `et(2026, 9, 14, 8, 30)`; also at a mid-session clock `et(2026, 9, 14, 10, 15)`.
    - `test_a_refresh_sets_logs_and_charts_the_regime`: `lines == [("Regime", -1, "Regime")]` for a bearish series. The log contains `"Market regime -1 (bearish)"`, `"score "`, `"VIX3M"`, `"SMA50"`, `"52w high"` and `"episode low"`.
    - `test_the_regime_never_reads_the_strategys_own_bars`: `broker.bars_calls == []`.
    - `test_the_regime_ignores_benchmark_symbol`: a subclass with `benchmark_symbol = "QQQ"` still asks the source, and the log names `SPY`.
    - `test_a_data_failure_keeps_the_previous_regime` (parametrised `RegimeDataError`, `ValueError`): regime stays at its prior value, and the log has `"Market regime not refreshed (still -1)"`.
    - `test_too_little_history_leaves_the_regime_unknown_and_warns_once`
    - `test_outside_backtesting_the_regime_is_set_and_nothing_is_charted`
  - `tests/backtesting/test_regime_backtest.py`:
    - Rename `REGIME_WARMUP_SLACK` uses to `DAILY_WARMUP_FLOOR`. Expected warmups stay `(0, 283), (10, 283), (400, 400)`. The preload window becomes `warmup_calendar_days(DAILY_WARMUP_FLOOR)`.
    - Rewrite `test_a_daily_backtest_has_a_regime_from_its_first_session_and_never_looks_ahead` to inject a `FakeRegimeSource` whose series crashes on the last session's own date. The last session's regime must not see it.
    - Delete `test_the_regime_warmup_covers_min_bars_sessions_for_every_2025_start_date`; its subject is gone.
  - `tests/strategies/test_backtest_defaults.py:103`: `regime_floor = DAILY_WARMUP_FLOOR`.

- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/core/test_strategy_regime.py tests/backtesting/test_regime_backtest.py tests/strategies/test_backtest_defaults.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.**
  - `_refresh_regime`: the session is `self.clock.now().astimezone(MARKET_TZ).date()`. Read `self.regime_source.closes_before(session, params.min_bars)`, then call `classify_regime(series.spy, series.vix, series.vix3m, params)`. Catch `(RegimeDataError, ValueError)`. The unavailable warning names `SPY/VIX/VIX3M` and the session. The INFO line carries the score and the four components with their values (SPY close vs SMA50, SPY vs 52w high in %, VIX vs VIX3M, VIX vs 1y median as a ratio), plus `episode low` when bearish. Then `add_line(REGIME_LINE, ...)` as today.
  - The `regime_params` comment says the regime always reads Yahoo SPY/VIX/VIX3M.
  - Delete cross_momentum's override.
  - Update the three CLAUDE.md passages to describe the new formula, the Yahoo source, the replay, and `DAILY_WARMUP_FLOOR`.

- [ ] **Step 4: Run the whole suite.** Run `uv run pytest`. Expected: all pass. Any other test still asserting the old log text or the old fields is updated, not deleted, unless its subject is gone.

- [ ] **Step 5: Commit.** `git commit -m "feat: Task 3 -- every strategy reads the SPY/VIX regime from Yahoo"`.

---

### Task 4: The scorecard and the `regime` CLI

**Files:**
- Create: `src/trading_agent_framework/core/regime_scorecard.py` (pure: lists and dates in, dataclasses out)
- Create: `src/trading_agent_framework/core/regime_cli.py` (`main()`, argparse subcommands)
- Modify: `pyproject.toml` `[project.scripts]` (add `regime = "trading_agent_framework.core.regime_cli:main"`), root `CLAUDE.md` Commands (two lines: `uv run regime scorecard ...`, `uv run regime recompute ...`)
- Test: `tests/core/test_regime_scorecard.py`, `tests/core/test_regime_cli.py`

**Interfaces:**
- Consumes: `regime_by_session`, `regime_path`, `RegimeParameters` (Task 1); `YahooRegimeData`, `RegimeSeries` (Task 2).
- Produces:
  - `Episode(peak: date, trough: date, depth: float)`.
  - `zigzag(sessions: Sequence[date], closes: Sequence[float], threshold: float) -> list[Episode]`: a swing high to swing low fall of at least `threshold`, ending on a `threshold` rebound off the low. An unfinished fall at the end counts.
  - `reference_labels(sessions, closes, bear=0.08, neutral=0.04) -> dict[date, int]`: -1 on `[peak, trough]` of bear episodes, 0 on 4% episodes not already bearish, else 1.
  - `EpisodeScore(episode: Episode, bear_on_fall: float | None, coverage: float, overstay: int)`.
  - `Scorecard(episodes: list[EpisodeScore], recall: float, false_bear_days: int, mean_on_fall: float, median_overstay: float, changes: int)`.
  - `score(sessions: Sequence[date], closes: Sequence[float], regimes: Mapping[date, int]) -> Scorecard`, with these definitions (they match the spike):
    - `bear_on_fall`: SPY's fall from the peak at the first -1 session in `[peak, trough]`, or `None` if there is none.
    - `mean_on_fall` averages `bear_on_fall`, using the episode's depth for `None`.
    - `coverage`: the share of `[peak, trough]` sessions at -1.
    - `overstay`: the run of consecutive -1 sessions strictly after the trough.
    - `false_bear_days`: -1 sessions outside every `[peak, trough + overstay]`.
    - `recall`: the share of reference -1 sessions read -1.
    - `changes`: the sessions whose regime differs from the previous session's.
  - The CLI maps `regime_by_session`'s readings to `{date: reading.regime}` before calling `score` (and `recompute_run` in Task 5).
  - CLI: `regime scorecard [--start 2016-01-04] [--end 2026-10-05]` downloads from `start - 800 days`, computes `regime_by_session`, scores the window, and prints the per-episode table and the summary. It also prints `replay disagreements: N`, the number of window sessions where `regime_by_session` differs from one continuous `regime_path(first=lookback - 1)` over the whole series (reading `i` applies to session `i + 1`).

- [ ] **Step 1: Write the failing tests.**
  - `test_zigzag_finds_falls_from_recent_highs`: on closes `[100, 110, 99, 105, 95, 106, 115, 100, 130]` with threshold 0.08 it returns peak 110 → trough 95 (depth -13.6%) and 115 → 100. The 105 bounce (+6%) does not end the first swing.
  - `test_reference_labels_mark_bear_neutral_and_bull`
  - `test_score_counts_switch_on_coverage_overstay_and_false_alarms` uses a hand-labelled 20-session example, asserting each `Scorecard` field.
  - `test_an_episode_never_read_bearish_uses_its_full_depth`
  - `test_the_cli_scorecard_prints_the_summary` uses a monkeypatched `YahooRegimeData` returning a synthetic series and checks that stdout contains `recall`, `false-alarm bear days`, `regime changes` and `replay disagreements`.

- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/core/test_regime_scorecard.py tests/core/test_regime_cli.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `regime_scorecard.py`, `regime_cli.py` (scorecard subcommand) and the entry point.** Run `uv sync` after editing `pyproject.toml`.

- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.

- [ ] **Step 5: Commit.** `git commit -m "feat: Task 4 -- regime scorecard against SPY swing episodes"`.

---

### Task 5: Recompute existing backtest runs

**Files:**
- Create: `src/trading_agent_framework/backtesting/regime_recompute.py`
- Modify: `src/trading_agent_framework/core/regime_cli.py` (add the `recompute` subcommand), `src/trading_agent_framework/backtesting/CLAUDE.md` (one gotcha: the command, the backup name, that `backtest.log` keeps old regime lines)
- Test: `tests/backtesting/test_regime_recompute.py`

**Interfaces:**
- Consumes: `regime_by_session` (Task 1), `YahooRegimeData` (Task 2), `REGIME_LINE`, the `indicators.parquet` schema of `report.write_indicators` (`datetime` as `datetime64[us, America/New_York]`, `name`, `value` float64, `color`, `style`, `plot_name`).
- Produces:
  - `BACKUP_NAME = "indicators.pre-vix-regime.parquet"`, `REGIME_TIME = time(8, 30)` (America/New_York), `REGIME_COLOR = "#d1d4dc"`.
  - `RecomputeResult(run_dir: Path, status: Literal["rewritten", "added", "skipped"], reason: str = "", rows: int = 0)`.
  - `run_dirs(logs_root: Path) -> list[Path]`: sorted `logs_root/*/backtesting/*/` directories.
  - `run_sessions(run_dir: Path) -> list[date] | None`: the dates of `equity.parquet`'s index in New York time, or `None` if it is absent.
  - `recompute_run(run_dir: Path, readings: Mapping[date, int], *, dry_run: bool = False) -> RecomputeResult`.
  - CLI: `regime recompute [--logs-root logs] [--dry-run]`. It downloads once from the earliest run session minus 800 days, computes `regime_by_session`, then calls `recompute_run` per run and prints one line per run plus totals.

- [ ] **Step 1: Write the failing tests** (tmp run directories built with small parquet files):
  - `test_regime_rows_are_replaced_and_other_lines_kept`: a run with a `Regime` line and an `Other` line comes back with the `Other` rows unchanged. The `Regime` rows are one per session in `readings`, at the run's existing time of day, with the given values. Status is `"rewritten"`.
  - `test_a_run_without_a_regime_line_gets_one_at_08_30`: status `"added"`, rows stamped at 08:30 America/New_York, color `#d1d4dc`, plot `Regime`.
  - `test_a_run_without_indicators_gets_a_new_file`: status `"added"`, no backup written.
  - `test_the_written_schema_matches_write_indicators`: dtypes are equal to a frame written by `report.write_indicators`.
  - `test_recompute_is_idempotent_and_keeps_the_first_backup`: two runs leave the backup equal to the original file and the Regime rows identical.
  - `test_a_run_without_equity_is_skipped_untouched`: status `"skipped"`, reason mentions `equity.parquet`, files unchanged.
  - `test_sessions_without_a_reading_get_no_row`
  - `test_dry_run_writes_nothing`
  - `test_the_cli_recompute_reports_each_run` uses a monkeypatched `YahooRegimeData`.

- [ ] **Step 2: Run them to verify they fail.** Run `uv run pytest tests/backtesting/test_regime_recompute.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.** Write the backup only if it does not exist (copy the current file). Write the new file to a sibling `.tmp` and `os.replace` it. Rows are sorted by `datetime` with a stable sort.

- [ ] **Step 4: Run the tests to verify they pass.** Same command, then `uv run pytest`. Expected: PASS.

- [ ] **Step 5: Commit.** `git commit -m "feat: Task 5 -- recompute the Regime line of existing backtest runs"`.

---

### Task 6: Acceptance on real data

**Files:** none changed unless a check fails. The recompute rewrites files under `logs/`, which is not tracked; check with `git check-ignore logs` and do not commit `logs/`.

- [ ] **Step 1: Run the scorecard.** Run `uv run regime scorecard --start 2016-01-04 --end 2026-10-05`. Expected (the spike's Finalist 1, Yahoo revisions allowed):
  - 13 episodes;
  - recall 0.584 ± 0.03;
  - false-alarm bear days 86 ± 20;
  - mean fall at switch-on −5.8% ± 0.6 pp;
  - median overstay 6 ± 2;
  - regime changes below 261 (the neutral hold lowers it);
  - replay disagreements reported.

  If recall falls outside the band, or the changes don't drop, report it and stop; don't tune. If changes don't drop or recall moves with the hold, set `neutral_hold=1` per the spec and say so.
- [ ] **Step 2: Dry-run the recompute.** Run `uv run regime recompute --dry-run`. Expected: 61 runs listed; about 40 "rewritten", about 21 "added", and a reason for any "skipped".
- [ ] **Step 3: Run the recompute.** Run `uv run regime recompute`, then run it a second time. Expected: the same counts, and the second run changes no Regime rows.
- [ ] **Step 4: Check acceptance 3.** In `logs/cross_momentum/backtesting/2026-11-07_152938_backtesting/indicators.parquet`, the `Regime` rows must include -1 within 2018-10..2018-12, 2020-02-24..2020-03-23, 2022-04..2022-06 and 2025-03..2025-04. Print one line per window.
- [ ] **Step 5: Check acceptance 2.** Write `scripts/tests/smoke_regime_backtest.py`. It runs a do-nothing `Strategy` subclass with `run_backtesting(start=2025-03-03, end=2025-04-30, timestep="day", data_source=YahooBacktestData, budget=10000)` and the real `YahooRegimeData`, in a temporary project root so nothing lands in `logs/`. It then asserts that the run's `Regime` values equal `regime_by_session(...)[session].regime` for every session. Run it with `uv run python scripts/tests/smoke_regime_backtest.py`. Expected: "regime matches on N sessions", where N is about 42.
- [ ] **Step 6: Look at the dashboard.** Run `uv run dashboard`, open the 10Y cross_momentum run's Trades chart, and screenshot the regime bands for the user.
- [ ] **Step 7: Run the quality gate.** Run the six commands of Global Constraints and report the numbers.
- [ ] **Step 8: Commit** any smoke script added in Step 5: `git commit -m "test: Task 6 -- regime backtest smoke check"`.

# Strategies Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplication and dead weight the 2026-11-06 `/simplify` review found in `src/trading_agent_framework/strategies/` (plus the backtest plumbing in `core/strategy.py`), without changing what any strategy trades.

**Architecture:**
- Code shared by several strategies moves into `strategies/common/`.
  - New modules: `json_state.py`, `pipeline.py` (the abstract `ReviewPipeline`), `stops.py`, `cooldowns.py`, `universe_params.py`.
  - Moved modules: `batch_us_stock_universe.py`, `universe_backtest_data.py`.
  - Extended modules: `handoff.py`, `run_log.py`, `universe_history.py`, `yahoo_daily_bars.py`.
- Run-directory, state-file and backtest-default plumbing moves into `core/strategy.py`.
- Each strategy keeps only its own validation, prompts, agent runs (`_run_stage` is NOT shared) and decision logic.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff, pandas.

**Spec:** this plan's "Source findings" and "User decisions" sections.

## User decisions (2026-11-06)

- Shared code lives in `strategies/common/`. The run/state paths and backtest defaults live in `core/strategy.py`.
- **`_run_stage` is not factorized.** Instead, `strategies/common/pipeline.py` holds an abstract class `ReviewPipeline`. Its abstract methods and its template `run()` carry only what ALL FOUR agent pipelines share: bill_ackman, bull_bear, congress_trades and insider_trades. Nothing goes in on a baseline that is not common to the four.
  - Exception (explicitly allowed): the pending-trade steps only congress_trades and insider_trades share go in `ReviewPipeline` as concrete helper methods. bill_ackman and bull_bear never call them, and their code must not change because of them (no new attribute, argument or hook they have to provide).
  - `retry_prompt` (3 of 4) stays in each strategy.
- `cross_momentum/batch_us_stock_universe.py` moves to `strategies/common/`, together with `cross_momentum/universe_backtest_data.py`. The universe thresholds (`min_price`, `min_dollar_volume`, `min_market_cap`) move to `strategies/common/universe_params.py`, which cross_momentum's `CONFIG` reads, so `common/` never imports a strategy.
- vwap_pullback adopts earnings_drift's stop-release rules (Task 11), an approved **BEHAVIOR CHANGE**.
- `advance_cooldowns` is shared (bill_ackman and bull_bear).

## Global Constraints

- Work on branch `refactor/strategies-simplify` (create it from `main` before Task 1). Never push or merge; the user squashes into `main` themselves.
- Behavior-preserving unless a step says **BEHAVIOR CHANGE**. Baseline: `uv run pytest` = 4577 passed, `uv run ruff check` clean. Both must hold after every task (the count may only grow).
- Model-visible text (handoff error messages, retry prompts, tool docstrings) stays byte-identical. Tests pin it.
- Log message text and `runs.jsonl` / `reviews.jsonl` record keys stay identical unless a step names the change. The dashboard reads `reviews.jsonl`.
- `strategies/common/handoff.py` must NOT use `from __future__ import annotations` (the agent layer reads real annotations).
- `strategies/common/` must not import any `strategies/<strategy>/` module.
- No re-export shims (`strategies/common/CLAUDE.md`): when something moves, update every importer, tests and `scripts/` included.
- Money stays `Decimal`; no new float boundary.
- No `time.sleep` / `datetime.now` in strategy or common code.
- Tests use hand-written fakes, never `MagicMock`; no network.
- Each task updates the `CLAUDE.md` files that name what it moved, renamed or removed (`strategies/common/CLAUDE.md`, the strategies' own, `core/CLAUDE.md`, `backtesting/CLAUDE.md`).
- Commit message per task: `refactor(<area>): Task N - <summary>`, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A state file written by the old code loads the same through the new store.** All five stores, including bull_bear's version-1 and version-2 files. Test: Task 5, `test_old_files_load_unchanged`.
2. **A strategy's `run_backtesting()` with no overrides passes the same window, budget, benchmark, slippage and warmup to `run_backtest` as before.** Test: Task 4, `test_strategy_backtest_defaults_unchanged`.
3. **An abandoned review or run still writes the same log line, `reviews.jsonl`/`runs.jsonl` record (keys and key order) and saved state in all four pipelines.** That covers the broker/execution split and congress/insider's `filings` stage. Test: Task 7, `test_abandoned_record_unchanged` in each pipeline's test file.
4. **The text the model sees is byte-identical after the handoff refactor.** Test: Task 6, `test_guard_messages_unchanged`.
5. **`intraday_contexts` gives identical output on gappy minute data.** Test: Task 3, `test_intraday_contexts_matches_reference`.

## Source findings (what each task fixes)

- Duplicated across strategies:
  - The JSON `StateStore` (×5) and the atomic JSON write (×3 more).
  - `ReviewLog`/`JsonlLog` (copies of `RunLog`).
  - The submit recorder skeleton (×4).
  - The pipelines' `run()` abandon handling, the agent protocols, the abandon exceptions and the outcome types (×4).
  - The handoff field readers in bill_ackman and bull_bear.
  - The congress/insider pending-trade twins.
  - `_market_day` and the point-in-time universe wiring.
  - The Yahoo retry and coverage check.
  - `_release_stop`.
  - `advance_cooldowns`.
  - The `logs/<name>/<mode>/<run_id>/<file>` path (×6).
  - The backtest defaults in every `run_backtesting`.
- Dead code and simplifications inside one strategy: Tasks 1-2.
- Hot paths: `vwap_pullback/features.intraday_contexts` (about 70 s of CPU per simulated session), `common/sessions.completed_bars`, and `bill_ackman/screen/splits.py` writing once per symbol.

**Out of scope (proposed, not done here):**
- Executor `FatalStrategyError` from every hook.
- A daily-bars provider seam on `Strategy`.
- Parallel and cached earnings_drift news.
- A SEC ticker→CIK memo in `fundamentals/`.
- The Alpaca rate limiter in cross_momentum.
- The `Rebalancer` treating an IBKR short as a holding: a possible bug, left to `/code-review`.
- bull_bear never resetting its pipeline `_stage` to `"setup"` between reviews: a possible bug, left to `/code-review`; the refactor keeps today's behavior.

---

### Task 1: cross_momentum dead code and local simplifications

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/utils.py`, `agent_cross_momentum.py`, `risk_diagnostics.py`, `log_diagnostics.py` (`batch_us_stock_universe.py` is handled in Task 9)
- Test: `tests/strategies/test_cross_momentum_*.py`

**Interfaces:**
- Produces: `CrossMomentumStrategy._send(side: str, quantity: Decimal, symbol: str, why: str) -> tuple[Order | None, Exception | None, bool]`. It returns the submitted order (or None), the exception raised (or None), and `unknown` (True on `BrokerUncertainError`).

- [ ] **Step 1: Delete dead code.**
  - `utils.py`: delete `_percentile_rank`, `compute_residual_momentum`, `compute_atr_from_df` and `_EQUITY_HISTORY_FILE` (make `filepath` a required parameter of the equity-history helpers).
  - `risk_diagnostics.py`: delete `compute_drawdown`.
  - `agent_cross_momentum.py`: drop the `"atr"` entry (about line 333) and remove the `"atr"` key from the fixture at `tests/strategies/test_cross_momentum_breadth.py:286`.
  - Before deleting each name, grep `src tests scripts` to confirm it has no caller.
- [ ] **Step 2: Simplify `utils.py` and the diagnostics.**
  - `diagnostics_to_dict(diag)` becomes `dataclasses.asdict(diag)` (inline it if there is one caller).
  - `save_equity_history` takes the value it persists (`history[-1]`), not the list.
  - Remove the `fractional_qty` / `parse_insufficient_buying_power` re-export at `utils.py:19`; `agent_cross_momentum.py` imports them from `trading_agent_framework.utils.helpers`.
  - `log_diagnostics._compute_portfolio_daily_returns`: weights are already normalised by `_build_weight_array`, so compute `pd.Series(returns_df.to_numpy() @ weights, index=returns_df.index)`.
  - `risk_diagnostics.py`: replace the weighted-correlation double loop (about lines 164-171) with `iu = np.triu_indices(n, 1); w = np.outer(weights, weights)[iu]; (w * corr[iu]).sum() / w.sum()`, and delete the unreachable truncate branch of `_normalize_weights`.
- [ ] **Step 3: Refactor `_rebalance_book` (about lines 592-841).**
  - Move the five copies of `try: submit_order(create_order(...)) / except BrokerUncertainError / except Exception` into `_send`. Keep each site's log text exactly; the exit site still counts `unknown` in `unconfirmed_count`, and the buy site still feeds the exception to `parse_insufficient_buying_power`.
  - Split the body into `_sell_phase`, `_park_targets` and `_buy_phase`, with the same order of operations.
  - Build `positions_by_symbol` once instead of the `next(...)` scan inside the buy loop.
- [ ] **Step 4: Small cuts in `agent_cross_momentum.py`.**
  - Move the `get_cross_momentum_universe_last_date()` call (about line 857) inside its paper/live branch.
  - Keep the frame on each scan entry and call `close_series` only for `selected[:top_n]`. Drop `"closes"` if every reader can use the series.
  - Pass `self.parameters` (not `dict(...)`) to `apply_filters`.
  - Add a `_sleeve_symbols()` helper for the three `set(sleeve_symbols(self.parameters["parking"]))` sites.
  - Drop the `vt_config.get(..., default)` defaults `CONFIG` always provides.
  - Remove the unreachable `return False` (about line 368) and the impossible `hasattr`/`isinstance` guards (about lines 309-312).
- [ ] **Step 5: Run** `uv run pytest tests/strategies -q && uv run ruff check`. Expected: all pass.
- [ ] **Step 6: Dispatch `code-simplifier:code-simplifier`** (low) on the diff, re-run Step 5.
- [ ] **Step 7: Commit** `refactor(cross_momentum): Task 1 - remove dead code, split _rebalance_book`.

### Task 2: Local simplifications, and `advance_cooldowns` in common

**Files:**
- Create: `strategies/common/cooldowns.py`, `tests/strategies/common/test_common_cooldowns.py`
- Modify:
  - `strategies/common/trade_desk.py`
  - `strategies/earnings_drift/desk.py`
  - `strategies/vwap_pullback/desk.py`
  - `strategies/bull_bear/pipeline.py`; delete `strategies/bull_bear/cooldown.py`
  - `strategies/insider_trades/pipeline.py`
  - `strategies/bill_ackman/pipeline.py`, `bill_ackman/fact_sheet.py`, `bill_ackman/hysteresis.py`
  - `strategies/news_builtin/agent_news_binary.py`
- Test: the matching `tests/strategies/...` directories

**Interfaces:**
- Consumes: `common.rebalancer.in_flight_quantities(open_orders)` and `common.scoring.compute_return_from_prices(closes, lookback, skip)`.
- Produces: `common.cooldowns.advance_cooldowns(...)`, with the same signature and behavior as today's bull_bear version. If the two copies' parameter names differ, use bull_bear's and update bill_ackman's callers.

- [ ] **Step 1: Write the failing test** `test_common_cooldowns.py`: move bull_bear's and bill_ackman's existing `advance_cooldowns` tests into it, importing from `strategies.common.cooldowns`. Run it: FAIL (module missing).
- [ ] **Step 2: Create `common/cooldowns.py`.** Delete `bull_bear/cooldown.py` and `hysteresis.advance_cooldowns`, and update the importers. Run: PASS.
- [ ] **Step 3: `common/trade_desk.py`.**
  - `_book`'s inline loop (about lines 181-187) becomes `in_flight_quantities(strategy.broker.tracker.get_active_orders())`.
  - Replace `_target: dict` plus `_has_target: bool` with `_target: dict[str, float] | None`.
- [ ] **Step 4: `earnings_drift/desk.py`.**
  - `_stop_is_working` uses `self._fetch(id, sym)[1]`; delete `_lookup`.
  - Extract `_open_trade_or_refusal(symbol, action) -> Trade | dict` from the preamble of `set_trailing_stop` and `sell`.
  - Extract `_trail_in_range(trail) -> bool`.
  - Remove the intermediate `record` dict (about lines 486-497).
  - Do NOT merge the order-ended branches.
- [ ] **Step 5: `vwap_pullback/desk.py`.**
  - Add `_stop(trade) -> Order | None` for the three `get_order(trade.stop_order_id) if ... else None` sites.
  - Add `_drop_ended_stop(trade, order) -> bool` for the duplicated block in `_reprotect` and `_sell_unprotected`.
  - Flatten `reconcile`'s nested condition (about lines 363-367).
- [ ] **Step 6: `bull_bear/pipeline.py`.**
  - Use one `score_stock` call with `params = self._holding_momentum if held else self._momentum`.
  - Split `_review` into `_rank`, `_research_notes` and `_debate_and_judge`, with the same order and record keys.
  - Rename the re-bound `now` (about line 225) to `as_of`.
- [ ] **Step 7: `insider_trades/pipeline.py`.** Compute `reasons = {ticker: self._change_reasons(...)}` once in `_run`, derive `changed` from it, pass `reasons` to `_full_run`, and delete `_changed`.
- [ ] **Step 8: `bill_ackman`.**
  - Rename the comprehension variable `record` (about line 158 of `pipeline.py`) to `entry`.
  - Split `_review` into researcher, short-seller and trader steps.
  - Replace `fact_sheet.price_return(closes)` with `compute_return_from_prices(closes, 252, 0)` and delete `price_return`, after checking that every caller uses a 252 lookback.
- [ ] **Step 9: `news_builtin`.** Delete the `self.vars.strategy_parameters` alias; readers use `self.strategy_parameters`.
- [ ] **Step 10: Run** `uv run pytest tests/strategies -q && uv run ruff check`.
- [ ] **Step 11: Run `code-simplifier`** (low), re-run, update the CLAUDE.md files (bill_ackman and bull_bear name `advance_cooldowns`), commit `refactor(strategies): Task 2 - local simplifications, shared advance_cooldowns`.

### Task 3: Hot paths: `intraday_contexts`, `completed_bars`, the splits cache

**Files:**
- Modify: `strategies/vwap_pullback/features.py:109-170`, `strategies/common/sessions.py:42-49`, `strategies/bill_ackman/screen/splits.py:131-160` and its caller in `screen/screen.py`
- Test: the vwap features test file, `tests/strategies/common/test_common_sessions.py` (create if absent), the splits tests

**Interfaces:**
- Produces: `SplitHistory.flush() -> None`, which writes the cache once if it changed. `QualityScreen.run` calls it in a `finally`.

- [ ] **Step 1: Write the equivalence test.**
  - Copy today's `intraday_contexts` body into the test module as `_reference_intraday_contexts`.
  - Write `test_intraday_contexts_matches_reference`, parametrized over:
    - full 60 / 200 / 390-minute sessions;
    - a bucket missing its last minute, with `now` both within and beyond one minute after its close;
    - a bucket with no rows;
    - an empty `benchmark_df`;
    - `now` in the middle of a bucket.
  - Assert the two lists are equal, using `pytest.approx` per float field.
- [ ] **Step 2: Run it.** It passes on today's code; it pins behavior before the rewrite.
- [ ] **Step 3: Vectorise `intraday_contexts`.**
  - One `df.groupby(buckets).agg(open=("open","first"), high=("high","max"), low=("low","min"), close=("close","last"), volume=("volume","sum"))`, plus the last index label per bucket.
  - Apply the completeness gate to the bucket array in order (stop at the first incomplete bucket).
  - Read `vwaps`, `cumulative` and `mos` at the last labels, and get `session_high` with `np.maximum.accumulate`.
  - Build the `BarContext` list in one comprehension.
- [ ] **Step 4: Run** `uv run pytest tests/strategies/vwap_pullback -q`. Expected: PASS.
- [ ] **Step 5: `completed_bars`.**
  - First add `test_completed_bars_drops_only_today`: bars over 3 days, with today's bar at 00:00 and at 16:00 market time, in a UTC index. It must pass on today's code.
  - Then replace the list comprehension with `idx = pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ).normalize(); return df.loc[idx != pd.Timestamp(today, tz=MARKET_TZ)]`. Re-run.
- [ ] **Step 6: Splits cache.**
  - `SplitHistory` marks itself dirty instead of calling `_save` after each symbol; `flush()` saves once.
  - Add `test_splits_written_once_per_run` (3 stale symbols, count the writes).
- [ ] **Step 7: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 8: Commit** `perf(strategies): Task 3 - vectorise intraday_contexts and completed_bars, write splits once`.

### Task 4: `core/strategy.py`: run file, state file, backtest defaults

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py:96-102, 528-640`, `core/CLAUDE.md`, the 8 `strategies/*/agent_*.py` `run_backtesting` overrides and their `_run_file` / `_run_log_path` / `_review_log_path` / trades-path helpers, and each strategy's `parameters` dict (drop `"slippage"`)
- Test: `tests/core/test_strategy.py`, new `tests/strategies/test_backtest_defaults.py`

**Interfaces:**
- Produces:
  - `Strategy.run_file(self, name: str) -> Path | None`: `None` when `run_id is None`, else `project_root / "logs" / self.name / self.trading_mode.value / run_id / name`.
  - `Strategy.state_file(self, name: str | None = None) -> Path`: `project_root / "data" / f"{name or self.name}_state_{self.trading_mode.value}.json"`.
  - `Strategy.backtesting_slippage: Decimal = Decimal("0.0005")`: the one default, 5 bps per side.
  - `run_backtesting(..., slippage: Number | None = None, warmup_trading_days: int | None = None, ...)`. Each of start, end, budget, benchmark, slippage and warmup resolves as: argument, then `self.parameters[key]`, then the class attribute (warmup: 0). The keys are `"backtesting_start"`, `"backtesting_end"`, `"budget"`, `"benchmark_symbol"`, `"slippage"` and `"warmup_trading_days"`.

- [ ] **Step 1: Write the defaults test** (must PASS on today's code):

```python
# tests/strategies/test_backtest_defaults.py
@pytest.mark.parametrize("strategy_name", [s.value for s in Strategies])
def test_strategy_backtest_defaults_unchanged(strategy_name, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("trading_agent_framework.backtesting.runner.run_backtest", lambda strategy, **kw: captured.update(kw))
    # build the strategy in BACKTESTING mode with fakes (reuse each strategy's test harness;
    # stub ensure_universe_history for cross_momentum and bull_bear)
    ...
    strategy.run_backtesting()
    p = strategy.parameters
    assert captured["start"] == p["backtesting_start"] and captured["end"] == p["backtesting_end"]
    assert captured["budget"] == Decimal(str(p["budget"]))
    assert captured["benchmark"] == p["benchmark_symbol"]
    assert captured["slippage"] == Decimal("0.0005")
    assert captured["warmup_trading_days"] >= p["warmup_trading_days"]
```

  If a strategy cannot be built cheaply, rely on its existing backtest test and say which in the commit message.

- [ ] **Step 2: Write the failing core tests** in `tests/core/test_strategy.py`:
  - `test_run_file_none_without_run_id`
  - `test_run_file_path`
  - `test_state_file_default_and_named`
  - `test_run_backtesting_reads_parameters`
  - `test_run_backtesting_default_slippage_5bps`
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement** the three members and the resolution; update the `run_backtesting` docstring.
- [ ] **Step 5: Run** `uv run pytest tests/core tests/backtesting -q`. A toy-strategy test whose numbers relied on zero slippage sets `backtesting_slippage = Decimal(0)` on its toy class; list each one in the commit message.
- [ ] **Step 6: Adopt it in the strategies.**
  - Each override drops `start`, `end`, `budget`, `benchmark`, `slippage` and `warmup_trading_days` from its defaults, and drops `"slippage"` from its `parameters`.
  - Each override keeps `data_source`, `preload_assets`, `timestep` and `agent_telemetry`.
  - The six `logs/.../<file>` helpers become `self.run_file("<file>")`.
  - earnings_drift and insider_trades read the merged window as `overrides.get("end", self.parameters["backtesting_end"])` (and `start` likewise).
- [ ] **Step 7: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 8: Update `core/CLAUDE.md`, commit** `refactor(core): Task 4 - Strategy.run_file/state_file and backtest defaults from parameters`.

### Task 5: `common/json_state.py` and a hardened `RunLog`

**Files:**
- Create: `strategies/common/json_state.py`, `tests/strategies/common/test_json_state.py`
- Modify:
  - `strategies/common/run_log.py`
  - the stores: `bill_ackman/state.py`, `bull_bear/state.py`, `congress_trades/state.py`, `insider_trades/state.py`, `earnings_drift/book.py`
  - `vwap_pullback/desk.py` (`_archive`)
  - `bill_ackman/screen/annual_store.py:~147`, `bill_ackman/screen/splits.py:~159`, `congress_trades/congress/clerk_client.py:~103`
  - the agents that build a store or a log
- Test: the existing state and log tests of those strategies

**Interfaces:**
- Consumes: `Strategy.state_file(name)` and `Strategy.run_file(name)` (Task 4).
- Produces:
  - `atomic_write_json(path: Path, payload: Any, *, log: ColorLogger, what: str = "state") -> bool`:
    - Serializes first (`json.dumps(payload, default=str)`), then writes with mkstemp → fdopen (closing the fd if fdopen fails) → `os.replace`.
    - On `OSError`/`TypeError`/`ValueError` it logs `f"{what} could not be saved to {path}: {exc}"`, removes the tmp file and returns False.
  - Generic state validators, public: `is_int(v) -> bool` (an int, not a bool), `is_weight(v) -> bool` (a finite number in [0, 1], not a bool), `valid_strings(v) -> bool`, `valid_weights(v) -> bool`. The bodies are today's congress/insider `_is_int` etc.
  - `class InvalidState(ValueError)`.
  - `class JsonStateStore[S]`:
    - Constructor: `__init__(self, path: Path, *, version: int, empty: Callable[[], S], decode: Callable[[dict[str, Any]], S], encode: Callable[[S], dict[str, Any]], log: ColorLogger, accepts: Collection[int] | None = None, fresh: bool = False)`.
    - `path`, `load() -> S`, `save(state: S) -> None`, `wipe() -> None`.
    - A missing file gives `empty()`.
    - A read `OSError`/`ValueError` logs `state file {path} is unreadable, starting from an empty state: {exc}`.
    - A non-dict, or a version not in `accepts` (default `{version}`), or `decode` raising `InvalidState`/`ValueError`/`KeyError`/`TypeError`/`AttributeError`: logs `state file {path} is not a valid state, starting from an empty state`, followed by `: {exc}` only when `str(exc)` is non-empty.
    - `save` writes `{"version": version, **encode(state)}` through `atomic_write_json`.
    - `fresh=True` wipes in `__init__`.
    - A failed wipe logs `state file {path} could not be deleted: {exc}`.
  - `RunLog(path: Path | Callable[[], Path | None] | None, *, what: str = "run log")`: serializes before opening, never raises on `OSError`/`TypeError`/`ValueError`, and logs `f"{what} {path} could not be written: {exc}"`.

- [ ] **Step 1: Write the failing tests** in `test_json_state.py`:
  - `test_missing_file_is_empty`
  - `test_round_trip`
  - `test_unreadable_file_warns_and_is_empty`
  - `test_wrong_version_is_empty`
  - `test_accepts_older_versions`
  - `test_decode_error_message_suffix`
  - `test_save_unserializable_leaves_no_tmp`: `encode` returns `{(1, 2): "x"}`, a tuple key that `json.dumps` rejects with `TypeError` even with `default=str`. Assert no `*.tmp` remains, nothing is raised, and a warning is logged.
  - `test_fresh_wipes`
  - `test_atomic_write_closes_fd_on_fdopen_failure`
  - `test_validators` (bool is not an int; NaN and 1.5 are not weights)
  - `test_run_log_callable_path`
  - `test_run_log_unserializable_never_raises`
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run: PASS.
- [ ] **Step 4: Write `test_old_files_load_unchanged`** in each strategy's state test file.
  - Use a literal dict equal to what the OLD `save()` wrote for a fully populated state; bull_bear also gets version-1 and version-2 literals.
  - Load it and assert the expected dataclass. It must PASS on today's code.
- [ ] **Step 5: Port each store.**
  - Make it a subclass `class StateStore(JsonStateStore[XState])` with `__init__(self, path: Path, *, fresh: bool = False)` that passes its `version`, `empty`, `decode` (`_valid` → `raise InvalidState()`, then the field mapping) and `encode`.
  - bull_bear passes `accepts={1, 2, 3}` and keeps its version-1 branch in `decode`.
  - congress and insider use the shared validators. `PendingTrade` and `_valid_pending` stay where they are until Task 8.
  - Delete each `state_path`. The agents build `StateStore(self.state_file("<name>"), fresh=self.is_backtesting)` and drop their `if is_backtesting: wipe()`. earnings_drift passes `self.state_file()`; the others pass their literal name so file names do not change.
- [ ] **Step 6: Port the logs.**
  - Delete `ReviewLog` (bill_ackman, bull_bear) and `JsonlLog` (earnings_drift).
  - Use `RunLog(self.run_file("reviews.jsonl"), what="review log")`. earnings_drift uses `RunLog(lambda: self.run_file("trades.jsonl"))` / `("decisions.jsonl")`.
  - vwap's `_archive` appends through a `RunLog`. **BEHAVIOR CHANGE:** an `OSError` there is logged instead of escaping the order hook.
- [ ] **Step 7: Port the three other atomic writers** to `atomic_write_json(..., what=<their noun>)`, keeping any warning text a test asserts.
- [ ] **Step 8: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 9: Run `code-simplifier`, update the CLAUDE.md files, commit** `refactor(strategies): Task 5 - JsonStateStore, atomic_write_json, one RunLog`.

### Task 6: `common/handoff.py`: shared readers, `covering`, `SubmitRecorder`

**Files:**
- Modify: `strategies/common/handoff.py`, `bill_ackman/handoff.py`, `bull_bear/handoff.py`, `congress_trades/handoff.py`, `insider_trades/handoff.py`
- Test: create `tests/strategies/common/test_common_handoff.py`; existing handoff tests of the 4 strategies

**Interfaces:**
- Produces, in `common/handoff.py` (no `from __future__ import annotations`):
  - `required_text(item: Mapping[str, Any], field: str, who: str, max_chars: int) -> str`. Errors: `f"{who} has no {field}: give one short sentence"` and `f"the {field} for {who} is {n} characters: keep it under {max_chars}"`.
  - `optional_text(item, field, who, max_chars) -> str | None`. Adds `f"the {field} for {who} must be text"`.
  - `weight(item, ticker, *, example: str = "0.08") -> float`. bill_ackman passes `example="0.25"`.
  - `covering(raw: Any, expected: Sequence[str], *, what: str, field: str = "symbol", unknown: Callable[[str], str], missing: Callable[[list[str]], str]) -> list[tuple[str, Mapping[str, Any]]]`. It returns one `(key, item)` per expected key, in submission order. Errors: `unknown(key)`, `f"{key} appears twice"`, `missing(keys_in_expected_order)`.
  - `class SubmitRecorder`:
    - Class variables: `TOOL_NAMES: ClassVar[Mapping[str, str]]`, `NOT_EXPECTED_SUFFIX: ClassVar[str] = ""` (bill_ackman, bull_bear: `" of the review"`), `ALREADY_RECORDED: ClassVar[str] = "already recorded for this run"` (bill_ackman: `"... for this review"`, bull_bear: `"... for this stage"`).
    - Attributes: `last_error: str | None`.
    - Properties: `submitted -> bool`, `submission -> Any`, `missing_tool -> str`.
    - Methods: `_arm(stage, context)`, `submit(stage, *args) -> dict[str, Any]`, an abstract `_validate(stage, *args) -> Any` that raises `HandoffError`, and `_fail(message)`.

- [ ] **Step 1: Write the tests.**
  - The common tests: `covering`'s errors and order, the text readers' messages, `weight(example=)`, and a toy `SubmitRecorder` (guards, record, `missing_tool`, re-arming).
  - In each strategy's handoff test file, `test_guard_messages_unchanged` with the exact "not expected" / "already recorded" texts; it must PASS on today's code.
- [ ] **Step 2: Run them.** The common tests FAIL; the guard tests PASS.
- [ ] **Step 3: Implement the common additions.**
- [ ] **Step 4: Port the strategies.**
  - bill_ackman and bull_bear delete `HandoffError`, `_items`, `_symbol`, `_weight`, `_reason`, `_text` and `_optional_text`, and use the common ones. `NoteTooLong` subclasses the common `HandoffError`.
  - bull_bear's `_covering` and bill_ackman's `validate_verdicts` key check use `covering` with their exact messages. congress and insider `validate_target` use it only if their text comes out byte-identical (congress: `sorted(expected)`).
  - The four recorders subclass `SubmitRecorder`. bull_bear overrides `_arm` (it also resets `_refused_note`) and builds the "same text" message inside `_validate`. insider keeps its own `submit` (target → report) on the base's `_arm`/`_fail`/`submitted`/`missing_tool`.
- [ ] **Step 5: Run** `uv run pytest tests/strategies -q && uv run ruff check`.
- [ ] **Step 6: Run `code-simplifier`, update the CLAUDE.md files, commit** `refactor(strategies): Task 6 - shared handoff readers, covering, SubmitRecorder`.

### Task 7: `common/pipeline.py`: the abstract `ReviewPipeline`

Only what all four pipelines share goes into the base. `_run_stage`, `retry_prompt` and each review's body stay in the strategies.

**Files:**
- Create: `strategies/common/pipeline.py`, `tests/strategies/common/test_common_pipeline.py`
- Modify:
  - `bill_ackman/pipeline.py` (class renamed `AckmanPipeline`), `bull_bear/pipeline.py` (renamed `BullBearPipeline`), `congress_trades/pipeline.py` (`CongressPipeline`), `insider_trades/pipeline.py` (`InsiderPipeline`)
  - the four agent files and the importers of the deleted names, tests included
- Test: each pipeline's existing test file

**Interfaces:**
- Consumes: `JsonStateStore` (Task 5), `RunLog`, `SubmitRecorder` (Task 6).
- Produces, in `common/pipeline.py`:
  - `class AgentLike(Protocol)`: `run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult`. bull_bear subclasses it with `max_steps`.
  - `class AgentLookup(Protocol)`: `__getitem__(self, name: str, /) -> AgentLike`.
  - `class StageAbandoned(Exception)`: `__init__(self, stage: str, message: str)`, `.stage`. It replaces `ReviewAbandoned` (bill_ackman, bull_bear) and `RunAbandoned` (congress, insider); bull_bear's `AgentLooped` subclasses it.
  - `@dataclass(frozen=True, slots=True) class RunOutcome`: `completed: bool`, `abandoned_streak: int`. It replaces `ReviewOutcome` and the old `RunOutcome`.
  - `class ReviewPipeline[S](ABC)`:
    - Class variables:
      - `LOG_PREFIX: ClassVar[str] = ""` (bull_bear `"[bull_bear] "`, congress `"[congress_trades] "`, insider `"[insider_trades] "`).
      - `WHAT: ClassVar[str] = "review"` (congress, insider: `"run"`).
      - `RECORD_FLAG: ClassVar[tuple[str, Any, Any]] = ("abandoned", False, True)` (congress, insider: `("outcome", None, "abandoned")`).
      - `DATA_ERRORS: ClassVar[tuple[type[Exception], ...]] = ()` (congress `(CongressDataError,)`, insider `(InsiderDataError,)`).
    - `__init__(self, *, strategy: Strategy, agents: AgentLookup, recorder: Any, state: JsonStateStore[S], log: RunLog) -> None` sets `_strategy`, `_agents`, `_recorder`, `_state` and `_log`. Subclasses call it, then set their own fields.
    - `run(self) -> RunOutcome` is the template, not overridden. It does what the four `run()` copies do today:

```python
state = self._state.load()
now = self._strategy.clock.now().astimezone(MARKET_TZ)
key, initial, abandoned = self.RECORD_FLAG
record = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, key: initial}
self._begin()
try:
    return self._run(now, state, record)
except StageAbandoned as exc:
    stage, error = exc.stage, exc
except self.DATA_ERRORS as exc:
    stage, error = "filings", exc
except (BrokerError, BacktestError) as exc:
    stage, error = ("execution" if self._executing() else "broker"), exc
streak = state.abandoned_streak + 1
self._strategy.log_error(f"{self.LOG_PREFIX}{self.WHAT} abandoned at the {stage} stage (streak {streak}): {error}")
saved, extra = self._on_abandoned(state, streak, stage)
self._state.save(saved)
self._log.append({**record, key: abandoned, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
return RunOutcome(completed=False, abandoned_streak=streak)
```

    - The abstract methods, each implemented by all four:
      - `_run(self, now: datetime, state: S, record: dict[str, Any]) -> RunOutcome`: the body. bill_ackman and bull_bear call their `_review` and return `RunOutcome(True, 0)`.
      - `_executing(self) -> bool`: bill_ackman/bull_bear `self._stage == "execution"`, congress/insider `self._phase == "trading"`.
      - `_on_abandoned(self, state: S, streak: int, stage: str) -> tuple[S, dict[str, Any]]`: the state to save and the extra record fields.
        - bill_ackman: the `"orders"` key at the execution stage.
        - bull_bear: the `"orders"` key, plus the verdicts and cooldowns logic and `"cooldowns_started"`.
        - congress/insider: the pending-trade bump, the `traded` merge and `"orders"`.
    - The default hook `_begin(self) -> None` does nothing. bill_ackman resets `_stage = "setup"`; congress/insider reset `_phase` and `_pending_retry`; bull_bear keeps the default, as today.

- [ ] **Step 1: Write the pinning tests** (must PASS on today's code). In each pipeline's test file, `test_abandoned_record_unchanged` checks, for a stage abandon and for a broker error at each phase:
  - the exact error log line;
  - the exact `reviews.jsonl`/`runs.jsonl` record, as a list of `(key, value)` pairs so the key order is checked;
  - the saved state.
  - congress and insider also cover their data error (`filings` stage).
- [ ] **Step 2: Write the failing common tests:**
  - `test_template_success_returns_body_outcome`
  - `test_stage_abandoned`
  - `test_data_error_is_filings_stage`
  - `test_broker_error_execution_vs_broker`
  - `test_record_flag_key_order`
  - `test_begin_called_once_per_run`
  - Each uses a toy subclass with a fake store and log.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement `common/pipeline.py`.** Run: PASS.
- [ ] **Step 5: Port the four pipelines.**
  - Each subclasses `ReviewPipeline[<its state>]`: its `run()` body becomes `_run`/`_executing`/`_on_abandoned`/`_begin`, and it deletes its protocols, abandon exception and outcome class.
  - Rename `ReviewPipeline` → `AckmanPipeline` / `BullBearPipeline` in bill_ackman and bull_bear and every importer.
  - `_run_stage` stays in each pipeline, raising `StageAbandoned`.
- [ ] **Step 6: Run** `uv run pytest -q && uv run ruff check`. Expected: PASS, Step 1's pinning tests included.
- [ ] **Step 7: Run `code-simplifier`, update the CLAUDE.md files** (bill_ackman's and bull_bear's say `ReviewPipeline._run_stage`, which becomes `AckmanPipeline._run_stage` / `BullBearPipeline._run_stage`; common's describes `pipeline.py`), then commit `refactor(strategies): Task 7 - abstract ReviewPipeline for the four agent pipelines`.

### Task 8: Pending-trade steps in `ReviewPipeline` (congress_trades and insider_trades only)

These are concrete methods that bill_ackman and bull_bear never call. They take everything they need as arguments, so bill_ackman and bull_bear gain no attribute, argument or hook.

**Files:**
- Modify: `strategies/common/pipeline.py`, `congress_trades/state.py`, `congress_trades/pipeline.py`, `insider_trades/state.py`, `insider_trades/pipeline.py`
- Test: `tests/strategies/common/test_common_pipeline.py`, the congress and insider pipeline and state tests

**Interfaces:**
- Consumes: `ReviewPipeline`, `RunOutcome` (Task 7), `is_int`/`valid_weights` (Task 5), `TradeDesk`, `Audit`.
- Produces, in `common/pipeline.py`:
  - `@dataclass(frozen=True, slots=True) class PendingTrade`: `target: dict[str, float]`, `days: int = 0`.
  - `valid_pending(value: Any) -> bool` and `decode_pending(raw: dict[str, Any] | None) -> PendingTrade | None`.
  - `ReviewPipeline` methods, each a copy of the congress/insider code it replaces:
    - `_orders_record(self, desk: TradeDesk) -> list[dict[str, Any]]`.
    - `_at_target(self, before: Audit) -> bool`: True when there are no shortfalls, after logging today's `[trader] ... counting orders still working (...)` or `[trader] the account is already at the target: no order needed` line.
    - `_audited(self, desk: TradeDesk, record: dict[str, Any]) -> Audit`: the after-audit block (record `orders`/`shortfalls_after`, the unfilled and off-target warnings).
    - `_give_up_pending(self, state: S, pending: PendingTrade, record: dict[str, Any], *, max_retries: int) -> RunOutcome | None`: logs `f"{self.LOG_PREFIX}giving up on the unfinished trade after {pending.days} later checks: the target was not reached"`, saves, appends `pending_gave_up`, and returns None when `pending.days < max_retries`.
    - `_finish_pending(self, desk: TradeDesk, state: S, pending: PendingTrade, audit: Audit, record: dict[str, Any], now: datetime) -> RunOutcome`: the save plus the `pending_complete`/`pending_retry` record.
    - `_abandoned_with_pending(self, desk: TradeDesk, state: S, streak: int, *, pending_retry: bool, trading: bool) -> tuple[S, dict[str, Any]]`: the congress/insider `_on_abandoned` body.

- [ ] **Step 1: Write the failing tests** in `test_common_pipeline.py`, one per method, with a fake desk, store and log. The expected log lines and record dicts are copied from today's congress pipeline.
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run: PASS.
- [ ] **Step 4: Port congress and insider.** Their state modules import `PendingTrade`, `valid_pending` and `decode_pending`, and their pipelines call the methods. Each keeps its own log lines around them: insider's "check N of M", `trader_retry triggered` and shortfall lines, and congress's single line.
- [ ] **Step 5: Confirm bill_ackman and bull_bear are untouched.** `git diff --stat HEAD -- src/trading_agent_framework/strategies/bill_ackman src/trading_agent_framework/strategies/bull_bear` prints nothing.
- [ ] **Step 6: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 7: Commit** `refactor(strategies): Task 8 - pending-trade steps in ReviewPipeline for congress and insider`.

### Task 9: Universe modules into common, plus the point-in-time helpers

**Files:**
- Create: `strategies/common/universe_params.py`
- Move (`git mv`):
  - `cross_momentum/batch_us_stock_universe.py` → `common/batch_us_stock_universe.py`
  - `cross_momentum/universe_backtest_data.py` → `common/universe_backtest_data.py`
  - `tests/strategies/test_batch_us_stock_universe.py` → `tests/strategies/common/`
  - `tests/strategies/test_universe_backtest_data.py` → `tests/strategies/common/`
- Modify:
  - `pyproject.toml:33` (`batch-universe = "trading_agent_framework.strategies.common.batch_us_stock_universe:main"`)
  - `cross_momentum/parameters.py`, `cross_momentum/agent_cross_momentum.py`, `cross_momentum/utils.py` (comments)
  - `bull_bear/agent_bull_bear.py`, `common/universe_history.py`
  - `scripts/tests/smoke_universe_history.py`, `brokers/alpaca/symbols.py` (docstring)
  - the CLAUDE.md files of common, cross_momentum, bull_bear and backtesting, plus the root `CLAUDE.md` commands
- Test: the moved tests, `tests/strategies/common/test_universe_history.py`, the cross_momentum and bull_bear universe-history tests

**Interfaces:**
- Produces:
  - `common/universe_params.py`: `MIN_PRICE = 10.0`, `MIN_DOLLAR_VOLUME = 20_000_000`, `MIN_MARKET_CAP = 2_000_000_000`, the values copied from today's `CONFIG`. `CONFIG` sets `"min_price": MIN_PRICE` etc.; `batch_us_stock_universe.py` reads these constants and no longer imports `cross_momentum`.
  - Public names in `common/batch_us_stock_universe.py` for what `universe_backtest_data.py` used privately: `rate_limit_gate` (a module-level instance replacing `_get_rate_limit_gate()`), `download_batch`, `HISTORY_BATCH_SIZE`, `MAX_ABSENT_FRACTION`. Also a module-level semaphore replacing `_get_semaphore()`.
  - In `common/universe_history.py`:
    - `market_day(moment: datetime) -> date`
    - `window_symbols(history: UniverseHistory, start: datetime, end: datetime, *, exclude: Collection[str]) -> list[str]`
    - `month_universe(history: UniverseHistory, day: date, *, exclude: Collection[str]) -> list[str]`: raises `FatalStrategyError(str(exc))` from `UniverseHistoryError`.

- [ ] **Step 1: Do the move.**
  - `git mv` the two modules and their two test files. Create `universe_params.py`, and point `CONFIG` and the batch module at it.
  - Update every importer: `grep -rn "cross_momentum.batch_us_stock_universe\|cross_momentum import batch_us_stock_universe\|cross_momentum.universe_backtest_data\|from .batch_us_stock_universe\|from .universe_backtest_data" src tests scripts pyproject.toml` must print nothing afterwards.
  - `uv sync` (the entry point changed), then `uv run batch-universe --help` must print usage.
- [ ] **Step 2: Run** `uv run pytest tests/strategies -q`. Expected: PASS (a pure move).
- [ ] **Step 3: Clean up the batch module.**
  - Merge `get_extra_universe_symbols` and `_strict_etf_symbols` into `_etf_symbols(..., strict: bool)`, keeping the public lenient name if it has outside callers.
  - Replace `_get_semaphore` / `_get_rate_limit_gate` with the module-level instances.
  - Use `results[:_HISTORY_TOP_N]` instead of `results[:1200]`.
  - Delete `get_history_pool` (tests use `build_history_pool(...).symbols`) and type `pool` as `HistoryPool` only.
  - Run the moved tests.
- [ ] **Step 4: Write the failing tests** in `test_universe_history.py`: `test_market_day_aware_and_naive`, `test_window_symbols_excludes`, `test_month_universe_missing_month_is_fatal`. Run: FAIL.
- [ ] **Step 5: Implement the three helpers.** Port both strategies: delete both `_market_day`; `bull_bear._universe_on` and cross_momentum's `_use_month_universe` call `month_universe`, and both `run_backtesting` pre-steps call `window_symbols`.
- [ ] **Step 6: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 7: Update the CLAUDE.md files** (every path to the two modules; the thresholds' new home), then commit `refactor(strategies): Task 9 - universe build and backtest data move to common`.

### Task 10: Yahoo retry and coverage

**Files:**
- Modify: `strategies/common/yahoo_daily_bars.py`, `cross_momentum/agent_cross_momentum.py:349-390`, `bull_bear/market_data.py:52-110`
- Test: `tests/strategies/test_cross_momentum_yahoo_bars.py`, the bull_bear market-data tests, `tests/strategies/common/test_yahoo_daily_bars.py` (create if absent)

**Interfaces:**
- Produces:
  - `class UnusableBatch(Exception)`: the message is the last problem.
  - `BarsSource` Protocol, moved here from `bull_bear/market_data.py`.
  - `fetch_usable_batch(source: BarsSource, universe: Sequence[str], *, extras: Sequence[str] = (), today: Callable[[], date], held: Callable[[], Collection[str]], min_coverage: float, retry_delays: Sequence[float], sleep: Callable[[float], None], on_retry: Callable[[str, float], None]) -> dict[str, Bars]`:
    - The problem texts are `Yahoo lookup failed ({exc})`, `Yahoo covers only {covered}/{n} symbols` and `Yahoo has no bars for held {...}`.
    - Coverage is counted over `universe` only; held stocks are counted only when they are in `universe`.
    - Raises `UnusableBatch(problem)` after the last attempt.

- [ ] **Step 1: Write the failing tests:**
  - `test_usable_first_try`
  - `test_retries_then_succeeds`: `on_retry` is called with `[(problem, 60.0)]` and `sleep` with `60.0`.
  - `test_raises_after_last_delay`
  - `test_held_outside_universe_ignored`
  - `test_extras_not_counted_in_coverage`
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Port both callers.**
  - cross_momentum: `on_retry` logs `f"{problem} — retry in {delay:.0f}s"`; catching `UnusableBatch` logs `f"{exc} — holding current positions."` and returns False.
  - bull_bear: `on_retry` warns `f"{problem}: retry in {delay:.0f}s"`; catching `UnusableBatch` raises `DataUnavailable(str(exc))`.
- [ ] **Step 5: Run** `uv run pytest tests/strategies -q && uv run ruff check`.
- [ ] **Step 6: Commit** `refactor(strategies): Task 10 - shared Yahoo batch retry and coverage check`.

### Task 11: `common/stops.py`: releasing a stop before an exit

**Files:**
- Create: `strategies/common/stops.py`, `tests/strategies/common/test_stops.py`
- Modify: `earnings_drift/desk.py` (`_release_stop`), `vwap_pullback/desk.py` (`_release_stop`)

**Interfaces:**
- Produces:
  - `class StopRelease(StrEnum)`: `GONE`, `FILLED`, `RELEASED`, `FAILED`.
  - `@dataclass(frozen=True, slots=True) class Release`: `status: StopRelease`, `order: Order | None = None`, `error: str | None = None`.
  - `release_stop(strategy, stop_order_id: str | None, *, expected_cancels: set[str], wait_seconds: float, warn_prefix: str) -> Release`. The logic is earnings_drift's:
    - A failed lookup gives `FAILED` with `could not look up the stop: {exc}` and logs a warning.
    - No id or an unknown order gives `GONE`.
    - A filled order gives `FILLED`.
    - An active order: add it to `expected_cancels`, then cancel. A `BrokerError` discards it again and gives `FILLED` if the order is filled, else `FAILED` with `could not cancel the stop: {exc}`. Otherwise wait `wait_seconds`; a filled order gives `FILLED`.
    - Still active after the wait: discard it from `expected_cancels`, log the timeout warning, and give `FAILED` with `the stop cancel was not confirmed in time; nothing else was changed`.
    - Otherwise `RELEASED` with the order.
- Each desk keeps its own partial-fill booking and its own return values (`ALREADY_CLOSED` / `STOPPED_OUT`).

- [ ] **Step 1: Write the failing tests** for each branch (fake strategy and orders).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement, then port both desks.**
- [ ] **Step 4: BEHAVIOR CHANGE (vwap_pullback, approved).** Add vwap desk tests that fail on the old code:
  - (a) A failed stop lookup is a refused exit with a warning, not an exception.
  - (b) A timed-out cancel is removed from `_expected_cancels`, so a late CANCELED re-places the stop.
- [ ] **Step 5: Run** `uv run pytest -q && uv run ruff check`.
- [ ] **Step 6: Update the CLAUDE.md files** (vwap_pullback's gotcha on stops), commit `refactor(strategies): Task 11 - shared release_stop (vwap adopts earnings_drift's timeout rule)`.

### Task 12: Whole-branch verification

- [ ] **Step 1: Run** `uv run pytest -q` and `uv run ruff check`. Expected: at least 4577 passed plus the new tests, and no lint.
- [ ] **Step 2: Grep for leftovers.** These must print nothing:
  - `grep -rn "def state_path\|class ReviewLog\|class JsonlLog\|def _market_day\|class ReviewAbandoned\|class RunAbandoned\|class ReviewOutcome" src/trading_agent_framework/strategies`
  - `grep -rn "tempfile.mkstemp" src/trading_agent_framework/strategies`, except in `common/json_state.py`.
  - `grep -rn "strategies\.\(cross_momentum\|bull_bear\|bill_ackman\|congress_trades\|insider_trades\|earnings_drift\|vwap_pullback\|news_builtin\)" src/trading_agent_framework/strategies/common`
- [ ] **Step 3: Run `/code-review`** on the branch (correctness pass). Point it at the two possible bugs listed under "Out of scope".

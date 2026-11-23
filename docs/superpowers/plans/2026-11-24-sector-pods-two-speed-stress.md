# sector_pods two-speed stress signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give sector_pods' risk manager and PM a fast early warning (`in_market_fall`, "VIX alone") next to the confirmed regime with its score, and tell them to tilt toward XLP/XLV/XLU when the warning is on, the book staying 98% invested.

**Architecture:** `core/regime_data` downloads VIXY in the same batched Yahoo call (optional column, `nan` when missing). `Strategy._refresh_regime` keeps the full `RegimeReading` and computes `market_fall` from the same no-look-ahead slice. sector_pods turns both into a `stress` dict in `MarketFacts`, shows it to the risk manager and PM (not the pods), records it in `reviews.jsonl`, and the prompts say how to use it.

**Tech Stack:** Python 3.14, uv, pytest, pandas/yfinance (Yahoo), LangGraph (sector_pods graph).

**Spec:** `docs/superpowers/specs/2026-11-24-sector-pods-two-speed-stress-design.md`

## Global Constraints

- `in_market_fall` (`utils/helpers.py`) and its thresholds are reused as is: no change.
- The 4-point regime rule, its chart line and `uv run regime recompute|scorecard` are unchanged.
- No code forces weights: no change to `check_weights`, `fill_book` or the caps (25% ETF, 40% pod, 98% total).
- `reviews.jsonl`'s `regime` stays a string; `stress` is a new key next to it.
- A missing, empty or unreadable VIXY is never an error: it becomes `nan`, and the regime is computed as before.
- No look-ahead: `market_fall` reads exactly the rows `_refresh_regime` already reads (`closes_before(session, 503)`, strictly before the session date).
- `DEFENSIVE_ETFS = ("XLP", "XLV", "XLU")`.
- Tests use the hand-written fakes (`tests/fakes.py`, `FakeDownload` in `tests/core/test_regime_data.py`), never the network.
- Quality gate on touched code before each commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest` (with `uv sync --extra dashboard` done), `uv run interrogate -v`. Google docstrings on every new public name. Before every commit: `/simplify` on the diff, stage, `/ponytail:ponytail-review` on the staged changes; fix every "Must fix".

## Review Focus

1. **VIXY absent from the download** (Yahoo drops it, a delisting, a bad day): the regime is still computed and `vixy` is all `nan`. Test in Task 1.
2. **VIXY with no early history** (a backtest window before 2011, `nan` at the start then valid closes): `before` must not raise and the warning runs on the ratio alone there. Tests in Tasks 1 and 2.
3. **A VIXY reverse split**: a jump in raw `Close` must not show up in `vixy` (it reads Adj Close), or it would fire a false warning. Test in Task 1.
4. **A failed refresh** (`RegimeDataError`): `market_fall` and `regime_reading` keep their previous values, like `regime`. Test in Task 2.
5. **No reading yet** (first 503 sessions, or a live start before the first successful refresh): `stress` is `None`, the review still runs, and `reviews.jsonl` records `null`. Test in Task 3.

---

### Task 1: VIXY in the regime download

**Files:**
- Modify: `src/trading_agent_framework/core/regime_data.py` (`SYMBOLS`, `RegimeSeries`, `before`, `series`, `_fetch`)
- Test: `tests/core/test_regime_data.py`

**Interfaces:**
- Produces: `REQUIRED = ("SPY", "^VIX", "^VIX3M")`, `SYMBOLS = (*REQUIRED, "VIXY")`; `RegimeSeries.vixy: tuple[float, ...] = ()` (empty = no VIXY data; otherwise aligned with `sessions`, `nan` where missing).

- [ ] **Step 1: Write the failing tests**

Extend `FakeDownload` with a VIXY part (base 20.0, volume 1e6), skippable through `absent`, and an optional `vixy_frame: pd.DataFrame | None` override. Then:

```python
def test_one_batched_download_of_the_four_symbols() -> None:  # replaces ..._three_symbols
    ...
    assert symbols == ["SPY", "^VIX", "^VIX3M", "VIXY"] == list(SYMBOLS)

def test_vixy_is_aligned_on_spy_sessions() -> None:
    series = _source(FakeDownload(_weekdays(10))).series(FIRST)
    assert series.vixy == tuple(20.0 + i for i in range(10))

def test_a_missing_vixy_is_nan_not_an_error() -> None:
    series = _source(FakeDownload(_weekdays(10), absent=("VIXY",))).series(FIRST)
    assert len(series.vixy) == 10 and all(v != v for v in series.vixy)
    assert series.spy == tuple(99.5 + i for i in range(10))  # the regime's inputs are untouched

def test_before_slices_vixy_and_ignores_its_nan() -> None:
    days = _weekdays(30)
    series = _source(FakeDownload(days, drop={"VIXY": days[:20]})).series(FIRST)  # VIXY starts late
    out = series.before(days[-1], 25)
    assert len(out.vixy) == 25 and out.vixy[-1] == series.vixy[-2]

def test_a_series_without_vixy_keeps_an_empty_column() -> None:
    old = RegimeSeries((date(2026, 1, 2),), (100.0,), (15.0,), (17.0,))
    assert old.vixy == () and old.before(date(2026, 1, 5), 1).vixy == ()

def test_vixy_reads_adjusted_closes_so_a_reverse_split_is_no_jump() -> None:
    # raw Close x4 from day 5 (1:4 reverse split), Adj Close smooth
    ...
    assert max(b / a for a, b in zip(series.vixy, series.vixy[1:])) < 1.1
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/core/test_regime_data.py -v`
Expected: the new tests FAIL (`AttributeError: ... 'vixy'`, symbols mismatch); the others pass.

- [ ] **Step 3: Implement**

- `REQUIRED` keeps today's rules (empty → `RegimeDataError`, partial download not cached). VIXY: `_closes` wrapped so a `RegimeDataError` or an empty series gives an empty series, then `reindex(spy.index).ffill(limit=self._max_fill)` like VIX, so missing → `nan`.
- `series()` and `before()` slice `vixy` with the other columns when it is non-empty, else keep `()`. `before`'s `nan` check stays on `spy`, `vix`, `vix3m` only.
- Error messages that list symbols keep listing all of `SYMBOLS`.
- Module docstring: "SPY, VIX, VIX3M and VIXY"; one line on VIXY being optional.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/ -v`
Expected: all PASS (`test_regime_cli.py` and `test_regime.py` unchanged and green).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/core/regime_data.py tests/core/test_regime_data.py
git commit -m "feat(core): download VIXY with the regime closes (optional, nan when missing)"
```

### Task 2: `Strategy.market_fall` and `Strategy.regime_reading`

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py` (`__init__` near line 171, `_refresh_regime` at line 372)
- Modify: `src/trading_agent_framework/core/CLAUDE.md` (regime bullet), root `CLAUDE.md` (regime bullet: one clause)
- Test: `tests/core/test_strategy_regime.py`

**Interfaces:**
- Consumes: `RegimeSeries.vixy` (Task 1); `in_market_fall(vix, vix3m, vixy) -> bool` (`utils/helpers.py`).
- Produces: `Strategy.regime_reading: RegimeReading | None` and `Strategy.market_fall: bool | None`, both `None` until the first refresh with enough history, set together, kept on a failed refresh.

- [ ] **Step 1: Write the failing tests**

Give `_series` a `vixy: list[float] | None = None` parameter (passed through when set). `BULL` (VIX 15 / VIX3M 18, ratio 0.83) is calm; `BEAR` (30 / 25, ratio 1.2) is stressed.

```python
def test_a_refresh_sets_the_reading_and_the_market_fall_flag() -> None:
    strategy, _, _, _ = _strategy(BEAR)
    strategy._refresh_regime()
    assert strategy.market_fall is True
    assert strategy.regime_reading is not None and strategy.regime_reading.regime == strategy.regime

def test_a_calm_market_has_no_fall() -> None:  # BULL, no VIXY column: ratio leg only
    ... assert strategy.market_fall is False

def test_a_vixy_jump_alone_turns_the_flag_on() -> None:
    vixy = [20.0] * (ROWS - 1) + [22.5]  # +12.5% over 5 sessions, ratio calm
    ... assert strategy.market_fall is True

def test_the_market_fall_is_logged(caplog) -> None:
    ... assert "Market fall warning ON" in caplog.text

def test_a_data_failure_keeps_the_previous_flag_and_reading() -> None:
    # refresh on BEAR, then source.error = RegimeDataError("x"), refresh again
    ... assert strategy.market_fall is True and strategy.regime_reading is first_reading

def test_too_little_history_leaves_the_flag_unknown() -> None:
    ... assert strategy.market_fall is None and strategy.regime_reading is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/core/test_strategy_regime.py -v`
Expected: the new tests FAIL with `AttributeError: ... 'market_fall'`.

- [ ] **Step 3: Implement**

In `_refresh_regime`, after `self.regime = reading.regime`: `self.regime_reading = reading`; `vixy = series.vixy or (math.nan,) * len(series)`; `self.market_fall = in_market_fall(series.vix, series.vix3m, vixy)`; one `log_info` line, e.g. `Market fall warning ON (VIX/VIX3M 0.97, VIXY +12.3% over 5 sessions)` / `OFF (...)` (VIXY part omitted when `nan`). Docstrings of both attributes say "information only, nothing trades on it".

Docs: `core/CLAUDE.md` regime bullet: VIXY in the download (optional, `nan` when missing), `regime_reading`, `market_fall` (the `in_market_fall` rule, same slice, set together with the regime). Root `CLAUDE.md` regime bullet: add "plus a fast early warning, `strategy.market_fall`".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/ tests/backtesting/test_regime_backtest.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py src/trading_agent_framework/core/CLAUDE.md CLAUDE.md tests/core/test_strategy_regime.py
git commit -m "feat(core): strategy.market_fall early warning and the full regime reading"
```

### Task 3: the `stress` field in sector_pods' state

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/pods.py` (`DEFENSIVE_ETFS`)
- Modify: `src/trading_agent_framework/strategies/sector_pods/facts.py` (`stress_view`)
- Modify: `src/trading_agent_framework/strategies/sector_pods/graph.py` (`PodsState`, `MarketFacts`, `risk_reads`, `pm_reads`, `Execute` record)
- Modify: `src/trading_agent_framework/strategies/sector_pods/agent_sector_pods.py:189` (wiring)
- Test: `tests/strategies/sector_pods/test_sector_pods_facts.py`, `test_sector_pods_graph.py` (update the `MarketFacts(...)` call at line 221)

**Interfaces:**
- Consumes: `Strategy.regime_reading`, `Strategy.market_fall` (Task 2); `RegimeReading` fields `score`, `inverted`, `vix_spike`, `below_sma`, `off_high`.
- Produces: `DEFENSIVE_ETFS: tuple[str, ...]`; `stress_view(reading: RegimeReading | None, market_fall: bool | None) -> dict[str, Any] | None`; `MarketFacts(..., stress: Callable[[], dict[str, Any] | None], ...)` (required keyword); state slot `stress`.

- [ ] **Step 1: Write the failing tests**

```python
def test_defensive_etfs_are_in_the_universe():
    assert set(DEFENSIVE_ETFS) == {"XLP", "XLV", "XLU"} <= set(ALL_SYMBOLS)

def test_stress_view_from_a_reading():
    view = stress_view(reading(score=2, inverted=True, vix_spike=False, below_sma=True, off_high=False), True)
    assert view == {"early_warning": True, "regime_score": 2,
                    "conditions": {"vix_above_vix3m": True, "vix_spike": False, "spy_below_sma50": True, "spy_5pct_off_high": False},
                    "defensive_etfs": ["XLP", "XLV", "XLU"]}

def test_stress_view_is_none_without_a_reading():
    assert stress_view(None, None) is None

def test_market_facts_puts_stress_in_the_state(): ...           # state["stress"] == the callable's dict
def test_risk_and_pm_see_stress_but_pods_do_not(): ...          # "stress" in risk_reads(s) and pm_reads(s); not in pod_reads(k)(s)
def test_the_review_log_records_stress(): ...                   # reviews.jsonl record has "stress"
def test_a_review_without_stress_records_null_and_completes(): ...  # stress=lambda: None: week completes, record["stress"] is None
```

(`reading(...)` = a small test helper building a `RegimeReading` with the other fields at neutral values.)

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/sector_pods/ -v`
Expected: the new tests FAIL (`ImportError: DEFENSIVE_ETFS` / `stress_view`, `TypeError` on `stress=`).

- [ ] **Step 3: Implement**

- `pods.py`: `DEFENSIVE_ETFS = ("XLP", "XLV", "XLU")`; membership in `ALL_SYMBOLS` is pinned by the test, not an import-time assert.
- `facts.py`: `stress_view` returns `None` when `reading` is None, else the dict above; the "spy_below_sma50" / "spy_5pct_off_high" keys name the default parameters, which are the only ones sector_pods uses.
- `graph.py`: `PodsState.stress: dict[str, Any] | None`; `MarketFacts` stores `stress` and adds `"stress": self._stress()` to `base`; `risk_reads` keys add `"stress"`; `pm_reads` keys add `"stress"`; `Execute`'s record keys add `"stress"` after `"regime"`. `pod_reads` unchanged.
- `agent_sector_pods.py`: `stress=lambda: stress_view(self.regime_reading, self.market_fall)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/sector_pods/ tests/dashboard/ -v`
Expected: all PASS (the dashboard's `reviews_reader` ignores the new key).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/sector_pods/ tests/strategies/sector_pods/
git commit -m "feat(sector_pods): stress field (early warning + regime score) for the risk manager and PM"
```

### Task 4: prompts and the `stress` objection kind

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/handoff.py:19` (`OBJECTION_KINDS`)
- Modify: `src/trading_agent_framework/strategies/sector_pods/prompts.py` (`risk_system`, `pm_system`)
- Modify: `src/trading_agent_framework/strategies/sector_pods/CLAUDE.md` (state table row, the `stress` field, defensive list, objection kind; "Regime is context only" stays)
- Test: `tests/strategies/sector_pods/test_sector_pods_handoff.py`, `test_sector_pods_prompts.py`

**Interfaces:**
- Consumes: `DEFENSIVE_ETFS` (Task 3), state slot `stress` (Task 3).
- Produces: `OBJECTION_KINDS = ("crowding", "macro", "drawdown", "reversal", "stress", "other")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stress_is_an_objection_kind(): ...              # a risk review with kind "stress" passes validation
def test_risk_prompt_explains_stress_and_its_kind():
    text = risk_system(SectorPodsParams())
    assert "`stress`" in text and "early_warning" in text and "defensive_etfs" in text
    assert "crowding, macro, drawdown, reversal, stress or other" in text
def test_pm_prompt_tilts_to_defensives_and_stays_invested():
    text = pm_system(SectorPodsParams())
    assert "early_warning" in text and "defensive_etfs" in text and "keep other ETFs" in text
def test_pod_prompts_do_not_mention_stress(): ...        # "early_warning" not in any pod_system(...)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/sector_pods/test_sector_pods_handoff.py tests/strategies/sector_pods/test_sector_pods_prompts.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement**

Prompt copy (the spec fixes the meaning; wording may be tightened, no absolutes):

- Risk context fields, after `regime`: *`stress`: `early_warning` (a fast VIX-based alarm: it fires near the start of a fall, and also about 4 times a year falsely), `regime_score` (0-4 conditions behind `regime`) and `conditions`, and `defensive_etfs`. Null when unavailable.*
- Risk "Look for", new item: *stress: when `stress.early_warning` is true, strong scores on cyclical or high-volatility ETFs (`volatility_20d_pct`); suggest the `defensive_etfs` instead. Weigh the alarm with `regime_score` and `conditions`.*
- Risk kinds list: `crowding, macro, drawdown, reversal, stress or other`.
- PM context fields: the same `stress` line. PM step 2 reads `stress` too. PM rules, new line: *When `stress.early_warning` is true, tilt toward `defensive_etfs`, up to the caps, but keep other ETFs: the book stays fully invested.*

Docs: `strategies/sector_pods/CLAUDE.md`, state table gets a `stress` row (writer `market_facts`, readers risk, PM, `Execute`); the market-facts bullet gains one sentence on `stress` and `DEFENSIVE_ETFS`; the persistence bullet lists `stress` in the `reviews.jsonl` keys.

- [ ] **Step 4: Run the full gate**

Run: `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov && uv run interrogate -v`
Expected: exit 0 everywhere, 0 pyright errors, coverage >= 80%, interrogate >= 80%.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/sector_pods/ tests/strategies/sector_pods/
git commit -m "feat(sector_pods): risk and PM prompts use the stress signal; stress objection kind"
```

### After the tasks (not code; the user runs or approves)

- Manual smoke: `uv run python scripts/tests/smoke_regime_backtest.py` still passes (the regime line is unchanged).
- Evaluation per the spec's "How we will know it worked": a 2-year backtest on 2024-09-03 to 2026-10-05, baseline rerun on today's code, 2-3 runs per arm; check the defensive share of `book` in warning weeks vs others.

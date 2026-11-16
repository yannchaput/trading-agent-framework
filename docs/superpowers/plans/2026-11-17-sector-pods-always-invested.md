# sector_pods Always Invested Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every completed sector_pods week trades a book invested at `1 - cash_buffer` (98%): the PM's weights set the mix, code scales them up.

**Architecture:** A pure `fill_book(weights, state, params)` next to `check_weights` in `strategies/sector_pods/handoff.py` scales the
PM's researched ETFs in proportion to 98%, clipping at the ETF and pod caps and spreading what a cap cuts over the ETFs still below
theirs. `Execute._trade` (`graph.py`) trades the filled book, logs one INFO line and records it as `book` in `reviews.jsonl`.
The PM prompt says the book is always invested.

**Tech Stack:** Python 3.14, uv, pytest, LangGraph (unchanged).

**Spec:** `docs/superpowers/specs/2026-11-17-sector-pods-always-invested-design.md`

## Global Constraints

- Paths below are relative to `src/trading_agent_framework/` unless they start with `tests/` or `docs/`.
- Target exposure is `1 - params.cash_buffer` (0.98 with defaults); caps `max_etf_weight` 0.25, `max_pod_weight` 0.40, unchanged.
- Unresearched ETFs (`_frozen(symbol, state)`) keep their submitted weight; ETFs with weight 0 stay out of the book.
- No regime-based exposure, no on/off switch, no change to pods, risk manager, cap values or cash buffer.
- The filled book always passes `check_weights` and never totals above `1 - cash_buffer`.
- `handoff.py` has NO `from __future__ import annotations` (it sits next to tool closures): keep it that way.
- Quality gate (global CLAUDE.md): ruff check, ruff format --check, pytest (no new warnings), pytest --cov >= 80%, interrogate >= 80%; `/ponytail:ponytail-review` on the staged change before every commit.

## Review Focus

1. Float drift: after scaling, the sum lands at 0.9800000001 and `check_weights` refuses the book in `Execute` -> week abandoned. Expect: never above target (pinned in Task 1, `test_fill_book_never_exceeds_the_target`).
2. A pod whose unresearched ETFs alone sit near the pod cap: the free ETFs of that pod get only what is left (never negative). Pinned in Task 1, `test_fill_book_frozen_etfs_count_toward_their_pod_cap`.
3. Every pod unavailable (all ETFs frozen): nothing scalable, the submitted book is traded as is with a log line, not an error. Pinned in Task 1 and Task 2.
4. The default 2-ETF test book (XLK 0.2, XLV 0.2) can only reach 50% (two ETFs at 25%): the week still trades and the log says a cap held it. Pinned in Task 2.
5. Turnover: the scale factor changes every week with the PM's total, so ETFs drift in and out of `rebalance_band` (0.03). Not testable here; check trade counts in the follow-up backtest.

---

### Task 0: Workspace

- [ ] **Step 1:** Another session works on `feat/bull-bear-length-gate` in the main checkout: do NOT switch branches there. Create a worktree from `main`: `git worktree add .worktrees/sector-pods-always-invested -b feat/sector-pods-always-invested main` (`.worktrees/` is git-ignored), then `cd` into it and `uv sync --extra dashboard`.
- [ ] **Step 2:** Copy the untracked spec from the main checkout (`docs/superpowers/specs/2026-11-17-sector-pods-always-invested-design.md`) and this plan into the worktree; commit both: `docs: sector_pods always invested: design and plan`. Remove the untracked copies from the main checkout only after the commit.
- [ ] **Step 3:** Baseline: `uv run pytest -q tests/strategies/sector_pods` passes.

Merge note: `feat/bull-bear-length-gate` rewrote much of `handoff.py` and `tests/strategies/sector_pods/test_sector_pods_handoff.py` (not `check_weights`, `_frozen` or `Execute`). Whichever branch merges second rebases; add `fill_book` and its tests in their own blocks to keep that conflict mechanical.

### Task 1: `fill_book`

**Files:**
- Modify: `strategies/sector_pods/handoff.py` (new function right after `check_weights`)
- Test: `tests/strategies/sector_pods/test_sector_pods_handoff.py` (new block; reuse `_full_state`, `_state`, `PARAMS`)

**Interfaces:**
- Consumes: `check_weights(weights, state, params) -> str | None`, `_frozen(symbol, state) -> bool`, `pod_of(symbol).key`, `_TOLERANCE`.
- Produces: `fill_book(weights: Mapping[str, float], state: Mapping[str, Any], params: SectorPodsParams) -> dict[str, float]`: the book to trade, zero weights dropped, insertion order of `weights`.

- [ ] **Step 1: Write the failing tests** (`approx = pytest.approx`; every test also asserts `check_weights(result, state, PARAMS) is None`)

```python
def test_fill_book_scales_in_proportion_to_98_percent():
    book = fill_book({"XLK": 0.12, "XLF": 0.08, "XLV": 0.10, "XLE": 0.10, "XLY": 0.10}, _full_state(), PARAMS)
    assert book == approx({"XLK": 0.2352, "XLF": 0.1568, "XLV": 0.196, "XLE": 0.196, "XLY": 0.196})


def test_fill_book_spreads_what_the_etf_cap_cuts():
    book = fill_book({"XLK": 0.20, "XLF": 0.05, "XLV": 0.05, "XLE": 0.05, "XLY": 0.05}, _full_state(), PARAMS)
    assert book == approx({"XLK": 0.25, "XLF": 0.1825, "XLV": 0.1825, "XLE": 0.1825, "XLY": 0.1825})


def test_fill_book_stops_below_98_when_every_etf_is_at_a_cap():
    book = fill_book({"XLK": 0.10, "SMH": 0.10, "IGV": 0.10, "XLV": 0.10}, _full_state(), PARAMS)
    assert book == approx({"XLK": 0.40 / 3, "SMH": 0.40 / 3, "IGV": 0.40 / 3, "XLV": 0.25})  # tech pod at 40%, XLV at 25%: 65%


def test_fill_book_keeps_unresearched_etfs_at_their_weight():
    state = _full_state(holdings={"XLF": 0.10}) | {"pods": {p.key: ({"unavailable": True} if p.key == "financials_pod" else {"scores": [], "sector_view": "x"}) for p in PODS}}
    book = fill_book({"XLF": 0.10, "XLK": 0.10, "XLV": 0.10, "XLE": 0.10, "XLY": 0.10}, state, PARAMS)
    assert book == approx({"XLF": 0.10, "XLK": 0.22, "XLV": 0.22, "XLE": 0.22, "XLY": 0.22})


def test_fill_book_frozen_etfs_count_toward_their_pod_cap():
    # financials_pod: KRE frozen at 0.30 (held), XLF researched; the pod can take only 0.10 more
    state = _full_state(holdings={"KRE": 0.30})
    del state["facts"]["financials_pod"]["KRE"]  # no facts: KRE is unresearched, the rest of the pod is not
    book = fill_book({"KRE": 0.30, "XLF": 0.05, "XLK": 0.05, "XLV": 0.05}, state, PARAMS)
    assert book["KRE"] == approx(0.30) and book["XLF"] == approx(0.10)


def test_fill_book_leaves_zero_weights_out_and_an_empty_book_empty():
    assert fill_book({"XLK": 0.0, "XLV": 0.10}, _full_state(), PARAMS) == approx({"XLV": 0.25})
    assert fill_book({}, _full_state(), PARAMS) == {}


def test_fill_book_with_nothing_researched_returns_the_submitted_book():
    state = _full_state(holdings={"XLK": 0.10, "XLV": 0.10}) | {"pods": {p.key: {"unavailable": True} for p in PODS}}
    assert fill_book({"XLK": 0.10, "XLV": 0.10}, state, PARAMS) == approx({"XLK": 0.10, "XLV": 0.10})


def test_fill_book_never_exceeds_the_target():
    weights = {s: 1 / 30 for s in ("XLK", "XLC", "SMH", "XLF", "KRE", "KIE", "XLV", "XBI", "IHI", "XLE", "XLU", "XLI", "XLY", "XLP", "XHB")}
    book = fill_book(weights, _full_state(), PARAMS)
    assert sum(book.values()) <= 0.98 and sum(book.values()) == approx(0.98)
    assert fill_book({"XLK": 0.25, "XLV": 0.25, "XLE": 0.25, "XLY": 0.23}, _full_state(), PARAMS) == approx({"XLK": 0.25, "XLV": 0.25, "XLE": 0.25, "XLY": 0.23})
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/sector_pods/test_sector_pods_handoff.py -k fill_book -q`. Expected: FAIL (`fill_book` not defined).

- [ ] **Step 3: Implement `fill_book`** (Google docstring: what is scaled, what is kept, when the rest stays in SHV). The algorithm (the tests do not pin the iteration order):

```
target = 1 - cash_buffer; book = positive weights only
free = [s for s in book if not _frozen(s, state)]
repeat (bounded by len(free) + 1 rounds):
    open = free ETFs below max_etf_weight whose pod total is below max_pod_weight (both with _TOLERANCE)
    room = target - sum(book)
    if room <= _TOLERANCE or not open: stop
    add room to the open ETFs in proportion to their current weight
    clip each free ETF at max_etf_weight
    for each pod above max_pod_weight: scale its FREE ETFs down so the pod (frozen included) sits at the cap
finally: if sum(book) > target, scale the free ETFs down by the excess (float drift), so the sum never exceeds target
```

- [ ] **Step 4: Run** the same command. Expected: PASS. Then `uv run pytest -q tests/strategies/sector_pods`.
- [ ] **Step 5: Commit** (after `/ponytail:ponytail-review` on the staged change): `feat(sector_pods): fill_book scales the PM's weights to the invested target within the caps`

### Task 2: `Execute` trades the filled book

**Files:**
- Modify: `strategies/sector_pods/graph.py` (`Execute.__init__`, `__call__`, `_trade`), `strategies/sector_pods/agent_sector_pods.py` (the `Execute(...)` call: `on_info=self.log_info`)
- Test: `tests/strategies/sector_pods/test_sector_pods_graph.py` (harness and the two direct `Execute(...)` calls gain `on_info=`; `Harness` gains `infos: list[str]`)

**Interfaces:**
- Consumes: `fill_book` (Task 1).
- Produces: `Execute(store, rebalancer, log, params, *, on_info, on_warning, on_error)`; `_trade` returns `(reason, orders, book)`; the `reviews.jsonl` record gains `"book"` right after `"pm"`: the traded weights, or `None` when the week traded nothing (abandoned or refused).

- [ ] **Step 1: Write the failing tests**

```python
def test_execute_fills_the_book_and_records_it(tmp_path: Path) -> None:
    book = {"XLK": 0.12, "XLF": 0.08, "XLV": 0.10, "XLE": 0.10, "XLY": 0.10}
    h = _harness(tmp_path, pm=weights(book))
    h.run()
    (line,) = h.log_lines()
    assert line["book"] == pytest.approx({"XLK": 0.2352, "XLF": 0.1568, "XLV": 0.196, "XLE": 0.196, "XLY": 0.196})
    assert [w["weight"] for w in line["pm"]["weights"]] == list(book.values())  # what the PM submitted is kept
    assert h.infos == ["book filled from 50.0% to 98.0% invested"]


def test_a_capped_book_trades_below_98_and_says_so(tmp_path: Path) -> None:
    h = _harness(tmp_path)  # default PM book XLK 0.2, XLV 0.2: two ETFs at the 25% cap
    final = h.run()
    (line,) = h.log_lines()
    assert final["abandoned"] is None and line["book"] == pytest.approx({"XLK": 0.25, "XLV": 0.25})
    assert h.infos == ["book filled from 40.0% to 50.0% invested (the caps hold it below 98.0%)"]


def test_a_refused_book_records_no_book(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    execute = Execute(h.store, h.rebalancer, RunLog(h.log_path), h.params, on_info=h.infos.append, on_warning=h.warnings.append, on_error=h.errors.append)
    week = PodsState(abandoned=None, date="2026-11-10", regime="bullish", holdings={}, facts={}, pods={}, pm={"weights": [{"symbol": "XLK", "weight": 0.9}]})
    assert execute(week)["abandoned"].startswith("execution")
    assert h.log_lines()[-1]["book"] is None and h.infos == []
```

Also update `test_a_book_already_held_within_the_band_completes_the_week_with_no_orders`: hold the FILLED book (`held={"XLK": 50, "XLV": 50, "SHV": 49}`, 25% + 25%). Expected log wording: `book filled from {submitted:.1%} to {filled:.1%} invested`, plus ` (the caps hold it below {target:.1%})` when the filled total is more than `_TOLERANCE` under the target; with nothing scalable (every ETF frozen) the line still reads `book filled from X% to X% invested (...)`.

- [ ] **Step 2: Run** `uv run pytest tests/strategies/sector_pods/test_sector_pods_graph.py -q`. Expected: the new tests FAIL (no `book`, no `on_info`).
- [ ] **Step 3: Implement.** `_trade`: after `check_weights` accepts the PM's weights, `book = fill_book(weights, state, self._params)`, re-check `check_weights(book, ...)` (a failure is refused like today), `self._on_info(...)`, `target_portfolio(book, ...)`, and `_trades_due(book, ...)`. `__call__` writes `book` into the record. Strategy wiring: `on_info=self.log_info`.
- [ ] **Step 4: Run** `uv run pytest -q tests/strategies/sector_pods`. Expected: PASS.
- [ ] **Step 5: Commit** (after `/ponytail:ponytail-review`): `feat(sector_pods): execute trades the filled book and records it`

### Task 3: PM prompt and docs

**Files:**
- Modify: `strategies/sector_pods/prompts.py` (`pm_system`), `strategies/sector_pods/CLAUDE.md`
- Test: `tests/strategies/sector_pods/test_sector_pods_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pm_prompt_says_the_book_is_always_invested():
    text = pm_system(PARAMS)
    assert "always invested" in text and "98%" in text and "mix" in text
    assert "small book" not in text and "risk-off" not in text
```

- [ ] **Step 2: Run** `uv run pytest tests/strategies/sector_pods/test_sector_pods_prompts.py -q`. Expected: FAIL.
- [ ] **Step 3: Edit `pm_system`.** Step 3 of the prompt becomes: higher scores get more, stay spread over several sectors; your weights set the MIX: the book is always invested at {total} (scaled up in proportion), so give each ETF a weight relative to the others. Rules: the caps apply to the scaled book; an ETF you leave out (or give 0) is sold; unresearched ETFs are kept at their weight, never scaled. Keep the existing SHV sentence accurate ("SHV only holds what the caps leave over"). Keep `test_pm_prompt_states_the_caps_from_params` and `test_no_prompt_contains_forced` passing (no "forced", no "held").
- [ ] **Step 4:** `sector_pods/CLAUDE.md`: Architecture "Caps" bullet: the PM's book is filled to `1 - cash_buffer` by `handoff.fill_book` in `Execute` (in proportion, clipped at the caps, the rest in SHV only when the caps leave it over); the record's `book` field next to `pm`; persistence bullet lists `book`. Run `uv run pytest -q tests/strategies/sector_pods`. Expected: PASS.
- [ ] **Step 5: Commit** (after `/ponytail:ponytail-review`): `feat(sector_pods): the PM prompt says the book is always invested`

### Task 4: Gate

- [ ] `uv run ruff check`, `uv run ruff format --check`, `uv run pytest --cov` (>= 80%, `fill_book` fully covered, no new warnings), `uv run interrogate -v` (>= 80%), `uv run pyright` on the touched files (0 new errors). Report the numbers.
- [ ] Follow-up for the user (not part of this branch): a sector_pods backtest on the same `BI_YEAR` window; expect `invested_weight` about 0.98 every week (or a cap log line), compare CAGR with the 15.6% replay.

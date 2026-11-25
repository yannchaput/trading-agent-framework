# sector_pods simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `strategies/sector_pods/` simpler and readable (fewer classes, plain functions and dicts) with byte-identical behaviour.

**Architecture:** One `SectorPodsReview` class in `graph.py` (nodes are methods, graph built from a `(name, method)` list, like `BullBearReview`), one conditional edge for an abandoned week, `Execute._trade` split into short steps, `fill_book` without nested closures, a slimmer `SectorPodsStrategy._node`.

**Tech Stack:** Python 3.14, uv, LangGraph 1.2.16, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-sector-pods-simplify-design.md`

## Global Constraints

- Scope: `src/trading_agent_framework/strategies/sector_pods/` and `tests/strategies/sector_pods/` only. No change to `agents/`, `common/`, other strategies, prompts, `SectorPodsParams` fields, the state-file format, `reviews.jsonl` or log lines.
- Behaviour is byte-identical: the Task 1 capture must match after Tasks 3, 4 and 5. `fill_book` keeps the same float operations in the same order, or the capture differs in the last digit.
- Plain functions and dicts; no new class, Protocol or type unless it removes more than it adds. No `# noqa`, `# type: ignore`, `# pragma: no cover`, lowered gate.
- Node names stay (`load_previous`, `market_facts`, the five pod keys, `risk_manager`, `portfolio_manager`, `execute`); LangGraph refuses a node named like a state key.
- Thread rules: with `invoke()` and no `step_timeout`, single nodes (order code included) run on the strategy thread; the five pods run on workers and call only read-only tools.
- Use `uv sync --extra dashboard` (plain `uv sync` drops the dashboard extra). Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%; it crashes on a clean main: report, do not fix).
- Branch `refactor/sector-pods-simplify`; never touch `TODO.md`; no push, PR or merge without the user's go-ahead.

## Review Focus

Each is pinned by an existing test that must keep passing (renamed to the new construction, never weakened):

- Abandoned week (data unavailable): no agent runs, the week is logged, the streak grows (`test_data_unavailable_abandons_before_agents`).
- A pod with no facts or a failed pod: its ETFs become "unresearched" and the join still fires (`test_failed_pod_keeps_or_trims_its_sleeve`).
- `fill_book` edges: every ETF at a cap, zero weights, empty book, float drift, frozen ETFs counting toward their pod cap (the `test_fill_book_*` tests plus the Task 1 table).
- A broker error escaping the graph, and three of them aborting a backtest only (`test_three_escaped_errors_*` in `test_sector_pods_strategy.py`).
- A restart on an already completed Tuesday does not run again (`test_sector_pods_strategy.py`).

---

### Task 1: Capture the current behaviour (scratchpad, not committed)

**Files:**
- Create: `<scratchpad>/capture_sector_pods.py`, `<scratchpad>/baseline.json`

**Interfaces:**
- Produces: `baseline.json`, the reference of Tasks 3-5. The script must run unchanged before and after the refactor: it uses only `tests/strategies/sector_pods/test_sector_pods_graph.py::_harness(tmp_path, ...)` and the `Harness` fields `broker.submitted`, `store.load()`, `log_lines()`, `infos`, `warnings`, `errors`, `run()`. **Task 4 keeps `_harness`'s signature and these `Harness` fields.**

- [ ] **Step 1: Write the capture script.** It runs these `_harness` scenarios, each in a fresh `tmp_path`, and dumps JSON `{scenario: {final, orders, state, log_lines, infos, warnings, errors}}` (`final` minus nothing; orders as `(symbol, side, quantity)`; state via `dataclasses.asdict`): default week; `held={"XLF": 20}` with `financials_pod` raising; PM unavailable; `FakeBars` unavailable (data abandoned); a capped book (the scenario of `test_a_capped_book_trades_below_98_and_says_so`); a refused book (`test_a_refused_book_records_no_book`); two consecutive weeks (`previous` fed back, as in `test_previous_week_reaches_agents_and_outcome`); stress `None`. Copy each scenario's setup from the matching test.
- [ ] **Step 2: Add the `fill_book` table.** 300 cases from `random.Random(7)`: weights over a random subset of `ALL_SYMBOLS` (0 to 0.5), a state with random `holdings`, random pods marked unavailable; dump `{case_index: {symbol: weight}}` with `repr(float)`. Include the six hand cases of `test_fill_book_*`.
- [ ] **Step 3: Run it on the untouched code and save `baseline.json`.** Run: `uv run python <scratchpad>/capture_sector_pods.py > <scratchpad>/baseline.json`. Run it twice: the two files must be identical (the capture is deterministic).
- [ ] **Step 4: Commit the spec and this plan only.** `git add docs/superpowers/specs/2026-11-25-sector-pods-simplify-design.md docs/superpowers/plans/2026-11-25-sector-pods-simplify.md && git commit -m "docs: sector_pods simplification design and plan"` (this is the only commit made before Task 8; end the message with the attribution line of the session).

### Task 2: Spike the conditional edge (scratchpad, not committed)

**Files:**
- Create: `<scratchpad>/spike_edge.py`

**Interfaces:**
- Produces: a yes/no on the wiring below, before Task 4 builds on it.

- [ ] **Step 1: Build a toy `StateGraph` with the same shape** (`load_previous -> market_facts`, a conditional edge from `market_facts` returning either the five pod names or `["execute"]`, `add_edge([five pods], "risk_manager")`, `risk_manager -> portfolio_manager -> execute -> END`), `abandoned` set by `market_facts` in one run and not in the other.
- [ ] **Step 2: Run both cases.** Expected: abandoned run visits only `load_previous`, `market_facts`, `execute` and the final state has no `pods` key; the normal run visits the five pods in parallel (threads differ from the caller's, as `test_single_nodes_run_on_the_calling_thread` needs) and `risk_manager` runs once, after all five.
- [ ] **Step 3: If either expectation fails**, keep the existing wrappers' behaviour with the conditional edge only on `market_facts -> execute` and report the finding before Task 4.

### Task 3: `fill_book` without nested closures

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/handoff.py` (`fill_book`, lines ~167-210)
- Test: `tests/strategies/sector_pods/test_sector_pods_handoff.py` (the `test_fill_book_*` tests stay as they are)

**Interfaces:**
- Consumes: `fill_book(weights: Mapping[str, float], state: Mapping[str, Any], params: SectorPodsParams) -> dict[str, float]` (unchanged signature and docstring).
- Produces: the same function, with module-level private helpers `_pod_totals(book: Mapping[str, float]) -> dict[str, float]` and `_scale(book: dict[str, float], symbols: Sequence[str], factor: float) -> None` replacing the closures `pod_totals` and `scale`.

- [ ] **Step 1: Run the existing tests** `uv run pytest tests/strategies/sector_pods/test_sector_pods_handoff.py -k fill_book -v`. Expected: PASS (the safety net before the rewrite).
- [ ] **Step 2: Move `pod_totals` and `scale` out of `fill_book` into the two module-level helpers**, pass `book` explicitly, and name the loop's steps (`room`, `open_`, the clip, the pod cut) so the loop reads as the docstring does. Keep each float expression and the order of operations unchanged.
- [ ] **Step 3: Verify.** Run the `fill_book` tests (PASS), then `uv run python <scratchpad>/capture_sector_pods.py > <scratchpad>/after.json && diff <scratchpad>/baseline.json <scratchpad>/after.json`. Expected: no difference.

### Task 4: `SectorPodsReview` replaces the node classes and the parts bundle

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/graph.py` (rewrite), `src/trading_agent_framework/strategies/sector_pods/agent_sector_pods.py`
- Test: `tests/strategies/sector_pods/test_sector_pods_graph.py` (the `_harness` and the three `Execute(...)` calls), `tests/strategies/sector_pods/test_sector_pods_strategy.py` (`_FakeGraph`, `_graph_parts`)

**Interfaces:**
- Produces, in `graph.py`:
  - `PodsState` (TypedDict, unchanged), `pod_reads`, `risk_reads`, `pm_reads` (unchanged).
  - `class SectorPodsReview`, `__init__(self, *, strategy: Strategy, params: SectorPodsParams, store: StateStore, rebalancer: Rebalancer, bars: DailyBars, log: RunLog, pods: Mapping[str, AgentNode], risk_manager: AgentNode, portfolio_manager: AgentNode) -> None`; builds the graph at the end of `__init__`. Public read-only attributes `pods`, `risk_manager`, `portfolio_manager` (the strategy test arms their recorders).
  - `SectorPodsReview.run(self) -> PodsState`: `self._graph.invoke({})` and returns the final state. The date comes from `market_facts`.
  - Node methods `_load_previous(state)`, `_market_facts(state)`, `_run_pod(key, state)`, `_execute(state)`; `_after_facts(state) -> list[str]`; `_build_graph() -> Any`.
- Removed: `GraphParts`, `CodeNode`, `LoadPrevious`, `MarketFacts`, `Execute`, `skip_if_abandoned`, `skip_pod_without_facts`, `build_graph`, and the strategy's `_info`, `_warn`, `_error`, `_graph_parts`, `_graph`.
- Produces, in the strategy: `self._review: SectorPodsReview | None`, built in `initialize` (replaces `self._graph`); `on_trading_iteration` calls `self._review.run()`.

- [ ] **Step 1: Rewrite the test harness first.** In `_harness`, build `SectorPodsReview(strategy=strategy, params=params, store=store, rebalancer=rebalancer, bars=bars, log=RunLog(log_path), pods=pods, risk_manager=..., portfolio_manager=...)`. Set `strategy.regime = 1` and `strategy.regime_reading`/`strategy.market_fall` so that `stress_view(...)` returns the harness's `stress` (the `stress` argument now builds those two attributes, or is replaced by them: keep the call sites of the tests readable); collect `infos`, `warnings`, `errors` by replacing `strategy.log_info`, `log_warning`, `log_error` with list appenders (the review logs through the strategy). `Harness.graph` becomes `Harness.review` and `run()` calls `self.review.run()`. Keep `_harness`'s keyword arguments and every other `Harness` field. The three `Execute(...)` tests call `h.review` instead.
- [ ] **Step 2: Run the graph tests.** Run: `uv run pytest tests/strategies/sector_pods/test_sector_pods_graph.py -q`. Expected: FAIL (import error on `SectorPodsReview`).
- [ ] **Step 3: Implement `SectorPodsReview` in `graph.py`** from the existing node bodies: `_market_facts` is `MarketFacts.__call__` plus `_outcome`, reading `strategy.regime`, `stress_view(strategy.regime_reading, strategy.market_fall)`, `rebalancer.current_weights()` and the market date from `strategy.clock`; warnings go to `strategy.log_warning`. `_execute` is `Execute.__call__` with `_trade`, `_filled_line`, `_trades_due` as private methods. `_run_pod(key, state)` returns `{"pods": {key: UNAVAILABLE}}` when `state["facts"].get(key)` is empty, else `self.pods[key](state)`. `_build_graph` adds the nodes from a `(name, method)` list like `BullBearReview._build_graph`, the five pods through `_run_pod`, `add_conditional_edges("market_facts", self._after_facts)` where `_after_facts` returns `["execute"]` for an abandoned week and the five pod keys otherwise, `add_edge([pod keys], "risk_manager")`, and the straight edges to `END`. No `skip_*` wrapper remains. Log the same message strings as today.
- [ ] **Step 4: Wire the strategy.** `initialize` builds the agents (`_node`, unchanged for now) and `self._review = SectorPodsReview(strategy=self, ...)`; delete `_info`, `_warn`, `_error`; `_daily_bars` stays. In the strategy tests `_FakeGraph` becomes a fake review with `run()`, and the `_graph_parts` test reads `strategy._review.pods`, `.risk_manager`, `.portfolio_manager`.
- [ ] **Step 5: Verify.** Run: `uv run pytest tests/strategies/sector_pods -q`. Expected: all PASS. Then `uv run python <scratchpad>/capture_sector_pods.py > <scratchpad>/after.json && diff <scratchpad>/baseline.json <scratchpad>/after.json`. Expected: no difference. `uv run pyright src/trading_agent_framework/strategies/sector_pods tests/strategies/sector_pods`: 0 errors.

### Task 5: Slimmer `_node` and one `UNAVAILABLE` helper

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/agent_sector_pods.py` (`_node`, `_graph_parts` successor)
- Test: `tests/strategies/sector_pods/test_sector_pods_strategy.py` (unchanged assertions)

**Interfaces:**
- Produces: `_node(self, name: str, *, system: str, tools: list[Callable[..., Any]], validator: Validator, submit: str, task: str, reads: Callable[..., Any], slot: Callable[[Any], dict[str, Any]], budget: int | None, exempt: Sequence[str] = ()) -> AgentNode`, keyword-only after `name`; and one module-level `_view(slot_key: str)`-style helper (the builder picks the name) returning the `write` callable `lambda view: {slot_key: UNAVAILABLE if view is None else view}`; the pods' write keeps its `{"pods": {key: ...}}` shape.

- [ ] **Step 1: Run `uv run pytest tests/strategies/sector_pods/test_sector_pods_strategy.py -q`.** Expected: PASS.
- [ ] **Step 2: Make `_node`'s parameters keyword-only and use the helper for the three write lambdas**; the call sites read `_node("risk_manager", system=..., tools=[], ...)`. Behaviour is unchanged.
- [ ] **Step 3: Verify** the strategy tests (PASS) and the capture diff (no difference).

### Task 6: Split `Execute._trade` (now `SectorPodsReview._trade`)

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/graph.py`
- Test: `tests/strategies/sector_pods/test_sector_pods_graph.py` (the `test_execute_*`, `test_a_*book*` and `test_broker_error_*` tests)

**Interfaces:**
- Produces: `_trade(self, state: PodsState) -> dict[str, Any]` returning `{"abandoned": str | None, "orders": list[dict], "book": dict[str, float] | None}` (no 3-tuple), built from short steps: `_target_book(state) -> dict[str, float]` (check, `fill_book`, check again; raises `ValueError(problem)` with today's problem text) and `_send_orders(book, state) -> dict[str, Any]` (the rebalance, the placed orders, the refused-everything and nothing-fundable cases). The `unreadable book (<Type>: <message>)` wording and the broad `except Exception` around the book build stay, because the PM payload is untrusted.

- [ ] **Step 1: Run the book tests.** Run: `uv run pytest tests/strategies/sector_pods/test_sector_pods_graph.py -q`. Expected: PASS.
- [ ] **Step 2: Split `_trade` into the two helpers above**; `_execute` reads the dict. Keep every log message and the order of the `on_info`/warning/error calls.
- [ ] **Step 3: Verify** the graph tests (PASS), the capture diff (no difference), and that no method of `graph.py` is longer than about 40 lines (`awk` or a look; a guide, not a gate).

### Task 7: Documentation

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/CLAUDE.md`, the module and class docstrings of `graph.py`, `agent_sector_pods.py`

- [ ] **Step 1:** Replace every mention of `GraphParts`, `LoadPrevious`, `MarketFacts`, `Execute`, `skip_if_abandoned`, `skip_pod_without_facts`, `build_graph` and `_graph_parts` in `strategies/sector_pods/CLAUDE.md` with `SectorPodsReview` and its methods; keep the architecture diagram, the state table and every gotcha, and add one line for the conditional edge (an abandoned week goes `market_facts -> execute`). `grep -rn "GraphParts\|LoadPrevious\|MarketFacts\|skip_if_abandoned\|build_graph" src tests docs/superpowers/specs docs/superpowers/plans/2026-11-09-sector-pods.md` must show no live reference in `src/` or `tests/` (old specs and plans are history: leave them).
- [ ] **Step 2:** Every new public class, method and function has a docstring (Google style); `uv run interrogate -v` is not lower than before.

### Task 8: Quality gate, simplification passes, commit

- [ ] **Step 1:** Run the full gate and report the numbers: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest`, `uv run pytest --cov`, `uv run interrogate -v`.
- [ ] **Step 2:** Run `code-simplifier:code-simplifier` on the changed files, then `/simplify` on the diff, as the user's rules say; re-run the gate.
- [ ] **Step 3:** Stage the diff (not `TODO.md`), run `/ponytail:ponytail-review` on it; fix every "Must fix", re-run the gate, list the "Should fix" to the user. Re-run the capture diff one last time (no difference).
- [ ] **Step 4:** Commit: `git commit -m "refactor(sector_pods): one review class, plain nodes, simpler fill_book"` with the session's attribution line. Do not push; ask the user.

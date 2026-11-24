# bull_bear as a LangGraph graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `strategies/bull_bear/pipeline.py` (`BullBearPipeline`) with `strategies/bull_bear/graph.py` (`BullBearReview`, a straight-line LangGraph `StateGraph`), with exactly the same behaviour.

**Architecture:** One plain class whose node methods are today's review code cut at the stage boundaries, wired `START -> rank -> fact_sheets -> research -> bull_case -> bear_case -> judge -> execute -> END` and run with a plain `invoke`. A failing stage raises `StageAbandoned` out of `invoke`; `run()` catches it the way `ReviewPipeline.run` does today. A golden capture of today's outputs, taken before any change, proves the new code is identical.

**Tech Stack:** Python 3.14, uv, LangGraph 1.2 (`langgraph>=1.2,<2`), pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-11-24-bull-bear-langgraph-design.md`

## Global Constraints

- Same behaviour, exactly: same orders, state file, `reviews.jsonl` line (keys, values, key order) and log messages, on completed and abandoned reviews.
- No code change in shared files (`common/pipeline.py`, `common/rebalancer.py`, `agents/`); wording-only fixes in `common/pipeline.py`'s docstring, `common/CLAUDE.md`, `agents/CLAUDE.md`.
- No new `Protocol`, no `ClassVar`, no generics, no decorators; `AgentLike` and `AgentLookup` move to `graph.py` unchanged.
- Plain `graph.invoke(...)` only: no streaming, `subgraphs=True`, `step_timeout` or checkpointer.
- Google-style docstring on every module, class and method; comments only for the why.
- Gate before every commit: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright` (0 errors), `uv run pytest` (all pass), `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%); no `# noqa` / `# type: ignore` added. Last step before every commit: `/simplify` on the diff, stage, `/ponytail:ponytail-review` on the staged changes.
- Every commit message ends with the attribution line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the model in use may differ; use the one the session gives).
- Run `uv sync --extra dashboard` once in the worktree before the first test run (a plain sync skips the dashboard tests).

## Review Focus

1. Two reviews on one `BullBearReview` (a restart in the same process, the next Tuesday of a backtest): the second must start from an empty record, `_stage = "setup"` and no `_pending`, even after an execution abandonment. Test in Task 2.
2. The `execute` node places orders and must run on the strategy (calling) thread. Test in Task 2.
3. The order of log lines and `record.update(...)` calls across node boundaries (bull/bear counts and the `[debate]` lines come after the bear). Covered by the golden comparison in Task 2.
4. An exception that is not an abandonment (`ConfigurationError` from an agent) must come out of `run()` unchanged with the state untouched. Already pinned by `test_a_configuration_error_from_an_agent_propagates_and_leaves_the_state`.
5. A `StageAbandoned` subclass (`AgentLooped`) raised inside a node must still be caught as an abandonment with its stage. Already pinned by the loop tests.

---

### Task 1: Golden capture of today's pipeline

**Files:**
- Create (scratchpad, never committed): `$SCRATCH/golden.py`, where `$SCRATCH` is the session scratchpad directory.

**Interfaces:**
- Consumes: from the test module given on the command line: `_harness`, `Harness` (`.run()`, `.store`, `.log_path`, `.orders`, `.agents`, `.rebalancer`), `FakeAgent`, `verdicts`, `does_nothing`, `_loops`, `_sells_s03`, `_raises_after`-style patching (done by hand: `setattr` on `h.rebalancer`), `UNIVERSE`, `BullBearState`, `BrokerError`, `PlacedOrder`.
- Produces: `golden.py <test_module> <out.json>` writes one JSON object per scenario name. Task 2 runs it again and compares.

- [ ] **Step 1: Write `golden.py`**

`uv run python $SCRATCH/golden.py tests.strategies.bull_bear.test_bull_bear_pipeline $SCRATCH/before.json` imports the module with `importlib.import_module`, and for each scenario builds a harness in a fresh `tempfile.TemporaryDirectory()`, attaches a `logging.Handler` collecting `record.getMessage()` (ANSI kept) at DEBUG on the `trading_agent_framework` logger (set the logger level to DEBUG for the run, restore after), runs `h.run()` and stores:
`outcome` (`asdict`), `orders` (`h.orders`), `state` (`asdict(h.store.load())`), `lines` (the raw text of `reviews.jsonl`, so key order counts), `messages`, and per agent the list of calls with `task`, `context`, `force_tool`, `tool_budget`, `max_steps` and `same_run_id_as_previous` (bool; uuids differ per run). Dump with `json.dumps(..., default=str, indent=1)`.

Scenarios (name: setup):
- `completed`: `_harness(tmp)`.
- `held_and_forced`: `held={"OLD": 20, "S03": 20}`, judge `_sells_s03`.
- `research_fails`: researcher overrides `[lambda tools, ctx: (_ for _ in ()).throw(AgentError("boom"))]` for its first two calls (both attempts of the first stock).
- `bull_silent`, `bear_silent`, `judge_silent`: that agent's script `does_nothing`.
- `bear_loops`: bear script `_loops`.
- `judge_retry`: judge overrides `[verdicts(UNIVERSE[:11])]`.
- `broker_before`: `h.rebalancer.holdings` raises `BrokerError("positions unavailable")`.
- `execution_half_way`: `held={"S03": 20}`, state `BullBearState(verdicts={"S03": {"verdict": "in", "reason": "old", "date": "2026-09-29", "strength": "standard"}}, cooldowns={"S05": 2}, abandoned_streak=1)`, judge `_sells_s03`, `h.rebalancer.rebalance` sets `placed = [PlacedOrder("S03", "sell", 20.0), PlacedOrder("S00", "buy", 10.0)]` then raises `BrokerError("connection reset")`.
- `previous_and_cooldown`: same state as above without the failure (judge `verdicts()`).

- [ ] **Step 2: Run it on today's code**

Run: `uv run python $SCRATCH/golden.py tests.strategies.bull_bear.test_bull_bear_pipeline $SCRATCH/before.json && python3 -c "import json;d=json.load(open('$SCRATCH/before.json'));print(len(d), sorted(d))"`
Expected: `11` and the 11 scenario names; `completed` has a non-empty `orders` list and `execution_half_way` an `"execution"` stage in its line.

Nothing to commit (scratchpad only). Ledger the path of `before.json`.

---

### Task 2: `BullBearReview` graph, the strategy wired to it, tests ported

**Files:**
- Move: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` -> `graph.py` (`git mv`, then rewrite)
- Move: `tests/strategies/bull_bear/test_bull_bear_pipeline.py` -> `test_bull_bear_graph.py` (`git mv`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` (import, construction, `self.pipeline` -> `self.review`, class docstring line 62, annotation line 127, error text line 221)
- Modify: `tests/strategies/bull_bear/test_bull_bear_strategy.py`, `tests/strategies/bull_bear/test_bull_bear_universe_history.py` (`strategy.pipeline` -> `strategy.review`, `_FakePipeline` -> `_FakeReview`)

**Interfaces:**
- Produces: `BullBearReview(*, strategy, params, agents, recorder, state, log, rebalancer, universe_on, bars, sector_of, fundamentals_of, momentum)` (the pipeline's keyword arguments, unchanged); `run() -> RunOutcome`; `completed_today() -> bool`; `_graph` (compiled, typed `Any`). Also `AgentLike`, `AgentLookup`, `AgentLooped`, `ReviewState` in `graph.py`.
- Consumes: `StageAbandoned`, `RunOutcome`, `AgentLike as CommonAgentLike` from `common/pipeline.py` (import only).

- [ ] **Step 1: Move the files and port the tests**

`git mv` both files. In the test module: import `BullBearReview` from `...bull_bear.graph`; the harness builds it (same arguments); the `Harness` field `pipeline` becomes `review` (update its 8 uses). Then add, in the test module:

```python
def test_the_review_graph_is_a_straight_line(tmp_path: Path) -> None:
    graph = _harness(tmp_path).review._graph.get_graph()
    steps = ["__start__", "rank", "fact_sheets", "research", "bull_case", "bear_case", "judge", "execute", "__end__"]
    assert sorted((edge.source, edge.target) for edge in graph.edges) == sorted(zip(steps, steps[1:], strict=False))


def test_a_review_after_an_execution_abandonment_starts_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)

    def half_way(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        h.rebalancer.placed = [PlacedOrder("S00", "buy", 10.0)]
        raise BrokerError("connection reset")

    def broken() -> list[str]:
        raise BrokerError("positions unavailable")

    monkeypatch.setattr(h.rebalancer, "rebalance", half_way)
    h.run()
    monkeypatch.undo()
    monkeypatch.setattr(h.rebalancer, "holdings", broken)
    h.run()
    first, second = h.log_lines()
    assert first["stage"] == "execution"
    # A stale `_stage` would call this "execution"; a stale record would carry the first review's verdicts and orders.
    assert second["stage"] == "broker" and "verdicts" not in second and "orders" not in second
```

The test module needs `import threading` for the next test:

```python
def test_the_orders_are_placed_on_the_calling_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _harness(tmp_path)
    rebalance, threads = h.rebalancer.rebalance, []

    def spy(target: Any, forced_exits: Any = ()) -> list[PlacedOrder]:
        threads.append(threading.get_ident())
        return rebalance(target, forced_exits)

    monkeypatch.setattr(h.rebalancer, "rebalance", spy)
    assert h.run().completed
    assert threads == [threading.get_ident()]
```

- [ ] **Step 2: Run the new tests to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_graph.py -q -x`
Expected: FAIL at collection: `ImportError: cannot import name 'BullBearReview'`.

- [ ] **Step 3: Rewrite `graph.py`**

Module docstring: the review as a graph (the node line above), what code does vs the agents, the abandonment rule, why `_record`/`_stage`/`_pending` are attributes (a failed `invoke` returns no state), the plain-`invoke` threading rule, the two spec links of today's docstring plus the new spec.

Contents, in this order:
- `AgentLike(CommonAgentLike, Protocol)`, `AgentLookup(Protocol)`, `AgentLooped(StageAbandoned)`, `_Pending`: unchanged from `pipeline.py`.
- `class ReviewState(TypedDict)` (every key required), slots and types: `now: datetime`, `saved: BullBearState`, `debate: DebateSet`, `sheets: dict[str, dict[str, Any]]`, `notes: dict[str, str]`, `bull: dict[str, BullCase]`, `bear: dict[str, BearCase]`, `verdicts: Verdicts`, `previous: dict[str, dict[str, str]]`. Runtime imports (not under `TYPE_CHECKING`): `StateGraph` reads the hints. Use the real fact-sheet row type if `fact_sheet_row` has one.
- `class BullBearReview`: no base class. `__init__` stores the arguments (as `pipeline.py` + `ReviewPipeline.__init__` did, same attribute names `_strategy`, `_agents`, `_recorder`, `_state`, `_log`, ...), sets `_record: dict[str, Any] = {}`, `_stage = "setup"`, `_pending = None`, then `self._graph = self._build_graph()`.
- `run(self) -> RunOutcome`: the body of `ReviewPipeline.run` specialised to bull_bear, this exact logic:

```python
state = self._state.load()
now = self._now()
self._record = {"date": now.date().isoformat(), "run_id": self._strategy.run_id, "abandoned": False}
self._stage, self._pending = "setup", None
try:
    self._graph.invoke({"now": now, "saved": state})
    return RunOutcome(completed=True, abandoned_streak=0)
except StageAbandoned as exc:
    stage, error = exc.stage, exc
except (BrokerError, BacktestError) as exc:
    stage, error = ("execution" if self._stage == "execution" else "broker"), exc
streak = state.abandoned_streak + 1
self._strategy.log_error(f"[bull_bear] review abandoned at the {stage} stage (streak {streak}): {error}")
saved, extra = self._on_abandoned(state, streak, stage)
self._state.save(saved)
self._log.append({**self._record, "abandoned": True, "stage": stage, "error": str(error), "abandoned_streak": streak, **extra})
return RunOutcome(completed=False, abandoned_streak=streak)
```

  Check `_now()` returns exactly what `ReviewPipeline.run` computed (`strategy.clock.now().astimezone(MARKET_TZ)`).
- `completed_today`, `_now`, `_fundamentals`, `_on_abandoned`, `_rank`, `_run_stage`: bodies unchanged (`_on_abandoned` reads `self._stage`/`self._pending` as before).
- Nodes, each `(self, state: ReviewState) -> dict[str, Any]`, bodies cut from `_review`, `_research_notes`, `_debate_and_judge` with every `record` replaced by `self._record`, every log line and `record.update` in its original order:
  - `_rank_node` -> `{"debate": debate}` (holdings, `_rank`, `build_debate_set` with `excluded=state["saved"].cooldowns`, record, log line, `StageAbandoned("data", "no stock to debate")`).
  - `_fact_sheets_node` -> `{"sheets": ...}`.
  - `_research_node` -> `{"notes": ...}` (today's `_research_notes` body, inlined or called).
  - `_bull_node` -> `{"bull": ...}`; `_bear_node` -> `{"bear": ...}` and then the `bull_conviction`/`bear_risk` record update and the `[debate]` lines. Both build `evidence` and `limits` the same way.
  - `_judge_node` -> `{"verdicts": ..., "previous": ...}`.
  - `_execute_node` -> `{}` (book, sizing, `_pending`, record, forced-exit warnings, `self._stage = "execution"`, rebalance, state save, log append).
- `_build_graph(self) -> Any`: `StateGraph(ReviewState)`, the 7 nodes named `rank`, `fact_sheets`, `research`, `bull_case`, `bear_case`, `judge`, `execute`, the 8 edges, `compile()`. Import langgraph inside, as ray_dalio does.
- `_stop_detail`, `_submission_lines`: unchanged.

Drop `ReviewPipeline`, `ClassVar`, `LOG_PREFIX`, `_executing`, `_run`, `_review`, `_debate_and_judge` (the tuple return goes with it).

- [ ] **Step 4: Wire the strategy**

`agent_bull_bear.py`: import `BullBearReview` from `.graph`; `self.review: BullBearReview | None = None`; `self.review = BullBearReview(...)`; `on_trading_iteration` uses `self.review` and says "before initialize() built the review"; the module and class docstrings name `BullBearReview`. Update the two strategy test files to `strategy.review` / `_FakeReview`.

- [ ] **Step 5: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass (the ported 78 plus the 3 new ones), no assertion edited except the `pipeline` -> `review` renames.

- [ ] **Step 6: Golden comparison**

Run: `uv run python $SCRATCH/golden.py tests.strategies.bull_bear.test_bull_bear_graph $SCRATCH/after.json && diff $SCRATCH/before.json $SCRATCH/after.json && echo IDENTICAL`
Expected: `IDENTICAL`. Any difference is a behaviour change: fix the code, never the golden file.

- [ ] **Step 7: Gate, simplify, review, commit**

Run the full gate (Global Constraints), `/simplify`, `/ponytail:ponytail-review`, then:

```bash
git add -A src/trading_agent_framework/strategies/bull_bear tests/strategies/bull_bear
git commit -m "refactor(bull_bear): the weekly review as a LangGraph graph (BullBearReview)"
```

---

### Task 3: Docs

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (architecture: `graph.py`, the node line, the `ReviewState` slots, attributes kept outside the graph and why, the plain-`invoke` threading rule as in `ray_dalio/CLAUDE.md`; every `pipeline`/`BullBearPipeline` mention)
- Modify: `src/trading_agent_framework/strategies/CLAUDE.md` (the bull_bear line, if it names the pipeline)
- Modify (wording only): `src/trading_agent_framework/strategies/common/pipeline.py` docstring line 3, `src/trading_agent_framework/strategies/common/CLAUDE.md` line 12, `src/trading_agent_framework/agents/CLAUDE.md` line 22
- Modify: `scripts/replays/bull_bear_bear_balance.py` docstring line 317 (`BullBearPipeline._run_stage` -> `BullBearReview._run_stage`)

- [ ] **Step 1: Find every stale mention**

Run: `git grep -n "BullBearPipeline\|bull_bear/pipeline\|bull_bear\.pipeline\|strategy\.pipeline" -- src tests scripts README.md`
Expected: only the files listed above (docs under `docs/superpowers/` are history and stay).

- [ ] **Step 2: Fix the wording**

In the shared files, say that bull_bear runs as a graph in `bull_bear/graph.py` and no longer uses `ReviewPipeline`; change nothing else there. In `agents/CLAUDE.md`, point at wherever `tool_failed` actually lives now (`git grep -n "def tool_failed"`).

- [ ] **Step 3: Re-run the search**

Run: the Step 1 command.
Expected: no match outside `docs/superpowers/`.

- [ ] **Step 4: Gate, simplify, review, commit**

Full gate, `/simplify`, `/ponytail:ponytail-review`, then:

```bash
git add -A src scripts
git commit -m "docs(bull_bear): the review graph in the CLAUDE.md files and docstrings"
```

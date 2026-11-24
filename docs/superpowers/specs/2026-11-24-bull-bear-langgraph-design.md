# bull_bear as a LangGraph graph

Date: 2026-11-24. Branch `refactor/bull-bear-langgraph`. Status: approved 2026-11-24.

## Intent

bull_bear's weekly review is a hand-written chain in `strategies/bull_bear/pipeline.py` (`BullBearPipeline`, 432
lines, a subclass of the shared `common/pipeline.ReviewPipeline` template). The user is moving the LLM strategies to
LangGraph one by one, the way `ray_dalio` already is (`ray_dalio/agent_ray_dalio.py`: a `TypedDict` state, one method
per node, `_build_graph` wiring them, a plain `graph.invoke`). bull_bear is next. The aim is code a person reads top
to bottom: the graph shows the review's steps, each step is one method.

## Goals

- The review is a LangGraph `StateGraph`, a straight line:
  `START -> rank -> fact_sheets -> research -> bull_case -> bear_case -> judge -> execute -> END`.
- **Same behaviour, exactly.** For the same agent answers and the same data, the review places the same orders, saves
  the same state file, appends the same `reviews.jsonl` line (same keys, same values, same key order) and writes the
  same log messages, on a completed review and on every kind of abandoned review. The 5Y and COVID results stay valid.
- Plain, documented code: no new `Protocol` (the two existing ones stay, see Agents), no `ClassVar`, no generics, no decorators. Every module, class and method has
  a docstring (Google style, the repo's convention) saying what it does and anything non-obvious.

## Non-goals

- No code change to the shared code (wording fixes in docs and docstrings only, below): `common/pipeline.py` (still used by `bill_ackman`, `congress_trades`,
  `insider_trades`), `common/rebalancer.py`, `agents/`. bull_bear may keep **importing** `StageAbandoned` and
  `RunOutcome` from `common/pipeline.py`; it stops subclassing `ReviewPipeline`.
- No change to the prompts, the handoff tools (`handoff.py`), the state file (`state.py`), the fact sheet, the debate
  set, the book, sizing, parameters, the agents' settings in `initialize`, or `run_backtesting`.
- No parallelism: the bull and the bear stay sequential (one `HandoffRecorder` is armed for one stage at a time), the
  researcher stays a loop over the stocks inside one node.

## Anti-goals (a failure even if it runs)

- Any difference in orders, state, `reviews.jsonl` or log text against today's pipeline.
- Streaming, `subgraphs=True`, `step_timeout` or a checkpointer on the graph: the first three send nodes to worker
  threads, and `execute` places orders, which must stay on the strategy thread (root `CLAUDE.md`).
- An abstraction the review does not need (a node base class, a node registry, a generic stage runner type).

## Approaches considered

- **A. A graph class in a new `bull_bear/graph.py` replacing `pipeline.py` (chosen).** A plain class `BullBearReview`
  with the pipeline's constructor arguments; one method per node; a straight-line graph; a failing stage raises the
  existing `StageAbandoned`, which passes through `graph.invoke`, and `run()` catches it as `ReviewPipeline.run` does
  today (about 30 lines copied, not shared). The 78 pipeline tests port with a class rename, which is what proves
  "same behaviour".
- B. Every node a method of `BullBearStrategy`, exactly like ray_dalio. Dropped: the strategy file (321 lines) would
  reach about 700, and every pipeline test would need a whole strategy around its fakes (a rewrite, not a port).
- C. Like A, with abandonment routed through the graph (an `abandoned` flag in the state, conditional edges to an
  `abandon` node, as sector_pods). Dropped: a check in every node and a router per edge, and an abandonment during
  execution (orders sent, cooldowns started) is easy to get subtly wrong when it moves across edges.

## The design

### Files

- New `strategies/bull_bear/graph.py`: `ReviewState`, `BullBearReview`, `AgentLooped`, `AgentLike`, `AgentLookup`, and the two log helpers
  `_stop_detail` and `_submission_lines` moved unchanged.
- Delete `strategies/bull_bear/pipeline.py` (`git mv` to `graph.py`, then rewrite, so history follows).
- `agent_bull_bear.py`: builds `BullBearReview` instead of `BullBearPipeline`; the attribute `self.pipeline` becomes
  `self.review` (our call: the name says what it is); the class docstring, the attribute annotation and the
  `FatalStrategyError` text ("before initialize() built the review") follow. Nothing else in it changes.
- Tests: `tests/strategies/bull_bear/test_bull_bear_pipeline.py` -> `test_bull_bear_graph.py` (`git mv`), the harness
  builds `BullBearReview`; `test_bull_bear_strategy.py` and `test_bull_bear_universe_history.py` (reads
  `strategy.pipeline._universe_on`) follow the attribute rename.
- Shared docs, wording only (approved by the user; no code change in these files): the module docstring of
  `common/pipeline.py` (line 3 lists `BullBearPipeline` among its users), `common/CLAUDE.md` line 12 (the same list,
  bull_bear's `_stage` reset and its `AgentLike` subclass) and `agents/CLAUDE.md` line 22 (points at
  `bull_bear/pipeline.py`). They now say bull_bear runs as a graph in `bull_bear/graph.py`.
- Docs: `bull_bear/CLAUDE.md` (architecture: the graph and its nodes, the threading rule), the bull_bear line of
  `strategies/CLAUDE.md` if it names the pipeline, the module docstring of `agent_bull_bear.py`, and the docstring
  mention in `scripts/replays/bull_bear_bear_balance.py` (`BullBearPipeline._run_stage`).

### State and attributes

Two kinds of data, kept apart on purpose:

- **Graph state** (`ReviewState`, a plain `TypedDict`, every key required: pyright's standard mode refuses
  `state["debate"]` on a `total=False` dict, and LangGraph does not check it at run time; its slot types are runtime
  imports, since `StateGraph` reads the hints with `get_type_hints`): what flows from one step to the next. One writer per
  slot. Slot names must differ from node names (LangGraph refuses a node named like a state key), hence `bull_case`
  the node and `bull` the slot.

  | Slot | Written by | What |
  |---|---|---|
  | `now` | `run()` (input) | the market-time `datetime` of the review |
  | `saved` | `run()` (input) | the `BullBearState` loaded at the start |
  | `debate` | `rank` | the `DebateSet` (candidates and forced exits) |
  | `sheets` | `fact_sheets` | symbol -> fact sheet row |
  | `notes` | `research` | symbol -> note (`UNAVAILABLE_NOTE` when research failed) |
  | `bull` | `bull_case` | symbol -> `BullCase` |
  | `bear` | `bear_case` | symbol -> `BearCase` |
  | `verdicts` | `judge` | the `Verdicts` |
  | `previous` | `judge` | symbol -> the judge's previous verdict entry |

- **Attributes of `BullBearReview`** that an abandonment needs after a node raised (LangGraph returns no state from a
  failed `invoke`): `self._record` (the `reviews.jsonl` line being filled in; an abandoned line carries what was
  filled so far, exactly as today), `self._stage` (`"setup"` / `"execution"`) and `self._pending` (the `_Pending`
  verdicts and sells, set just before execution). All three are reset at the start of every `run()`.

### Agents

bull_bear's two protocols move to `graph.py` unchanged (the user's call: cleaner than `Any`): `AgentLike` (the
shared `common/pipeline.AgentLike` plus `max_steps`) and `AgentLookup` (an agent by name: an `AgentManager` in the
strategy, a plain dict of fakes in the tests). They are the only exception to "no `Protocol`".

### Nodes

Each node is a method taking `state` and returning a dict of the slots it writes. The bodies are today's code from
`_review`, `_rank`, `_research_notes` and `_debate_and_judge`, cut at the stage boundaries, with the same log lines
and the same `record.update(...)` calls in the same order.

- `rank`: holdings, ranking (`_rank`, unchanged), the debate set, the `record` fields and the "N stocks ranked;
  debate: ...; forced exits: ..." line; `StageAbandoned("data", "no stock to debate")` on an empty set.
- `fact_sheets`: one fact sheet per candidate, fundamentals as of `strategy.clock.now()` (`_fundamentals`, unchanged).
- `research`: the researcher loop (one run per stock; a failure costs that stock its note).
- `bull_case`, `bear_case`: one run each through `_run_stage`. The `bull_conviction` / `bear_risk` counts and the
  per-stock `[debate]` lines are written by `bear_case`, right after the bear, as today.
- `judge`: previous verdicts, the recorder armed, the judge's run, the `[judge] in: ...; out: ...` line.
- `execute`: book, sizing, target, sells and `_Pending`, the record fields, the forced-exit warnings,
  `self._stage = "execution"`, the rebalance, then the state save and the `reviews.jsonl` append of a completed review.

`_run_stage` stays a method with today's body (attempt, forced retry, `AgentLooped` on a loop).

### Running a review

`run()` returns a `RunOutcome`, as today:

1. Load the state, take `now` (market time), reset `_record` to `{"date", "run_id", "abandoned": False}`, `_stage`,
   `_pending`.
2. `self._graph.invoke({"now": now, "saved": state})`; on return, `RunOutcome(completed=True, abandoned_streak=0)`.
3. `StageAbandoned` -> its stage; `BrokerError` / `BacktestError` -> `"execution"` when `_stage` is `"execution"`, else
   `"broker"`. Anything else propagates (a `ConfigurationError` from an agent, as today).
4. On an abandonment: streak + 1, the error line `[bull_bear] review abandoned at the <stage> stage (streak N): ...`,
   the state and the extra fields from `_on_abandoned` (today's body), the save, the abandoned line appended.

`completed_today()` is unchanged. The graph is compiled once in `__init__` and run with a plain `invoke`.

## Proof of "same behaviour"

1. **Before any change**, a throwaway script (scratchpad, not committed) runs the current pipeline's test harness on
   a set of scenarios (completed review; holdings sold; forced exits; failed research; each debater abandoned; a loop;
   a judge retry; a broker failure during execution) and writes, per scenario, the orders, the saved state, the
   `reviews.jsonl` lines and the log messages to a golden file. After the refactor the same script runs against
   `BullBearReview` and the outputs must be identical.
2. The 78 ported tests pass with their assertions unchanged (only the class name, the import and the harness change).
3. One new test: the compiled graph's nodes and edges are the straight line above.
4. The repo gate: ruff, ruff format, pyright, pytest with coverage >= 80%, interrogate.

## Left to the builder (the plan)

- The exact cut of `_review` into node bodies, as long as the order of log lines and record updates is unchanged.
- How the golden script reaches the harness (import it from the test module, or copy it).

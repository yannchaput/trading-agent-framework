# sector_pods: simpler, more readable code (design)

Date: 2026-11-25. Scope: `src/trading_agent_framework/strategies/sector_pods/` and its tests. Nothing else changes.

## Intent

`sector_pods` works, but it is hard to read: five graph types (`GraphParts`, `CodeNode`, `LoadPrevious`, `MarketFacts`, `Execute`) only to wire a seven-node linear graph, an agent factory with ten positional arguments, three one-line log forwarders, hand-made skip wrappers for what LangGraph does with a conditional edge, and `fill_book` as nested closures around a capped redistribution loop. The user wants the KISS principle applied: fewer classes, plain functions and dicts, code a human can follow, in the shape the `bull_bear` graph refactor (#45) already established.

## Goals

- Same behaviour, byte for byte: orders, `data/sector_pods_state_<mode>.json`, `reviews.jsonl`, log lines.
- One class for the weekly review, `SectorPodsReview` in `graph.py`, whose nodes are methods and whose graph is built from a `(name, method)` list, like `BullBearReview._build_graph`. It takes the strategy (logger, clock, regime) instead of the callbacks `on_info`, `on_warning`, `on_error`, `regime`, `stress`, `today`.
- Gone: `GraphParts`, `CodeNode`, `LoadPrevious`, `MarketFacts`, `Execute`, `skip_if_abandoned`, `skip_pod_without_facts`, and the strategy's `_info`, `_warn`, `_error` forwarders.
- Idiomatic LangGraph: an abandoned week leaves `market_facts` through one conditional edge to `execute`, instead of wrapping risk, PM and every pod. A pod with no facts is handled inside its own node. The five pods always run, because LangGraph's multi-source join waits for every listed source (to be confirmed by a spike, see Open points).
- `Execute._trade` split into short steps; no 3-tuple return.
- `_node` (ten positional arguments) reduced: keyword arguments, and the three `UNAVAILABLE if view is None else view` lambdas become one helper.
- `handoff.fill_book` rewritten without nested closures; same results.
- Tests updated to the new construction; `CLAUDE.md` of the strategy, docstrings and the spec/plan references updated.

## Non-goals and anti-goals

- No change to prompts, parameters (`SectorPodsParams` fields), the state-file or `reviews.jsonl` format, the submit tool signatures (LangChain reads them) or the TypedDicts behind them.
- No change to `agents/` (`AgentNode`, `NodeRecorder`, `LengthGate`), `common/` or other strategies.
- Anti-goal: a refactor that adds an abstraction. Plain functions and dicts; no new class, Protocol or type unless it removes more than it adds. No `# noqa`, `# type: ignore` or lowered gate.

## Approaches

- **A. One class, node methods (bull_bear's shape).** Consistent with `bull_bear`; removes every per-node class and the parts bundle. Trade-off: a constructor with several arguments.
- **B. Plain functions, nodes as closures in `build_graph(strategy)`.** Fewest types, but `Execute` is about 80 lines and becomes a long nested function; drifts from `bull_bear`.
- **C. A plus idiomatic routing and smaller steps (picked).** A, with the conditional edge, the split `_trade`, the slimmer `_node` and a readable `fill_book`.

## Safety net

Before the first edit, capture the current behaviour into the scratchpad as JSON: the existing graph-test scenarios (fake agents, fake broker) recorded as orders, saved state, `reviews.jsonl` text and log messages, plus a table of `fill_book` and `check_weights` inputs and outputs (caps binding, unresearched ETFs, all ETFs capped, float drift). After the refactor the same capture must be identical. Tests that built `GraphParts`/`Execute` directly move to the new construction without losing a case.

## Constraints

- The thread rules of the root `CLAUDE.md` hold: with `invoke()` and no `step_timeout`, single nodes (including order code) run on the strategy thread; the five pods run on workers and only call read-only tools. A conditional edge does not change that.
- LangGraph refuses a node named like a state key; keep the current node names.
- Quality gate: ruff, ruff format, pyright, pytest, coverage >= 80%, interrogate >= 80%.
- Pipeline: feature branch `refactor/sector-pods-simplify`; `code-simplifier` runs on the diff and `/simplify` once before the commit; no push, PR or merge without the user's go-ahead.

## Open points (the builder settles them, within the goals)

- Confirm by a short spike that a conditional edge from `market_facts` to `execute` and the fan-out to the five pods behave as today in LangGraph 1.2.16 (state after an abandoned week, parallel pods, join).
- The exact split of `Execute._trade`, the names of the helpers, and how far `fill_book` shrinks: any form is fine if the capture matches and the function has no nested closures.

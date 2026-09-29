"""The per-tick LangGraph (spec §4): classify, then the exit agent, then the entry agent, each only when due.

`build_tick_graph` takes the three node callables, so the routing is tested with fakes and the strategy
injects its real nodes. No checkpointer (what lasts between ticks lives in `strategy.vars`) and no retry
policy (re-running an agent node could place an order twice). `langgraph` is imported inside
`build_tick_graph`: `main.py` imports every strategy, and one that never builds this graph must not load it.
"""

import operator
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, TypedDict

CLASSIFY_NODE = "classify"
EXIT_NODE = "exit_agent"
ENTRY_NODE = "entry_agent"
_END = "__end__"  # langgraph.graph.END


class TickState(TypedDict, total=False):
    """What the nodes of one tick share. Rebuilt by every `invoke`; nothing here outlives the tick."""

    now: datetime  # the tick's time (strategy clock), set by the caller
    exit_due: list[str]  # symbols of open trades the exit agent should review (set by classify)
    entry_due: bool  # a setup triggered and an entry is allowed (set by classify, refreshed by the exit node)
    # Per-agent run summaries; `operator.add` makes LangGraph append each node's list instead of replacing it.
    runs: Annotated[list[dict[str, Any]], operator.add]


# A node receives the state and returns only the keys it updates.
Node = Callable[[TickState], dict[str, Any]]


def route_after_classify(state: TickState) -> str:
    """Exit reviews come first (they may free a slot for an entry this same tick), then entries, else stop."""
    if state.get("exit_due"):
        return EXIT_NODE
    if state.get("entry_due"):
        return ENTRY_NODE
    return _END


def route_after_exit(state: TickState) -> str:
    """After the exit agent, enter only if `entry_due` is still true (the exit node recomputes it)."""
    return ENTRY_NODE if state.get("entry_due") else _END


def build_tick_graph(*, classify: Node, run_exit: Node, run_entry: Node) -> Any:
    """Compile START -> classify -> [exit_agent] -> [entry_agent] -> END; each agent node only when due."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(TickState)
    builder.add_node(CLASSIFY_NODE, classify)
    builder.add_node(EXIT_NODE, run_exit)
    builder.add_node(ENTRY_NODE, run_entry)
    builder.add_edge(START, CLASSIFY_NODE)
    builder.add_conditional_edges(CLASSIFY_NODE, route_after_classify, {EXIT_NODE: EXIT_NODE, ENTRY_NODE: ENTRY_NODE, _END: END})
    builder.add_conditional_edges(EXIT_NODE, route_after_exit, {ENTRY_NODE: ENTRY_NODE, _END: END})
    builder.add_edge(ENTRY_NODE, END)
    return builder.compile()

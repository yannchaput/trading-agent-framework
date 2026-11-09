"""The sector_pods weekly graph: load last week, market facts, five parallel pods, risk, portfolio manager, execute.

The nodes share one `PodsState`. Code nodes are the classes below; the agent nodes are `AgentNode`s built by the
strategy. The graph stays linear: once a week is abandoned every agent node returns without running.
"""

import operator
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from typing import Annotated, Any, Protocol, TypedDict

from trading_agent_framework.agents.graph import AgentNode, merge_dicts
from trading_agent_framework.strategies.common.daily_bars import DailyBars, DataUnavailable
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.sector_pods.facts import etf_facts, outcome, week_return_pct
from trading_agent_framework.strategies.sector_pods.handoff import UNAVAILABLE, check_weights, unresearched
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, PODS
from trading_agent_framework.strategies.sector_pods.store import StateStore, WeekState
from trading_agent_framework.utils.errors import BacktestError, BrokerError

_SPY = "SPY"
_REGIME_LABELS: dict[int | None, str] = {1: "bullish", 0: "neutral", -1: "bearish", None: "unknown"}


class PodsState(TypedDict, total=False):
    """The shared state of one weekly review.

    `pods` is merged across the parallel pod nodes and `errors` is appended to; every other slot has one writer.
    """

    date: str
    regime: str  # bullish / neutral / bearish / unknown
    holdings: dict[str, float]
    previous: dict[str, Any]
    facts: dict[str, dict[str, dict[str, float]]]  # pod key -> symbol -> row
    outcome: dict[str, Any]
    pods: Annotated[dict[str, Any], merge_dicts]
    risk: dict[str, Any]
    pm: dict[str, Any]
    errors: Annotated[list[dict[str, str]], operator.add]
    orders: list[dict[str, Any]]
    abandoned: str | None  # why the week was abandoned; None when completed


class CodeNode(Protocol):
    """A graph node: reads the shared state, returns the slots it writes."""

    def __call__(self, state: PodsState) -> dict[str, Any]:
        """Run the node."""
        ...


@dataclass(frozen=True)
class GraphParts:
    """The nodes `build_graph` wires together."""

    load_previous: CodeNode
    market_facts: CodeNode
    pods: Mapping[str, AgentNode]
    risk_manager: AgentNode
    portfolio_manager: AgentNode
    execute: CodeNode


def _real(view: Any) -> Any:
    """A saved view, or None when there is none or it was the unavailable marker."""
    return None if not view or view == UNAVAILABLE else view


def _previous(state: Mapping[str, Any], slot: str) -> Any:
    return _real((state.get("previous") or {}).get(slot))


def pod_reads(key: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Slots a pod sees: its own facts and its own view of last week."""

    def reads(state: Mapping[str, Any]) -> dict[str, Any]:
        last = _real((_previous(state, "pods") or {}).get(key))
        return {"date": state.get("date"), "regime": state.get("regime"), "facts": state.get("facts", {}).get(key, {}), "previous": last}

    return reads


def risk_reads(state: Mapping[str, Any]) -> dict[str, Any]:
    """Slots the risk manager sees: every pod's facts and view, plus its own last review."""
    return {key: state.get(key) for key in ("date", "regime", "holdings", "facts", "pods")} | {"previous": _previous(state, "risk")}


def pm_reads(state: Mapping[str, Any]) -> dict[str, Any]:
    """Slots the portfolio manager sees: everything, plus the held ETFs nobody researched and last week's outcome."""
    keys = ("date", "regime", "holdings", "facts", "outcome", "pods", "risk")
    return {key: state.get(key) for key in keys} | {"unresearched": unresearched(state), "previous": _previous(state, "pm")}


def skip_if_abandoned(node: CodeNode) -> CodeNode:
    """Wrap an agent node so it does nothing once the week is abandoned."""

    def guarded(state: PodsState) -> dict[str, Any]:
        return {} if state.get("abandoned") else node(state)

    return guarded


def skip_pod_without_facts(key: str, node: CodeNode) -> CodeNode:
    """Wrap a pod node so a pod with no ETF facts is not run and is recorded as unavailable."""

    def guarded(state: PodsState) -> dict[str, Any]:
        return node(state) if state.get("facts", {}).get(key) else {"pods": {key: UNAVAILABLE}}

    return guarded


class LoadPrevious:
    """Node: put last completed week's views in the state."""

    def __init__(self, store: StateStore) -> None:
        """Read from `store`."""
        self._store = store

    def __call__(self, state: PodsState) -> dict[str, Any]:
        """Return `{"previous": ...}`."""
        return {"previous": self._store.load().previous}


class MarketFacts:
    """Node: today's date, regime label, holdings, per-ETF facts and the score of last week's views."""

    def __init__(
        self, bars: DailyBars, *, regime: Callable[[], int | None], holdings: Callable[[], dict[str, float]], today: Callable[[], date], on_warning: Callable[[str], None] | None = None
    ) -> None:
        """Create the node.

        Args:
            bars: Daily bars source.
            regime: The strategy's regime (1, 0, -1 or None).
            holdings: Current weight of each held ETF.
            today: The session date (never the wall clock).
            on_warning: Told when last week's saved book cannot be read (no outcome is then given).
        """
        self._warn = on_warning or (lambda _message: None)
        self._bars = bars
        self._regime = regime
        self._holdings = holdings
        self._today = today

    def __call__(self, state: PodsState) -> dict[str, Any]:
        """Load the bars; abandon the week (`data: ...`) when they are unusable or no ETF has enough history."""
        holdings = self._holdings()
        base: dict[str, Any] = {"date": self._today().isoformat(), "regime": _REGIME_LABELS[self._regime()], "holdings": holdings}
        try:
            series = self._bars.load((*ALL_SYMBOLS, _SPY), list(holdings))
        except DataUnavailable as exc:
            return {**base, "abandoned": f"data: {exc}"}
        spy = series[_SPY].closes if _SPY in series else []
        facts = {pod.key: {s: row for s in pod.sleeve if s in series and (row := etf_facts(series[s].closes, spy)) is not None} for pod in PODS}
        if not any(facts.values()):
            return {**base, "abandoned": "data: no ETF has enough history"}
        return {**base, "facts": facts, "outcome": self._outcome(state.get("previous") or {}, facts, spy)}

    def _outcome(self, previous: Mapping[str, Any], facts: dict[str, dict[str, dict[str, float]]], spy: list[float]) -> dict[str, Any]:
        """Last week's book and views against this week's returns; empty without a previous PM or when it is unreadable."""
        pm = previous.get("pm")
        if not pm:
            return {}
        try:
            weights = {w["symbol"]: float(w["weight"]) for w in pm["weights"]}
            return outcome(weights, previous.get("pods") or {}, facts, week_return_pct(spy))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            self._warn(f"last week's book is unreadable, no outcome given: {type(exc).__name__}: {exc}")
            return {}


class Execute:
    """Node: re-check the PM's book, trade it, save the week and write the run-log record."""

    def __init__(self, store: StateStore, rebalancer: Rebalancer, log: RunLog, params: SectorPodsParams, *, on_warning: Callable[[str], None], on_error: Callable[[str], None]) -> None:
        """Create the node.

        Args:
            store: Where the week is saved.
            rebalancer: Places the orders.
            log: One record is appended per call.
            params: Order limits and cash buffer.
            on_warning: Reports an abandoned week that is not a failure of execution.
            on_error: Reports a refused book or a broker failure.
        """
        self._store = store
        self._rebalancer = rebalancer
        self._log = log
        self._params = params
        self._on_warning = on_warning
        self._on_error = on_error

    def __call__(self, state: PodsState) -> dict[str, Any]:
        """Trade and save; returns `{"orders", "abandoned"}` (the reason, or None when completed)."""
        saved = self._store.load()
        reason = state.get("abandoned")
        orders: list[dict[str, Any]] = []
        if reason is None:
            reason, orders = self._trade(state)
        elif reason:
            self._on_warning(f"week abandoned: {reason}")
        if reason is None:
            week = WeekState(last_completed=state.get("date"), previous={"date": state.get("date"), "pods": state.get("pods"), "risk": state.get("risk"), "pm": state.get("pm")})
        else:
            week = replace(saved, abandoned_streak=saved.abandoned_streak + 1)
        self._store.save(week)
        record = {key: state.get(key) for key in ("date", "regime", "holdings", "facts", "outcome", "pods", "risk", "pm")}
        self._log.append({**record, "orders": orders, "errors": state.get("errors", []), "abandoned": reason, "abandoned_streak": week.abandoned_streak})
        return {"orders": orders, "abandoned": reason}

    def _trade(self, state: PodsState) -> tuple[str | None, list[dict[str, Any]]]:
        """Place the PM's book; returns why the week is abandoned (None when it went out) and the orders actually sent."""
        pm = state.get("pm")
        if not pm or pm.get("unavailable"):
            return "the portfolio manager gave no book", []
        weights: dict[str, float] = {}
        try:
            weights = {entry["symbol"]: float(entry["weight"]) for entry in pm["weights"]}
            problem = check_weights(weights, state, self._params)
            target = target_portfolio(weights, cash_buffer=self._params.cash_buffer) if problem is None else None
        except Exception as exc:  # hostile or malformed book: abandon the week, never crash half way
            problem, target = f"unreadable book ({type(exc).__name__}: {exc})", None
        if target is None:
            self._on_error(f"execute refused the book: {problem}")
            return f"execution: {problem}", []
        try:
            self._rebalancer.rebalance(target)
        except (BrokerError, BacktestError) as exc:
            orders = [asdict(order) for order in self._rebalancer.placed]
            self._on_error(f"rebalance failed after {len(orders)} orders: {exc}")
            return f"execution: {exc}", orders
        orders = [asdict(order) for order in self._rebalancer.placed]
        if not orders and self._trades_due(weights, state.get("holdings") or {}):
            # `Rebalancer.rebalance` logs a refused order and goes on, so a broker that refuses everything returns normally.
            self._on_error("execute: no order was accepted although the book needs trades")
            return "execution: no order was accepted", []
        return None, orders

    def _trades_due(self, weights: Mapping[str, float], holdings: Mapping[str, float]) -> bool:
        """Whether some ETF's target differs from its current weight by more than the rebalance band.

        If so the rebalancer had at least one order to send, so a run with no accepted order means the broker
        refused them all. SHV is left out of both sides: the PM never sets it and `holdings` drops the parking symbol.
        """
        return any(abs(weights.get(symbol, 0.0) - holdings.get(symbol, 0.0)) > self._params.rebalance_band for symbol in {*weights, *holdings})


def build_graph(parts: GraphParts) -> Any:
    """Compile the weekly flow: load_previous, market_facts, the pods in parallel, risk_manager, portfolio_manager, execute."""
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PodsState)
    graph.add_node("load_previous", parts.load_previous)
    graph.add_node("market_facts", parts.market_facts)
    for pod in PODS:
        graph.add_node(pod.key, skip_if_abandoned(skip_pod_without_facts(pod.key, parts.pods[pod.key])))
    graph.add_node("risk_manager", skip_if_abandoned(parts.risk_manager))
    graph.add_node("portfolio_manager", skip_if_abandoned(parts.portfolio_manager))
    graph.add_node("execute", parts.execute)
    graph.add_edge(START, "load_previous")
    graph.add_edge("load_previous", "market_facts")
    for pod in PODS:
        graph.add_edge("market_facts", pod.key)
    graph.add_edge([pod.key for pod in PODS], "risk_manager")
    graph.add_edge("risk_manager", "portfolio_manager")
    graph.add_edge("portfolio_manager", "execute")
    graph.add_edge("execute", END)
    return graph.compile()

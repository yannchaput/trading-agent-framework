"""The sector_pods weekly graph: load last week, market facts, five parallel pods, risk, portfolio manager, execute.

The nodes share one `PodsState`. `SectorPodsReview` holds the code nodes as methods and wires them with the agent nodes
(`AgentNode`s built by the strategy). An abandoned week leaves `market_facts` straight for `execute`.
"""

import operator
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from functools import partial
from typing import Annotated, Any, TypedDict

from trading_agent_framework.agents.graph import AgentNode, merge_dicts
from trading_agent_framework.core import Strategy
from trading_agent_framework.core.regime import REGIME_LABELS
from trading_agent_framework.strategies.common.daily_bars import DailyBars, DataUnavailable
from trading_agent_framework.strategies.common.portfolio import TargetPortfolio, target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.sector_pods.facts import etf_facts, outcome, stress_view, week_return_pct
from trading_agent_framework.strategies.sector_pods.handoff import UNAVAILABLE, check_weights, fill_book, unresearched
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, PODS
from trading_agent_framework.strategies.sector_pods.store import StateStore, WeekState
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError

_SPY = "SPY"


class PodsState(TypedDict, total=False):
    """The shared state of one weekly review.

    `pods` is merged across the parallel pod nodes and `errors` is appended to; every other slot has one writer.
    """

    date: str
    regime: str  # bullish / neutral / bearish / unknown
    stress: dict[str, Any] | None  # `facts.stress_view`; None without enough history
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
    return {key: state.get(key) for key in ("date", "regime", "stress", "holdings", "facts", "pods")} | {"previous": _previous(state, "risk")}


def pm_reads(state: Mapping[str, Any]) -> dict[str, Any]:
    """Slots the portfolio manager sees: everything, plus the held ETFs nobody researched and last week's outcome."""
    keys = ("date", "regime", "stress", "holdings", "facts", "outcome", "pods", "risk")
    return {key: state.get(key) for key in keys} | {"unresearched": unresearched(state), "previous": _previous(state, "pm")}


class SectorPodsReview:
    """The weekly review: a LangGraph graph compiled once, run by `run`."""

    def __init__(
        self,
        *,
        strategy: Strategy,
        params: SectorPodsParams,
        store: StateStore,
        rebalancer: Rebalancer,
        bars: DailyBars,
        log: RunLog,
        pods: Mapping[str, AgentNode],
        risk_manager: AgentNode,
        portfolio_manager: AgentNode,
    ) -> None:
        """Wire the review and compile its graph.

        Args:
            strategy: Gives the clock, the regime and the log the nodes report to.
            params: Order limits and cash buffer.
            store: Where the last completed week and the abandoned streak are kept.
            rebalancer: Reads the holdings and places the orders.
            bars: Daily bars source.
            log: One record is appended per week that reached `execute`.
            pods: The five pod agents by pod key.
            risk_manager: The risk manager agent.
            portfolio_manager: The portfolio manager agent.
        """
        self._strategy = strategy
        self._params = params
        self._store = store
        self._rebalancer = rebalancer
        self._bars = bars
        self._log = log
        self.pods = pods
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self._graph = self._build_graph()

    def run(self) -> dict[str, Any]:
        """Run one weekly review; returns the final state (`orders`, and `abandoned`: the reason, or None when completed)."""
        return self._graph.invoke({})

    def _build_graph(self) -> Any:
        """Compile load_previous, market_facts, the pods in parallel, risk_manager, portfolio_manager, execute."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(PodsState)
        graph.add_node("load_previous", self._load_previous)
        graph.add_node("market_facts", self._market_facts)
        for key in self.pods:
            graph.add_node(key, partial(self._run_pod, key))
        graph.add_node("risk_manager", partial(self._run_agent, self.risk_manager))
        graph.add_node("portfolio_manager", partial(self._run_agent, self.portfolio_manager))
        graph.add_node("execute", self._execute)
        graph.add_edge(START, "load_previous")
        graph.add_edge("load_previous", "market_facts")
        graph.add_conditional_edges("market_facts", self._after_facts)
        graph.add_edge(list(self.pods), "risk_manager")
        graph.add_edge("risk_manager", "portfolio_manager")
        graph.add_edge("portfolio_manager", "execute")
        graph.add_edge("execute", END)
        return graph.compile()

    def _after_facts(self, state: PodsState) -> list[str]:
        """Where `market_facts` leads: the five pods, or straight to `execute` when the week is abandoned."""
        return ["execute"] if state.get("abandoned") else list(self.pods)

    def _load_previous(self, state: PodsState) -> dict[str, Any]:
        """Put last completed week's views in the state."""
        return {"previous": self._store.load().previous}

    def _market_facts(self, state: PodsState) -> dict[str, Any]:
        """Today's date, regime label, holdings, per-ETF facts, the score of last week's views and the market `stress`.

        Abandons the week (`data: ...`) when the bars are unusable or no ETF has enough history.
        """
        strategy = self._strategy
        holdings = self._rebalancer.current_weights()
        today = strategy.clock.now().astimezone(MARKET_TZ).date()
        base: dict[str, Any] = {
            "date": today.isoformat(),
            "regime": "unknown" if strategy.regime is None else REGIME_LABELS[strategy.regime],
            "stress": stress_view(strategy.regime_reading, strategy.market_fall),
            "holdings": holdings,
        }
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
            self._strategy.log_warning(f"last week's book is unreadable, no outcome given: {type(exc).__name__}: {exc}")
            return {}

    def _run_agent(self, node: AgentNode, state: PodsState) -> dict[str, Any]:
        """Run an agent node on the week's state (LangGraph hands nodes the `PodsState`)."""
        return node(state)

    def _run_pod(self, key: str, state: PodsState) -> dict[str, Any]:
        """Run one pod; a pod with no ETF facts is not run and is recorded as unavailable."""
        if not state.get("facts", {}).get(key):
            return {"pods": {key: UNAVAILABLE}}
        return self._run_agent(self.pods[key], state)

    def _execute(self, state: PodsState) -> dict[str, Any]:
        """Re-check the PM's book, fill it to the invested target, trade it, save the week and write the run-log record.

        Returns:
            `{"orders", "abandoned"}`: the orders sent, and why the week is abandoned (None when completed).
        """
        saved = self._store.load()
        abandoned = state.get("abandoned")
        if abandoned is None:
            trade = self._trade(state)
        else:
            trade = {"abandoned": abandoned, "orders": [], "book": None}
            if abandoned:
                self._strategy.log_warning(f"week abandoned: {abandoned}")
        reason = trade["abandoned"]
        if reason is None:
            week = WeekState(last_completed=state.get("date"), previous={"date": state.get("date"), "pods": state.get("pods"), "risk": state.get("risk"), "pm": state.get("pm")})
        else:
            week = replace(saved, abandoned_streak=saved.abandoned_streak + 1)
        self._store.save(week)
        record = {key: state.get(key) for key in ("date", "regime", "stress", "holdings", "facts", "outcome", "pods", "risk", "pm")}
        self._log.append({**record, "book": trade["book"], "orders": trade["orders"], "errors": state.get("errors", []), "abandoned": reason, "abandoned_streak": week.abandoned_streak})
        return {"orders": trade["orders"], "abandoned": reason}

    def _trade(self, state: PodsState) -> dict[str, Any]:
        """Check the PM's book, fill it to the invested target and trade it.

        Returns:
            `{"abandoned", "orders", "book"}`: why the week is abandoned (None when it went out), the orders actually sent, and
            the weights traded (None when nothing was traded because the book was refused).
        """
        pm = state.get("pm")
        if not pm or pm.get("unavailable"):
            return {"abandoned": "the portfolio manager gave no book", "orders": [], "book": None}
        weights: dict[str, float] = {}
        book: dict[str, float] = {}
        try:
            weights = {entry["symbol"]: float(entry["weight"]) for entry in pm["weights"]}
            problem = check_weights(weights, state, self._params)
            if problem is None:
                book = fill_book(weights, state, self._params)
                problem = check_weights(book, state, self._params)
            target = target_portfolio(book, cash_buffer=self._params.cash_buffer) if problem is None else None
        except Exception as exc:  # hostile or malformed book: abandon the week, never crash half way
            problem, target = f"unreadable book ({type(exc).__name__}: {exc})", None
        if target is None:
            self._strategy.log_error(f"execute refused the book: {problem}")
            return {"abandoned": f"execution: {problem}", "orders": [], "book": None}
        self._strategy.log_info(self._filled_line(sum(weights.values()), sum(book.values())))
        return self._send_orders(target, book, state.get("holdings") or {})

    def _send_orders(self, target: TargetPortfolio, book: dict[str, float], holdings: Mapping[str, float]) -> dict[str, Any]:
        """Rebalance to `target` and say how it went; a broker failure or a book nobody accepted abandons the week."""
        strategy = self._strategy
        try:
            self._rebalancer.rebalance(target)
        except (BrokerError, BacktestError) as exc:
            orders = [asdict(order) for order in self._rebalancer.placed]
            strategy.log_error(f"rebalance failed after {len(orders)} orders: {exc}")
            return {"abandoned": f"execution: {exc}", "orders": orders, "book": book}
        orders = [asdict(order) for order in self._rebalancer.placed]
        if not orders and self._trades_due(book, holdings):
            if self._rebalancer.refused:
                # `Rebalancer.rebalance` logs a refused order and goes on, so a broker that refuses everything returns normally.
                strategy.log_error("execute: no order was accepted although the book needs trades")
                return {"abandoned": "execution: no order was accepted", "orders": [], "book": book}
            # Nothing was sent and nothing refused: the rebalancer found nothing it could fund. The week is not a failure.
            strategy.log_warning("execute: the rebalancer sent no order although the book needs trades: nothing could be funded")
        return {"abandoned": None, "orders": orders, "book": book}

    def _filled_line(self, submitted: float, filled: float) -> str:
        """The log line: how far `fill_book` took the PM's total, and when the caps held it under the target."""
        target = 1 - self._params.cash_buffer
        line = f"book filled from {submitted:.1%} to {filled:.1%} invested"
        return line if filled >= target - 1e-9 else f"{line} (the caps hold it below {target:.1%})"

    def _trades_due(self, weights: Mapping[str, float], holdings: Mapping[str, float]) -> bool:
        """Whether some ETF's target differs from its current weight by more than the rebalance band.

        If so the rebalancer had at least one order to send, so a run with no accepted order means the broker
        refused them all. SHV is left out of both sides: the PM never sets it and `holdings` drops the parking symbol.
        """
        return any(abs(weights.get(symbol, 0.0) - holdings.get(symbol, 0.0)) > self._params.rebalance_band for symbol in {*weights, *holdings})

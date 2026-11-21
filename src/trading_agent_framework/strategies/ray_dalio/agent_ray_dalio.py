"""RayDalioStrategy: a three-analyst, disagreement and trader LLM team that holds leveraged ETFs, once a session.

Adapted from LumiBot's idea-meritocracy example (see `prompts.py`). The flow is the LangGraph graph of
`_build_graph`, built once in `initialize`; agents are created once there and the nodes only run them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypedDict, cast

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.common.news import require_news_provider
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.ray_dalio.basket import MIN_STOCK_WEIGHT, basket_tool
from trading_agent_framework.strategies.ray_dalio.prompts import ANALYSTS, CONTEXT, DISAGREEMENT, SYSTEM_PROMPTS, TASK_PROMPTS, UNIVERSE
from trading_agent_framework.utils.helpers import BUY_BUFFER

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.runner import BacktestResult

REBALANCE = "rebalance"  # the plain-code node that trades the basket
_DEBATE = tuple(ANALYSTS.values())  # the state slots the analysts write


def _size(quantity: Decimal | None) -> str:
    """A share count for a log line; `?` for a notional order, which has none."""
    return "?" if quantity is None else f"{quantity.normalize():f}"


@dataclass(frozen=True)
class _RebalanceSettings:
    """The three settings `Rebalancer` reads (pyright refuses a plain namespace for them): sector_pods' band and minimum trade."""

    parking_symbol: str = "SHV"
    rebalance_band: float = MIN_STOCK_WEIGHT
    min_trade_pct: float = 0.005


class DalioState(TypedDict, total=False):
    """The shared state of one session: the date, each analyst's view, the disagreement round and its basket (one writer per slot)."""

    date: str
    growth: str
    inflation: str
    liquidity: str
    disagreement: str
    basket: dict[str, float]


class RayDalioStrategy(Strategy):
    """Once a session: growth, inflation and debt/liquidity analysts in parallel, a disagreement round, then code trades its basket."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.BI_YEAR_LATEST)[0],
        "backtesting_end": backtest_window(PredefinedWindow.BI_YEAR_LATEST)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 10,
        "budget": 10000,
    }

    _graph: Any = None

    def initialize(self) -> None:
        """Create the four agents (once per run: names are unique), the rebalancer, and compile the graph."""
        self.sleeptime = "1D"
        require_news_provider(self.broker, "ray_dalio's agents need")
        self._basket: dict[str, float] = {}  # filled by the disagreement agent's submit_basket tool
        for name in ANALYSTS:
            self.agents.create(name=name, system_prompt=SYSTEM_PROMPTS[name], tools=self._research_tools())
        self.agents.create(name=DISAGREEMENT, system_prompt=SYSTEM_PROMPTS[DISAGREEMENT], tools=[*self._research_tools(), basket_tool(self._basket, self.log_info)])
        self._rebalancer = Rebalancer(self, _RebalanceSettings())
        self._graph = self._build_graph()

    def _research_tools(self) -> list[Callable[..., dict[str, Any]]]:
        """Read-only tools, new instances on every call (the news tool keeps a per-instance run memo)."""
        return [*macro_tools(self), *news_tools(self), *fundamentals_tools(self), *market_data_tools(self), *account_tools(self), *indicator_tools(self)]

    def _node(self, name: str, slot: str | None, upstream: tuple[str, ...]):
        """A graph node that runs agent `name` on the date, the fixed context and the `upstream` slots and writes `slot`."""

        def run(state: DalioState) -> dict[str, Any]:
            values = cast(Mapping[str, Any], state)
            context = {"date": values["date"], **CONTEXT, **{key: values[key] for key in upstream}}
            result = self.agents[name].run(TASK_PROMPTS[name], context=context)
            self.log_info(f"[{name}] {result.output}")
            return {} if slot is None else {slot: result.output}

        return run

    def _disagreement_node(self, state: DalioState) -> dict[str, Any]:
        """Run the disagreement agent; its accepted basket (empty if it never submitted one) goes to the state."""
        self._basket.clear()  # a basket never carries over from an earlier session
        update = self._node(DISAGREEMENT, "disagreement", _DEBATE)(state)
        return {**update, "basket": dict(self._basket)}

    def _rebalance_node(self, state: DalioState) -> dict[str, Any]:
        """Trade the account to the basket: holdings it leaves out are sold, the rest of the book goes to SHV. Plain code, no agent."""
        basket = state.get("basket", {})
        if not basket:
            self.log_warning("no basket submitted: holding the current book")
            return {}
        stocks = {symbol: weight for symbol, weight in basket.items() if symbol != "SHV"}  # SHV is whatever the stocks leave over
        self._rebalancer.rebalance(target_portfolio(stocks, cash_buffer=float(BUY_BUFFER)))
        return {}

    def _build_graph(self) -> Any:
        """Compile START -> three analysts in parallel -> disagreement -> rebalance -> END."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(DalioState)
        for name, slot in ANALYSTS.items():  # node names are the agent names: a node may not share a name with a state slot
            graph.add_node(name, self._node(name, slot, ()))
            graph.add_edge(START, name)
        graph.add_node(DISAGREEMENT, self._disagreement_node)
        graph.add_node(REBALANCE, self._rebalance_node)
        graph.add_edge(list(ANALYSTS), DISAGREEMENT)
        graph.add_edge(DISAGREEMENT, REBALANCE)
        graph.add_edge(REBALANCE, END)
        return graph.compile()

    def on_trading_iteration(self) -> None:
        """Run the graph; an agent error goes to the executor, which logs it and runs again next session."""
        self._graph.invoke({"date": self.get_datetime().date().isoformat()})

    def run_backtesting(self, **overrides: Any) -> BacktestResult:
        """Backtest on Yahoo daily bars, the universe and SPY preloaded."""
        defaults: dict[str, Any] = dict(preload_assets=[Asset(symbol=symbol) for symbol in (*UNIVERSE, "SPY")], timestep="day", agent_telemetry=True)
        return super().run_backtesting(**{**defaults, **overrides})

    # --- order events: logged here so a paper/live run shows what the rebalance did ---------------------------

    def on_new_order(self, order: Order) -> None:
        """Log a newly accepted order.

        Args:
            order: The order the broker accepted.
        """
        self.log_info(f"new order: {order.side.value} {_size(order.quantity)} {order.asset.symbol} ({order.order_type.value}, id {order.identifier})")

    def on_partially_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        """Log a partial fill.

        Args:
            position: The position after the fill, or None if there is none.
            order: The order that was partially filled.
            price: Price of this fill.
            quantity: Quantity of this fill.
            multiplier: Contract multiplier (1 for stocks).
        """
        self.log_info(f"order partially filled: {order.side.value} {_size(quantity)} of {_size(order.quantity)} {order.asset.symbol} @ ${price:.2f} (id {order.identifier})")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        """Log a complete fill.

        Args:
            position: The position after the fill, or None if there is none.
            order: The order that was filled.
            price: Price of the last fill.
            quantity: Quantity of the last fill.
            multiplier: Contract multiplier (1 for stocks).
        """
        self.log_info(f"order filled: {order.side.value} {_size(quantity)} of {_size(order.quantity)} {order.asset.symbol} @ ${price:.2f} (id {order.identifier})")

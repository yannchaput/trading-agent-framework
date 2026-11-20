"""RayDalioStrategy: a three-analyst, disagreement and trader LLM team that holds leveraged ETFs, once a session.

Adapted from LumiBot's idea-meritocracy example (see `prompts.py`). The flow is the LangGraph graph of
`_build_graph`, built once in `initialize`; agents are created once there and the nodes only run them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypedDict, cast

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.agents.tools.trading import trading_tools
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.common.news import require_news_provider
from trading_agent_framework.strategies.ray_dalio.prompts import ANALYSTS, CONTEXT, DISAGREEMENT, SYSTEM_PROMPTS, TASK_PROMPTS, TRADER, UNIVERSE
from trading_agent_framework.strategies.ray_dalio.readiness import require_readiness_before_orders

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.runner import BacktestResult

_DEBATE = tuple(ANALYSTS.values())  # the state slots the analysts write


class DalioState(TypedDict, total=False):
    """The shared state of one session: the date, each analyst's view and the disagreement round (one writer per slot)."""

    date: str
    growth: str
    inflation: str
    liquidity: str
    disagreement: str


class RayDalioStrategy(Strategy):
    """Once a session: growth, inflation and debt/liquidity analysts in parallel, a disagreement round, then the trader."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.MONTH_LATEST)[0],
        "backtesting_end": backtest_window(PredefinedWindow.MONTH_LATEST)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 10,
        "budget": 10000,
    }

    _graph: Any = None

    def initialize(self) -> None:
        """Create the five agents (once per run: names are unique) and compile the graph."""
        self.sleeptime = "1D"
        require_news_provider(self.broker, "ray_dalio's agents need")
        for name in (*ANALYSTS, DISAGREEMENT):
            self.agents.create(name=name, system_prompt=SYSTEM_PROMPTS[name], tools=self._research_tools())
        trader_tools = require_readiness_before_orders([*self._research_tools(), *trading_tools(self)])
        self.agents.create(name=TRADER, system_prompt=SYSTEM_PROMPTS[TRADER], tools=trader_tools)
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

    def _build_graph(self) -> Any:
        """Compile START -> three analysts in parallel -> disagreement -> trader -> END."""
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(DalioState)
        for name, slot in ANALYSTS.items():  # node names are the agent names: a node may not share a name with a state slot
            graph.add_node(name, self._node(name, slot, ()))
            graph.add_edge(START, name)
        graph.add_node(DISAGREEMENT, self._node(DISAGREEMENT, "disagreement", _DEBATE))
        graph.add_node(TRADER, self._node(TRADER, None, (*_DEBATE, "disagreement")))
        graph.add_edge(list(ANALYSTS), DISAGREEMENT)
        graph.add_edge(DISAGREEMENT, TRADER)
        graph.add_edge(TRADER, END)
        return graph.compile()

    def on_trading_iteration(self) -> None:
        """Run the graph; an agent error goes to the executor, which logs it and runs again next session."""
        self._graph.invoke({"date": self.get_datetime().date().isoformat()})

    def run_backtesting(self, **overrides: Any) -> BacktestResult:
        """Backtest on Yahoo daily bars, the universe and SPY preloaded."""
        defaults: dict[str, Any] = dict(preload_assets=[Asset(symbol=symbol) for symbol in (*UNIVERSE, "SPY")], timestep="day", agent_telemetry=True)
        return super().run_backtesting(**{**defaults, **overrides})

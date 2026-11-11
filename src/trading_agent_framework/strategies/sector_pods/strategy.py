"""SectorPodsStrategy: five sector pods, a risk manager and a portfolio manager rebalance 27 sector ETFs once a week.

The weekly flow is the LangGraph graph of `graph.py`, built once in `initialize`. One review a week, on Tuesday at 12:00 ET.
"""

from __future__ import annotations

import calendar
import dataclasses
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.graph import AgentNode, SubmitSlot
from trading_agent_framework.agents.tools import only
from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.memory.tools import memory_tools
from trading_agent_framework.strategies.common.daily_bars import DailyBars, GateBars, YahooBars
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.strategies.common.yahoo_daily_bars import BarsSource, YahooDailyBars
from trading_agent_framework.strategies.sector_pods.graph import Execute, GraphParts, LoadPrevious, MarketFacts, build_graph, pm_reads, pod_reads, risk_reads
from trading_agent_framework.strategies.sector_pods.handoff import (
    PM_SUBMIT,
    POD_SUBMIT,
    RISK_SUBMIT,
    UNAVAILABLE,
    pm_submit_tool,
    pod_submit_tool,
    risk_submit_tool,
    validate_pm,
    validate_pod,
    validate_risk,
)
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, PODS
from trading_agent_framework.strategies.sector_pods.prompts import PM_TASK, POD_TASK, RISK_SYSTEM, RISK_TASK, pm_system, pod_system
from trading_agent_framework.strategies.sector_pods.store import StateStore
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BacktestError, BrokerError, ConfigurationError, FatalStrategyError

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.runner import BacktestResult

_PM_MEMORY_TOOLS = {"search_memory", "remember_lesson"}
_SUBMIT_TOOLS = {POD_SUBMIT: pod_submit_tool, RISK_SUBMIT: risk_submit_tool, PM_SUBMIT: pm_submit_tool}


def _size(quantity: Decimal | None) -> str:
    """A share count for a log line; `?` for a notional order, which has none."""
    return "?" if quantity is None else f"{quantity.normalize():f}"


class SectorPodsStrategy(Strategy):
    """One review a week (see `graph.py`); every other session's iteration returns at once."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.MONTH_LATEST)[0],
        "backtesting_end": backtest_window(PredefinedWindow.MONTH_LATEST)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 300,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        settings: SectorPodsParams | None = None,
        bars_source: BarsSource | None = None,
        fred_client_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the strategy.

        Args:
            broker: Broker the strategy trades through.
            mode: Trading mode.
            settings: Strategy parameters; the defaults when omitted.
            bars_source: Daily-bars source for paper/live (injected in tests); backtests read the data source.
            fred_client_factory: Builds the FRED client of the macro tool (injected in tests); the real client when omitted.
            **kwargs: Passed on to `Strategy`.
        """
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or SectorPodsParams()
        # Recorded in settings.json, so runs with other caps or budgets can be told apart.
        self.parameters = {
            **self.parameters,
            "max_etf_weight": self.settings.max_etf_weight,
            "max_pod_weight": self.settings.max_pod_weight,
            "thinking_token_budget": self.settings.thinking_token_budget,
            "max_output_tokens": self.settings.max_output_tokens,
        }
        self.iteration_start_time = parse_rebalance_time(self.settings.rebalance_time)
        if bars_source is None and self.trading_mode is not TradingMode.BACKTESTING:
            bars_source = YahooDailyBars()
        self._bars_source = bars_source
        self._fred_client_factory = fred_client_factory
        self._store: StateStore | None = None
        self._graph: Any = None

    def initialize(self) -> None:
        """Build the seven agents and the weekly graph (once per run)."""
        # fresh in a backtest: one run's streak and date must not leak into the next; paper and live state is never wiped
        self._store = StateStore(self.state_file("sector_pods"), log=self._log, fresh=self.is_backtesting)
        try:
            self._require_news_provider()
            parts = self._graph_parts(self._store)
        except ConfigurationError as exc:
            raise FatalStrategyError(str(exc)) from exc
        self._graph = build_graph(parts)
        self.log_info(f"SectorPodsStrategy initialized: {len(ALL_SYMBOLS)} ETFs in {len(PODS)} pods, review on weekday {self.settings.rebalance_weekday} at {self.iteration_start_time}")

    def _require_news_provider(self) -> None:
        """Refuse to start without a news source: every pod's `search_news` would only return `{"error": ...}`."""
        try:
            provider = self.broker.news_provider()
        except BrokerError as exc:
            raise ConfigurationError(f"the pods need a news provider: {exc}") from exc
        if provider is None:
            raise ConfigurationError("the pods need a news provider and this broker has none")

    def _node(self, name: str, system: str, tools: list[Callable[..., Any]], slot: SubmitSlot, submit: str, task: str, reads: Any, write: Any, budget: int | None) -> AgentNode:
        """Create one agent, which ends its run on its recorded `submit` tool, and the graph node that runs it."""
        params = self.settings
        agent = self.agents.create(
            name=name,
            system_prompt=system,
            tools=[*tools, _SUBMIT_TOOLS[submit](slot)],
            temperature=params.agent_temperature,
            exempt_tools=[submit],
            final_tools=[submit],
            max_output_tokens=params.max_output_tokens,
            thinking_token_budget=params.thinking_token_budget,
        )
        return AgentNode(name=name, agent=agent, slot=slot, submit_tool=submit, task=task, reads=reads, write=write, tool_budget=budget, max_steps=params.max_agent_steps)

    def _graph_parts(self, store: StateStore) -> GraphParts:
        """Create the agents and wire them, with the code nodes, into the graph's parts."""
        params = self.settings
        pods = {
            pod.key: self._node(
                pod.key,
                pod_system(pod, params),
                # a new tool set per pod: they hold per-run memo state behind a lock, so sharing would serialize the parallel pods
                [*news_tools(self), *macro_tools(self, fred_client_factory=self._fred_client_factory)],
                SubmitSlot(validate_pod(pod, params)),
                POD_SUBMIT,
                POD_TASK,
                pod_reads(pod.key),
                lambda view, key=pod.key: {"pods": {key: UNAVAILABLE if view is None else view}},
                params.pod_tool_budget,
            )
            for pod in PODS
        }
        risk = self._node("risk_manager", RISK_SYSTEM, [], SubmitSlot(validate_risk(params)), RISK_SUBMIT, RISK_TASK, risk_reads, lambda view: {"risk": UNAVAILABLE if view is None else view}, None)
        memory = only(memory_tools(self.memory), _PM_MEMORY_TOOLS)
        pm = self._node(
            "portfolio_manager",
            pm_system(params),
            memory,
            SubmitSlot(validate_pm(params)),
            PM_SUBMIT,
            PM_TASK,
            pm_reads,
            lambda view: {"pm": UNAVAILABLE if view is None else view},
            params.pm_memory_budget,
        )
        rebalancer = Rebalancer(self, params)
        return GraphParts(
            load_previous=LoadPrevious(store),
            market_facts=MarketFacts(self._daily_bars(), regime=lambda: self.regime, holdings=rebalancer.current_weights, today=self._market_date, on_warning=self._warn),
            pods=pods,
            risk_manager=risk,
            portfolio_manager=pm,
            execute=Execute(store, rebalancer, RunLog(self.run_file("weeks.jsonl"), what="week log"), params, on_warning=self._warn, on_error=self._error),
        )

    def on_trading_iteration(self) -> None:
        """The weekly review. A backtest aborts when too many reviews in a row were abandoned (a dead LLM or data source)."""
        if self._graph is None or self._store is None:
            raise FatalStrategyError("SectorPodsStrategy.on_trading_iteration ran before initialize() built the graph")
        today = self._market_date()
        self.log_info(f"Today is a {today.strftime('%A')} and the rebalance day is {calendar.day_name[self.settings.rebalance_weekday]}")
        if today.weekday() != self.settings.rebalance_weekday:
            self.log_info("Not a rebalance day — skipping.")
            return
        if self._store.load().last_completed == today.isoformat():
            self.log_info(f"[sector_pods] the week of {today} already completed: not run again")
            return
        try:
            abandoned = self._graph.invoke({"date": today.isoformat()}).get("abandoned") is not None
            streak = self._store.load().abandoned_streak
        except (BrokerError, BacktestError) as exc:
            # Escaped the graph (holdings read, rebalance): an abandoned week, but last week's views stay.
            self.log_error(f"[sector_pods] the week of {today} failed outside the graph: {exc}")
            saved = self._store.load()
            abandoned, streak = True, saved.abandoned_streak + 1
            self._store.save(dataclasses.replace(saved, abandoned_streak=streak))
        if abandoned and self.is_backtesting and streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

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

    def _warn(self, message: str) -> None:
        self.log_warning(message)

    def _error(self, message: str) -> None:
        self.log_error(message)

    def _market_date(self) -> date:
        return self.clock.now().astimezone(MARKET_TZ).date()

    def _daily_bars(self) -> DailyBars:
        if self._bars_source is None:
            return GateBars(lambda symbol, length: self.get_historical_prices(symbol, length, "day"), today=self._market_date, warn=self._warn)
        return YahooBars(
            self._bars_source,
            today=self._market_date,
            sleep=self.sleep,
            warn=self._warn,
            retry_delays=self.settings.yahoo_retry_delays,
            min_coverage=self.settings.min_yahoo_coverage,
        )

    def run_backtesting(self, **overrides: Any) -> BacktestResult:
        """Backtest on the framework's Yahoo daily bars, the 27 ETFs plus SHV and SPY preloaded."""
        defaults: dict[str, Any] = dict(
            preload_assets=[Asset(symbol=symbol) for symbol in (*ALL_SYMBOLS, "SHV", "SPY")],
            timestep="day",
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})

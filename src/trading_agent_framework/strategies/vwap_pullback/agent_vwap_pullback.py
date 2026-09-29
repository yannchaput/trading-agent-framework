"""VwapPullbackStrategy: an intraday VWAP pullback continuation strategy with an entry agent and an exit agent.

Every tick (5 minutes) runs one LangGraph pass (`graph.py`): the Python classifier (`Scanner` + `Desk.reconcile`)
always, then the exit agent only when an open trade had an event, then the entry agent only when a setup
triggered and a slot is free. Code owns the safety net (`Desk`): sizing, the protective stop on every fill,
the loss limit, the entry window and the 15:50 flatten. See docs/superpowers/specs/2026-09-29-vwap-pullback-continuation-design.md.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.graph import TickState, build_tick_graph
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.prompts import build_entry_prompt, build_exit_prompt
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState
from trading_agent_framework.strategies.vwap_pullback.tools import entry_tools, exit_tools, setup_rows, trade_rows
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, FatalStrategyError

# A persistent LLM misconfiguration would otherwise give a flat "successful" backtest (same rule as news_binary).
MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS = 3


class VwapPullbackStrategy(Strategy):
    ENTRY_AGENT = "vwap_entry"
    EXIT_AGENT = "vwap_exit"
    ENTRY_TASK = "Review the triggered pullback setups and enter or pass on each one. The current datetime and the setups are in the context below."
    EXIT_TASK = "Review the open trades and manage each one. The current datetime and the trades are in the context below."

    sleeptime = "5M"
    minutes_before_closing = 10  # before_market_closes, and so the flatten, runs at 15:50

    parameters = {
        "backtesting_start": datetime(2026, 8, 3, tzinfo=MARKET_TZ),
        "backtesting_end": datetime(2026, 8, 31, tzinfo=MARKET_TZ),
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 75,  # 70 daily bars for stage 1, plus the RVOL baseline sessions
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: VwapPullbackParameters | None = None,
        chat_model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or VwapPullbackParameters()
        self._chat_model = chat_model  # None: the model named by LLM_MODEL (tests inject a fake)
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None
        self._graph: Any = None

    # --- lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        self.vars.session = None
        self.vars.consecutive_agent_errors = 0
        if self.is_backtesting:
            # A backtest clock jumps a whole 5-minute tick at once: an entry filled on the tick's first bar would get
            # its stop (placed by the fill hook) only at the tick's end, with the bars in between never checked
            # against it. One-minute slices let the executor dispatch fills, and so place stops, bar by bar.
            # Set on this run's clock instance only: other strategies' clocks keep the class's infinite slice.
            self.clock.max_wait_slice = 60.0
        self._build_components()
        assert self.desk is not None
        flatten_time = (datetime(2000, 1, 1, 16, 0) - timedelta(minutes=self.minutes_before_closing)).strftime("%H:%M")
        self.agents.create(name=self.ENTRY_AGENT, system_prompt=build_entry_prompt(self.settings), model=self._chat_model, tools=entry_tools(self, self.desk))
        self.agents.create(name=self.EXIT_AGENT, system_prompt=build_exit_prompt(self.settings, flatten_time=flatten_time), model=self._chat_model, tools=exit_tools(self, self.desk))
        self._graph = build_tick_graph(classify=self._classify_node, run_exit=self._exit_node, run_entry=self._entry_node)
        self.log_info(f"VwapPullbackStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def _build_components(self) -> None:
        self.desk = Desk(self, self.settings, trade_log=self._trade_log_path)
        self.scanner = Scanner(
            self, self.settings, self.universe, benchmark=self.parameters["benchmark_symbol"], preload=self._preload if self.is_backtesting else None
        )

    def before_market_opens(self) -> None:
        self._ensure_session()

    def on_trading_iteration(self) -> None:
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the last minute bar be published
        state = self._ensure_session()
        if state is None or state.flattened:
            return
        assert self.desk is not None
        if not state.unknown_positions_checked:
            self.desk.close_unknown_positions()
        self._graph.invoke({"now": self.get_datetime()})

    def before_market_closes(self) -> None:
        if self.desk is not None:
            self.desk.flatten_all("end-of-day flatten")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        if self.desk is not None:
            self.desk.on_order_filled(order, price, quantity)

    def on_canceled_order(self, order: Order) -> None:
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    def _ensure_session(self) -> SessionState | None:
        """This session's state, prepared on first need: `before_market_opens` does not run when a live run starts mid-session."""
        today = self.get_datetime().astimezone(MARKET_TZ).date()
        state = self.vars.session
        if state is None or state.day != today:
            assert self.scanner is not None
            try:
                self.vars.session = self.scanner.prepare_session()
            except (BrokerError, BacktestError) as exc:
                self.log_error(f"session preparation failed, skipping this tick: {exc}")
                self.vars.session = None
        return self.vars.session

    # --- graph nodes -------------------------------------------------------------------

    def _classify_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None and self.scanner is not None
        session = self.vars.session
        now = state["now"]
        self.desk.reconcile(now)
        self.scanner.scan(session)
        session.decided.clear()
        return {"exit_due": self.desk.exit_review_due(now), "entry_due": self.desk.entry_due()}

    def _exit_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None
        now = state["now"]
        context = {"current_datetime": now.isoformat(), "open_trades": trade_rows(self.desk, now)}
        summary = self._run_agent(self.EXIT_AGENT, self.EXIT_TASK, context)
        if summary["ok"]:  # a failed run reviewed nothing: the same triggers must bring the trades back next tick
            self.desk.mark_reviewed(now)
        return {"runs": [summary], "entry_due": self.desk.entry_due()}

    def _entry_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None
        now = state["now"]
        context = {"current_datetime": now.isoformat(), "setups": setup_rows(self.desk), "free_slots": self.desk.free_slots()}
        summary = self._run_agent(self.ENTRY_AGENT, self.ENTRY_TASK, context)
        session = self.vars.session
        undecided = [s for s, setup in session.setups.items() if setup.state is SetupState.TRIGGERED and s not in session.decided]
        if undecided:
            self.log_warning(f"[{self.ENTRY_AGENT}] no enter_long or pass_on_setup for {', '.join(undecided)}; treated as a pass")
        return {"runs": [summary]}

    def _run_agent(self, name: str, task: str, context: dict[str, Any]) -> dict[str, Any]:
        """Run one agent; an `AgentError` is logged and counted, and aborts a backtest after 3 in a row."""
        try:
            result: AgentRunResult = self.agents[name].run(task, context=context, run_id=uuid.uuid4().hex)
        except AgentError as exc:
            self.vars.consecutive_agent_errors += 1
            self.log_error(f"[{name}] run failed: {exc}")
            if self.is_backtesting and self.vars.consecutive_agent_errors >= MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS:
                raise FatalStrategyError(f"[{name}] failed {self.vars.consecutive_agent_errors} runs in a row, aborting the backtest; last error: {exc}") from exc
            return {"agent": name, "ok": False, "tool_calls": 0}
        self.vars.consecutive_agent_errors = 0
        self.log_info(f"[{name}] output:\n{result.output}")
        for i, call in enumerate(result.tool_calls):
            self.log_info(f"[{name}] tool_call_{i}: {call}")
        return {"agent": name, "ok": True, "tool_calls": len(result.tool_calls)}

    # --- backtesting ---------------------------------------------------------------------

    def _trade_log_path(self) -> Path | None:
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "trades.jsonl"

    def _preload(self, assets: Sequence[Asset], timestep: str) -> None:
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.warmup import warmup_calendar_days

        if not isinstance(self.broker, BacktestBroker):
            return
        start = self.parameters["backtesting_start"] - timedelta(days=warmup_calendar_days(self.parameters["warmup_trading_days"]))
        self.broker.preload_bars(assets, start, self.parameters["backtesting_end"], timestep)

    def run_backtesting(self):
        # class parameters: the same window the data source is built with, so preload_bars matches it
        return super().run_backtesting(
            data_source=AlpacaBacktestData,  # minute bars with enough history (Yahoo keeps ~30 days of minutes)
            timestep="minute",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )

"""EarningsDriftStrategy: post-earnings announcement drift, decided by an agent that trades through a guarded desk.

Once per session, right after the close (`after_market_closes`): `Scanner.prepare` builds today's candidates
(SEC 8-K item 2.02, Benzinga surprise, reaction-day gates), `Desk.reconcile` applies the guardrails, then the agent
decides (or, in baseline mode, code buys every candidate). Orders fill at the next open. `on_trading_iteration`
only raises a fatal error recorded by the previous cycle: the executor swallows exceptions from every other hook.
See docs/superpowers/specs/2026-10-05-earnings-drift-design.md.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog, StateStore, state_path
from trading_agent_framework.strategies.earnings_drift.desk import Desk
from trading_agent_framework.strategies.earnings_drift.event_source import EventProvider, EventSource
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.prompts import DRIFT_SYSTEM, TASK_PROMPT
from trading_agent_framework.strategies.earnings_drift.scanner import Scanner, ScanResult
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS, desk_tools, research_tools
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, ConfigurationError, FatalStrategyError


class EarningsDriftStrategy(Strategy):
    sleeptime = "1D"  # on_trading_iteration only raises a pending fatal error; the work is in after_market_closes
    minutes_after_closing = 0  # at the exact close: a backtest order then fills at the next open, not a session later
    AGENT_NAME = "drift"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.YEAR)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 283,  # the regime minimum; also covers the 60-session run-up and the 20-session baselines
        "budget": 10000,
        "slippage": Decimal("0.0005"),
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: DriftParams | None = None,
        event_source: EventProvider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or DriftParams()
        self._event_source = event_source  # injected in tests; the SEC source is built in `initialize`
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None
        self._state = DriftState()  # replaced by the loaded state in `initialize`
        self._pending_fatal: str | None = None  # raised by the next on_trading_iteration (backtests only)

    # --- lifecycle --------------------------------------------------------------------------------

    def initialize(self) -> None:
        store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            store.wipe()  # one run's trades must not leak into the next; paper and live state is never wiped
        self._state = store.load()
        state = self._state
        try:
            source = self._event_source if self._event_source is not None else EventSource(_sec_client(self.project_root), reload_every_cycle=not self.is_backtesting)
            self.desk = Desk(
                self,
                self.settings,
                state,
                save=lambda: store.save(state),
                trade_log=JsonlLog(lambda: self._run_file("trades.jsonl")),
                decision_log=JsonlLog(lambda: self._run_file("decisions.jsonl")),
            )
            if self.settings.agent_enabled:
                self.agents.create(
                    name=self.AGENT_NAME,
                    system_prompt=DRIFT_SYSTEM,
                    tools=[*desk_tools(self.desk), *research_tools(self)],
                    temperature=self.settings.agent_temperature,
                    exempt_tools=list(ORDER_TOOLS),
                )
            else:
                self.log_warning("guardrail baseline: agent disabled, baseline mode (every gated candidate is bought at the default trail)")
        except ConfigurationError as exc:
            raise FatalStrategyError(str(exc)) from exc  # refuse to start rather than fail every cycle
        self.scanner = Scanner(self, self.settings, self.universe, source, benchmark=self.parameters["benchmark_symbol"])
        self.log_info(f"EarningsDriftStrategy initialized: {len(self.universe)} symbols, agent {'on' if self.settings.agent_enabled else 'off (baseline)'}")

    def on_trading_iteration(self) -> None:
        if self._pending_fatal is not None:
            raise FatalStrategyError(self._pending_fatal)

    def after_market_closes(self) -> None:
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the day's bar become final
        self.run_cycle()

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        if self.desk is not None:
            self.desk.on_order_filled(order)

    def on_canceled_order(self, order: Order) -> None:
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    # --- the daily cycle --------------------------------------------------------------------------

    def run_cycle(self) -> None:
        assert self.desk is not None and self.scanner is not None, "initialize() has not run"
        desk, settings, state = self.desk, self.settings, self._state
        try:
            scan = self.scanner.prepare(held=desk.exposed_symbols())
        except Exception as exc:  # the guardrails below never depend on the scan: carry on with an empty one
            self.log_error(f"[earnings_drift] scan failed ({type(exc).__name__}: {exc}); no candidates this cycle, the guardrails still run")
            scan = ScanResult(today=self.get_datetime().astimezone(MARKET_TZ).date(), trading_dates=[])
        if scan.hollow:
            state.hollow_scan_streak += 1
            if self.is_backtesting and state.hollow_scan_streak >= settings.max_consecutive_hollow_scans:
                self._pending_fatal = f"{state.hollow_scan_streak} hollow scans in a row (SEC unreachable?), aborting the backtest"
        else:
            state.hollow_scan_streak = 0
        try:
            desk.begin_session(scan.today, scan.trading_dates, scan.candidates, scan.rejections)
            desk.reconcile()
            if settings.agent_enabled:
                self._run_agent(scan)
            else:
                desk.baseline_entries()
        finally:
            # Whatever failed above, the stops are re-checked, the unanswered candidates are logged and the state is saved.
            desk.ensure_stops()
            desk.record_undecided()
            desk.save()

    def _run_agent(self, scan: ScanResult) -> None:
        assert self.desk is not None
        desk, state = self.desk, self._state
        candidates, holdings = desk.candidate_sheets(), desk.holdings_context()
        if not candidates and not holdings:
            return
        context = {"date": scan.today.isoformat(), "candidates": candidates, "holdings": holdings, "balances": desk.balances()}
        budget = self.settings.tool_budget_per_item * (len(candidates) + len(holdings))
        try:
            result = self.agents[self.AGENT_NAME].run(TASK_PROMPT, context=context, tool_budget=budget)
        except AgentError as exc:
            state.agent_failure_streak += 1
            self.log_error(f"[earnings_drift] agent run failed ({state.agent_failure_streak} in a row); today's candidates are dropped: {exc}")
            if self.is_backtesting and state.agent_failure_streak >= self.settings.max_consecutive_agent_failures:
                self._pending_fatal = f"{state.agent_failure_streak} agent runs failed in a row, aborting the backtest; last error: {exc}"
            return
        except Exception as exc:  # not an agent failure (it does not count toward the abort): log it and let the tail run
            self.log_error(f"[earnings_drift] agent run raised {type(exc).__name__}: {exc}; today's candidates are dropped")
            return
        state.agent_failure_streak = 0
        self.log_info(f"[earnings_drift] agent: {len(candidates)} candidates, {len(holdings)} holdings, {len(result.tool_calls)} tool calls")

    def _run_file(self, name: str) -> Path | None:
        """`<name>` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / name

    # --- backtesting --------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the `parameters` window on Alpaca SIP daily bars, the universe preloaded."""
        symbols = list(dict.fromkeys([*self.universe, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=AlpacaBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            slippage=self.parameters["slippage"],
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=self.settings.agent_enabled,
        )
        return super().run_backtesting(**{**defaults, **overrides})


def _sec_client(project_root: Path) -> SecEdgarClient:
    """The SEC client for the event source; `ConfigurationError` without `SEC_EDGAR_USER_AGENT`."""
    return SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), project_root / "cache" / "sec")

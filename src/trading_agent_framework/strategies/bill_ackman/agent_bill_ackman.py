"""BillAckmanStrategy: a researcher, a short seller and a trader over the fundamentals quality screen.

Port of lumibot's Bill Ackman example, hardened for a local model: code screens the universe and builds the fact
sheets, three agents hand structured results to each other through submit tools, and code keeps the state and
places every order (`ReviewPipeline`, `Rebalancer`). No agent has an order tool.
See docs/superpowers/specs/2026-10-02-bill-ackman-strategy-design.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bill_ackman.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewPipeline, ScreenLike
from trading_agent_framework.strategies.bill_ackman.prompts import RESEARCHER_SYSTEM, SHORT_SELLER_SYSTEM, TRADER_SYSTEM
from trading_agent_framework.strategies.bill_ackman.rebalancer import Rebalancer
from trading_agent_framework.strategies.bill_ackman.screen import build_quality_screen
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, StateStore, state_path
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError


class BillAckmanStrategy(Strategy):
    """One review per session: screen, researcher, short seller, trader, rebalancer (see `ReviewPipeline`)."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.BI_MONTH)[0],
        "backtesting_end": backtest_window(PredefinedWindow.BI_MONTH)[1],
        "benchmark_symbol": "SPY",
        # a year of daily bars before the first simulated day, so price_return_12m (TRADING_DAYS_PER_YEAR + 1 closes) exists from day one
        "warmup_trading_days": 260,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: AckmanParams | None = None,
        screen: ScreenLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or AckmanParams()
        self._screen = screen  # injected in tests; the real screen is built in `initialize`
        self.pipeline: ReviewPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the screen, the three agents and the pipeline (once per run)."""
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's counters must not leak into the next; paper and live state is never wiped
        try:
            screen = self._screen if self._screen is not None else build_quality_screen(self.project_root, params=self.settings.screen)
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools

            # No agent gets an order, account, indicator or memory tool: they research and hand over structured results.
            self.agents.create(name="researcher", system_prompt=RESEARCHER_SYSTEM, tools=[*fundamentals_tools(self), *market_data_tools(self), submit["submit_ranking"]])
            self.agents.create(
                name="short_seller",
                system_prompt=SHORT_SELLER_SYSTEM,
                tools=[*fundamentals_tools(self), *news_tools(self), *market_data_tools(self), submit["submit_verdicts"]],
            )
            self.agents.create(name="trader", system_prompt=TRADER_SYSTEM, tools=[submit["submit_portfolio"]])
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every day (SEC_EDGAR_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = ReviewPipeline(
            strategy=self,
            params=self.settings,
            screen=screen,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            review_log=ReviewLog(self._review_log_path()),
            rebalancer=Rebalancer(self, self.settings),
            universe=self.universe,
        )
        self.log_info(f"BillAckmanStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def on_trading_iteration(self) -> None:
        """One review. A backtest aborts when too many reviews in a row were abandoned (a dead LLM or data source)."""
        assert self.pipeline is not None, "initialize() has not run"
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    def _review_log_path(self) -> Path | None:
        """`reviews.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "reviews.jsonl"

    # --- backtesting ------------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window (`PredefinedWindow.BI_MONTH`) on Yahoo daily bars."""
        symbols = list(dict.fromkeys([*self.universe, self.settings.parking_symbol, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=YahooBacktestData,  # years of daily history
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],  # the screen's daily price lookups hit the cache
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})

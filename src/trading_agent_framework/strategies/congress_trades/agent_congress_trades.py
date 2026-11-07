"""CongressTradesStrategy: a researcher, a portfolio manager and a trader that mirror a member's disclosed stock holdings.

Port of lumibot's "Nancy Pelosi trading bot" agent example, rebuilt as a pipeline like `bill_ackman`: code finds the
filings known today (House Clerk disclosures) and decides whether anything is new; three agents hand structured results to
each other through submit tools; only the trading agent has order tools, and those are bounded by the `TradeDesk`.
The strategy ticks every day but acts only when the member has filed something new (see `CongressPipeline`).
See docs/superpowers/specs/2026-10-07-congress-trades-design.md.
"""

from __future__ import annotations

import os
from datetime import time
from typing import Any

from trading_agent_framework.agents.tools import only
from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.trade_desk import TradeDesk
from trading_agent_framework.strategies.congress_trades.congress import ClerkClient, CongressSource
from trading_agent_framework.strategies.congress_trades.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.strategies.congress_trades.pipeline import AGENT_PORTFOLIO, AGENT_RESEARCHER, AGENT_TRADER, CongressPipeline, SourceLike
from trading_agent_framework.strategies.congress_trades.prompts import portfolio_system, researcher_system, trader_system
from trading_agent_framework.strategies.congress_trades.state import StateStore
from trading_agent_framework.strategies.congress_trades.tools import congress_research_tools
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError


class CongressTradesStrategy(Strategy):
    """One check per session at 10:00 ET: new filing? research, portfolio, trading. Otherwise nothing."""

    sleeptime = "1D"  # a check every trading session (`D` counts sessions)
    iteration_start_time = time(10, 0)  # market time: after the open, so the first bar of the day has printed

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.YEAR)[1],
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 10,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        settings: CongressParams | None = None,
        source: SourceLike | None = None,
        **kwargs: Any,
    ) -> None:
        """Create the strategy.

        Args:
            broker: Broker the strategy trades through.
            mode: Trading mode.
            settings: Strategy parameters; the defaults when omitted.
            source: Disclosure source (injected in tests); the House Clerk source is built in `initialize` when omitted.
            **kwargs: Passed on to `Strategy`.
        """
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or CongressParams()
        self._source = source  # injected in tests; the real Clerk source is built in `initialize`
        self.pipeline: CongressPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the source, the three agents, the desk and the pipeline (once per run)."""
        params = self.settings
        state_store = StateStore(self.state_file("congress_trades"), fresh=self.is_backtesting)  # one run's history must not leak into the next; paper and live state is never wiped
        try:
            source = self._source if self._source is not None else self._build_source()
            recorder = HandoffRecorder(params)
            desk = TradeDesk(self, params, log_tag="congress_trades")
            submit = submit_tools(recorder)
            temperature = params.agent_temperature
            # The research agent reads filings; the portfolio agent only submits; only the trading agent can place orders (through the desk).
            self.agents.create(
                name=AGENT_RESEARCHER,
                system_prompt=researcher_system(params.politician),
                tools=[*congress_research_tools(self, source), *only(market_data_tools(self), {"get_last_price"}), submit["submit_holdings"]],
                temperature=temperature,
            )
            self.agents.create(name=AGENT_PORTFOLIO, system_prompt=portfolio_system(params.politician), tools=[submit["submit_target"]], temperature=temperature)
            self.agents.create(
                name=AGENT_TRADER,
                system_prompt=trader_system(),
                tools=[*only(account_tools(self), {"get_account_balance", "get_positions"}), *only(market_data_tools(self), {"get_last_price"}), *desk.tools(), submit["submit_trade_report"]],
                temperature=temperature,
            )
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every day (CONGRESS_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = CongressPipeline(
            strategy=self,
            params=params,
            source=source,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            log=RunLog(self.run_file("runs.jsonl")),
            desk=desk,
        )
        self.log_info(f"CongressTradesStrategy initialized: following {params.politician}, sleeptime {self.sleeptime}, first check at {self.iteration_start_time}")

    def _build_source(self) -> CongressSource:
        """The House Clerk source; `ConfigurationError` without `CONGRESS_USER_AGENT`."""
        client = ClerkClient(os.environ.get("CONGRESS_USER_AGENT", ""), self.project_root / "cache" / "house_clerk")
        return CongressSource(client, self.settings.politician)

    def on_trading_iteration(self) -> None:
        """One daily check. A backtest aborts when too many runs in a row were abandoned (a dead LLM or a broken data source)."""
        assert self.pipeline is not None, "initialize() has not run"
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} runs abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    # --- backtesting -------------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window on Yahoo daily bars.

        The tickers to trade are known only once the filings are read, so only the benchmark is preloaded: Yahoo's source fetches
        any other ticker lazily, once, the first time the desk asks for its price.
        """
        defaults: dict[str, Any] = dict(
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=self.parameters["benchmark_symbol"])],
            timestep="day",
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})

"""BullBearStrategy: a researcher, a bull, a bear and a judge debate the top of cross_momentum's ranking.

Port of lumibot's "Bull vs Bear AI Stock Trading Bot", hardened for a local model: code ranks the universe with
cross_momentum's own score and filters, builds the fact sheets, sizes the judge's picks by inverse volatility and
places every order (`ReviewPipeline`, `Rebalancer`). No agent has an order tool. One review a week, on Tuesday at
12:00 ET, as cross_momentum. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.tools import only
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bull_bear.market_data import BarsSource, DailyBars, GateBars, YahooBars
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewPipeline
from trading_agent_framework.strategies.bull_bear.prompts import BEAR_SYSTEM, BULL_SYSTEM, JUDGE_SYSTEM, RESEARCHER_SYSTEM
from trading_agent_framework.strategies.bull_bear.state import ReviewLog, StateStore, state_path
from trading_agent_framework.strategies.common.rebalancer import Rebalancer
from trading_agent_framework.strategies.common.sector_provider import SectorProvider
from trading_agent_framework.strategies.common.sessions import parse_rebalance_time
from trading_agent_framework.strategies.common.yahoo_daily_bars import YahooDailyBars
from trading_agent_framework.strategies.cross_momentum.parameters import CONFIG
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, FatalStrategyError

if TYPE_CHECKING:
    from trading_agent_framework.backtesting.runner import BacktestResult

_RESEARCH_TOOLS = {"search_news", "get_income_statement", "get_balance_sheet"}


class BullBearStrategy(Strategy):
    """One review a week (see `ReviewPipeline`); every other session's iteration returns at once."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.HALF_YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.HALF_YEAR)[1],
        "benchmark_symbol": "SPY",
        # 300 completed sessions before the first review: the 12-1 month return needs 274 (as cross_momentum)
        "warmup_trading_days": 300,
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: BullBearParams | None = None,
        bars_source: BarsSource | None = None,
        sector_of: Callable[[str], str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or BullBearParams()
        self.iteration_start_time = parse_rebalance_time(self.settings.rebalance_time)
        self.universe = [symbol for symbol in universe if symbol != self.settings.parking_symbol]  # SHV is parking, never a stock
        # Paper/live read Yahoo's daily bars (Alpaca's IEX volume would empty the dollar-volume filter); a backtest reads the gate.
        if bars_source is None and self.trading_mode is not TradingMode.BACKTESTING:
            bars_source = YahooDailyBars()
        self._bars_source = bars_source
        self._sector_of = sector_of  # injected in tests; yfinance's sectors otherwise
        self.pipeline: ReviewPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the four agents and the pipeline (once per run)."""
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's streak and date must not leak into the next; paper and live state is never wiped
        try:
            self._require_news_provider()
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools

            temperature = self.settings.agent_temperature
            research = only([*news_tools(self), *fundamentals_tools(self)], _RESEARCH_TOOLS)
            self.agents.create(name="researcher", system_prompt=RESEARCHER_SYSTEM, tools=[*research, submit["submit_note"]], temperature=temperature, exempt_tools=["submit_note"])
            # The bull, the bear and the judge argue from the same evidence pack: no research tool, only their submit tool.
            self.agents.create(name="bull", system_prompt=BULL_SYSTEM, tools=[submit["submit_bull_case"]], temperature=temperature)
            self.agents.create(name="bear", system_prompt=BEAR_SYSTEM, tools=[submit["submit_bear_case"]], temperature=temperature)
            self.agents.create(name="judge", system_prompt=JUDGE_SYSTEM, tools=[submit["submit_picks"]], temperature=temperature)
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every week (SEC_EDGAR_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = ReviewPipeline(
            strategy=self,
            params=self.settings,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            review_log=ReviewLog(self._review_log_path()),
            rebalancer=Rebalancer(self, self.settings),
            universe=self.universe,
            bars=self._daily_bars(),
            sector_of=self._sector_of or SectorProvider().get_sector,
            momentum=CONFIG,
        )
        self.log_info(f"BullBearStrategy initialized: {len(self.universe)} symbols, review on weekday {self.settings.rebalance_weekday} at {self.iteration_start_time}")

    def _require_news_provider(self) -> None:
        """Refuse to start without a news source: the researcher's `search_news` would only ever return `{"error": ...}`.

        A backtest builds its Alpaca provider from `ALPACA_NEWS_*` on first use, so a missing credential would
        otherwise surface only inside the tool, and every note of the run would be research-free.
        """
        try:
            provider = self.broker.news_provider()
        except BrokerError as exc:
            raise ConfigurationError(f"the researcher needs a news provider: {exc}") from exc
        if provider is None:
            raise ConfigurationError("the researcher needs a news provider and this broker has none")

    def on_trading_iteration(self) -> None:
        """The weekly review. A backtest aborts when too many reviews in a row were abandoned (a dead LLM or data source)."""
        if self.pipeline is None:
            raise FatalStrategyError("BullBearStrategy.on_trading_iteration ran before initialize() built the review pipeline")
        today = self._market_date()
        if today.weekday() != self.settings.rebalance_weekday:
            return
        if self.pipeline.completed_today():
            self.log_info(f"[bull_bear] the review of {today} already completed: not run again")
            return
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    def _market_date(self) -> date:
        return self.clock.now().astimezone(MARKET_TZ).date()

    def _warn(self, message: str) -> None:
        self.log_warning(message)

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

    def _review_log_path(self) -> Path | None:
        """`reviews.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "reviews.jsonl"

    # --- backtesting ------------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any) -> BacktestResult:
        """Backtest over the class `parameters` window on Yahoo daily bars, the universe preloaded."""
        symbols = list(dict.fromkeys([*self.universe, self.settings.parking_symbol, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],  # one Yahoo download, not 1,200 per review
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})

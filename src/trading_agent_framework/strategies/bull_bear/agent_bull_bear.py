"""BullBearStrategy: a researcher, a bull, a bear and a judge debate the top of cross_momentum's ranking.

Port of lumibot's "Bull vs Bear AI Stock Trading Bot", hardened for a local model: code ranks the universe with
cross_momentum's own score and filters, builds the fact sheets, builds the book from the stocks the judge puts in,
sizes it by inverse volatility and places every order (`ReviewPipeline`, `Rebalancer`). No agent has an order tool.
One review a week, on Tuesday at 12:00 ET, as cross_momentum. See docs/superpowers/specs/2026-10-09-bull-bear-strategy-design.md
and docs/superpowers/specs/2026-10-11-bull-bear-debate-realignment-design.md.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
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
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.fundamentals.edgar_client import default_client
from trading_agent_framework.strategies.bull_bear.fundamentals import SecFundamentals
from trading_agent_framework.strategies.bull_bear.handoff import HandoffRecorder, submit_tools
from trading_agent_framework.strategies.bull_bear.market_data import BarsSource, DailyBars, GateBars, YahooBars
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewPipeline
from trading_agent_framework.strategies.bull_bear.prompts import BEAR_SYSTEM, BULL_SYSTEM, JUDGE_SYSTEM, researcher_system
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
_JUDGE_MEMORY_TOOLS = {"search_memory", "remember_lesson"}


def _size(quantity: Decimal | None) -> str:
    """A share count for a log line: `10`, `7.842749`; `?` for a notional order, which has none."""
    return "?" if quantity is None else f"{quantity.normalize():f}"


class BullBearStrategy(Strategy):
    """One review a week (see `ReviewPipeline`); every other session's iteration returns at once."""

    sleeptime = "1D"

    parameters = {
        "backtesting_start": backtest_window(PredefinedWindow.YEAR)[0],
        "backtesting_end": backtest_window(PredefinedWindow.YEAR)[1],
        "benchmark_symbol": "SPY",
        # 300 completed sessions before the first review: the 12-1 month return needs 274 (as cross_momentum)
        "warmup_trading_days": 300,
        "slippage": Decimal("0"),
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
        fundamentals_of: Callable[[str, datetime], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or BullBearParams()
        # Recorded in settings.json, so the news-on and news-off runs, and runs with another thinking budget or output cap, can be told apart.
        self.parameters = {
            **self.parameters,
            "research_news": self.settings.research_news,
            "thinking_token_budget": self.settings.thinking_token_budget,
            "judge_thinking_token_budget": self.settings.judge_thinking_token_budget,
            "max_output_tokens": self.settings.max_output_tokens,
            "judge_max_output_tokens": self.settings.judge_max_output_tokens,
        }
        self.iteration_start_time = parse_rebalance_time(self.settings.rebalance_time)
        self.universe = [symbol for symbol in universe if symbol != self.settings.parking_symbol]  # SHV is parking, never a stock
        # Paper/live read Yahoo's daily bars (Alpaca's IEX volume would empty the dollar-volume filter); a backtest reads the gate.
        if bars_source is None and self.trading_mode is not TradingMode.BACKTESTING:
            bars_source = YahooDailyBars()
        self._bars_source = bars_source
        self._sector_of = sector_of  # injected in tests; yfinance's sectors otherwise
        self._fundamentals_of = fundamentals_of  # injected in tests; the SEC otherwise
        self.pipeline: ReviewPipeline | None = None

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the four agents and the pipeline (once per run)."""
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's streak and date must not leak into the next; paper and live state is never wiped
        try:
            news = self.settings.research_news
            if news:
                self._require_news_provider()
            recorder = HandoffRecorder(self.settings)
            submit = submit_tools(recorder)
            from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
            from trading_agent_framework.agents.tools.news import news_tools
            from trading_agent_framework.memory.tools import memory_tools

            temperature = self.settings.agent_temperature

            # A thinking budget and an output cap per model call: one of each for the researcher, bull and bear, and the judge has its own (`judge_*`).
            # The budget cuts the reasoning so the model still submits; the cap (budget + 8,192) is the backstop.
            # A run it stops, or whose submission is rejected, is not corrected in place: it ends and the pipeline's forced retry is a fresh run (corrective prompt, submit tool forced, thinking on).
            cap = self.settings.max_output_tokens  # the researcher, bull and bear
            budget = self.settings.thinking_token_budget

            # One SEC client (one rate limiter, one cache) serves the researcher's tools and the fact sheet.
            sec_client = default_client(self.project_root)
            sources = [*(news_tools(self) if news else []), *fundamentals_tools(self, client=sec_client)]
            research = only(sources, _RESEARCH_TOOLS if news else _RESEARCH_TOOLS - {"search_news"})
            self.agents.create(
                name="researcher",
                system_prompt=researcher_system(news),
                tools=[*research, submit["submit_note"]],
                temperature=temperature,
                exempt_tools=["submit_note"],
                final_tools=["submit_note"],  # a recorded note ends the run: no closing turn to think the window full
                compact_history=True,  # its tool loop (news, SEC) can grow the history: summarized at 80% of the window
                max_output_tokens=cap,
                thinking_token_budget=budget,
            )
            # The bull and the bear argue from the same evidence pack: no research tool, only their submit tool.
            for name, system_prompt, tool in (("bull", BULL_SYSTEM, "submit_bull_case"), ("bear", BEAR_SYSTEM, "submit_bear_case")):
                self.agents.create(name=name, system_prompt=system_prompt, tools=[submit[tool]], temperature=temperature, final_tools=[tool], max_output_tokens=cap, thinking_token_budget=budget)
            # The judge also keeps lessons between reviews (`strategy.memory`, wiped at the start of every backtest).
            lessons = only(memory_tools(self.memory), _JUDGE_MEMORY_TOOLS)
            self.agents.create(
                name="judge",
                system_prompt=JUDGE_SYSTEM,
                tools=[*lessons, submit["submit_verdicts"]],
                temperature=temperature,
                exempt_tools=["submit_verdicts"],
                final_tools=["submit_verdicts"],
                compact_history=True,
                max_output_tokens=self.settings.judge_max_output_tokens,
                thinking_token_budget=self.settings.judge_thinking_token_budget,
            )
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
            fundamentals_of=self._fundamentals_of or SecFundamentals(sec_client),
            momentum=CONFIG,
        )
        news_state = "on" if self.settings.research_news else "off"
        self.log_info(f"BullBearStrategy initialized: {len(self.universe)} symbols, review on weekday {self.settings.rebalance_weekday} at {self.iteration_start_time}, research news {news_state}")

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
        rebalance_day = calendar.day_name[self.settings.rebalance_weekday]
        self.log_info(f"Today is a {today.strftime('%A')} and the rebalance day is {rebalance_day}")
        if today.weekday() != self.settings.rebalance_weekday:
            self.log_warning("Not a rebalance day — skipping.")
            return
        if self.pipeline.completed_today():
            self.log_info(f"[bull_bear] the review of {today} already completed: not run again")
            return
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} reviews abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    # --- order events: logged here only (the base hooks and the executor stay silent for the other strategies) -------

    def on_new_order(self, order: Order) -> None:
        self.log_info(f"new order: {order.side.value} {_size(order.quantity)} {order.asset.symbol} ({order.order_type.value}, id {order.identifier})")

    def on_partially_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        self.log_info(f"order partially filled: {order.side.value} {_size(quantity)} of {_size(order.quantity)} {order.asset.symbol} @ ${price:.2f} (id {order.identifier})")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        self.log_info(f"order filled: {order.side.value} {_size(quantity)} of {_size(order.quantity)} {order.asset.symbol} @ ${price:.2f} (id {order.identifier})")

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
            slippage=self.parameters["slippage"],
            agent_telemetry=True,
        )
        return super().run_backtesting(**{**defaults, **overrides})

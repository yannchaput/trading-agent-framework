"""InsiderTradesStrategy: a researcher and a trader that follow what the officers and directors of a fixed watchlist do with their stock.

Built as a pipeline like `congress_trades`: code reads the Form 4 filings known today (SEC EDGAR), labels every open-market trade,
classes each stock (buying / selling / none) and decides whether anything changed; a researcher verifies the labels of one stock at
a time, and one trader sets the target weights and then places the orders. Only the two trading agents have order tools, and those
are bounded by the `TradeDesk`. The strategy ticks every day but acts only when a stock's insider filings changed (see `InsiderPipeline`).
See docs/superpowers/specs/2026-10-21-insider-trades-design.md.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
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
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.common.run_log import RunLog
from trading_agent_framework.strategies.common.trade_desk import TradeDesk
from trading_agent_framework.strategies.insider_trades.facts import InsiderFacts
from trading_agent_framework.strategies.insider_trades.form4.source import Form4Source
from trading_agent_framework.strategies.insider_trades.handoff import InsiderRecorder, submit_tools
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.strategies.insider_trades.pipeline import AGENT_RESEARCHER, AGENT_TRADER, AGENT_TRADER_RETRY, InsiderPipeline, SourceLike
from trading_agent_framework.strategies.insider_trades.prompts import RESEARCHER_TASK, TRADER_RETRY_TASK, TRADER_TASK, researcher_system, retry_prompt, trader_retry_system, trader_system
from trading_agent_framework.strategies.insider_trades.state import StateStore, state_path
from trading_agent_framework.strategies.insider_trades.tools import insider_research_tools
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError


class InsiderTradesStrategy(Strategy):
    """One check per session at 10:00 ET: did a watchlist stock's insider filings change? research, target, trade. Otherwise nothing."""

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
        settings: InsiderParams | None = None,
        source: SourceLike | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.settings = settings or InsiderParams()
        self._source = source  # injected in tests; the real EDGAR source is built in `initialize`
        self.pipeline: InsiderPipeline | None = None
        self._backtest_start: datetime | None = None  # set by run_backtesting: where the Form 4 history reaches back from
        self._backtest_end: datetime | None = None  # set by run_backtesting: the SEC cache's freshness time in a backtest

    # --- lifecycle ------------------------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the source, the agents, the desk and the pipeline (once per run). Nothing here reads the filings: the pipeline loads them first."""
        params = self.settings
        state_store = StateStore(state_path(self.project_root, self.trading_mode))
        if self.is_backtesting:
            state_store.wipe()  # one run's history must not leak into the next; paper and live state is never wiped
        try:
            source = self._source if self._source is not None else self._build_source()
            facts = InsiderFacts(self, source, params)
            recorder = InsiderRecorder(params)
            desk = TradeDesk(self, params, log_tag="insider_trades")
            submit = submit_tools(recorder)
            temperature = params.agent_temperature
            trading_tools = [*only(account_tools(self), {"get_account_balance", "get_positions"}), *only(market_data_tools(self), {"get_last_price"}), *desk.tools()]
            # The researcher reads filings and only submits a note; only the two trading agents can place orders (through the desk).
            self.agents.create(
                name=AGENT_RESEARCHER,
                system_prompt=researcher_system(),
                tools=[*insider_research_tools(self, facts, recorder.facts_fetched), submit["submit_note"]],
                temperature=temperature,
            )
            self.agents.create(
                name=AGENT_TRADER,
                system_prompt=trader_system(params),
                tools=[*trading_tools, submit["submit_target"], submit["submit_trade_report"]],
                temperature=temperature,
            )
            self.agents.create(
                name=AGENT_TRADER_RETRY,
                system_prompt=trader_retry_system(),
                tools=[*trading_tools, submit["submit_trade_report"]],
                temperature=temperature,
            )
        except ConfigurationError as exc:
            # The strategy refuses to start rather than fail every day (SEC_EDGAR_USER_AGENT missing, no LLM model).
            raise FatalStrategyError(str(exc)) from exc
        self.pipeline = InsiderPipeline(
            strategy=self,
            params=params,
            source=source,
            facts=facts,
            agents=self.agents,
            recorder=recorder,
            state=state_store,
            run_log=RunLog(self._run_log_path()),
            desk=desk,
            load_window=self.load_window,
            researcher_task=RESEARCHER_TASK,
            trader_task=TRADER_TASK,
            trader_retry_task=TRADER_RETRY_TASK,
            retry_prompt=retry_prompt,
        )
        self.log_info(f"InsiderTradesStrategy initialized: {len(params.watchlist)} stocks, sleeptime {self.sleeptime}, first check at {self.iteration_start_time}")

    def _build_source(self) -> Form4Source:
        """The EDGAR Form 4 source; `ConfigurationError` without `SEC_EDGAR_USER_AGENT`."""
        client = SecEdgarClient(os.environ.get("SEC_EDGAR_USER_AGENT", ""), self.project_root / "cache" / "sec")
        return Form4Source(client, self.settings.watchlist)

    def load_window(self) -> tuple[datetime, date]:
        """`(as_of, since)` for a Form 4 load: the instant the filings must be known by, and the first day of the history to read.

        `since` is January 1 of the year `history_years` before the year of the earliest trade the run can still see: a trade filed
        in early January was made the previous December (a filing lags its trade, and the window reaches back `signal_window_days`),
        and the labeller needs `history_years` full years before THAT trade's year, or its oldest year would sit empty and its
        trades be mislabelled unclassified.
        """
        if self.is_backtesting:
            assert self._backtest_start is not None and self._backtest_end is not None, "run_backtesting has not set the run's window"
            as_of, reference = self._backtest_end, self._backtest_start.astimezone(MARKET_TZ).date()
        else:
            as_of = self.clock.now()
            reference = as_of.astimezone(MARKET_TZ).date()
        params = self.settings
        earliest = reference - timedelta(days=params.signal_window_days + 10)
        return as_of, date(earliest.year - params.history_years, 1, 1)

    def on_trading_iteration(self) -> None:
        """One daily check. A backtest aborts when too many runs in a row were abandoned (a dead LLM or a broken data source)."""
        assert self.pipeline is not None, "initialize() has not run"
        outcome = self.pipeline.run()
        if not outcome.completed and self.is_backtesting and outcome.abandoned_streak >= self.settings.max_consecutive_abandoned:
            raise FatalStrategyError(f"{outcome.abandoned_streak} runs abandoned in a row, aborting the backtest (see the log for the stage and the error)")

    def _run_log_path(self) -> Path | None:
        """`runs.jsonl` in this run's log directory; None outside a runner (no run id), which disables the log."""
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "runs.jsonl"

    # --- backtesting -------------------------------------------------------------------------------------

    def run_backtesting(self, **overrides: Any):
        """Backtest over the class `parameters` window on Yahoo daily bars, the whole watchlist and the benchmark preloaded."""
        symbols = list(dict.fromkeys([*self.settings.watchlist, self.parameters["benchmark_symbol"]]))
        defaults: dict[str, Any] = dict(
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            budget=Decimal(str(self.parameters["budget"])),
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            benchmark=self.parameters["benchmark_symbol"],
            timestep="day",
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
        arguments = {**defaults, **overrides}
        start = arguments["start"]
        self._backtest_start = start if isinstance(start, datetime) else None
        self._backtest_end = _run_end(arguments["end"])
        return super().run_backtesting(**arguments)


def _run_end(end: Any) -> datetime | None:
    """The instant a backtest's filings must be known by: a bare-midnight `end` covers that whole day (backtesting/CLAUDE.md); None if naive."""
    if not isinstance(end, datetime) or end.tzinfo is None:
        return None  # run_backtest refuses a naive end itself
    return end + timedelta(days=1) if end.timetz().replace(tzinfo=None) == time() else end

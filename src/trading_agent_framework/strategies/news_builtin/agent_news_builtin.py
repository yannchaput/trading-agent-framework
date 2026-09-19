"""News-builtin strategy: an LLM agent reads broad-market news and holds SPY/QQQ (bullish) or a defensive ETF.

Port of the lumibot `agent_alpaca_news_builtin` strategy onto this framework: `PrebuiltTools` (trading,
account, market data, indicators, memory) plus the `search_news` tool. News is gated on the strategy clock,
so it works in backtests as well as paper/live.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from trading_agent_framework.agents.tools import PrebuiltTools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.news_builtin.prompts import TASK_PROMPT, build_system_prompt
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError

AGENT_NAME = "news_trader"


class NewsBuiltinStrategy(Strategy):
    backtesting_start = datetime(2025, 1, 1, tzinfo=MARKET_TZ)
    backtesting_end = datetime(2026, 4, 1, tzinfo=MARKET_TZ)
    budget = Decimal("10000")
    benchmark_symbol = "SPY"

    parameters = {
        "symbols": ("SPY", "QQQ"),
        "defensive_symbol": "SHV",
        "news_symbols": "SPY,QQQ,DIA,IWM",
        "backtest_every_n_iterations": 5,
        "warmup_trading_days": 0,
    }

    def initialize(self) -> None:
        self.sleeptime = "1D" if self.is_backtesting else "2H"
        self.vars.iteration_count = 0
        self.agents.create(
            name=AGENT_NAME,
            system_prompt=build_system_prompt(
                symbols=self.parameters["symbols"],
                defensive_symbol=self.parameters["defensive_symbol"],
                news_symbols=self.parameters["news_symbols"],
            ),
            tools=[*PrebuiltTools.all(self), *news_tools(self)],
        )
        self.log_info(f"NewsBuiltinStrategy initialized (sleeptime={self.sleeptime})")

    def on_trading_iteration(self) -> None:
        self.vars.iteration_count += 1
        if self.is_backtesting and not self._backtest_iteration_is_due():
            return
        try:
            result = self.agents[AGENT_NAME].run(TASK_PROMPT, context={"current_datetime": self.get_datetime().isoformat()})
        except AgentError as exc:
            # A failed LLM call must not kill a multi-week backtest or a live loop; ConfigurationError still propagates.
            self.log_error(f"[{AGENT_NAME}] run failed: {exc}")
            return
        self.log_info(f"[{AGENT_NAME}] {result.output}")

    def _backtest_iteration_is_due(self) -> bool:
        count = self.vars.iteration_count
        return count == 1 or count % self.parameters["backtest_every_n_iterations"] == 0

    def run_backtesting(self):
        symbols = [*self.parameters["symbols"], self.parameters["defensive_symbol"]]
        return super().run_backtesting(
            data_source=YahooBacktestData,
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            commission=Decimal("0.001"),
            warmup_trading_days=self.parameters["warmup_trading_days"],
        )

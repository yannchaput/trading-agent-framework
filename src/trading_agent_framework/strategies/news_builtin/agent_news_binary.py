"""News-builtin strategy: an LLM agent reads broad-market news and holds SPY/QQQ (bullish) or a defensive ETF.

Port of the lumibot `agent_alpaca_news_builtin` strategy onto this framework: `PrebuiltTools` (trading,
account, market data, indicators, memory) plus the `search_news` tool. News is gated on the strategy clock,
so it works in backtests as well as paper/live.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading_agent_framework.agents.tools import PrebuiltTools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, FatalStrategyError

# A persistent LLM misconfiguration (wrong URL/model, dead server) would otherwise yield a flat "successful" backtest,
# so a backtest aborts (FatalStrategyError, which the executor propagates) after this many failed runs in a row.
MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS = 3


def _build_system_prompt(*, symbols: Sequence[str], defensive_symbol: str, news_symbols: str) -> str:
    allowed = ", ".join([*symbols, defensive_symbol])
    return (
        f"You are a news-driven allocator. Your only allowed instruments are {allowed} "
        f"({defensive_symbol} is the defensive ETF). Never short, never use margin, never trade anything else, "
        "never trade USD or FOREX.\n\n"
        "On every run, follow this workflow:\n"
        "1. Call search_memory to recall recent decisions and the current regime thesis.\n"
        f"2. Scan broad-market news: call search_news with symbols='{news_symbols}', include_content=False and limit=30.\n"
        "3. Pick the single most relevant article. Call search_news again with a narrow start/end window around that "
        "article's created_at (ISO 8601 with timezone), include_content=True and limit=3, to read it in full.\n"
        "4. Compare article timestamps with the current datetime given in the task and ignore stale news.\n"
        f"5. Decide the regime: bullish evidence means hold {' or '.join(symbols)}; negative or unclear evidence means hold {defensive_symbol}.\n"
        "6. Check get_positions and get_orders, then trade only the difference. Sell the current position before buying "
        "the other. Never sell more than get_positions says you hold. Size a buy at no more than 95% of the SMALLER of "
        "the 'cash' and 'buying_power' reported by get_account_balance (quantity = floor(0.95 * that amount / last price)), "
        "leaving room for fees. Never size a buy against 'buying_power' alone: on a margin account it exceeds your cash, "
        "and buying on margin is forbidden. Orders fill on a later bar, so cash does not yet include proceeds of a sell you "
        "submitted in this run: if you sold this run, do not buy in the same run; the next run completes the rotation. "
        "If submit_order comes back with an 'error', the order was refused and nothing was placed -- read the reason, "
        "correct the quantity and try once more, or skip the trade this run. "
        "Keep position sizing reasonable and never place a duplicate order.\n"
        "7. Record the outcome: call remember_decision after every decision, and open_thesis or close_thesis when the "
        "regime call changes."
    )


class NewsBinaryStrategy(Strategy):
    AGENT_NAME = "news_binary_trader"
    TASK_PROMPT = "Research current broad-market news and rebalance if needed. The current datetime is in the context below."

    parameters = {
        "backtesting_start": datetime(2025, 1, 1, tzinfo=MARKET_TZ),
        "backtesting_end": datetime(2026, 8, 14, tzinfo=MARKET_TZ),
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 300,
        "budget": 10000,
        "commission": 0.001,
    }

    strategy_parameters = {
        "symbols": ("SPY", "QQQ"),
        "defensive_symbol": "SHV",
        "news_symbols": "SPY,QQQ,DIA,IWM",
        "backtest_every_n_iterations": 1,  # Number of days between 2 effective iterations
    }

    def initialize(self) -> None:
        self.sleeptime = "1D" if self.is_backtesting else "1H"
        self.vars.iteration_count = 0
        self.vars.consecutive_agent_errors = 0
        self.vars.strategy_parameters = self.strategy_parameters
        self.agents.create(
            name=self.AGENT_NAME,
            system_prompt=_build_system_prompt(
                symbols=self.vars.strategy_parameters["symbols"],
                defensive_symbol=self.vars.strategy_parameters["defensive_symbol"],
                news_symbols=self.vars.strategy_parameters["news_symbols"],
            ),
            tools=[*PrebuiltTools.all(self), *news_tools(self)],
        )
        self.log_info(f"NewsBuiltinStrategy initialized (sleeptime={self.sleeptime})")

    def on_trading_iteration(self) -> None:
        self.vars.iteration_count += 1
        if self.is_backtesting and not self._backtest_iteration_is_due():
            return
        try:
            result = self.agents[self.AGENT_NAME].run(self.TASK_PROMPT, context={"current_datetime": self.get_datetime().isoformat()})
        except AgentError as exc:
            # A failed LLM call must not kill a live loop, and one flaky call must not kill a backtest.
            self.vars.consecutive_agent_errors += 1
            self.log_error(f"[{self.AGENT_NAME}] run failed: {exc}")
            if self.is_backtesting and self.vars.consecutive_agent_errors >= MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS:
                raise FatalStrategyError(f"[{self.AGENT_NAME}] failed {self.vars.consecutive_agent_errors} runs in a row, aborting the backtest; last error: {exc}") from exc
            return
        self.vars.consecutive_agent_errors = 0
        self.log_info(f"[{self.AGENT_NAME}] Agent output: \n{result.output}")
        for i, tool_call in enumerate(result.tool_calls):
            self.log_info(f"tool_call_{i}: {tool_call}")

    def _backtest_iteration_is_due(self) -> bool:
        """
        Set the interval of iterations for backtesting (not to trade every day)
        """
        count = self.vars.iteration_count
        return count == 1 or count % self.vars.strategy_parameters["backtest_every_n_iterations"] == 0

    def run_backtesting(self):
        # Clean memory before backtest
        memory_path = Path(Path.cwd() / "memory" / self.name / "backtesting" / "memory.sqlite")
        if memory_path and memory_path.exists:
            self.log_warning("A memory remains from previous backtest. Deleting it.")
            memory_path.unlink()
        # use class strategy parameters as the strategy constructor is not run yet
        symbols = [*self.strategy_parameters["symbols"], self.strategy_parameters["defensive_symbol"]]
        return super().run_backtesting(
            data_source=AlpacaBacktestData,  # Use alpaca broker for intraday quotes (> 6 year history)
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            preload_assets=[Asset(symbol=symbol) for symbol in symbols],
            budget=Decimal(self.parameters["budget"]),
            commission=Decimal(self.parameters["commission"]),
            warmup_trading_days=self.parameters["warmup_trading_days"],
        )

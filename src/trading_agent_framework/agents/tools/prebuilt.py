"""`PrebuiltTools.all(strategy)`: trading, account, market-data, indicator and memory tools in one call."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.trading import trading_tools
from trading_agent_framework.memory.tools import memory_tools

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class PrebuiltTools:
    """Everything a trading agent needs out of the box, bundled in one call."""

    @staticmethod
    def all(strategy: "Strategy") -> list[Callable[..., Any]]:  # noqa: UP037
        """Trading, account, market-data, indicator and memory tools for `strategy`."""
        return [
            *memory_tools(strategy.memory),
            *trading_tools(strategy),
            *account_tools(strategy),
            *market_data_tools(strategy),
            *indicator_tools(strategy),
        ]

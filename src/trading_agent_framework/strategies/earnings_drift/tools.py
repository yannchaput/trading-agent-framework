"""The agent's tools (spec §5.1): the desk's order tools, `skip`, and the clock-gated research tools.

Each docstring is a single line on purpose: it is the tool description sent to the model on every call.
No `from __future__ import annotations`: the agent layer reads the real annotations.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.strategies.earnings_drift.desk import Desk

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

ORDER_TOOLS = ("buy", "set_trailing_stop", "sell", "skip")  # exempt from the per-run tool budget
_FILING_TOOLS = ("get_filings", "get_filing_document")


def desk_tools(desk: Desk) -> list[Callable[..., dict[str, Any]]]:
    def buy(symbol: str, quantity: int, trail_percent: float, reason: str) -> dict[str, Any]:
        """Buy a candidate at the next open; a trailing stop of trail_percent (3-15) is placed when it fills."""
        return desk.buy(symbol, quantity, trail_percent, reason)

    def set_trailing_stop(symbol: str, trail_percent: float, reason: str) -> dict[str, Any]:
        """Tighten a holding's trailing stop to trail_percent (never wider than the current one)."""
        return desk.set_trailing_stop(symbol, trail_percent, reason)

    def sell(symbol: str, reason: str) -> dict[str, Any]:
        """Sell a whole holding at the next open."""
        return desk.sell(symbol, reason)

    def skip(symbol: str, reason: str) -> dict[str, Any]:
        """Pass on a candidate, with the reason."""
        return desk.skip(symbol, reason)

    return [buy, set_trailing_stop, sell, skip]


def research_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """News search, the two SEC filing tools (the 8-K press release) and market data, all gated on the strategy clock."""
    from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
    from trading_agent_framework.agents.tools.market_data import market_data_tools
    from trading_agent_framework.agents.tools.news import news_tools

    filings = [tool for tool in fundamentals_tools(strategy) if tool.__name__ in _FILING_TOOLS]
    return [*news_tools(strategy), *filings, *market_data_tools(strategy)]

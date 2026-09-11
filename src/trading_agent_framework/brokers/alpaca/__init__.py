from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream

try:
    __version__ = version("trading_agent_framework")
except PackageNotFoundError:
    # Package is not installed (e.g., running from local source)
    __version__ = "unknown"

__all__ = [
    "AlpacaBroker",
    "AlpacaMarketClock",
    "AlpacaTradeStream",
]

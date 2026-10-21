"""The insider_trades strategy: a researcher and a trader that follow what a watchlist's officers and directors do with their stock."""

from trading_agent_framework.strategies.insider_trades.agent_insider_trades import InsiderTradesStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["InsiderTradesStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

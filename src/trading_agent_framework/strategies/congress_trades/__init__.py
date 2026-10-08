"""The congress_trades strategy: a researcher, a portfolio agent and a trader that mirror a member's disclosed stock holdings."""

from trading_agent_framework.strategies.congress_trades.agent_congress_trades import CongressTradesStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["CongressTradesStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

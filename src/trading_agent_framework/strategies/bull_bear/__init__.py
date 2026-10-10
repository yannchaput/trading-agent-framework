"""The bull vs bear strategy: a researcher, a bull, a bear and a judge debate the top of cross_momentum's ranking."""

from trading_agent_framework.strategies.bull_bear.agent_bull_bear import BullBearStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["BullBearStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

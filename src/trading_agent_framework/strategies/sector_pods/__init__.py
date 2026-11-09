"""sector_pods: five sector-pod agents, a risk manager and a portfolio manager over a sector-ETF universe."""

from trading_agent_framework.strategies.sector_pods.strategy import SectorPodsStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["SectorPodsStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

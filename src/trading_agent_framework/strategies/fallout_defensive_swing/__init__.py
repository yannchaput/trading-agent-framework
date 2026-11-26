"""fallout_defensive_swing: a defensive ETF mix sold at a small profit and bought back after a dip."""

from trading_agent_framework.strategies.fallout_defensive_swing.agent_fallout_defensive_swing import FalloutDefensiveSwingStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["FalloutDefensiveSwingStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

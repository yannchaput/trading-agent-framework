from trading_agent_framework.utils import get_version

from .agent_cross_momentum_v5 import CrossMomentumStrategyV5

__all__ = [
    "CrossMomentumStrategyV5",
]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

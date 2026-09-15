from trading_agent_framework.utils import get_version

from .agent_cross_momentum import CrossMomentumStrategy

__all__ = [
    "CrossMomentumStrategy",
]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

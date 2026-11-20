"""ray_dalio: an LLM team that reads the macro regime and holds a basket of at least three leveraged ETFs."""

from trading_agent_framework.strategies.ray_dalio.agent_ray_dalio import RayDalioStrategy
from trading_agent_framework.utils import get_version

__all__ = ["RayDalioStrategy"]


def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")

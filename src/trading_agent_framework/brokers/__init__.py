"""Broker-agnostic public API.

`Broker` and `OrderTracker` never import `alpaca`, so they are imported eagerly
here for convenience. `AlpacaBroker` and `AlpacaTradeStream` live in
`brokers.alpaca`, which does import `alpaca` (and therefore pandas) -- those
two names are wired through a module-level `__getattr__` lazy-import map so
that `import trading_agent_framework.brokers` alone stays import-light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.tracker import OrderTracker

if TYPE_CHECKING:
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
    from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
    from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trading_agent_framework")
except PackageNotFoundError:
    # Package is not installed (e.g., running from local source)
    __version__ = "unknown"

__all__ = [
    "AlpacaBroker",
    "AlpacaMarketClock",
    "AlpacaTradeStream",
    "Broker",
    "OrderTracker",
]

_LAZY = {
    "AlpacaBroker": (".alpaca.broker", "AlpacaBroker"),
    "AlpacaTradeStream": (".alpaca.stream", "AlpacaTradeStream"),
    "AlpacaMarketClock": (".alpaca.clock", "AlpacaMarketClock"),
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        module_path, attr = _LAZY[name]
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        # Cache in globals so __getattr__ is only called once per name
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + __all__

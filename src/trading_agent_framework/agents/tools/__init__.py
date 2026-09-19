"""Agent tools: PrebuiltTools.all() plus individually-wired tool factories.

`news_tools` and `fundamentals_tools` are wired through a module-level `__getattr__`
lazy-import map, mirroring `brokers/__init__.py`'s documented pattern: `news.py` imports
`AlpacaBroker` (and therefore `alpaca`/pandas) at module level, and `fundamentals.py` imports
`edgar_client` (which imports `httpx`) at module level, so importing them eagerly here would
mean `import trading_agent_framework.agents.tools` alone pulls in the entire fundamentals/news
stack just to reach `PrebuiltTools`. `macro_tools` stays eager: `fredapi` is already deferred
inside `_default_fred_client`, not imported at `macro.py` module level.
"""

from typing import TYPE_CHECKING

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.prebuilt import PrebuiltTools
from trading_agent_framework.agents.tools.trading import trading_tools

if TYPE_CHECKING:
    from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
    from trading_agent_framework.agents.tools.news import news_tools

__all__ = [
    "PrebuiltTools",
    "account_tools",
    "fundamentals_tools",
    "indicator_tools",
    "macro_tools",
    "market_data_tools",
    "news_tools",
    "trading_tools",
]

_LAZY = {
    "news_tools": (".news", "news_tools"),
    "fundamentals_tools": (".fundamentals", "fundamentals_tools"),
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

from typing import TYPE_CHECKING

from trading_agent_framework.utils.package_helper import get_version

if TYPE_CHECKING:
    from trading_agent_framework.strategies.congress_trades.tools.congress import congress_research_tools


__all__ = ["congress_research_tools"]


_LAZY = {
    "congress_research_tools": (".congress", "congress_research_tools"),
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


__version__ = get_version("trading_agent_framework")

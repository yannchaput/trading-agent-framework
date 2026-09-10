from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from trading_agent_framework.config.env import (
    TRADING_MODES,
    AlpacaCredentials,
    TradingMode,
    load_strategy_env,
)

try:
    __version__ = version("trading_agent_framework")
except PackageNotFoundError:
    # Package is not installed (e.g., running from local source)
    __version__ = "unknown"

__all__ = [
    "AlpacaCredentials",
    "TRADING_MODES",
    "TradingMode",
    "load_strategy_env",
]

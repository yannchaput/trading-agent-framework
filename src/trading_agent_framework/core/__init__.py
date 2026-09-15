"""Strategy framework: lifecycle hooks, executor, and the lumibot-style Strategy base."""

from __future__ import annotations

from trading_agent_framework.core.executor import StrategyExecutor
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")


__all__ = [
    "Strategy",
    "StrategyExecutor",
]

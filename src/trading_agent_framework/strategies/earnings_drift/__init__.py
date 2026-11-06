"""earnings_drift: post-earnings announcement drift with an agent that trades (spec 2026-10-05)."""

from trading_agent_framework.strategies.earnings_drift.agent_earnings_drift import EarningsDriftStrategy
from trading_agent_framework.utils import get_version

__version__ = get_version("trading_agent_framework")

__all__ = ["EarningsDriftStrategy"]

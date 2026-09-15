from __future__ import annotations


class TradingFrameworkError(Exception):
    """Base exception for all trading-agent-framework errors."""


class ConfigurationError(TradingFrameworkError):
    """Raised when application or environment configuration is invalid."""


class BrokerError(TradingFrameworkError):
    """Raised when a broker integration fails."""


class OrderValidationError(BrokerError, ValueError):
    """Raised when an order fails validation before submission."""


class OrderEventError(BrokerError, ValueError):
    """Raised when an order event cannot be processed."""


class MemoryStoreError(TradingFrameworkError):
    """Raised when the agent memory database cannot be read or written."""


class MemoryValidationError(MemoryStoreError, ValueError):
    """Raised when a memory write is rejected (bad text/kind/tags, unknown or non-open thesis)."""


class AgentError(TradingFrameworkError):
    """Raised when building or running a LangChain agent fails (never a raw SDK exception)."""


class BacktestError(TradingFrameworkError):
    """Raised when the backtesting simulation cannot proceed (never a raw exception)."""


class BacktestDataError(BacktestError):
    """Raised when a BacktestDataSource cannot fetch, cache, or parse historical data."""

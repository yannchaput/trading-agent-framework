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

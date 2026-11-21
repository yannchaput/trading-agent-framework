"""Exception hierarchy of the framework."""

from __future__ import annotations

from decimal import Decimal


class TradingFrameworkError(Exception):
    """Base exception for all trading-agent-framework errors."""


class ConfigurationError(TradingFrameworkError):
    """Raised when application or environment configuration is invalid."""


class BrokerError(TradingFrameworkError):
    """Raised when a broker integration fails."""


class BrokerUncertainError(BrokerError):
    """A broker write whose outcome is unknown (timeout, transport failure, 5xx): it may or may not have happened."""


class OrderValidationError(BrokerError, ValueError):
    """Raised when an order fails validation before submission."""


class InsufficientFundsError(OrderValidationError):
    """A buy costs more than `Strategy.spendable_cash()`; `max_quantity` is the most that fits."""

    def __init__(self, message: str, *, spendable: Decimal, max_quantity: Decimal) -> None:
        """Keep the spendable cash (may be negative) and the largest quantity that fits (0 when none)."""
        super().__init__(message)
        self.spendable = spendable
        self.max_quantity = max_quantity


class OrderEventError(BrokerError, ValueError):
    """Raised when an order event cannot be processed."""


class MemoryStoreError(TradingFrameworkError):
    """Raised when the agent memory database cannot be read or written."""


class MemoryValidationError(MemoryStoreError, ValueError):
    """Raised when a memory write is rejected (bad text/kind/tags, unknown or non-open thesis)."""


class LLMStatsError(TradingFrameworkError):
    """Raised when the LLM-stats (agent telemetry) database cannot be prepared or written."""


class AgentError(TradingFrameworkError):
    """Raised when building or running a LangChain agent fails (never a raw SDK exception)."""


class AgentLoopError(AgentError):
    """Raised when an agent run used up its step limit (`AgentHandle.run(max_steps=...)`): a loop, e.g. a submission rejected again and again."""


class FatalStrategyError(TradingFrameworkError):
    """Raised from `on_trading_iteration` to abort the run; the executor otherwise logs an iteration failure and carries on."""


class BacktestError(TradingFrameworkError):
    """Raised when the backtesting simulation cannot proceed (never a raw exception)."""


class BacktestDataError(BacktestError):
    """Raised when a BacktestDataSource cannot fetch, cache, or parse historical data."""


class UniverseHistoryError(BacktestError):
    """Raised when the point-in-time universe history cannot be built, or has no list for a requested month."""


class MacroDataError(TradingFrameworkError):
    """Raised when a macro (FRED) data lookup fails."""


class YahooDataError(TradingFrameworkError):
    """Raised when a strategy's Yahoo daily-bar lookup fails."""


class RegimeDataError(TradingFrameworkError):
    """Raised when the SPY/VIX/VIX3M series for the market regime cannot be fetched or is incomplete."""


class FundamentalsError(TradingFrameworkError):
    """Raised when a SEC EDGAR fundamentals lookup fails."""


class FundamentalsNotFoundError(FundamentalsError):
    """Raised when SEC has nothing for the request (unknown ticker, HTTP 404): a real absence, not a failed lookup."""


class CongressDataError(TradingFrameworkError):
    """Raised when a congressional-disclosure lookup or parse fails (never a raw httpx/pypdf exception)."""


class CongressNotFoundError(CongressDataError):
    """Raised when the Clerk has nothing for the request (HTTP 404, e.g. a year index not published yet): a real absence, not a failed lookup."""


class InsiderDataError(TradingFrameworkError):
    """Raised when an insider-transaction (SEC Form 4) lookup or parse fails (never a raw httpx/XML exception)."""

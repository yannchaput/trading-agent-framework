from __future__ import annotations

from trading_agent_framework.utils.errors import (
    BacktestDataError,
    BacktestError,
    FundamentalsError,
    MacroDataError,
    TradingFrameworkError,
)


def test_backtest_error_is_a_trading_framework_error() -> None:
    assert issubclass(BacktestError, TradingFrameworkError)


def test_backtest_data_error_is_a_backtest_error() -> None:
    assert issubclass(BacktestDataError, BacktestError)


def test_macro_data_error_is_a_trading_framework_error() -> None:
    assert issubclass(MacroDataError, TradingFrameworkError)


def test_fundamentals_error_is_a_trading_framework_error() -> None:
    assert issubclass(FundamentalsError, TradingFrameworkError)

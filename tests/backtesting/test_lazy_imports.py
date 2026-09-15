from __future__ import annotations

import subprocess
import sys

import pytest

from trading_agent_framework import backtesting
from trading_agent_framework.backtesting.data import base as base_module


def test_backtest_data_source_is_eagerly_reexported() -> None:
    assert backtesting.BacktestDataSource is base_module.BacktestDataSource


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        backtesting.does_not_exist  # noqa: B018


def test_lazy_attributes_resolve_to_the_real_classes_in_process() -> None:
    from trading_agent_framework.backtesting.broker import BacktestBroker as direct_broker
    from trading_agent_framework.backtesting.clock import BacktestClock as direct_clock
    from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData as direct_alpaca
    from trading_agent_framework.backtesting.data.cache import CachedDataSource as direct_cache
    from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData as direct_yahoo
    from trading_agent_framework.backtesting.runner import BacktestResult as direct_result
    from trading_agent_framework.backtesting.runner import run_backtest as direct_run

    assert backtesting.BacktestBroker is direct_broker
    assert backtesting.BacktestClock is direct_clock
    assert backtesting.AlpacaBacktestData is direct_alpaca
    assert backtesting.CachedDataSource is direct_cache
    assert backtesting.YahooBacktestData is direct_yahoo
    assert backtesting.BacktestResult is direct_result
    assert backtesting.run_backtest is direct_run
    # Second access hits the globals() cache set by __getattr__, not the _LAZY branch again.
    assert backtesting.BacktestBroker is direct_broker


def test_dir_includes_lazy_and_eager_names() -> None:
    names = dir(backtesting)
    for name in (
        "BacktestDataSource", "BacktestBroker", "BacktestClock", "CachedDataSource",
        "YahooBacktestData", "AlpacaBacktestData", "run_backtest", "BacktestResult",
    ):
        assert name in names


def test_importing_backtesting_package_does_not_import_vectorbt_or_yfinance() -> None:
    """Must run in a subprocess -- see tests/brokers/test_lazy_imports.py's identical
    reasoning: other test modules have already imported these by the time this test
    runs in-process, so only a fresh interpreter makes the assertion meaningful."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "import trading_agent_framework.backtesting\n"
            "import sys\n"
            "assert 'vectorbt' not in sys.modules\n"
            "assert 'numba' not in sys.modules\n"
            "assert 'yfinance' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_run_backtest_lazily_imports_vectorbt_is_not_true_until_called() -> None:
    """run_backtest itself is light to import (metrics.py's vectorbt import is deferred
    further, inside compute_metrics) -- confirm importing the function doesn't pull
    vectorbt in either, only actually calling compute_metrics does (not exercised here)."""
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from trading_agent_framework.backtesting import run_backtest\n"
            "import sys\n"
            "assert 'vectorbt' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_yahoo_backtest_data_does_not_import_yfinance_until_used() -> None:
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from trading_agent_framework.backtesting import YahooBacktestData\n"
            "import sys\n"
            "assert 'yfinance' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout

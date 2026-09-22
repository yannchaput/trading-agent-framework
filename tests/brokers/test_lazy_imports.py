from __future__ import annotations

import subprocess
import sys

import pytest

from trading_agent_framework import brokers
from trading_agent_framework.brokers import base as base_module
from trading_agent_framework.brokers import tracker as tracker_module


def test_broker_and_order_tracker_are_eagerly_reexported() -> None:
    assert brokers.Broker is base_module.Broker
    assert brokers.OrderTracker is tracker_module.OrderTracker


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        brokers.does_not_exist  # noqa: B018


def test_lazy_attribute_resolves_to_the_real_class_in_process() -> None:
    """Exercises the __getattr__ lazy-import branch itself (import + cache),
    as opposed to the subprocess tests below which check the *absence* of
    `alpaca` from `sys.modules` before this attribute is ever touched.
    """
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker as direct_broker
    from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock as direct_clock
    from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream as direct_stream

    assert brokers.AlpacaBroker is direct_broker
    assert brokers.AlpacaTradeStream is direct_stream
    assert brokers.AlpacaMarketClock is direct_clock
    # Second access hits the globals() cache set by __getattr__, not the _LAZY branch again.
    assert brokers.AlpacaBroker is direct_broker


def test_dir_includes_lazy_and_eager_names() -> None:
    names = dir(brokers)
    assert "AlpacaBroker" in names
    assert "AlpacaTradeStream" in names
    assert "AlpacaMarketClock" in names
    assert "Broker" in names
    assert "OrderTracker" in names


def test_importing_brokers_package_does_not_import_alpaca() -> None:
    """Must run in a subprocess.

    By the time this test runs, other test modules in the same pytest session
    have already imported `alpaca` (and therefore pandas) dozens of times, so
    checking `sys.modules` in-process would pass or fail for reasons unrelated
    to whether `brokers/__init__.py` itself is import-light. A fresh
    interpreter is the only way to make the assertion meaningful.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.brokers\n"
            "import sys\n"
            "assert 'alpaca' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_alpaca_broker_lazily_imports_alpaca() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trading_agent_framework.brokers import AlpacaBroker\n"
            "import sys\n"
            "assert 'alpaca' in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_alpaca_trade_stream_lazily_imports_alpaca() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trading_agent_framework.brokers import AlpacaTradeStream\n"
            "import sys\n"
            "assert 'alpaca' in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_accessing_alpaca_market_clock_lazily_imports_alpaca() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from trading_agent_framework.brokers import AlpacaMarketClock\n"
            "import sys\n"
            "assert 'alpaca' in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_the_factory_does_not_import_alpaca() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import trading_agent_framework.brokers.factory\nimport sys\nassert 'alpaca' not in sys.modules\nprint('OK')\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_importing_brokers_and_the_factory_does_not_import_ib_async() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import trading_agent_framework.brokers\nimport trading_agent_framework.brokers.factory\nimport sys\nassert 'ib_async' not in sys.modules\nprint('OK')\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_ibkr_broker_is_a_lazy_export() -> None:
    import trading_agent_framework.brokers as brokers
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    assert brokers.IbkrBroker is IbkrBroker
    assert "IbkrBroker" in dir(brokers)

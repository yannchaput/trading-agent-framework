"""Backtesting subsystem: the third trading mode.

`vectorbt`+`numba` (backtesting/metrics.py) and `yfinance`
(backtesting/data/yahoo.py) are heavy and imported only inside method bodies
there; importing this package alone must not pull them in -- only actually
calling into a codepath that needs them does. Mirrors `brokers/__init__.py`.
"""

from __future__ import annotations

from trading_agent_framework.backtesting.data.base import BacktestDataSource

__all__ = [
    "AlpacaBacktestData",
    "BacktestBroker",
    "BacktestClock",
    "BacktestDataSource",
    "BacktestResult",
    "CachedDataSource",
    "YahooBacktestData",
    "run_backtest",
]

_LAZY = {
    "BacktestBroker": (".broker", "BacktestBroker"),
    "BacktestClock": (".clock", "BacktestClock"),
    "CachedDataSource": (".data.cache", "CachedDataSource"),
    "YahooBacktestData": (".data.yahoo", "YahooBacktestData"),
    "AlpacaBacktestData": (".data.alpaca", "AlpacaBacktestData"),
    "run_backtest": (".runner", "run_backtest"),
    "BacktestResult": (".runner", "BacktestResult"),
}


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        module_path, attr = _LAZY[name]
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + __all__

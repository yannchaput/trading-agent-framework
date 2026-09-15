"""Alpaca support helpers — rate limiting and safe backtesting.

Two utilities for working with Alpaca's data/backtesting stack:

1. ``AlpacaApiRateLimiter`` — thread-safe token bucket that spaces out Alpaca
   market-data calls, keeping the strategy under Alpaca's ~200 req/min limit
   (universe filtering fires one ``get_historical_prices`` request per ticker).

2. ``SafeAlpacaBacktesting`` — an ``AlpacaBacktesting`` subclass that converts
   "no data" errors into empty DataFrames / ``None`` so a backtest can continue
   past symbols with no history (recent spinoff, IPO, or delisted).
"""

import logging
import threading
import time

from trading_agent_framework.config import TradingMode

logger = logging.getLogger(__name__)

# Comfortably below Alpaca's ~200 requests/minute market-data limit, leaving
# headroom for the benchmark/SPY calls and any other strategies sharing the key.
DEFAULT_CALLS_PER_MINUTE = 100.0


class AlpacaApiRateLimiter:
    """Thread-safe token bucket that spaces out Alpaca API calls."""

    def __init__(self, trading_mode: TradingMode, calls_per_minute: float = DEFAULT_CALLS_PER_MINUTE):
        self._interval = 60.0 / calls_per_minute
        self._lock = threading.Lock()
        self._last = 0.0
        self._trading_mode = trading_mode

    def wait(self) -> None:
        """Block until the next call is allowed under the configured rate.

        No-ops during backtesting: ``get_historical_prices`` then reads the local
        data cache (or data already downloaded up-front), never Alpaca's live
        market-data API — so there is nothing to rate-limit.
        """
        if self._trading_mode == TradingMode.BACKTESTING:
            return
        with self._lock:
            now = time.monotonic()  # in seconds
            sleep_for = self._last + self._interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.monotonic()

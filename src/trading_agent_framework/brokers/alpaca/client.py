"""Trivial factory functions for building Alpaca SDK clients from credentials.

Kept deliberately trivial: `AlpacaBroker` never builds its own client, so
tests can inject a mock/fake in its place. All three functions import `alpaca`
inside the function body, so importing this module doesn't force an `alpaca`
import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_agent_framework.config.env import AlpacaCredentials

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream


def build_trading_client(creds: AlpacaCredentials) -> TradingClient:
    from alpaca.trading.client import TradingClient

    return TradingClient(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper)


def build_trading_stream(creds: AlpacaCredentials) -> TradingStream:
    from alpaca.trading.stream import TradingStream

    return TradingStream(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper)


def build_stock_data_client(creds: AlpacaCredentials) -> StockHistoricalDataClient:
    """Market data uses the trading credentials; paper and live share one data endpoint."""
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(api_key=creds.api_key, secret_key=creds.api_secret)


def build_news_client(creds: AlpacaCredentials) -> NewsClient:
    """News uses the trading credentials, same as market data."""
    from alpaca.data.historical.news import NewsClient

    return NewsClient(api_key=creds.api_key, secret_key=creds.api_secret)

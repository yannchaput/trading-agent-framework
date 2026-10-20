"""Trivial factory functions for building Alpaca SDK clients from credentials.

Kept deliberately trivial: `AlpacaBroker` never builds its own client, so
tests can inject a mock/fake in its place. All three functions import `alpaca`
inside the function body, so importing this module doesn't force an `alpaca`
import.

alpaca-py sets no request timeout, so each REST client gets `REQUEST_TIMEOUT_SECONDS` here: a stalled read
would otherwise block forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from requests.adapters import HTTPAdapter

from trading_agent_framework.config.env import AlpacaCredentials

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream

REQUEST_TIMEOUT_SECONDS = 30.0


class _TimeoutAdapter(HTTPAdapter):
    """An `HTTPAdapter` that gives a request with no timeout of its own `timeout` seconds."""

    def __init__(self, timeout: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeout = timeout

    def send(self, request: Any, **kwargs: Any) -> Any:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


def _with_timeout[T](client: T) -> T:
    """Mount the timeout adapter on the SDK client's own `requests` session (`_session`, which has no public setter)."""
    adapter = _TimeoutAdapter(REQUEST_TIMEOUT_SECONDS)
    for prefix in ("https://", "http://"):
        client._session.mount(prefix, adapter)  # type: ignore[attr-defined]
    return client


def build_trading_client(creds: AlpacaCredentials) -> TradingClient:
    from alpaca.trading.client import TradingClient

    return _with_timeout(TradingClient(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper))


def build_trading_stream(creds: AlpacaCredentials) -> TradingStream:
    from alpaca.trading.stream import TradingStream

    return TradingStream(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper)


def build_stock_data_client(creds: AlpacaCredentials) -> StockHistoricalDataClient:
    """Market data uses the trading credentials; paper and live share one data endpoint."""
    from alpaca.data.historical import StockHistoricalDataClient

    return _with_timeout(StockHistoricalDataClient(api_key=creds.api_key, secret_key=creds.api_secret))


def build_news_client(creds: AlpacaCredentials) -> NewsClient:
    """News uses the trading credentials, same as market data."""
    from alpaca.data.historical.news import NewsClient

    return _with_timeout(NewsClient(api_key=creds.api_key, secret_key=creds.api_secret))

"""`AlpacaNewsProvider`: Alpaca's news feed behind the broker-agnostic `NewsProvider` seam.

Does the I/O (one `NewsClient.get_news` call) and wraps failures as `BrokerError`; all request
building and response parsing stays in the pure `market_data` module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, cast

from trading_agent_framework.brokers.alpaca import market_data
from trading_agent_framework.brokers.alpaca.client import build_news_client
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials


class AlpacaNewsProvider:
    def __init__(self, client: market_data.AlpacaNewsClient) -> None:
        self._client = client

    @classmethod
    def from_credentials(cls, creds: AlpacaCredentials) -> AlpacaNewsProvider:
        return cls(cast("market_data.AlpacaNewsClient", build_news_client(creds)))

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]:
        request = market_data.build_news_request(symbols, start=start, end=end, limit=limit, include_content=include_content)
        try:
            response = self._client.get_news(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch news: {exc}") from exc
        return market_data.parse_news(response)


def lazy_news_provider(credentials: Callable[[], AlpacaCredentials]) -> Callable[[], AlpacaNewsProvider]:
    """A factory reading the news credentials only when first called, so a strategy that never
    searches news needs no `ALPACA_NEWS_*`. Missing credentials become `BrokerError`, which the
    news tool turns into `{"error": ...}`."""

    def build() -> AlpacaNewsProvider:
        try:
            creds = credentials()
        except ConfigurationError as exc:
            raise BrokerError(f"no news source available: {exc}") from exc
        return AlpacaNewsProvider.from_credentials(creds)

    return build

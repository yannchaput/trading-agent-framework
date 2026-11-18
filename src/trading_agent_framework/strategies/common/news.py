"""The fail-fast news-source check of the strategies whose agents research news (sector_pods, bull_bear)."""

from __future__ import annotations

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError


def require_news_provider(broker: Broker, who: str) -> None:
    """Refuse to start without a news source: the agents' `search_news` would only ever return `{"error": ...}`.

    A backtest builds its Alpaca provider from `ALPACA_NEWS_*` on first use, so a missing credential would otherwise
    surface only inside the tool, and every note of the run would be research-free.

    Args:
        broker: The broker whose `news_provider()` is checked.
        who: The subject and verb of the message, so each caller keeps its grammar ("the pods need", "the researcher needs").

    Raises:
        ConfigurationError: The provider cannot be built (the broker's reason is kept) or the broker has none.
    """
    try:
        provider = broker.news_provider()
    except BrokerError as exc:
        raise ConfigurationError(f"{who} a news provider: {exc}") from exc
    if provider is None:
        raise ConfigurationError(f"{who} a news provider and this broker has none")

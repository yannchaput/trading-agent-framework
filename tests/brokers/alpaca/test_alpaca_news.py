from __future__ import annotations

import pytest
from tests.fakes import FakeNewsClient, et, make_alpaca_news_article

from trading_agent_framework.brokers.alpaca import news as news_module
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.utils.errors import BrokerError

_NOW = et(2026, 9, 14, 10)


def test_get_news_builds_a_request_and_parses_the_response() -> None:
    client = FakeNewsClient()
    client.articles = [make_alpaca_news_article(headline="Rates cut")]
    provider = AlpacaNewsProvider(client)

    articles = provider.get_news(["SPY", "QQQ"], start=None, end=_NOW, limit=5, include_content=False)

    assert articles[0]["headline"] == "Rates cut"
    [request] = client.news_requests
    assert request.symbols == "SPY,QQQ"
    assert request.limit == 5


def test_get_news_wraps_client_failures_as_broker_error() -> None:
    client = FakeNewsClient()
    client.raises = RuntimeError("rate limited")

    with pytest.raises(BrokerError, match="Failed to fetch news"):
        AlpacaNewsProvider(client).get_news(end=_NOW)


def test_from_credentials_builds_the_client_from_the_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNewsClient()
    seen: list[AlpacaCredentials] = []

    def fake_build(creds: AlpacaCredentials) -> FakeNewsClient:
        seen.append(creds)
        return client

    monkeypatch.setattr(news_module, "build_news_client", fake_build)
    creds = AlpacaCredentials(api_key="k", api_secret="s")

    provider = AlpacaNewsProvider.from_credentials(creds)
    provider.get_news(end=_NOW)

    assert seen == [creds]
    assert len(client.news_requests) == 1

from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.strategies.common.news import require_news_provider
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError


def _broker(news: object | None) -> FakeBroker:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="sector_pods")
    broker.news = news  # type: ignore[attr-defined]
    return broker


def test_a_broker_with_a_news_provider_passes() -> None:
    require_news_provider(_broker(object()), "the pods need")


def test_a_broker_without_a_news_provider_is_a_configuration_error_naming_who_needs_it() -> None:
    with pytest.raises(ConfigurationError) as plural:
        require_news_provider(_broker(None), "the pods need")
    with pytest.raises(ConfigurationError) as singular:
        require_news_provider(_broker(None), "the researcher needs")

    assert str(plural.value) == "the pods need a news provider and this broker has none"
    assert str(singular.value) == "the researcher needs a news provider and this broker has none"


def test_a_provider_that_cannot_be_built_keeps_the_brokers_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker(object())

    def broken() -> None:
        raise BrokerError("ALPACA_NEWS_API_KEY is not set")

    monkeypatch.setattr(broker, "news_provider", broken)

    with pytest.raises(ConfigurationError, match="the researcher needs a news provider: ALPACA_NEWS_API_KEY is not set"):
        require_news_provider(broker, "the researcher needs")

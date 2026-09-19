from __future__ import annotations

from decimal import Decimal

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import et

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.utils.errors import BrokerError


class _Source:
    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
        return []


def _broker(**kwargs: object) -> BacktestBroker:
    clock = BacktestClock(start=et(2026, 1, 5, 10), sessions=[])
    return BacktestBroker("news", data_source=FakeBacktestDataSource(), clock=clock, budget=Decimal("10000"), **kwargs)


def test_an_injected_news_source_is_returned_as_is() -> None:
    source = _Source()

    assert _broker(news_source=source).news_provider() is source


def test_the_default_news_source_is_built_lazily_from_env_credentials_and_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    sentinel = _Source()
    built: list[object] = []

    def fake_from_credentials(cls: type, creds: object) -> _Source:
        built.append(creds)
        return sentinel

    monkeypatch.setattr(AlpacaNewsProvider, "from_credentials", classmethod(fake_from_credentials))
    broker = _broker()

    assert built == []  # nothing is constructed until the news tool asks
    assert broker.news_provider() is sentinel
    assert broker.news_provider() is sentinel
    assert len(built) == 1


def test_missing_alpaca_credentials_surface_as_a_broker_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    with pytest.raises(BrokerError, match="no news source available for backtesting"):
        _broker().news_provider()

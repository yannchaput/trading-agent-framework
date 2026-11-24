from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import et

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.backtesting.news import CutoffNewsProvider
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider
from trading_agent_framework.utils.errors import BrokerError


class _Source:
    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False, sort=None):
        return []


def _broker(**kwargs: object) -> BacktestBroker:
    clock = BacktestClock(start=et(2026, 1, 5, 10), sessions=[])
    return BacktestBroker("news", data_source=FakeBacktestDataSource(), clock=clock, budget=Decimal("10000"), **kwargs)


def test_an_injected_news_source_is_returned_as_is() -> None:
    source = _Source()

    assert _broker(news_source=source).news_provider() is source


def test_the_default_news_source_is_the_cutoff_router_and_is_memoized() -> None:
    broker = _broker()

    assert isinstance(broker.news_provider(), CutoffNewsProvider)
    assert broker.news_provider() is broker.news_provider()


def test_the_default_builds_alpaca_lazily_and_only_for_a_window_reaching_the_cutoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPACA_NEWS_API_KEY", "k")
    monkeypatch.setenv("ALPACA_NEWS_API_SECRET", "s")
    monkeypatch.delenv("ALPACA_IS_PAPER", raising=False)
    (tmp_path / "cache/gdelt/gkg_stockmarket").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)  # the default GDELT root is relative to the working directory
    built: list[object] = []

    def fake_from_credentials(cls: type, creds: object) -> _Source:
        built.append(creds)
        return _Source()

    monkeypatch.setattr(AlpacaNewsProvider, "from_credentials", classmethod(fake_from_credentials))
    provider = _broker().news_provider()
    assert provider is not None

    assert provider.get_news(end=datetime(2026, 1, 5, tzinfo=UTC)) == []  # GDELT only: an empty cache dir
    assert built == []
    provider.get_news(start=datetime(2026, 5, 1, tzinfo=UTC), end=datetime(2026, 5, 4, tzinfo=UTC))
    provider.get_news(start=datetime(2026, 5, 1, tzinfo=UTC), end=datetime(2026, 5, 4, tzinfo=UTC))
    assert len(built) == 1


def test_missing_alpaca_credentials_surface_as_a_broker_error_after_the_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_NEWS_API_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_IS_PAPER", raising=False)
    provider = _broker().news_provider()  # building the router needs no credentials
    assert provider is not None

    with pytest.raises(BrokerError, match="no news source available"):
        provider.get_news(end=datetime(2026, 5, 4, tzinfo=UTC))

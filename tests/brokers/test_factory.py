from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.brokers import factory
from trading_agent_framework.config.env import BrokerKind, BrokerSettings
from trading_agent_framework.utils.errors import ConfigurationError


def _fake_broker(strategy_name: str) -> FakeBroker:
    return FakeBroker(FakeClock(et(2026, 9, 22, 10)), strategy_name=strategy_name)


def test_build_broker_dispatches_on_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, BrokerSettings]] = []

    def build(strategy_name, settings, env):
        calls.append((strategy_name, settings))
        return _fake_broker(strategy_name)

    monkeypatch.setitem(factory.BUILDERS, BrokerKind.ALPACA, build)

    broker = factory.build_broker("news_binary", env={"BROKER_API_IS_PAPER": "false"})

    assert broker.strategy_name == "news_binary"
    assert calls == [("news_binary", BrokerSettings(kind=BrokerKind.ALPACA, is_paper=False))]


def test_build_broker_reports_a_broker_without_a_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(factory.BUILDERS, BrokerKind.IBKR, raising=False)

    with pytest.raises(ConfigurationError, match="BROKER=ibkr is not supported"):
        factory.build_broker("s", env={"BROKER": "ibkr"})


def test_the_alpaca_builder_reads_trading_and_data_credentials_but_not_news(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker

    captured: dict[str, object] = {}

    def fake_from_credentials(cls, strategy_name, *, trading, data, news, with_stream=True):
        captured.update(trading=trading, data=data, news=news)
        return _fake_broker(strategy_name)

    monkeypatch.setattr(AlpacaBroker, "from_credentials", classmethod(fake_from_credentials))
    env = {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}

    factory.build_broker("s", env=env)

    assert captured["trading"].api_key == "k"
    assert captured["data"].api_key == "dk"
    with pytest.raises(ConfigurationError, match="ALPACA_NEWS_API_KEY"):
        captured["news"]()  # read lazily: missing news credentials only fail when news is used


def test_the_ibkr_builder_passes_settings_and_alpaca_data_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    captured: dict[str, object] = {}

    def fake_from_settings(cls, strategy_name, settings, *, data, news, connection=None, market_data=None):
        captured.update(settings=settings, data=data, news=news)
        return _fake_broker(strategy_name)

    monkeypatch.setattr(IbkrBroker, "from_settings", classmethod(fake_from_settings))
    env = {"BROKER": "ibkr", "BROKER_API_IS_PAPER": "false", "IBKR_CLIENT_ID": "3", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}

    factory.build_broker("s", env=env)

    settings = captured["settings"]
    assert (settings.port, settings.client_id, settings.is_paper) == (4001, 3, False)
    assert captured["data"].api_key == "dk"


def test_the_ibkr_builder_does_not_need_alpaca_trading_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    monkeypatch.setattr(IbkrBroker, "from_settings", classmethod(lambda cls, name, settings, **kwargs: _fake_broker(name)))

    broker = factory.build_broker("s", env={"BROKER": "ibkr", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"})

    assert broker.strategy_name == "s"

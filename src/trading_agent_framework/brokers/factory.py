"""Paper/live broker selection from the strategy env file (`BROKER=alpaca|ibkr`).

Backtests never come here (`main.py` gives them a `PlaceholderBroker`). Broker classes are
imported inside the builders, so importing this module stays as light as `brokers/__init__.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import AlpacaCredentials, BrokerKind, BrokerSettings, IbkrSettings
from trading_agent_framework.utils.errors import ConfigurationError

BrokerBuilder = Callable[[str, BrokerSettings, Mapping[str, str] | None], Broker]


def _build_alpaca(strategy_name: str, settings: BrokerSettings, env: Mapping[str, str] | None) -> Broker:
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker

    return AlpacaBroker.from_credentials(
        strategy_name,
        trading=AlpacaCredentials.for_trading(env),
        data=AlpacaCredentials.for_data(env),
        news=lambda: AlpacaCredentials.for_news(env),
    )


def _build_ibkr(strategy_name: str, settings: BrokerSettings, env: Mapping[str, str] | None) -> Broker:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    return IbkrBroker.from_settings(
        strategy_name,
        IbkrSettings.from_env(settings.is_paper, env),
        data=AlpacaCredentials.for_data(env),
        news=lambda: AlpacaCredentials.for_news(env),
    )


BUILDERS: dict[BrokerKind, BrokerBuilder] = {
    BrokerKind.ALPACA: _build_alpaca,
    BrokerKind.IBKR: _build_ibkr,
}


def build_broker(strategy_name: str, *, env: Mapping[str, str] | None = None) -> Broker:
    """The paper/live broker named by `BROKER` (default `alpaca`), connected and checked."""
    settings = BrokerSettings.from_env(env)
    builder = BUILDERS.get(settings.kind)
    if builder is None:
        raise ConfigurationError(f"BROKER={settings.kind.value} is not supported yet")
    return builder(strategy_name, settings, env)

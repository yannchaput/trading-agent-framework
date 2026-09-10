"""Trivial factory functions for building Alpaca SDK clients from credentials.

Kept deliberately trivial: `AlpacaBroker` never builds its own client, so
tests can inject a mock/fake in its place. Both functions import `alpaca`
inside the function body, so importing this module doesn't force an `alpaca`
import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_agent_framework.config.env import AlpacaCredentials

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream


def build_trading_client(creds: AlpacaCredentials) -> TradingClient:
    from alpaca.trading.client import TradingClient

    return TradingClient(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper)


def build_trading_stream(creds: AlpacaCredentials) -> TradingStream:
    from alpaca.trading.stream import TradingStream

    return TradingStream(api_key=creds.api_key, secret_key=creds.api_secret, paper=creds.is_paper)

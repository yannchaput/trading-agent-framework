from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trading_agent_framework.brokers.alpaca.client import (
    build_trading_client,
    build_trading_stream,
)
from trading_agent_framework.config.env import AlpacaCredentials


@pytest.mark.parametrize("is_paper", [True, False])
def test_build_trading_client_threads_paper_flag(monkeypatch, is_paper: bool) -> None:
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=is_paper)
    mock_trading_client = MagicMock()
    monkeypatch.setattr("alpaca.trading.client.TradingClient", mock_trading_client)

    result = build_trading_client(creds)

    mock_trading_client.assert_called_once_with(api_key="key", secret_key="secret", paper=is_paper)
    assert result is mock_trading_client.return_value


@pytest.mark.parametrize("is_paper", [True, False])
def test_build_trading_stream_threads_paper_flag(monkeypatch, is_paper: bool) -> None:
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=is_paper)
    mock_trading_stream = MagicMock()
    monkeypatch.setattr("alpaca.trading.stream.TradingStream", mock_trading_stream)

    result = build_trading_stream(creds)

    mock_trading_stream.assert_called_once_with(api_key="key", secret_key="secret", paper=is_paper)
    assert result is mock_trading_stream.return_value

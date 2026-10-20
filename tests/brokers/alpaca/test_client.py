from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests
from requests.adapters import HTTPAdapter

from trading_agent_framework.brokers.alpaca.client import (
    build_news_client,
    build_stock_data_client,
    build_trading_client,
    build_trading_stream,
)
from trading_agent_framework.config.env import AlpacaCredentials

_CREDS = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)


def _capture_send(monkeypatch) -> list[dict]:
    """Replace the real transport: record the kwargs each send gets and answer 200 with an empty JSON body."""
    calls: list[dict] = []

    def fake_send(self, request, **kwargs):
        calls.append(kwargs)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    monkeypatch.setattr(HTTPAdapter, "send", fake_send)
    return calls


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


def test_build_stock_data_client_uses_the_same_credentials(monkeypatch) -> None:
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)
    mock_data_client = MagicMock()
    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", mock_data_client)

    result = build_stock_data_client(creds)

    mock_data_client.assert_called_once_with(api_key="key", secret_key="secret")
    assert result is mock_data_client.return_value


def test_build_news_client_uses_the_same_credentials(monkeypatch) -> None:
    from trading_agent_framework.brokers.alpaca.client import build_news_client

    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)
    mock_news_client = MagicMock()
    monkeypatch.setattr("alpaca.data.historical.news.NewsClient", mock_news_client)

    result = build_news_client(creds)

    mock_news_client.assert_called_once_with(api_key="key", secret_key="secret")
    assert result is mock_news_client.return_value


@pytest.mark.parametrize("build", [build_trading_client, build_stock_data_client, build_news_client])
@pytest.mark.parametrize("url", ["https://data.alpaca.markets/v1beta1/news", "http://localhost/v2/account"])
def test_built_clients_send_requests_with_the_timeout(monkeypatch, build, url: str) -> None:
    calls = _capture_send(monkeypatch)
    client = build(_CREDS)

    client._session.request("GET", url)

    assert calls[-1]["timeout"] == 30.0


def test_built_clients_keep_an_explicit_timeout(monkeypatch) -> None:
    calls = _capture_send(monkeypatch)
    client = build_news_client(_CREDS)

    client._session.request("GET", "https://data.alpaca.markets/v1beta1/news", timeout=5)

    assert calls[-1]["timeout"] == 5

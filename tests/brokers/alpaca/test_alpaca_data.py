from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from tests.fakes import (
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    bar_payload,
    et,
    make_alpaca_calendar,
    make_alpaca_trade,
)

from trading_agent_framework.brokers.alpaca import data as data_module
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BrokerError

AAPL = Asset("AAPL")
_NOW = et(2026, 9, 10, 9, 32)


def test_get_bars_ends_at_the_given_end_and_windows_on_the_calendar_client() -> None:
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAPL": [bar_payload("2026-09-09T04:00:00Z", 100.0), bar_payload("2026-09-10T04:00:00Z", 101.0)]}
    calendar = FakeTradingClient()
    calendar.calendar_response = [make_alpaca_calendar("2026-09-09"), make_alpaca_calendar("2026-09-10")]

    bars = AlpacaMarketData(data, calendar).get_bars([AAPL], 2, "day", end=_NOW)

    assert AAPL in bars
    [request] = data.bars_requests
    # alpaca-py's BaseTimeseriesDataRequest.__init__ converts a tz-aware end/start to naive UTC
    # before storing it, so the request never keeps _NOW's tzinfo; compare against that UTC value.
    assert request.end == datetime(2026, 9, 10, 13, 32)
    assert calendar.calendar_requests  # the session window came from the calendar client


def test_get_last_price_reads_the_latest_trade() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    assert AlpacaMarketData(data, FakeTradingClient()).get_last_price(AAPL) == Decimal("100.15")


def test_market_data_without_a_data_client_raises() -> None:
    with pytest.raises(BrokerError, match="no market data client"):
        AlpacaMarketData(None, FakeTradingClient()).get_last_price(AAPL)


def test_from_credentials_builds_both_clients_from_the_data_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, AlpacaCredentials]] = []
    calendar = FakeTradingClient()

    def fake_data(creds: AlpacaCredentials) -> FakeStockHistoricalDataClient:
        seen.append(("data", creds))
        return FakeStockHistoricalDataClient()

    def fake_trading(creds: AlpacaCredentials) -> FakeTradingClient:
        seen.append(("calendar", creds))
        return calendar

    monkeypatch.setattr(data_module, "build_stock_data_client", fake_data)
    monkeypatch.setattr(data_module, "build_trading_client", fake_trading)
    creds = AlpacaCredentials(api_key="dk", api_secret="ds", is_paper=True)

    market_data = AlpacaMarketData.from_credentials(creds)

    assert seen == [("data", creds), ("calendar", creds)]
    assert market_data.calendar_client is calendar

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal

import pytest
from alpaca.data.enums import DataFeed
from tests.fakes import (
    FakeClock,
    FakeNewsClient,
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    bar_payload,
    et,
    make_alpaca_calendar,
    make_alpaca_news_article,
    make_alpaca_quote,
    make_alpaca_trade,
)

from trading_agent_framework.brokers.alpaca import broker as broker_module
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.errors import BrokerError

AAPL = Asset("AAPL")
MSFT = Asset("MSFT")
_NOW = et(2026, 9, 10, 9, 32)  # Thursday, two minutes after the open
_MINUTES = {
    "AAPL": [
        bar_payload("2026-09-10T13:29:00Z", 99.0),  # 09:29, pre-market
        bar_payload("2026-09-10T13:30:00Z", 100.0),
        bar_payload("2026-09-10T13:31:00Z", 101.0),
    ]
}


def _broker(
    data: FakeStockHistoricalDataClient | None, trading: FakeTradingClient | None = None
) -> AlpacaBroker:
    return AlpacaBroker(
        "momentum", trading or FakeTradingClient(), clock=FakeClock(_NOW), data_client=data
    )


def _trading_with_calendar() -> FakeTradingClient:
    trading = FakeTradingClient()
    trading.calendar_response = [make_alpaca_calendar("2026-09-09"), make_alpaca_calendar("2026-09-10")]
    return trading


def _data_with_minutes() -> FakeStockHistoricalDataClient:
    data = FakeStockHistoricalDataClient()
    data.bars = _MINUTES
    return data


# --- last prices and quotes ------------------------------------------------------


def test_get_last_prices_fetches_every_symbol_in_one_iex_request() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    prices = _broker(data).get_last_prices([AAPL, MSFT])

    assert prices == {AAPL: Decimal("100.15"), MSFT: None}
    [request] = data.trade_requests
    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.feed == DataFeed.IEX


def test_get_last_prices_splits_large_universes_into_150_symbol_requests() -> None:
    data = FakeStockHistoricalDataClient()

    prices = _broker(data).get_last_prices([Asset(f"S{i}") for i in range(151)])

    assert len(prices) == 151
    assert [len(r.symbol_or_symbols) for r in data.trade_requests] == [150, 1]


def test_get_last_price_returns_the_single_price() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 99.5)}

    assert _broker(data).get_last_price(AAPL) == Decimal("99.5")


def test_get_quote_returns_bid_and_ask_from_the_iex_feed() -> None:
    data = FakeStockHistoricalDataClient()
    data.quotes = {"AAPL": make_alpaca_quote(bid=100.1, ask=100.2)}

    quote = _broker(data).get_quote(AAPL)

    assert isinstance(quote, Quote)
    assert quote.mid == Decimal("100.15")
    [request] = data.quote_requests
    assert request.feed == DataFeed.IEX


def test_get_quote_is_none_when_alpaca_has_no_quote() -> None:
    assert _broker(FakeStockHistoricalDataClient()).get_quote(AAPL) is None


# --- bars ------------------------------------------------------------------------


def test_get_bars_starts_the_window_at_the_earliest_session_needed() -> None:
    data = _data_with_minutes()
    trading = _trading_with_calendar()

    _broker(data, trading).get_bars([AAPL, MSFT], 30, "minute")

    [calendar_request] = trading.calendar_requests
    assert (calendar_request.start, calendar_request.end) == (date(2026, 8, 28), date(2026, 9, 10))
    [bars_request] = data.bars_requests
    assert bars_request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert bars_request.timeframe.value == "1Min"
    # Midnight 2026-09-09 and 09:32 2026-09-10 in New York, stored by alpaca-py as naive UTC.
    assert bars_request.start == datetime(2026, 9, 9, 4, 0)
    assert bars_request.end == datetime(2026, 9, 10, 13, 32)


def test_get_bars_keeps_extended_hours_by_default() -> None:
    result = _broker(_data_with_minutes(), _trading_with_calendar()).get_bars([AAPL], 30, "minute")

    assert list(result[AAPL].df["close"]) == [99.0, 100.0, 101.0]


def test_get_bars_drops_extended_hours_on_request() -> None:
    broker = _broker(_data_with_minutes(), _trading_with_calendar())

    result = broker.get_bars([AAPL], 30, "minute", include_after_hours=False)

    assert list(result[AAPL].df["close"]) == [100.0, 101.0]


def test_get_bars_leaves_out_assets_without_data() -> None:
    broker = _broker(_data_with_minutes(), _trading_with_calendar())

    assert set(broker.get_bars([AAPL, MSFT], 30, "minute")) == {AAPL}


def test_get_bars_rejects_an_unknown_timestep_before_any_request() -> None:
    data = _data_with_minutes()
    trading = _trading_with_calendar()

    with pytest.raises(ValueError, match="Unsupported timestep"):
        _broker(data, trading).get_bars([AAPL], 5, "hour")

    assert trading.calendar_requests == []
    assert data.bars_requests == []


def test_get_bars_with_no_assets_makes_no_network_calls() -> None:
    data = _data_with_minutes()
    trading = _trading_with_calendar()

    assert _broker(data, trading).get_bars([], 5, "day") == {}

    assert trading.calendar_requests == []
    assert data.bars_requests == []


# --- failures --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "call", "match"),
    [
        ("get_stock_latest_trade", lambda b: b.get_last_prices([AAPL]), "latest trades"),
        ("get_stock_latest_quote", lambda b: b.get_quote(AAPL), "latest quote"),
        ("get_stock_bars", lambda b: b.get_bars([AAPL], 5, "day"), "bars"),
    ],
)
def test_data_client_failures_become_broker_errors(
    method: str, call: Callable[[AlpacaBroker], object], match: str
) -> None:
    data = FakeStockHistoricalDataClient()
    data.raises[method] = RuntimeError("boom")

    with pytest.raises(BrokerError, match=match):
        call(_broker(data, _trading_with_calendar()))


def test_calendar_failures_become_broker_errors() -> None:
    trading = FakeTradingClient()
    trading.raises["get_calendar"] = RuntimeError("boom")

    with pytest.raises(BrokerError, match="calendar"):
        _broker(FakeStockHistoricalDataClient(), trading).get_bars([AAPL], 5, "day")


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.get_last_price(AAPL),
        lambda b: b.get_last_prices([AAPL]),
        lambda b: b.get_quote(AAPL),
        lambda b: b.get_bars([AAPL], 5, "day"),
    ],
)
def test_market_data_without_a_data_client_raises(call: Callable[[AlpacaBroker], object]) -> None:
    with pytest.raises(BrokerError, match="no market data client"):
        call(_broker(None))


def test_from_credentials_wires_the_stock_data_client(monkeypatch: pytest.MonkeyPatch) -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 101.0)}
    monkeypatch.setattr(broker_module, "build_trading_client", lambda creds: FakeTradingClient())
    monkeypatch.setattr(broker_module, "build_stock_data_client", lambda creds: data)
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)

    broker = AlpacaBroker.from_credentials("momentum", creds, with_stream=False)

    assert broker.get_last_price(AAPL) == Decimal("101.0")


# --- news -------------------------------------------------------------------


def _broker_with_news(news: FakeNewsClient) -> AlpacaBroker:
    return AlpacaBroker(
        "momentum", FakeTradingClient(), clock=FakeClock(_NOW), news_client=news
    )


def test_get_news_builds_a_request_and_parses_the_response() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    broker = _broker_with_news(news)

    articles = broker.get_news(["SPY"], start=None, end=_NOW, limit=5, include_content=False)

    assert articles[0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY"


def test_get_news_without_a_configured_client_raises() -> None:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_NOW))

    with pytest.raises(BrokerError, match="no news client configured"):
        broker.get_news(end=_NOW)


def test_get_news_wraps_client_failures_as_broker_error() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("rate limited")
    broker = _broker_with_news(news)

    with pytest.raises(BrokerError, match="Failed to fetch news"):
        broker.get_news(end=_NOW)


def test_news_provider_is_available_when_a_news_client_is_configured() -> None:
    broker = _broker_with_news(FakeNewsClient())

    assert broker.news_provider() is not None


def test_news_provider_is_none_without_a_news_client() -> None:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_NOW))

    assert broker.news_provider() is None

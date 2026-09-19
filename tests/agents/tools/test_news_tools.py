from __future__ import annotations

from datetime import UTC, datetime

from tests.fakes import FakeClock, FakeNewsClient, FakeTradingClient, et, make_alpaca_news_article

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.core.strategy import Strategy


def _now() -> datetime:
    return et(2026, 9, 14, 10)


def _alpaca_strategy(news: FakeNewsClient) -> Strategy:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_now()), news_client=news)
    return Strategy(broker)


def _tool(strategy: Strategy):
    [tool] = news_tools(strategy)
    return tool


def test_search_news_returns_articles_from_the_broker() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    tool = _tool(_alpaca_strategy(news))

    result = tool(symbols="SPY,QQQ")

    assert result["count"] == 1
    assert result["articles"][0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY,QQQ"


def test_search_news_defaults_end_to_the_strategy_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool()

    [request] = news.news_requests
    # alpaca-py normalises start/end to naive UTC (same as test_market_data_requests.py's
    # test_bars_request_asks_for_adjusted_iex_bars_of_every_symbol)
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_clamps_end_that_is_after_the_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool(end=et(2026, 9, 20).isoformat())

    [request] = news.news_requests
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_returns_an_error_dict_on_broker_failure() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("boom")
    tool = _tool(_alpaca_strategy(news))

    result = tool()

    assert "error" in result


def test_search_news_without_an_alpaca_broker_returns_an_error() -> None:
    class _OtherBroker(Broker):
        name = "other"
        def _conform_order(self, order): return order
        def _submit_order(self, order): return order
        def cancel_order(self, order): pass
        def pull_order(self, identifier): return None
        def pull_orders(self, limit=100): return []
        def pull_positions(self): return []
        def get_account(self): raise NotImplementedError
        def modify_order(self, order, *, limit_price=None, stop_price=None): raise NotImplementedError
        def close_position(self, asset, fraction=1): return None
        def close_all_positions(self, cancel_orders=True): return []
        def sync_open_orders(self): return []
        def get_last_price(self, asset): return None
        def get_last_prices(self, assets): return {}
        def get_quote(self, asset): return None
        def get_bars(self, assets, length, timestep="day", *, include_after_hours=True): return {}

    strategy = Strategy(_OtherBroker("momentum", clock=FakeClock(_now())))
    tool = _tool(strategy)

    assert tool() == {"error": "news requires an Alpaca broker"}

from __future__ import annotations

from datetime import UTC, datetime

from tests.fakes import FakeBroker, FakeClock, FakeNewsClient, FakeTradingClient, et, make_alpaca_news_article

from trading_agent_framework.agents.tools.news import MAX_CONTENT_LIMIT, news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.utils.errors import BrokerError


def _now() -> datetime:
    return et(2026, 9, 14, 10)


def _alpaca_strategy(news: FakeNewsClient) -> Strategy:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_now()), news_client=news)
    return Strategy(broker)


def _tool(strategy: Strategy):
    [tool] = news_tools(strategy)
    return tool


class _StubProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_news(self, symbols=(), *, start=None, end, limit=10, include_content=False):
        self.calls.append({"symbols": list(symbols), "start": start, "end": end, "limit": limit, "include_content": include_content})
        return [{"id": 1, "headline": "Stub headline"}]


def _provider_strategy(provider: object = None, *, lookup_error: BrokerError | None = None) -> Strategy:
    class _Broker(FakeBroker):
        def news_provider(self):
            if lookup_error is not None:
                raise lookup_error
            return provider

    return Strategy(_Broker(FakeClock(_now())))


def test_search_news_returns_articles_from_the_broker() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    tool = _tool(_alpaca_strategy(news))

    result = tool(symbols="SPY,QQQ")

    assert result["count"] == 1
    assert result["articles"][0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY,QQQ"


def test_search_news_works_with_any_broker_that_provides_news() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))

    result = tool(symbols="SPY")

    assert result == {"count": 1, "articles": [{"id": 1, "headline": "Stub headline"}]}
    assert provider.calls[0]["symbols"] == ["SPY"]
    assert provider.calls[0]["end"] == _now()


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


def test_search_news_clamps_limit_to_the_content_cap_when_content_is_requested() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))

    tool(limit=30, include_content=True)
    tool(limit=30, include_content=False)

    assert provider.calls[0]["limit"] == MAX_CONTENT_LIMIT
    assert provider.calls[1]["limit"] == 30


def test_search_news_returns_an_error_dict_on_broker_failure() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("boom")
    tool = _tool(_alpaca_strategy(news))

    result = tool()

    assert "error" in result


def test_search_news_without_a_news_provider_returns_an_error() -> None:
    strategy = Strategy(FakeBroker(FakeClock(_now())))

    assert _tool(strategy)() == {"error": "no news provider for this broker"}


def test_search_news_reports_a_failing_provider_lookup_as_an_error() -> None:
    strategy = _provider_strategy(lookup_error=BrokerError("no news source configured"))

    assert _tool(strategy)() == {"error": "no news source configured"}


def test_search_news_returns_an_error_on_naive_datetime_end() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    # Naive datetime strings (no offset) should return error, not raise TypeError
    result = tool(end="2026-09-20T00:00:00")

    assert "error" in result


def test_search_news_returns_an_error_on_naive_datetime_start() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    # Naive datetime strings (no offset) should return error, not raise TypeError
    result = tool(start="2026-09-10T00:00:00")

    assert "error" in result


# --- per-run cache: the model re-sent the identical scan (31 times in one backtest) ----------------


def test_an_identical_search_within_one_run_reuses_the_first_result() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))
    with agent_call_context(run_id="run-1"):
        first = tool(symbols="SPY,QQQ", limit=30)
        again = tool(symbols="SPY,QQQ", limit=30)

    assert again == first
    assert len(provider.calls) == 1


def test_a_different_search_or_a_later_run_queries_the_provider_again() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))
    with agent_call_context(run_id="run-1"):
        tool(symbols="SPY,QQQ", limit=30)
        tool(symbols="SPY,QQQ", limit=3, include_content=True)
    with agent_call_context(run_id="run-2"):
        tool(symbols="SPY,QQQ", limit=30)

    assert len(provider.calls) == 3


def test_a_failed_search_is_not_cached() -> None:
    class _FlakyProvider(_StubProvider):
        def get_news(self, symbols=(), **kwargs):
            if not self.calls:
                self.calls.append({})
                raise BrokerError("news API down")
            return super().get_news(symbols, **kwargs)

    provider = _FlakyProvider()
    tool = _tool(_provider_strategy(provider))
    with agent_call_context(run_id="run-1"):
        assert "error" in tool(symbols="SPY")
        assert tool(symbols="SPY")["count"] == 1


def test_outside_an_agent_run_every_search_queries_the_provider() -> None:
    provider = _StubProvider()
    tool = _tool(_provider_strategy(provider))
    tool(symbols="SPY")
    tool(symbols="SPY")

    assert len(provider.calls) == 2

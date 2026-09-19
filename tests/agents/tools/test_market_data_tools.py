from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def _tools(strategy: Strategy) -> dict[str, Callable[..., dict[str, Any]]]:
    return {tool.__name__: tool for tool in market_data_tools(strategy)}  # ty: ignore[unresolved-attribute]


def test_returns_three_tools_with_one_line_docstrings() -> None:
    tools = market_data_tools(_strategy()[0])
    assert [t.__name__ for t in tools] == ["get_last_price", "get_quote", "get_bars"]  # ty: ignore[unresolved-attribute]
    for tool in tools:
        assert len(tool.__doc__.splitlines()) == 1  # ty: ignore[unresolved-attribute]


def test_get_last_price_returns_the_price() -> None:
    strategy, broker = _strategy()
    broker.last_prices["SPY"] = Decimal("450.10")
    tools = _tools(strategy)

    assert tools["get_last_price"]("SPY") == {"symbol": "SPY", "price": 450.10}


def test_get_last_price_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_last_price"]("SPY") == {"error": "no trade data for 'SPY'"}


def test_get_quote_returns_bid_ask_and_mid() -> None:
    strategy, broker = _strategy()
    broker.quotes["SPY"] = Quote(
        asset=Asset("SPY"), bid=Decimal("450"), ask=Decimal("451"),
        bid_size=None, ask_size=None, timestamp=datetime(2026, 9, 14, 14, tzinfo=UTC),
    )
    tools = _tools(strategy)

    result = tools["get_quote"]("SPY")

    assert (result["bid"], result["ask"], result["mid"]) == (450.0, 451.0, 450.5)
    assert result["timestamp"] == "2026-09-14T14:00:00+00:00"


def test_get_quote_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_quote"]("SPY") == {"error": "no quote for 'SPY'"}


def test_get_bars_returns_oldest_first_ohlcv_rows() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["SPY"] = make_bars_frame([100.0, 101.0, 102.0], start=et(2026, 9, 10))
    tools = _tools(strategy)

    result = tools["get_bars"]("SPY", length=3)

    assert result["symbol"] == "SPY"
    assert result["timestep"] == "day"
    assert [row["close"] for row in result["bars"]] == [100.0, 101.0, 102.0]
    assert set(result["bars"][0]) == {"date", "open", "high", "low", "close", "volume"}


def test_get_bars_clamps_length_to_the_allowed_range() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["SPY"] = make_bars_frame([100.0] * 250, start=et(2026, 9, 10))
    tools = _tools(strategy)

    tools["get_bars"]("SPY", length=1000)

    [(_, length, _, _)] = broker.bars_calls
    assert length == 200


def test_get_bars_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_bars"]("SPY") == {"error": "no bars for 'SPY'"}

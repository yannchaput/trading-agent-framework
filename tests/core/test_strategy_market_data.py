from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def test_get_last_price_accepts_a_symbol() -> None:
    strategy, broker = _strategy()
    broker.last_prices["SPY"] = Decimal("450.10")

    assert strategy.get_last_price("SPY") == Decimal("450.10")


def test_get_last_price_is_none_without_data() -> None:
    strategy, _ = _strategy()

    assert strategy.get_last_price("SPY") is None


def test_get_last_prices_keys_results_by_asset() -> None:
    strategy, broker = _strategy()
    broker.last_prices = {"SPY": Decimal("450.10"), "QQQ": Decimal("380.5")}

    prices = strategy.get_last_prices(["SPY", Asset("QQQ"), "TLT"])

    assert prices == {
        Asset("SPY"): Decimal("450.10"),
        Asset("QQQ"): Decimal("380.5"),
        Asset("TLT"): None,
    }


def test_get_quote_delegates_to_the_broker() -> None:
    strategy, broker = _strategy()
    quote = Quote(
        asset=Asset("SPY"),
        bid=Decimal(1),
        ask=Decimal(2),
        bid_size=None,
        ask_size=None,
        timestamp=datetime(2026, 9, 14, 14, tzinfo=UTC),
    )
    broker.quotes["SPY"] = quote

    assert strategy.get_quote("SPY") is quote


def test_get_historical_prices_returns_the_last_length_bars() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["AAPL"] = make_bars_frame([1, 2, 3, 4])

    bars = strategy.get_historical_prices("AAPL", 2, "day")

    assert bars is not None
    assert bars.asset == Asset("AAPL")
    assert list(bars.df["close"]) == [3.0, 4.0]
    assert broker.bars_calls == [(("AAPL",), 2, "day", True)]


def test_get_historical_prices_is_none_without_data() -> None:
    strategy, _ = _strategy()

    assert strategy.get_historical_prices("AAPL", 2) is None


def test_get_historical_prices_for_assets_forwards_every_option() -> None:
    strategy, broker = _strategy()
    broker.bar_frames = {"AAPL": make_bars_frame([1, 2]), "MSFT": make_bars_frame([3, 4])}

    result = strategy.get_historical_prices_for_assets(
        ["AAPL", Asset("MSFT"), "TLT"], 2, "minute", include_after_hours=False
    )

    assert set(result) == {Asset("AAPL"), Asset("MSFT")}
    assert broker.bars_calls == [(("AAPL", "MSFT", "TLT"), 2, "minute", False)]


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get_last_price("SPY"),
        lambda s: s.get_last_prices(["SPY"]),
        lambda s: s.get_quote("SPY"),
        lambda s: s.get_historical_prices("SPY", 5),
        lambda s: s.get_historical_prices_for_assets(["SPY"], 5),
    ],
)
def test_broker_errors_reach_the_strategy(call: Callable[[Strategy], object]) -> None:
    strategy, broker = _strategy()
    broker.market_data_error = BrokerError("feed down")

    with pytest.raises(BrokerError, match="feed down"):
        call(strategy)

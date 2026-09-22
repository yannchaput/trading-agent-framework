from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError

AAPL = Asset("AAPL")


def _order() -> Order:
    return Order("s", AAPL, OrderSide.BUY, quantity=Decimal(1))


def test_it_carries_the_strategy_name_and_a_simulated_clock() -> None:
    broker = PlaceholderBroker("news_binary")

    assert broker.strategy_name == "news_binary"
    assert isinstance(broker.clock, BacktestClock)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.submit_order(_order()),
        lambda b: b.cancel_order(_order()),
        lambda b: b.pull_order("x"),
        lambda b: b.pull_orders(),
        lambda b: b.pull_positions(),
        lambda b: b.get_account(),
        lambda b: b.modify_order(_order(), limit_price=Decimal(1)),
        lambda b: b.close_position(AAPL),
        lambda b: b.close_all_positions(),
        lambda b: b.sync_open_orders(),
        lambda b: b.get_last_price(AAPL),
        lambda b: b.get_last_prices([AAPL]),
        lambda b: b.get_quote(AAPL),
        lambda b: b.get_bars([AAPL], 5),
    ],
)
def test_every_operation_raises(call) -> None:
    with pytest.raises(BrokerError, match="not available before the backtest starts"):
        call(PlaceholderBroker("s"))

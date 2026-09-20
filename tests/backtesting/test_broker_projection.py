"""Submission-time rejection and `buying_power`, both driven by `BacktestBroker._projection`:
what cash and holdings will be once every still-pending order has filled.

Backtest fills land a session after submission, so a strategy that reads only settled `cash`
double-spends it, and the fill-time rejection it earns arrives an iteration too late to react
to. That matters most for an LLM agent, whose only feedback channel is the tool's return value.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

AAPL = Asset("AAPL")
MSFT = Asset("MSFT")  # deliberately never given bars: the unpriceable case
DAY1 = datetime(2026, 1, 5, 16, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
DAY3 = DAY2 + timedelta(days=1)


def _account(budget: Decimal) -> tuple[BacktestBroker, BacktestClock]:
    """Flat $100 bars, so an estimate at submission equals the price it fills at."""
    source = FakeBacktestDataSource()
    source.set_bars(AAPL, make_close_indexed_frame([100.0, 100.0, 100.0], start=DAY1))
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=budget)
    clock.on_advance = broker.on_advance
    return broker, clock


def _order(side: OrderSide, quantity: int, asset: Asset = AAPL) -> Order:
    return Order(strategy_name="momentum", asset=asset, side=side, quantity=Decimal(quantity))


def _advance(broker: BacktestBroker, clock: BacktestClock, previous: datetime, now: datetime) -> None:
    clock._now = now  # test-only direct time jump; production code goes through clock.wait()
    broker.on_advance(previous, now)


def _holding(quantity: int, budget: Decimal = Decimal(10000)) -> tuple[BacktestBroker, BacktestClock]:
    """A broker holding `quantity` AAPL, bought and filled, with nothing left pending."""
    broker, clock = _account(budget)
    broker.submit_order(_order(OrderSide.BUY, quantity))
    _advance(broker, clock, DAY1, DAY2)
    assert broker._pending == {}
    return broker, clock


# --- sells -----------------------------------------------------------------------------


def test_selling_more_than_the_holding_is_rejected_at_submission() -> None:
    broker, _ = _holding(10)

    with pytest.raises(OrderValidationError, match="insufficient position: selling 15 AAPL, available 10"):
        broker.submit_order(_order(OrderSide.SELL, 15))


def test_selling_what_is_held_is_accepted() -> None:
    broker, _ = _holding(10)

    order = broker.submit_order(_order(OrderSide.SELL, 10))

    assert order.status is OrderStatus.NEW


def test_selling_shares_a_pending_sell_has_already_claimed_is_rejected() -> None:
    # The Jan-3 SHV case from the news_binary backtest: the holding still reads 10 because the
    # first sell has not filled yet, but those shares are spoken for.
    broker, _ = _holding(10)
    broker.submit_order(_order(OrderSide.SELL, 6))

    with pytest.raises(OrderValidationError, match="available 4 \\(10 held, the rest already pending sale\\)"):
        broker.submit_order(_order(OrderSide.SELL, 5))


def test_selling_a_symbol_that_is_not_held_at_all_is_rejected() -> None:
    broker, _ = _account(Decimal(10000))

    with pytest.raises(OrderValidationError, match="selling 1 AAPL, available 0"):
        broker.submit_order(_order(OrderSide.SELL, 1))


# --- buys ------------------------------------------------------------------------------


def test_a_buy_whose_cash_a_pending_buy_has_already_committed_is_rejected() -> None:
    # The Jan-6 SPY case: settled cash still reads the full budget because the first buy has
    # not filled, so sizing against `cash` spends the same money twice.
    broker, _ = _account(Decimal(1000))
    broker.submit_order(_order(OrderSide.BUY, 9))  # ~900 of the 1000 committed

    with pytest.raises(OrderValidationError, match="insufficient buying power: buying 9 AAPL"):
        broker.submit_order(_order(OrderSide.BUY, 9))


def test_a_buy_funded_by_a_pending_sell_is_accepted() -> None:
    """The rebalance case: pending sells are credited, because fills happen in submission order."""
    broker, clock = _holding(10, budget=Decimal(1000))
    assert broker._cash == Decimal(0)

    sell = broker.submit_order(_order(OrderSide.SELL, 10))
    buy = broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY2, DAY3)

    assert (sell.status, buy.status) == (OrderStatus.FILL, OrderStatus.FILL)


def test_an_order_whose_price_cannot_be_estimated_is_left_to_the_fill_time_check() -> None:
    broker, _ = _account(Decimal(1))

    order = broker.submit_order(_order(OrderSide.BUY, 1000, asset=MSFT))

    assert order.status is OrderStatus.NEW


def test_a_rejected_submission_leaves_no_trace() -> None:
    broker, _ = _account(Decimal(100))
    order = _order(OrderSide.BUY, 1000)

    with pytest.raises(OrderValidationError):
        broker.submit_order(order)

    assert broker._pending == {}
    assert broker.tracker.get_all_tracked_orders() == []
    assert broker._cash == Decimal(100)


# --- balances --------------------------------------------------------------------------


def test_buying_power_excludes_cash_a_pending_buy_has_committed() -> None:
    broker, _ = _account(Decimal(1000))
    assert broker.get_account().buying_power == Decimal(1000)

    broker.submit_order(_order(OrderSide.BUY, 4))

    assert broker.get_account().buying_power == Decimal(600)


def test_buying_power_credits_a_pending_sell() -> None:
    broker, _ = _holding(10, budget=Decimal(1000))
    assert broker.get_account().buying_power == Decimal(0)

    broker.submit_order(_order(OrderSide.SELL, 10))

    assert broker.get_account().buying_power == Decimal(1000)


def test_cash_and_portfolio_value_stay_settled_while_an_order_is_pending() -> None:
    # `Strategy.get_cash()` reads `cash`, and `cross_momentum.rebalance` adds its own estimate of
    # pending sell proceeds to it -- netting `cash` here would make that strategy double-count.
    broker, _ = _account(Decimal(1000))
    before = broker.get_account()

    broker.submit_order(_order(OrderSide.BUY, 4))
    after = broker.get_account()

    assert after.cash == before.cash == Decimal(1000)
    assert after.portfolio_value == before.portfolio_value == Decimal(1000)


def test_buying_power_returns_to_cash_once_the_order_fills() -> None:
    broker, clock = _account(Decimal(1000))
    broker.submit_order(_order(OrderSide.BUY, 4))

    _advance(broker, clock, DAY1, DAY2)

    account = broker.get_account()
    assert account.cash == Decimal(600)
    assert account.buying_power == Decimal(600)

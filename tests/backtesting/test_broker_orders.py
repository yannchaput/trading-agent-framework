from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BacktestError, OrderValidationError

AAPL = Asset("AAPL")
NOW = datetime(2026, 1, 5, 16, tzinfo=UTC)


def _broker(budget: Decimal = Decimal(10000)) -> BacktestBroker:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=NOW)
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=NOW, sessions=[])
    return BacktestBroker("momentum", data_source=source, clock=clock, budget=budget)


def test_submitting_a_market_order_tracks_it_as_new_and_pending() -> None:
    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))

    result = broker.submit_order(order)

    assert result.status is OrderStatus.NEW
    assert result.client_order_id == f"momentum:{result.identifier}"
    assert result in broker.tracker.new.snapshot()


def test_submitting_a_notional_order_raises() -> None:
    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, notional=Decimal(1000))
    with pytest.raises(OrderValidationError, match="notional"):
        broker.submit_order(order)


def test_cancel_order_removes_it_from_pending_and_the_tracker() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    broker.cancel_order(order)

    assert order.status is OrderStatus.CANCELED
    assert order not in broker.tracker.new.snapshot()
    assert order in broker.tracker.canceled.snapshot()


def test_modify_order_updates_the_pending_orders_prices() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(
            strategy_name="momentum", asset=AAPL, side=OrderSide.BUY,
            quantity=Decimal(10), limit_price=Decimal(140),
        )
    )

    replacement = broker.modify_order(order, limit_price=Decimal(145))

    assert replacement.limit_price == Decimal(145)


def test_modify_order_returns_a_distinct_replacement_and_re_keys_pending() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(
            strategy_name="momentum", asset=AAPL, side=OrderSide.BUY,
            quantity=Decimal(10), limit_price=Decimal(140), stop_price=Decimal(130),
        )
    )
    old_identifier = order.identifier

    replacement = broker.modify_order(order, limit_price=Decimal(145))

    assert replacement is not order
    assert replacement.identifier != old_identifier
    assert replacement.limit_price == Decimal(145)
    assert replacement.stop_price == Decimal(130)

    # the new identifier is now the one findable via pull_order and as pending;
    # the old identifier is still tracked (now canceled, per mark_replaced) but
    # no longer resolves to a pending order.
    assert broker.pull_order(replacement.identifier) == replacement
    assert old_identifier not in broker._pending
    assert replacement.identifier in broker._pending

    # the old order object is no longer resolvable as pending under itself
    with pytest.raises(BacktestError):
        broker.modify_order(order, limit_price=Decimal(150))


def test_modify_order_on_an_unpending_order_raises() -> None:
    broker = _broker()
    order = Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    with pytest.raises(BacktestError):
        broker.modify_order(order, limit_price=Decimal(100))


def test_sync_open_orders_returns_nothing_for_a_fresh_backtest() -> None:
    assert _broker().sync_open_orders() == []


def test_pull_orders_and_pull_order_read_from_the_tracker() -> None:
    broker = _broker()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    assert broker.pull_order(order.identifier) == order
    assert order in broker.pull_orders()
    assert broker.pull_order("does-not-exist") is None


def test_get_last_price_returns_the_latest_visible_close() -> None:
    broker = _broker()
    assert broker.get_last_price(AAPL) == Decimal("150.0")


def test_get_last_price_returns_none_for_an_asset_with_no_data() -> None:
    broker = _broker()
    assert broker.get_last_price(Asset("MISSING")) is None


def test_get_quote_synthesises_bid_and_ask_from_the_last_close() -> None:
    broker = _broker()
    quote = broker.get_quote(AAPL)
    assert quote is not None
    assert quote.bid == quote.ask == Decimal("150.0")


def test_get_bars_delegates_to_the_data_source() -> None:
    broker = _broker()
    result = broker.get_bars([AAPL], 2, "day")
    assert AAPL in result
    assert len(result[AAPL].df) <= 2


def test_get_account_reports_cash_only_when_no_positions_are_held() -> None:
    broker = _broker(budget=Decimal(5000))
    account = broker.get_account()
    assert account.cash == Decimal(5000)
    assert account.portfolio_value == Decimal(5000)


def test_pull_positions_is_empty_for_a_fresh_backtest() -> None:
    assert _broker().pull_positions() == []


def test_close_position_with_no_position_returns_none() -> None:
    assert _broker().close_position(AAPL) is None


def test_start_and_stop_stream_are_no_ops() -> None:
    broker = _broker()
    broker.start_stream()
    broker.stop_stream()  # must not raise

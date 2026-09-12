from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order

AAPL = Asset("AAPL")
DAY1 = datetime(2026, 1, 5, 16, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
DAY3 = DAY2 + timedelta(days=1)


def _broker_with_two_bars(budget: Decimal = Decimal(10000)) -> BacktestBroker:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=budget)
    clock.on_advance = broker.on_advance
    return broker, clock, source


def test_market_order_fills_on_the_next_bar_not_the_submission_bar() -> None:
    broker, clock, _ = _broker_with_two_bars()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )

    # Advancing time without a new bar closing yet: still pending (no bar past DAY1 exists yet
    # at exactly DAY1's cutoff -- the fake's only bar closed *at* DAY1, same as last_evaluated).
    broker.on_advance(clock.now(), clock.now())
    assert order.status is OrderStatus.NEW

    # Simulate the clock reaching DAY2, where the second bar has closed.
    clock._now = DAY2  # test-only direct time jump; production code goes through clock.wait()
    broker.on_advance(DAY1, DAY2)

    assert order.status is OrderStatus.FILL
    assert order.avg_fill_price == Decimal("151.0")  # bar 2's open == its close in this fixture
    assert order.identifier not in broker._pending


def test_fill_updates_cash_and_creates_a_long_position() -> None:
    broker, clock, _ = _broker_with_two_bars(budget=Decimal(10000))
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert broker._cash == Decimal(10000) - Decimal(10) * Decimal("151.0")
    position = broker.pull_positions()[0]
    assert position.asset == AAPL
    assert position.quantity == Decimal(10)
    assert position.side is PositionSide.LONG


def test_fill_records_a_fill_in_the_ledger() -> None:
    broker, clock, _ = _broker_with_two_bars()
    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert len(broker.ledger.fills) == 1
    record = broker.ledger.fills[0]
    assert record.identifier == order.identifier
    assert record.symbol == "AAPL"
    assert record.filled_quantity == Decimal(10)
    assert record.price == Decimal("151.0")


def test_commission_and_slippage_reduce_cash_beyond_the_raw_notional() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([100.0, 100.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker(
        "momentum", data_source=source, clock=clock, budget=Decimal(10000),
        commission=Decimal("0.01"), slippage=Decimal("0.01"),
    )
    clock.on_advance = broker.on_advance
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    # execution_price = 100 * 1.01 = 101; commission = 101 * 0.01 = 1.01/share
    expected_cash = Decimal(10000) - (Decimal(10) * Decimal("101.00") + Decimal(10) * Decimal("1.0100"))
    assert broker._cash == expected_cash


def test_unfilled_limit_order_stays_pending_and_is_retried_next_bar() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 200.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance
    order = broker.submit_order(
        Order(
            strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=Decimal(10), limit_price=Decimal(90),  # never touched by this fixture's bars
        )
    )

    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert order.status is OrderStatus.NEW
    assert order.identifier in broker._pending

    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)
    assert order.status is OrderStatus.NEW  # still never touched -- stays pending, doesn't error


def test_on_advance_samples_equity_after_processing_fills() -> None:
    broker, clock, _ = _broker_with_two_bars(budget=Decimal(10000))
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    assert len(broker.ledger.equity) == 1
    sample = broker.ledger.equity[0]
    assert sample.time == DAY2
    assert sample.positions_value == Decimal(10) * Decimal("151.0")
    assert sample.cash == broker._cash
    assert sample.portfolio_value == broker._cash + sample.positions_value


def test_selling_closes_the_position_when_quantity_returns_to_zero() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert len(broker.pull_positions()) == 1

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(10)))
    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)

    assert broker.pull_positions() == []


def test_adding_to_an_existing_long_position_increases_quantity() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(100000))
    clock.on_advance = broker.on_advance

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(5)))
    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)

    position = broker.pull_positions()[0]
    assert position.quantity == Decimal(15)
    assert position.side is PositionSide.LONG


def test_reducing_a_long_position_keeps_it_long_with_smaller_quantity() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(100000))
    clock.on_advance = broker.on_advance

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(4)))
    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)

    position = broker.pull_positions()[0]
    assert position.quantity == Decimal(6)
    assert position.side is PositionSide.LONG


def test_flipping_a_long_position_to_short_crosses_through_zero() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(100000))
    clock.on_advance = broker.on_advance

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(15)))
    clock._now = DAY3
    broker.on_advance(DAY2, DAY3)

    position = broker.pull_positions()[0]
    assert position.quantity == Decimal(5)
    assert position.side is PositionSide.SHORT

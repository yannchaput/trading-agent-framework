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


def test_order_submitted_mid_bar_formation_skips_that_bar_and_fills_on_the_one_after() -> None:
    # Submission happens strictly BEFORE any bar has closed (clock starts before DAY1, the
    # first bar's close) -- the mid-bar-formation case the boundary-aligned fixture above
    # can't exercise. The bar that closes at DAY1 is the one "in progress" at submission and
    # must be skipped; the fill must happen on the bar that closes at DAY2, not DAY1.
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([150.0, 151.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    before_day1 = DAY1 - timedelta(hours=1)
    clock = BacktestClock(start=before_day1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance

    order = broker.submit_order(
        Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10))
    )
    assert broker._pending[order.identifier].needs_skip is True

    # DAY1's bar closes: this is the submitting (in-progress) bar -- must be skipped, not filled.
    clock._now = DAY1
    broker.on_advance(before_day1, DAY1)
    assert order.status is OrderStatus.NEW
    assert order.identifier in broker._pending
    assert broker._pending[order.identifier].needs_skip is False

    # DAY2's bar closes: the first bar genuinely eligible to fill this order.
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)
    assert order.status is OrderStatus.FILL
    assert order.avg_fill_price == Decimal("151.0")
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


# --- long-only cash account: unaffordable buys and unheld sells are rejected at fill time ------


def _account(closes: list[float], *, budget: Decimal, commission: Decimal = Decimal(0)) -> tuple[BacktestBroker, BacktestClock]:
    source = FakeBacktestDataSource()
    source.set_bars(AAPL, make_close_indexed_frame(closes, start=DAY1, freq="1D"))
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=budget, commission=commission)
    clock.on_advance = broker.on_advance
    return broker, clock


def _order(side: OrderSide, quantity: int) -> Order:
    return Order(strategy_name="momentum", asset=AAPL, side=side, quantity=Decimal(quantity))


def _advance(broker: BacktestBroker, clock: BacktestClock, previous: datetime, now: datetime) -> None:
    clock._now = now  # test-only direct time jump; production code goes through clock.wait()
    broker.on_advance(previous, now)


def test_a_buy_the_cash_cannot_cover_is_rejected_and_changes_nothing() -> None:
    broker, clock = _account([150.0, 151.0], budget=Decimal(1000))
    order = broker.submit_order(_order(OrderSide.BUY, 10))  # fills at 151 -> costs 1510

    _advance(broker, clock, DAY1, DAY2)

    assert order.status is OrderStatus.ERROR
    assert "insufficient cash" in (order.error_message or "")
    assert broker._cash == Decimal(1000)
    assert broker.pull_positions() == []
    assert broker.ledger.fills == []
    assert order.identifier not in broker._pending


def test_a_buy_costing_exactly_the_available_cash_fills_and_one_cent_more_is_rejected() -> None:
    exact, exact_clock = _account([150.0, 151.0], budget=Decimal("1510"))
    exact_order = exact.submit_order(_order(OrderSide.BUY, 10))
    _advance(exact, exact_clock, DAY1, DAY2)

    short, short_clock = _account([150.0, 151.0], budget=Decimal("1509.99"))
    short_order = short.submit_order(_order(OrderSide.BUY, 10))
    _advance(short, short_clock, DAY1, DAY2)

    assert exact_order.status is OrderStatus.FILL
    assert exact._cash == Decimal(0)
    assert short_order.status is OrderStatus.ERROR
    assert short._cash == Decimal("1509.99")


def test_commission_counts_toward_the_cash_a_buy_needs() -> None:
    # 10 shares at 100 = 1000 notional + 1% commission (10) = 1010 needed.
    covered, covered_clock = _account([100.0, 100.0], budget=Decimal(1010), commission=Decimal("0.01"))
    covered_order = covered.submit_order(_order(OrderSide.BUY, 10))
    _advance(covered, covered_clock, DAY1, DAY2)

    notional_only, notional_only_clock = _account([100.0, 100.0], budget=Decimal(1005), commission=Decimal("0.01"))
    notional_only_order = notional_only.submit_order(_order(OrderSide.BUY, 10))
    _advance(notional_only, notional_only_clock, DAY1, DAY2)

    assert covered_order.status is OrderStatus.FILL
    assert covered._cash == Decimal(0)
    assert notional_only_order.status is OrderStatus.ERROR  # the notional alone (1000) fits, the commission does not
    assert notional_only._cash == Decimal(1005)


def test_a_buy_funded_by_a_sell_filling_on_the_same_bar_fills() -> None:
    # The cross_momentum pattern: sells and buys are submitted together and the buys are sized
    # against estimated sell proceeds. Orders fill in submission order, so the sell frees the cash.
    broker, clock = _account([100.0, 100.0, 100.0], budget=Decimal(1000))
    broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY1, DAY2)
    assert broker._cash == Decimal(0)

    sell = broker.submit_order(_order(OrderSide.SELL, 10))
    buy = broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY2, DAY3)

    assert sell.status is OrderStatus.FILL
    assert buy.status is OrderStatus.FILL
    assert broker._cash == Decimal(0)


def test_a_sell_larger_than_the_holding_is_rejected_and_the_position_is_unchanged() -> None:
    broker, clock = _account([150.0, 151.0, 152.0], budget=Decimal(100000))
    broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY1, DAY2)
    cash_after_buy = broker._cash

    oversized = broker.submit_order(_order(OrderSide.SELL, 15))
    _advance(broker, clock, DAY2, DAY3)

    assert oversized.status is OrderStatus.ERROR
    assert "insufficient position" in (oversized.error_message or "")
    [position] = broker.pull_positions()
    assert position.quantity == Decimal(10)
    assert position.side is PositionSide.LONG
    assert broker._cash == cash_after_buy


def test_selling_something_that_is_not_held_is_rejected() -> None:
    broker, clock = _account([150.0, 151.0], budget=Decimal(10000))
    order = broker.submit_order(_order(OrderSide.SELL, 5))

    _advance(broker, clock, DAY1, DAY2)

    assert order.status is OrderStatus.ERROR
    assert broker.pull_positions() == []
    assert broker._cash == Decimal(10000)


def test_the_second_of_two_full_size_sells_is_rejected() -> None:
    broker, clock = _account([150.0, 151.0, 152.0], budget=Decimal(100000))
    broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY1, DAY2)

    first = broker.submit_order(_order(OrderSide.SELL, 10))
    second = broker.submit_order(_order(OrderSide.SELL, 10))
    _advance(broker, clock, DAY2, DAY3)

    assert first.status is OrderStatus.FILL
    assert second.status is OrderStatus.ERROR
    assert broker.pull_positions() == []

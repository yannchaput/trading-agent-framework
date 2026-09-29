from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import FakeClock, FrameDataSource, et, minute_ohlc

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

ROWS = [
    (100, 100, 100, 100, 1000),  # closes 09:31
    (100, 101, 99.5, 100.5, 1000),  # 09:32: the buy fills at this open (100)
    (100.5, 103, 100.4, 102.8, 1000),  # 09:33: level 98.5, reference -> 103
    (102.8, 104, 103, 103.5, 1000),  # 09:34: level 101, reference -> 104
    (103.5, 103.6, 101.9, 102, 1000),  # 09:35: level 102 touched -> fills at 102
    (102, 102.5, 101.5, 102, 1000),  # 09:36
]
AAPL = Asset("AAPL")


def _broker() -> tuple[BacktestBroker, FakeClock, FrameDataSource]:
    source = FrameDataSource({("AAPL", "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)})
    clock = FakeClock(et(2026, 9, 1, 9, 31))
    broker = BacktestBroker("s", data_source=source, clock=clock, budget=Decimal("100000"), timestep="minute")
    return broker, clock, source


def _advance(broker: BacktestBroker, clock: FakeClock, seconds: float) -> None:
    before = clock.now()
    clock.advance(seconds)
    broker.on_advance(before, clock.now())


def _buy_ten(broker: BacktestBroker, clock: FakeClock) -> None:
    broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    _advance(broker, clock, 60)  # 09:32


def _trail(quantity: int = 10, **trail: Decimal) -> Order:
    return Order(strategy_name="s", asset=AAPL, side=OrderSide.SELL, order_type=OrderType.TRAIL, quantity=Decimal(quantity), **trail)


def test_trailing_stop_ratchets_across_a_multi_bar_jump_and_fills_at_its_level() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    trail = broker.submit_order(_trail(trail_price=Decimal(2)))  # reference seeded at the 09:32 close, 100.5
    _advance(broker, clock, 180)  # 09:35 in one jump: three bars walked
    assert trail.is_filled()
    assert broker.pull_positions() == []
    assert broker.get_account().cash == Decimal("100020")  # bought 10 at 100, sold 10 at 102


def test_a_pending_trailing_sell_counts_against_the_position() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    broker.submit_order(_trail(trail_price=Decimal(2)))
    with pytest.raises(OrderValidationError, match="insufficient position"):
        broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(1)))


def test_a_trailing_stop_without_a_trail_is_refused_at_submission() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    with pytest.raises(OrderValidationError, match="exactly one"):
        broker.submit_order(_trail())


def test_preload_bars_forwards_to_the_data_source() -> None:
    broker, _, source = _broker()
    broker.preload_bars([AAPL], et(2026, 5, 1), et(2026, 9, 30), "day")
    assert source.load_calls == [(("AAPL",), et(2026, 5, 1), et(2026, 9, 30), "day")]

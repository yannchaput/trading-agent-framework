from __future__ import annotations

from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order


def _limit_order() -> Order:
    return Order(
        strategy_name="momentum",
        asset=Asset("AAPL"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal(1),
        limit_price=Decimal("100"),
    )


def test_broker_stores_clock_and_account_kind() -> None:
    clock = FakeClock(et(2026, 9, 14, 9))
    broker = FakeBroker(clock, is_paper=False)
    assert broker.clock is clock
    assert broker.is_paper is False
    assert broker.strategy_name == "momentum"


def test_default_stream_hooks_are_no_ops() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 9)))
    assert Broker.start_stream(broker) is None
    assert Broker.stop_stream(broker) is None


def test_fake_broker_modify_marks_the_original_replaced() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 9)))
    order = broker.submit_order(_limit_order())

    replacement = broker.modify_order(order, limit_price=Decimal("101"))

    assert replacement.identifier != order.identifier
    assert replacement.limit_price == Decimal("101")
    assert order.status == OrderStatus.CANCELED
    assert broker.tracker.get_tracked_order(replacement.identifier) is replacement


def test_market_data_methods_are_part_of_the_broker_interface() -> None:
    expected = {"get_last_price", "get_last_prices", "get_quote", "get_bars"}
    assert expected <= Broker.__abstractmethods__


def test_news_provider_defaults_to_none() -> None:
    assert FakeBroker(FakeClock(et(2026, 9, 14, 10))).news_provider() is None

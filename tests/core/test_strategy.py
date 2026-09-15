from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import _FINAL_STATUSES, Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from trading_agent_framework.entities.position import Position

_START = et(2026, 9, 14, 9, 0)


def _broker(**kwargs: Any) -> FakeBroker:
    return FakeBroker(FakeClock(_START), **kwargs)


def _position(symbol: str) -> Position:
    return Position(
        strategy_name="momentum",
        asset=Asset(symbol),
        quantity=Decimal(10),
        side=PositionSide.LONG,
    )


class ParamStrategy(Strategy):
    parameters = {"symbol": "SPY", "quantity": 1}


class CrashRecorder(Strategy):
    def __init__(self, broker: FakeBroker) -> None:
        super().__init__(broker)
        self.abrupt_closings = 0

    def on_abrupt_closing(self) -> None:
        self.abrupt_closings += 1


# --- configuration ------------------------------------------------------------


def test_name_comes_from_the_broker() -> None:
    assert Strategy(_broker(strategy_name="momo")).name == "momo"


def test_defaults_match_lumibot() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    assert strategy.sleeptime == "1M"
    assert (strategy.minutes_before_opening, strategy.minutes_before_closing) == (60, 1)
    assert strategy.minutes_after_closing == 0
    assert strategy.parameters == {}
    assert strategy.vars == SimpleNamespace()
    assert strategy.first_iteration is True
    assert strategy.trading_mode is TradingMode.PAPER
    assert strategy.is_backtesting is False
    assert strategy.clock is broker.clock
    assert strategy.get_datetime() == _START


def test_parameters_merge_class_defaults_with_constructor_values() -> None:
    strategy = ParamStrategy(_broker(), parameters={"quantity": 5})
    assert strategy.parameters == {"symbol": "SPY", "quantity": 5}
    assert ParamStrategy.parameters == {"symbol": "SPY", "quantity": 1}


def test_explicit_clock_overrides_the_broker_clock() -> None:
    clock = FakeClock(et(2026, 9, 15, 10))
    assert Strategy(_broker(), clock=clock).get_datetime() == et(2026, 9, 15, 10)


def test_is_backtesting_follows_the_mode() -> None:
    assert Strategy(_broker(), mode=TradingMode.BACKTESTING).is_backtesting is True


def test_backtest_class_attribute_defaults() -> None:
    assert Strategy.backtesting_start is None
    assert Strategy.backtesting_end is None
    assert Strategy.budget == Decimal("10000")
    assert Strategy.benchmark_symbol == "SPY"


def test_add_line_is_a_no_op_outside_backtesting() -> None:
    strategy = Strategy(_broker())  # mode defaults to PAPER

    strategy.add_line("sma_200", 148.5)  # must not raise, must not touch anything

    assert strategy.trading_mode is TradingMode.PAPER  # sanity: no exception occurred


def test_add_line_records_an_indicator_line_in_backtesting() -> None:
    from tests.backtesting.fakes import FakeBacktestDataSource

    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.backtesting.clock import BacktestClock

    source = FakeBacktestDataSource()
    clock = BacktestClock(start=_START, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10000))
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING)

    strategy.add_line("sma_200", 148.5, color="red", style="dashed", plot_name="overlay")

    [line] = broker.ledger.lines
    assert line.name == "sma_200"
    assert line.value == Decimal("148.5")
    assert line.color == "red"
    assert line.style == "dashed"
    assert line.plot_name == "overlay"
    assert line.time == clock.now()


def test_final_statuses_includes_unknown_so_it_never_blocks_a_wait() -> None:
    """UNKNOWN is what map_status returns for an unrecognised Alpaca status; it must
    count as final or wait_for_order_execution would block on it for the full timeout."""
    assert OrderStatus.UNKNOWN in _FINAL_STATUSES


# --- hooks ----------------------------------------------------------------------


def test_hooks_default_to_no_ops() -> None:
    strategy = Strategy(_broker())
    order = strategy.create_order("AAPL", 1, "buy")
    strategy.initialize()
    strategy.on_trading_iteration()
    strategy.before_market_opens()
    strategy.before_starting_trading()
    strategy.before_market_closes()
    strategy.after_market_closes()
    strategy.on_strategy_end()
    strategy.on_abrupt_closing()
    strategy.on_new_order(order)
    strategy.on_canceled_order(order)
    strategy.on_partially_filled_order(None, order, Decimal(1), Decimal(1), 1)
    strategy.on_filled_order(None, order, Decimal(1), Decimal(1), 1)


def test_default_on_bot_crash_calls_on_abrupt_closing() -> None:
    strategy = CrashRecorder(_broker())
    strategy.on_bot_crash(RuntimeError("boom"))
    assert strategy.abrupt_closings == 1


# --- logging --------------------------------------------------------------------


def test_log_info_returns_the_message_and_names_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    strategy = Strategy(_broker())
    with caplog.at_level(logging.INFO, logger="trading_agent_framework"):
        returned = strategy.log_info("hello")
    record = caplog.records[-1]
    assert returned == "hello"
    assert record.filename == "test_strategy.py"
    assert "[momentum]" in record.getMessage()
    assert "hello" in record.getMessage()


# --- accounting -----------------------------------------------------------------


def test_cash_and_portfolio_value_come_from_the_account() -> None:
    strategy = Strategy(_broker())
    assert strategy.get_cash() == strategy.cash == Decimal("10000")
    assert strategy.get_portfolio_value() == strategy.portfolio_value == Decimal("25000")


def test_get_position_accepts_a_symbol_or_an_asset() -> None:
    broker = _broker()
    aapl = _position("AAPL")
    broker.positions = [aapl]
    strategy = Strategy(broker)
    assert strategy.get_positions() == [aapl]
    assert strategy.get_position("AAPL") is aapl
    assert strategy.get_position(Asset("AAPL")) is aapl
    assert strategy.get_position("TSLA") is None


def test_get_order_checks_the_tracker_then_the_broker() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    tracked = strategy.submit_order(strategy.create_order("AAPL", 1, "buy"))
    remote = strategy.create_order("TSLA", 1, "buy")
    broker.remote_orders[remote.identifier] = remote
    assert strategy.get_orders() == [tracked]
    assert strategy.get_order(tracked.identifier) is tracked
    assert strategy.get_order(remote.identifier) is remote
    assert strategy.get_order("missing") is None


# --- create_order -----------------------------------------------------------------


def test_create_market_order_from_plain_values() -> None:
    order = Strategy(_broker()).create_order("AAPL", 10, "buy")
    assert order.asset == Asset("AAPL")
    assert order.side is OrderSide.BUY
    assert order.order_type is OrderType.MARKET
    assert order.quantity == Decimal(10)
    assert order.time_in_force is TimeInForce.DAY
    assert order.strategy_name == "momentum"
    assert order.status is OrderStatus.UNPROCESSED


def test_create_limit_order() -> None:
    order = Strategy(_broker()).create_order(
        Asset("AAPL"), "2.5", OrderSide.SELL, limit_price=101.25, time_in_force="gtc"
    )
    assert order.order_type is OrderType.LIMIT
    assert order.limit_price == Decimal("101.25")
    assert order.quantity == Decimal("2.5")
    assert order.time_in_force is TimeInForce.GTC


def test_create_stop_order() -> None:
    order = Strategy(_broker()).create_order("AAPL", 1, "sell", stop_price=95)
    assert order.order_type is OrderType.STOP
    assert order.stop_price == Decimal(95)


def test_create_stop_limit_order_maps_limit_to_stop_limit_price() -> None:
    order = Strategy(_broker()).create_order("AAPL", 1, "sell", limit_price=94, stop_price=95)
    assert order.order_type is OrderType.STOP_LIMIT
    assert order.stop_price == Decimal(95)
    assert order.stop_limit_price == Decimal(94)
    assert order.limit_price is None


# --- trading ------------------------------------------------------------------------


def test_submit_orders_submits_each() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    orders = [strategy.create_order("AAPL", 1, "buy"), strategy.create_order("TSLA", 1, "buy")]
    assert strategy.submit_orders(orders) == orders
    assert broker.submitted == orders


def test_cancel_open_orders_skips_finished_orders() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    open_order = strategy.submit_order(strategy.create_order("AAPL", 1, "buy"))
    filled = strategy.submit_order(strategy.create_order("TSLA", 1, "buy"))
    broker.tracker.process_trade_event(
        filled, OrderEvent.FILLED, price=Decimal("100"), filled_quantity=Decimal(1)
    )

    strategy.cancel_open_orders()

    assert broker.canceled == [open_order]


def test_cancel_orders_cancels_each() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    orders = [strategy.create_order("AAPL", 1, "buy"), strategy.create_order("TSLA", 1, "buy")]
    strategy.cancel_orders(orders)
    assert broker.canceled == orders


def test_modify_order_converts_prices_to_decimal() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    order = strategy.submit_order(strategy.create_order("AAPL", 1, "buy", limit_price=100))

    replacement = strategy.modify_order(order, limit_price=101.5)

    assert broker.modified == [(order, Decimal("101.5"), None)]
    assert replacement is not order


def test_sell_all_closes_every_position() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.sell_all()
    strategy.sell_all(cancel_open_orders=False)
    assert broker.close_all_calls == [True, False]


def test_close_position_converts_symbol_and_fraction() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.close_position("AAPL", 0.5)
    strategy.close_position(Asset("TSLA"))
    assert broker.closed == [(Asset("AAPL"), Decimal("0.5")), (Asset("TSLA"), Decimal(1))]


def test_close_positions_defaults_to_every_held_position() -> None:
    broker = _broker()
    broker.positions = [_position("AAPL"), _position("TSLA")]
    strategy = Strategy(broker)

    strategy.close_positions()
    strategy.close_positions(["MSFT"])

    assert [asset.symbol for asset, _ in broker.closed] == ["AAPL", "TSLA", "MSFT"]

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.strategy import Strategy

MONDAY = date(2026, 9, 14)


class OrderHooks(Strategy):
    """Records order hooks (with the thread they ran on) and lifecycle milestones."""

    sleeptime = "1D"

    def __init__(self, broker: FakeBroker) -> None:
        super().__init__(broker)
        self.events: list[tuple[object, ...]] = []
        self.threads: set[str] = set()
        self.milestones: list[str] = []
        self.errors: list[BaseException] = []

    def _note(self, *event: object) -> None:
        self.events.append(event)
        self.threads.add(threading.current_thread().name)

    def on_new_order(self, order: Order) -> None:
        self._note("new", order.identifier)

    def on_canceled_order(self, order: Order) -> None:
        self._note("canceled", order.identifier)

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        self._note("partial", order.identifier, position, price, quantity, multiplier)

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        self._note("filled", order.identifier, position, price, quantity, multiplier)

    def on_trading_iteration(self) -> None:
        self.milestones.append("on_trading_iteration")

    def on_abrupt_closing(self) -> None:
        self.milestones.append("on_abrupt_closing")

    def on_strategy_end(self) -> None:
        self.milestones.append("on_strategy_end")

    def on_bot_crash(self, error: BaseException) -> None:
        self.errors.append(error)

    @property
    def fake_clock(self) -> FakeClock:
        return cast(FakeClock, self.clock)

    @property
    def fake_broker(self) -> FakeBroker:
        return cast(FakeBroker, self.broker)


def _strategy(cls: type[OrderHooks] = OrderHooks, sessions: int = 1) -> OrderHooks:
    clock = FakeClock(et(2026, 9, 14, 7), weekday_sessions(MONDAY, sessions))
    return cls(FakeBroker(clock))


def _aapl_position() -> Position:
    return Position(
        strategy_name="momentum", asset=Asset("AAPL"), quantity=Decimal(10), side=PositionSide.LONG
    )


def _on_first_wait(clock: FakeClock, action: Callable[[], None]) -> None:
    """Run `action` on a separate "stream" thread during the executor's first wait."""

    def hook() -> None:
        clock.on_wait = None
        thread = threading.Thread(target=action, name="fake-stream")
        thread.start()
        thread.join()

    clock.on_wait = hook


def test_events_from_the_stream_thread_run_hooks_on_the_executor_thread() -> None:
    strategy = _strategy()
    broker = strategy.fake_broker
    broker.positions = [_aapl_position()]
    order = broker.submit_order(strategy.create_order("AAPL", 10, "buy"))

    def stream() -> None:
        broker.tracker.process_trade_event(order, OrderEvent.NEW)
        broker.tracker.process_trade_event(
            order, OrderEvent.PARTIALLY_FILLED, price=Decimal(100), filled_quantity=Decimal(4)
        )
        broker.tracker.process_trade_event(
            order, OrderEvent.FILLED, price=Decimal(101), filled_quantity=Decimal(6)
        )

    _on_first_wait(strategy.fake_clock, stream)
    strategy.executor.run()

    position = _aapl_position()
    assert strategy.events == [
        ("new", order.identifier),
        ("partial", order.identifier, position, Decimal(100), Decimal(4), 1),
        ("filled", order.identifier, position, Decimal(101), Decimal(6), 1),
    ]
    assert strategy.threads == {threading.current_thread().name}


def test_cancel_error_and_modified_events(caplog: pytest.LogCaptureFixture) -> None:
    strategy = _strategy()
    broker = strategy.fake_broker
    canceled = broker.submit_order(strategy.create_order("AAPL", 1, "buy"))
    rejected = broker.submit_order(strategy.create_order("TSLA", 1, "buy"))
    modified = broker.submit_order(strategy.create_order("MSFT", 1, "buy"))

    def stream() -> None:
        broker.tracker.process_trade_event(canceled, OrderEvent.CANCELED)
        broker.tracker.process_trade_event(rejected, OrderEvent.ERROR)
        broker.tracker.process_trade_event(modified, OrderEvent.MODIFIED)

    _on_first_wait(strategy.fake_clock, stream)
    with caplog.at_level(logging.WARNING, logger="trading_agent_framework"):
        strategy.executor.run()

    assert strategy.events == [("canceled", canceled.identifier)]
    assert rejected.identifier in caplog.text


def test_order_hook_crash_goes_to_on_bot_crash_and_trading_continues() -> None:
    class BadHook(OrderHooks):
        def on_new_order(self, order: Order) -> None:
            raise RuntimeError("hook failed")

    strategy = _strategy(BadHook)
    broker = strategy.fake_broker
    order = broker.submit_order(strategy.create_order("AAPL", 1, "buy"))
    _on_first_wait(
        strategy.fake_clock, lambda: broker.tracker.process_trade_event(order, OrderEvent.NEW)
    )

    strategy.executor.run()

    assert [str(e) for e in strategy.errors] == ["hook failed"]
    assert strategy.milestones == ["on_trading_iteration", "on_strategy_end"]


def test_fill_hook_still_fires_with_no_position_when_position_lookup_fails() -> None:
    class BrokenPositionsBroker(FakeBroker):
        def pull_positions(self) -> list[Position]:
            raise RuntimeError("broker unavailable")

    clock = FakeClock(et(2026, 9, 14, 7), weekday_sessions(MONDAY, 1))
    strategy = OrderHooks(BrokenPositionsBroker(clock))
    broker = strategy.fake_broker
    order = broker.submit_order(strategy.create_order("AAPL", 10, "buy"))

    _on_first_wait(
        strategy.fake_clock,
        lambda: broker.tracker.process_trade_event(
            order, OrderEvent.FILLED, price=Decimal(100), filled_quantity=Decimal(10)
        ),
    )
    strategy.executor.run()

    assert strategy.events == [("filled", order.identifier, None, Decimal(100), Decimal(10), 1)]
    assert strategy.errors == []


class Waiter(OrderHooks):
    """Submits an order in its first iteration and waits for it."""

    fill_it = True

    def on_trading_iteration(self) -> None:
        broker = self.fake_broker
        order = self.submit_order(self.create_order("AAPL", 1, "buy"))
        if self.fill_it:
            _on_first_wait(
                self.fake_clock,
                lambda: broker.tracker.process_trade_event(
                    order, OrderEvent.FILLED, price=Decimal(100), filled_quantity=Decimal(1)
                ),
            )
        started = self.get_datetime()
        self.vars.result = self.wait_for_order_execution(order, timeout=600)
        self.vars.waited = self.get_datetime() - started
        self.vars.events_at_return = list(self.events)


def test_wait_for_order_execution_returns_once_filled() -> None:
    strategy = _strategy(Waiter)
    strategy.executor.run()
    assert strategy.vars.result is True
    assert strategy.vars.waited.total_seconds() == 0
    assert [event[0] for event in strategy.vars.events_at_return] == ["filled"]


def test_wait_for_order_execution_times_out() -> None:
    class NeverFilled(Waiter):
        fill_it = False

    strategy = _strategy(NeverFilled)
    strategy.executor.run()
    assert strategy.vars.result is False
    assert strategy.vars.waited.total_seconds() == 600


def test_keyboard_interrupt_runs_abrupt_closing_then_strategy_end() -> None:
    class Interrupted(OrderHooks):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            raise KeyboardInterrupt

    strategy = _strategy(Interrupted, sessions=2)
    strategy.executor.run()  # returns normally
    assert strategy.milestones == ["on_trading_iteration", "on_abrupt_closing", "on_strategy_end"]
    assert strategy.fake_broker.calls[-1] == "stop_stream"


def test_sigterm_is_handled_like_ctrl_c_and_the_handler_restored() -> None:
    class Terminated(OrderHooks):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # interrupted by the handler raising KeyboardInterrupt

    previous = signal.getsignal(signal.SIGTERM)
    strategy = _strategy(Terminated)
    strategy.executor.run()
    assert strategy.milestones == ["on_trading_iteration", "on_abrupt_closing", "on_strategy_end"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_run_off_the_main_thread_skips_signal_handling() -> None:
    strategy = _strategy()
    errors: list[BaseException] = []

    def target() -> None:
        try:
            strategy.executor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=10)
    assert errors == []
    assert strategy.milestones == ["on_trading_iteration", "on_strategy_end"]


def test_waiting_is_cut_short_by_stop_from_another_thread() -> None:
    strategy = _strategy()
    started: list[datetime] = []

    def stop_during_pre_open_wait() -> None:
        started.append(strategy.get_datetime())
        strategy.stop()

    _on_first_wait(strategy.fake_clock, stop_during_pre_open_wait)
    strategy.executor.run()
    assert strategy.milestones == ["on_strategy_end"]
    assert strategy.get_datetime() == started[0]

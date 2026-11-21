"""`BacktestBroker` is single-threaded by design: a second thread entering while one is inside is a bug, reported loudly.

LangGraph runs the tool calls of one model turn on worker threads; `AgentManager` serializes the calls of an agent that can
trade. The tripwire catches whatever would still reach the broker from two threads at once: the first entrant's state is
never read half-written, and the run aborts at the next clock advance even if a tool swallowed the first error.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BacktestError

AAPL = Asset("AAPL")
DAY1 = datetime(2026, 1, 5, 16, tzinfo=UTC)


class _BlockingSource(FakeBacktestDataSource):
    """Holds the caller inside `bars()` until released, once `block` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.block = False
        self.inside = threading.Event()
        self.release = threading.Event()

    def bars(self, *args: Any, **kwargs: Any) -> Any:
        if self.block:
            self.inside.set()
            self.release.wait(5)
        return super().bars(*args, **kwargs)


def _broker() -> tuple[BacktestBroker, BacktestClock, _BlockingSource]:
    source = _BlockingSource()
    source.set_bars(AAPL, make_close_indexed_frame([100.0, 100.0, 100.0], start=DAY1))
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=Decimal(10_000))
    clock.on_advance = broker.on_advance
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(1)))  # pending: get_account prices it
    return broker, clock, source


def _hold_inside(broker: BacktestBroker, source: _BlockingSource) -> threading.Thread:
    """A worker thread parked inside `get_account`."""
    source.block = True
    worker = threading.Thread(target=broker.get_account)
    worker.start()
    assert source.inside.wait(5)
    return worker


def test_a_read_while_another_thread_writes_raises() -> None:
    broker, _, source = _broker()
    source.block = True
    # submit_order prices the pending orders through bars(), so the worker is held inside a write
    worker = threading.Thread(target=broker.submit_order, args=(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(1)),))
    worker.start()
    assert source.inside.wait(5)

    with pytest.raises(BacktestError, match="two threads"):
        broker.get_account()

    source.release.set()
    worker.join()


def test_the_next_clock_advance_aborts_the_run_even_if_the_first_error_was_swallowed() -> None:
    broker, clock, source = _broker()
    worker = _hold_inside(broker, source)
    try:
        broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(1)))
    except BacktestError:
        pass  # what a tool's broad `except Exception` does
    source.release.set()
    worker.join()
    source.block = False

    with pytest.raises(BacktestError, match="two threads"):
        clock.wait(timedelta(days=1).total_seconds(), threading.Event())


def test_one_thread_at_a_time_is_fine_even_from_different_threads() -> None:
    broker, _, _ = _broker()
    worker = threading.Thread(target=broker.get_account)
    worker.start()
    worker.join()

    assert broker.get_account().cash == Decimal(10_000)


def test_parallel_readers_are_fine() -> None:
    # ray_dalio's three analysts run in parallel and each may read the account: reads alone never race.
    broker, _, source = _broker()
    worker = _hold_inside(broker, source)
    source.block = False  # only the worker stays parked

    assert broker.get_account().cash == Decimal(10_000)  # read while another thread reads

    source.release.set()
    worker.join()


def test_a_write_while_another_thread_reads_raises() -> None:
    broker, _, source = _broker()
    worker = _hold_inside(broker, source)

    with pytest.raises(BacktestError, match="two threads"):
        broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(1)))

    source.release.set()
    worker.join()

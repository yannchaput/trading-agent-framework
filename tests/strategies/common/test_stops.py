from __future__ import annotations

from decimal import Decimal as D

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.common.stops import Release, StopRelease, release_stop
from trading_agent_framework.utils.errors import BrokerError


def _stop(status: OrderStatus = OrderStatus.NEW) -> Order:
    order = Order(strategy_name="t", asset=Asset("AAA"), side=OrderSide.SELL, quantity=D(10))
    order.set_identifier("stop-1")
    order.status = status
    return order


class FakeStrategy:
    def __init__(self, order: Order | None = None, *, lookup_error: Exception | None = None, cancel_error: Exception | None = None) -> None:
        self.order = order
        self.lookup_error = lookup_error
        self.cancel_error = cancel_error
        self.expected: set[str] | None = None
        self.at_cancel: bool | None = None
        self.on_wait = lambda order: None
        self.cancelled: list[Order] = []
        self.waits: list[float] = []
        self.warnings: list[str] = []

    def get_order(self, identifier: str) -> Order | None:
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.order

    def cancel_order(self, order: Order) -> None:
        self.at_cancel = order.identifier in self.expected
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancelled.append(order)

    def wait_for_order_execution(self, order: Order, timeout: float) -> None:
        self.waits.append(timeout)
        self.on_wait(order)

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)


def _release(strategy: FakeStrategy, stop_id: str | None = "stop-1", expected: set[str] | None = None) -> Release:
    strategy.expected = expected if expected is not None else set()
    return release_stop(strategy, stop_id, expected_cancels=strategy.expected, wait_seconds=7.0, warn_prefix="[x] AAA")  # type: ignore[arg-type]


def test_no_stop_id_is_gone() -> None:
    assert _release(FakeStrategy(), None) == Release(StopRelease.GONE)


def test_an_unknown_order_is_gone() -> None:
    assert _release(FakeStrategy(None)).status is StopRelease.GONE


def test_a_filled_stop_is_filled_and_not_cancelled() -> None:
    strategy = FakeStrategy(_stop(OrderStatus.FILL))
    assert _release(strategy).status is StopRelease.FILLED
    assert strategy.cancelled == []


def test_an_active_stop_is_cancelled_waited_on_and_released() -> None:
    order = _stop()
    strategy = FakeStrategy(order)

    def cancelled(stop: Order) -> None:
        stop.status = OrderStatus.CANCELED

    strategy.on_wait = cancelled
    expected: set[str] = set()
    result = _release(strategy, expected=expected)
    assert result == Release(StopRelease.RELEASED, order=order)
    assert strategy.cancelled == [order] and strategy.waits == [7.0]
    assert expected == {"stop-1"}  # the late CANCELED hook is swallowed as the desk's own


def test_the_id_is_expected_before_the_cancel_is_sent() -> None:
    strategy = FakeStrategy(_stop())
    strategy.on_wait = lambda stop: setattr(stop, "status", OrderStatus.CANCELED)
    _release(strategy)
    assert strategy.at_cancel is True


def test_an_inactive_unfilled_stop_is_released_without_a_cancel() -> None:
    order = _stop(OrderStatus.CANCELED)
    strategy = FakeStrategy(order)
    assert _release(strategy) == Release(StopRelease.RELEASED, order=order)
    assert strategy.cancelled == [] and strategy.waits == []


def test_a_failed_lookup_fails_with_a_warning() -> None:
    strategy = FakeStrategy(lookup_error=RuntimeError("boom"))
    result = _release(strategy)
    assert result == Release(StopRelease.FAILED, error="could not look up the stop: boom")
    assert len(strategy.warnings) == 1 and strategy.warnings[0].startswith("[x] AAA: could not look up the stop stop-1")


def test_a_cancel_error_fails_and_discards_the_expected_id() -> None:
    expected: set[str] = set()
    strategy = FakeStrategy(_stop(), cancel_error=BrokerError("nope"))
    result = _release(strategy, expected=expected)
    assert result.status is StopRelease.FAILED and result.error == "could not cancel the stop: nope"
    assert expected == set() and strategy.waits == []


def test_a_cancel_error_on_an_already_filled_stop_is_filled() -> None:
    order = _stop()
    strategy = FakeStrategy(order)

    def raising(stop: Order) -> None:
        stop.status = OrderStatus.FILL
        raise BrokerError("already filled")

    strategy.cancel_order = raising  # type: ignore[method-assign]
    expected: set[str] = set()
    assert _release(strategy, expected=expected).status is StopRelease.FILLED
    assert expected == set()


def test_a_stop_that_fills_during_the_wait_is_filled() -> None:
    strategy = FakeStrategy(_stop())
    strategy.on_wait = lambda stop: setattr(stop, "status", OrderStatus.FILL)
    assert _release(strategy).status is StopRelease.FILLED


def test_a_cancel_not_confirmed_in_time_fails_and_is_no_longer_expected() -> None:
    expected: set[str] = set()
    strategy = FakeStrategy(_stop())
    result = _release(strategy, expected=expected)
    assert result == Release(StopRelease.FAILED, error="the stop cancel was not confirmed in time; nothing else was changed", timed_out=True)
    assert expected == set()
    assert strategy.warnings == ["[x] AAA: the stop cancel was not confirmed in 7.0s; the stop stays tracked"]


def test_only_the_timeout_outcome_is_flagged_timed_out() -> None:
    assert _release(FakeStrategy(_stop())).timed_out is True
    assert _release(FakeStrategy(_stop(), cancel_error=BrokerError("nope"))).timed_out is False
    assert _release(FakeStrategy(lookup_error=RuntimeError("x"))).timed_out is False

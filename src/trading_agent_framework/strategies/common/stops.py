"""Releasing a working stop before an exit sell, shared by the earnings_drift and vwap_pullback desks.

Brokers refuse a sell above held minus pending sells, so a stop must go before any other exit sell. If the stop filled
while the cancel was on its way (or before; its hook not seen yet), the position is already out. The desk keeps its own
partial-fill booking and its own return values; this module only talks to the broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.entities.order import Order


class StopRelease(StrEnum):
    GONE = "gone"  # no stop id, or the broker does not know the order
    FILLED = "filled"  # the stop sold the shares: the trade is already out
    RELEASED = "released"  # cancelled (or already ended unfilled): `Release.order` may hold a partial fill to book
    FAILED = "failed"  # nothing was changed; `Release.error` says why


@dataclass(frozen=True, slots=True)
class Release:
    status: StopRelease
    order: Order | None = None
    error: str | None = None
    timed_out: bool = False  # FAILED only: the cancel was sent but not confirmed in time (the id was un-expected)


def release_stop(strategy: Strategy, stop_order_id: str | None, *, expected_cancels: set[str], wait_seconds: float, warn_prefix: str) -> Release:
    """Cancel the working stop and wait `wait_seconds` for it. `warn_prefix` (e.g. "[earnings_drift] AAPL") starts the warnings.

    The order id is in `expected_cancels` for the whole cancel, so the desk's CANCELED hook does not read it as an outside
    cancel; it is taken out again when the cancel fails or is not confirmed in time, so a late CANCELED goes through the
    hook's stop path. The wait dispatches order hooks; in a backtest the cancel is synchronous.
    """
    if not stop_order_id:
        return Release(StopRelease.GONE)
    try:
        order = strategy.get_order(stop_order_id)
    except Exception as exc:  # a failed lookup is not "no stop": the stop may be live, so nothing else is sent
        strategy.log_warning(f"{warn_prefix}: could not look up the stop {stop_order_id}: {exc}")
        return Release(StopRelease.FAILED, error=f"could not look up the stop: {exc}")
    if order is None:
        return Release(StopRelease.GONE)
    if order.is_filled():
        return Release(StopRelease.FILLED)
    if order.is_active():
        expected_cancels.add(order.identifier)
        try:
            strategy.cancel_order(order)
        except BrokerError as exc:
            expected_cancels.discard(order.identifier)
            if order.is_filled():  # Alpaca raises when cancelling an already-filled order
                return Release(StopRelease.FILLED)
            return Release(StopRelease.FAILED, error=f"could not cancel the stop: {exc}")
        strategy.wait_for_order_execution(order, timeout=wait_seconds)
        if order.is_filled():
            return Release(StopRelease.FILLED)
        if order.is_active():
            expected_cancels.discard(order.identifier)
            strategy.log_warning(f"{warn_prefix}: the stop cancel was not confirmed in {wait_seconds}s; the stop stays tracked")
            return Release(StopRelease.FAILED, error="the stop cancel was not confirmed in time; nothing else was changed", timed_out=True)
    return Release(StopRelease.RELEASED, order=order)

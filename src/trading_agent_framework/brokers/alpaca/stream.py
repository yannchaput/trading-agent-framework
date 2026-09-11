"""Alpaca trade-update websocket stream: the async handler (Task 15) and its
background-thread lifecycle (Task 16).

`AlpacaTradeStream` wraps an `alpaca.trading.stream.TradingStream`, translating
each `TradeUpdate` it receives into a call against
`OrderTracker.process_trade_event`. It deliberately owns none of the
reconnect/backoff logic itself: `TradingStream.run()` already calls
`asyncio.run(self._run_forever())` internally, and `_run_forever` reconnects
with its own exponential backoff -- wrapping that in another event loop or an
outer while-loop (as lumibot does) is wrong against alpaca-py 0.44.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from trading_agent_framework.brokers.alpaca import orders
from trading_agent_framework.errors import BrokerError

if TYPE_CHECKING:
    from alpaca.trading.stream import TradingStream

    from trading_agent_framework.brokers.tracker import OrderTracker

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0


class AlpacaTradeStream:
    """Bridges Alpaca's `TradingStream` trade updates into an `OrderTracker`.

    `stream` may be `None` so `handle_trade_update` is testable with no real
    `TradingStream` at all; `start()` raises `BrokerError` if no stream was
    supplied, and `stop()` is a safe no-op in that case.
    """

    def __init__(self, tracker: OrderTracker, stream: TradingStream | None = None) -> None:
        self._tracker = tracker
        self._stream = stream
        self._thread: threading.Thread | None = None

    async def handle_trade_update(self, trade_update: Any) -> bool:
        """Apply one Alpaca `TradeUpdate` to the tracker.

        Returns `False` -- and never raises -- for a deliberately ignored or
        unrecognised event, or for an order we aren't tracking. The one
        exception is deliberate: `OrderEventError` raised by
        `process_trade_event` (a FILLED/PARTIALLY_FILLED event missing price
        or filled_quantity) is NOT caught here and propagates to the caller,
        so a future change to that contract is a conscious decision.
        """
        raw_order = orders._field(trade_update, "order")
        identifier = str(orders._field(raw_order, "id"))
        raw_event = orders._field(trade_update, "event")
        event = orders.map_event(raw_event)
        if event is None:
            logger.debug("ignoring event %r for order %s", raw_event, identifier)
            return False

        stored = self._tracker.get_tracked_order(identifier)
        if stored is None:
            # The stream can report an order's "new" event before this order's own
            # submit_order() call has learned its broker-assigned id (a network race
            # between the REST response and the websocket push) -- fall back to the
            # client_order_id we always know before submission, then promote it.
            client_order_id = orders._field(raw_order, "client_order_id")
            if client_order_id:
                stored = self._tracker.get_tracked_order_by_client_order_id(str(client_order_id))
                if stored is not None:
                    stored.set_identifier(identifier)
        if stored is None:
            logger.debug("untracked order %s (event %r)", identifier, raw_event)
            return False

        avg_fill_price = orders._to_decimal(orders._field(raw_order, "filled_avg_price"))
        if avg_fill_price is not None:
            stored.avg_fill_price = avg_fill_price

        price = orders._to_decimal(orders._field(trade_update, "price"))
        filled_quantity = orders._to_decimal(orders._field(trade_update, "qty"))
        self._tracker.process_trade_event(
            stored, event, price=price, filled_quantity=filled_quantity
        )
        return True

    def start(self, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS) -> None:
        """Subscribe the handler, run the websocket loop in a daemon thread, and
        block until the connect/auth/subscribe handshake actually completes.

        `stream.run()` spawns the handshake asynchronously and returns only when
        the loop stops -- if `start()` returned as soon as the thread was
        launched, a strategy could submit an order before the subscription was
        live and silently miss that order's first trade update (Alpaca doesn't
        replay events that happened before a stream subscribed). `_running` is
        `TradingStream`'s own private flag for "handshake complete"; polling it
        is the same kind of deliberate internals-reach as `stop()`'s `_loop`
        check below.
        """
        stream = self._stream
        if stream is None:
            raise BrokerError(
                "no TradingStream configured; construct AlpacaTradeStream with a "
                "real (or mocked) stream before calling start()"
            )
        stream.subscribe_trade_updates(self.handle_trade_update)
        self._thread = threading.Thread(target=stream.run, daemon=True, name="alpaca-trade-stream")
        self._thread.start()
        deadline = time.monotonic() + connect_timeout
        while not getattr(stream, "_running", False):
            if time.monotonic() >= deadline:
                raise BrokerError(
                    f"alpaca trade stream did not connect within {connect_timeout}s"
                )
            time.sleep(0.05)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the websocket loop and join its thread.

        `TradingStream.stop()` does `if self._loop.is_running()`, but
        `_loop` stays `None` until `_run_forever` has actually started inside
        the thread -- so calling `stop()` before `start()` (or before the
        loop has spun up) must not raise. Idempotent: safe to call more than
        once, and any exception from the underlying `stream.stop()` is
        logged, never propagated.
        """
        stream = self._stream
        if stream is not None and getattr(stream, "_loop", None) is not None:
            try:
                stream.stop()
            except Exception:
                logger.exception("error stopping alpaca trade stream")
        if self._thread is not None:
            self._thread.join(timeout)

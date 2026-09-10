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
from typing import TYPE_CHECKING, Any

from trading_agent_framework.brokers.alpaca import orders

if TYPE_CHECKING:
    from alpaca.trading.stream import TradingStream

    from trading_agent_framework.brokers.tracker import OrderTracker

logger = logging.getLogger(__name__)


class AlpacaTradeStream:
    """Bridges Alpaca's `TradingStream` trade updates into an `OrderTracker`.

    `stream` may be `None` so `handle_trade_update` is testable with no real
    `TradingStream` at all; `start()`/`stop()` require a real (or mocked)
    stream to have been supplied, which every caller of the lifecycle methods
    does.
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

    def start(self) -> None:
        """Subscribe the handler and run the websocket loop in a daemon thread."""
        self._stream.subscribe_trade_updates(self.handle_trade_update)
        self._thread = threading.Thread(
            target=self._stream.run, daemon=True, name="alpaca-trade-stream"
        )
        self._thread.start()

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
        loop = getattr(self._stream, "_loop", None)
        if loop is not None:
            try:
                self._stream.stop()
            except Exception:
                logger.exception("error stopping alpaca trade stream")
        if self._thread is not None:
            self._thread.join(timeout)

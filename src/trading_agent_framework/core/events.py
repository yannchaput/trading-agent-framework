"""Hand-off of order events from the broker's stream thread to the executor thread.

`OrderEventQueue` is registered as an `OrderTracker` listener. The tracker calls it
on the stream thread (inside its transition lock), so it only records the event
and wakes the executor; the strategy's order hooks run later, on the executor
thread, and therefore never concurrently with other strategy code.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderEvent
from trading_agent_framework.entities.order import Order

_FILL_EVENTS = frozenset({OrderEvent.FILLED, OrderEvent.PARTIALLY_FILLED})


@dataclass(frozen=True, slots=True)
class QueuedOrderEvent:
    """An order event waiting for the executor to dispatch it to the strategy.

    Attributes:
        order: The order the event is about.
        event: What happened to it.
        price: Fill price of the latest fill, for fill events.
        quantity: Fill quantity of the latest fill, for fill events.
    """

    order: Order
    event: OrderEvent
    price: Decimal | None = None
    quantity: Decimal | None = None


class OrderEventQueue:
    """Thread-safe FIFO of order events, doubling as an `OrderTracker` listener."""

    def __init__(self, wake: threading.Event) -> None:
        """Create an empty queue.

        Args:
            wake: Event set whenever an order event is queued, to wake the executor from its wait.
        """
        self._queue: queue.SimpleQueue[QueuedOrderEvent] = queue.SimpleQueue()
        self._wake = wake

    def __call__(self, order: Order, event: OrderEvent) -> None:
        """Queue an order event (called from the broker's stream thread).

        For fill events the latest fill's price and quantity are read now, because later fills append more transactions
        before the executor drains the queue.

        Args:
            order: The order the event is about.
            event: What happened to it.
        """
        price = quantity = None
        if event in _FILL_EVENTS and order.transactions:
            # Read now: later fills append more transactions before the executor drains.
            last_fill = order.transactions[-1]
            price, quantity = last_fill.price, last_fill.quantity
        self._queue.put(QueuedOrderEvent(order, event, price, quantity))
        self._wake.set()

    def drain(self) -> Iterator[QueuedOrderEvent]:
        """Pop queued events until the queue is empty.

        Yields:
            Each queued event, oldest first.
        """
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return

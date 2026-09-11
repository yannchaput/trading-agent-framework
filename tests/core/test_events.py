from __future__ import annotations

import threading
from decimal import Decimal

from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.core.events import OrderEventQueue, QueuedOrderEvent
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide
from trading_agent_framework.entities.order import Order


def _order() -> Order:
    return Order(
        strategy_name="momentum", asset=Asset("AAPL"), side=OrderSide.BUY, quantity=Decimal(10)
    )


def test_listener_queues_the_event_and_sets_wake() -> None:
    wake = threading.Event()
    queue = OrderEventQueue(wake)
    order = _order()

    queue(order, OrderEvent.NEW)

    assert wake.is_set()
    assert list(queue.drain()) == [QueuedOrderEvent(order, OrderEvent.NEW)]


def test_fill_events_capture_the_fill_at_call_time() -> None:
    queue = OrderEventQueue(threading.Event())
    order = _order()
    order.add_transaction(Decimal("100.5"), Decimal("4"))

    queue(order, OrderEvent.PARTIALLY_FILLED)
    order.add_transaction(Decimal("101"), Decimal("6"))  # a later fill must not leak in

    [item] = queue.drain()
    assert item.price == Decimal("100.5")
    assert item.quantity == Decimal("4")


def test_fill_event_without_transactions_has_no_price() -> None:
    queue = OrderEventQueue(threading.Event())
    queue(_order(), OrderEvent.FILLED)
    [item] = queue.drain()
    assert item.price is None
    assert item.quantity is None


def test_drain_is_fifo_and_empties_the_queue() -> None:
    queue = OrderEventQueue(threading.Event())
    order = _order()
    queue(order, OrderEvent.NEW)
    queue(order, OrderEvent.CANCELED)

    assert [item.event for item in queue.drain()] == [OrderEvent.NEW, OrderEvent.CANCELED]
    assert list(queue.drain()) == []


def test_events_posted_by_the_tracker_on_another_thread_are_queued() -> None:
    wake = threading.Event()
    queue = OrderEventQueue(wake)
    tracker = OrderTracker()
    tracker.listeners.append(queue)
    order = _order()
    tracker.track_unprocessed(order)

    def stream_thread() -> None:
        tracker.process_trade_event(order, OrderEvent.NEW)
        tracker.process_trade_event(
            order, OrderEvent.FILLED, price=Decimal("99"), filled_quantity=Decimal("10")
        )

    thread = threading.Thread(target=stream_thread)
    thread.start()
    thread.join()

    assert wake.is_set()
    items = list(queue.drain())
    assert [item.event for item in items] == [OrderEvent.NEW, OrderEvent.FILLED]
    assert (items[1].price, items[1].quantity) == (Decimal("99"), Decimal("10"))

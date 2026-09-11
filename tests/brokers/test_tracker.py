from __future__ import annotations

import threading
import time
from datetime import datetime
from decimal import Decimal

import pytest

from trading_agent_framework.brokers.tracker import OrderTracker, SafeList
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderEventError


def make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "strategy_name": "momentum",
        "asset": Asset(symbol="AAPL"),
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
    }
    defaults.update(overrides)
    return Order(**defaults)  # ty: ignore[invalid-argument-type]


# --- Task 5: SafeList -------------------------------------------------------


def test_remove_matches_by_identifier_across_distinct_instances() -> None:
    safe_list: SafeList[Order] = SafeList()
    order = make_order()
    safe_list.append(order)

    lookalike = make_order()
    lookalike.set_identifier(order.identifier)
    assert lookalike is not order

    removed = safe_list.remove(lookalike)

    assert removed is order
    assert len(safe_list) == 0


def test_remove_of_absent_item_returns_none_and_does_not_raise() -> None:
    safe_list: SafeList[Order] = SafeList()
    safe_list.append(make_order())

    result = safe_list.remove(make_order())

    assert result is None
    assert len(safe_list) == 1


def test_snapshot_is_an_independent_copy() -> None:
    safe_list: SafeList[Order] = SafeList()
    order1 = make_order()
    order2 = make_order()
    safe_list.append(order1)
    safe_list.append(order2)

    snapshot = safe_list.snapshot()
    snapshot.append(make_order())
    snapshot.pop(0)

    assert len(safe_list) == 2
    assert safe_list.snapshot() == [order1, order2]


def test_concurrent_appends_from_many_threads_are_all_recorded() -> None:
    safe_list: SafeList[object] = SafeList()

    def worker() -> None:
        for _ in range(50):
            safe_list.append(object())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(safe_list) == 400


# --- Task 6: OrderTracker ----------------------------------------------------


def test_new_event_moves_order_from_unprocessed_to_new_bucket() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)

    tracker.process_trade_event(order, OrderEvent.NEW)

    assert order.status == OrderStatus.NEW
    assert order in tracker.new.snapshot()
    assert order not in tracker.unprocessed.snapshot()


def test_partial_fill_then_full_fill_updates_buckets_and_transactions() -> None:
    tracker = OrderTracker()
    order = make_order(quantity=Decimal("10"))
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    tracker.process_trade_event(
        order,
        OrderEvent.PARTIALLY_FILLED,
        price=Decimal("10"),
        filled_quantity=Decimal("3"),
    )

    assert order in tracker.partially_filled.snapshot()
    assert order not in tracker.new.snapshot()
    assert order.filled_quantity == Decimal("3")
    assert len(order.transactions) == 1

    tracker.process_trade_event(
        order,
        OrderEvent.FILLED,
        price=Decimal("11"),
        filled_quantity=Decimal("7"),
    )

    assert order in tracker.filled.snapshot()
    assert order not in tracker.partially_filled.snapshot()
    assert order.filled_quantity == Decimal("10")
    assert len(order.transactions) == 2


def test_filled_without_price_raises_and_leaves_buckets_untouched() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    with pytest.raises(OrderEventError):
        tracker.process_trade_event(
            order,
            OrderEvent.FILLED,
            price=None,
            filled_quantity=Decimal("5"),
        )

    assert order in tracker.new.snapshot()
    assert len(tracker.filled) == 0
    assert order.status == OrderStatus.NEW


def test_filled_without_filled_quantity_raises() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    with pytest.raises(OrderEventError):
        tracker.process_trade_event(
            order,
            OrderEvent.FILLED,
            price=Decimal("10"),
            filled_quantity=None,
        )

    assert order in tracker.new.snapshot()
    assert len(tracker.filled) == 0
    assert order.status == OrderStatus.NEW


def test_canceled_event_from_new_moves_to_canceled_bucket() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    tracker.process_trade_event(order, OrderEvent.CANCELED)

    assert order.status == OrderStatus.CANCELED
    assert order in tracker.canceled.snapshot()
    assert order not in tracker.new.snapshot()


def test_error_event_moves_to_error_bucket() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)

    tracker.process_trade_event(order, OrderEvent.ERROR)

    assert order.status == OrderStatus.ERROR
    assert order in tracker.error.snapshot()
    assert order not in tracker.unprocessed.snapshot()


def test_modified_event_does_not_change_any_bucket() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    tracker.process_trade_event(order, OrderEvent.MODIFIED)

    assert order in tracker.new.snapshot()
    assert order.status == OrderStatus.NEW
    assert len(tracker.get_all_tracked_orders()) == 1


def test_listener_that_raises_is_logged_and_does_not_propagate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)

    calls: list[str] = []

    def bad_listener(order: Order, event: OrderEvent) -> None:
        raise RuntimeError("listener boom")

    def good_listener(order: Order, event: OrderEvent) -> None:
        calls.append(order.identifier)

    tracker.listeners.append(bad_listener)
    tracker.listeners.append(good_listener)

    with caplog.at_level("ERROR"):
        tracker.process_trade_event(order, OrderEvent.NEW)

    assert calls == [order.identifier]
    assert any("listener boom" in record.message or record.exc_info for record in caplog.records)


def test_concurrent_processing_of_same_order_lands_in_exactly_one_bucket() -> None:
    """Two threads race PARTIALLY_FILLED/FILLED events for the SAME order.

    A `threading.Barrier` lines both threads up before each call so they enter
    `process_trade_event` at (as close to) the same instant as possible on
    every iteration. To reliably widen the "remove from every bucket, then
    append to destination bucket" window that the tracker-level lock closes
    (a natural, unassisted race is too rare on the GIL to trigger in a fast
    test), the order's `add_transaction` is wrapped with a short sleep: this
    forces whichever thread reaches it first to hold the destination-bucket
    decision open long enough for the other thread's own remove-then-append
    sequence to interleave, if nothing is serializing the two calls.
    """
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)
    tracker.process_trade_event(order, OrderEvent.NEW)

    original_add_transaction = order.add_transaction

    def slow_add_transaction(
        price: Decimal, filled_quantity: Decimal, timestamp: datetime | None = None
    ) -> None:
        original_add_transaction(price, filled_quantity, timestamp)
        time.sleep(0.002)

    order.add_transaction = slow_add_transaction  # ty: ignore[invalid-assignment]

    iterations = 30
    barrier = threading.Barrier(2)

    def partial_fill_worker() -> None:
        for _ in range(iterations):
            barrier.wait()
            tracker.process_trade_event(
                order,
                OrderEvent.PARTIALLY_FILLED,
                price=Decimal("10"),
                filled_quantity=Decimal("1"),
            )

    def fill_worker() -> None:
        for _ in range(iterations):
            barrier.wait()
            tracker.process_trade_event(
                order,
                OrderEvent.FILLED,
                price=Decimal("10"),
                filled_quantity=Decimal("1"),
            )

    thread_a = threading.Thread(target=partial_fill_worker)
    thread_b = threading.Thread(target=fill_worker)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    membership_count = sum(1 for bucket in tracker._buckets() if order in bucket.snapshot())
    assert membership_count == 1


def test_get_tracked_order_searches_all_buckets() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)

    assert tracker.get_tracked_order(order.identifier) is order

    tracker.process_trade_event(order, OrderEvent.NEW)
    assert tracker.get_tracked_order(order.identifier) is order

    assert tracker.get_tracked_order("does-not-exist") is None


def test_get_tracked_order_by_client_order_id_finds_order_before_broker_id_known() -> None:
    tracker = OrderTracker()
    order = make_order()
    order.client_order_id = "momentum:local-uuid"
    tracker.track_unprocessed(order)

    assert tracker.get_tracked_order_by_client_order_id("momentum:local-uuid") is order
    assert tracker.get_tracked_order_by_client_order_id("does-not-exist") is None


def test_untrack_removes_order_from_whichever_bucket_it_is_in() -> None:
    tracker = OrderTracker()
    order = make_order()
    tracker.track_unprocessed(order)

    tracker.untrack(order)

    assert tracker.get_tracked_order(order.identifier) is None
    assert order not in tracker.unprocessed.snapshot()


def test_mark_replaced_cancels_old_tracks_new_and_notifies_nobody() -> None:
    tracker = OrderTracker()
    notified: list[tuple[Order, OrderEvent]] = []
    tracker.listeners.append(lambda order, event: notified.append((order, event)))
    old = make_order()
    tracker.track_unprocessed(old)
    tracker.process_trade_event(old, OrderEvent.NEW)
    notified.clear()
    new = make_order()

    tracker.mark_replaced(old, new)

    assert old.status == OrderStatus.CANCELED
    assert old in tracker.canceled
    assert old not in tracker.new
    assert new in tracker.unprocessed
    assert tracker.get_tracked_order(new.identifier) is new
    assert notified == []

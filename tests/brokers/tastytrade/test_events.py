from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from tests.fakes import make_tt_fill, make_tt_order

from trading_agent_framework.brokers.tastytrade.events import TastytradeOrderEvents
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

D = Decimal


@pytest.fixture
def tracker() -> OrderTracker:
    return OrderTracker()


@pytest.fixture
def events(tracker: OrderTracker) -> TastytradeOrderEvents:
    return TastytradeOrderEvents(tracker)


@pytest.fixture
def seen(tracker: OrderTracker) -> list[OrderEvent]:
    received: list[OrderEvent] = []
    tracker.listeners.append(lambda order, event: received.append(event))
    return received


def _tracked(tracker: OrderTracker, events: TastytradeOrderEvents, *, notional: str | None = None, register: bool = True) -> Order:
    if notional is None:
        order = Order("s", Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, quantity=D(10), limit_price=D(150), identifier="abc", client_order_id="s:abc")
    else:
        order = Order("s", Asset("AAPL"), OrderSide.BUY, OrderType.MARKET, notional=D(notional), identifier="abc", client_order_id="s:abc")
    order.status = OrderStatus.SUBMITTED
    tracker.track_unprocessed(order)
    if register:
        events.register(order, "1001")
    return order


def test_live_makes_a_submitted_order_new(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)

    assert events.apply(make_tt_order(status="Live"))
    assert order.status is OrderStatus.NEW
    assert seen == [OrderEvent.NEW]
    assert not events.apply(make_tt_order(status="Live"))  # already new: nothing to do


def test_an_order_that_is_not_ours_is_ignored(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    _tracked(tracker, events)

    assert not events.apply(make_tt_order(order_id=9999, external="other:xyz"))


def test_an_unregistered_id_falls_back_to_the_external_identifier(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)

    assert events.apply(make_tt_order(order_id=1001, status="Live"))
    assert order.status is OrderStatus.NEW


def test_a_partial_fill_then_the_rest(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)
    first = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Live", fills=[first]))
    assert (order.status, order.filled_quantity) == (OrderStatus.PARTIAL_FILL, D(4))

    final = make_tt_order(status="Filled", fills=[first, make_tt_fill("f2", "6", "151")])
    events.apply(final)
    assert (order.status, order.filled_quantity, len(order.transactions)) == (OrderStatus.FILL, D(10), 2)

    assert not events.apply(final)  # a replayed snapshot changes nothing
    assert order.filled_quantity == D(10)
    assert seen == [OrderEvent.PARTIALLY_FILLED, OrderEvent.FILLED]


@pytest.mark.parametrize("status", ["Cancelled", "Removed", "Partially Removed", "Expired"])
def test_ending_statuses_cancel(tracker: OrderTracker, events: TastytradeOrderEvents, status: str) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status=status))

    assert order.status is OrderStatus.CANCELED


def test_expired_emits_a_canceled_event_not_an_expired_one(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    # orders.map_status says EXPIRED for it, but the tracker has no expiry bucket: it ends as CANCELED.
    order = _tracked(tracker, events)

    assert events.apply(make_tt_order(status="Expired"))

    assert (order.status, seen) == (OrderStatus.CANCELED, [OrderEvent.CANCELED])
    assert order in tracker.canceled.snapshot()
    assert not events.apply(make_tt_order(status="Expired"))  # replayed: nothing more to cancel


def test_a_rejection_carries_its_reason(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status="Rejected", **{"reject-reason": "Insufficient buying power"}))

    assert (order.status, order.error_message) == (OrderStatus.ERROR, "Insufficient buying power")
    assert seen == [OrderEvent.ERROR]


def test_cancel_requested_marks_the_order_cancelling(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)

    events.apply(make_tt_order(status="Cancel Requested"))

    assert order.status is OrderStatus.CANCELLING


def test_a_cancel_of_the_replaced_id_racing_the_put_is_ignored(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.apply(make_tt_order(status="Live"))

    assert events.begin_replace(order) == "1001"
    events.apply(make_tt_order(status="Cancelled"))  # the original's cancel arrives before the PUT returns
    assert order.status is OrderStatus.NEW

    events.finish_replace(order, "1002")
    assert events.tt_id_of(order) == "1002"
    events.apply(make_tt_order(order_id=1002, status="Filled", fills=[make_tt_fill("f1", "10", "151")]))
    assert order.status is OrderStatus.FILL


def test_fills_on_a_replaced_id_still_count(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.finish_replace(order, "1002")

    events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f9", "2", "150")]))

    assert (order.filled_quantity, order.status) == (D(2), OrderStatus.PARTIAL_FILL)


def test_an_aborted_replace_listens_to_the_original_again(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    tt_id = events.begin_replace(order)
    assert tt_id is not None
    events.abort_replace(tt_id)

    events.apply(make_tt_order(status="Cancelled"))

    assert order.status is OrderStatus.CANCELED


def test_a_replace_that_keeps_its_id_is_not_superseded(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.finish_replace(order, "1001")

    events.apply(make_tt_order(status="Cancelled"))

    assert order.status is OrderStatus.CANCELED


def test_a_notional_orders_fills_wait_for_filled(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, notional="250")

    events.apply(make_tt_order(status="Live", quantity=None, fills=[make_tt_fill("f1", "1.2", "150")]))
    assert (order.status, order.filled_quantity) == (OrderStatus.NEW, D(0))  # Live counts, the fill waits

    events.apply(make_tt_order(status="Filled", quantity=None, fills=[make_tt_fill("f1", "1.2", "150"), make_tt_fill("f2", "0.46", "151")]))
    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D("1.66"))


def test_registering_with_a_snapshot_marks_its_fills_as_counted(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)
    snapshot = make_tt_order(status="Live", fills=[make_tt_fill("f1", "4", "150")])
    order.filled_quantity = D(4)  # as parse_order sets it for an adopted order
    events.register(order, "1001", snapshot=snapshot)

    events.apply(snapshot)

    assert order.filled_quantity == D(4)


def test_a_fill_arriving_after_the_order_ended_is_counted_without_reviving_it(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)
    events.apply(make_tt_order(status="Live"))
    events.apply(make_tt_order(status="Cancelled"))

    assert events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f1", "4", "150")]))

    assert (order.status, order.filled_quantity, len(order.transactions)) == (OrderStatus.CANCELED, D(4), 1)
    assert seen == [OrderEvent.NEW, OrderEvent.CANCELED]
    assert tracker.canceled.snapshot() == [order]
    assert tracker.partially_filled.snapshot() == []
    assert not events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f1", "4", "150")]))  # replayed
    assert order.filled_quantity == D(4)


def test_a_fallback_adoption_registers_the_id_and_counts_only_new_fills(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)
    order.filled_quantity = D(4)  # already reflected, as parse_order sets it for an adopted order
    f1 = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Live", fills=[f1]))

    assert events.tt_id_of(order) == "1001"
    assert (order.filled_quantity, len(order.transactions)) == (D(4), 0)

    events.apply(make_tt_order(status="Live", fills=[f1, make_tt_fill("f2", "2", "151")]))
    assert (order.filled_quantity, len(order.transactions)) == (D(6), 1)


def test_a_fallback_adoption_of_an_order_without_fills_counts_the_first_ones(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)

    events.apply(make_tt_order(status="Live", fills=[make_tt_fill("f1", "4", "150")]))

    assert (order.filled_quantity, len(order.transactions)) == (D(4), 1)


def test_two_orders_may_share_a_fill_id(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    first = _tracked(tracker, events)
    second = Order("s", Asset("MSFT"), OrderSide.BUY, OrderType.LIMIT, quantity=D(5), limit_price=D(300), identifier="def", client_order_id="s:def")
    second.status = OrderStatus.SUBMITTED
    tracker.track_unprocessed(second)
    events.register(second, "1002")

    events.apply(make_tt_order(order_id=1001, status="Live", fills=[make_tt_fill("f1", "4", "150")]))
    events.apply(make_tt_order(order_id=1002, status="Live", symbol="MSFT", external="s:def", fills=[make_tt_fill("f1", "3", "300")]))

    assert (first.filled_quantity, second.filled_quantity) == (D(4), D(3))


def test_avg_fill_price_is_the_weighted_average_of_the_fills(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    assert order.avg_fill_price is None
    first = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Live", fills=[first]))
    assert order.avg_fill_price == D(150)

    final = make_tt_order(status="Filled", fills=[first, make_tt_fill("f2", "6", "151")])
    events.apply(final)
    assert order.avg_fill_price == D("150.6")

    events.apply(final)  # replayed
    assert order.avg_fill_price == D("150.6")


def test_avg_fill_price_includes_fills_already_reflected_on_an_adopted_order(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)
    order.filled_quantity = D(4)
    order.avg_fill_price = D(150)  # as parse_order sets it
    f1 = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Live", fills=[f1]))
    assert order.avg_fill_price == D(150)

    events.apply(make_tt_order(status="Filled", fills=[f1, make_tt_fill("f2", "6", "151")]))
    assert order.avg_fill_price == D("150.6")


def test_avg_fill_price_when_an_adopted_order_has_none_yet(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events, register=False)
    order.filled_quantity = D(4)
    f1 = make_tt_fill("f1", "4", "150")

    events.apply(make_tt_order(status="Filled", fills=[f1, make_tt_fill("f2", "6", "151")]))

    assert order.avg_fill_price == D("150.6")


def test_a_late_fill_updates_avg_fill_price(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.apply(make_tt_order(status="Live"))
    events.apply(make_tt_order(status="Cancelled"))
    assert order.avg_fill_price is None

    events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f1", "4", "150")]))
    assert order.avg_fill_price == D(150)

    events.apply(make_tt_order(status="Cancelled", fills=[make_tt_fill("f1", "4", "150"), make_tt_fill("f2", "4", "152")]))
    assert order.avg_fill_price == D(151)


def test_a_snapshot_of_this_strategys_untracked_order_is_logged_as_an_error(tracker: OrderTracker, caplog: pytest.LogCaptureFixture) -> None:
    events = TastytradeOrderEvents(tracker, strategy_name="s")

    with caplog.at_level(logging.ERROR):
        assert not events.apply(make_tt_order(order_id=77, external="s:ghost"))

    assert [r for r in caplog.records if r.levelno == logging.ERROR and "s:ghost" in r.getMessage()]


def test_a_snapshot_of_another_strategys_order_is_not_an_error(tracker: OrderTracker, caplog: pytest.LogCaptureFixture) -> None:
    events = TastytradeOrderEvents(tracker, strategy_name="s")

    with caplog.at_level(logging.ERROR):
        assert not events.apply(make_tt_order(order_id=77, external="other:ghost"))
        assert not events.apply(make_tt_order(order_id=78, external=None))

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def _fill_snapshot(order_id: int, status: str, fills: list[dict[str, object]], **extra: object) -> dict[str, object]:
    return make_tt_order(order_id=order_id, status=status, external="s:abc", fills=fills, **extra)


def test_a_replaced_originals_cancel_never_cancels_the_order_that_adopted_its_replacement(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events, register=False)
    events.register(order, "2002")  # a restart adopted the live replacement

    events.apply(make_tt_order(order_id=1001, status="Cancelled", external="s:abc", **{"replacing-order-id": 2002}))

    assert order.is_active()
    assert events.tt_id_of(order) == "2002"
    assert OrderEvent.CANCELED not in seen


def test_a_stale_snapshot_with_an_unrelated_id_is_superseded_not_current(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)  # current 1001

    events.apply(make_tt_order(order_id=900, status="Cancelled", external="s:abc"))

    assert order.is_active()
    assert events.tt_id_of(order) == "1001"


def test_a_replacement_first_seen_with_a_fill_adds_it_to_the_originals(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.apply(_fill_snapshot(1001, "Live", [make_tt_fill("f1", "4", "150")]))
    assert order.filled_quantity == D(4)
    events.begin_replace(order)

    events.apply(_fill_snapshot(2002, "Live", [make_tt_fill("g1", "3", "151")], **{"replaces-order-id": 1001}))

    assert order.filled_quantity == D(7)
    assert events.tt_id_of(order) == "2002"


def test_a_replacement_seen_before_the_replace_returns_becomes_current_and_the_original_is_ignored(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)  # no begin_replace: the stream is ahead of the PUT's answer

    events.apply(make_tt_order(order_id=2002, status="Live", external="s:abc", **{"replaces-order-id": 1001}))
    events.apply(make_tt_order(order_id=1001, status="Cancelled", external="s:abc"))

    assert events.tt_id_of(order) == "2002"
    assert order.is_active()
    assert OrderEvent.CANCELED not in seen


def test_the_normal_modify_flow_still_works(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.finish_replace(order, "2002")

    events.apply(make_tt_order(order_id=1001, status="Cancelled", external="s:abc"))
    assert order.is_active()
    events.apply(make_tt_order(order_id=2002, status="Live", external="s:abc", **{"replaces-order-id": 1001}))
    events.apply(_fill_snapshot(2002, "Filled", [make_tt_fill("g1", "10", "151")], **{"replaces-order-id": 1001}))

    assert (order.status, order.filled_quantity) == (OrderStatus.FILL, D(10))
    assert seen.count(OrderEvent.FILLED) == 1


def test_register_superseded_counts_fills_but_ignores_statuses(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events, register=False)
    events.register(order, "2002")
    events.register_superseded(order, "1001", snapshot=make_tt_order(order_id=1001, fills=[make_tt_fill("f1", "4", "150")]))

    events.apply(_fill_snapshot(1001, "Cancelled", [make_tt_fill("f1", "4", "150")]))  # already counted, status ignored

    assert order.is_active()
    assert order.filled_quantity == D(0)
    assert events.tt_id_of(order) == "2002"


def test_a_replacement_without_replaces_order_id_seen_during_a_replace_is_still_adopted(tracker: OrderTracker, events: TastytradeOrderEvents, seen: list[OrderEvent]) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)  # the PUT is in flight: the stream is ahead of its answer

    events.apply(make_tt_order(order_id=2002, status="Live", external="s:abc"))  # no replaces-order-id
    events.finish_replace(order, "2002")
    events.apply(make_tt_order(order_id=2002, status="Cancelled", external="s:abc"))

    assert events.tt_id_of(order) == "2002"
    assert order.status is OrderStatus.CANCELED  # the replacement's own statuses are not ignored
    assert seen.count(OrderEvent.CANCELED) == 1


def test_finish_replace_always_makes_the_new_id_live(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)
    events.begin_replace(order)
    events.register_superseded(order, "2002")  # wrongly superseded earlier

    events.finish_replace(order, "2002")
    events.apply(make_tt_order(order_id=2002, status="Rejected", external="s:abc"))

    assert order.status is OrderStatus.ERROR


def test_an_unknown_id_with_no_replace_in_flight_stays_superseded(tracker: OrderTracker, events: TastytradeOrderEvents) -> None:
    order = _tracked(tracker, events)  # nothing superseded

    events.apply(make_tt_order(order_id=900, status="Cancelled", external="s:abc"))

    assert events.tt_id_of(order) == "1001"
    assert order.is_active()

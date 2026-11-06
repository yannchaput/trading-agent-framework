from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig, _spy_sells

from trading_agent_framework.strategies.vwap_pullback.desk import STOPPED_OUT
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus
from trading_agent_framework.utils.errors import BrokerError


def _open(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)
    rig.open_trade()
    return rig


def test_releasing_a_stop_that_already_filled_reports_it_and_sells_nothing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    rig.advance(120)  # the stop fills in the broker; the hook has not reached the desk yet
    assert rig.desk._release_stop(trade) == STOPPED_OUT
    assert trade.exit_order_ids == []


def test_the_flatten_sells_nothing_when_the_stop_already_filled(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.advance(120)  # the stop fills at the broker; its hook has not reached the desk yet
    attempts = _spy_sells(rig)
    rig.desk.flatten_all("end-of-day flatten")
    assert attempts == []


def test_a_stop_that_cannot_be_placed_reports_false(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk._submit_stop(trade, D(10_000)) is False  # more than is held: refused, and so is the market-sell fallback


def test_a_pending_full_exit_frees_its_slot_before_the_sell_fills(tmp_path: Path) -> None:
    # Final review I3 (spec §4): pending entries count as taken, pending full exits as freed.
    rig = _open(tmp_path)
    for symbol in ("BBB", "CCC", "DDD"):
        rig.state.book.add(
            Trade(
                symbol=symbol,
                entry_order_id=f"entry-{symbol}",
                planned_quantity=D(10),
                stop_price=D(90),
                r_per_share=D(1),
                entered_at=rig.clock.now(),
                status=TradeStatus.OPEN,
                quantity=D(10),
            )
        )
    assert rig.desk.free_slots() == 0
    trade = rig.state.book.get("AAA")
    assert rig.desk._release_stop(trade) is None
    sell = rig.desk._market_sell(trade, D(249), "flatten")
    assert sell is not None and sell.is_active() and trade.status is TradeStatus.OPEN
    assert rig.desk.free_slots() == 1


def test_releasing_a_stop_that_already_ended_part_filled_books_the_partial(tmp_path: Path) -> None:
    # Final review M1: a stop already cancelled after a partial fill, its hook not seen yet, when the flatten releases it.
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    stop.filled_quantity, stop.avg_fill_price = D(100), D("99.30")
    rig.broker.cancel_order(stop)
    assert rig.desk._release_stop(trade) is None
    assert trade.quantity == D(149) and trade.realised_pnl == D("-70.00") and trade.stop_order_id is None


def test_a_failed_stop_lookup_is_a_refused_exit_with_a_warning(tmp_path: Path) -> None:
    # Approved change (Task 11): a lookup that raises is not an exception out of the desk, nor "no stop".
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    stop_id = trade.stop_order_id
    warnings: list[str] = []
    rig.strategy.log_warning = warnings.append

    def failing_lookup(identifier: str):
        raise RuntimeError("broker down")

    rig.strategy.get_order = failing_lookup
    released = rig.desk._release_stop(trade)
    assert released == "could not look up the stop: broker down"
    assert trade.stop_order_id == stop_id
    assert any("could not look up the stop" in message for message in warnings)


def test_a_cancel_not_confirmed_in_time_is_not_expected_so_its_late_cancel_replaces_the_stop(tmp_path: Path) -> None:
    # Approved change (Task 11): the timeout rule of earnings_drift.
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    rig.strategy.cancel_order = lambda order: None  # the cancel request goes out ...
    rig.strategy.wait_for_order_execution = lambda order, timeout=None: False  # ... and is not confirmed within the wait
    assert rig.desk._release_stop(trade) == "the stop cancel was not confirmed in time; nothing else was changed"
    assert stop.identifier not in rig.desk._expected_cancels
    rig.broker.cancel_order(stop)  # the late confirmation
    rig.desk.on_order_canceled(stop)
    assert trade.stop_order_id is not None and trade.stop_order_id != stop.identifier
    assert rig.strategy.get_order(trade.stop_order_id).is_active()


def test_a_stop_whose_cancel_was_refused_in_the_flatten_is_replaced_when_it_later_ends(tmp_path: Path) -> None:
    # Task 11 fix: only a TIMED-OUT cancel is re-expected by the flatten; a refused cancel leaves the stop live, and when
    # it later ends (a DAY stop expiring at the close) that CANCELED is an outside cancel and the stop is placed again.
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)

    def refused(order) -> None:
        raise BrokerError("cancel refused")

    rig.strategy.cancel_order = refused
    rig.desk.flatten_all("end of day")
    assert stop.is_active() and trade.stop_order_id == stop.identifier
    rig.broker.cancel_order(stop)  # the stop expires later
    rig.desk.on_order_canceled(stop)
    assert trade.stop_order_id is not None and trade.stop_order_id != stop.identifier
    assert rig.strategy.get_order(trade.stop_order_id).is_active()

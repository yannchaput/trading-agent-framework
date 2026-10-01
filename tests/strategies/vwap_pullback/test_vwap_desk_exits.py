from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig

from trading_agent_framework.strategies.vwap_pullback.desk import STOPPED_OUT
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus


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


def test_a_stop_that_cannot_be_placed_reports_false(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk._submit_stop(trade, D(10_000)) is False  # more than is held: refused, and so is the market-sell fallback


def test_a_pending_full_exit_frees_its_slot_before_the_sell_fills(tmp_path: Path) -> None:
    # Final review I3 (spec §4): pending entries count as taken, pending full exits as freed.
    rig = _open(tmp_path)
    for symbol in ("BBB", "CCC", "DDD"):
        rig.state.book.add(Trade(
            symbol=symbol, entry_order_id=f"entry-{symbol}", planned_quantity=D(10), stop_price=D(90), r_per_share=D(1),
            entered_at=rig.clock.now(), status=TradeStatus.OPEN, quantity=D(10),
        ))
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

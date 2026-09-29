from __future__ import annotations

import dataclasses
from decimal import Decimal as D

from tests.fakes import et

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeBook, TradeStatus, exit_review_due, trade_flags

PARAMS = VwapPullbackParameters()
T0 = et(2026, 9, 1, 10, 0)


def _trade(symbol: str = "AAA") -> Trade:
    return Trade(symbol=symbol, entry_order_id=f"{symbol}-entry", planned_quantity=D(100), stop_price=D("99.00"), r_per_share=D("1.00"), catalyst="earnings", reason="clean pullback", entered_at=T0)


def test_a_trade_lifecycle_records_pnl_and_closes() -> None:
    trade = _trade()
    assert trade.status is TradeStatus.PENDING and trade.stop_level == D("99.00")
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.status is TradeStatus.OPEN and trade.quantity == D(100)
    trade.record_exit_fill(D(50), D("101.00"), et(2026, 9, 1, 10, 30))
    assert trade.realised_pnl == D("50.00") and trade.quantity == D(50)
    trade.record_exit_fill(D(50), D("99.00"), et(2026, 9, 1, 11, 0))
    assert trade.status is TradeStatus.CLOSED
    assert trade.realised_pnl == D("0.00")
    assert trade.closed_at == et(2026, 9, 1, 11, 0)
    row = trade.to_json()
    assert row["symbol"] == "AAA" and row["realised_pnl"] == "0.00" and row["catalyst"] == "earnings"


def test_unrealised_pnl_and_r() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.unrealised_pnl(D("101.50")) == D("150.00")
    assert trade.unrealised_r(D("101.50")) == 1.5


def test_book_indexes_trades_by_every_order_id_and_sums_session_pnl() -> None:
    book = TradeBook()
    a, b = _trade("AAA"), _trade("BBB")
    book.add(a)
    book.add(b)
    a.record_entry_fill(D(100), D("100.00"))
    a.stop_order_id = "AAA-stop"
    a.exit_order_ids.append("AAA-exit")
    assert book.by_order_id("AAA-stop") is a and book.by_order_id("AAA-exit") is a and book.by_order_id("BBB-entry") is b
    assert book.open_trades() == [a] and book.pending() == [b]
    a.record_exit_fill(D(100), D("101.00"), T0)
    book.archive(a)
    book.discard("BBB")
    assert book.active() == [] and book.closed == [a]
    assert book.session_pnl({}) == D("100.00")


def test_exit_review_is_due_on_a_new_flag_a_new_headline_or_elapsed_time() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    flags = trade_flags(trade, last_close=D("101.10"), vwap=100.5, ema=100.8)
    assert flags == frozenset({"reached_1r"})
    assert exit_review_due(trade, flags, now=T0, has_new_headline=False, params=PARAMS)
    trade.review_flags, trade.last_review_at = flags, T0
    assert not exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 5), has_new_headline=False, params=PARAMS)
    assert exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 5), has_new_headline=True, params=PARAMS)
    assert exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 15), has_new_headline=False, params=PARAMS)
    below = trade_flags(trade, last_close=D("100.40"), vwap=100.5, ema=100.8)
    assert below == frozenset({"below_vwap", "below_ema"})
    tp1 = dataclasses.replace(trade, tp1_done=True)
    assert trade_flags(tp1, last_close=D("101.10"), vwap=100.5, ema=100.8) == frozenset()

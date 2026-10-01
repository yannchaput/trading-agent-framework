from __future__ import annotations

from decimal import Decimal as D

from tests.fakes import et

from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeBook, TradeStatus

T0 = et(2026, 9, 1, 10, 0)


def _trade(symbol: str = "AAA") -> Trade:
    return Trade(symbol=symbol, entry_order_id=f"{symbol}-entry", planned_quantity=D(100), stop_price=D("99.00"), r_per_share=D("1.00"), entered_at=T0)


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
    assert row["symbol"] == "AAA" and row["realised_pnl"] == "0.00" and "catalyst" not in row and "reason" not in row


def test_unrealised_pnl() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.unrealised_pnl(D("101.50")) == D("150.00")


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


def test_to_json_has_exactly_the_trade_log_fields() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    trade.exit_reason = "stop"
    trade.record_exit_fill(D(100), D("99.00"), et(2026, 9, 1, 11, 0))
    assert trade.to_json() == {
        "symbol": "AAA",
        "entered_at": T0.isoformat(),
        "closed_at": et(2026, 9, 1, 11, 0).isoformat(),
        "entry_price": "100.00",
        "filled_quantity": "100",
        "stop_price": "99.00",
        "r_per_share": "1.00",
        "realised_pnl": "-100.00",
        "realised_r": -1.0,
        "exit_reason": "stop",
    }

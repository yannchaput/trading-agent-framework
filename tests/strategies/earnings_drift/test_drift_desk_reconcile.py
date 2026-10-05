from __future__ import annotations

import logging
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import RIG_DATES, DeskRig

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.strategies.earnings_drift.book import TradeState
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.utils.errors import BrokerError


def _lookup_raises_for(rig: DeskRig, monkeypatch: pytest.MonkeyPatch, order_id: str | None) -> None:
    """`strategy.get_order` fails for one order id (a transient broker error) and works for every other."""
    real = rig.strategy.get_order

    def get_order(identifier: str):  # noqa: ANN202
        if identifier == order_id:
            raise BrokerError("transient broker error")
        return real(identifier)

    monkeypatch.setattr(rig.strategy, "get_order", get_order)


def test_max_hold_sells_at_the_next_open(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_holding_sessions=2))
    rig.open_aaa()  # filled day 1
    assert rig.desk.reconcile() == []  # 1 session held
    rig.next_close()  # day 2: 2 sessions held
    with caplog.at_level(logging.WARNING):
        assert rig.desk.reconcile() == ["AAA"]
    assert "guardrail max_hold: AAA held 2 sessions" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.exit_reason == "max_hold" and trade.stop_order_id is None
    rig.next_close()  # day 3: the sell fills at the open (104)
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "max_hold" and D(line["exit_price"]) == D("104.0")


def test_ensure_stops_replaces_a_stop_that_vanished_without_a_hook(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]  # its CANCELED hook never reaches the desk
    with caplog.at_level(logging.WARNING):
        rig.desk.ensure_stops()
    assert "guardrail stop_backstop: AAA has no working stop" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.backstop and rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]
    assert rig.desk.holdings_context()[0]["stop_status"] == "backstop"
    assert rig.desk.holdings_context()[0]["stop_status"] == "working"  # shown once


def test_reconcile_settles_a_fill_whose_hook_was_missed(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    rig.advance()  # filled at the broker, hook not delivered
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [], {})
    rig.desk.reconcile()
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_reconcile_drops_an_entry_rejected_at_fill_time(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, budget=D("2000"))
    assert "error" not in rig.desk.buy("AAA", 2, 8.0, "x")  # 2 x 100 passes the submission check...
    entry = rig.strategy.get_order(rig.state.trades["AAA"].entry_order_id)
    entry.quantity = D(25)  # type: ignore[union-attr]  # ...then grows past the cash, so the fill is rejected whole
    rig.advance()
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [], {})
    rig.desk.reconcile()
    assert rig.state.trades == {}


def test_a_position_never_traded_by_this_strategy_is_left_alone(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))  # bought outside the desk
    rig.next_close()
    with caplog.at_level(logging.INFO):
        rig.desk.reconcile()
        rig.desk.reconcile()
    assert "BBB" not in rig.state.trades
    assert caplog.text.count("never traded by this strategy") == 1


def test_an_orphan_of_a_traded_symbol_is_adopted_with_a_stop(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.state.traded_symbols.add("BBB")
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))
    rig.next_close()
    with caplog.at_level(logging.WARNING):
        rig.desk.reconcile()
    assert "guardrail orphan: adopted 10 BBB" in caplog.text
    trade = rig.state.trades["BBB"]
    assert trade.state is TradeState.OPEN and trade.trail_percent == D("8.0") and trade.opened_on == RIG_DATES[1]
    assert rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_a_traded_symbol_without_a_position_is_not_adopted(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.state.traded_symbols.add("BBB")  # recorded before a buy that never reached the broker
    rig.desk.reconcile()
    assert rig.state.trades == {}


def test_an_orphan_with_a_working_sell_adopts_it_as_its_stop(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.state.traded_symbols.add("BBB")
    rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "buy"))
    rig.next_close()
    working = rig.strategy.submit_order(rig.strategy.create_order("BBB", 10, "sell", trail_percent=6, time_in_force="gtc"))
    rig.desk.reconcile()
    trade = rig.state.trades["BBB"]
    assert trade.stop_order_id == working.identifier and trade.trail_percent == D("6")
    sells = [o for o in rig.strategy.get_orders() if o.side is OrderSide.SELL and o.order_type is OrderType.TRAIL]
    assert sells == [working]


def test_a_position_gone_outside_the_strategy_closes_the_trade_as_unknown(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]
    rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell"))  # sold by hand
    rig.advance()
    rig.desk.begin_session(RIG_DATES[2], RIG_DATES[:3], [], {})
    rig.desk.reconcile()
    assert rig.state.trades == {}
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "unknown"


# --- a failed lookup is not a missing order (controller ruling) -------------------------------------


def test_ensure_stops_does_not_place_a_second_stop_when_the_stop_lookup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop_id = rig.state.trades["AAA"].stop_order_id
    orders_before = len(rig.strategy.get_orders())
    _lookup_raises_for(rig, monkeypatch, stop_id)
    with caplog.at_level(logging.WARNING):
        rig.desk.ensure_stops()
    monkeypatch.undo()
    assert rig.state.trades["AAA"].stop_order_id == stop_id and not rig.state.trades["AAA"].backstop
    assert len(rig.strategy.get_orders()) == orders_before
    assert "AAA" in caplog.text and stop_id in caplog.text  # type: ignore[operator]


def test_reconcile_keeps_a_pending_entry_whose_lookup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    rig.advance()
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [], {})
    _lookup_raises_for(rig, monkeypatch, rig.state.trades["AAA"].entry_order_id)
    rig.desk.reconcile()
    assert rig.state.trades["AAA"].state is TradeState.PENDING
    monkeypatch.undo()
    rig.desk.reconcile()  # the lookup works again: the fill is settled
    assert rig.state.trades["AAA"].state is TradeState.OPEN


def test_reconcile_drops_a_pending_entry_the_broker_does_not_know(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    entry_id = rig.state.trades["AAA"].entry_order_id
    real = rig.strategy.get_order
    monkeypatch.setattr(rig.strategy, "get_order", lambda identifier: None if identifier == entry_id else real(identifier))
    rig.desk.reconcile()
    assert rig.state.trades == {}


def test_reconcile_keeps_a_trade_whose_stop_lookup_fails_in_the_settle_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop_id = rig.state.trades["AAA"].stop_order_id
    _lookup_raises_for(rig, monkeypatch, stop_id)
    rig.desk.reconcile()
    trade = rig.state.trades["AAA"]
    assert trade.stop_order_id == stop_id and trade.state is TradeState.OPEN


def test_a_vanished_position_is_closed_even_when_the_stop_lookup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop_id = rig.state.trades["AAA"].stop_order_id
    stop = rig.strategy.get_order(stop_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]
    rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell"))  # sold by hand
    rig.advance()
    rig.desk.begin_session(RIG_DATES[2], RIG_DATES[:3], [], {})
    _lookup_raises_for(rig, monkeypatch, stop_id)
    with caplog.at_level(logging.WARNING):
        rig.desk.reconcile()
    assert rig.state.trades == {}  # the position is gone whatever the stop's state: the trade closes
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "unknown"

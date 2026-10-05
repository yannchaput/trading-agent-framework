from __future__ import annotations

import logging
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import RIG_DATES, DeskRig

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.strategies.earnings_drift.book import DriftState, JsonlLog, TradeState
from trading_agent_framework.strategies.earnings_drift.desk import Desk
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.utils.errors import BrokerError


def test_an_empty_date_list_keeps_the_known_dates_so_max_hold_still_fires(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_holding_sessions=2))
    rig.open_aaa()  # filled day 1; the dates known so far are days 0 and 1
    rig.advance()  # day 2: a cycle whose scan failed begins with no dates at all
    rig.deliver()
    rig.desk.begin_session(RIG_DATES[2], [], [], {})
    assert rig.desk._trading_dates == RIG_DATES[:3]
    assert rig.desk.reconcile() == ["AAA"]  # 2 sessions held
    assert rig.state.trades["AAA"].exit_reason == "max_hold"


def test_a_first_session_with_no_dates_knows_today(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    fresh = Desk(rig.strategy, DriftParams(), DriftState(), trade_log=JsonlLog(lambda: None), decision_log=JsonlLog(lambda: None))
    fresh.begin_session(RIG_DATES[0], [], [], {})
    assert fresh._trading_dates == [RIG_DATES[0]]


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
    assert "AAA" in rig.state.trades  # absent once: could be a stale positions list
    rig.desk.reconcile()
    assert rig.state.trades == {}  # absent on two consecutive reconciles
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


def _positions_empty(rig: DeskRig, monkeypatch: pytest.MonkeyPatch, calls: list[int] | None = None, *, times: int | None = None) -> None:
    """`strategy.get_positions` answers an empty list (a stale snapshot), `times` times (always when None), counting the calls."""
    real = rig.strategy.get_positions
    counter = calls if calls is not None else []

    def get_positions():  # noqa: ANN202
        counter.append(1)
        if times is None or len(counter) <= times:
            return []
        return real()

    monkeypatch.setattr(rig.strategy, "get_positions", get_positions)


def test_a_failed_stop_lookup_keeps_a_trade_whose_position_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop_id = rig.state.trades["AAA"].stop_order_id
    orders_before = len(rig.strategy.get_orders())
    _positions_empty(rig, monkeypatch)
    rig.desk.reconcile()  # absent once
    _lookup_raises_for(rig, monkeypatch, stop_id)
    rig.desk.reconcile()  # absent twice, but the stop may be live: nothing changes
    rig.desk.ensure_stops()
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and trade.stop_order_id == stop_id
    assert len(rig.strategy.get_orders()) == orders_before and rig.lines(rig.trades_path) == []
    monkeypatch.undo()
    _positions_empty(rig, monkeypatch)
    rig.desk.reconcile()  # the lookup works again: the stop is cancelled and the trade closes
    assert rig.state.trades == {}
    assert not rig.strategy.get_order(stop_id).is_active()  # type: ignore[arg-type, union-attr]
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "unknown"


def test_an_empty_positions_snapshot_on_one_reconcile_closes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    stop_id = rig.state.trades["AAA"].stop_order_id
    orders_before = len(rig.strategy.get_orders())
    _positions_empty(rig, monkeypatch, times=1)
    with caplog.at_level(logging.WARNING):
        rig.desk.reconcile()
        rig.desk.ensure_stops()  # the strategy calls it again after the agent run
    assert "will close as unknown if still absent next cycle" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and trade.stop_order_id == stop_id
    assert rig.strategy.get_order(stop_id).is_active()  # type: ignore[union-attr]
    assert len(rig.strategy.get_orders()) == orders_before and rig.lines(rig.trades_path) == []
    rig.desk.reconcile()  # the real positions are back
    assert rig.desk._missing_once == set()
    assert rig.state.trades["AAA"].state is TradeState.OPEN and rig.lines(rig.trades_path) == []


def test_reconcile_reads_the_positions_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    calls: list[int] = []
    real = rig.strategy.get_positions

    def counting():  # noqa: ANN202
        calls.append(1)
        return real()

    monkeypatch.setattr(rig.strategy, "get_positions", counting)
    rig.desk.reconcile()
    assert len(calls) == 1
    rig.desk.reconcile()
    assert len(calls) == 2

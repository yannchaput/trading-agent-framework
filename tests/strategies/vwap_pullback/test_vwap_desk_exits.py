from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig

from trading_agent_framework.entities.enums import OrderType


def _open(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)
    rig.open_trade()
    return rig


def test_take_partial_profit_sells_the_fraction_and_resizes_the_stop(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    old_stop = rig.strategy.get_order(trade.stop_order_id)
    result = rig.desk.take_partial_profit("AAA", 0.5)
    assert result == {"status": "partial profit taken", "sold": 124, "remaining": 125, "stop_price": 99.3}
    assert old_stop.is_canceled()
    sell = rig.strategy.get_order(trade.exit_order_ids[-1])
    assert sell.order_type is OrderType.MARKET and sell.quantity == D(124)
    new_stop = rig.strategy.get_order(trade.stop_order_id)
    assert new_stop.quantity == D(125) and new_stop.stop_price == D("99.30")
    assert trade.tp1_done
    assert "already" in rig.desk.take_partial_profit("AAA", 0.5)["error"]
    assert "between" in rig.desk.take_partial_profit("AAA", 0.9)["error"]


def test_tighten_stop_only_moves_up(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk.tighten_stop("AAA", 99.8) == {"status": "stop raised", "stop_price": 99.8}
    assert trade.stop_level == D("99.80")
    assert rig.strategy.get_order(trade.stop_order_id).stop_price == D("99.80")
    assert "only move up" in rig.desk.tighten_stop("AAA", 99.5)["error"]
    assert "below the last price" in rig.desk.tighten_stop("AAA", 100.5)["error"]


def test_replace_stop_with_trailing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    old_stop = rig.strategy.get_order(trade.stop_order_id)
    result = rig.desk.replace_stop_with_trailing("AAA", 1.0)  # 5-minute ATR 0.4 -> trail 0.40, starting at 99.60
    assert result == {"status": "trailing stop placed", "trail_price": 0.4, "starts_at": 99.6}
    assert old_stop.is_canceled()
    trail = rig.strategy.get_order(trade.stop_order_id)
    assert trail.order_type is OrderType.TRAIL and trail.trail_price == D("0.40") and trail.quantity == D(249)
    assert trade.stop_kind == "trail"
    assert "already a trailing stop" in rig.desk.tighten_stop("AAA", 99.9)["error"]


def test_a_trail_starting_below_the_current_stop_is_refused(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.desk.tighten_stop("AAA", 99.8)
    assert "below the current stop" in rig.desk.replace_stop_with_trailing("AAA", 1.0)["error"]  # would start at 99.60


def test_exit_position_releases_the_stop_and_sells_everything(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk.exit_position("AAA", "downgrade headline") == {"status": "exit submitted", "quantity": 249}
    assert trade.stop_order_id is None and trade.exit_reason == "downgrade headline"
    assert rig.strategy.get_order(trade.exit_order_ids[-1]).quantity == D(249)


def test_a_stop_that_already_filled_is_reported_not_sold_twice(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    rig.advance(120)  # the stop fills in the broker; the hook has not reached the desk yet
    assert rig.desk.exit_position("AAA", "x") == {"status": "already_stopped_out"}
    assert trade.exit_order_ids == []


def test_hold_and_unknown_symbols(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    assert rig.desk.hold("AAA", "inside 1R") == {"symbol": "AAA", "status": "holding"}
    assert "no open trade" in rig.desk.exit_position("ZZZ", "x")["error"]

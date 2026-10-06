from __future__ import annotations

import logging
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import DeskRig

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.utils.errors import OrderValidationError


def _open(tmp_path: Path) -> DeskRig:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    return rig


def test_tightening_replaces_the_stop(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    old = rig.state.trades["AAA"].stop_order_id
    result = rig.desk.set_trailing_stop("AAA", 5.0, "lock in gains")
    assert result == {"symbol": "AAA", "trail_percent": 5.0, "status": "stop replaced"}
    trade = rig.state.trades["AAA"]
    assert trade.trail_percent == D("5.0") and trade.stop_order_id != old
    assert rig.strategy.get_order(old).is_canceled()  # type: ignore[union-attr]
    new = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert new is not None and new.is_active() and new.trail_percent == D("5.0")


def test_widening_is_refused_and_the_same_trail_is_unchanged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = _open(tmp_path)
    with caplog.at_level(logging.WARNING):
        assert "tighten-only" in rig.desk.set_trailing_stop("AAA", 9.0, "more room")["error"]
    assert "guardrail order_limits" in caplog.text
    assert rig.desk.set_trailing_stop("AAA", 8.0, "same")["status"] == "unchanged"
    assert "between" in rig.desk.set_trailing_stop("AAA", 2.0, "too tight")["error"]
    assert "no open position" in rig.desk.set_trailing_stop("BBB", 5.0, "x")["error"]


def test_a_refused_new_stop_puts_the_old_trail_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    real_submit = rig.strategy.submit_order

    def refuse_five(order):  # noqa: ANN001, ANN202
        if order.order_type is OrderType.TRAIL and order.trail_percent == D("5.0"):
            raise OrderValidationError("no")
        return real_submit(order)

    monkeypatch.setattr(rig.strategy, "submit_order", refuse_five)
    assert "the 8.0% trail was placed again" in rig.desk.set_trailing_stop("AAA", 5.0, "x")["error"]
    trade = rig.state.trades["AAA"]
    stop = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert trade.trail_percent == D("8.0") and stop is not None and stop.is_active() and stop.trail_percent == D("8.0")


def test_sell_cancels_the_stop_and_sells_at_the_next_open(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    stop_id = rig.state.trades["AAA"].stop_order_id
    result = rig.desk.sell("AAA", "thesis broken")
    assert result["side"] == "sell"
    assert rig.strategy.get_order(stop_id).is_canceled()  # type: ignore[union-attr]
    assert rig.desk.free_slots() == 8  # a pending exit frees its slot
    assert "no open position" in rig.desk.sell("AAA", "again")["error"]  # being sold
    rig.next_close()
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "agent_sell" and D(line["exit_price"]) == D("102.0")


def test_sell_after_the_stop_filled_sells_nothing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.next_close()
    rig.advance()  # day 3: the stop fills at the broker; its hook has not reached the desk yet
    sells_before = [o for o in rig.strategy.get_orders() if o.side.value == "sell"]
    assert rig.desk.sell("AAA", "x") == {"status": "already_closed"}
    assert rig.desk.set_trailing_stop("AAA", 5.0, "x") == {"status": "already_closed"}
    assert [o for o in rig.strategy.get_orders() if o.side.value == "sell"] == sells_before


def test_a_stop_cancelled_outside_is_placed_again(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    rig = _open(tmp_path)
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING):
        rig.desk.on_order_canceled(stop)  # type: ignore[arg-type]
    assert "guardrail stop_backstop" in caplog.text
    trade = rig.state.trades["AAA"]
    assert trade.stop_order_id != stop.identifier and trade.backstop  # type: ignore[union-attr]
    assert rig.strategy.get_order(trade.stop_order_id).is_active()  # type: ignore[arg-type, union-attr]


def test_the_desk_s_own_cancel_hook_is_ignored(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    rig.desk.set_trailing_stop("AAA", 5.0, "x")
    new_id = rig.state.trades["AAA"].stop_order_id
    rig.desk.on_order_canceled(stop)  # type: ignore[arg-type]  # delivered late, as the executor would
    assert rig.state.trades["AAA"].stop_order_id == new_id


def test_an_unfilled_entry_that_ends_is_dropped_and_a_partial_one_is_protected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.buy("AAA", 10, 8.0, "x")
    rig.desk.buy("BBB", 10, 8.0, "y")
    aaa = rig.strategy.get_order(rig.state.trades["AAA"].entry_order_id)
    bbb = rig.strategy.get_order(rig.state.trades["BBB"].entry_order_id)
    protected: list[str] = []
    monkeypatch.setattr(rig.desk, "_protect", lambda trade, trail: protected.append(trade.symbol) or True)
    rig.broker.cancel_order(aaa)  # type: ignore[arg-type]
    rig.desk.on_order_canceled(aaa)  # type: ignore[arg-type]
    bbb.filled_quantity, bbb.avg_fill_price = D(4), D("50.1")  # type: ignore[union-attr]
    rig.broker.cancel_order(bbb)  # type: ignore[arg-type]
    rig.desk.on_order_canceled(bbb)  # type: ignore[arg-type]
    assert "AAA" not in rig.state.trades
    assert rig.state.trades["BBB"].quantity == D(4) and protected == ["BBB"]


def test_a_cancel_not_confirmed_in_time_does_not_hide_a_late_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = _open(tmp_path)
    stop = rig.strategy.get_order(rig.state.trades["AAA"].stop_order_id)  # type: ignore[arg-type]
    sells_before = [o.identifier for o in rig.strategy.get_orders() if o.side.value == "sell"]
    real_cancel = rig.strategy.cancel_order
    monkeypatch.setattr(rig.strategy, "cancel_order", lambda order: None)
    monkeypatch.setattr(rig.strategy, "wait_for_order_execution", lambda order, timeout=None: False)
    with caplog.at_level(logging.WARNING):
        assert "not confirmed in time" in rig.desk.sell("AAA", "x")["error"]
    assert "AAA" in caplog.text
    assert [o.identifier for o in rig.strategy.get_orders() if o.side.value == "sell"] == sells_before
    assert stop.identifier not in rig.desk._expected_cancels  # type: ignore[union-attr]  # noqa: SLF001
    monkeypatch.setattr(rig.strategy, "cancel_order", real_cancel)
    rig.broker.cancel_order(stop)  # type: ignore[arg-type]  # the cancel lands late
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rig.desk.on_order_canceled(stop)  # type: ignore[arg-type]
    assert "guardrail stop_backstop" in caplog.text
    trade = rig.state.trades["AAA"]
    new = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert trade.stop_order_id != stop.identifier and trade.backstop  # type: ignore[union-attr]
    assert new is not None and new.is_active() and new.order_type is OrderType.TRAIL


def test_a_failed_stop_lookup_sells_and_places_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    stop_id = rig.state.trades["AAA"].stop_order_id
    real_get = rig.strategy.get_order

    def failing_get(identifier: str):  # noqa: ANN202
        if identifier == stop_id:
            raise RuntimeError("503")
        return real_get(identifier)

    monkeypatch.setattr(rig.strategy, "get_order", failing_get)
    orders_before = [o.identifier for o in rig.strategy.get_orders()]
    assert "could not look up the stop" in rig.desk.sell("AAA", "x")["error"]
    assert "could not look up the stop" in rig.desk.set_trailing_stop("AAA", 5.0, "x")["error"]
    assert [o.identifier for o in rig.strategy.get_orders()] == orders_before
    assert rig.state.trades["AAA"].stop_order_id == stop_id


# --- B5: truthful messages when the stop could not be put back --------------------------------------


def _refusing(rig: DeskRig, monkeypatch: pytest.MonkeyPatch, *, market_sells_refused: int, trails_refused: bool = True, trail_refused_below: D | None = None) -> None:
    """Refuse the first `market_sells_refused` market sells, and every trailing stop (or only those tighter than a trail)."""
    real_submit = rig.strategy.submit_order
    refused = {"market": 0}

    def submit(order):  # noqa: ANN001, ANN202
        if order.order_type is OrderType.TRAIL and trails_refused and (trail_refused_below is None or order.trail_percent < trail_refused_below):
            raise OrderValidationError("stops are down")
        if order.order_type is OrderType.MARKET and order.side is OrderSide.SELL and refused["market"] < market_sells_refused:
            refused["market"] += 1
            raise OrderValidationError("sells are down")
        return real_submit(order)

    monkeypatch.setattr(rig.strategy, "submit_order", submit)


def test_a_refused_sell_whose_stop_cannot_come_back_says_a_backstop_sell_was_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    _refusing(rig, monkeypatch, market_sells_refused=1)
    result = rig.desk.sell("AAA", "thesis broken")
    assert "backstop market sell" in result["error"] and "placed again" not in result["error"]
    trade = rig.state.trades["AAA"]
    assert trade.exit_reason == "backstop_sell" and trade.exit_order_id is not None
    assert [line["decision"] for line in rig.lines(rig.decisions_path)] == ["buy"]


def test_a_refused_sell_with_nothing_placed_says_the_position_has_no_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    _refusing(rig, monkeypatch, market_sells_refused=2)
    result = rig.desk.sell("AAA", "thesis broken")
    assert "no stop" in result["error"] and "placed again" not in result["error"]
    assert rig.state.trades["AAA"].exit_order_id is None and rig.state.trades["AAA"].stop_order_id is None


def test_a_refused_sell_whose_stop_comes_back_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    _refusing(rig, monkeypatch, market_sells_refused=1, trails_refused=False)
    assert rig.desk.sell("AAA", "x") == {"error": "the sell was refused; the stop was placed again"}


def test_a_refused_new_trail_whose_old_one_cannot_come_back_says_a_backstop_sell_was_sent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = _open(tmp_path)
    _refusing(rig, monkeypatch, market_sells_refused=0)
    result = rig.desk.set_trailing_stop("AAA", 5.0, "tighten")
    assert "backstop market sell" in result["error"] and "placed again" not in result["error"]
    assert rig.state.trades["AAA"].exit_reason == "backstop_sell"


def test_a_sell_that_finds_the_trade_already_closed_logs_no_sell_decision(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.next_close()
    rig.advance()  # day 3: the stop fills at the broker; its hook has not reached the desk yet
    assert rig.desk.sell("AAA", "x") == {"status": "already_closed"}
    assert [line["decision"] for line in rig.lines(rig.decisions_path)] == ["buy"]

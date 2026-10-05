from __future__ import annotations

import logging
import threading
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.strategies.earnings_drift.drift_helpers import RIG_DATES, DeskRig, make_candidate

from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.strategies.earnings_drift.book import TradeState
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.utils.errors import OrderValidationError


def test_max_quantity_is_the_slot_budget_over_the_reaction_close(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    assert rig.desk.max_quantity("AAA") == 125  # 100000 / 8 / 100
    assert rig.desk.max_quantity("BBB") == 250
    assert rig.desk.max_quantity("ZZZ") == 0


def test_buy_submits_a_day_market_order_and_records_a_pending_trade(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    result = rig.desk.buy("aaa", 10, 8.0, "beat and raise")
    assert result["symbol"] == "AAA" and result["trail_percent"] == 8.0
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.PENDING and trade.trail_percent == D("8.0") and trade.thesis == "beat and raise"
    assert trade.reaction_low == D("99.0") and trade.accession_number == "0000000000-26-AAA"
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order is not None and order.side is OrderSide.BUY and order.order_type is OrderType.MARKET and order.time_in_force is TimeInForce.DAY
    assert rig.state.traded_symbols == {"AAA"}
    assert rig.desk.max_quantity("BBB") == 250  # still the 12500 slot cap: 99000 is left after the 1000 committed
    [line] = rig.lines(rig.decisions_path)
    assert line["decision"] == "buy" and line["quantity"] == 10 and line["features"]["abnormal_pct"] == 0.075


@pytest.mark.parametrize(
    ("symbol", "quantity", "trail", "message"),
    [
        ("ZZZ", 10, 8.0, "not one of today's candidates"),
        ("AAA", 2.5, 8.0, "whole number"),
        ("AAA", 0, 8.0, "whole number"),
        ("AAA", -3, 8.0, "whole number"),
        ("AAA", float("nan"), 8.0, "whole number"),
        ("AAA", "ten", 8.0, "whole number"),
        ("AAA", 10, 2.9, "trail_percent"),
        ("AAA", 10, 15.1, "trail_percent"),
        ("AAA", 126, 8.0, "above max_quantity 125"),
    ],
)
def test_buy_refuses_bad_quantities_and_trails(tmp_path: Path, caplog: pytest.LogCaptureFixture, symbol: str, quantity: object, trail: float, message: str) -> None:
    rig = DeskRig(tmp_path)
    with caplog.at_level(logging.WARNING):
        result = rig.desk.buy(symbol, quantity, trail, "x")  # type: ignore[arg-type]
    assert message in result["error"]
    assert "guardrail order_limits" in caplog.text
    assert rig.state.trades == {}


def test_buy_refuses_a_second_entry_and_a_full_book(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=1))
    assert "error" not in rig.desk.buy("AAA", 10, 8.0, "x")
    assert "already held or being bought" in rig.desk.buy("AAA", 1, 8.0, "x")["error"]
    assert "max_positions (1) reached" in rig.desk.buy("BBB", 1, 8.0, "x")["error"]


def test_two_buys_on_parallel_threads_respect_max_positions(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=1))
    results: list[dict] = []
    barrier = threading.Barrier(2)

    def buy(symbol: str) -> None:
        barrier.wait()
        results.append(rig.desk.buy(symbol, 5, 8.0, "x"))

    threads = [threading.Thread(target=buy, args=(s,)) for s in ("AAA", "BBB")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum("error" in r for r in results) == 1 and len(rig.state.trades) == 1


def test_a_broker_refusal_is_returned_as_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = DeskRig(tmp_path)

    def refuse(order):  # noqa: ANN001, ANN202
        raise OrderValidationError("insufficient cash")

    monkeypatch.setattr(rig.strategy, "submit_order", refuse)
    assert rig.desk.buy("AAA", 10, 8.0, "x") == {"error": "insufficient cash"}
    assert rig.state.trades == {}


def test_an_unconfirmed_buy_is_recorded_before_submission_so_it_can_be_adopted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[set[str]] = []
    rig = DeskRig(tmp_path, save=lambda: saved.append(set(rig.state.traded_symbols)))

    def raises(order):  # noqa: ANN001, ANN202
        raise OrderValidationError("timeout")

    monkeypatch.setattr(rig.strategy, "submit_order", raises)
    result = rig.desk.buy("AAA", 10, 8.0, "x")
    assert "error" in result and rig.state.trades == {}
    assert "AAA" in rig.state.traded_symbols
    assert saved and saved[0] == {"AAA"}  # persisted before the submit, not after


def test_a_refused_buy_does_not_record_the_symbol(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    assert "error" in rig.desk.buy("AAA", 126, 8.0, "x")
    assert rig.state.traded_symbols == set()


def test_the_entry_fills_at_the_next_open_and_gets_its_trailing_stop(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    trade = rig.state.trades["AAA"]
    assert trade.state is TradeState.OPEN and trade.quantity == D(10) and trade.entry_price == D("101.0")
    assert trade.opened_on == RIG_DATES[1]
    stop = rig.strategy.get_order(trade.stop_order_id)  # type: ignore[arg-type]
    assert stop is not None and stop.order_type is OrderType.TRAIL and stop.trail_percent == D("8.0") and stop.time_in_force is TimeInForce.GTC
    assert stop.quantity == D(10) and stop.is_active()


def test_the_stop_fill_closes_the_trade_into_trades_jsonl(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    rig.next_close()
    rig.next_close()  # day 3: the low 95 crosses 106 x 0.92 = 97.52
    assert rig.state.trades == {}
    [line] = rig.lines(rig.trades_path)
    assert line["symbol"] == "AAA" and line["exit_reason"] == "trail"
    assert D(line["entry_price"]) == D("101") and D(line["exit_price"]) == D("97.52")
    assert line["sessions_held"] == 3 and line["entry_date"] == RIG_DATES[1].isoformat() and line["exit_date"] == RIG_DATES[3].isoformat()
    assert D(line["pnl"]) == D("-34.80") and line["agent_enabled"] is True and line["thesis"] == "beat and raise"


def test_a_stop_refused_twice_is_replaced_by_a_market_sell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    rig = DeskRig(tmp_path)
    assert "error" not in rig.desk.buy("AAA", 10, 8.0, "x")
    real_submit = rig.strategy.submit_order

    def no_trailing_stops(order):  # noqa: ANN001, ANN202
        if order.order_type is OrderType.TRAIL:
            raise OrderValidationError("trailing stops are down")
        return real_submit(order)

    monkeypatch.setattr(rig.strategy, "submit_order", no_trailing_stops)
    with caplog.at_level(logging.WARNING):
        rig.next_close()
    trade = rig.state.trades["AAA"]
    assert trade.stop_order_id is None and trade.exit_reason == "backstop_sell"
    sell = rig.strategy.get_order(trade.exit_order_id)  # type: ignore[arg-type]
    assert sell is not None and sell.side is OrderSide.SELL and sell.order_type is OrderType.MARKET
    rig.next_close()
    [line] = rig.lines(rig.trades_path)
    assert line["exit_reason"] == "backstop_sell"


def test_skip_and_undecided_are_recorded_once(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    assert rig.desk.skip("BBB", "guidance cut") == {"symbol": "BBB", "status": "skipped"}
    assert "already decided" in rig.desk.skip("BBB", "again")["error"]
    assert "not one of today's candidates" in rig.desk.skip("ZZZ", "x")["error"]
    rig.desk.record_undecided()
    rig.desk.record_undecided()
    decisions = [(line["symbol"], line["decision"]) for line in rig.lines(rig.decisions_path)]
    assert decisions == [("BBB", "skip"), ("AAA", "undecided")]


def test_rejections_are_logged_at_begin_session(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.desk.begin_session(RIG_DATES[0], RIG_DATES[:1], [], {"CCC": "faded"})
    assert rig.lines(rig.decisions_path) == [{"date": RIG_DATES[0].isoformat(), "symbol": "CCC", "decision": "rejected", "reason": "faded"}]


def test_baseline_buys_every_candidate_best_first_up_to_the_free_slots(tmp_path: Path) -> None:
    candidates = [make_candidate("BBB", day=RIG_DATES[0], close=50.0, abnormal_pct=0.04), make_candidate("AAA", day=RIG_DATES[0], close=100.0, abnormal_pct=0.09)]
    rig = DeskRig(tmp_path, params=DriftParams(agent_enabled=False, max_positions=1), candidates=candidates)
    assert rig.desk.baseline_entries() == ["AAA"]
    trade = rig.state.trades["AAA"]
    assert trade.trail_percent == D("8.0") and trade.thesis == "baseline"
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order is not None and order.quantity == D(1000)  # one slot: the whole 100000, at the reaction close of 100


def test_candidate_sheets_and_balances(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    sheets = rig.desk.candidate_sheets()
    assert [s["symbol"] for s in sheets] == ["AAA", "BBB"]  # best abnormal return first
    assert sheets[0]["max_quantity"] == 125
    assert rig.desk.balances() == {"portfolio_value": 100000.0, "cash": 100000.0, "buying_power": 100000.0, "free_slots": 8}


def test_holdings_context_lists_open_trades(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path)
    rig.open_aaa()
    [row] = rig.desk.holdings_context()
    assert row == {
        "symbol": "AAA",
        "entry_date": RIG_DATES[1].isoformat(),
        "entry_price": 101.0,
        "last_close": 102.0,
        "pnl_pct": 1.0,
        "sessions_held": 1,
        "trail_percent": 8.0,
        "stop_status": "working",
        "thesis": "beat and raise",
        "reaction_low": 99.0,
    }


# --- A1: the cycle's buys are deducted once (buying_power already nets them) ---------------------


def test_a_cycle_buy_is_counted_once_in_max_quantity(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=2), budget=D("10000"))
    assert rig.desk.max_quantity("AAA") == 50  # the 5000 slot cap at 100
    assert "error" not in rig.desk.buy("AAA", 50, 8.0, "x")
    account = rig.broker.get_account()
    assert account.cash == D("10000") and account.buying_power == D("5000")  # the broker already nets the pending buy
    assert rig.desk.max_quantity("BBB") == 100  # the second 5000 slot at 50, not 0
    assert "error" not in rig.desk.buy("BBB", 100, 8.0, "y")


def test_sell_proceeds_are_credited_and_cycle_buys_deducted_once(tmp_path: Path) -> None:
    rig = DeskRig(tmp_path, params=DriftParams(max_positions=1), budget=D("10000"))
    rig.open_aaa(quantity=50)  # filled at 101: cash 4950
    rig.desk.begin_session(RIG_DATES[1], RIG_DATES[:2], [make_candidate("BBB", day=RIG_DATES[1], close=50.0)], {})
    assert rig.desk.max_quantity("BBB") == 99  # 4950 left, before any sell
    assert rig.desk.sell("AAA", "rotate")["side"] == "sell"  # 50 x 102 = 5100 credited
    assert rig.desk.max_quantity("BBB") == 201  # the whole 10050 slot
    assert "error" not in rig.desk.buy("BBB", 100, 8.0, "y")
    assert rig.desk.max_quantity("BBB") == 101  # 10050 - 5000 committed = 5050, deducted once

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest
from tests.fakes import FakeClock, FrameDataSource, et, make_session, minute_ohlc

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.strategies.vwap_pullback.trades import TradeStatus

DAY = date(2026, 9, 1)
FLAT = (100.0, 100.2, 99.8, 100.0, 1000.0)
# Close-stamped minute bars 09:31..11:00: flat at 100, then a drop through 99.30 at 10:03.
ROWS = [FLAT] * 32 + [(100.0, 100.0, 99.0, 99.2, 1000.0)] + [(99.2, 99.4, 99.0, 99.2, 1000.0)] * 57


class Rig:
    def __init__(self, tmp_path: Path, *, now: datetime | None = None, params: VwapPullbackParameters | None = None) -> None:
        self.clock = FakeClock(now or et(2026, 9, 1, 10, 0), [make_session(DAY)])
        frames = {(symbol, "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS) for symbol in ("AAA", "MSFT")}
        self.broker = BacktestBroker("vwap", data_source=FrameDataSource(frames), clock=self.clock, budget=D("100000"), timestep="minute")
        self.strategy = Strategy(self.broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
        self.strategy.minutes_before_closing = 10  # as VwapPullbackStrategy: the flatten runs at 15:50
        self.strategy.vars.session = SessionState(day=DAY, session=make_session(DAY), bar_stamp="close", session_open_equity=D("100000"))
        self.state.candidates["AAA"] = CandidateInfo(symbol="AAA", daily_atr=2.0, beta=1.0, z_rs=2.5, z_rvol=2.5)
        self.state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
        self.state.contexts["AAA"] = [BarContext(time=et(2026, 9, 1, 10, 0), open=100, high=100.2, low=99.8, close=100, volume=5000, vwap=99.9, rs=0.01, rvol=2.0,
            session_open=99.0, session_high=100.2,
        )]
        self.log = tmp_path / "trades.jsonl"
        self.desk = Desk(self.strategy, params or VwapPullbackParameters(), trade_log=lambda: self.log)

    @property
    def state(self) -> SessionState:
        return self.strategy.vars.session

    def advance(self, seconds: float) -> None:
        before = self.clock.now()
        self.clock.advance(seconds)
        self.broker.on_advance(before, self.clock.now())

    def fill_hook(self, order: Order) -> None:
        assert order.is_filled()
        self.desk.on_order_filled(order, order.avg_fill_price, order.filled_quantity)

    def open_trade(self) -> None:
        self.desk.enter_long("AAA", "earnings", "clean pullback")
        entry = self.strategy.get_order(self.state.book.get("AAA").entry_order_id)
        self.advance(60)
        self.fill_hook(entry)


def test_enter_long_sizes_the_trade_and_submits_a_limit_buy(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    result = rig.desk.enter_long("aaa", "earnings", "clean pullback")
    assert result == {"symbol": "AAA", "quantity": 249, "limit_price": 100.1, "stop_price": 99.3, "r_per_share": 0.7, "status": "entry submitted"}
    trade = rig.state.book.get("AAA")
    assert trade.status is TradeStatus.PENDING and trade.catalyst == "earnings"
    assert rig.state.setups["AAA"].state is SetupState.IN_TRADE
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order.order_type is OrderType.LIMIT and order.limit_price == D("100.10")
    assert "AAA" in rig.state.decided


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rig: rig.state.setups.__setitem__("AAA", Setup(symbol="AAA", state=SetupState.PULLBACK)), "no triggered setup"),
        (lambda rig: setattr(rig.state, "flattened", True), "flattened"),
        (lambda rig: rig.clock.advance(-20 * 60), "only allowed between"),
    ],
)
def test_enter_long_refusals(tmp_path: Path, change, message: str) -> None:
    rig = Rig(tmp_path)
    change(rig)
    assert message in rig.desk.enter_long("AAA", "earnings", "x")["error"]


def test_enter_long_refuses_an_unknown_catalyst_and_a_full_book(tmp_path: Path) -> None:
    assert "catalyst" in Rig(tmp_path).desk.enter_long("AAA", "rumour", "x")["error"]
    full = Rig(tmp_path, params=dataclasses.replace(VwapPullbackParameters(), max_positions=0))
    assert "slot" in full.desk.enter_long("AAA", "earnings", "x")["error"]


def test_an_entry_fill_places_the_protective_stop(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    assert trade.status is TradeStatus.OPEN and trade.quantity == D(249) and trade.entry_price == D("100")
    stop = rig.strategy.get_order(trade.stop_order_id)
    assert stop.order_type is OrderType.STOP and stop.stop_price == D("99.30") and stop.quantity == D(249) and stop.side is OrderSide.SELL


def test_a_stop_fill_closes_the_trade_and_writes_the_trade_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    stop = rig.strategy.get_order(rig.state.book.get("AAA").stop_order_id)
    rig.advance(120)  # the 10:03 bar trades through 99.30
    rig.fill_hook(stop)
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.DONE
    row = json.loads(rig.log.read_text().splitlines()[0])
    assert row["symbol"] == "AAA" and row["exit_reason"] == "stop" and row["realised_pnl"] == str(D("-0.70") * 249)


def test_reconcile_expires_an_unfilled_entry_from_an_earlier_tick(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.desk.enter_long("AAA", "earnings", "x")
    entry = rig.strategy.get_order(rig.state.book.get("AAA").entry_order_id)
    rig.clock.advance(5)  # still the same bar: nothing filled
    rig.desk.reconcile(rig.clock.now())
    assert entry.is_canceled()
    rig.desk.on_order_canceled(entry)
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.PULLBACK


def test_reconcile_drops_an_entry_the_broker_rejected(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.desk.enter_long("AAA", "earnings", "x")
    entry = rig.strategy.get_order(rig.state.book.get("AAA").entry_order_id)
    entry.set_error("insufficient cash")
    rig.broker.tracker.process_trade_event(entry, OrderEvent.ERROR)
    rig.desk.reconcile(rig.clock.now())
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.PULLBACK
    assert rig.desk.free_slots() == 4


def test_reconcile_replaces_a_stop_that_errored(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    rig.broker.cancel_order(old)  # drop it from the broker's queue ...
    old.set_error("rejected")  # ... and make it look rejected, with no hook reaching the strategy
    rig.desk.reconcile(rig.clock.now())
    assert trade.stop_order_id != old.identifier
    assert rig.strategy.get_order(trade.stop_order_id).is_active()


def test_unexpected_stop_cancel_places_it_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    rig.broker.cancel_order(old)
    rig.desk.on_order_canceled(old)
    assert trade.stop_order_id != old.identifier and rig.strategy.get_order(trade.stop_order_id).is_active()


def test_flatten_all_cancels_the_stop_and_sells_and_later_entry_fills_are_sold(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    rig.desk.flatten_all("end of day")
    assert stop.is_canceled()
    sell = rig.strategy.get_order(trade.exit_order_ids[-1])
    assert sell.order_type is OrderType.MARKET and sell.quantity == D(249)
    assert trade.exit_reason == "end of day" and rig.state.flattened


def test_close_unknown_positions_leaves_foreign_positions_alone(tmp_path: Path) -> None:
    # AAA: held, with a stop of ours from before a restart, but no trade in the session -> closed.
    # MSFT: held with no order of ours (a shared account) -> untouched.
    rig = Rig(tmp_path)
    for symbol in ("AAA", "MSFT"):
        rig.broker.submit_order(Order(strategy_name="vwap", asset=Asset(symbol), side=OrderSide.BUY, quantity=D(10)))
    rig.advance(60)
    ours = rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell", stop_price=90))  # a stop from before the restart
    rig.desk.close_unknown_positions()
    assert ours.is_canceled()
    pending = [o for o in rig.broker.tracker.get_active_orders() if o.side is OrderSide.SELL]
    assert [(o.asset.symbol, o.quantity) for o in pending] == [("AAA", D(10))]
    assert rig.state.unknown_positions_checked


def test_entry_due_and_exit_review_due(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.desk.entry_due()
    rig.open_trade()
    assert not rig.desk.entry_due()  # the only triggered setup is now in a trade
    assert rig.desk.exit_review_due(rig.clock.now()) == []  # 100 is above VWAP 99.9 and the EMA, below +1R
    rig.clock.advance(timedelta(minutes=15).total_seconds())
    assert rig.desk.exit_review_due(rig.clock.now()) == ["AAA"]
    rig.desk.mark_reviewed(rig.clock.now())
    assert rig.desk.exit_review_due(rig.clock.now()) == []


def test_a_partially_filled_stop_that_is_cancelled_reduces_the_trade_and_is_replaced(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    old.filled_quantity = D(100)
    old.avg_fill_price = D("99.30")
    rig.broker.cancel_order(old)
    rig.desk.on_order_canceled(old)
    assert trade.quantity == D(149) and trade.realised_pnl == D("-70.00")
    replacement = rig.strategy.get_order(trade.stop_order_id)
    assert replacement.identifier != old.identifier and replacement.quantity == D(149) and replacement.is_active()


def test_release_stop_books_a_partial_fill(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    old.filled_quantity = D(100)
    old.avg_fill_price = D("99.30")
    assert rig.desk._release_stop(trade) is None
    assert trade.quantity == D(149) and trade.stop_order_id is None


def test_enter_long_refuses_a_symbol_that_is_not_a_candidate(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    del rig.state.candidates["AAA"]
    assert "not a candidate" in rig.desk.enter_long("AAA", "earnings", "x")["error"]


def test_close_unknown_positions_cancels_a_pending_entry_of_ours_for_a_symbol_not_held(tmp_path: Path) -> None:
    # Final review I2(a): after a restart, an adopted entry for a symbol with no position and no trade must not stay working.
    rig = Rig(tmp_path)
    adopted = rig.strategy.submit_order(rig.strategy.create_order("MSFT", 10, "buy", limit_price=90))
    rig.desk.close_unknown_positions()
    assert adopted.is_canceled()
    assert rig.broker.tracker.get_active_orders() == []


def test_a_fill_of_our_buy_that_matches_no_trade_is_sold_at_once(tmp_path: Path) -> None:
    # Final review I2(b): belt and braces -- a buy of ours that fills with no trade to protect it is never carried.
    rig = Rig(tmp_path)
    orphan = rig.strategy.submit_order(rig.strategy.create_order("MSFT", 10, "buy"))
    rig.advance(60)
    rig.fill_hook(orphan)
    sells = [o for o in rig.broker.tracker.get_active_orders() if o.side is OrderSide.SELL]
    assert [(o.asset.symbol, o.quantity, o.order_type) for o in sells] == [("MSFT", D(10), OrderType.MARKET)]
    assert rig.state.book.get("MSFT") is None


def test_flatten_without_a_session_cancels_our_orders_and_closes_only_our_positions(tmp_path: Path) -> None:
    # Final review I5(a): session preparation kept failing (no session), yet what this strategy holds must not ride overnight.
    rig = Rig(tmp_path)
    rig.broker._data_source.frames[("BBB", "minute")] = minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)
    for symbol in ("AAA", "MSFT"):
        rig.broker.submit_order(Order(strategy_name="vwap", asset=Asset(symbol), side=OrderSide.BUY, quantity=D(10)))
    rig.advance(60)
    stop = rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell", stop_price=90))
    entry = rig.strategy.submit_order(rig.strategy.create_order("BBB", 5, "buy", limit_price=90))
    rig.strategy.vars.session = None
    rig.desk.flatten_all("end-of-day flatten")
    assert stop.is_canceled() and entry.is_canceled()
    pending = [o for o in rig.broker.tracker.get_active_orders()]
    assert [(o.asset.symbol, o.side, o.quantity) for o in pending] == [("AAA", OrderSide.SELL, D(10))]  # MSFT is not ours


def test_flatten_sells_the_shares_of_a_stop_whose_cancel_was_confirmed_late(tmp_path: Path) -> None:
    # Final review I5(b): the stop cancel is confirmed only after the 10 s wait; its hook is swallowed as expected,
    # so without a re-check after the flatten wait those shares would have neither a stop nor a sell.
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    rig.strategy.trading_mode = TradingMode.PAPER  # the live path: waits happen
    rig.strategy.cancel_order = lambda order: None  # the cancel request goes out ...
    rig.strategy.wait_for_order_execution = lambda order, timeout=None: False  # ... and is not confirmed within the wait

    def flatten_wait(orders, timeout=None) -> bool:
        rig.broker.cancel_order(stop)  # ... it is confirmed during the flatten wait
        rig.desk.on_order_canceled(stop)
        return True

    rig.strategy.wait_for_orders_execution = flatten_wait
    rig.desk.flatten_all("end of day")
    assert stop.is_canceled()
    sells = [o for o in rig.broker.tracker.get_active_orders() if o.side is OrderSide.SELL]
    assert [(o.order_type, o.quantity) for o in sells] == [(OrderType.MARKET, D(249))]
    assert trade.exit_reason == "end of day"


def _working_exit_sell(rig: Rig, quantity: int, filled: int):
    trade = rig.state.book.get("AAA")
    rig.broker.cancel_order(rig.strategy.get_order(trade.stop_order_id))  # the stop would otherwise already cover every share
    sell = rig.strategy.submit_order(rig.strategy.create_order("AAA", quantity, "sell", limit_price=200))
    trade.exit_order_ids.append(sell.identifier)
    sell.filled_quantity = D(filled)  # partly filled by the tracker, not yet booked by the desk hook
    return trade, sell


def test_free_quantity_counts_a_partly_filled_exit_sells_full_quantity_as_not_free(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade, sell = _working_exit_sell(rig, 124, 60)
    assert sell.is_active() and trade.quantity == D(249)
    assert rig.desk._free_quantity(trade) == D(125)


def test_flatten_does_not_resell_the_filled_part_of_a_partly_filled_flatten_sell(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    rig.strategy.trading_mode = TradingMode.PAPER  # the live path: waits happen
    rig.strategy.wait_for_order_execution = lambda order, timeout=None: True

    def flatten_wait(orders, timeout=None) -> bool:
        for order in orders:
            if order.side is OrderSide.SELL and order.order_type is OrderType.MARKET:
                order.filled_quantity = D(100)  # partly filled, still working when the wait ends
        return False

    rig.strategy.wait_for_orders_execution = flatten_wait
    attempts: list[D] = []
    real_sell = rig.desk._market_sell
    rig.desk._market_sell = lambda t, quantity, reason: attempts.append(quantity) or real_sell(t, quantity, reason)
    rig.desk.flatten_all("end of day")
    assert attempts == [D(249)]  # no second sell for the part the first one already filled
    assert trade.status is TradeStatus.OPEN


def test_free_slots_counts_a_partly_filled_full_exit_as_freed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    before = rig.desk.free_slots()
    _working_exit_sell(rig, 249, 100)
    assert rig.desk.free_slots() == before + 1


def test_flatten_sells_the_other_shares_when_a_tp1_sell_is_working_and_the_stop_cancel_is_late(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    rig.desk.take_partial_profit("AAA", 0.5)
    tp1 = rig.strategy.get_order(trade.exit_order_ids[-1])
    stop = rig.strategy.get_order(trade.stop_order_id)
    assert tp1.is_active() and tp1.quantity == D(124) and stop.quantity == D(125)
    rig.strategy.trading_mode = TradingMode.PAPER
    rig.strategy.cancel_order = lambda order: None
    rig.strategy.wait_for_order_execution = lambda order, timeout=None: False

    def flatten_wait(orders, timeout=None) -> bool:
        rig.broker.cancel_order(stop)  # confirmed only during the flatten wait
        rig.desk.on_order_canceled(stop)
        return True

    rig.strategy.wait_for_orders_execution = flatten_wait
    attempts: list[D] = []
    real_sell = rig.desk._market_sell
    rig.desk._market_sell = lambda t, quantity, reason: attempts.append(quantity) or real_sell(t, quantity, reason)
    rig.desk.flatten_all("end of day")
    assert attempts == [D(125)]


def test_a_partly_filled_exit_sell_that_ends_cancelled_books_its_fill_and_the_rest_is_stopped_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    rig.broker.cancel_order(rig.strategy.get_order(trade.stop_order_id))
    trade.stop_order_id = None
    sell = rig.desk._market_sell(trade, D(249), "exit")
    sell.filled_quantity = D(100)
    sell.avg_fill_price = D("101")
    rig.broker.cancel_order(sell)
    rig.desk.on_order_canceled(sell)
    assert trade.quantity == D(149) and trade.realised_pnl != D(0)
    stop = rig.strategy.get_order(trade.stop_order_id)
    assert stop.is_active() and stop.quantity == D(149)
    rig.desk.on_order_canceled(sell)  # a repeated hook books nothing twice
    assert trade.quantity == D(149)

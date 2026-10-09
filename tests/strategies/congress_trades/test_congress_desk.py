from __future__ import annotations

import inspect
import logging
import threading
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderStatus, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.congress_trades.desk import TradeDesk, order_status
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError

# Portfolio value 10,000 in every test: the band is 100 (1%), the smallest order 50 (0.5%).
PARAMS = CongressParams(order_wait_seconds=30.0)


class _Book:
    def __init__(self, strategy: Strategy, broker: FakeBroker, desk: TradeDesk) -> None:
        self.strategy, self.broker, self.desk = strategy, broker, desk
        tools = {tool.__name__: tool for tool in desk.tools()}
        self.place, self.check = tools["place_order"], tools["check_orders"]

    def orders(self) -> list[tuple[str, str, float]]:
        return [(o.asset.symbol, o.side.value, float(o.quantity or 0)) for o in self.broker.submitted]


def _book(
    tmp_path: Path,
    *,
    target: dict[str, float] | None = None,
    traded: tuple[str, ...] = (),
    positions: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    cash: float = 5_000.0,
    buying_power: float = 5_000.0,
    portfolio_value: float = 10_000.0,
    params: CongressParams = PARAMS,
) -> _Book:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="congress_trades")
    broker.positions = [Position(strategy_name="congress_trades", asset=Asset(s), quantity=Decimal(str(q)), side=PositionSide.LONG) for s, q in (positions or {}).items()]
    broker.last_prices = {s: Decimal(str(p)) for s, p in (prices or {"AAA": 50.0, "BBB": 25.0, "OLD": 10.0, "ZZZ": 10.0}).items()}
    broker.account = AccountBalances(cash=Decimal(str(cash)), portfolio_value=Decimal(str(portfolio_value)), buying_power=Decimal(str(buying_power)))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    desk = TradeDesk(strategy, params)
    desk.begin_run({"AAA": 0.15, "BBB": 0.05} if target is None else target, traded)
    return _Book(strategy, broker, desk)


# --- buys ---------------------------------------------------------------------------------------------


def test_a_buy_up_to_the_target_weight_plus_the_band_is_accepted_and_one_more_share_is_not(tmp_path: Path) -> None:
    book = _book(tmp_path)  # AAA target $1,500 + band $100 = $1,600 = 32 shares at $50

    refused = book.place("AAA", "buy", 33)
    accepted = book.place("AAA", "buy", 32)

    assert "at most 32 shares" in refused["error"] and "target weight of 0.1500" in refused["error"]
    assert accepted["symbol"] == "AAA" and accepted["side"] == "buy" and accepted["quantity"] == 32 and accepted["status"] == "working"
    assert book.orders() == [("AAA", "buy", 32.0)]


def test_a_second_buy_counts_the_first_one_still_open(tmp_path: Path) -> None:
    book = _book(tmp_path)
    book.place("AAA", "buy", 30)  # $1,500 on order

    assert "error" in book.place("AAA", "buy", 5)  # $250 more is beyond $1,600
    assert "error" not in book.place("AAA", "buy", 2)  # $100 more reaches exactly $1,600


def test_held_shares_count_toward_the_target(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 28})  # $1,400 held: room for $200 = 4 shares

    assert "at most 4 shares" in book.place("AAA", "buy", 5)["error"]
    assert "error" not in book.place("AAA", "buy", 4)


def test_a_buy_of_a_stock_outside_the_target_is_refused(tmp_path: Path) -> None:
    book = _book(tmp_path)

    assert "not in the target portfolio" in book.place("ZZZ", "buy", 10)["error"]
    assert book.orders() == []


@pytest.mark.parametrize("quantity", [0, -1, float("nan"), float("inf"), "lots", None, True, 1e-9])
def test_a_quantity_that_is_not_a_positive_number_of_shares_is_refused(tmp_path: Path, quantity: Any) -> None:
    book = _book(tmp_path)

    assert "error" in book.place("AAA", "buy", quantity)
    assert book.orders() == []


def test_the_side_must_be_buy_or_sell(tmp_path: Path) -> None:
    book = _book(tmp_path)

    assert "side must be" in book.place("AAA", "short", 1)["error"]


def test_symbol_and_side_are_normalised(tmp_path: Path) -> None:
    book = _book(tmp_path)

    assert book.place(" aaa ", " BUY ", 10)["symbol"] == "AAA"


def test_fractional_quantities_are_floored_not_rounded(tmp_path: Path) -> None:
    book = _book(tmp_path)

    book.place("AAA", "buy", 1.23456789)

    assert book.orders() == [("AAA", "buy", 1.234567)]


def test_an_order_below_the_minimum_size_is_refused(tmp_path: Path) -> None:
    book = _book(tmp_path)

    assert "below the minimum order" in book.place("AAA", "buy", 0.5)["error"]  # $25 < $50


def test_a_buy_beyond_the_smaller_of_buying_power_and_cash_is_refused(tmp_path: Path) -> None:
    low_cash = _book(tmp_path, cash=1_000.0, buying_power=20_000.0)
    low_power = _book(tmp_path, cash=5_000.0, buying_power=1_000.0)

    for book in (low_cash, low_power):
        refused = book.place("AAA", "buy", 30)  # $1,500 against $1,000
        assert "at most 20 shares" in refused["error"]
        assert "error" not in book.place("AAA", "buy", 20)


def test_buying_power_alone_never_funds_a_buy_on_a_margin_account(tmp_path: Path) -> None:
    book = _book(tmp_path, cash=100.0, buying_power=40_000.0)

    assert "exceeds the money available" in book.place("AAA", "buy", 30)["error"]


def test_this_runs_buys_are_deducted_from_the_money_available(tmp_path: Path) -> None:
    book = _book(tmp_path, cash=2_000.0, buying_power=2_000.0)
    book.place("AAA", "buy", 30)  # $1,500

    assert "at most 20 shares" in book.place("BBB", "buy", 21)["error"]  # $500 left; BBB $25 a share
    assert "error" not in book.place("BBB", "buy", 20)


def _fill(book: _Book, order_id: str, price: float) -> None:
    """A live fill: the order is done and the account's cash already reflects it."""
    order = next(o for o in book.broker.submitted if o.identifier == order_id)
    order.status, order.filled_quantity = OrderStatus.FILL, order.quantity
    cash = book.broker.account.cash - Decimal(str(price)) * order.quantity
    book.broker.account = AccountBalances(cash=cash, portfolio_value=book.broker.account.portfolio_value, buying_power=cash)


def test_a_filled_buy_is_not_deducted_twice_once_the_account_cash_reflects_it(tmp_path: Path) -> None:
    """Live market buys fill in seconds and cash drops by their cost: counting the same cost again left $0.00 available (2026-10-09)."""
    book = _book(tmp_path, cash=2_000.0, buying_power=2_000.0)
    first = book.place("AAA", "buy", 30)  # $1,500
    _fill(book, first["order_id"], 50.0)  # cash is now $500

    assert "at most 20 shares" in book.place("BBB", "buy", 21)["error"]  # $500 available, BBB $25 a share
    assert "error" not in book.place("BBB", "buy", 20)


def test_a_filled_sells_proceeds_are_not_added_twice_once_the_account_cash_reflects_them(tmp_path: Path) -> None:
    book = _book(tmp_path, cash=0.0, buying_power=0.0, traded=("OLD",), positions={"OLD": 100})  # OLD: $1,000
    sold = book.place("OLD", "sell", 100)
    order = next(o for o in book.broker.submitted if o.identifier == sold["order_id"])
    order.status, order.filled_quantity = OrderStatus.FILL, order.quantity
    book.broker.account = AccountBalances(cash=Decimal(1_000), portfolio_value=Decimal(10_000), buying_power=Decimal(20_000))
    book.broker.positions = []

    assert "at most 20 shares" in book.place("AAA", "buy", 30)["error"]  # $1,000, not $2,000
    assert "error" not in book.place("AAA", "buy", 20)


def test_the_proceeds_of_this_runs_sells_fund_the_buys(tmp_path: Path) -> None:
    book = _book(tmp_path, cash=0.0, buying_power=20_000.0, traded=("OLD",), positions={"OLD": 100})  # OLD: $1,000

    assert "error" in book.place("AAA", "buy", 10)  # no cash yet
    book.desk.begin_run({"AAA": 0.15, "BBB": 0.05}, ("OLD",))
    assert "error" not in book.place("OLD", "sell", 100)
    assert "at most 20 shares" in book.place("AAA", "buy", 30)["error"]  # the $1,000 of proceeds
    assert "error" not in book.place("AAA", "buy", 20)


# --- sells --------------------------------------------------------------------------------------------


def test_a_position_this_strategy_never_ordered_is_left_alone(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"ZZZ": 100})

    assert "not a position of this strategy" in book.place("ZZZ", "sell", 10)["error"]


def test_a_dropped_stock_this_strategy_bought_before_can_be_sold_in_full(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",), positions={"OLD": 100})  # $1,000

    result = book.place("OLD", "sell", 100)

    assert result["side"] == "sell"
    assert "error" in book.place("OLD", "sell", 1)  # already being sold: nothing left to sell


def test_no_shorts_a_sell_is_capped_at_held_minus_what_is_already_being_sold(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",), positions={"OLD": 100})

    assert "exceeds the 100 shares held" in book.place("OLD", "sell", 101)["error"]
    book.place("OLD", "sell", 60)
    assert "exceeds the 40 shares held and not already being sold" in book.place("OLD", "sell", 60)["error"]
    assert "error" not in book.place("OLD", "sell", 40)


def test_a_sell_may_not_take_a_target_stock_below_its_weight(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 40})  # $2,000 held, target $1,500, floor $1,400

    assert "at most 12 shares" in book.place("AAA", "sell", 13)["error"]
    assert "error" not in book.place("AAA", "sell", 12)


def test_closing_a_tiny_position_is_allowed_below_the_minimum_but_a_partial_sell_is_not(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",), positions={"OLD": 4})  # $40 of OLD, below the $50 minimum

    assert "below the minimum order" in book.place("OLD", "sell", 2)["error"]
    assert "error" not in book.place("OLD", "sell", 4)


def test_sells_go_out_before_the_first_buy(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",), positions={"OLD": 100})
    book.place("AAA", "buy", 10)

    assert "before the first buy" in book.place("OLD", "sell", 100)["error"]


# --- the broker and the state -------------------------------------------------------------------------


def test_a_broker_refusal_is_an_error_payload_and_the_stock_is_still_recorded_as_ours(tmp_path: Path) -> None:
    class _Rejecting(FakeBroker):
        def _submit_order(self, order: Order) -> Order:
            raise OrderValidationError("not tradable")

    book = _book(tmp_path)
    broker = _Rejecting(book.broker.clock, strategy_name="congress_trades")
    broker.last_prices, broker.account = book.broker.last_prices, book.broker.account
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    desk = TradeDesk(strategy, PARAMS)
    desk.begin_run({"AAA": 0.15}, ())
    place = {t.__name__: t for t in desk.tools()}["place_order"]

    result = place("AAA", "buy", 10)

    assert "refused by the broker: not tradable" in result["error"]
    assert desk.traded == ["AAA"]  # an order a client raised on may still have reached the broker
    assert desk.orders == []


def test_traded_accumulates_and_a_new_run_resets_the_orders_but_keeps_the_history(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",))
    book.place("AAA", "buy", 10)

    assert book.desk.traded == ["AAA", "OLD"]
    assert len(book.desk.orders) == 1

    book.desk.begin_run({"AAA": 0.15}, book.desk.traded)
    assert book.desk.orders == []
    assert book.desk.orders_view() == {}
    assert book.desk.traded == ["AAA", "OLD"]


def test_a_data_failure_is_an_error_payload(tmp_path: Path) -> None:
    book = _book(tmp_path)
    book.broker.market_data_error = BrokerError("no data")

    assert "cannot read the account or the price" in book.place("AAA", "buy", 10)["error"]


def test_a_missing_price_or_a_zero_portfolio_refuses_every_order(tmp_path: Path) -> None:
    assert "no price for AAA" in _book(tmp_path, prices={"BBB": 25.0}).place("AAA", "buy", 10)["error"]
    assert "nothing can be traded" in _book(tmp_path, portfolio_value=0.0).place("AAA", "buy", 10)["error"]


def test_every_refusal_is_logged_as_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    book = _book(tmp_path)

    with caplog.at_level(logging.WARNING):
        book.place("ZZZ", "buy", 10)

    assert "order refused" in caplog.text


# --- check_orders ---------------------------------------------------------------------------------------


def test_check_orders_with_no_orders_says_so(tmp_path: Path) -> None:
    assert _book(tmp_path).check() == {"orders": [], "all_filled": True, "note": "no order was placed in this run"}


def test_check_orders_waits_the_configured_time_and_reports_working_orders(tmp_path: Path) -> None:
    book = _book(tmp_path)
    placed = book.place("AAA", "buy", 10)

    before = book.desk.orders_view()[placed["order_id"]]
    result = book.check()
    after = book.desk.orders_view()[placed["order_id"]]

    assert book.broker.clock.waits == [30.0]  # type: ignore[attr-defined]
    assert result["all_filled"] is False
    assert result["orders"][0]["status"] == "working"
    assert (before.checked, after.checked) == (False, True)


def test_check_orders_reports_filled_orders_with_their_fill(tmp_path: Path) -> None:
    book = _book(tmp_path)
    placed = book.place("AAA", "buy", 10)
    order = book.desk.orders[0]
    order.status, order.filled_quantity, order.avg_fill_price = OrderStatus.FILL, Decimal(10), Decimal("50.1")

    result = book.check()

    assert result["all_filled"] is True
    assert result["orders"][0] == {"order_id": placed["order_id"], "symbol": "AAA", "side": "buy", "quantity": 10.0, "status": "filled", "filled_quantity": 10.0, "avg_price": 50.1}
    assert book.broker.clock.waits == []  # type: ignore[attr-defined]  # nothing to wait for


def test_check_orders_reports_a_rejected_order_with_its_error(tmp_path: Path) -> None:
    book = _book(tmp_path)
    book.place("AAA", "buy", 10)
    order = book.desk.orders[0]
    order.status, order.error_message = OrderStatus.ERROR, "insufficient buying power"

    item = book.check()["orders"][0]

    assert (item["status"], item["error"]) == ("rejected", "insufficient buying power")


@pytest.mark.parametrize(
    "status, expected",
    [
        (OrderStatus.UNPROCESSED, "working"),
        (OrderStatus.NEW, "working"),
        (OrderStatus.OPEN, "working"),
        (OrderStatus.SUBMITTED, "working"),
        (OrderStatus.CANCELLING, "working"),
        (OrderStatus.PARTIAL_FILL, "partially_filled"),
        (OrderStatus.FILL, "filled"),
        (OrderStatus.CANCELED, "canceled"),
        (OrderStatus.ERROR, "rejected"),
        (OrderStatus.EXPIRED, "expired"),
        (OrderStatus.UNKNOWN, "unknown"),
    ],
)
def test_order_status_names(status: OrderStatus, expected: str) -> None:
    order = Order(strategy_name="s", asset=Asset("AAA"), side=__import__("trading_agent_framework.entities.enums", fromlist=["OrderSide"]).OrderSide.BUY, quantity=Decimal(1), status=status)

    assert order_status(order) == expected


# --- the audit ------------------------------------------------------------------------------------------


def test_the_audit_lists_unfilled_orders(tmp_path: Path) -> None:
    book = _book(tmp_path, target={"AAA": 0.15})
    book.place("AAA", "buy", 30)

    audit = book.desk.audit()

    assert [(v.symbol, v.status) for v in audit.unfilled] == [("AAA", "working")]
    assert audit.shortfalls == []  # the open order covers the target
    assert not audit.complete


def test_a_filled_order_that_reaches_the_target_is_a_complete_audit(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 30, "BBB": 20})  # AAA $1,500 (target 1,500), BBB $500 (target 500)

    assert book.desk.audit().complete


def test_a_target_stock_below_its_weight_with_nothing_on_order_is_a_buy_shortfall(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 10})  # $500 held against $1,500; BBB not held at all (target $500)

    audit = book.desk.audit()

    assert [(s.symbol, s.kind, round(s.value)) for s in audit.shortfalls] == [("AAA", "buy", 1000), ("BBB", "buy", 500)]


def test_a_drift_inside_the_band_is_not_a_shortfall(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 29, "BBB": 20})  # AAA $1,450: $50 short, inside the $100 band

    assert book.desk.audit().shortfalls == []


def test_a_dropped_stock_still_held_is_a_sell_shortfall_but_a_stock_not_ours_is_ignored(tmp_path: Path) -> None:
    book = _book(tmp_path, traded=("OLD",), positions={"AAA": 30, "BBB": 20, "OLD": 100, "ZZZ": 500})

    audit = book.desk.audit()

    assert [(s.symbol, s.kind) for s in audit.shortfalls] == [("OLD", "sell")]
    assert "dropped from the target" in audit.shortfalls[0].detail


def test_an_overweight_target_stock_is_a_sell_shortfall(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 50, "BBB": 20})  # AAA $2,500 against $1,500

    assert [(s.symbol, s.kind, round(s.value)) for s in book.desk.audit().shortfalls] == [("AAA", "sell", 1000)]


def test_the_audit_skips_a_stock_without_a_price_and_survives_an_unavailable_account(tmp_path: Path) -> None:
    book = _book(tmp_path, prices={"AAA": 50.0})  # no BBB price
    assert [s.symbol for s in book.desk.audit().shortfalls] == ["AAA"]

    book.broker.market_data_error = BrokerError("no data")
    placed = _book(tmp_path)
    placed.place("AAA", "buy", 10)
    placed.broker.account = AccountBalances(cash=Decimal(0), portfolio_value=Decimal(0), buying_power=Decimal(0))
    assert [v.symbol for v in placed.desk.audit().unfilled] == ["AAA"]


# --- the tools --------------------------------------------------------------------------------------------


def test_the_tools_have_one_line_docstrings_and_real_annotations(tmp_path: Path) -> None:
    book = _book(tmp_path)

    assert [t.__name__ for t in book.desk.tools()] == ["place_order", "check_orders"]
    for tool in book.desk.tools():
        assert tool.__doc__ is not None and "\n" not in tool.__doc__.strip()
    params = inspect.signature(book.place).parameters
    assert [params[name].annotation for name in ("symbol", "side", "quantity")] == [str, str, float]


def test_parallel_tool_calls_never_over_buy(tmp_path: Path) -> None:
    """LangGraph runs the tool calls of one model response on worker threads: the lock makes the limit hold."""
    book = _book(tmp_path)  # room for 32 shares in total
    results: list[dict[str, Any]] = []
    original = book.broker._submit_order

    def slow_submit(order: Order) -> Order:
        time.sleep(0.02)  # widens the window between a check and the order showing up as in flight
        return original(order)

    book.broker._submit_order = slow_submit  # type: ignore[method-assign]

    def buy() -> None:
        results.append(book.place("AAA", "buy", 8))

    threads = [threading.Thread(target=buy) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(1 for r in results if "error" not in r) == 4
    assert sum(q for _, _, q in book.orders()) == 32


# --- a daily backtest: orders fill on the NEXT bar, so the first check finds them working ------------------

DAY1 = datetime(2026, 1, 5, 16, tzinfo=UTC)


def test_in_a_daily_backtest_the_orders_are_still_working_at_the_tick_and_filled_a_session_later(tmp_path: Path) -> None:
    source = FakeBacktestDataSource()
    source.set_bars(Asset("AAPL"), make_close_indexed_frame([150.0, 151.0, 152.0], start=DAY1, freq="1D"))
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("congress_trades", data_source=source, clock=clock, budget=Decimal(10000))
    clock.on_advance = broker.on_advance
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
    desk = TradeDesk(strategy, CongressParams(order_wait_seconds=60.0))
    desk.begin_run({"AAPL": 0.15}, ())
    place, check = (t for t in desk.tools())

    placed = place("AAPL", "buy", 10)  # $1,500 at $150
    first = check()
    working_view = desk.orders_view()[placed["order_id"]]
    audit = desk.audit()

    assert "error" not in placed
    assert first["all_filled"] is False and first["orders"][0]["status"] == "working"  # a 60 s wait does not reach the next bar
    assert working_view.checked and not working_view.filled
    assert [v.symbol for v in audit.unfilled] == ["AAPL"] and audit.shortfalls == []  # the open order covers the target

    clock.wait(timedelta(days=1).total_seconds(), threading.Event())  # the next session's bar closes
    second = check()

    assert second["all_filled"] is True
    assert second["orders"][0]["status"] == "filled" and second["orders"][0]["avg_price"] == 151.0
    assert desk.audit().complete


def test_an_order_from_an_earlier_run_still_working_keeps_the_audit_incomplete(tmp_path: Path) -> None:
    book = _book(tmp_path, target={"AAA": 0.15})
    book.place("AAA", "buy", 30)  # covers the target and stays working at the broker
    book.desk.begin_run({"AAA": 0.15}, book.desk.traded)  # a later run: this run has placed nothing

    audit = book.desk.audit()

    assert audit.unfilled == [] and audit.shortfalls == []
    assert audit.in_flight == ["AAA"]
    assert not audit.complete


def test_no_order_working_means_nothing_in_flight(tmp_path: Path) -> None:
    book = _book(tmp_path, positions={"AAA": 30, "BBB": 20})

    assert book.desk.audit().in_flight == []

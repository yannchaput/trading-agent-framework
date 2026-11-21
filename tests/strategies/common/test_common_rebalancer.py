from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, TickingBroker, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.common.portfolio import target_portfolio
from trading_agent_framework.strategies.common.rebalancer import PlacedOrder, Rebalancer, submit_buy
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError, InsufficientFundsError

# Portfolio value 10,000 in every test: the band is 500 (5%), the smallest order 50 (0.5%), the cash reserve 200 (2%).


def _book(
    tmp_path: Path,
    *,
    positions: dict[str, float],
    prices: dict[str, float],
    cash: float,
    portfolio_value: float = 10_000.0,
    buying_power: float = 1_000_000.0,
    costs: dict[str, float] | None = None,
) -> tuple[Strategy, FakeBroker, Rebalancer]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.positions = [
        Position(
            strategy_name="bill_ackman",
            asset=Asset(symbol),
            quantity=Decimal(str(quantity)),
            side=PositionSide.LONG,
            avg_fill_price=Decimal(str((costs or {})[symbol])) if symbol in (costs or {}) else None,
        )
        for symbol, quantity in positions.items()
    ]
    broker.last_prices = {symbol: Decimal(str(price)) for symbol, price in prices.items()}
    broker.account = AccountBalances(cash=Decimal(str(cash)), portfolio_value=Decimal(str(portfolio_value)), buying_power=Decimal(str(buying_power)))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return strategy, broker, Rebalancer(strategy, AckmanParams())


def _orders(broker: FakeBroker) -> list[tuple[str, str, float]]:
    return [(order.asset.symbol, order.side.value, float(order.quantity)) for order in broker.submitted]


def test_an_empty_book_buys_the_parking_instrument_up_to_everything_but_the_cash_buffer(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"SHV": 100}, cash=10_000)

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "buy", 98.0)]
    assert placed == [PlacedOrder("SHV", "buy", 98.0)]


def test_new_stocks_are_funded_by_selling_parking_and_every_sell_goes_out_before_any_buy(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "sell", 60.0), ("A", "buy", 70.0), ("B", "buy", 100.0)]


def test_a_forced_exit_is_sold_in_full_and_its_money_is_parked(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)

    rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert _orders(broker) == [("A", "sell", 100.0), ("SHV", "buy", 98.0)]


def test_a_forced_exit_is_sold_even_if_it_is_in_the_target_weights(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02), forced_exits=["A"])

    assert _orders(broker)[0] == ("A", "sell", 100.0)
    assert not any(symbol == "A" and side == "buy" for symbol, side, _ in _orders(broker))


def test_a_holding_missing_from_the_target_is_sold_in_full(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("B", "sell", 100.0), ("A", "buy", 70.0), ("SHV", "buy", 63.0)]


def test_a_drift_inside_the_band_does_not_trade(tmp_path: Path) -> None:
    # A is held at 3,300 against a target of 3,500: 200 below, inside the 500 band. Only the parking buy happens.
    _, broker, rebalancer = _book(tmp_path, positions={"A": 66}, prices={"A": 50, "SHV": 100}, cash=6700)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("SHV", "buy", 63.0)]


def test_a_drift_beyond_the_band_buys_the_difference(tmp_path: Path) -> None:
    # 2,950 against 3,500: 550 below, beyond the band.
    _, broker, rebalancer = _book(tmp_path, positions={"A": 59}, prices={"A": 50, "SHV": 100}, cash=7050)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert ("A", "buy", 11.0) in _orders(broker)


def test_a_drift_of_exactly_the_band_does_not_trade(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 60}, prices={"A": 50, "SHV": 100}, cash=7000)  # 3,000 against 3,500

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert not any(symbol == "A" for symbol, _, _ in _orders(broker))


def test_a_stock_above_its_band_is_trimmed_and_the_proceeds_are_parked(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 90}, prices={"A": 50, "SHV": 100}, cash=5500)  # 4,500 against 3,500

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("A", "sell", 20.0), ("SHV", "buy", 63.0)]


def test_parking_above_its_target_is_sold_down(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({}, cash_buffer=0.5))  # parking target 5,000, holding 9,800

    assert _orders(broker) == [("SHV", "sell", 48.0)]


def test_a_refused_buy_resyncs_the_available_cash_from_the_brokers_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    original = strategy.submit_order

    def submit(order):  # the broker refuses A for buying power and says what it really has
        if order.asset.symbol == "A" and order.side is OrderSide.BUY:
            raise BrokerError(json.dumps({"buying_power": "1000", "message": "insufficient buying power"}))
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    # B is sized against 1,000 (the broker's figure) less the 200 reserve = 800, not against our own estimate.
    assert placed == [PlacedOrder("SHV", "sell", 60.0), PlacedOrder("B", "buy", 32.0)]


def test_a_buy_over_the_framework_budget_resyncs_from_its_spendable_figure_and_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    calls = _raising_for(strategy, monkeypatch, "A", OrderSide.BUY, InsufficientFundsError("A ticked up", spendable=Decimal(1000), max_quantity=Decimal(19)))

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    # A is retried once at the 19 that fit, refused again, not a third time; B is sized against the 1,000 the check reported
    assert placed == [PlacedOrder("SHV", "sell", 60.0), PlacedOrder("B", "buy", 40.0)]
    assert calls["A"] == 2


def test_a_refused_buy_whose_most_that_fits_is_under_the_minimum_trade_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    calls = _raising_for(strategy, monkeypatch, "A", OrderSide.BUY, InsufficientFundsError("A ticked up", spendable=Decimal(40), max_quantity=Decimal("0.8")))

    rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert calls["A"] == 1  # 0.8 x 50 = 40 is under the 50 minimum trade
    assert not any(symbol == "A" for symbol, _, _ in _orders(broker))


def test_the_buy_that_spends_the_last_of_the_money_is_retried_when_its_price_ticks_up(tmp_path: Path) -> None:
    """Live, B ticks a cent between the review's price read and the check: it is bought at the most that fits, not dropped."""
    broker = TickingBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman", ticks={"B": Decimal("100.01")})
    broker.last_prices = {"A": Decimal(100), "B": Decimal(100), "SHV": Decimal(100)}
    broker.account = AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    rebalancer = Rebalancer(strategy, AckmanParams())

    placed = rebalancer.rebalance(target_portfolio({"A": 0.49, "B": 0.49}, cash_buffer=0.02))

    # B: 49 x 100.01 is over the 4,900 left; 4,900 / 100.01 = 48.9951 shares fit
    assert placed == [PlacedOrder("A", "buy", 49.0), PlacedOrder("B", "buy", 48.9951)]
    assert _orders(broker) == [("A", "buy", 49.0), ("B", "buy", 48.9951)]
    assert rebalancer.refused == 0


def _refused_once_then(strategy: Strategy, monkeypatch: pytest.MonkeyPatch, retry_error: Exception | None = None) -> None:
    """The first A buy is refused for money (40 shares fit); the retry raises `retry_error`, or goes through. Spendable stays 5,000."""
    original = strategy.submit_order
    refusals = [InsufficientFundsError("A ticked up", spendable=Decimal(2000), max_quantity=Decimal(40))]

    def submit(order):
        if order.asset.symbol == "A" and refusals:
            raise refusals.pop()
        if order.asset.symbol == "A" and retry_error is not None:
            raise retry_error
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)
    monkeypatch.setattr(strategy, "spendable_cash", lambda account=None: Decimal(5000))  # large: the per-buy cap never hides the running figure


def test_the_running_budget_deducts_what_a_retry_really_bought(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=10_000)
    _refused_once_then(strategy, monkeypatch)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.35}, cash_buffer=0.02))

    # 5,000 - the 40 x 50 really bought = 3,000 for B (120 shares), not 5,000 - the 70 x 50 planned (60 shares)
    assert placed == [PlacedOrder("A", "buy", 40.0), PlacedOrder("B", "buy", 120.0)]


def test_a_retry_with_an_unknown_outcome_reports_the_retried_quantity_and_keeps_the_planned_cost_spent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=10_000)
    _refused_once_then(strategy, monkeypatch, BrokerUncertainError("timeout"))

    with caplog.at_level(logging.WARNING):
        placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.35}, cash_buffer=0.02))

    # the retry sent 40; the planned 70 x 50 stays spent (the safe side), so B gets 1,500 (60 shares)
    assert placed == [PlacedOrder("A", "buy", 40.0, confirmed=False), PlacedOrder("B", "buy", 60.0)]
    assert "buy 40 A (toward target): outcome unknown at the broker" in caplog.text


def _strategy_at(tmp_path: Path, price: str, ticked: str) -> tuple[Strategy, TickingBroker]:
    """A paper strategy with 10,000 in cash (spendable 9,800) whose AAA quote moves from `price` to `ticked` after one read."""
    broker = TickingBroker(FakeClock(et(2026, 9, 14, 10)), ticks={"AAA": Decimal(ticked)})
    broker.last_prices = {"AAA": Decimal(price)}
    broker.account = AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    return Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path), broker


def test_submit_buy_sends_a_buy_that_fits_unchanged(tmp_path: Path) -> None:
    strategy, broker = _strategy_at(tmp_path, "100", "100")

    order, sent = submit_buy(strategy, "AAA", 98.0, 100.0)

    assert sent == Decimal(98) and order.quantity == Decimal(98)
    assert [o.quantity for o in broker.submitted] == [Decimal(98)]


def test_submit_buy_resends_a_refused_buy_once_at_the_most_that_fits(tmp_path: Path) -> None:
    strategy, broker = _strategy_at(tmp_path, "100", "100.5")
    price = float(strategy.get_last_price("AAA") or 0)  # the caller's sizing read; the quote then ticks up

    order, sent = submit_buy(strategy, "AAA", 98.0, price, min_trade=50.0)

    assert sent == Decimal("97.512437")  # 9,800 / 100.50, floored to 6 decimals
    assert order.quantity == Decimal("97.512437") and [o.quantity for o in broker.submitted] == [Decimal("97.512437")]


def test_submit_buy_keeps_the_callers_whole_shares(tmp_path: Path) -> None:
    strategy, broker = _strategy_at(tmp_path, "100", "100.5")
    strategy.get_last_price("AAA")

    _, sent = submit_buy(strategy, "AAA", 98.0, 100.0, decimals=0)

    assert sent == Decimal(97) and [o.quantity for o in broker.submitted] == [Decimal(97)]


def test_submit_buy_sends_nothing_when_the_most_that_fits_is_under_the_minimum_trade(tmp_path: Path) -> None:
    strategy, broker = _strategy_at(tmp_path, "100", "10000")  # 9,800 buys 0.98 shares at 10,000: worth 98 at the caller's 100
    strategy.get_last_price("AAA")

    with pytest.raises(InsufficientFundsError):
        submit_buy(strategy, "AAA", 98.0, 100.0, min_trade=100.0)
    assert broker.submitted == []


def test_the_buys_are_sized_from_spendable_cash_so_a_stop_sell_funds_nothing(tmp_path: Path) -> None:
    """An earlier stop-loss on A may never trigger: `spendable_cash()` gives it no credit, so no buy is sent only to be refused."""
    strategy, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    broker.tracker.track_unprocessed(strategy.create_order("A", 100, "sell", stop_price=45))

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    assert placed == []
    assert rebalancer.refused == 0


def test_a_refusal_never_raises_the_available_cash_back_above_the_running_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A's outcome is unknown (its cost stays spent, but the broker does not track it here); B is then refused with a spendable that misses A."""
    strategy, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "C": 20, "SHV": 100}, cash=5200)
    original = strategy.submit_order

    def submit(order):
        if order.asset.symbol == "A":
            raise BrokerUncertainError("timeout")
        if order.asset.symbol == "B":
            raise InsufficientFundsError("B ticked up", spendable=Decimal(5000), max_quantity=Decimal(199))
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25, "C": 0.25}, cash_buffer=0.02))

    # 5,000 - A's 3,500 = 1,500 left for C (75 shares), not the refusal's 5,000 (125 shares)
    assert placed == [PlacedOrder("A", "buy", 70.0, confirmed=False), PlacedOrder("C", "buy", 75.0)]


def test_a_refused_sell_is_not_counted_as_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)
    original = strategy.submit_order

    def submit(order):
        if order.asset.symbol == "B":
            raise BrokerError("cannot sell")
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    # available = min(bp, 7,500 + 0) - 200 = 7,300: A takes 3,500, the parking buy gets the remaining 3,800
    assert _orders(broker) == [("A", "buy", 70.0), ("SHV", "buy", 38.0)]


def _raising_for(strategy: Strategy, monkeypatch: pytest.MonkeyPatch, symbol: str, side: OrderSide, error: Exception) -> dict[str, int]:
    """Make `symbol`'s `side` order raise `error`; returns how many times each symbol's orders were submitted."""
    original = strategy.submit_order
    calls: dict[str, int] = {}

    def submit(order):
        calls[order.asset.symbol] = calls.get(order.asset.symbol, 0) + 1
        if order.asset.symbol == symbol and order.side is side:
            raise error
        return original(order)

    monkeypatch.setattr(strategy, "submit_order", submit)
    return calls


def test_an_unknown_buy_counts_as_spent_for_the_buys_after_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=5200)
    calls = _raising_for(strategy, monkeypatch, "A", OrderSide.BUY, BrokerUncertainError("timeout"))

    with caplog.at_level(logging.WARNING):
        placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.35}, cash_buffer=0.02))

    # available 5,000 less A's 3,500 leaves 1,500 = 60 B (a refused A would have left B 140)
    assert placed == [PlacedOrder("A", "buy", 70.0, confirmed=False), PlacedOrder("B", "buy", 60.0)]
    assert calls["A"] == 1
    assert "A" not in [symbol for symbol, _, _ in _orders(broker)]  # the fake never saw A: the broker owns its tracking
    assert "buy 70 A (toward target): outcome unknown at the broker, not resent -- check the account: timeout" in caplog.text
    assert "was refused" not in caplog.text


def test_an_unknown_sell_brings_no_proceeds_and_is_not_waited_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)
    calls = _raising_for(strategy, monkeypatch, "B", OrderSide.SELL, BrokerUncertainError("timeout"))
    waits: list[object] = []
    monkeypatch.setattr(strategy, "wait_for_orders_execution", lambda *args, **kwargs: waits.append(args) or True)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    # no proceeds: available = 7,500 - 200 = 7,300, A takes 3,500 and the parking buy gets the remaining 3,800
    assert placed == [PlacedOrder("B", "sell", 100.0, confirmed=False), PlacedOrder("A", "buy", 70.0), PlacedOrder("SHV", "buy", 38.0)]
    assert calls["B"] == 1
    assert waits == []


def test_an_unknown_parking_sell_does_not_fund_the_buys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    calls = _raising_for(strategy, monkeypatch, "SHV", OrderSide.SELL, BrokerUncertainError("timeout"))

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert placed == [PlacedOrder("SHV", "sell", 60.0, confirmed=False)]  # the available cash is still 0: no buy
    assert calls == {"SHV": 1}


def test_an_unknown_outcome_does_not_feed_the_buying_power_resync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=5200)
    message = json.dumps({"buying_power": "100000", "message": "insufficient buying power"})
    _raising_for(strategy, monkeypatch, "A", OrderSide.BUY, BrokerUncertainError(message))

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.35}, cash_buffer=0.02))

    assert placed == [PlacedOrder("A", "buy", 70.0, confirmed=False), PlacedOrder("B", "buy", 60.0)]  # not resynced up to 140


def test_a_target_without_a_price_is_skipped_with_a_warning_and_the_others_proceed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"B": 25, "SHV": 100}, cash=200)

    with caplog.at_level(logging.WARNING):
        rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert not any(symbol == "A" and side == "buy" for symbol, side, _ in _orders(broker))
    assert ("B", "buy", 100.0) in _orders(broker)
    assert "A" in caplog.text and "No price" in caplog.text


def test_a_failed_price_lookup_abandons_the_review_after_the_forced_exit_went_out(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={}, cash=0)
    broker.market_data_error = BrokerError("market data is down")

    # The quantity needs no price, so the sell goes out; sizing the buys then needs the open sell's price, and the
    # rebalance raises like any broker failure after a sell (the pipeline abandons the review and logs `placed`).
    with pytest.raises(BrokerError, match="market data is down"):
        rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert rebalancer.placed == [PlacedOrder("A", "sell", 100.0)]


def test_an_order_below_the_minimum_size_is_skipped(tmp_path: Path) -> None:
    # Only 240 in cash: 240 - 200 reserve = 40 available, below the 50 minimum trade.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"SHV": 100}, cash=240)

    assert rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02)) == []
    assert broker.submitted == []


def test_quantities_are_floored_so_a_cost_never_exceeds_the_money(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 33.33, "SHV": 100}, cash=10_000)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    (_, _, quantity) = next(order for order in _orders(broker) if order[0] == "A")
    assert quantity == 105.010501
    assert quantity * 33.33 <= 3500


def test_each_buy_is_capped_by_a_fresh_spendable_cash_that_counts_the_earlier_buys_fees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backtest's buying power nets the fees of pending buys (`BacktestBroker._projection`); our estimate only their cost."""
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=10_000)

    def get_account() -> AccountBalances:
        pending = sum((order.quantity or Decimal(0)) * broker.last_prices[order.asset.symbol] for order in broker.tracker.get_active_orders() if order.side is OrderSide.BUY)
        return AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(10_000) - pending * Decimal("1.001"))

    monkeypatch.setattr(broker, "get_account", get_account)

    placed = rebalancer.rebalance(target_portfolio({"A": 0.49, "B": 0.49}, cash_buffer=0.02))

    # B gets what is left after A's 4,900 and its 4.90 fee, less the 200 buffer: 4,895.10 = 195.804 shares, not a refused 196
    assert placed == [PlacedOrder("A", "buy", 98.0), PlacedOrder("B", "buy", 195.804)]
    assert rebalancer.refused == 0


def test_buying_power_below_cash_limits_the_buys(tmp_path: Path) -> None:
    # A margin-free account reports less buying power than cash (unsettled funds): the smaller figure sizes the buys.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000, buying_power=2_200)

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert _orders(broker)[0] == ("A", "buy", 40.0)  # (2,200 - 200 reserve) / 50


def test_a_portfolio_value_that_is_not_positive_places_nothing(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 1}, prices={"A": 50}, cash=0, portfolio_value=0.0)

    assert rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02)) == []
    assert rebalancer.current_weights() == {}


def test_holdings_exclude_the_parking_instrument_and_empty_positions(tmp_path: Path) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"B": 10, "A": 100, "SHV": 10, "C": 0}, prices={"A": 50, "B": 25, "SHV": 100}, cash=0)

    assert rebalancer.holdings() == ["A", "B"]


def _with_short(broker: FakeBroker, symbol: str, quantity: float) -> None:
    """IBKR and Tastytrade report a short as a POSITIVE quantity with side=SHORT (brokers/CLAUDE.md)."""
    broker.positions = [*broker.positions, Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.SHORT)]


def test_a_short_position_is_not_a_holding_and_has_no_weight(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "X": 20, "SHV": 100}, cash=0)
    _with_short(broker, "X", 100)

    assert rebalancer.holdings() == ["A"]
    assert rebalancer.current_weights() == {"A": 0.5}


def test_a_short_position_is_never_sold_and_the_run_warns_about_it(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Selling against a short opens more short: X (short 100, not in the target) must get no order."""
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "X": 20, "SHV": 100}, cash=5_000)
    _with_short(broker, "X", 100)

    with caplog.at_level(logging.WARNING):
        rebalancer.rebalance(target_portfolio({"A": 0.5}, cash_buffer=0.02))

    assert all(symbol != "X" for symbol, _, _ in _orders(broker))
    assert any("short" in record.getMessage().lower() and "X" in record.getMessage() for record in caplog.records)


def test_current_weights_are_each_stocks_share_of_portfolio_value(tmp_path: Path) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 100, "B": 10, "SHV": 10}, prices={"A": 50, "B": 25, "SHV": 100}, cash=0)

    assert rebalancer.current_weights() == {"A": 0.5, "B": 0.025}


def test_current_weights_count_open_orders(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100, "B": 10, "SHV": 10}, prices={"A": 50, "B": 25, "C": 20, "SHV": 100}, cash=0)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.SELL, quantity=Decimal(100)))  # A is being sold in full
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("C"), side=OrderSide.BUY, quantity=Decimal(50)))  # C is being bought
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("SHV"), side=OrderSide.BUY, quantity=Decimal(5)))  # parking is never a weight

    assert rebalancer.current_weights() == {"B": 0.025, "C": 0.1}
    assert rebalancer.holdings() == ["A", "B"]  # holdings stay the filled positions


# --- orders still open from an earlier review -----------------------------------------------------------


def test_a_review_does_not_send_again_what_is_already_in_flight(tmp_path: Path) -> None:
    # FakeBroker tracks every submitted order as open and never fills it: the second review finds the first one's orders pending.
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000)
    target = target_portfolio({"A": 0.35}, cash_buffer=0.02)

    first = rebalancer.rebalance(target)
    second = rebalancer.rebalance(target)

    assert first == [PlacedOrder("A", "buy", 70.0), PlacedOrder("SHV", "buy", 63.0)]
    assert second == []
    assert len(broker.submitted) == 2


def test_a_forced_exit_is_not_sold_twice_while_the_first_sell_is_open(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)
    target = target_portfolio({}, cash_buffer=0.02)

    first = rebalancer.rebalance(target, forced_exits=["A"])
    second = rebalancer.rebalance(target, forced_exits=["A"])

    assert first == [PlacedOrder("A", "sell", 100.0), PlacedOrder("SHV", "buy", 98.0)]
    assert second == []


def test_only_the_unfilled_part_of_a_partly_filled_order_counts_as_in_flight(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "SHV": 100}, cash=10_000)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.BUY, quantity=Decimal(70), filled_quantity=Decimal(30)))

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    # 40 shares (2,000) are still to come against a 3,500 target: 1,500 more, i.e. 30 shares
    assert PlacedOrder("A", "buy", 30.0) in placed


def test_a_sell_never_exceeds_what_is_held_and_not_already_being_sold(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.SELL, quantity=Decimal(60)))

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert placed[0] == PlacedOrder("A", "sell", 40.0)


# --- sizing counts the sells, never a stale buying power ------------------------------------------------


def test_an_earlier_pending_sell_funds_todays_buy(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.SELL, quantity=Decimal(100)))

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    # credit 5,000 -> available 5,200 - 200 reserve = 5,000: B takes 3,500, the other 1,500 is parked
    assert placed == [PlacedOrder("B", "buy", 140.0), PlacedOrder("SHV", "buy", 15.0)]


def test_an_earlier_pending_buy_is_debited_from_todays_cash(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={}, prices={"A": 50, "B": 25, "SHV": 100}, cash=5000)
    broker.tracker.track_unprocessed(Order(strategy_name="bill_ackman", asset=Asset("A"), side=OrderSide.BUY, quantity=Decimal(40)))

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    # cash 5,000 - 2,000 already committed = 3,000; less the 200 reserve = 2,800 -> 112 shares of B, nothing left to park
    assert placed == [PlacedOrder("B", "buy", 112.0)]


def test_this_runs_sell_credit_is_not_lost_to_a_stale_buying_power(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=200)
    reads = {"count": 0}

    def get_account() -> AccountBalances:
        # The backtest projection: buying power only credits the sell once it has been submitted.
        reads["count"] += 1
        return AccountBalances(cash=Decimal(200), portfolio_value=Decimal(10_000), buying_power=Decimal(200 if reads["count"] == 1 else 5200))

    monkeypatch.setattr(broker, "get_account", get_account)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    assert placed[0] == PlacedOrder("A", "sell", 100.0)
    assert PlacedOrder("B", "buy", 140.0) in placed


class _LiveFillBroker(FakeBroker):
    """Fills every order at submission, at its last price, and moves cash at once: Alpaca with a market order.

    Buying power is 4x cash (a margin account), so `min(buying_power, ...)` cannot hide a double count.
    """

    def _submit_order(self, order: Order) -> Order:
        self.submitted.append(order)
        cost = self.last_prices[order.asset.symbol] * order.quantity
        cash = self.account.cash - cost if order.side is OrderSide.BUY else self.account.cash + cost
        self.account = AccountBalances(cash=cash, portfolio_value=self.account.portfolio_value, buying_power=4 * cash)
        order.status, order.filled_quantity = OrderStatus.FILL, order.quantity
        return order  # never tracked as active: it is already filled


def _live_book(tmp_path: Path, *, positions: dict[str, float], prices: dict[str, float], cash: float) -> tuple[_LiveFillBroker, Rebalancer]:
    broker = _LiveFillBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG) for symbol, quantity in positions.items()]
    broker.last_prices = {symbol: Decimal(str(price)) for symbol, price in prices.items()}
    # Portfolio value 10,000 while only cash + A is visible: the money, not the targets, binds the buys.
    broker.account = AccountBalances(cash=Decimal(str(cash)), portfolio_value=Decimal(10_000), buying_power=Decimal(str(4 * cash)))
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return broker, Rebalancer(strategy, AckmanParams())


def test_a_live_filled_sell_funds_the_buys_once(tmp_path: Path) -> None:
    """Sell A (5,000) then buy B and C: at most pre-sell cash 1,000 + 5,000 - the 200 reserve = 5,800 is spent."""
    broker, rebalancer = _live_book(tmp_path, positions={"A": 100}, prices={"A": 50, "B": 25, "C": 20, "SHV": 100}, cash=1_000)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.49, "C": 0.49}, cash_buffer=0.02))

    # B wants 4,900 (196 shares), C gets the 900 left (45 shares); re-reading the filled cash would have bought far more C
    assert placed == [PlacedOrder("A", "sell", 100.0), PlacedOrder("B", "buy", 196.0), PlacedOrder("C", "buy", 45.0)]
    assert broker.account.cash >= Decimal(200)


def test_a_live_filled_buy_is_not_deducted_twice_from_the_next_buy(tmp_path: Path) -> None:
    """Two buys from 5,000 cash: B 2,400 then C 2,400; C must not be sized on (cash after B) - B's cost again."""
    broker, rebalancer = _live_book(tmp_path, positions={}, prices={"B": 25, "C": 20, "SHV": 100}, cash=5_000)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.24, "C": 0.24}, cash_buffer=0.02))

    assert placed[:2] == [PlacedOrder("B", "buy", 96.0), PlacedOrder("C", "buy", 120.0)]
    assert broker.account.cash >= Decimal(0)


class _DeferredFillBroker(FakeBroker):
    """Alpaca on a cash account (multiplier 1): a sell is accepted at once but its money (and buying power) lands only when it fills.

    `fill_sells` is what the fill looks like; the test's clock runs it during the rebalancer's wait.
    """

    def fill_sells(self, symbols: set[str] | None = None) -> None:
        for order in self.submitted:
            if order.side is OrderSide.SELL and order.status is not OrderStatus.FILL and (symbols is None or order.asset.symbol in symbols):
                proceeds = self.last_prices[order.asset.symbol] * order.quantity
                cash = self.account.cash + proceeds
                self.account = AccountBalances(cash=cash, portfolio_value=self.account.portfolio_value, buying_power=cash)
                order.status, order.filled_quantity = OrderStatus.FILL, order.quantity
                self.tracker.process_trade_event(order, OrderEvent.FILLED, price=self.last_prices[order.asset.symbol], filled_quantity=order.quantity)


def _deferred_book(tmp_path: Path, *, fills_on_wait: bool = True) -> tuple[_DeferredFillBroker, FakeClock, Rebalancer]:
    """A held (100 x 50), B to buy (25), cash 200 = exactly the reserve: the sale of A is the only money for B."""
    clock = FakeClock(et(2026, 9, 14, 10))
    broker = _DeferredFillBroker(clock, strategy_name="bill_ackman")
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset("A"), quantity=Decimal(100), side=PositionSide.LONG)]
    broker.last_prices = {"A": Decimal(50), "B": Decimal(25), "SHV": Decimal(100)}
    broker.account = AccountBalances(cash=Decimal(200), portfolio_value=Decimal(10_000), buying_power=Decimal(200))
    if fills_on_wait:
        clock.on_wait = broker.fill_sells
    strategy = Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    return broker, clock, Rebalancer(strategy, AckmanParams())


def test_buys_funded_by_the_parking_sell_wait_for_it_to_fill(tmp_path: Path) -> None:
    """A cash account (buying power = cash): the SHV sale is the only money for A and B, and buying power has it only once it fills."""
    broker, clock, rebalancer = _deferred_book(tmp_path)
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset("SHV"), quantity=Decimal(98), side=PositionSide.LONG)]
    broker.last_prices = {"A": Decimal(50), "B": Decimal(25), "SHV": Decimal(100)}

    placed = rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert placed == [PlacedOrder("SHV", "sell", 60.0), PlacedOrder("A", "buy", 70.0), PlacedOrder("B", "buy", 100.0)]
    assert clock.waits  # it waited on the clock for the SHV sell


def test_buys_wait_for_the_sells_to_fill_so_buying_power_has_the_proceeds(tmp_path: Path) -> None:
    """2026-10-27 bull_bear: SYRE's sell was accepted at 16:18:21.86, buying power was read at .05 (still the old 193.87), the fill came at .40."""
    broker, clock, rebalancer = _deferred_book(tmp_path)

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    # buying power 5,200 once A has filled; available = min(5,200, 200 + 5,000) - 200 reserve = 5,000: B takes 3,500, 1,500 is parked
    assert placed == [PlacedOrder("A", "sell", 100.0), PlacedOrder("B", "buy", 140.0), PlacedOrder("SHV", "buy", 15.0)]
    assert clock.waits  # it did wait on the clock for the sell


def test_funding_trims_are_waited_on_too_before_the_buys_in_paper_and_live(tmp_path: Path) -> None:
    broker, clock, rebalancer = _deferred_book(tmp_path)
    broker.positions = [
        Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(quantity), side=PositionSide.LONG) for symbol, quantity in (("A", 50), ("B", 50), ("C", 50), ("D", 23))
    ]
    broker.last_prices = {"A": Decimal(50), "B": Decimal(50), "C": Decimal(50), "D": Decimal(100), "SHV": Decimal(100)}

    placed = rebalancer.rebalance(target_portfolio({"A": 0.21, "B": 0.21, "C": 0.21, "D": 0.35}, cash_buffer=0.02))

    assert placed == [PlacedOrder("A", "sell", 8.0), PlacedOrder("B", "sell", 8.0), PlacedOrder("C", "sell", 8.0), PlacedOrder("D", "buy", 12.0)]
    assert clock.waits  # the buys waited for the funding trims to fill, like for any other sell


def test_a_sell_that_never_fills_costs_the_buys_but_not_the_review(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, clock, rebalancer = _deferred_book(tmp_path, fills_on_wait=False)

    with caplog.at_level(logging.WARNING):
        placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    # buying power is still 200: nothing to spend after the reserve, and the wait ended at its timeout
    assert placed == [PlacedOrder("A", "sell", 100.0)]
    assert "did not fill" in caplog.text
    assert sum(clock.waits) == pytest.approx(30.0)


def test_a_parking_sell_with_no_buy_planned_is_not_waited_on(tmp_path: Path) -> None:
    """SHV is sold only because it sits above its target: no buy needs its money, so nothing waits for it."""
    broker, clock, rebalancer = _deferred_book(tmp_path)
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset("SHV"), quantity=Decimal(98), side=PositionSide.LONG)]

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.5))  # parking target 5,000, holding 9,800

    assert placed == [PlacedOrder("SHV", "sell", 48.0)]
    assert clock.waits == []


def test_after_a_timed_out_wait_only_the_later_sells_are_waited_on(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A's sell never fills (first wait times out); the SHV sell sent next fills at once and is all the second wait waits for."""
    broker, clock, rebalancer = _deferred_book(tmp_path, fills_on_wait=False)
    broker.positions.append(Position(strategy_name="bill_ackman", asset=Asset("SHV"), quantity=Decimal(30), side=PositionSide.LONG))
    clock.on_wait = lambda: broker.fill_sells({"SHV"})

    with caplog.at_level(logging.WARNING):
        placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    # spendable after the SHV fill: min(3,200, 3,200 + A's open 5,000) - 200 = 3,000 for B
    assert placed == [PlacedOrder("A", "sell", 100.0), PlacedOrder("SHV", "sell", 30.0), PlacedOrder("B", "buy", 120.0)]
    # the second wait ends once SHV has filled: it no longer waits (and times out) on A again
    assert "the buys go out on the broker's buying power as it stands" not in caplog.text


def test_no_wait_without_a_sell(tmp_path: Path) -> None:
    broker, clock, rebalancer = _deferred_book(tmp_path)
    broker.positions = []
    broker.account = AccountBalances(cash=Decimal(10_000), portfolio_value=Decimal(10_000), buying_power=Decimal(10_000))

    placed = rebalancer.rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    assert PlacedOrder("B", "buy", 140.0) in placed
    assert clock.waits == []


def test_a_backtest_does_not_wait_for_its_sells(tmp_path: Path) -> None:
    """A backtest fills on the next bar and its buying power already credits a submitted sell: waiting would only move the simulated clock."""
    broker, clock, _ = _deferred_book(tmp_path, fills_on_wait=False)
    strategy = Strategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)

    Rebalancer(strategy, AckmanParams()).rebalance(target_portfolio({"B": 0.35}, cash_buffer=0.02))

    assert clock.waits == []


def test_a_sell_logs_the_estimated_gain_against_the_positions_average_cost(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500, costs={"B": 20})

    with caplog.at_level(logging.INFO):
        rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert "sell 100 B @ $25.00 (not in the target portfolio, est. P&L +$500.00 / +25.0%)" in caplog.text


def test_a_sell_logs_the_estimated_loss_with_a_minus_sign(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 100}, prices={"A": 50, "SHV": 100}, cash=5000, costs={"A": 60})

    with caplog.at_level(logging.INFO):
        rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert "sell 100 A @ $50.00 (forced exit, est. P&L -$1000.00 / -16.7%)" in caplog.text


def test_a_trim_reports_the_pnl_of_the_quantity_sold_only(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 90}, prices={"A": 50, "SHV": 100}, cash=5500, costs={"A": 40})  # sells 20

    with caplog.at_level(logging.INFO):
        rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert "sell 20 A @ $50.00 (trim above target, est. P&L +$200.00 / +25.0%)" in caplog.text


def test_a_sell_without_a_known_cost_logs_no_pnl(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)

    with caplog.at_level(logging.INFO):
        rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert "sell 100 B @ $25.00 (not in the target portfolio)" in caplog.text
    assert "P&L" not in caplog.text


def test_a_buy_logs_no_pnl(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 59}, prices={"A": 50, "SHV": 100}, cash=7050, costs={"A": 40})

    with caplog.at_level(logging.INFO):
        rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))

    assert "buy 11 A @ $50.00 (toward target)" in caplog.text
    assert "P&L" not in caplog.text


def test_a_sell_priced_at_zero_logs_no_pnl(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={}, cash=0, costs={"A": 60})
    broker.market_data_error = BrokerError("market data is down")

    with caplog.at_level(logging.INFO), pytest.raises(BrokerError):  # sizing the buys then fails: see the test above
        rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert "sell 100 A @ $0.00 (forced exit)" in caplog.text
    assert "P&L" not in caplog.text


# --- a buy with nothing left to pay for it: parking first, then trims -------------------------------------


def test_a_buy_beyond_the_band_with_no_cash_is_funded_by_trimming_stocks_inside_their_band(tmp_path: Path) -> None:
    # fully invested: A, B and C are 400 over their targets (inside the 500 band, so no trim of their own), D is 900 under
    _, broker, rebalancer = _book(tmp_path, positions={"A": 50, "B": 50, "C": 50, "D": 23}, prices={"A": 50, "B": 50, "C": 50, "D": 100, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.22, "B": 0.22, "C": 0.22, "D": 0.32}, cash_buffer=0.02))

    assert _orders(broker) == [("A", "sell", 6.0), ("B", "sell", 6.0), ("C", "sell", 6.0), ("D", "buy", 9.0)]


def test_parking_is_sold_first_and_only_the_rest_of_the_shortfall_is_trimmed(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 6, "A": 50, "B": 50, "C": 50, "D": 17}, prices={"A": 50, "B": 50, "C": 50, "D": 100, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.21, "B": 0.21, "C": 0.21, "D": 0.35}, cash_buffer=0.02))

    # D needs 1,800: the 600 of parking first, then 1,200 from the three 400 excesses
    assert _orders(broker) == [("SHV", "sell", 6.0), ("A", "sell", 8.0), ("B", "sell", 8.0), ("C", "sell", 8.0), ("D", "buy", 18.0)]


def test_a_funding_trim_sells_only_the_shortfall_largest_excess_first(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 52, "B": 50, "C": 48, "D": 23}, prices={"A": 50, "B": 50, "C": 50, "D": 100, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.21, "B": 0.21, "C": 0.21, "D": 0.29}, cash_buffer=0.02))

    # D is 600 under: A is 500 over (10 shares, inside its band), B 400 over covers the remaining 100 (2 shares); C is not touched
    assert _orders(broker) == [("A", "sell", 10.0), ("B", "sell", 2.0), ("D", "buy", 6.0)]


def test_refused_orders_are_counted_and_the_count_starts_over_each_rebalance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _, rebalancer = _book(tmp_path, positions={"B": 100}, prices={"A": 50, "B": 25, "SHV": 100}, cash=7500)
    _raising_for(strategy, monkeypatch, "B", OrderSide.SELL, BrokerError("cannot sell"))

    rebalancer.rebalance(target_portfolio({"A": 0.35}, cash_buffer=0.02))
    assert rebalancer.refused == 1

    rebalancer.rebalance(target_portfolio({"B": 0.25}, cash_buffer=0.02))
    assert rebalancer.refused == 0


def test_a_refused_funding_trim_is_not_counted_as_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"A": 50, "B": 50, "C": 50, "D": 23}, prices={"A": 50, "B": 50, "C": 50, "D": 100, "SHV": 100}, cash=200)
    _raising_for(strategy, monkeypatch, "A", OrderSide.SELL, BrokerError("cannot sell"))

    rebalancer.rebalance(target_portfolio({"A": 0.22, "B": 0.22, "C": 0.22, "D": 0.32}, cash_buffer=0.02))

    # D needs 900 and A's 300 never came: B and C bring 600, so D buys 600, not 900
    assert _orders(broker) == [("B", "sell", 6.0), ("C", "sell", 6.0), ("D", "buy", 6.0)]
    assert rebalancer.refused == 1


def test_a_funding_trim_skips_a_stock_whose_excess_is_below_the_minimum_trade(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 50, "B": 50, "D": 23}, prices={"A": 50, "B": 50, "D": 100, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.21, "B": 0.247, "D": 0.40}, cash_buffer=0.02))

    # A is 400 over, B only 30 over (the smallest order is 50): only A is trimmed, D buys what that brings
    assert _orders(broker) == [("A", "sell", 8.0), ("D", "buy", 4.0)]


def test_a_stock_already_trimmed_to_its_target_is_not_trimmed_again_to_fund_the_buys(tmp_path: Path) -> None:
    # A is 1,000 over (beyond its band): step 1 sells it to target. D needs 1,200 and only 1,000 is there:
    # the missing 200 comes from B (200 over), not from A, which has no excess left (2026-11-18 run: 18 ETFs sold twice)
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100, "B": 25, "D": 23}, prices={"A": 50, "B": 100, "D": 100, "SHV": 100}, cash=200)

    rebalancer.rebalance(target_portfolio({"A": 0.40, "B": 0.23, "D": 0.35}, cash_buffer=0.02))

    assert _orders(broker) == [("A", "sell", 20.0), ("B", "sell", 2.0), ("D", "buy", 12.0)]


def test_a_sell_with_an_unknown_outcome_is_not_sold_again_to_fund_the_buys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker, rebalancer = _book(tmp_path, positions={"A": 100, "B": 25, "D": 23}, prices={"A": 50, "B": 100, "D": 100, "SHV": 100}, cash=200)
    calls = _raising_for(strategy, monkeypatch, "A", OrderSide.SELL, BrokerUncertainError("timeout"))

    rebalancer.rebalance(target_portfolio({"A": 0.40, "B": 0.23, "D": 0.35}, cash_buffer=0.02))

    # A's unknown sell is sent once, never again, and brings no proceeds: only B funds the buys
    assert calls["A"] == 1
    assert _orders(broker) == [("B", "sell", 2.0), ("D", "buy", 2.0)]


def test_nothing_is_trimmed_when_no_buy_is_due_even_if_cash_is_below_the_reserve(tmp_path: Path) -> None:
    # cash 100 is under the 200 reserve, but A is 500 over (inside its band) and nothing is waiting to be bought: no money is needed
    _, broker, rebalancer = _book(tmp_path, positions={"A": 50, "B": 73}, prices={"A": 52, "B": 100, "SHV": 100}, cash=100)

    rebalancer.rebalance(target_portfolio({"A": 0.21, "B": 0.70}, cash_buffer=0.02))

    assert _orders(broker) == []

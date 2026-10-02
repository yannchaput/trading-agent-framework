from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.portfolio import target_portfolio
from trading_agent_framework.strategies.bill_ackman.rebalancer import PlacedOrder, Rebalancer
from trading_agent_framework.utils.errors import BrokerError

# Portfolio value 10,000 in every test: the band is 500 (5%), the smallest order 50 (0.5%), the cash reserve 200 (2%).


def _book(
    tmp_path: Path,
    *,
    positions: dict[str, float],
    prices: dict[str, float],
    cash: float,
    portfolio_value: float = 10_000.0,
    buying_power: float = 1_000_000.0,
) -> tuple[Strategy, FakeBroker, Rebalancer]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    broker.positions = [Position(strategy_name="bill_ackman", asset=Asset(symbol), quantity=Decimal(str(quantity)), side=PositionSide.LONG) for symbol, quantity in positions.items()]
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


def test_a_target_without_a_price_is_skipped_with_a_warning_and_the_others_proceed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"SHV": 98}, prices={"B": 25, "SHV": 100}, cash=200)

    with caplog.at_level(logging.WARNING):
        rebalancer.rebalance(target_portfolio({"A": 0.35, "B": 0.25}, cash_buffer=0.02))

    assert not any(symbol == "A" and side == "buy" for symbol, side, _ in _orders(broker))
    assert ("B", "buy", 100.0) in _orders(broker)
    assert "A" in caplog.text and "No price" in caplog.text


def test_a_failed_price_lookup_does_not_stop_a_forced_exit(tmp_path: Path) -> None:
    _, broker, rebalancer = _book(tmp_path, positions={"A": 100}, prices={}, cash=0)
    broker.market_data_error = BrokerError("market data is down")

    placed = rebalancer.rebalance(target_portfolio({}, cash_buffer=0.02), forced_exits=["A"])

    assert placed == [PlacedOrder("A", "sell", 100.0)]  # the quantity needs no price; no parking buy without one


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


def test_current_weights_are_each_stocks_share_of_portfolio_value(tmp_path: Path) -> None:
    _, _, rebalancer = _book(tmp_path, positions={"A": 100, "B": 10, "SHV": 10}, prices={"A": 50, "B": 25, "SHV": 100}, cash=0)

    assert rebalancer.current_weights() == {"A": 0.5, "B": 0.025}


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

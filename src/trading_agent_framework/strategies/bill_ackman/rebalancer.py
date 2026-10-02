"""The only module that places orders: sells first, then buys, then the parking instrument (SHV).

Follows the shape of `cross_momentum.rebalance` and the repo's cash-account rules: sell orders are submitted
before any buy, a buy is sized against the SMALLER of `buying_power` and cash plus the estimated proceeds of the
sells submitted in this run, minus the cash buffer (never `buying_power` alone: on a margin account it exceeds
cash), and quantities are floored (`fractional_qty`) so a cost never exceeds the money available.

Orders still open from an earlier review count: a pending buy counts toward the position and a pending sell is
deducted, so a review never sends again what is already in flight. (A backtest fills an order on the next bar, so a
review that follows closely finds the previous orders still pending; in paper/live a market order is normally filled by then.)
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.portfolio import TargetPortfolio
from trading_agent_framework.utils.errors import TradingFrameworkError
from trading_agent_framework.utils.helpers import fractional_qty, parse_insufficient_buying_power

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float


class Rebalancer:
    def __init__(self, strategy: Strategy, params: AckmanParams) -> None:
        self._strategy = strategy
        self._params = params
        self.placed: list[PlacedOrder] = []  # the orders accepted by the current/last `rebalance`, readable if it raises half way

    # --- what is held ----------------------------------------------------------------------------

    def holdings(self) -> list[str]:
        """The stocks held (every long position except the parking instrument), sorted."""
        parking = self._params.parking_symbol
        return sorted(position.asset.symbol for position in self._strategy.get_positions() if position.quantity > 0 and position.asset.symbol != parking)

    def current_weights(self) -> dict[str, float]:
        """Each stock's share of portfolio value counting open orders (held + pending buys - pending sells), rounded to 4 decimals.

        A stock whose effective quantity is not positive is omitted; empty if the portfolio value is not positive.
        """
        portfolio_value = float(self._strategy.broker.get_account().portfolio_value)
        if portfolio_value <= 0:
            return {}
        parking = self._params.parking_symbol
        held = {position.asset.symbol: float(position.quantity) for position in self._strategy.get_positions() if position.quantity > 0}
        incoming, outgoing = self._in_flight()
        weights = {}
        for symbol in dict.fromkeys([*held, *incoming, *outgoing]):
            quantity = held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0)
            if quantity > 0 and symbol != parking:
                weights[symbol] = round(quantity * self._price(symbol) / portfolio_value, 4)
        return dict(sorted(weights.items()))

    def _price(self, symbol: str) -> float:
        """The last price, or 0.0 when the lookup fails: a failed quote must not abort a rebalance after sells went out."""
        try:
            return float(self._strategy.get_last_price(symbol) or 0.0)
        except TradingFrameworkError as exc:
            self._strategy.log_warning(f"No price for {symbol}: {exc}")
            return 0.0

    def _in_flight(self) -> tuple[dict[str, float], dict[str, float]]:
        """The quantity still to fill per symbol from open orders: (buys, sells)."""
        buys: dict[str, float] = {}
        sells: dict[str, float] = {}
        for order in self._strategy.broker.tracker.get_active_orders():
            if order.quantity is None:
                continue
            remaining = float(order.quantity - order.filled_quantity)
            book = buys if order.side is OrderSide.BUY else sells
            book[order.asset.symbol] = book.get(order.asset.symbol, 0.0) + remaining
        return buys, sells

    # --- the rebalance -----------------------------------------------------------------------------

    def rebalance(self, target: TargetPortfolio, forced_exits: Collection[str] = ()) -> list[PlacedOrder]:
        """Trade the book toward `target`; returns the orders that were accepted, in submission order."""
        strategy, params = self._strategy, self._params
        self.placed = []
        account = strategy.broker.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            strategy.log_warning(f"Portfolio value is {portfolio_value}: nothing to rebalance")
            return []
        parking = params.parking_symbol
        forced = set(forced_exits)
        targets = {symbol: weight for symbol, weight in target.weights.items() if symbol not in forced and symbol != parking}
        held = {position.asset.symbol: float(position.quantity) for position in strategy.get_positions() if position.quantity > 0}
        incoming, outgoing = self._in_flight()
        prices = {symbol: self._price(symbol) for symbol in dict.fromkeys([*held, *incoming, *outgoing, *targets, parking])}
        band = params.rebalance_band * portfolio_value
        min_trade = params.min_trade_pct * portfolio_value
        reserve = params.cash_buffer * portfolio_value
        refusals: list[Exception] = []  # the last refused order's error, read by the buy loop

        def sellable(symbol: str) -> float:
            """What is held and not already being sold."""
            return max(0.0, held.get(symbol, 0.0) - outgoing.get(symbol, 0.0))

        def value_of(symbol: str) -> float:
            """The position's value counting open orders: held, plus buys in flight, minus sells in flight."""
            return max(0.0, held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0)) * prices.get(symbol, 0.0)

        def submit(symbol: str, side: str, quantity: float, why: str) -> bool:
            try:
                strategy.submit_order(strategy.create_order(symbol, quantity, side, time_in_force="day"))
            except TradingFrameworkError as exc:
                refusals[:] = [exc]
                strategy.log_warning(f"{side} {quantity:g} {symbol} ({why}) was refused: {exc}")
                return False
            strategy.log_info(f"{side} {quantity:g} {symbol} @ ${prices[symbol]:.2f} ({why})")
            self.placed.append(PlacedOrder(symbol, side, quantity))
            return True

        # 1. Sells: forced exits and dropped stocks in full, stocks above their band trimmed.
        proceeds = 0.0
        for symbol in held:
            if symbol == parking:
                continue
            price = prices[symbol]
            if symbol in forced or symbol not in targets:
                quantity, why = sellable(symbol), "forced exit" if symbol in forced else "not in the target portfolio"
            else:
                excess = value_of(symbol) - targets[symbol] * portfolio_value
                if price <= 0 or excess <= band or excess < min_trade:
                    continue
                quantity, why = min(fractional_qty(excess / price), sellable(symbol)), "trim above target"
            if quantity > 0 and submit(symbol, "sell", quantity, why):
                proceeds += quantity * price

        # What the buys need, and what is available for them: the smaller of buying power and cash plus the
        # estimated proceeds of the sells just submitted (and the net credit of earlier reviews' open orders), less the
        # cash buffer.
        plan: list[tuple[str, float]] = []
        for symbol, weight in targets.items():
            difference = weight * portfolio_value - value_of(symbol)
            if difference > band and difference >= min_trade:
                if prices[symbol] <= 0:
                    strategy.log_warning(f"No price for {symbol}: not buying it this review")
                    continue
                plan.append((symbol, difference))
        earlier_credit = sum(quantity * prices.get(symbol, 0.0) for symbol, quantity in outgoing.items()) - sum(
            quantity * prices.get(symbol, 0.0) for symbol, quantity in incoming.items()
        )
        cash_term = float(account.cash) + proceeds + earlier_credit
        # A second buying power read, after the sells: BacktestBroker's projection credits them once submitted.
        buying_power_now = float(strategy.broker.get_account().buying_power)
        available = min(buying_power_now, cash_term) - reserve

        # The parking instrument gives back what it holds above its target, and funds the buys.
        parking_price = prices[parking]
        parking_value = value_of(parking)
        parking_target = target.parking_weight * portfolio_value
        if parking_price > 0 and sellable(parking) > 0:
            sell_value = 0.0
            excess = parking_value - parking_target
            if excess > band and excess >= min_trade:
                sell_value = excess
            shortfall = sum(difference for _, difference in plan) - available
            if shortfall > 0:
                sell_value = max(sell_value, min(shortfall, parking_value))
            if sell_value >= min_trade or (sell_value > 0 and sell_value >= parking_value):
                quantity = sellable(parking) if sell_value >= parking_value else min(fractional_qty(sell_value / parking_price), sellable(parking))
                if quantity > 0 and submit(parking, "sell", quantity, "above target or funding the buys"):
                    available += quantity * parking_price
                    parking_value -= quantity * parking_price

        # 2. Buys, in the trader's order.
        for symbol, difference in plan:
            if available <= 0:
                strategy.log_warning("No cash left for buying: skipping the remaining stocks")
                break
            price = prices[symbol]
            quantity = fractional_qty(min(difference, available) / price)
            if quantity * price < min_trade:
                continue
            if submit(symbol, "buy", quantity, "toward target"):
                available -= quantity * price
            else:
                # Refused for buying power: continue from the broker's own figure, not our (drifted) estimate.
                real = parse_insufficient_buying_power(refusals[0])
                if real is not None:
                    strategy.log_warning(f"Resyncing available cash to the broker-reported buying power: ${real:.2f}")
                    available = real - reserve

        # 3. Park what the stock buys left, up to the parking target.
        if parking_price > 0 and available > 0 and parking_target - parking_value > band:
            quantity = fractional_qty(min(parking_target - parking_value, available) / parking_price)
            if quantity * parking_price >= min_trade:
                submit(parking, "buy", quantity, "parking")
        return list(self.placed)

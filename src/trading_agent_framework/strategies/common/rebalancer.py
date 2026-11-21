"""The only module that places orders: sells first, then buys, then the parking instrument (SHV).

Follows the shape of `cross_momentum.rebalance` and the repo's cash-account rules: sell orders are submitted
before any buy, the buys are sized from `Strategy.spendable_cash()` read after the sells (the framework's budget:
the smaller of `buying_power` and cash after open orders, less 2% of the book), and quantities are floored
(`fractional_qty`) so a cost never exceeds the money available.

Orders still open from an earlier review count: a pending buy counts toward the position and a pending sell is
deducted, so a review never sends again what is already in flight. (A backtest fills an order on the next bar, so a
review that follows closely finds the previous orders still pending; in paper/live a market order is normally filled by then.)

An order whose outcome the broker could not tell (`BrokerUncertainError`) counts as sent, never as refused and never
resent: a buy's cost stays deducted from the cash available, a sell's proceeds are never counted, and the order is
reported with `confirmed=False`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal, Protocol

from trading_agent_framework.entities.enums import OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.common.portfolio import TargetPortfolio
from trading_agent_framework.utils.errors import BrokerUncertainError, InsufficientFundsError, TradingFrameworkError
from trading_agent_framework.utils.helpers import BUY_BUFFER, fractional_qty, parse_insufficient_buying_power

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class RebalanceParams(Protocol):
    """What the rebalancer reads from a strategy's parameters (`AckmanParams`, `BullBearParams`).

    The cash kept back from the buys is the framework's (`Strategy.spendable_cash`); a strategy's own `cash_buffer`
    only shrinks its target weights.
    """

    @property
    def parking_symbol(self) -> str:
        """Return the instrument unallocated money is parked in.

        Returns:
            The parking ticker.
        """
        ...

    @property
    def rebalance_band(self) -> float:
        """Return the drift tolerated before a position is traded.

        Returns:
            The band, as a fraction of portfolio value.
        """
        ...

    @property
    def min_trade_pct(self) -> float:
        """Return the smallest trade worth sending.

        Returns:
            The minimum trade size, as a fraction of portfolio value.
        """
        ...


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    """One order sent by a rebalance.

    Attributes:
        symbol: Ticker.
        side: `buy` or `sell`.
        quantity: Shares ordered.
        confirmed: False when the broker could not say whether it took the order (it is kept, never resent).
    """

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    confirmed: bool = True  # False: the broker could not say whether it took the order (it is kept, never resent)


def in_flight_quantities(open_orders: Iterable[Order]) -> tuple[dict[str, float], dict[str, float]]:
    """The quantity still to fill per symbol from open orders: (buys, sells). Orders sized by notional are skipped."""
    buys: dict[str, float] = {}
    sells: dict[str, float] = {}
    for order in open_orders:
        if order.quantity is None:
            continue
        remaining = float(order.quantity - order.filled_quantity)
        book = buys if order.side is OrderSide.BUY else sells
        book[order.asset.symbol] = book.get(order.asset.symbol, 0.0) + remaining
    return buys, sells


def submit_buy(strategy: Strategy, symbol: str, quantity: float | Decimal, price: float, *, min_trade: float = 0.0, decimals: int = 6) -> tuple[Order, Decimal]:
    """Submit a day market buy; one the framework refuses for money is sent once more at the most that fits.

    A caller sizes its last buy to exactly `spendable_cash()` at a price read earlier; live, any uptick since then
    makes `Strategy.submit_order` refuse it (`InsufficientFundsError`). The retry sends `exc.max_quantity` floored
    to `decimals` (6: fractional shares, 0: whole shares), unless that is nothing or worth less than `min_trade` at
    the caller's `price`. The first attempt sends `quantity` as given. Shared by the Rebalancer, cross_momentum and
    earnings_drift.

    Args:
        strategy: Strategy the order goes through.
        symbol: Ticker to buy.
        quantity: Shares the caller sized.
        price: The caller's price estimate, used only to value the reduced buy against `min_trade`.
        min_trade: Smallest buy worth sending, in money.
        decimals: Decimal places a quantity keeps.

    Returns:
        The order and the quantity actually sent (the caller's running budget deducts that one).

    Raises:
        InsufficientFundsError: The most that fits is nothing or under `min_trade` (no order sent), or the retry
            was refused too. Any other error of `Strategy.submit_order` propagates unchanged.
    """
    try:
        return strategy.submit_order(strategy.create_order(symbol, quantity, "buy")), Decimal(str(quantity))
    except InsufficientFundsError as exc:
        reduced = exc.max_quantity.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_DOWN)
        if reduced <= 0 or float(reduced) * price < min_trade:
            raise
        strategy.log_warning(f"buy {quantity:g} {symbol} refused ({exc}): sending {reduced:g} instead")
        return strategy.submit_order(strategy.create_order(symbol, reduced, "buy")), reduced


Outcome = Literal["accepted", "refused", "unknown"]


def _estimated_pnl(price: float, cost: float, quantity: float) -> str:
    """The log suffix for a sell: its gain or loss against the position's average cost, at the quoted price."""
    pnl = (price - cost) * quantity
    sign = "+" if pnl >= 0 else "-"
    return f", est. P&L {sign}${abs(pnl):.2f} / {(price / cost - 1) * 100:+.1f}%"


class Rebalancer:
    """Rebalances the account toward target weights: sells go out before the buys they fund."""

    def __init__(self, strategy: Strategy, params: RebalanceParams, *, fill_wait_seconds: float = 30.0) -> None:
        """Create a rebalancer.

        Args:
            strategy: Strategy whose account, positions and orders are used.
            params: Rebalance band, minimum trade size and parking instrument.
            fill_wait_seconds: How long the buys wait for this run's sells to fill (paper/live only).
        """
        self._strategy = strategy
        self._params = params
        self._fill_wait_seconds = fill_wait_seconds  # how long the buys wait for this run's sells to fill (paper/live only)
        self.placed: list[PlacedOrder] = []  # the orders sent by the current/last `rebalance` (accepted, or with an unknown outcome: `confirmed=False`), readable if it raises half way
        self.refused = 0  # the orders the broker refused in the current/last `rebalance`: with none placed, it tells a refusal from a rebalance that had nothing to send

    # --- what is held ----------------------------------------------------------------------------

    def _long_positions(self, *, warn: bool = False) -> list:
        """The long positions with a positive quantity.

        IBKR and Tastytrade report a short as a POSITIVE quantity with
        `side=SHORT`, and a sell against it would open more short, so a short is never a holding here: `warn` logs each one
        (the order path asks for it; the read-only views stay quiet).
        """
        positions = []
        for position in self._strategy.get_positions():
            if position.quantity <= 0:
                continue
            if position.side is PositionSide.LONG:
                positions.append(position)
            elif warn:
                self._strategy.log_warning(f"{position.asset.symbol}: short position of {position.quantity} ignored (this strategy is long-only and will not trade it)")
        return positions

    def holdings(self) -> list[str]:
        """The stocks held (every long position except the parking instrument), sorted."""
        parking = self._params.parking_symbol
        return sorted(position.asset.symbol for position in self._long_positions() if position.asset.symbol != parking)

    def current_weights(self) -> dict[str, float]:
        """Each stock's share of portfolio value counting open orders (held + pending buys - pending sells), rounded to 4 decimals.

        A stock whose effective quantity is not positive is omitted; empty if the portfolio value is not positive.
        """
        portfolio_value = float(self._strategy.broker.get_account().portfolio_value)
        if portfolio_value <= 0:
            return {}
        parking = self._params.parking_symbol
        held = {position.asset.symbol: float(position.quantity) for position in self._long_positions()}
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
        return in_flight_quantities(self._strategy.broker.tracker.get_active_orders())

    # --- the rebalance -----------------------------------------------------------------------------

    def rebalance(self, target: TargetPortfolio, forced_exits: Collection[str] = ()) -> list[PlacedOrder]:
        """Trade the book toward `target`; returns the orders sent (accepted, or with an unknown outcome: `confirmed=False`), in submission order.

        Cash invariant (live fills must never be counted twice, cf. congress_trades 275c2f3): the money for the buys
        is `strategy.spendable_cash()`, read once after the stock sells. It reads the account FIRST and the open
        orders second, counting each open order's unfilled part only, so a sell of this run that fills within seconds
        counts once (in cash, no longer open, or still open, not yet in cash) and an earlier fill is never counted
        twice. The account read at the start gives only the portfolio value the targets are measured in.
        After the stock sells go out, paper/live waits (up to `fill_wait_seconds`, 30 s) for them to fill before that
        read: a live broker credits buying power only on a fill (2026-10-27 bull_bear: read 350 ms before SYRE's fill,
        the buy was skipped). A backtest never waits (its buying power credits a submitted sell). On a timeout the
        buys are sized on buying power as it stands, with a warning. The parking sell and the funding trims that follow
        add their estimated proceeds to that figure and are waited on the same way before the first buy (every sell of
        this run, parking too, is in the waited list), each buy takes its cost off it, and every buy (parking too) is
        capped by `spendable_cash()` read again just before it (a backtest's buying power nets the earlier buys' fees).
        A failed `spendable_cash()` raises, like the account read it replaced: the caller abandons the review and
        `placed` keeps the sells already sent.
        NOT covered: an open BUY that fills between `spendable_cash()`'s account read and its open-order read (its cost
        is counted nowhere); buying power, one side of its `min()`, bounds that on a cash account only. A buy refused
        by the framework's check (`InsufficientFundsError`) lowers `available` to its `spendable` (never raises it: an
        unknown buy's cost stays spent), one refused by the broker for buying power replaces it with the broker's
        figure, and the next stock is sized on it.
        """
        strategy, params = self._strategy, self._params
        self.placed = []
        self.refused = 0
        account = strategy.broker.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            strategy.log_warning(f"Portfolio value is {portfolio_value}: nothing to rebalance")
            return []
        parking = params.parking_symbol
        forced = set(forced_exits)
        targets = {symbol: weight for symbol, weight in target.weights.items() if symbol not in forced and symbol != parking}
        positions = self._long_positions(warn=True)
        held = {position.asset.symbol: float(position.quantity) for position in positions}
        costs = {position.asset.symbol: float(position.avg_fill_price) for position in positions if position.avg_fill_price}
        incoming, outgoing = self._in_flight()
        prices = {symbol: self._price(symbol) for symbol in dict.fromkeys([*held, *incoming, *outgoing, *targets, parking])}
        band = params.rebalance_band * portfolio_value
        min_trade = params.min_trade_pct * portfolio_value
        refusals: list[Exception] = []  # the last refused order's error, read by the buy loop

        sold: dict[str, float] = {}  # this run's sells per symbol, apart from the earlier runs' `outgoing`

        def sellable(symbol: str) -> float:
            """What is held and not already being sold, by earlier runs or by this one."""
            return max(0.0, held.get(symbol, 0.0) - outgoing.get(symbol, 0.0) - sold.get(symbol, 0.0))

        def value_of(symbol: str) -> float:
            """The position's value counting open orders: held, plus buys in flight, minus sells in flight and sold by this run."""
            return max(0.0, held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0) - sold.get(symbol, 0.0)) * prices.get(symbol, 0.0)

        sells: list[Order] = []  # this run's sells (parking too), whose fills the buys wait for

        def submit(symbol: str, side: str, quantity: float, why: str) -> Outcome:
            try:
                if side == "buy":  # refused for money, retried once at the most that fits: `quantity` is what was sent
                    order, sent = submit_buy(strategy, symbol, quantity, prices[symbol], min_trade=min_trade)
                    quantity = float(sent)
                else:
                    order = strategy.submit_order(strategy.create_order(symbol, quantity, side, time_in_force="day"))
            except BrokerUncertainError as exc:  # the broker keeps the order tracked and resolves it later: never resend, never refund
                strategy.log_warning(f"{side} {quantity:g} {symbol} ({why}): outcome unknown at the broker, not resent -- check the account: {exc}")
                self.placed.append(PlacedOrder(symbol, side, quantity, confirmed=False))
                if side == "sell":  # it may have gone through: never sell the same shares twice
                    sold[symbol] = sold.get(symbol, 0.0) + quantity
                return "unknown"
            except TradingFrameworkError as exc:
                refusals[:] = [exc]
                self.refused += 1
                strategy.log_warning(f"{side} {quantity:g} {symbol} ({why}) was refused: {exc}")
                return "refused"
            if side == "sell":
                sold[symbol] = sold.get(symbol, 0.0) + quantity
                sells.append(order)
            note = _estimated_pnl(prices[symbol], costs[symbol], quantity) if side == "sell" and symbol in costs and prices[symbol] > 0 else ""
            strategy.log_info(f"{side} {quantity:g} {symbol} @ ${prices[symbol]:.2f} ({why}{note})")
            self.placed.append(PlacedOrder(symbol, side, quantity))
            return "accepted"

        # 1. Sells: forced exits and dropped stocks in full, stocks above their band trimmed.
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
            if quantity > 0:
                submit(symbol, "sell", quantity, why)

        # A live broker credits buying power only when a sell FILLS (2026-10-27: read 350 ms before SYRE's fill, it still
        # showed the old figure and the buy was skipped). A backtest's buying power already credits a submitted sell and
        # fills on the next bar, so waiting there would only move the simulated clock.
        if sells and not strategy.is_backtesting and not strategy.wait_for_orders_execution(sells, self._fill_wait_seconds):
            strategy.log_warning(f"{len(sells)} sell(s) did not fill within {self._fill_wait_seconds:g}s: sizing the buys on the broker's buying power as it stands")

        # What the buys need, and what is available for them: the framework's budget, now that the sells are out.
        plan: list[tuple[str, float]] = []
        for symbol, weight in targets.items():
            difference = weight * portfolio_value - value_of(symbol)
            if difference > band and difference >= min_trade:
                if prices[symbol] <= 0:
                    strategy.log_warning(f"No price for {symbol}: not buying it this review")
                    continue
                plan.append((symbol, difference))
        available = float(strategy.spendable_cash())  # raises like the account read it replaces: `placed` keeps the sells already sent

        # The parking instrument gives back what it holds above its target, and funds the buys.
        parking_price = prices[parking]
        parking_value = value_of(parking)
        parking_target = target.parking_weight * portfolio_value
        parked = False
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
                if quantity > 0 and submit(parking, "sell", quantity, "above target or funding the buys") == "accepted":
                    available += quantity * parking_price
                    parking_value -= quantity * parking_price
                    parked = True

        # The buys may still lack money: a fully invested book has no parking to sell and no cash. Parking went first above; now trim
        # the stocks above their targets, inside their band too, the largest excess first, and only by what the buys are short.
        unfunded = sum(difference for _, difference in plan) - available if plan else 0.0  # no buy, no money needed (cash under the reserve is not a reason to trim)
        trimmed = False
        overweight = sorted(((value_of(symbol) - targets[symbol] * portfolio_value, symbol) for symbol in held if symbol in targets), key=lambda item: -item[0])
        for excess, symbol in overweight:
            if unfunded < min_trade:
                break
            if excess < min_trade or prices[symbol] <= 0:
                continue
            quantity = min(fractional_qty(min(excess, unfunded) / prices[symbol]), sellable(symbol))
            if quantity > 0 and submit(symbol, "sell", quantity, "trim to fund the buys") == "accepted":
                available += quantity * prices[symbol]
                unfunded -= quantity * prices[symbol]
                trimmed = True
        if (parked or trimmed) and not strategy.is_backtesting and not strategy.wait_for_orders_execution(sells, self._fill_wait_seconds):
            strategy.log_warning(f"{len(sells)} sell(s) did not fill within {self._fill_wait_seconds:g}s: the buys go out on the broker's buying power as it stands")

        # 2. Buys, in the trader's order. Each buy reads the framework's figure again: it counts what our running
        # estimate cannot (a backtest's buying power nets the earlier buys' fees, which would refuse a buy sized to the last dollar).
        for symbol, difference in plan:
            available = min(available, float(strategy.spendable_cash()))
            if available <= 0:
                strategy.log_warning("No cash left for buying: skipping the remaining stocks")
                break
            price = prices[symbol]
            quantity = fractional_qty(min(difference, available) / price)
            if quantity * price < min_trade:
                continue
            outcome = submit(symbol, "buy", quantity, "toward target")
            if outcome != "refused":  # an unknown outcome may have gone through: its cost stays spent
                available -= self.placed[-1].quantity * price  # what was sent: a retry at the most that fits buys less
            elif isinstance(refusals[0], InsufficientFundsError):  # refused again after the retry, or too small to retry: never above the framework's figure
                strategy.log_warning(f"Resyncing available cash to the spendable cash: ${refusals[0].spendable:.2f}")
                available = min(available, float(refusals[0].spendable))
            elif (real := parse_insufficient_buying_power(refusals[0])) is not None:  # the broker's own figure, not our (drifted) estimate
                strategy.log_warning(f"Resyncing available cash to the broker-reported buying power: ${real:.2f}")
                available = real - float(BUY_BUFFER) * portfolio_value

        # 3. Park what the stock buys left, up to the parking target (the framework's figure read again, as for each buy).
        if parking_price > 0 and available > 0 and parking_target - parking_value > band:
            available = min(available, float(strategy.spendable_cash()))
            quantity = fractional_qty(min(parking_target - parking_value, available) / parking_price)
            if quantity > 0 and quantity * parking_price >= min_trade:
                submit(parking, "buy", quantity, "parking")
        return list(self.placed)

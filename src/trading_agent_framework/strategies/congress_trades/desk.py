"""The only code that places orders for congress_trades: the trading agent's `place_order` and `check_orders`, and the audit.

The trading agent decides which orders to send, but this desk makes a bad order impossible rather than unlikely. Every
refusal comes back as `{"error": ...}` (with the numbers the agent needs to correct it) and is logged as a warning:

- a buy only for a stock in the target portfolio, never beyond its target weight (held + open buys + this order, within
  the rebalance band), and never beyond the money: the SMALLER of `buying_power` and `cash` + the estimated proceeds of the
  sells placed this run, less what the buys placed this run already cost (never `buying_power` alone: on a margin account it
  exceeds cash);
- a sell only of a position this strategy owns (a target stock or one it ordered before: a shared account's other positions are
  left alone), never above what is held and not already being sold, and never below a target stock's weight;
- sells go out before the first buy of the run, there are no shorts, and an order below the minimum trade size is refused
  (unless it closes a position).

`check_orders` waits (`strategy.wait_for_orders_execution`, a simulated wait in a backtest) and reports each order. A backtest
fills a market order on the NEXT bar, so a daily backtest finds the orders still `working` at the tick that sent them: the
report is then accepted with that status and `audit` flags the unfilled orders, which makes the pipeline re-check on later ticks.

Tools may run on parallel LangGraph worker threads: every public method that touches the desk's state holds one re-entrant lock
(the wait for fills does not, so a slow wait never blocks another tool).

This module deliberately has NO `from __future__ import annotations` for the tool closures: the agent layer builds each tool's
schema from the function's real annotations.
"""

import math
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.enums import OrderSide, OrderStatus, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.congress_trades.handoff import OrderView
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.utils.errors import BacktestError, BrokerError, TradingFrameworkError
from trading_agent_framework.utils.helpers import fractional_qty

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

_DATA_ERRORS = (BrokerError, BacktestError)
_EPS = 1e-9
_WORKING = frozenset({OrderStatus.UNPROCESSED, OrderStatus.SUBMITTED, OrderStatus.OPEN, OrderStatus.NEW, OrderStatus.CANCELLING})
_STATUS_NAMES = {OrderStatus.FILL: "filled", OrderStatus.PARTIAL_FILL: "partially_filled", OrderStatus.CANCELED: "canceled", OrderStatus.ERROR: "rejected", OrderStatus.EXPIRED: "expired"}


def order_status(order: Order) -> str:
    """The agent-facing status of an order: filled, partially_filled, working, canceled, rejected, expired or unknown."""
    if order.status in _WORKING:
        return "working"
    return _STATUS_NAMES.get(order.status, "unknown")


@dataclass(frozen=True, slots=True)
class Shortfall:
    """A target stock below its weight (buy) or a stock above it / dropped but still held (sell), counting open orders."""

    symbol: str
    kind: str  # "buy" | "sell"
    value: float  # dollars still to buy or sell
    detail: str


@dataclass(frozen=True, slots=True)
class Audit:
    unfilled: list[OrderView]  # this run's orders that are not fully filled
    shortfalls: list[Shortfall]
    in_flight: list[str] = field(default_factory=list)  # owned stocks with an order still working at the broker (this run's or an earlier one's)

    @property
    def complete(self) -> bool:
        """Nothing unfilled, nothing off target, nothing still working: the trade is finished."""
        return not self.unfilled and not self.shortfalls and not self.in_flight


class TradeDesk:
    def __init__(self, strategy: "Strategy", params: CongressParams) -> None:  # noqa: UP037
        self._strategy = strategy
        self._params = params
        self._lock = threading.RLock()
        self._target: dict[str, float] = {}
        self._traded: set[str] = set()
        self._orders: dict[str, Order] = {}
        self._checked: set[str] = set()
        self._buys_placed = False
        self._sell_proceeds = 0.0
        self._buy_cost = 0.0

    # --- the run -----------------------------------------------------------------------------------

    def begin_run(self, target: Mapping[str, float], traded: Iterable[str]) -> None:
        """Start a trading stage: the target weights (ticker -> fraction of portfolio value) and the tickers ordered in earlier runs."""
        with self._lock:
            self._target = {symbol.upper(): weight for symbol, weight in target.items() if weight > 0}
            self._traded = {symbol.upper() for symbol in traded}
            self._orders = {}
            self._checked = set()
            self._buys_placed = False
            self._sell_proceeds = 0.0
            self._buy_cost = 0.0

    @property
    def traded(self) -> list[str]:
        """Every ticker this strategy has ordered (earlier runs and this one), sorted: the state keeps it."""
        with self._lock:
            return sorted(self._traded)

    @property
    def orders(self) -> list[Order]:
        with self._lock:
            return list(self._orders.values())

    def orders_view(self) -> dict[str, OrderView]:
        """This run's orders as they are NOW, for the trade report's validation."""
        with self._lock:
            return {
                identifier: OrderView(
                    order_id=identifier,
                    symbol=order.asset.symbol,
                    side=order.side.value,
                    quantity=float(order.quantity or 0),
                    status=order_status(order),
                    filled_quantity=float(order.filled_quantity),
                    checked=identifier in self._checked,
                )
                for identifier, order in self._orders.items()
            }

    # --- the agent's tools -------------------------------------------------------------------------

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        def place_order(symbol: str, side: str, quantity: float) -> dict[str, Any]:
            """Place a market order, side buy or sell, for a target-portfolio stock; sells first, quantity in shares (fractions allowed)."""
            return self._place(symbol, side, quantity)

        def check_orders() -> dict[str, Any]:
            """Wait for the orders you placed to fill and report each one's status; call it again while some are still working."""
            return self._check()

        return [place_order, check_orders]

    # --- reading the book ----------------------------------------------------------------------------

    def _book(self) -> tuple[float, float, float, dict[str, float], dict[str, float], dict[str, float]]:
        """(portfolio value, cash, buying power, held, buys in flight, sells in flight); quantities per symbol."""
        strategy = self._strategy
        account = strategy.broker.get_account()
        held = {p.asset.symbol: float(p.quantity) for p in strategy.get_positions() if p.side is PositionSide.LONG and p.quantity > 0}
        incoming: dict[str, float] = {}
        outgoing: dict[str, float] = {}
        for order in strategy.broker.tracker.get_active_orders():
            if order.quantity is None:
                continue
            book = incoming if order.side is OrderSide.BUY else outgoing
            book[order.asset.symbol] = book.get(order.asset.symbol, 0.0) + float(order.quantity - order.filled_quantity)
        return float(account.portfolio_value), float(account.cash), float(account.buying_power), held, incoming, outgoing

    def _price(self, symbol: str) -> float | None:
        price = self._strategy.get_last_price(symbol)
        return float(price) if price is not None and float(price) > 0 else None

    def _refuse(self, message: str) -> dict[str, Any]:
        self._strategy.log_warning(f"[congress_trades] order refused: {message}")
        return {"error": message}

    # --- place_order -------------------------------------------------------------------------------

    def _place(self, symbol: Any, side: Any, quantity: Any) -> dict[str, Any]:
        symbol = str(symbol).strip().upper()
        side = str(side).strip().lower()
        if side not in ("buy", "sell"):
            return self._refuse(f"side must be 'buy' or 'sell', got {side!r}")
        if isinstance(quantity, bool):
            return self._refuse("quantity must be a number of shares")
        try:
            raw_quantity = float(quantity)
        except TypeError, ValueError, OverflowError:
            return self._refuse("quantity must be a number of shares")
        quantity = fractional_qty(raw_quantity) if math.isfinite(raw_quantity) else 0.0
        if quantity <= 0:
            return self._refuse(f"quantity must be above 0 shares, got {raw_quantity!r}")
        with self._lock:
            try:
                portfolio_value, cash, buying_power, held, incoming, outgoing = self._book()
                price = self._price(symbol)
            except _DATA_ERRORS as exc:
                return self._refuse(f"cannot read the account or the price of {symbol}: {exc}")
            if portfolio_value <= 0:
                return self._refuse(f"the portfolio value is {portfolio_value}: nothing can be traded")
            if price is None:
                return self._refuse(f"no price for {symbol}: it cannot be traded")
            params = self._params
            min_trade = params.min_trade_pct * portfolio_value
            band = params.rebalance_band * portfolio_value
            effective = (held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0)) * price
            cost = quantity * price
            if side == "buy":
                problem = self._check_buy(symbol, quantity, price, cost, effective, portfolio_value, band, min_trade, cash, buying_power)
            else:
                problem = self._check_sell(symbol, quantity, price, cost, effective, held, outgoing, portfolio_value, band, min_trade)
            if problem is not None:
                return self._refuse(problem)
            return self._submit(symbol, side, quantity, price, cost)

    def _check_buy(
        self, symbol: str, quantity: float, price: float, cost: float, effective: float, portfolio_value: float, band: float, min_trade: float, cash: float, buying_power: float
    ) -> str | None:
        weight = self._target.get(symbol)
        if weight is None:
            return f"{symbol} is not in the target portfolio ({', '.join(sorted(self._target)) or 'it is empty'}): do not buy it"
        if cost < min_trade:
            return f"buy {quantity:g} {symbol} is ${cost:,.2f}, below the minimum order of ${min_trade:,.2f}"
        room = weight * portfolio_value + band - effective
        if cost > room + _EPS:
            return (
                f"buy {quantity:g} {symbol} (${cost:,.2f}) would exceed its target weight of {weight:.4f} (${weight * portfolio_value:,.2f} of ${portfolio_value:,.2f}; "
                f"${effective:,.2f} held or on order): at most {fractional_qty(max(room, 0.0) / price):g} shares"
            )
        available = min(buying_power, cash + self._sell_proceeds - self._buy_cost)
        if cost > available + _EPS:
            return (
                f"buy {quantity:g} {symbol} (${cost:,.2f}) exceeds the money available (${max(available, 0.0):,.2f}: the smaller of buying power ${buying_power:,.2f} "
                f"and cash plus this run's sells less this run's buys): at most {fractional_qty(max(available, 0.0) / price):g} shares"
            )
        return None

    def _check_sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        cost: float,
        effective: float,
        held: Mapping[str, float],
        outgoing: Mapping[str, float],
        portfolio_value: float,
        band: float,
        min_trade: float,
    ) -> str | None:
        if symbol not in self._target and symbol not in self._traded:
            return f"{symbol} is not a position of this strategy (not in the target and never ordered by it): do not touch it"
        if self._buys_placed:
            return "sell orders go out before the first buy of a run: this run has already placed a buy"
        sellable = max(0.0, held.get(symbol, 0.0) - outgoing.get(symbol, 0.0))
        if quantity > sellable + _EPS:
            return f"sell {quantity:g} {symbol} exceeds the {sellable:g} shares held and not already being sold: there are no shorts"
        closing = quantity >= sellable - _EPS
        if cost < min_trade and not closing:
            return f"sell {quantity:g} {symbol} is ${cost:,.2f}, below the minimum order of ${min_trade:,.2f}"
        weight = self._target.get(symbol)
        if weight is not None:
            floor = weight * portfolio_value - band
            if effective - cost < floor - _EPS:
                return (
                    f"sell {quantity:g} {symbol} would leave ${effective - cost:,.2f}, below its target weight of {weight:.4f} (${weight * portfolio_value:,.2f}): "
                    f"at most {fractional_qty(max(effective - floor, 0.0) / price):g} shares"
                )
        return None

    def _submit(self, symbol: str, side: str, quantity: float, price: float, cost: float) -> dict[str, Any]:
        strategy = self._strategy
        # Recorded BEFORE the submission: an order the client raised on may still have reached the broker, and a stock we may hold must stay ours to sell.
        self._traded.add(symbol)
        try:
            order = strategy.submit_order(strategy.create_order(symbol, quantity, side, time_in_force="day"))
        except TradingFrameworkError as exc:
            return self._refuse(f"{side} {quantity:g} {symbol} was refused by the broker: {exc}")
        self._orders[order.identifier] = order
        if side == "buy":
            self._buys_placed = True
            self._buy_cost += cost
        else:
            self._sell_proceeds += cost
        strategy.log_info(f"[congress_trades] {side} {quantity:g} {symbol} @ ~${price:,.2f}")
        return {"order_id": order.identifier, "symbol": symbol, "side": side, "quantity": quantity, "status": order_status(order)}

    # --- check_orders ------------------------------------------------------------------------------

    def _check(self) -> dict[str, Any]:
        with self._lock:
            orders = list(self._orders.values())
        if not orders:
            return {"orders": [], "all_filled": True, "note": "no order was placed in this run"}
        try:
            self._strategy.wait_for_orders_execution(orders, self._params.order_wait_seconds)
        except TradingFrameworkError as exc:
            return {"error": f"could not wait for the orders: {exc}"}
        with self._lock:
            self._checked.update(order.identifier for order in orders)
        report = [self._lean(order) for order in orders]
        return {"orders": report, "all_filled": all(item["status"] == "filled" for item in report)}

    @staticmethod
    def _lean(order: Order) -> dict[str, Any]:
        item: dict[str, Any] = {
            "order_id": order.identifier,
            "symbol": order.asset.symbol,
            "side": order.side.value,
            "quantity": float(order.quantity or 0),
            "status": order_status(order),
            "filled_quantity": float(order.filled_quantity),
        }
        if order.avg_fill_price is not None:
            item["avg_price"] = float(order.avg_fill_price)
        if order.error_message:
            item["error"] = order.error_message[:200]
        return item

    # --- the audit ---------------------------------------------------------------------------------

    def audit(self) -> Audit:
        """This run's unfilled orders, and the stocks whose position (counting orders still open) is off its target by more than the band."""
        unfilled = [view for view in self.orders_view().values() if not view.filled]
        shortfalls: list[Shortfall] = []
        with self._lock:
            try:
                portfolio_value, _cash, _bp, held, incoming, outgoing = self._book()
            except _DATA_ERRORS as exc:
                self._strategy.log_warning(f"[congress_trades] audit: the account is unavailable, only unfilled orders are reported: {exc}")
                return Audit(unfilled, shortfalls)
            if portfolio_value <= 0:
                return Audit(unfilled, shortfalls)
            band = self._params.rebalance_band * portfolio_value
            min_trade = self._params.min_trade_pct * portfolio_value
            owned = set(self._target) | self._traded
            in_flight = sorted(symbol for symbol in owned if incoming.get(symbol, 0.0) > _EPS or outgoing.get(symbol, 0.0) > _EPS)
            for symbol in sorted(owned):
                try:
                    price = self._price(symbol)
                except _DATA_ERRORS as exc:
                    self._strategy.log_warning(f"[congress_trades] audit: no price for {symbol}: {exc}")
                    continue
                if price is None:
                    continue
                effective = (held.get(symbol, 0.0) + incoming.get(symbol, 0.0) - outgoing.get(symbol, 0.0)) * price
                wanted = self._target.get(symbol, 0.0) * portfolio_value
                gap = wanted - effective
                if abs(gap) <= band or abs(gap) < min_trade:
                    continue
                if gap > 0:
                    shortfalls.append(Shortfall(symbol, "buy", gap, f"{symbol} is ${effective:,.2f} against a target of ${wanted:,.2f}"))
                else:
                    detail = (
                        f"{symbol} is ${effective:,.2f}, ${-gap:,.2f} above its target of ${wanted:,.2f}"
                        if wanted
                        else f"{symbol} was dropped from the target but ${effective:,.2f} is still held or on order"
                    )
                    shortfalls.append(Shortfall(symbol, "sell", -gap, detail))
        return Audit(unfilled, shortfalls, in_flight)

"""Pure OHLC fill rules for the backtest broker: no I/O, no state, no client instances.

Fills evaluate one already-closed bar against one order's request. `BacktestBroker`
only ever calls this against a bar that closed strictly after the order was submitted
(next-bar-open) -- this module has no notion of "which bar" beyond the one it's given.

Fill rules, all "ties resolved pessimistically" (design spec, section 2):
1. When the bar's range only *touches* the trigger price (doesn't gap through it),
   the trader gets exactly that price, not a better one.
2. When the bar gaps through a LIMIT or STOP_LIMIT-limit price favorably, the trader
   gets a price improvement (e.g., buy limit at 96, gap open at 90 → fill at 90).
3. When the bar gaps through a STOP or STOP_LIMIT-stop price, the trader gets a
   worse fill price (slippage on trigger is unavoidable -- that's the point of a stop).

- MARKET: always fills, at the bar's open.
- LIMIT buy: fills iff low <= limit_price, at min(open, limit_price).
- LIMIT sell: fills iff high >= limit_price, at max(open, limit_price).
- STOP buy: triggers iff high >= stop_price, at max(open, stop_price).
- STOP sell: triggers iff low <= stop_price, at min(open, stop_price).
- STOP_LIMIT buy: triggers iff high >= stop_price AND low <= stop_limit_price,
  at min(open, stop_limit_price).
- STOP_LIMIT sell: triggers iff low <= stop_price AND high >= stop_limit_price,
  at max(open, stop_limit_price).
- TRAIL: not handled by `evaluate_fill` (it raises ValueError): a trailing stop needs a
  reference price carried from bar to bar, so `BacktestBroker` calls
  `evaluate_trailing_stop` with the reference it keeps per pending order.

Slippage is a fraction applied against the trader: buys pay price * (1 + slippage), sells
receive price * (1 - slippage). Fees are not applied here: `BacktestBroker` asks its
`TradingFeeFactory` (`brokers/fees.py`) for each order's fee.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderSide, OrderType


@dataclass(frozen=True, slots=True)
class Bar:
    """One already-closed bar's OHLC, as Decimal."""

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


@dataclass(frozen=True, slots=True)
class FillResult:
    """Where an order filled, before slippage."""

    price: Decimal


def evaluate_fill(
    *,
    order_type: OrderType,
    side: OrderSide,
    bar: Bar,
    limit_price: Decimal | None = None,
    stop_price: Decimal | None = None,
    stop_limit_price: Decimal | None = None,
) -> FillResult | None:
    """The raw fill price for one order against one bar, or None if it doesn't fill this bar."""
    if order_type is OrderType.MARKET:
        return FillResult(price=bar.open)
    if order_type is OrderType.LIMIT:
        return _limit_fill(side, bar, _require(limit_price, "limit_price"))
    if order_type is OrderType.STOP:
        return _stop_fill(side, bar, _require(stop_price, "stop_price"))
    if order_type is OrderType.STOP_LIMIT:
        return _stop_limit_fill(
            side, bar,
            _require(stop_price, "stop_price"),
            _require(stop_limit_price, "stop_limit_price"),
        )
    raise ValueError(f"evaluate_fill does not handle order_type={order_type}; use evaluate_trailing_stop for TRAIL")


def _require(value: Decimal | None, name: str) -> Decimal:
    if value is None:
        raise ValueError(f"{name} is required for this order_type")
    return value


def _limit_fill(side: OrderSide, bar: Bar, limit_price: Decimal) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.low > limit_price:
            return None
        return FillResult(price=min(bar.open, limit_price))
    if bar.high < limit_price:
        return None
    return FillResult(price=max(bar.open, limit_price))


def _stop_fill(side: OrderSide, bar: Bar, stop_price: Decimal) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.high < stop_price:
            return None
        return FillResult(price=max(bar.open, stop_price))
    if bar.low > stop_price:
        return None
    return FillResult(price=min(bar.open, stop_price))


def _stop_limit_fill(
    side: OrderSide, bar: Bar, stop_price: Decimal, stop_limit_price: Decimal
) -> FillResult | None:
    if side is OrderSide.BUY:
        if bar.high < stop_price or bar.low > stop_limit_price:
            return None
        return FillResult(price=min(bar.open, stop_limit_price))
    if bar.low > stop_price or bar.high < stop_limit_price:
        return None
    return FillResult(price=max(bar.open, stop_limit_price))


def evaluate_trailing_stop(
    *,
    side: OrderSide,
    bar: Bar,
    reference: Decimal,
    trail_price: Decimal | None = None,
    trail_percent: Decimal | None = None,
) -> tuple[FillResult | None, Decimal]:
    """One bar of a trailing stop: `(fill or None, the reference to carry to the next bar)`.

    `reference` is the high-water mark (sell) or low-water mark (buy) of the bars before this one;
    `trail_percent` is in percent (5 = 5%), as Alpaca and IBKR take it. Ties go against the trader,
    like every other rule here: the level built from the previous reference is tested first, and only
    a bar that does not trigger may move the reference with its own high (sell) or low (buy) -- the
    bar's high might have come after its low.
    """
    if (trail_price is None) == (trail_percent is None):
        raise ValueError("a trailing stop needs exactly one of trail_price or trail_percent")
    if side is OrderSide.SELL:
        level = reference - trail_price if trail_price is not None else reference * (1 - trail_percent / 100)  # ty: ignore[unsupported-operator]
        if bar.low <= level:
            return FillResult(price=min(bar.open, level)), reference
        return None, max(reference, bar.high)
    level = reference + trail_price if trail_price is not None else reference * (1 + trail_percent / 100)  # ty: ignore[unsupported-operator]
    if bar.high >= level:
        return FillResult(price=max(bar.open, level)), reference
    return None, min(reference, bar.low)


def apply_slippage(price: Decimal, side: OrderSide, *, slippage: Decimal) -> Decimal:
    """The execution price after slippage, which always goes against the trader."""
    return price * (1 + slippage) if side is OrderSide.BUY else price * (1 - slippage)

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
- TRAIL: not supported (needs a trailing reference price tracked across bars,
  which is out of scope -- design spec, section 1.3); raises ValueError.

Commission is a fraction of trade notional (e.g. Decimal("0.001") = 10bps), returned
as a per-share rate -- multiply by fill quantity for the total dollar cost. Slippage
is a fraction applied against the trader: buys pay price * (1 + slippage), sells
receive price * (1 - slippage).
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
    """Where an order filled, before commission/slippage."""

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
    raise ValueError(f"backtesting does not support order_type={order_type} (TRAIL)")


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


def apply_commission_and_slippage(
    price: Decimal, side: OrderSide, *, commission: Decimal, slippage: Decimal
) -> tuple[Decimal, Decimal]:
    """Return (execution_price, commission_per_share). The caller multiplies
    commission_per_share by the fill quantity for the total dollar cost."""
    execution_price = price * (1 + slippage) if side is OrderSide.BUY else price * (1 - slippage)
    commission_per_share = execution_price * commission
    return execution_price, commission_per_share

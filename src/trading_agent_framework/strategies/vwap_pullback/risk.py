"""Pure risk rules (spec §5): stop placement, sizing, the entry window, slots and the daily circuit breaker.

Everything that becomes an order price or a quantity is `Decimal`; float inputs come from bar maths and are
converted through `to_price`, rounded to the cent in the direction that is safe for that price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_FLOOR, ROUND_HALF_UP, ROUND_UP, Decimal

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.utils.clock import MARKET_TZ

_CENT = Decimal("0.01")


class EntryRefused(Exception):
    """Why `plan_entry` refused; the message reaches the entry agent as `{"error": ...}`."""


@dataclass(frozen=True, slots=True)
class EntryPlan:
    quantity: Decimal
    limit_price: Decimal
    stop_price: Decimal
    r_per_share: Decimal


def _ratio(value: float) -> Decimal:
    return Decimal(str(value))


def to_price(value: float | Decimal, rounding: str) -> Decimal:
    """A price in cents, rounded as asked (down for a sell stop, up for a buy limit)."""
    return Decimal(str(value)).quantize(_CENT, rounding=rounding)


def planned_stop(pullback_low: float, daily_atr: float, params: VwapPullbackParameters) -> Decimal:
    return to_price(pullback_low - params.stop_buffer_atr * daily_atr, ROUND_UP)


def plan_entry(
    *,
    trigger_close: float,
    pullback_low: float,
    last_price: Decimal,
    daily_atr: float,
    equity: Decimal,
    buying_power: Decimal,
    cash: Decimal,
    pending_sell_proceeds: Decimal,
    params: VwapPullbackParameters,
) -> EntryPlan:
    """The size, limit and stop of an entry, or `EntryRefused` with the reason."""
    stop = planned_stop(pullback_low, daily_atr, params)
    trigger = to_price(trigger_close, ROUND_HALF_UP)
    r = trigger - stop
    atr = _ratio(daily_atr)
    low, high = params.r_band_atr
    if r <= 0:
        raise EntryRefused(f"the stop {stop} is not below the trigger close {trigger}")
    if r < _ratio(low) * atr or r > _ratio(high) * atr:
        raise EntryRefused(f"risk per share {r} is outside {low}-{high} x daily ATR ({atr.quantize(_CENT)})")
    if last_price > trigger + _ratio(params.chase_guard_r) * r:
        raise EntryRefused(f"price {last_price} is more than {params.chase_guard_r}R above the trigger close {trigger}; not chasing")
    limit = (last_price + _ratio(params.entry_limit_atr) * atr).quantize(_CENT, rounding=ROUND_UP)
    by_risk = equity * _ratio(params.risk_per_trade) / r
    by_size = equity * _ratio(params.max_position_pct) / limit
    by_cash = min(buying_power, cash + pending_sell_proceeds) * _ratio(params.cash_buffer) / limit
    quantity = min(by_risk, by_size, by_cash).to_integral_value(rounding=ROUND_FLOOR)
    if quantity <= 0:
        raise EntryRefused("the position size rounds to 0 shares (not enough equity or cash for this stop distance)")
    return EntryPlan(quantity=quantity, limit_price=limit, stop_price=stop, r_per_share=r)


def in_entry_window(now: datetime, params: VwapPullbackParameters) -> bool:
    return params.no_entry_before <= now.astimezone(MARKET_TZ).time() <= params.no_entry_after


def free_slots(open_trades: int, pending_entries: int, params: VwapPullbackParameters) -> int:
    return max(0, params.max_positions - open_trades - pending_entries)


def circuit_breaker_tripped(session_pnl: Decimal, session_open_equity: Decimal, params: VwapPullbackParameters) -> bool:
    return session_open_equity > 0 and session_pnl <= -_ratio(params.max_daily_loss_pct) * session_open_equity

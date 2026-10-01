"""Pure risk rules (spec §5): stop placement, sizing, the entry window, slots and the daily circuit breaker.

Everything that becomes an order price or a quantity is `Decimal`; float inputs come from bar maths and are
converted through `to_price`, rounded to the cent in the direction that is safe for that price.

Size and prices are always the code's: `Desk.enter_long` calls `plan_entry`, and any rule it breaks comes
back as the text of an `EntryRefused`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, ROUND_UP, Decimal

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.utils.clock import MARKET_TZ

_CENT = Decimal("0.01")


class EntryRefused(Exception):
    """Why `plan_entry` refused; `Desk.enter_long` returns the message as `{"error": ...}` and logs it."""


@dataclass(frozen=True, slots=True)
class EntryPlan:
    """A sized entry: buy `quantity` at a marketable `limit_price`, protect with a stop at `stop_price`."""

    quantity: Decimal  # whole shares
    limit_price: Decimal
    stop_price: Decimal  # the initial protective stop, placed by the desk once the entry fills
    r_per_share: Decimal  # trigger close - stop: the risk unit ("R") of this trade


def _ratio(value: float) -> Decimal:
    """A float parameter as an exact Decimal (via `str`, so 0.1 stays 0.1 and not 0.1000000000000000055...)."""
    return Decimal(str(value))


def to_price(value: float | Decimal, rounding: str) -> Decimal:
    """A price in cents, rounded as asked (down for a sell stop, up for a buy limit)."""
    return Decimal(str(value)).quantize(_CENT, rounding=rounding)


def planned_stop(pullback_low: float, daily_atr: float, params: VwapPullbackParameters) -> Decimal:
    """Protective stop: `stop_buffer_atr` daily ATRs under the pullback low, rounded DOWN to the cent.

    Computed in exact Decimal: doing the subtraction in float first gives e.g. 101.1 - 0.2 = 100.8999...,
    which would round down a whole cent too far. Rounding down keeps a sell stop at or below the intended
    level (slightly wider, never tighter).
    """
    exact = Decimal(str(pullback_low)) - _ratio(params.stop_buffer_atr) * Decimal(str(daily_atr))
    return exact.quantize(_CENT, rounding=ROUND_DOWN)


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
    """The size, limit and stop of an entry, or `EntryRefused` with the reason.

    Refusals, in order: stop not below the trigger; R outside the ATR band; price already too far above the
    trigger (chasing); a size that rounds to zero shares.
    """
    stop = planned_stop(pullback_low, daily_atr, params)
    trigger = to_price(trigger_close, ROUND_HALF_UP)
    r = trigger - stop
    atr = _ratio(daily_atr)
    low, high = params.r_band_atr
    if r <= 0:
        raise EntryRefused(f"the stop {stop} is not below the trigger close {trigger}")
    # Too tight a stop is noise and gets hit by the bar's wiggle; too wide a stop makes the trade meaningless.
    if r < _ratio(low) * atr or r > _ratio(high) * atr:
        raise EntryRefused(f"risk per share {r} is outside {low}-{high} x daily ATR ({atr.quantize(_CENT)})")
    # The trigger may be several minutes old; if price has already run, the risk/reward is gone.
    if last_price > trigger + _ratio(params.chase_guard_r) * r:
        raise EntryRefused(f"price {last_price} is more than {params.chase_guard_r}R above the trigger close {trigger}; not chasing")
    # Marketable limit: a little above the last price, so it fills like a market order but caps the price paid.
    limit = (last_price + _ratio(params.entry_limit_atr) * atr).quantize(_CENT, rounding=ROUND_UP)
    # Size = the smallest of three caps:
    by_risk = equity * _ratio(params.risk_per_trade) / r  # losing R per share costs risk_per_trade of equity
    by_size = equity * _ratio(params.max_position_pct) / limit  # no position above max_position_pct of equity
    # Cash-account rule: never size against buying_power alone (a margin account's is a
    # multiple of equity); sells already submitted are credited because they fund this buy.
    by_cash = min(buying_power, cash + pending_sell_proceeds) * _ratio(params.cash_buffer) / limit
    quantity = min(by_risk, by_size, by_cash).to_integral_value(rounding=ROUND_FLOOR)
    if quantity <= 0:
        raise EntryRefused("the position size rounds to 0 shares (not enough equity or cash for this stop distance)")
    return EntryPlan(quantity=quantity, limit_price=limit, stop_price=stop, r_per_share=r)


def in_entry_window(now: datetime, params: VwapPullbackParameters) -> bool:
    """Whether `now` (any timezone) is inside the entry window, both ends inclusive, in market time."""
    return params.no_entry_before <= now.astimezone(MARKET_TZ).time() <= params.no_entry_after


def free_slots(open_trades: int, pending_entries: int, params: VwapPullbackParameters) -> int:
    """Position slots left: pending entries count as taken (they will most likely fill)."""
    return max(0, params.max_positions - open_trades - pending_entries)


def circuit_breaker_tripped(session_pnl: Decimal, session_open_equity: Decimal, params: VwapPullbackParameters) -> bool:
    """True once the session's realised + open P&L has lost `max_daily_loss_pct` of the opening equity."""
    return session_open_equity > 0 and session_pnl <= -_ratio(params.max_daily_loss_pct) * session_open_equity

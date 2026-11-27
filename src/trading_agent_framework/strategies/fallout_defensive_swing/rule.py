"""The fallout_defensive_swing decision: a pure function of the saved state and the hour's numbers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum, auto


class Phase(StrEnum):
    """Where the strategy stands: never bought, holding the mix, or in cash after a sale."""

    FRESH = auto()
    HOLDING = auto()
    CASH = auto()


class Action(StrEnum):
    """What a tick does."""

    BUY = auto()
    RESET = auto()  # buys like BUY, but only because the money sat in cash too long
    SELL = auto()
    HOLD = auto()


@dataclass(frozen=True)
class SwingState:
    """Carried from one tick to the next.

    Attributes:
        phase: Current phase.
        basis: USD spent on the last buy, buy fees included; None before the first buy.
        sale_prices: Price each ETF was last sold at.
        last_sale_session: Market date of the last sale.
        selling: A sale was started and some of the mix is still held.
        peak: Highest value after sell fees seen since the mix reached the sell target; None before that.
    """

    phase: Phase = Phase.FRESH
    basis: Decimal | None = None
    sale_prices: dict[str, Decimal] = field(default_factory=dict)
    last_sale_session: date | None = None
    selling: bool = False
    peak: Decimal | None = None


def _banks_closed(day: date) -> bool:
    """Columbus Day and Veterans Day: the exchanges trade but nothing settles."""
    columbus = day.month == 10 and day.weekday() == 0 and 8 <= day.day <= 14
    veterans = day.month == 11 and (day.day == 11 or (day.day == 12 and day.weekday() == 0))  # a Sunday one moves to Monday
    return columbus or veterans


def _weekdays_after(sale: date, session: date) -> list[date]:
    """Weekdays after the sale session, up to and including `session`."""
    days = (sale + timedelta(days=n) for n in range(1, (session - sale).days + 1))
    return [day for day in days if day.weekday() < 5]


def _settled(sale: date, session: date) -> bool:
    """Whether the proceeds of a sale made on `sale` have settled on the trading day `session` (T+1)."""
    return any(not _banks_closed(day) for day in _weekdays_after(sale, session))


def trading_days_since(sale: date, session: date) -> int:
    """Trading days after the sale session, up to and including `session`."""
    # shortcut: a market holiday counts as a trading day; use the market calendar if one day early ever matters
    return len(_weekdays_after(sale, session))


def sell_target(basis: Decimal, take_profit: Decimal) -> Decimal:
    """Value, net of sell fees, at which the mix is sold."""
    return basis * (1 + take_profit)


def updated_peak(state: SwingState, net_value: Decimal, take_profit: Decimal) -> Decimal | None:
    """The peak after seeing `net_value` (after sell fees): starts at the sell target, then only rises while holding."""
    if state.phase is not Phase.HOLDING or state.basis is None or net_value < sell_target(state.basis, take_profit):
        return state.peak
    return net_value if state.peak is None else max(state.peak, net_value)


def mean_price_ratio(prices: Mapping[str, Decimal], sale_prices: Mapping[str, Decimal]) -> Decimal:
    """Mean over the sold ETFs of today's price divided by the price it was sold at."""
    return sum((prices[symbol] / sold for symbol, sold in sale_prices.items()), Decimal(0)) / len(sale_prices)


def dip_reached(prices: Mapping[str, Decimal], sale_prices: Mapping[str, Decimal], buyback_dip: Decimal) -> bool:
    """Whether the mix has fallen `buyback_dip` or more under the prices it was sold at."""
    return mean_price_ratio(prices, sale_prices) <= 1 - buyback_dip


def decide(
    state: SwingState,
    *,
    value: Decimal,
    sell_fees: Decimal,
    prices: Mapping[str, Decimal],
    session: date,
    take_profit: Decimal,
    buyback_dip: Decimal,
    cash_reset_days: int | None,
    trail: Decimal | None,
) -> Action:
    """Choose this tick's action.

    Args:
        state: State left by the previous tick.
        value: Market value of the held mix, in USD.
        sell_fees: Estimated fees of selling all of it now.
        prices: Last price of each ETF.
        session: Today's market date.
        take_profit: Gain over the basis, net of sell fees, that triggers a sale (0.01 = 1%); with a `trail`, it starts it.
        buyback_dip: Mean fall below the sale prices that triggers the buy-back.
        cash_reset_days: Trading days in cash after which the mix is bought back whatever the price; None never does.
        trail: Fall under the saved `peak` that sells, if the value is still above the basis; None sells at the target.

    Returns:
        BUY, RESET, SELL or HOLD.
    """
    if state.phase is Phase.FRESH:
        return Action.BUY
    if state.phase is Phase.HOLDING:
        # Cash account: a sale's proceeds settle the next day, and what they bought must not be sold before then.
        if state.last_sale_session is not None and not _settled(state.last_sale_session, session):
            return Action.HOLD
        assert state.basis is not None  # every path into HOLDING sets it
        net = value - sell_fees
        if trail is None:
            reached = net >= sell_target(state.basis, take_profit)
        else:  # never under the basis: a fall past it waits (no stop-loss)
            reached = state.peak is not None and net <= state.peak * (1 - trail) and net > state.basis
        return Action.SELL if state.selling or reached else Action.HOLD
    # the state file does not require the date in CASH: without one there is nothing to count from, so only the dip buys
    expired = cash_reset_days is not None and state.last_sale_session is not None and trading_days_since(state.last_sale_session, session) >= cash_reset_days
    if dip_reached(prices, state.sale_prices, buyback_dip):
        return Action.BUY
    return Action.RESET if expired else Action.HOLD

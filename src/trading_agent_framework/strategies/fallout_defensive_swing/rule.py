"""The fallout_defensive_swing decision: a pure function of the saved state and the hour's numbers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
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
    """

    phase: Phase = Phase.FRESH
    basis: Decimal | None = None
    sale_prices: dict[str, Decimal] = field(default_factory=dict)
    last_sale_session: date | None = None
    selling: bool = False


def decide(
    state: SwingState,
    *,
    value: Decimal,
    sell_fees: Decimal,
    prices: Mapping[str, Decimal],
    session: date,
    take_profit: Decimal,
    buyback_dip: Decimal,
) -> Action:
    """Choose this tick's action.

    Args:
        state: State left by the previous tick.
        value: Market value of the held mix, in USD.
        sell_fees: Estimated fees of selling all of it now.
        prices: Last price of each ETF.
        session: Today's market date.
        take_profit: Gain over the basis, net of sell fees, that triggers a sale (0.01 = 1%).
        buyback_dip: Mean fall below the sale prices that triggers the buy-back.

    Returns:
        BUY, SELL or HOLD.
    """
    if state.phase is Phase.FRESH:
        return Action.BUY
    if state.phase is Phase.HOLDING:
        # Cash account: a sale's proceeds settle the next day, and what they bought must not be sold before then.
        if state.last_sale_session is not None and session <= state.last_sale_session:
            return Action.HOLD
        assert state.basis is not None  # every path into HOLDING sets it
        reached = value - sell_fees >= state.basis * (1 + take_profit)
        return Action.SELL if state.selling or reached else Action.HOLD
    ratios = [prices[symbol] / sold for symbol, sold in state.sale_prices.items()]
    return Action.BUY if sum(ratios) / len(ratios) <= 1 - buyback_dip else Action.HOLD

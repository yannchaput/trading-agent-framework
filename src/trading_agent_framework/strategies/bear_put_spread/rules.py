"""Pure decisions of the bear put spread: when to open and close, which strikes and expiry, how many contracts.

The stage-1 research loop calls these, and stage 2's paper strategy will call them unchanged. Holiday-shifted
expiries (a third Friday that is a market holiday settles on Thursday) are ignored: a one-day error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, slots=True)
class SpreadParameters:
    """Knobs of the strategy.

    Attributes:
        long_otm: The bought put sits this fraction below spot.
        width: Dollar distance between the two strikes.
        min_dte: Fewest calendar days to expiry at entry; the nearest monthly expiry past it is used.
        exit_dte: Close when this many days or fewer remain, so a spread never reaches expiry.
        take_profit: Close when the spread is worth this fraction of its width.
        risk_fraction: Share of equity one entry may cost (debit plus entry costs).
        min_one_fraction: When `risk_fraction` buys no contract, one is still taken if it costs at most this share.
    """

    long_otm: float = 0.02
    width: float = 5.0
    min_dte: int = 30
    exit_dte: int = 7
    take_profit: float = 0.80
    risk_fraction: float = 0.15
    min_one_fraction: float = 0.20


def third_friday(year: int, month: int) -> date:
    """The monthly option expiry: the month's third Friday."""
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def monthly_expiry(today: date, params: SpreadParameters) -> date:
    """The nearest third Friday at least `min_dte` calendar days away.

    Third Fridays are 28 to 35 days apart, so a fixed window such as 30-60 days can hold none (2026-12-17: Jan 15
    is 29 days away, Feb 19 is 64); taking the nearest one past `min_dte` always answers, and is the one inside
    `[min_dte, min_dte + 35)`.
    """
    month_index = today.year * 12 + today.month - 1
    while True:
        expiry = third_friday(month_index // 12, month_index % 12 + 1)
        if (expiry - today).days >= params.min_dte:
            return expiry
        month_index += 1


def strikes(spot: float, params: SpreadParameters) -> tuple[float, float]:
    """The `(long, short)` strikes: the long put `long_otm` below spot, rounded to a dollar; the short `width` below it."""
    long_strike = float(round(spot * (1 - params.long_otm)))
    return long_strike, long_strike - params.width


def should_open(regime: int, has_open: bool) -> bool:
    """Open on a bearish regime (-1) when no spread is open."""
    return regime == -1 and not has_open


def exit_reason(today: date, expiry: date, regime: int, value: float, width: float, params: SpreadParameters) -> str | None:
    """Why an open spread closes today, or None to hold; the first match wins.

    Args:
        today: The session.
        expiry: The spread's expiry.
        regime: Today's regime reading.
        value: The spread's model mid per share.
        width: Distance between its strikes.
        params: Thresholds.

    Returns:
        `"time"`, `"target"`, `"regime"` or None.
    """
    if (expiry - today).days <= params.exit_dte:
        return "time"
    if value >= params.take_profit * width:
        return "target"
    if regime != -1:
        return "regime"
    return None


def contracts_for(equity: float, debit: float, open_cost_per_contract: float, params: SpreadParameters) -> int:
    """Whole contracts to open.

    Args:
        equity: Account equity at entry.
        debit: Net debit per share (mid).
        open_cost_per_contract: Entry costs of one contract, dollars.
        params: Thresholds.

    Returns:
        `floor(risk_fraction * equity / cost)`; when that is 0, 1 if one contract costs at most
        `min_one_fraction * equity`, else 0.
    """
    cost = debit * 100 + open_cost_per_contract
    count = math.floor(params.risk_fraction * equity / cost)
    if count == 0 and cost <= params.min_one_fraction * equity:
        return 1
    return max(count, 0)

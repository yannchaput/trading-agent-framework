"""Pure Black-Scholes put pricing, volatility skew and per-contract costs for the bear put spread research.

Rate 0 and dividend yield 0, time in calendar days / 365. Nothing here does I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def put_price(spot: float, strike: float, vol: float, years: float) -> float:
    """Black-Scholes European put, per share, with a zero rate and no dividend.

    Args:
        spot: Underlying price.
        strike: Strike price.
        vol: Annualised volatility as a fraction (0.2 = 20%).
        years: Time to expiry in years.

    Returns:
        The put's value per share.

    Raises:
        ValueError: If an input is not finite and positive.
    """
    for name, value in (("spot", spot), ("strike", strike), ("vol", vol), ("years", years)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    root = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * years) / root
    return strike * _normal_cdf(-(d1 - root)) - spot * _normal_cdf(-d1)


@dataclass(frozen=True, slots=True)
class VolModel:
    """Volatility of a put at a strike: the VIX, scaled, plus a skew below spot.

    Attributes:
        vol_scale: Multiplier on VIX / 100.
        skew: Volatility added per unit of `ln(spot / strike)` for strikes below spot (0.8: a put 10% below spot
            gets about 8 volatility points).
    """

    vol_scale: float = 1.0
    skew: float = 0.8

    def vol(self, spot: float, strike: float, vix: float) -> float:
        """Annualised volatility for `strike`, as a fraction.

        Args:
            spot: Underlying price.
            strike: Strike price.
            vix: VIX close in index points (20 = 20%).

        Returns:
            `vix / 100 * vol_scale + skew * max(0, ln(spot / strike))`.
        """
        return vix / 100 * self.vol_scale + self.skew * max(0.0, math.log(spot / strike))


def spread_mid(spot: float, long_strike: float, short_strike: float, vix: float, years: float, model: VolModel) -> float:
    """Mid value of a bear put spread (long the higher put, short the lower), per share.

    Args:
        spot: Underlying price.
        long_strike: Strike of the bought put.
        short_strike: Strike of the sold put, below `long_strike`.
        vix: VIX close in index points.
        years: Time to expiry in years.
        model: Volatility model; each leg uses its own strike's volatility.

    Returns:
        The spread's value per share, clamped to `[0, long_strike - short_strike]` (skew can push the model
        marginally outside it).
    """
    value = put_price(spot, long_strike, model.vol(spot, long_strike, vix), years) - put_price(spot, short_strike, model.vol(spot, short_strike, vix), years)
    return min(max(value, 0.0), long_strike - short_strike)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Per-contract trading costs of a two-leg spread (Tastytrade's schedule).

    Attributes:
        half_spread: Half the bid/ask spread, dollars per share, paid on every leg at every open and close.
        commission: Commission per contract per leg, on opens only.
        regulatory: Regulatory fees per contract per leg, on opens and closes.
    """

    half_spread: float = 0.05
    commission: float = 1.00
    regulatory: float = 0.10

    def open_cost(self, contracts: int) -> float:
        """Cost of opening `contracts` spreads (two legs each), in dollars."""
        return contracts * 2 * (self.half_spread * 100 + self.commission + self.regulatory)

    def close_cost(self, contracts: int) -> float:
        """Cost of closing `contracts` spreads (two legs each), in dollars."""
        return contracts * 2 * (self.half_spread * 100 + self.regulatory)

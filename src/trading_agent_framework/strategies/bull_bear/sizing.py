"""Target weights from the judge's picks (pure): inverse volatility, bounded per stock, summing to what is investable.

cross_momentum's `inverse_volatility_weights` renormalises after capping, so with few names it breaks the cap; this
one finds the single scale at which the clipped weights sum to the total, so both bounds hold exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

_ITERATIONS = 200  # bisection steps: the scale is exact to float precision long before
_EPS = 1e-9


def capped_inverse_volatility(volatilities: Mapping[str, float], *, total: float, min_weight: float, max_weight: float) -> dict[str, float]:
    """Each weight is `clip(scale / volatility, min_weight, max_weight)`, with the one `scale` that makes them sum to `total`.

    The clipped sum only grows with `scale`, so bisection finds it; the clip is the last step, so both bounds hold
    exactly. Keys keep the input order (the rebalancer buys in that order). Raises `ValueError` for no stock, a
    volatility that is not finite and above zero, or bounds that cannot hold (`n x min_weight > total` or
    `n x max_weight < total`).
    """
    if not volatilities:
        raise ValueError("no stock to size")
    for symbol, volatility in volatilities.items():
        if not (math.isfinite(volatility) and volatility > 0):
            raise ValueError(f"the volatility of {symbol} must be finite and above zero, got {volatility}")
    count = len(volatilities)
    if count * min_weight > total + _EPS or count * max_weight < total - _EPS:
        raise ValueError(f"{count} stocks cannot sum to {total} with weights in [{min_weight}, {max_weight}]")
    raw = {symbol: 1.0 / volatility for symbol, volatility in volatilities.items()}

    def weights(scale: float) -> dict[str, float]:
        return {symbol: min(max(scale * value, min_weight), max_weight) for symbol, value in raw.items()}

    low, high = 0.0, max_weight / min(raw.values())  # at `high` every weight is at its cap
    for _ in range(_ITERATIONS):
        middle = (low + high) / 2
        if sum(weights(middle).values()) < total:
            low = middle
        else:
            high = middle
    return weights(high)

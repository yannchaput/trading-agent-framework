"""Target weights for the book (pure): every stock the judge put in gets the same weight.

Nothing else enters: no volatility, no strength, no conviction, no risk rating. On the same books over the 1Y window
(2026-10-29 study: the final-prompt judge's books in 2 replayed runs, and the 11 older runs' books), equal weights beat
both the judge's strength (14% strong / 9% standard; no edge, more turnover) and capped inverse volatility (behind in 10
of 11 runs). The weight is fixed, so a short book is not scaled up: `target_portfolio` parks the rest, as it always did.
"""

from __future__ import annotations

from collections.abc import Sequence


def equal_weights(symbols: Sequence[str], *, weight: float) -> dict[str, float]:
    """Each stock gets `weight`. Keys keep the input order (the rebalancer buys in that order)."""
    return {symbol: weight for symbol in symbols}

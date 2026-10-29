"""Target weights for the book (pure): the judge's strength alone sizes a stock.

A strong stock gets `strong_weight`, a standard one `standard_weight`; nothing else enters (no volatility, no conviction,
no risk rating). When the weights sum to more than `total` every weight is scaled by one factor, so the ratios between
strengths hold; below it they are used as they are and `target_portfolio` parks the rest, as it always did.
"""

from __future__ import annotations

from collections.abc import Mapping


def strength_weights(strengths: Mapping[str, str], *, weights: Mapping[str, float], total: float) -> dict[str, float]:
    """Each stock's weight is `weights[strength]`, all multiplied by `total / sum` when the sum exceeds `total`.

    Keys keep the input order (the rebalancer buys in that order). Raises `ValueError` naming a strength that has no weight
    (unknown, or `weak`: a weak stock is never in the book).
    """
    sized: dict[str, float] = {}
    for symbol, strength in strengths.items():
        if strength not in weights:
            raise ValueError(f"{symbol} has no weight for the strength {strength!r}: only {', '.join(weights)} can be in the book")
        sized[symbol] = weights[strength]
    invested = sum(sized.values())
    if invested > total:
        scale = total / invested
        sized = {symbol: weight * scale for symbol, weight in sized.items()}
    return sized

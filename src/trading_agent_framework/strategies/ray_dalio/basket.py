"""The `submit_basket` tool: how the disagreement agent hands its basket to the code that trades it.

The first backtest showed the trader agent ignoring the basket (it kept TMF in 50 of 103 sessions that left it out),
so the basket is now a checked tool call and code, not an agent, trades it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trading_agent_framework.strategies.ray_dalio.prompts import MIN_POSITIONS, UNIVERSE
from trading_agent_framework.utils.helpers import BUY_BUFFER

MIN_STOCK_WEIGHT = 0.03  # the rebalance band: the Rebalancer never buys a fund into an empty position for a target this small


def basket_problem(weights: dict[str, float]) -> str | None:
    """Why `weights` (symbol -> fraction of the portfolio) is refused, or None if it is fine.

    SHV is the parking remainder, so it counts as a position but not toward the invested total.
    """
    unknown = [symbol for symbol in weights if symbol not in UNIVERSE]
    if unknown:
        return f"not in the universe: {', '.join(unknown)}"
    if not all(weight > 0 for weight in weights.values()):  # written this way so NaN is refused too
        return "every weight must be above 0 (drop a symbol you do not want)"
    if len(weights) < MIN_POSITIONS:
        return f"a basket needs at least {MIN_POSITIONS} symbols (SHV counts as one)"
    too_small = [symbol for symbol, weight in weights.items() if symbol != "SHV" and weight <= MIN_STOCK_WEIGHT]
    if too_small:
        return f"{', '.join(too_small)}: a weight of {MIN_STOCK_WEIGHT:g} or less is never bought, raise it above {MIN_STOCK_WEIGHT:g} or drop the symbol"
    limit = 1 - float(BUY_BUFFER)
    invested = sum(weight for symbol, weight in weights.items() if symbol != "SHV")
    if invested > limit + 1e-9:
        return f"the weights other than SHV add up to {invested:g}, above {limit:g}: weights are fractions of the portfolio (0.35 for 35%)"
    return None


def basket_tool(store: dict[str, float]) -> Callable[..., dict[str, Any]]:
    """Build the `submit_basket` tool; an accepted basket replaces what is in `store`, a refused one leaves it alone."""

    def submit_basket(weights: dict[str, float], reason: str) -> dict[str, Any]:
        """Submit the basket to hold, as symbol -> fraction of the portfolio (0.35 means 35%), and why.

        It is executed exactly as submitted: holdings you leave out are sold, and whatever the weights leave
        unallocated is parked in SHV. If it is refused, read the error, fix the basket and call this again.
        """
        problem = basket_problem(weights)
        if problem is not None:
            return {"error": problem}
        store.clear()
        store.update(weights)
        return {"accepted": True, "basket": weights}

    return submit_basket

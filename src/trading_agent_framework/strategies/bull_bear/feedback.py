"""What the judge sees of its recent `out` verdicts: how each stock did since, against the stocks it put in (pure).

Prices are the closes stored when the judge decided and the last closes the review loaded now, so no price after
`clock.now()` is ever read: in a backtest both come from the gated bars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any


def _since(close_then: float, close_now: float | None) -> float | None:
    if close_now is None or close_then <= 0:
        return None
    return close_now / close_then - 1


def recent_outcomes(history: Sequence[Mapping[str, Any]], closes_now: Mapping[str, float]) -> list[dict[str, Any]]:
    """One entry per `out` verdict in `history`, newest review first: symbol, date, reason, `return_since_pct`, and
    `in_names_return_pct` (the mean return of that review's `in` stocks over the same span; None when none is priced).
    An `out` stock without a price now is left out, never estimated."""
    outcomes: list[dict[str, Any]] = []
    for review in reversed(history):
        decided = review["verdicts"]
        in_returns = [since for symbol, item in decided.items() if item["verdict"] == "in" and (since := _since(item["close"], closes_now.get(symbol))) is not None]
        in_mean = round(fmean(in_returns) * 100, 1) if in_returns else None
        for symbol, item in decided.items():
            since = _since(item["close"], closes_now.get(symbol)) if item["verdict"] == "out" else None
            if since is not None:
                outcomes.append({"symbol": symbol, "date": review["date"], "reason": item["reason"], "return_since_pct": round(since * 100, 1), "in_names_return_pct": in_mean})
    return outcomes

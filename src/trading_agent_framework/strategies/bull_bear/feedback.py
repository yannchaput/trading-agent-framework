"""What the judge sees of its recent `out` verdicts: how each stock did since, against the stocks it put in (pure).

Prices are the last closes the review loaded now and, for the day the judge decided, the close of the last session
before that day in the same loaded series (so a split since then moves both the same way); the close stored with
the verdict is the fallback. No price after `clock.now()` is ever read: in a backtest everything comes from the
gated bars.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from statistics import fmean
from typing import Any

REASON_CHARS = 150  # a stored reason is cut to this many characters in the block the judge sees


def _since(close_then: float, close_now: float | None) -> float | None:
    if close_now is None or close_then <= 0:
        return None
    return close_now / close_then - 1


def _cut(reason: str) -> str:
    return reason if len(reason) <= REASON_CHARS else reason[:REASON_CHARS] + "..."


def recent_outcomes(
    history: Sequence[Mapping[str, Any]],
    closes_now: Mapping[str, float],
    close_before: Callable[[str, str], float | None] | None = None,
) -> list[dict[str, Any]]:
    """One entry per `out` verdict in `history`, newest review first: symbol, date, reason (cut to 150 characters),
    `return_since_pct`, and `in_names_return_pct` (the mean return of that review's `in` stocks over the same span;
    None when none is priced). An `out` stock without a price now is left out, never estimated.

    `close_before(symbol, review_date)` is the close of the last session strictly before the review's date, read from
    today's loaded series (the same adjustment basis as `closes_now`), or None; the stored close is used then.
    """

    def then(symbol: str, review_date: str, stored: float) -> float:
        adjusted = close_before(symbol, review_date) if close_before is not None else None
        return adjusted if adjusted is not None else stored

    outcomes: list[dict[str, Any]] = []
    for review in reversed(history):
        decided = review["verdicts"]
        day = review["date"]
        in_returns = [since for symbol, item in decided.items() if item["verdict"] == "in" and (since := _since(then(symbol, day, item["close"]), closes_now.get(symbol))) is not None]
        in_mean = round(fmean(in_returns) * 100, 1) if in_returns else None
        for symbol, item in decided.items():
            since = _since(then(symbol, day, item["close"]), closes_now.get(symbol)) if item["verdict"] == "out" else None
            if since is not None:
                outcomes.append({"symbol": symbol, "date": day, "reason": _cut(item["reason"]), "return_since_pct": round(since * 100, 1), "in_names_return_pct": in_mean})
    return outcomes

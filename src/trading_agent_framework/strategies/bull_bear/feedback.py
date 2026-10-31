"""What the judge sees of its recent verdicts, in and out: how each stock did since, against both sides of that day (pure).

Showing only the `out` verdicts taught it one lesson: in the 2026-10-30 2Y run, 11 of its 14 lessons said an out was costly
and none spoke of an `in` that lost, although an out stock beat the `in` stocks only 39 to 46% of the time.

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


def _mean_pct(returns: Sequence[float]) -> float | None:
    return round(fmean(returns) * 100, 1) if returns else None


def _cut(reason: str) -> str:
    return reason if len(reason) <= REASON_CHARS else reason[:REASON_CHARS] + "..."


def recent_outcomes(
    history: Sequence[Mapping[str, Any]],
    closes_now: Mapping[str, float],
    close_before: Callable[[str, str], float | None] | None = None,
) -> list[dict[str, Any]]:
    """One entry per verdict in `history`, in or out, newest review first: symbol, date, verdict, reason (cut to 150
    characters), `return_since_pct`, and `in_names_return_pct` and `out_names_return_pct` (the mean return of that review's
    `in` and `out` stocks over the same span; None when none is priced). A stock without a price now is left out, never
    estimated.

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
        returns = {symbol: since for symbol, item in decided.items() if (since := _since(then(symbol, day, item["close"]), closes_now.get(symbol))) is not None}
        means = {side: _mean_pct([since for symbol, since in returns.items() if decided[symbol]["verdict"] == side]) for side in ("in", "out")}
        for symbol, since in returns.items():
            item = decided[symbol]
            outcomes.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "verdict": item["verdict"],
                    "reason": _cut(item["reason"]),
                    "return_since_pct": round(since * 100, 1),
                    "in_names_return_pct": means["in"],
                    "out_names_return_pct": means["out"],
                }
            )
    return outcomes

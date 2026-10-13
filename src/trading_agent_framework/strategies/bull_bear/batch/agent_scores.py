"""Score the bull, the bear and the judge of a bull_bear run against forward returns (pure: no I/O, no network).

A review's candidates are its `debate_set`, the stocks every agent was shown. Each candidate's forward return runs
from the open of the first session after the review date (the fill: a Tuesday 12:00 review's market orders fill at
the next bar's open, `backtesting/fills.py`) to the open `h` sessions later; its excess return subtracts the mean raw
return of that review's candidates at the same horizon, so it measures discrimination, not the market.
Spec: docs/superpowers/specs/2026-10-13-bull-bear-agent-scores-design.md.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

HORIZONS = (5, 20)  # sessions: about one and four weeks


@dataclass(frozen=True, slots=True)
class ForwardReturn:
    review: date
    symbol: str
    horizon: int  # sessions
    raw: float
    excess: float  # raw minus the review's candidate mean at this horizon


@dataclass(frozen=True, slots=True)
class Returns:
    rows: list[ForwardReturn]
    dropped: int  # candidate x horizon pairs with no entry or no exit open


def scored_reviews(reviews: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The reviews whose calls count: not abandoned (an execution-stage abandonment keeps its verdicts, but its book
    never fully traded) and with a debate set (another strategy's records have none)."""
    return [review for review in reviews if not review.get("abandoned") and review.get("debate_set")]


def candidates(review: Mapping[str, Any]) -> list[str]:
    """The review's debate set, in rank order, each symbol once."""
    return list(dict.fromkeys(entry["symbol"] for entry in review["debate_set"]))


def forward_returns(
    reviews: Sequence[Mapping[str, Any]],
    opens: Mapping[str, Mapping[date, float]],
    calendar: Sequence[date],
    horizons: Sequence[int] = HORIZONS,
) -> Returns:
    """Each scored review's candidates, at each horizon: entry at the open of the first `calendar` session strictly
    after the review date, exit at the open `horizon` sessions later. A candidate with no open on either day, or a
    review whose exit is past the calendar, is dropped and counted."""
    sessions = sorted(calendar)
    rows: list[ForwardReturn] = []
    dropped = 0
    for review in scored_reviews(reviews):
        day = date.fromisoformat(review["date"])
        entry_index = bisect_right(sessions, day)  # the first session strictly after the review date
        symbols = candidates(review)
        for horizon in horizons:
            exit_index = entry_index + horizon
            if exit_index >= len(sessions):
                dropped += len(symbols)
                continue
            entry_day, exit_day = sessions[entry_index], sessions[exit_index]
            raw: dict[str, float] = {}
            for symbol in symbols:
                series = opens.get(symbol, {})
                entry, exit_ = series.get(entry_day), series.get(exit_day)
                if entry is None or exit_ is None or entry <= 0:
                    dropped += 1
                    continue
                raw[symbol] = exit_ / entry - 1
            if raw:
                mean = sum(raw.values()) / len(raw)
                rows.extend(ForwardReturn(day, symbol, horizon, value, value - mean) for symbol, value in raw.items())
    return Returns(rows, dropped)

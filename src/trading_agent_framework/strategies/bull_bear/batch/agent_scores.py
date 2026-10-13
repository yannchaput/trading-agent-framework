"""Score the bull, the bear and the judge of a bull_bear run against forward returns (pure: no I/O, no network).

A review's candidates are its `debate_set`, the stocks every agent was shown. Each candidate's forward return runs
from the open of the first session after the review date (the fill: a Tuesday 12:00 review's market orders fill at
the next bar's open, `backtesting/fills.py`) to the open `h` sessions later; its excess return subtracts the mean raw
return of that review's candidates at the same horizon, so it measures discrimination, not the market.
Spec: docs/superpowers/specs/2026-10-13-bull-bear-agent-scores-design.md.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

HORIZONS = (5, 20)  # sessions: about one and four weeks
LEVELS = ("low", "medium", "high")
VERDICTS = ("in", "out")
_SESSIONS_PER_REVIEW = 5  # one review a week


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


@dataclass(frozen=True, slots=True)
class GroupStats:
    n: int
    mean_excess: float | None  # None when n == 0
    hit_rate: float | None  # share with a positive excess return


EMPTY_GROUP = GroupStats(0, None, None)


@dataclass(frozen=True, slots=True)
class Spread:
    mean: float | None  # mean across reviews of (better group's mean excess - worse group's); None without a review
    t: float | None  # across reviews; None under 2 reviews or with zero variance
    reviews: int  # reviews with both groups
    skipped: int  # scored reviews missing a group


@dataclass(frozen=True, slots=True)
class HorizonScore:
    groups: dict[str, GroupStats]  # in display order
    spread: Spread
    concerns: dict[str, GroupStats] = field(default_factory=dict)  # the bear's, by concern; empty for the others


@dataclass(frozen=True, slots=True)
class AgentScore:
    agent: str  # "bull", "bear" or "judge"
    spread_label: str  # e.g. "high - low"
    by_horizon: dict[int, HorizonScore]


@dataclass(frozen=True, slots=True)
class RunScores:
    agents: list[AgentScore]  # bull, bear, judge
    dropped: int  # from `Returns.dropped`


def bull_calls(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: case["conviction"] for case in review.get("bull") or [] if case.get("conviction") in LEVELS}


def bear_calls(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: case["risk"] for case in review.get("bear") or [] if case.get("risk") in LEVELS}


def bear_concerns(review: Mapping[str, Any]) -> dict[str, str]:
    return {case["symbol"]: str(case.get("concern") or "none") for case in review.get("bear") or []}


def judge_calls(review: Mapping[str, Any]) -> dict[str, str]:
    """The in/out judge's `verdicts`; a picker record's `picks` are in and the rest of its debate set out."""
    if "verdicts" in review:
        return {verdict["symbol"]: verdict["verdict"] for verdict in review["verdicts"] or [] if verdict.get("verdict") in VERDICTS}
    if "picks" in review:
        picked = {pick["symbol"] for pick in review["picks"] or []}
        return {symbol: "in" if symbol in picked else "out" for symbol in candidates(review)}
    return {}


# (agent, its groups in display order, (the group that should do better, the one that should do worse), its calls)
_AGENTS: tuple[tuple[str, tuple[str, ...], tuple[str, str], Callable[[Mapping[str, Any]], dict[str, str]]], ...] = (
    ("bull", LEVELS, ("high", "low"), bull_calls),
    ("bear", LEVELS, ("low", "high"), bear_calls),
    ("judge", VERDICTS, ("in", "out"), judge_calls),
)


def score_agents(reviews: Sequence[Mapping[str, Any]], returns: Returns, horizons: Sequence[int] = HORIZONS) -> RunScores:
    """Each agent's calls against the excess returns: pooled group stats, and a spread computed per review (a review
    missing either group is skipped and counted), averaged across reviews, with a t-stat across reviews."""
    excess = {(row.review, row.symbol, row.horizon): row.excess for row in returns.rows}
    counted = scored_reviews(reviews)
    agents: list[AgentScore] = []
    for name, groups, (better, worse), calls_of in _AGENTS:
        by_horizon: dict[int, HorizonScore] = {}
        for horizon in horizons:
            pooled: dict[str, list[float]] = {group: [] for group in groups}
            concerns: dict[str, list[float]] = {}
            spreads: list[float] = []
            skipped = 0
            for review in counted:
                day = date.fromisoformat(review["date"])
                in_review: dict[str, list[float]] = {group: [] for group in groups}
                for symbol, group in calls_of(review).items():
                    value = excess.get((day, symbol, horizon))
                    if value is not None:
                        in_review[group].append(value)
                        pooled[group].append(value)
                if in_review[better] and in_review[worse]:
                    spreads.append(_mean(in_review[better]) - _mean(in_review[worse]))
                else:
                    skipped += 1
                if name == "bear":
                    for symbol, concern in bear_concerns(review).items():
                        value = excess.get((day, symbol, horizon))
                        if value is not None:
                            concerns.setdefault(concern, []).append(value)
            spread = Spread(_mean(spreads) if spreads else None, newey_west_t(spreads, lags=_overlap_lags(horizon)), len(spreads), skipped)
            by_horizon[horizon] = HorizonScore(
                groups={group: _stats(values) for group, values in pooled.items()},
                spread=spread,
                concerns={concern: _stats(values) for concern, values in sorted(concerns.items())},
            )
        agents.append(AgentScore(name, f"{better} - {worse}", by_horizon))
    return RunScores(agents, returns.dropped)


def newey_west_t(values: Sequence[float], *, lags: int) -> float | None:
    """t-stat of the mean of `values`, with a Newey-West (Bartlett) long-run variance over `lags` lags scaled by
    n / (n - 1): at 0 lags this is the textbook t, the mean over (sample standard deviation / root n). None under two
    values or with no variance."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    demeaned = [value - mean for value in values]
    variance = sum(x * x for x in demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * sum(demeaned[i] * demeaned[i - lag] for i in range(lag, n)) / n
    variance *= n / (n - 1)
    if variance <= 0:
        return None
    return mean / math.sqrt(variance / n)


def _overlap_lags(horizon: int) -> int:
    """The later reviews a horizon's window overlaps: 0 at 5 sessions, 3 at 20."""
    return max(0, math.ceil(horizon / _SESSIONS_PER_REVIEW) - 1)


def _stats(values: Sequence[float]) -> GroupStats:
    if not values:
        return EMPTY_GROUP
    return GroupStats(len(values), _mean(values), sum(1 for value in values if value > 0) / len(values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)

"""The hand-off between the three agents: submit tools, their validation and the recorder.

Each agent ends its run by calling one submit tool. A tool validates its argument against what the pipeline
armed the recorder with for this stage, records the first valid submission, and otherwise returns
`{"error": ...}` so the model can correct itself in the same run. Free text from an agent is never trusted.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's
schema from the function's real annotations (as `memory/tools.py` does).
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.common.handoff import (
    HandoffError,
    SubmitRecorder,
    items,
    key,
    optional_text,
    required_text,
)
from trading_agent_framework.strategies.common.handoff import weight as shared_weight

SURVIVE = "survive"
FAIL = "fail"
VERDICTS = (SURVIVE, FAIL)
CONCERNS = ("debt", "margin", "competition", "management", "accounting", "valuation")
_EPS = 1e-9

RANKING, VERDICTS_STAGE, PORTFOLIO = "ranking", "verdicts", "portfolio"


@dataclass(frozen=True, slots=True)
class Idea:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str
    reason: str
    concern: str | None = None  # one of CONCERNS for a short seller's fail; None for a survive or a code verdict
    what_changed: str | None = None  # required when the verdict differs from the previous review's


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    symbol: str
    weight: float
    reason: str


# --- validation (pure) --------------------------------------------------------------------------


def _concern(item: Mapping[str, Any], symbol: str, verdict: str) -> str | None:
    if verdict != FAIL:
        return None
    value = item.get("concern")
    concern = value.strip().lower() if isinstance(value, str) else None
    if concern not in CONCERNS:
        raise HandoffError(f"the fail for {symbol} needs a concern: one of {', '.join(CONCERNS)}")
    return concern


def validate_ranking(raw: Any, *, candidates: Sequence[str], top_n: int, reason_max_chars: int) -> list[Idea]:
    """1 to `top_n` unique ideas (or all candidates if fewer), each from `candidates`, best first."""
    entries = items(raw, "ideas")
    limit = min(top_n, len(candidates))
    if not entries:
        raise HandoffError("ideas is empty: submit at least one idea chosen from the candidates")
    if len(entries) > limit:
        raise HandoffError(f"submit at most {limit} ideas, got {len(entries)}")
    allowed = set(candidates)
    seen: set[str] = set()
    ideas = []
    for index, item in enumerate(entries, start=1):
        symbol = key(item, "symbol", index)
        if symbol not in allowed:
            raise HandoffError(f"{symbol} is not one of the candidates ({', '.join(candidates)})")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        ideas.append(Idea(symbol, required_text(item, "reason", symbol, reason_max_chars)))
    return ideas


def validate_verdicts(raw: Any, *, expected: Collection[str], reason_max_chars: int, previous: Mapping[str, str] | None = None) -> list[Verdict]:
    """Exactly one verdict (`survive` or `fail`) per symbol in `expected`, and no other symbol.

    A `fail` names its `concern` (one of `CONCERNS`); a verdict that differs from `previous` (symbol -> the last
    review's verdict) says `what_changed`.
    """
    entries = items(raw, "verdicts")
    wanted = set(expected)
    before = previous or {}
    seen: set[str] = set()
    verdicts = []
    for index, item in enumerate(entries, start=1):
        symbol = key(item, "symbol", index)
        if symbol not in wanted:
            raise HandoffError(f"{symbol} was not asked about: judge only {', '.join(sorted(wanted))}")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        verdict = item.get("verdict")
        verdict = verdict.strip().lower() if isinstance(verdict, str) else verdict
        if verdict not in VERDICTS:
            raise HandoffError(f"the verdict for {symbol} must be 'survive' or 'fail'")
        reason = required_text(item, "reason", symbol, reason_max_chars)
        concern = _concern(item, symbol, verdict)
        what_changed = optional_text(item, "what_changed", symbol, reason_max_chars)
        if before.get(symbol, verdict) != verdict and what_changed is None:
            raise HandoffError(f"{symbol} was '{before[symbol]}' at the previous review: say in what_changed what changed since then")
        verdicts.append(Verdict(symbol, verdict, reason, concern, what_changed))
    missing = sorted(wanted - seen)
    if missing:
        raise HandoffError(f"no verdict for {', '.join(missing)}: give one verdict per symbol")
    return verdicts


def validate_portfolio(
    raw: Any,
    *,
    allowed: Collection[str],
    max_positions: int,
    min_weight: float,
    max_weight: float,
    max_total_weight: float,
    reason_max_chars: int,
    required: Collection[str] = (),
) -> list[PortfolioPosition]:
    """At most `max_positions` unique stocks from `allowed`, each weight in [min, max], their sum at most `max_total_weight`.

    An empty list is valid: it means everything goes to the parking instrument. Every symbol in `required` (a holding that
    has not failed twice) must be held.
    """
    entries = items(raw, "positions")
    if len(entries) > max_positions:
        raise HandoffError(f"submit at most {max_positions} positions, got {len(entries)}")
    allowed_set = set(allowed)
    seen: set[str] = set()
    positions = []
    for index, item in enumerate(entries, start=1):
        symbol = key(item, "symbol", index)
        if symbol not in allowed_set:
            raise HandoffError(f"{symbol} is not allowed (choose from {', '.join(sorted(allowed_set)) or 'nothing: submit an empty list'})")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        weight = shared_weight(item, symbol, example="0.25")
        if not min_weight - _EPS <= weight <= max_weight + _EPS:
            raise HandoffError(f"the weight for {symbol} is {weight:.4f}: it must be between {min_weight} and {max_weight}")
        positions.append(PortfolioPosition(symbol, weight, required_text(item, "reason", symbol, reason_max_chars)))
    held = {position.symbol for position in positions}
    missing = [symbol for symbol in required if symbol not in held]
    if len(missing) == 1:
        raise HandoffError(f"{missing[0]} is held and has not failed twice: keep it with a weight of at least {min_weight}")
    if missing:
        raise HandoffError(f"{', '.join(sorted(missing))} are held and have not failed twice: keep each with a weight of at least {min_weight}")
    total = sum(position.weight for position in positions)
    if total > max_total_weight + _EPS:
        raise HandoffError(f"the weights sum to {total:.4f}: they must sum to at most {max_total_weight:.2f}")
    return positions


# --- the recorder ---------------------------------------------------------------------------------


class HandoffRecorder(SubmitRecorder):
    """Holds what the current stage must validate against, and the first valid submission for it."""

    TOOL_NAMES = {RANKING: "submit_ranking", VERDICTS_STAGE: "submit_verdicts", PORTFOLIO: "submit_portfolio"}
    NOT_EXPECTED_SUFFIX = " of the review"
    ALREADY_RECORDED = "already recorded for this review"

    def __init__(self, params: AckmanParams) -> None:
        super().__init__()
        self._params = params

    def expect_ranking(self, candidates: Sequence[str]) -> None:
        self._arm(RANKING, {"candidates": list(candidates)})

    def expect_verdicts(self, symbols: Sequence[str], previous: Mapping[str, str] | None = None) -> None:
        self._arm(VERDICTS_STAGE, {"expected": list(symbols), "previous": dict(previous or {})})

    def expect_portfolio(self, allowed: Sequence[str], required: Sequence[str] = ()) -> None:
        self._arm(PORTFOLIO, {"allowed": list(allowed), "required": list(required)})

    def _validate(self, stage: str, *args: Any) -> Any:
        raw, params, context = args[0], self._params, self._context
        if stage == RANKING:
            return validate_ranking(raw, candidates=context["candidates"], top_n=params.research_top_n, reason_max_chars=params.reason_max_chars)
        if stage == VERDICTS_STAGE:
            return validate_verdicts(raw, expected=context["expected"], reason_max_chars=params.reason_max_chars, previous=context["previous"])
        return validate_portfolio(
            raw,
            allowed=context["allowed"],
            max_positions=params.max_positions,
            min_weight=params.min_weight,
            max_weight=params.max_weight,
            max_total_weight=params.max_total_weight,
            reason_max_chars=params.reason_max_chars,
            required=context["required"],
        )


# --- the tools ------------------------------------------------------------------------------------


def submit_tools(recorder: HandoffRecorder) -> dict[str, Callable[..., dict[str, Any]]]:
    """The three submit tools, closures over `recorder`, keyed by name. Each agent is given only its own."""

    def submit_ranking(ideas: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit your ranked ideas, best first: a list of objects with symbol and reason."""
        return recorder.submit(RANKING, ideas)

    def submit_verdicts(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit one verdict per symbol you were asked about: objects with symbol, verdict (survive or fail), reason, concern (for a fail) and what_changed (when the verdict changed)."""
        return recorder.submit(VERDICTS_STAGE, verdicts)

    def submit_portfolio(positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit the portfolio to hold: objects with symbol, weight (fraction of portfolio value) and reason; an empty list holds nothing."""
        return recorder.submit(PORTFOLIO, positions)

    return {"submit_ranking": submit_ranking, "submit_verdicts": submit_verdicts, "submit_portfolio": submit_portfolio}

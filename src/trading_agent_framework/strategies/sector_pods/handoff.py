"""Submit schemas, validators and submit tools of the sector_pods agents.

No `from __future__ import annotations` here: LangChain reads the tool signatures and the TypedDicts at runtime.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from trading_agent_framework.agents.graph import SubmitSlot
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, Pod, pod_of

POD_SUBMIT = "submit_pod_view"
RISK_SUBMIT = "submit_risk_review"
PM_SUBMIT = "submit_target_weights"

OBJECTION_KINDS = ("crowding", "macro", "drawdown", "reversal", "other")
SEVERITIES = ("low", "high")
SCORE_RANGE = (-2, 2)
UNAVAILABLE: dict[str, Any] = {"unavailable": True}  # what `AgentNode.write(None)` stores for a pod

_TOLERANCE = 1e-9

Validator = Callable[[Any, Mapping[str, Any]], str | None]


class Score(TypedDict):
    """One ETF's score from a pod: -2 (strong avoid) to 2 (strong conviction)."""

    symbol: str
    score: int
    reason: str


class Objection(TypedDict):
    """One risk objection against a group of ETFs."""

    symbols: list[str]
    kind: str
    severity: str
    reason: str


class Weight(TypedDict):
    """One target weight of the portfolio, as a share of its value."""

    symbol: str
    weight: float
    reason: str


def _text(value: Any, label: str, limit: int, *, required: bool = True) -> str | None:
    """Error unless `value` is a string of at most `limit` characters (and non-blank when `required`)."""
    if not isinstance(value, str) or (required and not value.strip()):
        return f"{label} must be a non-empty string" if required else f"{label} must be a string"
    if len(value) > limit:
        return f"{label} is {len(value)} characters; the limit is {limit}"
    return None


def _entries(payload: Any, key: str) -> tuple[list[Any] | None, str | None]:
    """The list under `key` of a dict payload, or an error."""
    if not isinstance(payload, dict):
        return None, "the payload must be an object"
    items = payload.get(key)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return None, f"{key} must be a list of objects"
    return items, None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_pod(pod: Pod, params: SectorPodsParams) -> Validator:
    """Validator for `submit_pod_view`: one score per ETF the pod has facts for, plus a sector view."""

    def check(payload: Any, state: Mapping[str, Any]) -> str | None:
        scores, error = _entries(payload, "scores")
        if scores is None:
            return error
        symbols = [score.get("symbol") for score in scores]
        expected = set(state.get("facts", {}).get(pod.key, {}))
        duplicates = sorted({str(s) for s in symbols if symbols.count(s) > 1})
        missing = sorted(expected - set(symbols))
        unknown = sorted(str(s) for s in set(symbols) - expected)
        if duplicates or missing or unknown:
            parts = [f"{label}: {', '.join(names)}" for label, names in (("duplicate", duplicates), ("missing", missing), ("unknown", unknown)) if names]
            return f"scores must cover exactly {', '.join(sorted(expected))} once each ({'; '.join(parts)})"
        for score in scores:
            value = score.get("score")
            if isinstance(value, bool) or not isinstance(value, int) or not SCORE_RANGE[0] <= value <= SCORE_RANGE[1]:
                return f"{score['symbol']}: score must be an integer from {SCORE_RANGE[0]} to {SCORE_RANGE[1]}"
            if (message := _text(score.get("reason"), f"{score['symbol']}: reason", params.reason_max_chars)) is not None:
                return message
        return _text(payload.get("sector_view"), "sector_view", params.summary_max_chars)

    return check


def validate_risk(params: SectorPodsParams) -> Validator:
    """Validator for `submit_risk_review`: any number of objections (possibly none) plus a summary."""

    def check(payload: Any, state: Mapping[str, Any]) -> str | None:
        objections, error = _entries(payload, "objections")
        if objections is None:
            return error
        for objection in objections:
            symbols = objection.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                return "each objection needs a non-empty list of symbols"
            if unknown := sorted(str(s) for s in symbols if s not in ALL_SYMBOLS):
                return f"unknown symbols in an objection: {', '.join(unknown)}"
            if objection.get("kind") not in OBJECTION_KINDS:
                return f"kind must be one of {', '.join(OBJECTION_KINDS)}"
            if objection.get("severity") not in SEVERITIES:
                return f"severity must be one of {', '.join(SEVERITIES)}"
            if (message := _text(objection.get("reason"), "reason", params.reason_max_chars)) is not None:
                return message
        return _text(payload.get("summary"), "summary", params.summary_max_chars)

    return check


def _frozen(symbol: str, state: Mapping[str, Any]) -> bool:
    """Whether `symbol` was not researched this week: its pod gave no view, or it is in no pod's facts."""
    view = state.get("pods", {}).get(pod_of(symbol).key)
    if not view or view.get("unavailable"):
        return True
    return not any(symbol in rows for rows in state.get("facts", {}).values())


def unresearched(state: Mapping[str, Any]) -> dict[str, float]:
    """Held ETFs nobody researched this week, with their current weight (SHV and unknown symbols excluded).

    These can only be kept at up to their current weight or cut, the same rule `check_weights` applies.
    """
    return {s: w for s, w in state.get("holdings", {}).items() if s in ALL_SYMBOLS and _frozen(s, state)}


def check_weights(weights: Mapping[str, float], state: Mapping[str, Any], params: SectorPodsParams) -> str | None:
    """Check a whole target book against the order limits; the PM validator and the execute node both call it.

    Args:
        weights: Target weight per ETF (shares of the portfolio value).
        state: The graph state; reads `facts`, `holdings` and `pods`.
        params: Supplies the ETF, pod and cash-buffer caps.

    Returns:
        An error message naming the first broken limit, or `None` when the book is acceptable.
    """
    holdings = state.get("holdings", {})
    pod_totals: dict[str, float] = {}
    for symbol, weight in weights.items():
        if symbol not in ALL_SYMBOLS:
            return f"{symbol}: not one of the sector ETFs"
        if not 0 <= weight <= params.max_etf_weight + _TOLERANCE:
            return f"{symbol}: weight {weight:.4f} must be between 0 and {params.max_etf_weight}"
        current = holdings.get(symbol, 0.0)
        if _frozen(symbol, state) and weight > current + _TOLERANCE:
            return f"{symbol}: its pod gave no view this week, so it can only be kept at up to its current weight ({current:.4f}) or cut"
        key = pod_of(symbol).key
        pod_totals[key] = pod_totals.get(key, 0.0) + weight
    for key, total in pod_totals.items():
        if total > params.max_pod_weight + _TOLERANCE:
            return f"{key}: weights sum to {total:.4f}; the pod limit is {params.max_pod_weight}"
    limit = 1 - params.cash_buffer
    if (total := sum(weights.values())) > limit + _TOLERANCE:
        return f"weights sum to {total:.4f}; the limit is {limit:.4f} (cash buffer {params.cash_buffer})"
    return None


def validate_pm(params: SectorPodsParams) -> Validator:
    """Validator for `submit_target_weights`: per-entry checks, then `check_weights` on the whole book."""

    def check(payload: Any, state: Mapping[str, Any]) -> str | None:
        entries, error = _entries(payload, "weights")
        if entries is None:
            return error
        weights: dict[str, float] = {}
        for entry in entries:
            symbol, weight = entry.get("symbol"), entry.get("weight")
            if not isinstance(symbol, str) or not _is_number(weight):
                return "each weight needs a symbol string and a numeric weight"
            if symbol in weights:
                return f"{symbol}: listed more than once"
            if (message := _text(entry.get("reason"), f"{symbol}: reason", params.reason_max_chars)) is not None:
                return message
            weights[symbol] = float(weight)
        for label in ("overrides", "summary"):
            if (message := _text(payload.get(label), label, params.summary_max_chars, required=label == "summary")) is not None:
                return message
        return check_weights(weights, state, params)

    return check


def pod_submit_tool(slot: SubmitSlot) -> Callable[..., dict[str, Any]]:
    """Build the pod agents' submit tool, recording into `slot`."""

    def submit_pod_view(scores: list[Score], sector_view: str) -> dict[str, Any]:
        """Submit your score for every ETF and your view of the sector."""
        return slot.submit({"scores": scores, "sector_view": sector_view})

    return submit_pod_view


def risk_submit_tool(slot: SubmitSlot) -> Callable[..., dict[str, Any]]:
    """Build the risk agent's submit tool, recording into `slot`."""

    def submit_risk_review(objections: list[Objection], summary: str) -> dict[str, Any]:
        """Submit your objections to the pods' views (none is fine) and a summary."""
        return slot.submit({"objections": objections, "summary": summary})

    return submit_risk_review


def pm_submit_tool(slot: SubmitSlot) -> Callable[..., dict[str, Any]]:
    """Build the portfolio manager's submit tool, recording into `slot`."""

    def submit_target_weights(weights: list[Weight], overrides: str, summary: str) -> dict[str, Any]:
        """Submit the target weight of each ETF you want to hold, what you overrode, and a summary."""
        return slot.submit({"weights": weights, "overrides": overrides, "summary": summary})

    return submit_target_weights

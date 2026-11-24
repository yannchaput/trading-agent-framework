"""Submit schemas, validators and submit tools of the sector_pods agents.

No `from __future__ import annotations` here: LangChain reads the tool signatures and the TypedDicts at runtime.
"""

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from trading_agent_framework.agents.submit import HandoffError, NodeRecorder
from trading_agent_framework.agents.tools.text import LengthGate, review_texts
from trading_agent_framework.strategies.common.handoff import optional_text, required_text
from trading_agent_framework.strategies.sector_pods.params import SectorPodsParams
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS, PODS, Pod, pod_of

POD_SUBMIT = "submit_pod_view"
RISK_SUBMIT = "submit_risk_review"
PM_SUBMIT = "submit_target_weights"

OBJECTION_KINDS = ("crowding", "macro", "drawdown", "reversal", "stress", "other")
SEVERITIES = ("low", "high")
SCORE_RANGE = (-2, 2)
UNAVAILABLE: dict[str, Any] = {"unavailable": True}  # what `AgentNode.write(None)` stores for a pod

_TOLERANCE = 1e-9

Validator = Callable[[Any, Mapping[str, Any]], None]


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


def _entries(payload: Any, key: str) -> list[Any]:
    """The list under `key` of a dict payload; raise `HandoffError` when it is not a list of objects."""
    if not isinstance(payload, dict):
        raise HandoffError("the payload must be an object")
    items = payload.get(key)
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise HandoffError(f"{key} must be a list of objects")
    return items


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def validate_pod(pod: Pod, params: SectorPodsParams) -> Validator:
    """Validator for `submit_pod_view`: one score per ETF the pod has facts for, plus a sector view."""

    def check(payload: Any, state: Mapping[str, Any]) -> None:
        scores = _entries(payload, "scores")
        symbols = [score.get("symbol") for score in scores]
        expected = set(state.get("facts", {}).get(pod.key, {}))
        duplicates = sorted({str(s) for s in symbols if symbols.count(s) > 1})
        missing = sorted(expected - set(symbols))
        unknown = sorted(str(s) for s in set(symbols) - expected)
        if duplicates or missing or unknown:
            parts = [f"{label}: {', '.join(names)}" for label, names in (("duplicate", duplicates), ("missing", missing), ("unknown", unknown)) if names]
            raise HandoffError(f"scores must cover exactly {', '.join(sorted(expected))} once each ({'; '.join(parts)})")
        for score in scores:
            value = score.get("score")
            if isinstance(value, bool) or not isinstance(value, int) or not SCORE_RANGE[0] <= value <= SCORE_RANGE[1]:
                raise HandoffError(f"{score['symbol']}: score must be an integer from {SCORE_RANGE[0]} to {SCORE_RANGE[1]}")
            required_text(score, "reason", str(score["symbol"]), params.reason_hard_max_chars)
        required_text(payload, "sector_view", "your submission", params.summary_hard_max_chars)

    return check


def validate_risk(params: SectorPodsParams) -> Validator:
    """Validator for `submit_risk_review`: any number of objections (possibly none) plus a summary."""

    def check(payload: Any, state: Mapping[str, Any]) -> None:
        objections = _entries(payload, "objections")
        for objection in objections:
            symbols = objection.get("symbols")
            if not isinstance(symbols, list) or not symbols:
                raise HandoffError("each objection needs a non-empty list of symbols")
            if unknown := sorted(str(s) for s in symbols if s not in ALL_SYMBOLS):
                raise HandoffError(f"unknown symbols in an objection: {', '.join(unknown)}")
            if objection.get("kind") not in OBJECTION_KINDS:
                raise HandoffError(f"kind must be one of {', '.join(OBJECTION_KINDS)}")
            if objection.get("severity") not in SEVERITIES:
                raise HandoffError(f"severity must be one of {', '.join(SEVERITIES)}")
            required_text(objection, "reason", f"the objection on {', '.join(symbols)}", params.reason_hard_max_chars)
        required_text(payload, "summary", "your submission", params.summary_hard_max_chars)

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


def fill_book(weights: Mapping[str, float], state: Mapping[str, Any], params: SectorPodsParams) -> dict[str, float]:
    """The book to trade: the PM's weights scaled up to `1 - params.cash_buffer`, within the ETF and pod caps.

    The PM's weights set the mix. The researched ETFs are scaled in proportion; an ETF over `max_etf_weight` is clipped and a
    pod over `max_pod_weight` is cut back, and what a cap cuts is spread in proportion over the ETFs still below their caps.
    Unresearched ETFs keep their weight (they count toward their pod's cap) and zero weights stay out. When every researched
    ETF is at a cap the rest is left unallocated (parked in the parking symbol by the rebalancer).

    Args:
        weights: The PM's target weight per ETF, already accepted by `check_weights`.
        state: The graph state; reads `facts`, `holdings` and `pods`.
        params: Supplies the ETF and pod caps and the cash buffer.

    Returns:
        Target weight per ETF, in the order of `weights`; it passes `check_weights` and never totals above the target.
    """
    target = 1 - params.cash_buffer
    book = {symbol: weight for symbol, weight in weights.items() if weight > 0}
    free = [symbol for symbol in book if not _frozen(symbol, state)]

    def pod_totals() -> dict[str, float]:
        totals: dict[str, float] = {}
        for symbol, weight in book.items():
            totals[pod_of(symbol).key] = totals.get(pod_of(symbol).key, 0.0) + weight
        return totals

    def scale(symbols: list[str], factor: float) -> None:
        for symbol in symbols:
            book[symbol] *= factor

    for _ in range(len(free) + len(PODS) + 1):  # each round saturates an ETF or a pod, or ends the loop
        totals = pod_totals()
        room = target - sum(book.values())
        open_ = [s for s in free if book[s] < params.max_etf_weight - _TOLERANCE and totals[pod_of(s).key] < params.max_pod_weight - _TOLERANCE]
        if room <= _TOLERANCE or not open_:
            break
        scale(open_, 1 + room / sum(book[s] for s in open_))
        for symbol in open_:
            book[symbol] = min(book[symbol], params.max_etf_weight)
        for key, total in pod_totals().items():
            if total > params.max_pod_weight:
                members = [s for s in free if pod_of(s).key == key]
                held = sum(book[s] for s in members)
                scale(members, max(held - (total - params.max_pod_weight), 0.0) / held)
    if (excess := sum(book.values()) - target) > 0 and free:  # float drift: never above the target
        held = sum(book[s] for s in free)
        scale(free, max(held - excess, 0.0) / held)
    return book


def validate_pm(params: SectorPodsParams) -> Validator:
    """Validator for `submit_target_weights`: per-entry checks, then `check_weights` on the whole book."""

    def check(payload: Any, state: Mapping[str, Any]) -> None:
        entries = _entries(payload, "weights")
        weights: dict[str, float] = {}
        for entry in entries:
            symbol, weight = entry.get("symbol"), entry.get("weight")
            if not isinstance(symbol, str) or not _is_number(weight):
                raise HandoffError("each weight needs a symbol string and a numeric weight")
            if symbol in weights:
                raise HandoffError(f"{symbol}: listed more than once")
            required_text(entry, "reason", symbol, params.reason_hard_max_chars)
            weights[symbol] = float(weight)
        optional_text(payload, "overrides", "your submission", params.summary_hard_max_chars)
        required_text(payload, "summary", "your submission", params.summary_hard_max_chars)
        if (message := check_weights(weights, state, params)) is not None:
            raise HandoffError(message)

    return check


def _subject(entry: Mapping[str, Any]) -> str:
    """What an entry's reason is about, for the trim warning: its `symbol`, or its `symbols` joined."""
    symbols = entry.get("symbols")
    return ", ".join(map(str, symbols)) if isinstance(symbols, list) else str(entry.get("symbol"))


def _submit(recorder: NodeRecorder, params: SectorPodsParams, key: str, entries: Any, **long_texts: Any) -> dict[str, Any]:
    """Record `{key: entries, **long_texts}` once the recorder's length gate reviewed each entry's `reason` and the `long_texts`.

    The gate's refusal is returned instead. On its last attempt the texts are stripped and cut at the hard cap. Without a
    gate, or for a malformed list, the texts reach the validator untouched, for it to refuse.
    """

    def review(gate: LengthGate) -> dict[str, str] | tuple[Any, ...]:
        out = review_texts(gate, entries, {"reason": params.reason_limit}, _subject, {label: (value, params.summary_limit) for label, value in long_texts.items()})
        return out if isinstance(out, dict) else ({key: out[0], **out[1]},)

    return recorder.submit_texts(recorder.tool, review, {key: entries, **long_texts})


def pod_submit_tool(recorder: NodeRecorder, params: SectorPodsParams) -> Callable[..., dict[str, Any]]:
    """Build the pod agents' submit tool, recording into `recorder`; its texts go through the recorder's length gate first."""

    def submit_pod_view(scores: list[Score], sector_view: str) -> dict[str, Any]:
        """Submit your score for every ETF and your view of the sector."""
        return _submit(recorder, params, "scores", scores, sector_view=sector_view)

    return submit_pod_view


def risk_submit_tool(recorder: NodeRecorder, params: SectorPodsParams) -> Callable[..., dict[str, Any]]:
    """Build the risk agent's submit tool, recording into `recorder`; its texts go through the recorder's length gate first."""

    def submit_risk_review(objections: list[Objection], summary: str) -> dict[str, Any]:
        """Submit your objections to the pods' views (none is fine) and a summary."""
        return _submit(recorder, params, "objections", objections, summary=summary)

    return submit_risk_review


def pm_submit_tool(recorder: NodeRecorder, params: SectorPodsParams) -> Callable[..., dict[str, Any]]:
    """Build the portfolio manager's submit tool, recording into `recorder`; its texts go through the recorder's length gate first."""

    def submit_target_weights(weights: list[Weight], overrides: str, summary: str) -> dict[str, Any]:
        """Submit the target weight of each ETF you want to hold, what you overrode, and a summary."""
        return _submit(recorder, params, "weights", weights, overrides=overrides, summary=summary)

    return submit_target_weights

"""The pieces of an agent hand-off that strategies share: the error, the trade report's validation and the field readers.

A strategy's submit tools validate an agent's argument in code and return `{"error": ...}` so the model can correct itself
in the same run (see `congress_trades/handoff.py`). What is common to them lives here: `HandoffError`, the pure readers of
list items (`items`, `key`, `text`, `required_text`, `optional_text`, `weight`, `covering`) and the trade report, accepted only when it names every order the desk placed and
every order has been checked (`validate_report` against the desk's `OrderView`s). `SubmitRecorder` is the base of every
strategy's recorder: it arms a stage, guards the submit tools and records the first valid submission.

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from the
function's real annotations (as `memory/tools.py` does).
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar


class HandoffError(ValueError):
    """A submission that breaks a rule; the message tells the model what is wrong and what is allowed."""


@dataclass(frozen=True, slots=True)
class OrderReport:
    order_id: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class OrderView:
    """What the trade desk knows about one order it placed this run (the report is validated against it)."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    status: str  # "filled" | "working" | "partially_filled" | "canceled" | "rejected" | ...
    filled_quantity: float
    checked: bool  # `check_orders` ran after this order was placed

    @property
    def filled(self) -> bool:
        return self.status == "filled"


# --- shared field readers (pure) -------------------------------------------------------------------


def items(raw: Any, what: str) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list | tuple):
        raise HandoffError(f"{what} must be a list of objects")
    result = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise HandoffError(f"{what} item {index} must be an object, got {type(item).__name__}")
        result.append(item)
    return result


def key(item: Mapping[str, Any], field: str, index: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"item {index} has no {field}")
    return value.strip().upper() if field in ("ticker", "symbol") else value.strip()


def _clean_text(value: Any, field: str, who: str, max_chars: int) -> str:
    """The stripped text of a non-empty `value`, at most `max_chars` long."""
    if not isinstance(value, str):
        raise HandoffError(f"the {field} for {who} must be text")
    stripped = value.strip()
    if len(stripped) > max_chars:
        raise HandoffError(f"the {field} for {who} is {len(stripped)} characters: keep it under {max_chars}")
    return stripped


def required_text(item: Mapping[str, Any], field: str, who: str, max_chars: int) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{who} has no {field}: give one short sentence")
    return _clean_text(value, field, who, max_chars)


def optional_text(item: Mapping[str, Any], field: str, who: str, max_chars: int) -> str | None:
    value = item.get(field)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _clean_text(value, field, who, max_chars)


def text(item: Mapping[str, Any], who: str, max_chars: int, *, required: bool) -> str | None:
    """The item's `reason`; a missing one is refused with "needs a reason" when `required`, else None."""
    value = item.get("reason")
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise HandoffError(f"{who} needs a reason: give one short sentence")
        return None
    return _clean_text(value, "reason", who, max_chars)


def weight(item: Mapping[str, Any], ticker: str, *, example: str = "0.08") -> float:
    value = item.get("weight")
    if isinstance(value, bool):
        raise HandoffError(f"the weight for {ticker} must be a number")
    try:
        result = float(value)  # type: ignore[arg-type]  # an int, a float or a numeric string
    except TypeError, ValueError, OverflowError:
        raise HandoffError(f"the weight for {ticker} must be a number, a fraction of portfolio value such as {example}") from None
    if not math.isfinite(result):
        raise HandoffError(f"the weight for {ticker} must be a finite number")
    return result


def covering(
    raw: Any,
    expected: Sequence[str],
    *,
    what: str,
    field: str = "symbol",
    unknown: Callable[[str], str],
    missing: Callable[[list[str]], str],
) -> list[tuple[str, Mapping[str, Any]]]:
    """(key, item) for exactly one item per key in `expected`, in submission order; the messages for an unknown or a missing key are the caller's."""
    entries = items(raw, what)
    wanted = set(expected)
    seen: set[str] = set()
    covered = []
    for index, item in enumerate(entries, start=1):
        name = key(item, field, index)
        if name not in wanted:
            raise HandoffError(unknown(name))
        if name in seen:
            raise HandoffError(f"{name} appears twice")
        seen.add(name)
        covered.append((name, item))
    absent = [name for name in expected if name not in seen]
    if absent:
        raise HandoffError(missing(absent))
    return covered


# --- the trade report (pure) ------------------------------------------------------------------------


def validate_report(raw: Any, *, orders: Mapping[str, OrderView], reason_max_chars: int) -> list[OrderReport]:
    """One entry per order the desk placed (by order_id). Every order must have been checked; one not fully filled needs a reason."""
    entries = items(raw, "orders")
    seen: set[str] = set()
    reports = []
    for index, item in enumerate(entries, start=1):
        order_id = key(item, "order_id", index)
        view = orders.get(order_id)
        if view is None:
            raise HandoffError(f"{order_id} is not an order you placed in this run ({', '.join(sorted(orders)) or 'there are none: submit an empty list'})")
        if order_id in seen:
            raise HandoffError(f"{order_id} appears twice")
        seen.add(order_id)
        label = f"{view.side} {view.symbol} ({order_id})"
        if not view.checked:
            raise HandoffError(f"{label} has not been checked: call check_orders first")
        reason = text(item, f"{label}, which is {view.status}", reason_max_chars, required=not view.filled)
        reports.append(OrderReport(order_id, reason))
    missing = sorted(set(orders) - seen)
    if missing:
        raise HandoffError(f"no entry for {', '.join(missing)}: report every order you placed")
    return reports


# --- the recorder -----------------------------------------------------------------------------------


class SubmitRecorder:
    """Holds what the armed stage must validate against, and the first valid submission for it.

    A subclass sets `TOOL_NAMES` (stage -> tool name), arms a stage with `_arm(stage, context)` and implements `_validate`,
    which returns the submission or raises `HandoffError`. `submit` is what a submit tool returns: `{"status": "recorded"}`
    or `{"error": ...}` (also kept in `last_error`). Insider keeps its own final-submission attributes (note, target, report),
    so `submission` stays None there.
    """

    TOOL_NAMES: ClassVar[Mapping[str, str]] = {}
    NOT_EXPECTED_SUFFIX: ClassVar[str] = ""
    ALREADY_RECORDED: ClassVar[str] = "already recorded for this run"

    def __init__(self) -> None:
        self._stage: str | None = None
        self._context: dict[str, Any] = {}
        self._submission: Any = None
        self._submitted = False
        self.last_error: str | None = None

    def _arm(self, stage: str, context: dict[str, Any]) -> None:
        self._stage, self._context = stage, context
        self._submission, self._submitted, self.last_error = None, False, None

    @property
    def submitted(self) -> bool:
        """Whether the armed stage has a valid submission."""
        return self._submitted

    @property
    def submission(self) -> Any:
        """The armed stage's valid submission, else None."""
        return self._submission

    @property
    def missing_tool(self) -> str:
        """The tool the armed stage still waits for: what a forced retry forces."""
        return self.TOOL_NAMES.get(self._stage or "", "")

    def submit(self, stage: str, *args: Any) -> dict[str, Any]:
        if self._stage != stage:
            return self._fail(f"{self.TOOL_NAMES[stage]} is not expected at this point{self.NOT_EXPECTED_SUFFIX}")
        if self._submitted:
            return self._fail(self.ALREADY_RECORDED)
        try:
            value = self._validate(stage, *args)
        except HandoffError as exc:
            return self._fail(str(exc))
        self._submission, self._submitted, self.last_error = value, True, None
        return {"status": "recorded"}

    def _validate(self, stage: str, *args: Any) -> Any:
        """The submission for `stage`, or a `HandoffError` that tells the model what to fix."""
        raise NotImplementedError

    def _fail(self, message: str) -> dict[str, Any]:
        self.last_error = message
        return {"error": message}

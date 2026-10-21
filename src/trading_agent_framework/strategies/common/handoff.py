"""The pieces of an agent hand-off that strategies share: the error, the trade report's validation and the field readers.

A strategy's submit tools validate an agent's argument in code and return `{"error": ...}` so the model can correct itself
in the same run (see `congress_trades/handoff.py`). What is common to them lives here: `HandoffError`, the pure readers of
list items (`items`, `key`, `text`, `weight`) and the trade report, accepted only when it names every order the desk placed and
every order has been checked (`validate_report` against the desk's `OrderView`s).

This module deliberately has NO `from __future__ import annotations`: the agent layer builds each tool's schema from the
function's real annotations (as `memory/tools.py` does).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


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


def text(item: Mapping[str, Any], who: str, max_chars: int, *, required: bool) -> str | None:
    value = item.get("reason")
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise HandoffError(f"{who} needs a reason: give one short sentence")
        return None
    if not isinstance(value, str):
        raise HandoffError(f"the reason for {who} must be text")
    stripped = value.strip()
    if len(stripped) > max_chars:
        raise HandoffError(f"the reason for {who} is {len(stripped)} characters: keep it under {max_chars}")
    return stripped


def weight(item: Mapping[str, Any], ticker: str) -> float:
    value = item.get("weight")
    if isinstance(value, bool):
        raise HandoffError(f"the weight for {ticker} must be a number")
    try:
        result = float(value)  # type: ignore[arg-type]  # an int, a float or a numeric string
    except TypeError, ValueError, OverflowError:
        raise HandoffError(f"the weight for {ticker} must be a number, a fraction of portfolio value such as 0.08") from None
    if not math.isfinite(result):
        raise HandoffError(f"the weight for {ticker} must be a finite number")
    return result


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

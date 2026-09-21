"""Pure agent-call telemetry: one `CallRecord` per model call, and the per-agent totals.

Duck-typed like `results.py`: `usage_from_message` reads `usage_metadata` off whatever message it is
given, so this module never imports `langchain_core`. A local server that reports no usage yields
`None` token counts (unknown), never `0`, and `summarize` skips them instead of counting them as zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model call. `ts` is the strategy clock's time: simulated in a backtest, real otherwise."""

    ts: datetime
    agent: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    latency_ms: float
    tool_calls: int


_TOKEN_FIELDS = ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens")


def usage_from_message(message: Any) -> Usage:
    """Token usage of an `AIMessage`-like object (LangChain's `usage_metadata`), `None` where unreported."""
    usage = getattr(message, "usage_metadata", None) or {}
    details = usage.get("output_token_details") or {}
    return Usage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=details.get("reasoning"),
        total_tokens=usage.get("total_tokens"),
    )


def summarize(records: Iterable[CallRecord]) -> dict[str, dict[str, Any]]:
    """Per-agent totals, keyed by agent name, in first-seen order; a token total is `None` if never reported."""
    by_agent: dict[str, list[CallRecord]] = {}
    for record in records:
        by_agent.setdefault(record.agent, []).append(record)

    summary: dict[str, dict[str, Any]] = {}
    for agent, calls in by_agent.items():
        latency_total = sum(call.latency_ms for call in calls)
        summary[agent] = {
            "model": next((call.model for call in reversed(calls) if call.model is not None), None),
            "calls": len(calls),
            "tool_calls": sum(call.tool_calls for call in calls),
            **{field: _sum_known(getattr(call, field) for call in calls) for field in _TOKEN_FIELDS},
            "latency_ms_total": latency_total,
            "latency_ms_avg": latency_total / len(calls),
        }
    return summary


def _sum_known(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None

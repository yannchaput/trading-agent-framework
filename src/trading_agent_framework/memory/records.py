"""Pure helpers for the agent memory store: names, ids, JSON, lean items and search scoring.

No I/O and no state (same rules as `brokers/alpaca/orders.py`).
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from uuid import uuid4

SEARCH_TEXT_CHARS = 500
TRUNCATION_SUFFIX = "... [truncated]"

# Event types that write `memory_index`. One that is not an index row's latest event is an
# older version of its memory.
PROJECTED_EVENT_TYPES = frozenset(
    {
        "memory.created",
        "proposal.recorded",
        "risk_note.recorded",
        "decision.recorded",
        "lesson.proposed",
        "lesson.validated",
        "thesis.opened",
        "thesis.updated",
        "thesis.closed",
    }
)

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


def safe_name(value: object) -> str:
    """Lumibot's `_safe_name`; a name made only of dots (or nothing) becomes "strategy"."""
    name = _UNSAFE_NAME_CHARS.sub("_", str(value or "").strip()).strip("_")
    return name if name.strip(".") else "strategy"


def normalize_symbol(value: object) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def new_memory_id(kind: str) -> str:
    """Short on purpose: the model copies thesis ids back into tool calls."""
    return f"{safe_name(kind)}_{secrets.token_hex(4)}"


def new_event_id() -> str:
    return f"event_{uuid4().hex}"


def new_retrieval_id() -> str:
    return f"retrieval_{uuid4().hex}"


def json_dumps(value: object) -> str:
    """Sorted-key JSON; `Decimal`, `datetime` and other non-JSON values become strings."""
    return json.dumps(
        value if value is not None else {}, sort_keys=True, default=str, ensure_ascii=False
    )


def json_loads(value: str | None, default: Any) -> Any:
    return json.loads(value) if value else default


def compact_text(value: object, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - len(TRUNCATION_SUFFIX), 1)].rstrip() + TRUNCATION_SUFFIX


def index_item(row: Mapping[str, Any]) -> dict[str, Any]:
    """A `memory_index` row as a memory dict, JSON columns decoded."""
    return {
        "id": row["memory_id"],
        "latest_event_id": row["latest_event_id"],
        "kind": row["kind"],
        "status": row["status"],
        "symbol": row["symbol"],
        "text": row["text"],
        "tags": json_loads(row["tags_json"], []),
        "metadata": json_loads(row["metadata_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def event_item(row: Mapping[str, Any]) -> dict[str, Any]:
    """A `memory_events` row as a search candidate.

    Only for events that are not an index row's latest event: a projected event type is then an
    older version of its memory, hence "superseded" rather than its original status.
    """
    payload = json_loads(row["payload_json"], {})
    superseded = row["event_type"] in PROJECTED_EVENT_TYPES
    return {
        "id": row["event_id"],
        "memory_id": row["subject_id"],
        "event_type": row["event_type"],
        "kind": payload.get("kind") or row["subject_type"],
        "status": "superseded" if superseded else payload.get("status"),
        "symbol": row["symbol"],
        "text": row["text"],
        "tags": json_loads(row["tags_json"], []),
        "metadata": json_loads(row["metadata_json"], {}),
        "created_at": row["timestamp"],
        "updated_at": row["timestamp"],
    }


def lean_item(item: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]:
    """What the model sees of a memory: no metadata, minute-precision time, truncated text."""
    lean: dict[str, Any] = {
        "id": item["id"],
        "kind": item["kind"],
        "status": item["status"],
        "symbol": item["symbol"],
        "updated_at": str(item["updated_at"])[:16],
        "text": compact_text(item["text"], limit=max_chars),
    }
    if item.get("tags"):
        lean["tags"] = list(item["tags"])
    if "event_type" in item:
        lean["event_type"] = item["event_type"]
        lean["memory_id"] = item["memory_id"]
    return lean


def query_terms(query: str) -> list[str]:
    return [term.lower() for term in query.split()]


def _flatten_values(value: object) -> Iterable[str]:
    """Every leaf value (never a key) in a JSON-shaped structure, as text.

    Matching on `json_dumps(item)` directly would search JSON key names too, so a term like
    "data" would match any item merely for having a "metadata" key. Search must only ever match
    what's actually in the item, not its shape.
    """
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _flatten_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _flatten_values(nested)
    elif value is not None:
        yield str(value)


def score(item: Mapping[str, Any], terms: Sequence[str]) -> int:
    """Lumibot's relevance: how many terms appear in the item's values (1 for an empty query)."""
    if not terms:
        return 1
    haystack = " ".join(_flatten_values(item)).lower()
    return sum(1 for term in terms if term in haystack)


def rank(items: Iterable[Mapping[str, Any]], terms: Sequence[str]) -> list[Mapping[str, Any]]:
    """Items with a score above 0: best score first, then most recently updated first.

    Falls back to every item ordered by recency alone when `terms` is non-empty but nothing
    scores above 0 -- a query worded in terms that never appear verbatim in stored text
    (e.g. "risk_on"/"SHV" when past decisions only ever wrote "bullish"/"SPY") should still
    recall recent memory rather than silently return nothing.
    """
    scored = [(score(item, terms), str(item["updated_at"]), item) for item in items]
    matching = [entry for entry in scored if entry[0] > 0]
    if not matching and terms:
        matching = scored
    matching.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [item for _, _, item in matching]


def render_retrieval_text(items: Iterable[Mapping[str, Any]]) -> str:
    """Lumibot's `rendered_text` column of `memory_retrievals`."""
    return "\n\n".join(
        f"{item['kind']} {item['symbol'] or ''} {item['status'] or ''}: {item['text'] or ''}".strip()
        for item in items
    )

# Agent Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port lumibot's agent memory: a SQLite store per strategy and trading mode (`strategy.memory`) plus lumibot's 9 memory tools as plain typed Python functions.

**Architecture:** New package `src/trading_agent_framework/memory/`. `records.py` is pure (ids, JSON, lean items, search scoring). `store.py` holds `MemoryStore`, the only code touching SQLite: an append-only `memory_events` ledger, a `memory_index` projection of each memory's current state, and a `memory_retrievals` log of every search. `tools.py` wraps a store into the 9 tool functions the future LangChain agent layer will bind. `Strategy.memory` lazily opens `<project_root>/memory/<strategy>/<mode>/memory.sqlite`, and a backtest starts from an empty DB.

**Tech Stack:** Python 3.14, `uv`, stdlib `sqlite3` / `contextvars`, `pytest`, `ruff`, `uv check`. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-11-agent-memory-design.md`. Read it before starting any task, **including §10 (amendments made while planning), which overrides the sections it names**.

## Global Constraints

- Python `>=3.14`, package manager `uv`. **No new dependency** (no `langchain`, no `pyarrow`). SQLite through stdlib `sqlite3` only.
- `memory/records.py` is pure: no I/O, no `sqlite3` import, no state, stdlib only.
- `memory/store.py` is the only source module that imports `sqlite3`. It never imports `core/` or `Strategy`; it gets time from injected `now` / `wall_clock` callables and held positions as arguments.
- The only real-clock call in the package is `store._utc_now` (the `wall_clock` default). No `time.sleep`.
- Every `sqlite3.Error` reaches callers as `MemoryStoreError`; every rejected input (empty text or kind, non-list tags, unknown/non-open thesis) as `MemoryValidationError` (a subclass of both `MemoryStoreError` and `ValueError`). Tools turn **only** `MemoryValidationError` into `{"error": "<message>"}`.
- DB path: `<project_root>/memory/<safe_name(strategy)>/<mode>/memory.sqlite`. Schema identical to lumibot's (§4.2), `SCHEMA_VERSION = 1`.
- Tool names, parameter names, order and defaults exactly as spec §6. One-line docstrings. `memory/tools.py` must **not** use `from __future__ import annotations` (the agent layer reads real annotation objects from `inspect.signature`).
- Money stays `Decimal` in Python. It becomes a JSON string only when stored (`records.json_dumps` uses `default=str`).
- Tests never touch the network, use `tmp_path` for every DB, and use hand-written fakes (`tests/fakes.py`), never `MagicMock`. Test-file basenames must be unique across the repo (there is no `__init__.py` under `tests/`).
- Every task ends green on `uv run pytest`, `uv run ruff check` and `uv check`. Ruff allows 200-character lines, but code stays near 100. If ruff reports import order, run `uv run ruff check --fix` and re-check.
- Work on branch `feature/agent-memory`. Commit once per task with the message `Task N: <summary>`, ending with the attribution trailer lines from your instructions. **Never stage `TODO.md`**: it carries the user's own uncommitted edits.

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/trading_agent_framework/utils/errors.py` | 1 | `+ MemoryStoreError`, `MemoryValidationError` |
| `src/trading_agent_framework/memory/__init__.py` | 1, 5 | Package docstring (1); public exports (5) |
| `src/trading_agent_framework/memory/records.py` | 1 | Pure helpers |
| `.gitignore` | 1 | `/memory/` |
| `src/trading_agent_framework/memory/store.py` | 2, 3, 4 | Foundation + `remember`/`get` (2); other writes (3); search + compact state (4) |
| `tests/fakes.py` | 2 | `make_memory_store`, `memory_rows`, `MEMORY_START`, `MEMORY_WALL_TIME` |
| `src/trading_agent_framework/memory/tools.py` | 5 | `memory_tools`, `agent_call_context` |
| `src/trading_agent_framework/core/strategy.py` | 6 | Lazy `memory` property |
| `CLAUDE.md` | 6 | Architecture bullet + gotchas |
| `tests/memory/test_memory_records.py` | 1 | |
| `tests/memory/test_memory_store.py` | 2, 3, 4 | |
| `tests/memory/test_memory_tools.py` | 5 | |
| `tests/core/test_strategy_memory.py` | 6 | |

---

### Task 1: Errors, pure `records.py` helpers and the package skeleton

**Files:**
- Modify: `src/trading_agent_framework/utils/errors.py` (append two classes)
- Create: `src/trading_agent_framework/memory/__init__.py`
- Create: `src/trading_agent_framework/memory/records.py`
- Modify: `.gitignore` (append)
- Test: `tests/memory/test_memory_records.py`

**Interfaces:**
- Produces (used by Tasks 2-5):
  - `utils.errors.MemoryStoreError(TradingFrameworkError)`, `utils.errors.MemoryValidationError(MemoryStoreError, ValueError)`
  - `records.SEARCH_TEXT_CHARS = 500`, `records.TRUNCATION_SUFFIX = "... [truncated]"`, `records.PROJECTED_EVENT_TYPES: frozenset[str]`
  - `records.safe_name(value: object) -> str`
  - `records.normalize_symbol(value: object) -> str | None`
  - `records.new_memory_id(kind: str) -> str` (`<safe kind>_<8 hex>`), `records.new_event_id() -> str` (`event_<32 hex>`), `records.new_retrieval_id() -> str` (`retrieval_<32 hex>`)
  - `records.json_dumps(value: object) -> str`, `records.json_loads(value: str | None, default: Any) -> Any`
  - `records.compact_text(value: object, *, limit: int) -> str`
  - `records.index_item(row: Mapping[str, Any]) -> dict[str, Any]`: keys `id, latest_event_id, kind, status, symbol, text, tags, metadata, created_at, updated_at`
  - `records.event_item(row: Mapping[str, Any]) -> dict[str, Any]`: keys `id, memory_id, event_type, kind, status, symbol, text, tags, metadata, created_at, updated_at`
  - `records.lean_item(item: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]`
  - `records.query_terms(query: str) -> list[str]`, `records.score(item, terms) -> int`, `records.rank(items, terms) -> list[Mapping[str, Any]]`, `records.render_retrieval_text(items) -> str`

- [ ] **Step 1: Create the branch**

```bash
git switch -c feature/agent-memory
```

- [ ] **Step 2: Write the failing tests**

Create `tests/memory/test_memory_records.py`:

```python
from __future__ import annotations

import re
from decimal import Decimal

import pytest

from trading_agent_framework.memory import records

_INDEX_ROW = {
    "memory_id": "thesis_1a2b3c4d",
    "latest_event_id": "event_latest",
    "kind": "thesis",
    "status": "open",
    "symbol": "SPY",
    "text": "Long SPY on breadth",
    "tags_json": '["macro"]',
    "metadata_json": '{"symbol": "SPY"}',
    "created_at": "2026-09-14T10:00:00-04:00",
    "updated_at": "2026-09-14T10:05:30-04:00",
}


def _event_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "event_e1",
        "event_type": "order.submitted",
        "subject_type": "order",
        "subject_id": "order_abc",
        "symbol": "SPY",
        "text": "Submitted order buy 10 SPY as market",
        "tags_json": "[]",
        "metadata_json": "{}",
        "payload_json": '{"kind": "order", "status": "submitted"}',
        "timestamp": "2026-09-14T10:01:00-04:00",
    }
    return row | overrides


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("momentum", "momentum"),
        ("my strat/v2", "my_strat_v2"),
        ("  a=b-c.d  ", "a=b-c.d"),
        ("../etc", ".._etc"),
        ("..", "strategy"),
        (".", "strategy"),
        ("///", "strategy"),
        ("", "strategy"),
        (None, "strategy"),
    ],
)
def test_safe_name(raw: object, expected: str) -> None:
    assert records.safe_name(raw) == expected


def test_normalize_symbol() -> None:
    assert records.normalize_symbol(" spy ") == "SPY"
    assert records.normalize_symbol("") is None
    assert records.normalize_symbol(None) is None


def test_memory_ids_are_the_kind_plus_8_hex_chars() -> None:
    assert re.fullmatch(r"thesis_[0-9a-f]{8}", records.new_memory_id("thesis"))
    assert re.fullmatch(r"Macro_View_[0-9a-f]{8}", records.new_memory_id("Macro View"))


def test_event_and_retrieval_ids_are_full_uuids() -> None:
    assert re.fullmatch(r"event_[0-9a-f]{32}", records.new_event_id())
    assert re.fullmatch(r"retrieval_[0-9a-f]{32}", records.new_retrieval_id())


def test_json_dumps_sorts_keys_keeps_unicode_and_stringifies_decimals() -> None:
    assert records.json_dumps({"b": Decimal("1.50"), "a": "é"}) == '{"a": "é", "b": "1.50"}'
    assert records.json_dumps(None) == "{}"


def test_json_loads_falls_back_to_the_default() -> None:
    assert records.json_loads(None, []) == []
    assert records.json_loads("", {}) == {}
    assert records.json_loads('["x"]', []) == ["x"]


def test_compact_text() -> None:
    assert records.compact_text("  short  ", limit=10) == "short"
    truncated = records.compact_text("x" * 100, limit=40)
    assert len(truncated) == 40
    assert truncated.endswith(records.TRUNCATION_SUFFIX)


def test_index_item_decodes_the_json_columns() -> None:
    item = records.index_item(_INDEX_ROW)
    assert item["id"] == "thesis_1a2b3c4d"
    assert item["latest_event_id"] == "event_latest"
    assert item["tags"] == ["macro"]
    assert item["metadata"] == {"symbol": "SPY"}
    assert (item["created_at"], item["updated_at"]) == (
        "2026-09-14T10:00:00-04:00",
        "2026-09-14T10:05:30-04:00",
    )


def test_event_item_takes_kind_and_status_from_the_payload() -> None:
    item = records.event_item(_event_row())
    assert (item["id"], item["memory_id"], item["event_type"]) == (
        "event_e1",
        "order_abc",
        "order.submitted",
    )
    assert (item["kind"], item["status"]) == ("order", "submitted")
    assert item["updated_at"] == "2026-09-14T10:01:00-04:00"


def test_event_item_falls_back_to_the_subject_type() -> None:
    row = _event_row(
        event_type="agent_warning", subject_type="warning", payload_json='{"warning": "w"}'
    )
    item = records.event_item(row)
    assert (item["kind"], item["status"]) == ("warning", None)


def test_older_versions_of_a_memory_are_superseded() -> None:
    row = _event_row(
        event_type="thesis.opened",
        subject_type="thesis",
        subject_id="thesis_1a2b3c4d",
        payload_json='{"kind": "thesis", "status": "open"}',
    )
    assert records.event_item(row)["status"] == "superseded"


def test_lean_item_shape() -> None:
    assert records.lean_item(records.index_item(_INDEX_ROW), max_chars=500) == {
        "id": "thesis_1a2b3c4d",
        "kind": "thesis",
        "status": "open",
        "symbol": "SPY",
        "updated_at": "2026-09-14T10:05",
        "text": "Long SPY on breadth",
        "tags": ["macro"],
    }


def test_lean_item_omits_empty_tags_and_truncates_text() -> None:
    row = _INDEX_ROW | {"tags_json": "[]", "text": "y" * 50}
    lean = records.lean_item(records.index_item(row), max_chars=20)
    assert "tags" not in lean
    assert len(lean["text"]) == 20
    assert lean["text"].endswith(records.TRUNCATION_SUFFIX)


def test_lean_event_item_links_back_to_its_memory() -> None:
    lean = records.lean_item(records.event_item(_event_row()), max_chars=500)
    assert (lean["event_type"], lean["memory_id"]) == ("order.submitted", "order_abc")


def test_score_counts_matching_terms_case_insensitively() -> None:
    item = records.index_item(_INDEX_ROW)
    assert records.query_terms("  SPY   Breadth gold ") == ["spy", "breadth", "gold"]
    assert records.score(item, records.query_terms("spy BREADTH gold")) == 2
    assert records.score(item, []) == 1
    assert records.score(item, ["gold"]) == 0


def test_rank_drops_misses_and_orders_by_score_then_recency() -> None:
    two_terms_early = {"id": "a", "updated_at": "2026-09-14T09:00", "text": "spy breadth"}
    two_terms_late = {"id": "b", "updated_at": "2026-09-14T11:00", "text": "spy breadth"}
    one_term_latest = {"id": "c", "updated_at": "2026-09-14T12:00", "text": "spy"}
    miss = {"id": "d", "updated_at": "2026-09-14T13:00", "text": "gold"}
    ranked = records.rank(
        [two_terms_early, one_term_latest, miss, two_terms_late], ["spy", "breadth"]
    )
    assert [item["id"] for item in ranked] == ["b", "a", "c"]


def test_render_retrieval_text_matches_lumibot() -> None:
    items = [
        {"kind": "thesis", "symbol": "SPY", "status": "open", "text": "Long"},
        {"kind": "memory", "symbol": None, "status": "active", "text": "note"},
    ]
    assert records.render_retrieval_text(items) == "thesis SPY open: Long\n\nmemory  active: note"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/memory/test_memory_records.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.memory'`.

- [ ] **Step 4: Add the two error classes**

Append to `src/trading_agent_framework/utils/errors.py`:

```python


class MemoryStoreError(TradingFrameworkError):
    """Raised when the agent memory database cannot be read or written."""


class MemoryValidationError(MemoryStoreError, ValueError):
    """Raised when a memory write is rejected (bad text/kind/tags, unknown or non-open thesis)."""
```

- [ ] **Step 5: Create the package**

Create `src/trading_agent_framework/memory/__init__.py`:

```python
"""Agent memory: lumibot's SQLite-backed memory store and its 9 agent tools."""
```

Create `src/trading_agent_framework/memory/records.py`:

```python
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


def score(item: Mapping[str, Any], terms: Sequence[str]) -> int:
    """Lumibot's relevance: how many terms appear in the item's JSON (1 for an empty query)."""
    if not terms:
        return 1
    haystack = json_dumps(item).lower()
    return sum(1 for term in terms if term in haystack)


def rank(items: Iterable[Mapping[str, Any]], terms: Sequence[str]) -> list[Mapping[str, Any]]:
    """Items with a score above 0: best score first, then most recently updated first."""
    scored = [(score(item, terms), str(item["updated_at"]), item) for item in items]
    matching = [entry for entry in scored if entry[0] > 0]
    matching.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [item for _, _, item in matching]


def render_retrieval_text(items: Iterable[Mapping[str, Any]]) -> str:
    """Lumibot's `rendered_text` column of `memory_retrievals`."""
    return "\n\n".join(
        f"{item['kind']} {item['symbol'] or ''} {item['status'] or ''}: {item['text'] or ''}".strip()
        for item in items
    )
```

- [ ] **Step 6: Ignore the runtime memory folder**

Append to `.gitignore`:

```
# Agent memory databases (memory/<strategy>/<mode>/memory.sqlite)
/memory/
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/memory/test_memory_records.py -q`
Expected: all pass.

- [ ] **Step 8: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/utils/errors.py src/trading_agent_framework/memory/ .gitignore tests/memory/test_memory_records.py
git commit -m "Task 1: add memory errors and pure memory record helpers"
```

(Add the attribution trailer lines to the message.)

---

### Task 2: `MemoryStore` foundation: schema, transactions, `remember` and `get`

**Files:**
- Create: `src/trading_agent_framework/memory/store.py`
- Modify: `tests/fakes.py` (imports + helpers appended at the end)
- Test: `tests/memory/test_memory_store.py`

**Interfaces:**
- Consumes: `records.*` and the error classes from Task 1.
- Produces (used by Tasks 3-6):
  - `store.SCHEMA_VERSION = 1`, `store.DB_FILE_NAME = "memory.sqlite"`
  - `store.memory_db_path(project_root: Path, strategy_name: str, mode: TradingMode) -> Path`
  - `store.HeldPosition(symbol: str, quantity: Decimal, last_price: Decimal | None = None)`, frozen, with property `market_value -> Decimal | None`
  - `MemoryStore(db_path: Path, *, strategy_name: str, now: Callable[[], datetime], wall_clock: Callable[[], datetime] = _utc_now, fresh: bool = False)`, attributes `db_path`, `strategy_name`
  - `MemoryStore.remember(text, *, kind="memory", tags=None, metadata=None, agent_name=None, model_call_id=None, retrieval_id=None) -> dict[str, Any]` (the `records.index_item` dict)
  - `MemoryStore.get(memory_id: str) -> dict[str, Any] | None`
  - Private helpers later tasks call: `_transaction()`, `_connect()`, `_create(...)`, `_write_memory(conn, ...)`, `_append_event(conn, ...)`, `_timestamp()`, `_wall_time()`, module functions `_require_text`, `_require_tags`
  - `tests.fakes`: `MEMORY_START = et(2026, 9, 14, 10, 0)`, `MEMORY_WALL_TIME = datetime(2026, 9, 14, 14, 0, 5, tzinfo=UTC)`, `make_memory_store(directory: Path, clock: FakeClock | None = None, *, fresh: bool = False) -> MemoryStore`, `memory_rows(store: MemoryStore, sql: str, params: Sequence[object] = ()) -> list[dict[str, Any]]`

- [ ] **Step 1: Add the shared test helpers to `tests/fakes.py`**

In the import block of `tests/fakes.py`, add `import sqlite3` (next to `import threading`), `from pathlib import Path`, and change `from typing import ClassVar` to `from typing import Any, ClassVar`. After the existing `trading_agent_framework` imports, add:

```python
from trading_agent_framework.memory.store import DB_FILE_NAME, MemoryStore
```

Append to the end of `tests/fakes.py`:

```python


# --- agent memory ---------------------------------------------------------------

MEMORY_START = et(2026, 9, 14, 10, 0)
MEMORY_WALL_TIME = datetime(2026, 9, 14, 14, 0, 5, tzinfo=UTC)


def make_memory_store(
    directory: Path, clock: FakeClock | None = None, *, fresh: bool = False
) -> MemoryStore:
    """A `MemoryStore` at `directory/memory.sqlite`, on fake time (`MEMORY_START` by default)."""
    clock = clock if clock is not None else FakeClock(MEMORY_START)
    return MemoryStore(
        directory / DB_FILE_NAME,
        strategy_name="momentum",
        now=clock.now,
        wall_clock=lambda: MEMORY_WALL_TIME,
        fresh=fresh,
    )


def memory_rows(
    store: MemoryStore, sql: str, params: Sequence[object] = ()
) -> list[dict[str, Any]]:
    """Raw rows from the store's database, read on the test's own connection."""
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params)]
    finally:
        conn.close()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/memory/test_memory_store.py`:

```python
from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import make_memory_store, memory_rows

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.memory.store import (
    DB_FILE_NAME,
    HeldPosition,
    MemoryStore,
    memory_db_path,
)
from trading_agent_framework.utils.errors import MemoryStoreError, MemoryValidationError


def _events(store: MemoryStore) -> list[dict[str, Any]]:
    return memory_rows(store, "SELECT * FROM memory_events ORDER BY sequence")


# --- location, schema, lifecycle ---------------------------------------------------


def test_memory_db_path_mirrors_the_logs_layout() -> None:
    assert memory_db_path(Path("/root"), "my strat/v2", TradingMode.PAPER) == Path(
        "/root/memory/my_strat_v2/paper/memory.sqlite"
    )


def test_constructor_creates_parent_dirs_and_the_schema_in_wal_mode(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path / "a" / "b")
    assert store.db_path == tmp_path / "a" / "b" / DB_FILE_NAME
    tables = {
        row["name"] for row in memory_rows(store, "SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert tables == {"memory_events", "memory_index", "memory_retrievals"}
    assert memory_rows(store, "PRAGMA journal_mode") == [{"journal_mode": "wal"}]


def test_memories_persist_across_store_instances(tmp_path: Path) -> None:
    item = make_memory_store(tmp_path).remember("keep me")
    assert make_memory_store(tmp_path).get(item["id"]) == item


def test_a_fresh_store_deletes_the_previous_database(tmp_path: Path) -> None:
    old = make_memory_store(tmp_path).remember("stale")
    for suffix in ("-wal", "-shm"):
        (tmp_path / f"{DB_FILE_NAME}{suffix}").write_bytes(b"stale")
    store = make_memory_store(tmp_path, fresh=True)
    assert store.get(old["id"]) is None
    for suffix in ("-wal", "-shm"):
        side_file = tmp_path / f"{DB_FILE_NAME}{suffix}"
        assert not side_file.exists() or side_file.read_bytes() != b"stale"


def test_sqlite_errors_surface_as_memory_store_error(tmp_path: Path) -> None:
    (tmp_path / DB_FILE_NAME).mkdir()
    with pytest.raises(MemoryStoreError, match="memory database"):
        make_memory_store(tmp_path)


def test_held_position_market_value() -> None:
    assert HeldPosition("SPY", Decimal(10), Decimal("512.3")).market_value == Decimal("5123.0")
    assert HeldPosition("SPY", Decimal(10)).market_value is None


# --- remember / get -----------------------------------------------------------------


def test_remember_writes_an_event_and_a_projection(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember("Fed pivot likely in Q4", tags=["macro"], metadata={"source": "fomc"})

    assert re.fullmatch(r"memory_[0-9a-f]{8}", item["id"])
    assert (item["kind"], item["status"], item["symbol"]) == ("memory", "active", None)
    assert (item["text"], item["tags"]) == ("Fed pivot likely in Q4", ["macro"])
    assert item["metadata"] == {"source": "fomc"}
    assert item["created_at"] == item["updated_at"] == "2026-09-14T10:00:00-04:00"

    [event] = _events(store)
    assert (event["event_type"], event["subject_type"], event["subject_id"]) == (
        "memory.created",
        "memory",
        item["id"],
    )
    assert (event["sequence"], event["strategy"], event["schema_version"]) == (1, "momentum", 1)
    assert event["timestamp"] == "2026-09-14T10:00:00-04:00"
    assert event["wall_time"] == "2026-09-14T14:00:05Z"
    assert (event["agent_name"], event["model_call_id"], event["retrieval_id"]) == (None, None, None)
    assert json.loads(event["payload_json"]) == {
        "kind": "memory",
        "status": "active",
        "text": "Fed pivot likely in Q4",
        "symbol": None,
        "tags": ["macro"],
        "metadata": {"source": "fomc"},
    }
    assert event["payload_hash"] == hashlib.sha256(event["payload_json"].encode()).hexdigest()
    assert item["latest_event_id"] == event["event_id"]
    assert store.get(item["id"]) == item


def test_remember_with_a_custom_kind_takes_the_symbol_from_metadata(tmp_path: Path) -> None:
    item = make_memory_store(tmp_path).remember(
        "  SPY breadth is thin  ", kind="macro_view", metadata={"symbol": "spy"}
    )
    assert item["id"].startswith("macro_view_")
    assert (item["kind"], item["symbol"], item["text"]) == ("macro_view", "SPY", "SPY breadth is thin")


def test_remember_records_provenance(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.remember("x", agent_name="analyst", model_call_id="call-1", retrieval_id="retrieval_1")
    [event] = _events(store)
    assert (event["agent_name"], event["model_call_id"], event["retrieval_id"]) == (
        "analyst",
        "call-1",
        "retrieval_1",
    )


def test_event_sequence_increments(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.remember("one")
    store.remember("two")
    assert [event["sequence"] for event in _events(store)] == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"text": "   "}, "text must be a non-empty string"),
        ({"text": "x", "kind": " "}, "kind must be a non-empty string"),
        ({"text": "x", "tags": "macro"}, "tags must be a list of strings"),
        ({"text": "x", "tags": [1]}, "tags must be a list of strings"),
    ],
)
def test_remember_rejects_bad_input_without_writing(
    tmp_path: Path, kwargs: dict[str, Any], message: str
) -> None:
    store = make_memory_store(tmp_path)
    with pytest.raises(MemoryValidationError, match=message):
        store.remember(**kwargs)
    assert _events(store) == []


def test_get_returns_none_for_an_unknown_id(tmp_path: Path) -> None:
    assert make_memory_store(tmp_path).get("memory_00000000") is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/memory/test_memory_store.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.memory.store'` (raised from `tests/fakes.py`).

- [ ] **Step 4: Create `store.py`**

Create `src/trading_agent_framework/memory/store.py`:

```python
"""SQLite-backed agent memory: lumibot's `MemoryStore`, one database per strategy and mode.

`memory_events` is the append-only ledger of every write, `memory_index` projects each memory's
current state (what search and `compact_state` read), and `memory_retrievals` records every
search. This is the only module that touches SQLite.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework.memory import records
from trading_agent_framework.utils.errors import MemoryStoreError, MemoryValidationError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import TradingMode

SCHEMA_VERSION = 1
DB_FILE_NAME = "memory.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    wall_time TEXT NOT NULL,
    strategy TEXT NOT NULL,
    event_type TEXT NOT NULL,
    subject_type TEXT,
    subject_id TEXT,
    symbol TEXT,
    agent_name TEXT,
    model_call_id TEXT,
    retrieval_id TEXT,
    text TEXT,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_events_subject ON memory_events(subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_memory_events_symbol ON memory_events(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_events_type ON memory_events(event_type);

CREATE TABLE IF NOT EXISTS memory_index (
    memory_id TEXT PRIMARY KEY,
    latest_event_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT,
    symbol TEXT,
    text TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_index_kind ON memory_index(kind);
CREATE INDEX IF NOT EXISTS idx_memory_index_symbol ON memory_index(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_index_status ON memory_index(status);

CREATE TABLE IF NOT EXISTS memory_retrievals (
    retrieval_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    wall_time TEXT NOT NULL,
    strategy TEXT NOT NULL,
    agent_name TEXT,
    model_call_id TEXT,
    query TEXT NOT NULL,
    kind TEXT,
    symbol TEXT,
    status TEXT,
    result_limit INTEGER NOT NULL,
    candidate_ids_json TEXT NOT NULL,
    selected_ids_json TEXT NOT NULL,
    rendered_text TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_retrievals_symbol ON memory_retrievals(symbol);
CREATE INDEX IF NOT EXISTS idx_memory_retrievals_agent ON memory_retrievals(agent_name);
"""

_INSERT_EVENT = """
INSERT INTO memory_events (
    event_id, sequence, timestamp, wall_time, strategy, event_type,
    subject_type, subject_id, symbol, agent_name, model_call_id,
    retrieval_id, text, tags_json, metadata_json, payload_json,
    payload_hash, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# `created_at` is not in the DO UPDATE list: an update keeps the memory's creation time.
_UPSERT_INDEX = """
INSERT INTO memory_index (
    memory_id, latest_event_id, kind, status, symbol, text,
    tags_json, metadata_json, created_at, updated_at, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(memory_id) DO UPDATE SET
    latest_event_id = excluded.latest_event_id,
    kind = excluded.kind,
    status = excluded.status,
    symbol = excluded.symbol,
    text = excluded.text,
    tags_json = excluded.tags_json,
    metadata_json = excluded.metadata_json,
    updated_at = excluded.updated_at,
    schema_version = excluded.schema_version
"""


def memory_db_path(project_root: Path, strategy_name: str, mode: TradingMode) -> Path:
    """`<project_root>/memory/<strategy>/<mode>/memory.sqlite` (mirrors `logs/<strategy>/<mode>/`)."""
    return project_root / "memory" / records.safe_name(strategy_name) / mode.value / DB_FILE_NAME


@dataclass(frozen=True, slots=True)
class HeldPosition:
    """A held position as `compact_state` needs it; the caller builds it from the strategy."""

    symbol: str
    quantity: Decimal
    last_price: Decimal | None = None

    @property
    def market_value(self) -> Decimal | None:
        return None if self.last_price is None else self.quantity * self.last_price


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise MemoryValidationError("text must be a non-empty string")
    return text.strip()


def _require_tags(tags: object) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, (list, tuple)) or not all(isinstance(tag, str) for tag in tags):
        raise MemoryValidationError("tags must be a list of strings")
    return list(tags)


class MemoryStore:
    """Lumibot's agent memory store (`strategy.memory`) over one SQLite file.

    Every operation opens a connection, runs one transaction and closes it, so a write is atomic
    and the file can be inspected with `sqlite3` while the bot runs (WAL journal).
    """

    def __init__(
        self,
        db_path: Path,
        *,
        strategy_name: str,
        now: Callable[[], datetime],
        wall_clock: Callable[[], datetime] = _utc_now,
        fresh: bool = False,
    ) -> None:
        self.db_path = db_path
        self.strategy_name = strategy_name
        self._now = now
        self._wall_clock = wall_clock
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if fresh:
                for suffix in ("", "-wal", "-shm"):
                    db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        except OSError as exc:
            raise MemoryStoreError(f"Cannot prepare memory database {db_path}: {exc}") from exc
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    # --- writes -----------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        kind: str = "memory",
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a free-form memory or note (`memory.created`)."""
        if not isinstance(kind, str) or not kind.strip():
            raise MemoryValidationError("kind must be a non-empty string")
        return self._create(
            event_type="memory.created",
            kind=kind.strip(),
            status="active",
            text=text,
            symbol=None,
            tags=tags,
            metadata=dict(metadata or {}),
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    # --- reads ------------------------------------------------------------------------

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """The current state of one memory (full row, JSON decoded), or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_index WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return None if row is None else records.index_item(dict(row))

    # --- internals --------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection in autocommit mode, always closed; SQLite errors become `MemoryStoreError`."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Error in memory database {self.db_path}: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction: committed on success, rolled back on any exception."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _create(
        self,
        *,
        event_type: str,
        kind: str,
        status: str,
        text: object,
        symbol: str | None,
        tags: object,
        metadata: dict[str, Any],
        agent_name: str | None,
        model_call_id: str | None,
        retrieval_id: str | None,
    ) -> dict[str, Any]:
        """Validate, then write a new memory (new id) in one transaction."""
        clean_text = _require_text(text)
        clean_tags = _require_tags(tags)
        with self._transaction() as conn:
            return self._write_memory(
                conn,
                memory_id=self._unused_memory_id(conn, kind),
                event_type=event_type,
                kind=kind,
                status=status,
                text=clean_text,
                symbol=symbol,
                tags=clean_tags,
                metadata=metadata,
                agent_name=agent_name,
                model_call_id=model_call_id,
                retrieval_id=retrieval_id,
            )

    @staticmethod
    def _unused_memory_id(conn: sqlite3.Connection, kind: str) -> str:
        while True:
            memory_id = records.new_memory_id(kind)
            taken = conn.execute(
                "SELECT 1 FROM memory_index WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if taken is None:
                return memory_id

    def _write_memory(
        self,
        conn: sqlite3.Connection,
        *,
        memory_id: str,
        event_type: str,
        kind: str,
        status: str,
        text: str,
        symbol: str | None,
        tags: list[str],
        metadata: dict[str, Any],
        agent_name: str | None,
        model_call_id: str | None,
        retrieval_id: str | None,
    ) -> dict[str, Any]:
        """Append a projected event and upsert its `memory_index` row; return the new state."""
        normalized_symbol = records.normalize_symbol(symbol or metadata.get("symbol"))
        event = self._append_event(
            conn,
            event_type=event_type,
            subject_type=kind,
            subject_id=memory_id,
            text=text,
            symbol=normalized_symbol,
            tags=tags,
            metadata=metadata,
            payload={
                "kind": kind,
                "status": status,
                "text": text,
                "symbol": normalized_symbol,
                "tags": tags,
                "metadata": metadata,
            },
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )
        conn.execute(
            _UPSERT_INDEX,
            (
                memory_id,
                event["event_id"],
                kind,
                status,
                normalized_symbol,
                text,
                records.json_dumps(tags),
                records.json_dumps(metadata),
                event["timestamp"],
                event["timestamp"],
                SCHEMA_VERSION,
            ),
        )
        row = conn.execute("SELECT * FROM memory_index WHERE memory_id = ?", (memory_id,)).fetchone()
        return records.index_item(dict(row))

    def _append_event(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        subject_type: str,
        subject_id: str,
        text: str,
        symbol: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert one `memory_events` row; `sequence` is allocated inside the caller's transaction."""
        metadata = dict(metadata or {})
        payload_json = records.json_dumps(payload or {})
        event = {
            "event_id": records.new_event_id(),
            "sequence": conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM memory_events"
            ).fetchone()[0],
            "timestamp": self._timestamp(),
            "event_type": event_type,
            "subject_id": subject_id,
            "symbol": records.normalize_symbol(symbol or metadata.get("symbol")),
        }
        conn.execute(
            _INSERT_EVENT,
            (
                event["event_id"],
                event["sequence"],
                event["timestamp"],
                self._wall_time(),
                self.strategy_name,
                event_type,
                subject_type,
                subject_id,
                event["symbol"],
                agent_name,
                model_call_id,
                retrieval_id,
                text,
                records.json_dumps(list(tags)),
                records.json_dumps(metadata),
                payload_json,
                hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                SCHEMA_VERSION,
            ),
        )
        return event

    def _timestamp(self) -> str:
        """Strategy time (simulated in backtests), as lumibot's `timestamp` column."""
        return self._now().isoformat()

    def _wall_time(self) -> str:
        return self._wall_clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/memory/ -q`
Expected: all pass.

- [ ] **Step 6: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/memory/store.py tests/fakes.py tests/memory/test_memory_store.py
git commit -m "Task 2: add the SQLite MemoryStore foundation with remember and get"
```

(Add the attribution trailer lines to the message.)

---

### Task 3: The other memory writes: proposals, risk notes, decisions, lessons, theses, warnings, orders

**Files:**
- Modify: `src/trading_agent_framework/memory/store.py`
- Test: `tests/memory/test_memory_store.py` (append)

**Interfaces:**
- Consumes: Task 2's `MemoryStore` internals (`_create`, `_write_memory`, `_append_event`, `_transaction`, `_require_text`).
- Produces (used by Tasks 5-6). All keyword arguments after `text` / `thesis_id` are keyword-only, and every method also takes `agent_name=None, model_call_id=None, retrieval_id=None`:
  - `remember_proposal(text, *, symbol=None, action=None, tags=None, metadata=None) -> dict` (index item)
  - `remember_risk_note(text, *, symbol=None, tags=None, metadata=None) -> dict`
  - `remember_decision(text, *, symbol=None, action=None, evidence=None, tags=None) -> dict`
  - `remember_lesson(text, *, symbol=None, outcome=None, tags=None) -> dict`
  - `open_thesis(text, *, symbol=None, tags=None, metadata=None) -> dict`
  - `update_thesis(thesis_id, text, *, metadata=None) -> dict`
  - `close_thesis(thesis_id, text, *, outcome=None) -> dict`
  - `record_warning(text, *, kind="agent_warning", symbol=None, metadata=None) -> dict` (event dict: `event_id, sequence, timestamp, event_type, subject_id, symbol`)
  - `record_order_submitted(order: Order, *, metadata=None) -> dict` (same event dict)

- [ ] **Step 1: Write the failing tests**

Add these imports to the top of `tests/memory/test_memory_store.py` (keep the import blocks sorted; `uv run ruff check --fix` sorts them):

```python
from tests.fakes import MEMORY_START, FakeClock

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
```

Append:

```python
# --- proposals, risk notes, decisions, lessons -------------------------------------------


def test_remember_proposal(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember_proposal(
        "Buy SPY on a pullback", symbol="spy", action="buy", tags=["idea"],
        metadata={"confidence": "low"},
    )
    assert re.fullmatch(r"proposal_[0-9a-f]{8}", item["id"])
    assert (item["kind"], item["status"], item["symbol"], item["tags"]) == (
        "proposal", "proposed", "SPY", ["idea"],
    )
    assert item["metadata"] == {"symbol": "SPY", "action": "buy", "confidence": "low"}
    assert _events(store)[-1]["event_type"] == "proposal.recorded"


def test_remember_risk_note(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember_risk_note("Earnings next week", symbol="SPY", metadata={"src": "cal"})
    assert (item["kind"], item["status"], item["symbol"]) == ("risk_note", "active", "SPY")
    assert item["metadata"] == {"symbol": "SPY", "src": "cal"}
    assert _events(store)[-1]["event_type"] == "risk_note.recorded"


def test_remember_decision_stores_evidence_as_json(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember_decision(
        "Bought SPY on oversold RSI", symbol="SPY", action="buy", evidence={"rsi": Decimal("28.5")}
    )
    assert (item["kind"], item["status"]) == ("decision", "recorded")
    assert item["metadata"] == {"symbol": "SPY", "action": "buy", "evidence": {"rsi": "28.5"}}
    assert _events(store)[-1]["event_type"] == "decision.recorded"


@pytest.mark.parametrize(
    ("outcome", "status"),
    [(None, "proposed"), ({"validated": True}, "validated"), ({"validated": "yes"}, "proposed")],
)
def test_remember_lesson_is_validated_only_by_an_explicit_true(
    tmp_path: Path, outcome: dict[str, Any] | None, status: str
) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember_lesson("Don't chase opening gaps", outcome=outcome)
    assert (item["kind"], item["status"]) == ("lesson", status)
    assert item["metadata"] == {"symbol": None, "outcome": outcome or {}}
    assert _events(store)[-1]["event_type"] == f"lesson.{status}"


# --- theses ----------------------------------------------------------------------


def test_open_thesis(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.open_thesis("Long SPY on breadth", symbol="spy", tags=["macro"], metadata={"horizon": "3m"})
    assert re.fullmatch(r"thesis_[0-9a-f]{8}", item["id"])
    assert (item["kind"], item["status"], item["symbol"]) == ("thesis", "open", "SPY")
    assert item["metadata"] == {"symbol": "SPY", "horizon": "3m", "status": "open"}
    assert _events(store)[-1]["event_type"] == "thesis.opened"


def test_update_thesis_keeps_id_creation_time_symbol_and_tags(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    thesis = store.open_thesis("Long SPY", symbol="SPY", tags=["macro"])
    clock.advance(60)
    updated = store.update_thesis(thesis["id"], "Long SPY, breadth still improving")

    assert updated["id"] == thesis["id"]
    assert (updated["status"], updated["symbol"], updated["tags"]) == ("open", "SPY", ["macro"])
    assert updated["text"] == "Long SPY, breadth still improving"
    assert updated["created_at"] == "2026-09-14T10:00:00-04:00"
    assert updated["updated_at"] == "2026-09-14T10:01:00-04:00"
    assert updated["metadata"] == {"thesis_id": thesis["id"], "status": "open"}
    assert [e["event_type"] for e in _events(store)] == ["thesis.opened", "thesis.updated"]
    assert updated["latest_event_id"] == _events(store)[-1]["event_id"]


def test_update_thesis_symbol_can_come_from_metadata(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    thesis = store.open_thesis("Long SPY", symbol="SPY")
    assert store.update_thesis(thesis["id"], "Now via QQQ", metadata={"symbol": "qqq"})["symbol"] == "QQQ"


def test_close_thesis(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    thesis = store.open_thesis("Long SPY", symbol="SPY", tags=["macro"])
    closed = store.close_thesis(thesis["id"], "Target hit", outcome={"pnl": Decimal("12.5")})
    assert (closed["status"], closed["symbol"], closed["tags"]) == ("closed", "SPY", ["macro"])
    assert closed["metadata"] == {
        "thesis_id": thesis["id"],
        "outcome": {"pnl": "12.5"},
        "status": "closed",
    }
    assert _events(store)[-1]["event_type"] == "thesis.closed"


def test_thesis_changes_reject_unknown_and_non_open_ids(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    closed = store.open_thesis("Long QQQ", symbol="QQQ")
    store.close_thesis(closed["id"], "Stopped out")
    note = store.remember("A thesis-looking note", kind="thesis")
    events_before = len(_events(store))

    for change in (store.update_thesis, store.close_thesis):
        with pytest.raises(MemoryValidationError, match="unknown thesis_id 'thesis_00000000'"):
            change("thesis_00000000", "x")
        with pytest.raises(MemoryValidationError, match=r"is not open \(status: closed\)"):
            change(closed["id"], "x")
        with pytest.raises(MemoryValidationError, match=r"is not open \(status: active\)"):
            change(note["id"], "x")
        with pytest.raises(MemoryValidationError, match="text must be a non-empty string"):
            change(closed["id"], " ")
    assert len(_events(store)) == events_before


# --- history-only events: warnings and orders --------------------------------------------


def test_record_warning_is_history_only(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    event = store.record_warning(
        "Ordered SPY without checking its thesis",
        kind="position_order_without_memory_thesis",
        symbol="spy",
        metadata={"symbols": ["SPY"]},
        agent_name="trader",
    )
    [row] = _events(store)
    assert event["event_id"] == row["event_id"]
    assert (row["event_type"], row["subject_type"], row["symbol"], row["agent_name"]) == (
        "position_order_without_memory_thesis", "warning", "SPY", "trader",
    )
    assert row["subject_id"].startswith("warning_")
    assert json.loads(row["payload_json"]) == {
        "warning": "Ordered SPY without checking its thesis",
        "metadata": {"symbols": ["SPY"]},
    }
    assert memory_rows(store, "SELECT * FROM memory_index") == []


def test_record_warning_defaults_to_agent_warning(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    assert store.record_warning("Something odd")["event_type"] == "agent_warning"


def test_record_order_submitted_links_the_decision_of_the_same_model_call(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    order = Order(
        strategy_name="momentum",
        asset=Asset("SPY"),
        side=OrderSide.BUY,
        quantity=Decimal(10),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("512.30"),
        client_order_id="momentum-1",
    )
    decision = store.remember_decision(
        "Buy SPY", symbol="SPY", action="buy", agent_name="trader", model_call_id="call-7"
    )
    event = store.record_order_submitted(
        order, metadata={"reason": "breakout"}, agent_name="trader", model_call_id="call-7"
    )

    row = _events(store)[-1]
    assert event["event_id"] == row["event_id"]
    assert (row["event_type"], row["subject_type"], row["subject_id"]) == (
        "order.submitted", "order", f"order_{order.identifier}",
    )
    assert row["text"] == "Submitted order buy 10 SPY as limit"
    payload = json.loads(row["payload_json"])
    assert payload == {
        "kind": "order",
        "status": "submitted",
        "symbol": "SPY",
        "side": "buy",
        "quantity": "10",
        "order_type": "limit",
        "asset_type": "stock",
        "order": {
            "identifier": order.identifier,
            "client_order_id": "momentum-1",
            "status": "unprocessed",
            "time_in_force": "day",
            "limit_price": "512.30",
        },
        "metadata": {"reason": "breakout"},
        "decision_id": decision["id"],
    }
    assert json.loads(row["metadata_json"]) == payload
    assert memory_rows(store, "SELECT memory_id FROM memory_index") == [{"memory_id": decision["id"]}]


def test_record_order_submitted_without_a_matching_decision(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.remember_decision("Buy SPY", model_call_id="call-1")
    order = Order(strategy_name="momentum", asset=Asset("SPY"), side=OrderSide.BUY, notional=Decimal("500"))
    store.record_order_submitted(order, model_call_id="call-2")
    row = _events(store)[-1]
    assert row["text"] == "Submitted order buy $500 SPY as market"
    payload = json.loads(row["payload_json"])
    assert payload["decision_id"] is None
    assert payload["quantity"] is None
    assert payload["order"]["notional"] == "500"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/memory/test_memory_store.py -q`
Expected: the new tests fail with `AttributeError: 'MemoryStore' object has no attribute 'remember_proposal'` (and similar); Task 2's tests still pass.

- [ ] **Step 3: Implement the writes**

In `src/trading_agent_framework/memory/store.py`:

1. Extend the `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from trading_agent_framework.config.env import TradingMode
    from trading_agent_framework.entities.order import Order
```

2. Add this module function after `_require_tags`:

```python
def _order_fields(order: Order) -> dict[str, Any]:
    fields = {
        "identifier": order.identifier,
        "client_order_id": order.client_order_id,
        "status": order.status,
        "time_in_force": order.time_in_force,
        "notional": order.notional,
        "limit_price": order.limit_price,
        "stop_price": order.stop_price,
        "stop_limit_price": order.stop_limit_price,
        "trail_price": order.trail_price,
        "trail_percent": order.trail_percent,
    }
    return {key: value for key, value in fields.items() if value is not None}
```

3. Add these methods to `MemoryStore` right after `remember` (still in the `# --- writes` section):

```python
    def remember_proposal(
        self,
        text: str,
        *,
        symbol: str | None = None,
        action: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a research proposal or non-final trade idea (`proposal.recorded`)."""
        symbol = records.normalize_symbol(symbol)
        return self._create(
            event_type="proposal.recorded",
            kind="proposal",
            status="proposed",
            text=text,
            symbol=symbol,
            tags=tags,
            metadata={"symbol": symbol, "action": action, **(metadata or {})},
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    def remember_risk_note(
        self,
        text: str,
        *,
        symbol: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a risk note or bear case (`risk_note.recorded`)."""
        symbol = records.normalize_symbol(symbol)
        return self._create(
            event_type="risk_note.recorded",
            kind="risk_note",
            status="active",
            text=text,
            symbol=symbol,
            tags=tags,
            metadata={"symbol": symbol, **(metadata or {})},
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    def remember_decision(
        self,
        text: str,
        *,
        symbol: str | None = None,
        action: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record an actual trading decision (`decision.recorded`)."""
        symbol = records.normalize_symbol(symbol)
        return self._create(
            event_type="decision.recorded",
            kind="decision",
            status="recorded",
            text=text,
            symbol=symbol,
            tags=tags,
            metadata={"symbol": symbol, "action": action, "evidence": dict(evidence or {})},
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    def remember_lesson(
        self,
        text: str,
        *,
        symbol: str | None = None,
        outcome: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a lesson: `lesson.validated` only when `outcome["validated"] is True`."""
        symbol = records.normalize_symbol(symbol)
        outcome = dict(outcome or {})
        status = "validated" if outcome.get("validated") is True else "proposed"
        return self._create(
            event_type=f"lesson.{status}",
            kind="lesson",
            status=status,
            text=text,
            symbol=symbol,
            tags=tags,
            metadata={"symbol": symbol, "outcome": outcome},
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    def open_thesis(
        self,
        text: str,
        *,
        symbol: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Open an investment thesis (`thesis.opened`)."""
        symbol = records.normalize_symbol(symbol)
        return self._create(
            event_type="thesis.opened",
            kind="thesis",
            status="open",
            text=text,
            symbol=symbol,
            tags=tags,
            metadata={"symbol": symbol, **(metadata or {}), "status": "open"},
            agent_name=agent_name,
            model_call_id=model_call_id,
            retrieval_id=retrieval_id,
        )

    def update_thesis(
        self,
        thesis_id: str,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Replace an open thesis's text (`thesis.updated`); symbol and tags carry over."""
        clean_text = _require_text(text)
        merged = {"thesis_id": thesis_id, **(metadata or {}), "status": "open"}
        with self._transaction() as conn:
            thesis = self._open_thesis_row(conn, thesis_id)
            return self._write_memory(
                conn,
                memory_id=thesis["id"],
                event_type="thesis.updated",
                kind="thesis",
                status="open",
                text=clean_text,
                symbol=records.normalize_symbol(merged.get("symbol")) or thesis["symbol"],
                tags=thesis["tags"],
                metadata=merged,
                agent_name=agent_name,
                model_call_id=model_call_id,
                retrieval_id=retrieval_id,
            )

    def close_thesis(
        self,
        thesis_id: str,
        text: str,
        *,
        outcome: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Close an open thesis with its outcome/reflection (`thesis.closed`)."""
        clean_text = _require_text(text)
        with self._transaction() as conn:
            thesis = self._open_thesis_row(conn, thesis_id)
            return self._write_memory(
                conn,
                memory_id=thesis["id"],
                event_type="thesis.closed",
                kind="thesis",
                status="closed",
                text=clean_text,
                symbol=thesis["symbol"],
                tags=thesis["tags"],
                metadata={"thesis_id": thesis_id, "outcome": dict(outcome or {}), "status": "closed"},
                agent_name=agent_name,
                model_call_id=model_call_id,
                retrieval_id=retrieval_id,
            )

    def record_warning(
        self,
        text: str,
        *,
        kind: str = "agent_warning",
        symbol: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a runtime warning; `kind` becomes the event type (history only)."""
        clean_text = _require_text(text)
        details = dict(metadata or {})
        with self._transaction() as conn:
            return self._append_event(
                conn,
                event_type=kind,
                subject_type="warning",
                subject_id=records.new_memory_id("warning"),
                text=clean_text,
                symbol=symbol,
                metadata=details,
                payload={"warning": clean_text, "metadata": details},
                agent_name=agent_name,
                model_call_id=model_call_id,
                retrieval_id=retrieval_id,
            )

    def record_order_submitted(
        self,
        order: Order,
        *,
        metadata: Mapping[str, Any] | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
        retrieval_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a submitted order (`order.submitted`, history only), linked to the decision
        recorded in the same model call."""
        symbol = order.asset.symbol.upper()
        size = str(order.quantity) if order.quantity is not None else f"${order.notional}"
        text = f"Submitted order {order.side} {size} {symbol} as {order.order_type}"
        with self._transaction() as conn:
            payload = {
                "kind": "order",
                "status": "submitted",
                "symbol": symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "asset_type": order.asset.asset_type,
                "order": _order_fields(order),
                "metadata": dict(metadata or {}),
                "decision_id": self._decision_for_call(conn, agent_name, model_call_id),
            }
            return self._append_event(
                conn,
                event_type="order.submitted",
                subject_type="order",
                subject_id=f"order_{order.identifier}",
                text=text,
                symbol=symbol,
                metadata=payload,
                payload=payload,
                agent_name=agent_name,
                model_call_id=model_call_id,
                retrieval_id=retrieval_id,
            )
```

4. Add these helpers to the `# --- internals` section:

```python
    @staticmethod
    def _open_thesis_row(conn: sqlite3.Connection, thesis_id: str) -> dict[str, Any]:
        row = conn.execute("SELECT * FROM memory_index WHERE memory_id = ?", (thesis_id,)).fetchone()
        if row is None or row["kind"] != "thesis":
            raise MemoryValidationError(f"unknown thesis_id {thesis_id!r}")
        if row["status"] != "open":
            raise MemoryValidationError(f"thesis {thesis_id!r} is not open (status: {row['status']})")
        return records.index_item(dict(row))

    @staticmethod
    def _decision_for_call(
        conn: sqlite3.Connection, agent_name: str | None, model_call_id: str | None
    ) -> str | None:
        """The latest decision recorded in the same model call (lumibot's decision provenance)."""
        if not model_call_id:
            return None
        row = conn.execute(
            """
            SELECT subject_id FROM memory_events
            WHERE event_type = 'decision.recorded' AND model_call_id = ?
              AND (? IS NULL OR agent_name = ?)
            ORDER BY sequence DESC LIMIT 1
            """,
            (model_call_id, agent_name, agent_name),
        ).fetchone()
        return None if row is None else row["subject_id"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/memory/ -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/memory/store.py tests/memory/test_memory_store.py
git commit -m "Task 3: add proposal, risk note, decision, lesson, thesis, warning and order memory events"
```

(Add the attribution trailer lines to the message.)

---

### Task 4: Search, the retrieval log and compact state

**Files:**
- Modify: `src/trading_agent_framework/memory/store.py`
- Test: `tests/memory/test_memory_store.py` (append)

**Interfaces:**
- Consumes: Tasks 2-3.
- Produces (used by Tasks 5-6):
  - `store.RETRIEVAL_POLICY: str`
  - `MemoryStore.search(query: str, *, limit: int = 10, kind: str | None = None, symbol: str | None = None, status: str | None = None, agent_name: str | None = None, model_call_id: str | None = None) -> dict[str, Any]`, returning `{"count": int, "retrieval_id": str, "results": list[lean item]}`
  - `MemoryStore.compact_state(held: Sequence[HeldPosition] = (), *, max_theses: int = 8, max_lessons: int = 8, max_chars_per_item: int = 900, update_open_thesis_outcomes: bool = True) -> dict[str, Any]`, returning keys `as_of, held_symbols, open_theses, validated_lessons, retrieval_policy`

- [ ] **Step 1: Write the failing tests**

Add `RETRIEVAL_POLICY` to the `trading_agent_framework.memory.store` import in `tests/memory/test_memory_store.py`, then append:

```python
# --- search -------------------------------------------------------------------------


def test_search_ranks_by_matching_terms_then_recency(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    one_term = store.remember("SPY looks heavy")
    clock.advance(60)
    two_terms_old = store.remember("SPY breadth is improving")
    clock.advance(60)
    two_terms_new = store.remember("Breadth thrust on SPY")
    store.remember("Gold miners")

    result = store.search("spy breadth")

    assert [item["id"] for item in result["results"]] == [
        two_terms_new["id"], two_terms_old["id"], one_term["id"],
    ]
    assert result["count"] == 3
    assert result["retrieval_id"].startswith("retrieval_")


def test_search_results_are_lean(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember("x" * 600, metadata={"secret": "not for the model"})
    [lean] = store.search("")["results"]
    assert set(lean) == {"id", "kind", "status", "symbol", "updated_at", "text"}
    assert (lean["id"], lean["updated_at"]) == (item["id"], "2026-09-14T10:00")
    assert len(lean["text"]) == 500


def test_search_filters_on_kind_symbol_and_status(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    spy = store.open_thesis("Long SPY", symbol="SPY")
    store.open_thesis("Long QQQ", symbol="QQQ")
    closed = store.open_thesis("Old SPY idea", symbol="SPY")
    store.close_thesis(closed["id"], "Invalidated")
    store.remember_lesson("SPY gaps fill", symbol="SPY")

    result = store.search("", kind="thesis", symbol="spy", status="open")

    assert [item["id"] for item in result["results"]] == [spy["id"]]


def test_search_includes_history_once_and_marks_it_superseded(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    thesis = store.open_thesis("Long SPY", symbol="SPY")
    clock.advance(60)
    store.update_thesis(thesis["id"], "Long SPY, raising target")
    opened_event = _events(store)[0]

    results = store.search("")["results"]

    assert [item["id"] for item in results] == [thesis["id"], opened_event["event_id"]]
    history = results[1]
    assert (history["status"], history["event_type"], history["memory_id"]) == (
        "superseded", "thesis.opened", thesis["id"],
    )
    assert history["text"] == "Long SPY"


def test_search_finds_history_only_events(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.record_warning("Ordered without data", symbol="SPY")
    order = Order(strategy_name="momentum", asset=Asset("SPY"), side=OrderSide.SELL, quantity=Decimal(5))
    store.record_order_submitted(order)

    [found] = store.search("", kind="order", status="submitted")["results"]
    assert (found["kind"], found["event_type"], found["memory_id"]) == (
        "order", "order.submitted", f"order_{order.identifier}",
    )
    [warning] = store.search("without data")["results"]
    assert (warning["kind"], warning["status"]) == ("warning", None)


def test_empty_query_returns_the_most_recent_first_up_to_the_limit(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    ids = []
    for text in ("first", "second", "third"):
        ids.append(store.remember(text)["id"])
        clock.advance(60)

    result = store.search("", limit=2)

    assert [item["id"] for item in result["results"]] == [ids[2], ids[1]]
    assert result["count"] == 3
    assert len(store.search("", limit=0)["results"]) == 1


def test_search_logs_a_retrieval(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.remember("SPY breadth", metadata={"symbol": "SPY"})
    store.remember("Gold", metadata={"symbol": "GLD"})

    result = store.search("spy", symbol="spy", limit=5, agent_name="analyst", model_call_id="call-3")

    [row] = memory_rows(store, "SELECT * FROM memory_retrievals")
    assert row["retrieval_id"] == result["retrieval_id"]
    assert (row["query"], row["kind"], row["symbol"], row["status"], row["result_limit"]) == (
        "spy", None, "SPY", None, 5,
    )
    assert (row["agent_name"], row["model_call_id"], row["strategy"]) == ("analyst", "call-3", "momentum")
    assert (row["timestamp"], row["wall_time"]) == ("2026-09-14T10:00:00-04:00", "2026-09-14T14:00:05Z")
    selected = [item["id"] for item in result["results"]]
    assert json.loads(row["selected_ids_json"]) == selected
    assert json.loads(row["candidate_ids_json"]) == selected
    assert row["rendered_text"] == "memory SPY active: SPY breadth"


# --- compact state ----------------------------------------------------------------------


def test_compact_state_lists_open_theses_and_validated_lessons(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    spy = store.open_thesis("Long SPY on breadth", symbol="SPY")
    qqq = store.open_thesis("Long QQQ", symbol="QQQ")
    store.close_thesis(qqq["id"], "Stopped out")
    lesson = store.remember_lesson("Don't chase gaps", outcome={"validated": True})
    store.remember_lesson("Maybe fade the open")

    held = [
        HeldPosition("spy", Decimal(10), Decimal("512.3")),
        HeldPosition("IWM", Decimal(0), Decimal("200")),
    ]
    state = store.compact_state(held, update_open_thesis_outcomes=False)

    assert state == {
        "as_of": "2026-09-14T10:00:00-04:00",
        "held_symbols": ["SPY"],
        "open_theses": [
            {
                "id": spy["id"], "kind": "thesis", "status": "open", "symbol": "SPY",
                "updated_at": "2026-09-14T10:00", "text": "Long SPY on breadth",
            }
        ],
        "validated_lessons": [
            {
                "id": lesson["id"], "kind": "lesson", "status": "validated", "symbol": None,
                "updated_at": "2026-09-14T10:00", "text": "Don't chase gaps",
            }
        ],
        "retrieval_policy": RETRIEVAL_POLICY,
    }
    assert not any(e["event_type"] == "thesis.outcome_observed" for e in _events(store))


def test_compact_state_limits_and_truncates(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    ids = []
    for index in range(3):
        ids.append(store.open_thesis(f"Thesis {index} " + "z" * 50, symbol="SPY")["id"])
        clock.advance(60)

    state = store.compact_state(max_theses=2, max_chars_per_item=30)

    assert [item["id"] for item in state["open_theses"]] == [ids[2], ids[1]]
    assert all(len(item["text"]) == 30 for item in state["open_theses"])


def test_compact_state_observes_held_open_theses_once_per_day(tmp_path: Path) -> None:
    clock = FakeClock(MEMORY_START)
    store = make_memory_store(tmp_path, clock)
    spy = store.open_thesis("Long SPY", symbol="SPY")
    store.open_thesis("Long QQQ", symbol="QQQ")
    held = [HeldPosition("SPY", Decimal(10), Decimal("512.3"))]

    store.compact_state(held)
    store.compact_state(held)

    observed = [e for e in _events(store) if e["event_type"] == "thesis.outcome_observed"]
    assert len(observed) == 1
    [event] = observed
    assert (event["subject_type"], event["subject_id"], event["symbol"]) == ("thesis", spy["id"], "SPY")
    assert event["text"] == (
        "Observed open thesis for SPY: quantity=10 last_price=512.3 market_value=5123.0"
    )
    assert json.loads(event["payload_json"]) == {
        "kind": "thesis_outcome",
        "status": "observed",
        "thesis_id": spy["id"],
        "symbol": "SPY",
        "position": {
            "symbol": "SPY", "quantity": "10", "last_price": "512.3", "market_value": "5123.0",
        },
        "observed_date": "2026-09-14",
    }
    thesis_now = store.get(spy["id"])
    assert thesis_now is not None and thesis_now["status"] == "open"

    clock.advance(24 * 3600)
    store.compact_state(held)
    assert sum(e["event_type"] == "thesis.outcome_observed" for e in _events(store)) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/memory/test_memory_store.py -q`
Expected: collection error, `ImportError: cannot import name 'RETRIEVAL_POLICY'`.

- [ ] **Step 3: Implement search and compact state**

In `src/trading_agent_framework/memory/store.py`:

1. After `DB_FILE_NAME = "memory.sqlite"`, add:

```python
RETRIEVAL_POLICY = (
    "If you hold a symbol and plan to add, reduce, or sell it, "
    "call search_memory for its open thesis first."
)
```

2. After `_UPSERT_INDEX`, add:

```python
_INSERT_RETRIEVAL = """
INSERT INTO memory_retrievals (
    retrieval_id, timestamp, wall_time, strategy, agent_name, model_call_id,
    query, kind, symbol, status, result_limit, candidate_ids_json,
    selected_ids_json, rendered_text, schema_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
```

3. In the `# --- reads` section, after `get`, add:

```python
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: str | None = None,
        symbol: str | None = None,
        status: str | None = None,
        agent_name: str | None = None,
        model_call_id: str | None = None,
    ) -> dict[str, Any]:
        """Lumibot's search: current memories plus event history, ranked by matching query terms.

        Every call is logged to `memory_retrievals`.
        """
        query_text = str(query or "").strip()
        normalized_symbol = records.normalize_symbol(symbol)
        max_results = max(int(limit), 1)
        with self._transaction() as conn:
            index_rows = conn.execute(
                """
                SELECT * FROM memory_index
                WHERE (? IS NULL OR kind = ?) AND (? IS NULL OR symbol = ?)
                  AND (? IS NULL OR status = ?)
                """,
                (kind, kind, normalized_symbol, normalized_symbol, status, status),
            ).fetchall()
            event_rows = conn.execute(
                """
                SELECT * FROM memory_events
                WHERE subject_id IS NOT NULL AND (? IS NULL OR symbol = ?)
                  AND event_id NOT IN (SELECT latest_event_id FROM memory_index)
                """,
                (normalized_symbol, normalized_symbol),
            ).fetchall()
            candidates = [records.index_item(dict(row)) for row in index_rows]
            for row in event_rows:
                item = records.event_item(dict(row))
                if (kind is None or item["kind"] == kind) and (
                    status is None or item["status"] == status
                ):
                    candidates.append(item)
            ranked = records.rank(candidates, records.query_terms(query_text))
            selected = ranked[:max_results]
            retrieval_id = records.new_retrieval_id()
            conn.execute(
                _INSERT_RETRIEVAL,
                (
                    retrieval_id,
                    self._timestamp(),
                    self._wall_time(),
                    self.strategy_name,
                    agent_name,
                    model_call_id,
                    query_text,
                    kind,
                    normalized_symbol,
                    status,
                    max_results,
                    records.json_dumps([item["id"] for item in ranked]),
                    records.json_dumps([item["id"] for item in selected]),
                    records.render_retrieval_text(selected),
                    SCHEMA_VERSION,
                ),
            )
        return {
            "count": len(ranked),
            "retrieval_id": retrieval_id,
            "results": [
                records.lean_item(item, max_chars=records.SEARCH_TEXT_CHARS) for item in selected
            ],
        }

    def compact_state(
        self,
        held: Sequence[HeldPosition] = (),
        *,
        max_theses: int = 8,
        max_lessons: int = 8,
        max_chars_per_item: int = 900,
        update_open_thesis_outcomes: bool = True,
    ) -> dict[str, Any]:
        """The memory summary injected into agent prompts: open theses, validated lessons and the
        retrieval policy. Also records a daily `thesis.outcome_observed` for each held open thesis.
        """
        held_by_symbol: dict[str, HeldPosition] = {}
        for position in held:
            held_symbol = records.normalize_symbol(position.symbol)
            if held_symbol and position.quantity:
                held_by_symbol[held_symbol] = position
        with self._transaction() as conn:
            theses = self._latest_index_items(conn, "thesis", "open", max_theses)
            lessons = self._latest_index_items(conn, "lesson", "validated", max_lessons)
            if update_open_thesis_outcomes:
                self._observe_open_theses(conn, theses, held_by_symbol)
        return {
            "as_of": self._timestamp(),
            "held_symbols": sorted(held_by_symbol),
            "open_theses": [records.lean_item(i, max_chars=max_chars_per_item) for i in theses],
            "validated_lessons": [
                records.lean_item(i, max_chars=max_chars_per_item) for i in lessons
            ],
            "retrieval_policy": RETRIEVAL_POLICY,
        }
```

4. In the `# --- internals` section, add:

```python
    @staticmethod
    def _latest_index_items(
        conn: sqlite3.Connection, kind: str, status: str, limit: int
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM memory_index WHERE kind = ? AND status = ?
            ORDER BY updated_at DESC LIMIT ?
            """,
            (kind, status, max(int(limit), 1)),
        ).fetchall()
        return [records.index_item(dict(row)) for row in rows]

    def _observe_open_theses(
        self,
        conn: sqlite3.Connection,
        theses: Sequence[Mapping[str, Any]],
        held_by_symbol: Mapping[str, HeldPosition],
    ) -> None:
        """At most one `thesis.outcome_observed` per held open thesis per (strategy-time) day."""
        today = self._timestamp()[:10]
        for thesis in theses:
            position = held_by_symbol.get(thesis["symbol"]) if thesis["symbol"] else None
            if position is None:
                continue
            already_observed = conn.execute(
                """
                SELECT 1 FROM memory_events
                WHERE event_type = 'thesis.outcome_observed' AND subject_id = ?
                  AND timestamp LIKE ?
                LIMIT 1
                """,
                (thesis["id"], f"{today}%"),
            ).fetchone()
            if already_observed is not None:
                continue
            symbol = thesis["symbol"]
            metadata = {
                "thesis_id": thesis["id"],
                "symbol": symbol,
                "status": "observed",
                "position": {
                    "symbol": symbol,
                    "quantity": position.quantity,
                    "last_price": position.last_price,
                    "market_value": position.market_value,
                },
                "observed_date": today,
            }
            self._append_event(
                conn,
                event_type="thesis.outcome_observed",
                subject_type="thesis",
                subject_id=thesis["id"],
                text=(
                    f"Observed open thesis for {symbol}: quantity={position.quantity} "
                    f"last_price={position.last_price} market_value={position.market_value}"
                ),
                symbol=symbol,
                metadata=metadata,
                payload={"kind": "thesis_outcome", "status": "observed", **metadata},
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/memory/ -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/memory/store.py tests/memory/test_memory_store.py
git commit -m "Task 4: add memory search with a retrieval log, and the compact memory state"
```

(Add the attribution trailer lines to the message.)

---

### Task 5: The 9 memory tools and `agent_call_context`

**Files:**
- Create: `src/trading_agent_framework/memory/tools.py`
- Modify: `src/trading_agent_framework/memory/__init__.py`
- Test: `tests/memory/test_memory_tools.py`

**Interfaces:**
- Consumes: `MemoryStore` public methods from Tasks 2-4, `MemoryValidationError`.
- Produces (the future agent layer relies on these):
  - `tools.MAX_SEARCH_LIMIT = 20`
  - `memory_tools(store: MemoryStore) -> list[Callable[..., dict[str, Any]]]`: the 9 functions in lumibot's registration order: `remember, search_memory, remember_proposal, remember_risk_note, remember_decision, remember_lesson, open_thesis, update_thesis, close_thesis`. `remember_decision` carries the function attribute `mutates_trading = True`.
  - `agent_call_context(agent_name: str | None = None, model_call_id: str | None = None)`: a context manager.
  - `trading_agent_framework.memory` exports `HeldPosition, MemoryStore, agent_call_context, memory_db_path, memory_tools`.

- [ ] **Step 1: Write the failing tests**

Create `tests/memory/test_memory_tools.py`:

```python
from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import make_memory_store, memory_rows

from trading_agent_framework.memory import agent_call_context, memory_tools
from trading_agent_framework.memory.store import MemoryStore
from trading_agent_framework.memory.tools import MAX_SEARCH_LIMIT
from trading_agent_framework.utils.errors import MemoryStoreError

_REQUIRED = inspect.Parameter.empty

# Lumibot's tool signatures (lumibot/components/agents/builtins.py `_bind_*`), in registration order.
_LUMIBOT_SIGNATURES = {
    "remember": [("text", _REQUIRED), ("kind", "memory"), ("tags", None)],
    "search_memory": [
        ("query", _REQUIRED), ("limit", 10), ("kind", None), ("symbol", None), ("status", None),
    ],
    "remember_proposal": [("text", _REQUIRED), ("symbol", None), ("action", None), ("tags", None)],
    "remember_risk_note": [("text", _REQUIRED), ("symbol", None), ("tags", None)],
    "remember_decision": [("text", _REQUIRED), ("symbol", None), ("action", None)],
    "remember_lesson": [("text", _REQUIRED), ("symbol", None)],
    "open_thesis": [("text", _REQUIRED), ("symbol", None), ("tags", None)],
    "update_thesis": [("thesis_id", _REQUIRED), ("text", _REQUIRED)],
    "close_thesis": [("thesis_id", _REQUIRED), ("text", _REQUIRED)],
}


def _tools(store: MemoryStore) -> dict[str, Callable[..., dict[str, Any]]]:
    return {tool.__name__: tool for tool in memory_tools(store)}


# --- shape: what the agent layer will bind ------------------------------------------------


def test_the_nine_lumibot_tools_in_registration_order(tmp_path: Path) -> None:
    names = [tool.__name__ for tool in memory_tools(make_memory_store(tmp_path))]
    assert names == list(_LUMIBOT_SIGNATURES)


def test_tool_parameters_match_lumibot(tmp_path: Path) -> None:
    for name, tool in _tools(make_memory_store(tmp_path)).items():
        parameters = inspect.signature(tool).parameters.values()
        assert [(p.name, p.default) for p in parameters] == _LUMIBOT_SIGNATURES[name], name


def test_tools_have_real_annotations_and_one_line_docstrings(tmp_path: Path) -> None:
    tools = _tools(make_memory_store(tmp_path))
    for name, tool in tools.items():
        signature = inspect.signature(tool)
        assert all(p.annotation is not _REQUIRED for p in signature.parameters.values()), name
        assert signature.return_annotation == dict[str, Any], name
        docstring = inspect.getdoc(tool)
        assert docstring and "\n" not in docstring and len(docstring) <= 120, name
    remember = inspect.signature(tools["remember"]).parameters
    assert remember["text"].annotation is str
    assert remember["tags"].annotation == list[str] | None
    assert inspect.signature(tools["search_memory"]).parameters["limit"].annotation is int


def test_only_remember_decision_mutates_trading(tmp_path: Path) -> None:
    tools = _tools(make_memory_store(tmp_path))
    flagged = {name for name, tool in tools.items() if getattr(tool, "mutates_trading", False)}
    assert flagged == {"remember_decision"}


# --- behaviour ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kwargs", "kind", "status"),
    [
        ("remember", {"text": "note"}, "memory", "active"),
        ("remember", {"text": "note", "kind": "macro", "tags": ["fed"]}, "macro", "active"),
        ("remember_proposal", {"text": "buy dip", "symbol": "spy", "action": "buy"}, "proposal", "proposed"),
        ("remember_risk_note", {"text": "earnings", "symbol": "SPY"}, "risk_note", "active"),
        ("remember_decision", {"text": "bought", "symbol": "SPY", "action": "buy"}, "decision", "recorded"),
        ("remember_lesson", {"text": "don't chase", "symbol": "SPY"}, "lesson", "proposed"),
        ("open_thesis", {"text": "long SPY", "symbol": "SPY", "tags": ["macro"]}, "thesis", "open"),
    ],
)
def test_write_tools_return_a_lean_summary(
    tmp_path: Path, name: str, kwargs: dict[str, Any], kind: str, status: str
) -> None:
    store = make_memory_store(tmp_path)
    result = _tools(store)[name](**kwargs)
    assert result == {"id": result["id"], "kind": kind, "status": status}
    stored = store.get(result["id"])
    assert stored is not None
    assert stored["text"] == kwargs["text"]


def test_thesis_lifecycle_through_the_tools(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    tools = _tools(store)
    thesis_id = tools["open_thesis"]("Long SPY", symbol="SPY")["id"]
    assert tools["update_thesis"](thesis_id, "Long SPY, raise target") == {
        "id": thesis_id, "kind": "thesis", "status": "open",
    }
    assert tools["close_thesis"](thesis_id, "Target hit") == {
        "id": thesis_id, "kind": "thesis", "status": "closed",
    }
    assert tools["close_thesis"](thesis_id, "again") == {
        "error": f"thesis '{thesis_id}' is not open (status: closed)",
    }


@pytest.mark.parametrize(
    ("name", "args", "error"),
    [
        ("update_thesis", ("thesis_00000000", "x"), "unknown thesis_id 'thesis_00000000'"),
        ("remember", ("   ",), "text must be a non-empty string"),
        ("open_thesis", ("x", None, "macro"), "tags must be a list of strings"),
    ],
)
def test_validation_errors_come_back_as_an_error_dict(
    tmp_path: Path, name: str, args: tuple[Any, ...], error: str
) -> None:
    assert _tools(make_memory_store(tmp_path))[name](*args) == {"error": error}


def test_database_failures_still_raise(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    store.db_path.unlink()
    store.db_path.mkdir()
    with pytest.raises(MemoryStoreError):
        _tools(store)["remember"]("note")


def test_search_memory_returns_lean_results_and_clamps_the_limit(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    for index in range(25):
        store.remember(f"note {index}")
    search = _tools(store)["search_memory"]

    result = search("note", limit=100)
    assert set(result) == {"count", "retrieval_id", "results"}
    assert (result["count"], len(result["results"])) == (25, MAX_SEARCH_LIMIT)
    assert len(search("note", limit=0)["results"]) == 1
    rows = memory_rows(store, "SELECT result_limit FROM memory_retrievals ORDER BY rowid")
    limits = [row["result_limit"] for row in rows]
    assert limits == [MAX_SEARCH_LIMIT, 1]


# --- provenance ------------------------------------------------------------------------


def test_agent_call_context_is_recorded_on_writes_and_searches(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    tools = _tools(store)
    with agent_call_context(agent_name="analyst", model_call_id="call-9"):
        tools["remember"]("inside")
        tools["search_memory"]("inside")
    tools["remember"]("outside")

    events = memory_rows(store, "SELECT agent_name, model_call_id FROM memory_events ORDER BY sequence")
    assert events == [
        {"agent_name": "analyst", "model_call_id": "call-9"},
        {"agent_name": None, "model_call_id": None},
    ]
    assert memory_rows(store, "SELECT agent_name, model_call_id FROM memory_retrievals") == [
        {"agent_name": "analyst", "model_call_id": "call-9"},
    ]


def test_agent_call_context_nests_and_restores(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    remember = _tools(store)["remember"]
    with agent_call_context(agent_name="committee"):
        with agent_call_context(agent_name="analyst", model_call_id="call-1"):
            remember("inner")
        remember("outer")

    events = memory_rows(store, "SELECT agent_name, model_call_id FROM memory_events ORDER BY sequence")
    assert events == [
        {"agent_name": "analyst", "model_call_id": "call-1"},
        {"agent_name": "committee", "model_call_id": None},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/memory/test_memory_tools.py -q`
Expected: collection error, `ImportError: cannot import name 'agent_call_context' from 'trading_agent_framework.memory'`.

- [ ] **Step 3: Create `tools.py`**

Create `src/trading_agent_framework/memory/tools.py`. **Do not add `from __future__ import annotations`**: the agent layer reads real annotation objects from `inspect.signature`, and the tests assert them.

```python
"""Lumibot's 9 memory tools as plain typed functions; the agent layer wraps them as LLM tools.

Each docstring is a single line on purpose: it becomes the tool description sent to the model
on every call. Validation errors come back as `{"error": ...}` so the model can correct itself.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from trading_agent_framework.memory.store import MemoryStore
from trading_agent_framework.utils.errors import MemoryValidationError

MAX_SEARCH_LIMIT = 20


@dataclass(frozen=True, slots=True)
class _AgentCall:
    agent_name: str | None = None
    model_call_id: str | None = None


_CURRENT_CALL: ContextVar[_AgentCall] = ContextVar("memory_agent_call", default=_AgentCall())


@contextmanager
def agent_call_context(
    agent_name: str | None = None, model_call_id: str | None = None
) -> Iterator[None]:
    """Attribute memory tool calls made inside this block to `agent_name` / `model_call_id`."""
    token = _CURRENT_CALL.set(_AgentCall(agent_name, model_call_id))
    try:
        yield
    finally:
        _CURRENT_CALL.reset(token)


def _provenance() -> dict[str, str | None]:
    call = _CURRENT_CALL.get()
    return {"agent_name": call.agent_name, "model_call_id": call.model_call_id}


def _write(action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        item = action()
    except MemoryValidationError as exc:
        return {"error": str(exc)}
    return {"id": item["id"], "kind": item["kind"], "status": item["status"]}


def memory_tools(store: MemoryStore) -> list[Callable[..., dict[str, Any]]]:
    """Lumibot's memory tools bound to `store`, in lumibot's registration order."""

    def remember(text: str, kind: str = "memory", tags: list[str] | None = None) -> dict[str, Any]:
        """Store a memory or note."""
        return _write(lambda: store.remember(text, kind=kind, tags=tags, **_provenance()))

    def search_memory(
        query: str,
        limit: int = 10,
        kind: str | None = None,
        symbol: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Search memories, lessons, decisions and theses; filter by symbol/status to find a held position's open thesis."""
        return store.search(
            query,
            limit=min(max(int(limit), 1), MAX_SEARCH_LIMIT),
            kind=kind,
            symbol=symbol,
            status=status,
            **_provenance(),
        )

    def remember_proposal(
        text: str,
        symbol: str | None = None,
        action: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a non-final trade idea (not an executed decision)."""
        return _write(
            lambda: store.remember_proposal(
                text, symbol=symbol, action=action, tags=tags, **_provenance()
            )
        )

    def remember_risk_note(
        text: str, symbol: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Record a compact risk note or bear case."""
        return _write(
            lambda: store.remember_risk_note(text, symbol=symbol, tags=tags, **_provenance())
        )

    def remember_decision(
        text: str, symbol: str | None = None, action: str | None = None
    ) -> dict[str, Any]:
        """Record an actual trading decision."""
        return _write(
            lambda: store.remember_decision(text, symbol=symbol, action=action, **_provenance())
        )

    def remember_lesson(text: str, symbol: str | None = None) -> dict[str, Any]:
        """Record a compact lesson for future runs."""
        return _write(lambda: store.remember_lesson(text, symbol=symbol, **_provenance()))

    def open_thesis(
        text: str, symbol: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Open an investment thesis."""
        return _write(lambda: store.open_thesis(text, symbol=symbol, tags=tags, **_provenance()))

    def update_thesis(thesis_id: str, text: str) -> dict[str, Any]:
        """Update an open thesis."""
        return _write(lambda: store.update_thesis(thesis_id, text, **_provenance()))

    def close_thesis(thesis_id: str, text: str) -> dict[str, Any]:
        """Close a thesis and record its outcome."""
        return _write(lambda: store.close_thesis(thesis_id, text, **_provenance()))

    # Lumibot's `mutates_trading` tool metadata: the agent layer decides which agents get it.
    vars(remember_decision)["mutates_trading"] = True

    return [
        remember,
        search_memory,
        remember_proposal,
        remember_risk_note,
        remember_decision,
        remember_lesson,
        open_thesis,
        update_thesis,
        close_thesis,
    ]
```

- [ ] **Step 4: Export the public API**

Replace the contents of `src/trading_agent_framework/memory/__init__.py` with:

```python
"""Agent memory: lumibot's SQLite-backed memory store and its 9 agent tools."""

from trading_agent_framework.memory.store import HeldPosition, MemoryStore, memory_db_path
from trading_agent_framework.memory.tools import agent_call_context, memory_tools

__all__ = [
    "HeldPosition",
    "MemoryStore",
    "agent_call_context",
    "memory_db_path",
    "memory_tools",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/memory/ -q`
Expected: all pass.

- [ ] **Step 6: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green. (The long `search_memory` docstring line is under ruff's 200-character limit. If `uv check` flags the `vars(...)` assignment, run it on `tools.py` alone and report the exact diagnostic; don't silence it with a blanket ignore.)

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/memory/tools.py src/trading_agent_framework/memory/__init__.py tests/memory/test_memory_tools.py
git commit -m "Task 5: add lumibot's 9 memory tools and agent call provenance"
```

(Add the attribution trailer lines to the message.)

---

### Task 6: `Strategy.memory` and documentation

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py` (imports, `__init__`, new property after `indicators`)
- Modify: `CLAUDE.md`
- Test: `tests/core/test_strategy_memory.py`

**Interfaces:**
- Consumes: `MemoryStore`, `memory_db_path` (Task 2), `find_project_root` (`config/env.py`, existing).
- Produces: `Strategy.memory -> MemoryStore` (lazy, cached per trading mode, fresh for backtesting).

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_strategy_memory.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


def _strategy(
    tmp_path: Path, mode: TradingMode = TradingMode.PAPER, name: str = "momentum"
) -> Strategy:
    return Strategy(
        FakeBroker(FakeClock(_START), strategy_name=name), mode=mode, project_root=tmp_path
    )


def test_memory_is_opened_lazily_and_cached(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert not (tmp_path / "memory").exists()

    store = strategy.memory

    assert store.db_path == tmp_path / "memory" / "momentum" / "paper" / "memory.sqlite"
    assert store.db_path.is_file()
    assert store.strategy_name == "momentum"
    assert strategy.memory is store


@pytest.mark.parametrize("mode", list(TradingMode))
def test_each_trading_mode_has_its_own_database(tmp_path: Path, mode: TradingMode) -> None:
    assert _strategy(tmp_path, mode).memory.db_path.parent == tmp_path / "memory" / "momentum" / mode.value


def test_paper_and_live_memory_survive_a_new_run(tmp_path: Path) -> None:
    for mode in (TradingMode.PAPER, TradingMode.LIVE):
        memory_id = _strategy(tmp_path, mode).memory.remember("keep me")["id"]
        assert _strategy(tmp_path, mode).memory.get(memory_id) is not None


def test_backtesting_memory_starts_empty_on_every_run(tmp_path: Path) -> None:
    first_run = _strategy(tmp_path, TradingMode.BACKTESTING)
    memory_id = first_run.memory.remember("would leak into the next backtest")["id"]
    assert first_run.memory.get(memory_id) is not None

    second_run = _strategy(tmp_path, TradingMode.BACKTESTING)
    assert second_run.memory.get(memory_id) is None


def test_memory_reopens_when_the_trading_mode_changes(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER)
    paper = strategy.memory
    strategy.trading_mode = TradingMode.LIVE

    live = strategy.memory

    assert live is not paper
    assert live.db_path.parent.name == "live"


def test_memory_events_are_stamped_with_strategy_time(tmp_path: Path) -> None:
    item = _strategy(tmp_path).memory.remember("x")
    assert item["created_at"] == _START.isoformat()


def test_the_memory_folder_name_is_sanitised(tmp_path: Path) -> None:
    store = _strategy(tmp_path, name="orb v2/beta").memory
    assert store.db_path.parent.parent == tmp_path / "memory" / "orb_v2_beta"
    assert store.strategy_name == "orb v2/beta"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_strategy_memory.py -q`
Expected: failures with `AttributeError: 'Strategy' object has no attribute 'memory'`.

- [ ] **Step 3: Add the property**

In `src/trading_agent_framework/core/strategy.py`:

1. Change `from trading_agent_framework.config.env import TradingMode` to:

```python
from trading_agent_framework.config.env import TradingMode, find_project_root
```

and add, after the `trading_agent_framework.entities.quote` import:

```python
from trading_agent_framework.memory.store import MemoryStore, memory_db_path
```

2. In `Strategy.__init__`, after `self._indicators: Indicators | None = None`, add:

```python
        self._memory: MemoryStore | None = None
        self._memory_mode: TradingMode | None = None
```

3. After the `indicators` property, add:

```python
    @property
    def memory(self) -> MemoryStore:
        """Agent memory for this strategy and trading mode (lumibot's `strategy.memory`).

        Opened lazily at `memory/<strategy>/<mode>/memory.sqlite`; a backtest run starts empty.
        """
        if self._memory is None or self._memory_mode is not self.trading_mode:
            root = self.project_root if self.project_root is not None else find_project_root()
            self._memory = MemoryStore(
                memory_db_path(root, self.name, self.trading_mode),
                strategy_name=self.name,
                now=self.clock.now,
                fresh=self.is_backtesting,
            )
            self._memory_mode = self.trading_mode
        return self._memory
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/core/ -q`
Expected: all pass.

- [ ] **Step 5: Document the memory layer in `CLAUDE.md`**

In `## Architecture`, add this bullet after the `core/` bullet:

```markdown
- `memory/` -- agent memory (lumibot's `strategy.memory`): `records.py` (**pure**: ids, JSON, lean items, search scoring), `store.py` (`MemoryStore`, the only SQLite code: append-only `memory_events`, the `memory_index` projection, `memory_retrievals`; one DB per strategy and mode at `memory/<strategy>/<mode>/memory.sqlite`), `tools.py` (lumibot's 9 memory tools as plain typed functions, plus `agent_call_context` for provenance). No LangChain here: the agent layer wraps the tools.
```

In `## Key patterns / gotchas`, add these bullets after the **Market data** bullet:

```markdown
- **Memory tools are token-budgeted.** They return lean payloads (`{id, kind, status}`, or lean search items without metadata) and have one-line docstrings, because both reach the LLM on every call. Weigh the token cost before adding a field. Validation problems come back as `{"error": ...}`; database failures raise `MemoryStoreError`. `tools.py` deliberately has no `from __future__ import annotations` (the agent layer reads real annotations).
- **`MemoryStore` never knows `Strategy`.** It gets time from injected `now` / `wall_clock` callables (the strategy passes `clock.now`) and held positions as `HeldPosition` arguments to `compact_state`. Keep it that way so the store stays testable and backtest-clock friendly.
- **Backtesting memory is wiped** at the first `strategy.memory` access of every backtest run (`fresh=True`), so one run's lessons can't leak into the next. Paper and live memory are never wiped.
```

- [ ] **Step 6: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/core/strategy.py tests/core/test_strategy_memory.py CLAUDE.md
git commit -m "Task 6: add Strategy.memory and document the memory layer"
```

(Add the attribution trailer lines to the message. Do not stage `TODO.md`.)

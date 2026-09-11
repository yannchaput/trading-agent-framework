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

"""SQLite store for agent-call telemetry: one row per model call, one database per strategy and mode.

Lives next to `memory.sqlite` (`memory/<strategy>/<mode>/llm_stats.sqlite`) but as its own file, so it
can be deleted when it grows without touching agent memory. Like `MemoryStore` it is wiped when a
backtest starts (`fresh=True`) and kept across paper/live restarts; every row carries the `run_id`
(the run directory name) that wrote it. This is the only module that writes this database.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from trading_agent_framework.agents.telemetry import CallRecord
from trading_agent_framework.memory.store import memory_db_path
from trading_agent_framework.utils.errors import LLMStatsError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import TradingMode

logger = logging.getLogger(__name__)

DB_FILE_NAME = "llm_stats.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    agent TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms REAL NOT NULL,
    tool_calls INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls (run_id);
"""

_INSERT = """
INSERT INTO llm_calls (
    run_id, ts, agent, model, input_tokens, output_tokens, reasoning_tokens, total_tokens, latency_ms, tool_calls
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def llm_stats_db_path(project_root: Path, strategy_name: str, mode: TradingMode) -> Path:
    """`<project_root>/memory/<strategy>/<mode>/llm_stats.sqlite`, beside (but apart from) `memory.sqlite`."""
    return memory_db_path(project_root, strategy_name, mode).with_name(DB_FILE_NAME)


class LLMStatsStore:
    """Append-only log of model calls. Each write opens a connection and closes it (WAL journal), so the
    file can be read by the dashboard or `sqlite3` while a run is writing."""

    def __init__(self, db_path: Path, *, run_id: str, fresh: bool = False) -> None:
        self.db_path = db_path
        self.run_id = run_id
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if fresh:
                if db_path.exists():
                    logger.info("LLM stats database %s holds data from a previous run; deleting it to start fresh.", db_path)
                for suffix in ("", "-wal", "-shm"):
                    db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        except OSError as exc:
            raise LLMStatsError(f"Cannot prepare LLM stats database {db_path}: {exc}") from exc
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def record(self, call: CallRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                _INSERT,
                (
                    self.run_id, call.ts.isoformat(), call.agent, call.model, call.input_tokens, call.output_tokens,
                    call.reasoning_tokens, call.total_tokens, call.latency_ms, call.tool_calls,
                ),
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection in autocommit mode, always closed; SQLite errors become `LLMStatsError`."""
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            yield conn
        except sqlite3.Error as exc:
            raise LLMStatsError(f"Error in LLM stats database {self.db_path}: {exc}") from exc
        finally:
            if conn is not None:
                conn.close()

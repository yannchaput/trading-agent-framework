from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_agent_framework.agents.stats_store import LLMStatsStore, llm_stats_db_path
from trading_agent_framework.agents.telemetry import CallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.utils.errors import LLMStatsError

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def _record(**overrides: object) -> CallRecord:
    fields: dict[str, object] = {
        "ts": T0, "agent": "trader", "model": "qwen3-8b", "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": 5, "total_tokens": 120, "latency_ms": 1234.5, "tool_calls": 2,
    }
    return CallRecord(**{**fields, **overrides})  # type: ignore[arg-type]


def _rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM llm_calls ORDER BY id").fetchall()
    finally:
        conn.close()


def test_the_database_sits_next_to_memory_sqlite_but_is_a_separate_file() -> None:
    path = llm_stats_db_path(Path("/proj"), "news_binary", TradingMode.BACKTESTING)

    assert path == Path("/proj/memory/news_binary/backtesting/llm_stats.sqlite")


def test_a_recorded_call_is_stored_with_every_field_and_its_run_id(tmp_path: Path) -> None:
    db = tmp_path / "memory" / "s" / "backtesting" / "llm_stats.sqlite"
    store = LLMStatsStore(db, run_id="2026-01-05_210000_backtesting")

    store.record(_record())

    [row] = _rows(db)
    assert dict(row) == {
        "id": 1, "run_id": "2026-01-05_210000_backtesting", "ts": "2026-01-05T21:00:00+00:00",
        "agent": "trader", "model": "qwen3-8b", "input_tokens": 100, "output_tokens": 20,
        "reasoning_tokens": 5, "total_tokens": 120, "latency_ms": 1234.5, "tool_calls": 2,
    }


def test_unreported_token_counts_are_stored_as_null_not_zero(tmp_path: Path) -> None:
    db = tmp_path / "llm_stats.sqlite"
    store = LLMStatsStore(db, run_id="r")

    store.record(_record(model=None, input_tokens=None, output_tokens=None, reasoning_tokens=None, total_tokens=None))

    [row] = _rows(db)
    assert (row["model"], row["input_tokens"], row["output_tokens"], row["reasoning_tokens"], row["total_tokens"]) == (None,) * 5


def test_reopening_keeps_earlier_rows_so_live_runs_accumulate(tmp_path: Path) -> None:
    db = tmp_path / "llm_stats.sqlite"
    LLMStatsStore(db, run_id="run_a").record(_record(agent="first"))

    LLMStatsStore(db, run_id="run_b").record(_record(agent="second"))

    assert [(row["run_id"], row["agent"]) for row in _rows(db)] == [("run_a", "first"), ("run_b", "second")]


def test_a_fresh_store_deletes_the_previous_runs_rows(tmp_path: Path) -> None:
    db = tmp_path / "llm_stats.sqlite"
    LLMStatsStore(db, run_id="old_run").record(_record())

    store = LLMStatsStore(db, run_id="new_run", fresh=True)

    assert _rows(db) == []
    store.record(_record(agent="new"))
    assert [row["run_id"] for row in _rows(db)] == ["new_run"]


def test_an_unusable_location_raises_llm_stats_error_not_a_raw_os_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")

    with pytest.raises(LLMStatsError):
        LLMStatsStore(blocker / "memory" / "llm_stats.sqlite", run_id="r")


def test_a_write_failure_raises_llm_stats_error_not_a_raw_sqlite_error(tmp_path: Path) -> None:
    db = tmp_path / "llm_stats.sqlite"
    store = LLMStatsStore(db, run_id="r")
    for suffix in ("", "-wal", "-shm"):
        db.with_name(db.name + suffix).unlink(missing_ok=True)
    db.mkdir()  # the database path is now a directory, so SQLite cannot open it

    with pytest.raises(LLMStatsError):
        store.record(_record())

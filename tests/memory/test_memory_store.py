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

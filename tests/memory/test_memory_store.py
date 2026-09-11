from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.fakes import MEMORY_START, FakeClock, make_memory_store, memory_rows

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
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


def test_explicit_symbol_argument_wins_over_a_conflicting_metadata_symbol(tmp_path: Path) -> None:
    store = make_memory_store(tmp_path)
    item = store.remember_proposal("Buy SPY", symbol="SPY", metadata={"symbol": "QQQ"})
    assert item["symbol"] == "SPY"
    assert _events(store)[-1]["symbol"] == "SPY"


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

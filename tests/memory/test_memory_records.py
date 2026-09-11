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

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
    # memory_tools returns real functions (each always has __name__ in CPython); a structural
    # Callable type can't express that, so ty can't verify it statically.
    return {tool.__name__: tool for tool in memory_tools(store)}  # ty: ignore[unresolved-attribute]


# --- shape: what the agent layer will bind ------------------------------------------------


def test_the_nine_lumibot_tools_in_registration_order(tmp_path: Path) -> None:
    names = [tool.__name__ for tool in memory_tools(make_memory_store(tmp_path))]  # ty: ignore[unresolved-attribute]
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

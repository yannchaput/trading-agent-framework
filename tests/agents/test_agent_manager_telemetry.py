from __future__ import annotations

import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.stats_store import LLMStatsStore
from trading_agent_framework.agents.telemetry import CallRecord
from trading_agent_framework.utils.errors import AgentError, LLMStatsError

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://x/v1", api_key="k", default_model="m"))


def _two_call_model() -> FakeToolCallingChatModel:
    """Call 1 asks for one tool (usage 100 in / 20 out / 5 reasoning); call 2 answers (usage 130 in / 10 out)."""
    return FakeToolCallingChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="add", args={"a": 1, "b": 2}, id="call_1")],
                    usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "output_token_details": {"reasoning": 5}},
                ),
                AIMessage(content="3", usage_metadata={"input_tokens": 130, "output_tokens": 10, "total_tokens": 140}),
            ]
        )
    )


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    def record(self, call: CallRecord) -> None:
        self.calls.append(call)


class BrokenStore:
    def record(self, call: CallRecord) -> None:
        raise LLMStatsError("database is locked")


def test_every_model_call_is_counted_with_its_usage_and_requested_tool_calls() -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0)
    handle = manager.create(name="trader", system_prompt="x", model=_two_call_model(), tools=[add])

    handle.run("add 1 and 2")

    summary = manager.telemetry_summary()["trader"]
    assert (summary["calls"], summary["tool_calls"]) == (2, 1)
    assert (summary["input_tokens"], summary["output_tokens"], summary["reasoning_tokens"], summary["total_tokens"]) == (230, 30, 5, 260)


def test_each_call_is_written_to_the_store_as_it_happens_stamped_with_the_injected_clock() -> None:
    store = RecordingStore()
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=store)  # type: ignore[arg-type]
    handle = manager.create(name="trader", system_prompt="x", model=_two_call_model(), tools=[add])

    handle.run("add 1 and 2")

    assert [(c.ts, c.agent, c.input_tokens, c.tool_calls) for c in store.calls] == [(T0, "trader", 100, 1), (T0, "trader", 130, 0)]


def test_calls_land_in_a_real_sqlite_store(tmp_path: Path) -> None:
    db = tmp_path / "llm_stats.sqlite"
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=LLMStatsStore(db, run_id="run_1"))
    handle = manager.create(name="trader", system_prompt="x", model=_two_call_model(), tools=[add])

    handle.run("add 1 and 2")

    rows = sqlite3.connect(db).execute("SELECT run_id, agent, input_tokens, tool_calls FROM llm_calls ORDER BY id").fetchall()
    assert rows == [("run_1", "trader", 100, 1), ("run_1", "trader", 130, 0)]


def test_the_model_name_is_recorded_when_the_chat_model_has_one() -> None:
    class NamedModel(FakeToolCallingChatModel):
        model_name: str = "qwen3-8b"

    store = RecordingStore()
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=store)  # type: ignore[arg-type]
    handle = manager.create(name="trader", system_prompt="x", model=NamedModel(messages=iter([AIMessage("hi")])))

    handle.run("hello")

    assert [c.model for c in store.calls] == ["qwen3-8b"]
    assert manager.telemetry_summary()["trader"]["model"] == "qwen3-8b"


def test_latency_is_the_real_elapsed_time_of_the_model_call() -> None:
    class SlowModel(FakeToolCallingChatModel):
        def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    store = RecordingStore()
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=store)  # type: ignore[arg-type]
    handle = manager.create(name="trader", system_prompt="x", model=SlowModel(messages=iter([AIMessage("hi")])))

    handle.run("hello")

    assert 50 <= store.calls[0].latency_ms < 5000


def test_without_enabling_nothing_is_recorded() -> None:
    manager = _manager()
    handle = manager.create(name="trader", system_prompt="x", model=_two_call_model(), tools=[add])

    result = handle.run("add 1 and 2")

    assert result.output == "3"
    assert manager.telemetry_summary() == {}


def test_a_failing_model_call_is_not_recorded_and_still_raises_agent_error() -> None:
    class FailingModel(FakeToolCallingChatModel):
        def _generate(self, messages: list[Any], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> Any:
            raise RuntimeError("server down")

    store = RecordingStore()
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=store)  # type: ignore[arg-type]
    handle = manager.create(name="trader", system_prompt="x", model=FailingModel(messages=iter([])))

    with pytest.raises(AgentError, match="trader"):
        handle.run("hello")

    assert store.calls == []
    assert manager.telemetry_summary() == {}


def test_a_store_failure_is_logged_but_never_breaks_the_agent_run(caplog: pytest.LogCaptureFixture) -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0, store=BrokenStore())  # type: ignore[arg-type]
    handle = manager.create(name="trader", system_prompt="x", model=_two_call_model(), tools=[add])

    with caplog.at_level(logging.WARNING):
        result = handle.run("add 1 and 2")

    assert result.output == "3"
    assert manager.telemetry_summary()["trader"]["calls"] == 2  # the in-memory totals survive the store failure
    assert "database is locked" in caplog.text


def test_enabling_telemetry_after_an_agent_exists_is_rejected_because_it_would_silently_miss_it() -> None:
    manager = _manager()
    manager.create(name="trader", system_prompt="x", model=FakeToolCallingChatModel(messages=iter([AIMessage("hi")])))

    with pytest.raises(ValueError, match="before"):
        manager.enable_telemetry(now=lambda: T0)

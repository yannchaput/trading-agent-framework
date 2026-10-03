from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.messages import AIMessage
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


class _WarmModel(FakeToolCallingChatModel):
    temperature: float | None = 0.3


def test_a_temperature_reaches_the_chat_model() -> None:
    assert _manager()._resolve_model(None, None, 0.3).temperature == 0.3


def test_without_a_temperature_nothing_is_sent_and_the_server_default_applies() -> None:
    chat_model = _manager()._resolve_model(None, None, None)

    assert chat_model.temperature is None
    assert "temperature" not in chat_model._default_params


def test_a_prebuilt_model_is_used_as_is_whatever_the_temperature() -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))

    assert _manager()._resolve_model(model, None, 0.9) is model


def test_the_telemetry_summary_records_each_agents_temperature() -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0)
    manager.create(name="warm", system_prompt="x", model=_WarmModel(messages=iter([AIMessage(content="ok")]))).run("go")
    manager.create(name="plain", system_prompt="x", model=FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))).run("go")

    summary = manager.telemetry_summary()

    assert summary["warm"]["temperature"] == 0.3
    assert summary["plain"]["temperature"] is None

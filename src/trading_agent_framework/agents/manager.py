"""AgentManager / AgentHandle: build and run LangChain agents (`create_agent`) for a strategy.

The only module in this package that imports `langchain` / `langchain_openai`, and only inside
method bodies -- importing this module (or `trading_agent_framework.core.strategy`, which imports
it) does not pull LangChain into a strategy that never calls `strategy.agents.create(...)`.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.results import AgentRunResult, parse_agent_messages
from trading_agent_framework.agents.telemetry import CallRecord, summarize, usage_from_message
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.utils.errors import AgentError, ConfigurationError, LLMStatsError
from trading_agent_framework.utils.log import ColorLogger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from trading_agent_framework.agents.stats_store import LLMStatsStore

logger = ColorLogger(logging.getLogger(__name__), "AgentRunResult")


class AgentHandle:
    """A single named agent, built once by `AgentManager.create` and run repeatedly."""

    def __init__(self, name: str, agent: Any) -> None:
        self.name = name
        self._agent = agent

    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None) -> AgentRunResult:
        message = task_prompt if context is None else f"{task_prompt}\n\nContext:\n{context}"
        try:
            with agent_call_context(run_id=uuid.uuid4().hex):  # one id per run: memory tools dedupe repeats inside it
                raw_result = self._agent.invoke({"messages": [{"role": "user", "content": message}]})
            logger.log_debug(f"Model returned this raw messages: {str(raw_result)}")
            return parse_agent_messages(raw_result["messages"])
        except Exception as exc:
            raise AgentError(f"agent {self.name!r} failed: {exc}") from exc


class AgentManager:
    """`strategy.agents`: a named registry of LangChain agents built from this framework's tools."""

    def __init__(self, credentials_source: Callable[[], LLMCredentials]) -> None:
        self._credentials_source = credentials_source
        self._agents: dict[str, AgentHandle] = {}
        self._telemetry_now: Callable[[], datetime] | None = None
        self._telemetry_store: LLMStatsStore | None = None
        self._calls: list[CallRecord] = []

    def enable_telemetry(self, *, now: Callable[[], datetime], store: LLMStatsStore | None = None) -> None:
        """Record every model call of the agents created from now on (tokens, latency, requested tool calls).

        `now` stamps each call (the strategy clock, so a backtest logs simulated time); `store` also
        persists each call. Must run before `create()`: an agent built earlier has no recording hook.
        """
        if self._agents:
            raise ValueError("enable_telemetry() must be called before any agent is created.")
        self._telemetry_now = now
        self._telemetry_store = store

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        """Per-agent totals of the calls recorded so far (empty when telemetry is off or nothing ran)."""
        return summarize(self._calls)

    def _telemetry_middleware(self, agent_name: str) -> Any:
        from langchain.agents.middleware import wrap_model_call

        @wrap_model_call
        def record_call(request: Any, handler: Callable[[Any], Any]) -> Any:
            started = time.perf_counter()
            response = handler(request)
            self._record_call(agent_name, request.model, response, (time.perf_counter() - started) * 1000)
            return response

        return record_call

    def _record_call(self, agent_name: str, chat_model: Any, response: Any, latency_ms: float) -> None:
        assert self._telemetry_now is not None
        messages = getattr(response, "result", None) or [response]
        ai_message = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai"), None)
        usage = usage_from_message(ai_message)
        call = CallRecord(
            ts=self._telemetry_now(),
            agent=agent_name,
            model=getattr(chat_model, "model_name", None),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            tool_calls=len(getattr(ai_message, "tool_calls", None) or []),
        )
        self._calls.append(call)
        if self._telemetry_store is not None:
            try:
                self._telemetry_store.record(call)
            except LLMStatsError as exc:
                logger.log_warning(f"agent telemetry not persisted: {exc}")  # observational: never break the agent run

    def create(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str | BaseChatModel | None = None,
        tools: Sequence[Callable[..., Any] | BaseTool] | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentHandle:
        """Build and register a named agent; raises `ValueError` if `name` is already taken.

        `timeout_seconds` defaults to `None` -- no request timeout at all, deliberately, since a
        reasoning-capable local model can legitimately take minutes per call and a
        framework-imposed default would silently break that use case. It only applies when
        `model` is a string resolved through `LLMCredentials`; a pre-built `BaseChatModel`
        instance passed as `model` configures its own timeout and ignores this parameter.
        """
        if name in self._agents:
            raise ValueError(f"Agent with name {name!r} already exists.")

        chat_model = self._resolve_model(model, timeout_seconds)
        try:
            from langchain.agents import create_agent

            middleware = [self._telemetry_middleware(name)] if self._telemetry_now is not None else []
            agent = create_agent(model=chat_model, tools=list(tools or []), system_prompt=system_prompt, middleware=middleware)
        except Exception as exc:
            raise AgentError(f"failed to create agent {name!r}: {exc}") from exc

        handle = AgentHandle(name, agent)
        self._agents[name] = handle
        return handle

    def _resolve_model(self, model: str | BaseChatModel | None, timeout_seconds: float | None) -> Any:
        if model is not None and not isinstance(model, str):
            return model  # already a chat-model instance -- used as-is, LLMCredentials and timeout_seconds untouched

        try:
            from langchain_openai import ChatOpenAI
            from pydantic import SecretStr
        except Exception as exc:
            raise AgentError(f"could not import langchain_openai: {exc}") from exc

        credentials = self._credentials_source()
        model_id = model if model is not None else credentials.default_model
        if not model_id:
            raise ConfigurationError("No model id given and LLM_MODEL is not set; pass model=... explicitly or set LLM_MODEL in the strategy's env file")
        try:
            return ChatOpenAI(
                model=model_id,
                base_url=credentials.base_url,
                api_key=SecretStr(credentials.api_key),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise AgentError(f"could not build chat model {model_id!r}: {exc}") from exc

    def __getitem__(self, name: str) -> AgentHandle:
        return self._agents[name]

    def __contains__(self, name: str) -> bool:
        return name in self._agents

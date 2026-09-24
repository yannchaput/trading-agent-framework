"""AgentManager / AgentHandle: build and run LangChain agents (`create_agent`) for a strategy.

The only module in this package that imports `langchain` / `langchain_openai`, and only inside
method bodies -- importing this module (or `trading_agent_framework.core.strategy`, which imports
it) does not pull LangChain into a strategy that never calls `strategy.agents.create(...)`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.results import AgentRunResult, parse_agent_messages
from trading_agent_framework.agents.telemetry import CallRecord, summarize, usage_from_message
from trading_agent_framework.memory.tools import agent_call_context, current_forced_tool
from trading_agent_framework.utils.errors import AgentError, ConfigurationError, LLMStatsError
from trading_agent_framework.utils.log import ColorLogger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from trading_agent_framework.agents.stats_store import LLMStatsStore

logger = ColorLogger(logging.getLogger(__name__), "AgentRunResult")


def _parse_pseudo_tool_call(content: Any, valid_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    """Recover a `{"name": ..., "arguments": {...}}` call a model wrote as plain text instead of a real one.

    Observed with local models under long contexts (many tool round trips, large tool results): instead
    of using the API's structured tool-calling channel for its next step, the model writes out what
    should have been the call as JSON in the message content, sometimes with trailing junk -- a stray
    `</tool_call>` tag, an extra `}`. `raw_decode` reads just the first complete JSON object at `start`
    and ignores anything after it, so that junk doesn't stop the parse. Only a `name` naming a tool this
    agent actually has is accepted, so ordinary prose that happens to contain a `{` (or JSON naming
    something else entirely) is left alone.
    """
    if not isinstance(content, str):
        return None
    start = content.find("{")
    if start == -1:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(content, start)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    name, arguments = parsed.get("name"), parsed.get("arguments")
    if not isinstance(name, str) or name not in valid_names or not isinstance(arguments, dict):
        return None
    return name, arguments


class AgentHandle:
    """A single named agent, built once by `AgentManager.create` and run repeatedly."""

    def __init__(self, name: str, agent: Any) -> None:
        self.name = name
        self._agent = agent

    def run(
        self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None
    ) -> AgentRunResult:
        """`run_id` defaults to a fresh id (one per run: memory tools dedupe repeats inside it).

        A caller that needs to re-prompt the *same* logical run -- e.g. a corrective follow-up turn
        after the model skipped a mandatory tool call -- passes back the id from the first call, so a
        per-run gate (like `news_builtin`'s search-before-decide grounding) doesn't ask it to re-ground
        itself, and per-run dedupe still treats them as one run.

        `force_tool`, when given, forces this run's first model turn to call that tool via the LLM
        API's `tool_choice` (see `AgentManager._forced_tool_choice_middleware`), instead of relying on
        the model to follow a text instruction -- for a corrective retry after the model narrated a
        decision without ever calling the tool that should have recorded it. Later turns in the same
        run are left unforced.
        """
        message = task_prompt if context is None else f"{task_prompt}\n\nContext:\n{context}"
        run_id = run_id or uuid.uuid4().hex
        try:
            with agent_call_context(run_id=run_id, force_tool=force_tool):
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

    def _tool_call_repair_middleware(self) -> Any:
        """Middleware for every agent: repairs a pseudo tool call caught by `_parse_pseudo_tool_call`.

        Registered first (outermost) in `create()`'s middleware list, so it sees the model's raw response
        last, after any inner middleware (telemetry) has already recorded it -- telemetry keeps reporting
        what the model actually produced, while this repairs the message the graph acts on. Runs on every
        agent unconditionally: this is a local-model reliability issue (see `_parse_pseudo_tool_call`),
        not something specific to one strategy's tools.
        """
        from langchain.agents.middleware import ModelResponse, wrap_model_call
        from langchain_core.messages import AIMessage
        from langchain_core.messages.tool import tool_call as make_tool_call

        @wrap_model_call
        def repair_pseudo_tool_call(request: Any, handler: Callable[[Any], Any]) -> Any:
            response = handler(request)
            ai_message = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
            if ai_message is None or ai_message.tool_calls:
                return response
            valid_names = {tool.name for tool in request.tools if hasattr(tool, "name")}
            parsed = _parse_pseudo_tool_call(ai_message.content, valid_names)
            if parsed is None:
                return response
            name, arguments = parsed
            logger.log_warning(f"model wrote a {name!r} call as plain text instead of a real tool call; repairing it so it actually runs: {arguments}")
            repaired = AIMessage(content="", tool_calls=[make_tool_call(name=name, args=arguments, id=uuid.uuid4().hex)])
            return ModelResponse(result=[repaired], structured_response=response.structured_response)

        return repair_pseudo_tool_call

    def _forced_tool_choice_middleware(self) -> Any:
        """Middleware for every agent: forces `current_forced_tool()`'s tool as this run's first model turn.

        Registered unconditionally, like `_tool_call_repair_middleware`; a no-op whenever
        `AgentHandle.run` was called without `force_tool` (the common case). `request.messages`
        accumulates every message of the current `.invoke()`, so "no `AIMessage` yet" identifies the
        run's first model turn -- the only one forced. Once the model has answered once (whether with
        a tool call or plain text), later turns in the same run get the model's normal free choice, so
        a forced `remember_decision` call can still be followed by an ordinary closing summary.
        """
        from langchain.agents.middleware import wrap_model_call
        from langchain_core.messages import AIMessage

        @wrap_model_call
        def force_tool_choice(request: Any, handler: Callable[[Any], Any]) -> Any:
            tool_name = current_forced_tool()
            if tool_name is not None and not any(isinstance(m, AIMessage) for m in request.messages):
                request = request.override(tool_choice=tool_name)
            return handler(request)

        return force_tool_choice

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

            middleware = [self._tool_call_repair_middleware(), self._forced_tool_choice_middleware()]
            if self._telemetry_now is not None:
                middleware.append(self._telemetry_middleware(name))
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

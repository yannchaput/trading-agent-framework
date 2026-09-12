"""AgentManager / AgentHandle: build and run LangChain agents (`create_agent`) for a strategy.

The only module in this package that imports `langchain` / `langchain_openai`, and only inside
method bodies -- importing this module (or `trading_agent_framework.core.strategy`, which imports
it) does not pull LangChain into a strategy that never calls `strategy.agents.create(...)`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.results import AgentRunResult, parse_agent_messages
from trading_agent_framework.utils.errors import AgentError, ConfigurationError

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool


class AgentHandle:
    """A single named agent, built once by `AgentManager.create` and run repeatedly."""

    def __init__(self, name: str, agent: Any) -> None:
        self.name = name
        self._agent = agent

    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None) -> AgentRunResult:
        message = task_prompt if context is None else f"{task_prompt}\n\nContext:\n{context}"
        try:
            raw_result = self._agent.invoke({"messages": [{"role": "user", "content": message}]})
        except Exception as exc:
            raise AgentError(f"agent {self.name!r} failed: {exc}") from exc
        return parse_agent_messages(raw_result["messages"])


class AgentManager:
    """`strategy.agents`: a named registry of LangChain agents built from this framework's tools."""

    def __init__(self, credentials_source: Callable[[], LLMCredentials]) -> None:
        self._credentials_source = credentials_source
        self._agents: dict[str, AgentHandle] = {}

    def create(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str | BaseChatModel | None = None,
        tools: Sequence[Callable[..., Any] | BaseTool] | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentHandle:
        if name in self._agents:
            raise ValueError(f"Agent with name {name!r} already exists.")

        chat_model = self._resolve_model(model, timeout_seconds)
        try:
            from langchain.agents import create_agent

            agent = create_agent(model=chat_model, tools=list(tools or []), system_prompt=system_prompt)
        except Exception as exc:
            raise AgentError(f"failed to create agent {name!r}: {exc}") from exc

        handle = AgentHandle(name, agent)
        self._agents[name] = handle
        return handle

    def _resolve_model(self, model: str | BaseChatModel | None, timeout_seconds: float | None) -> Any:
        if model is not None and not isinstance(model, str):
            return model  # already a chat-model instance -- used as-is, LLMCredentials untouched

        try:
            from langchain_openai import ChatOpenAI
        except Exception as exc:
            raise AgentError(f"could not import langchain_openai: {exc}") from exc

        credentials = self._credentials_source()
        model_id = model if model is not None else credentials.default_model
        if not model_id:
            raise ConfigurationError(
                "No model id given and LLM_MODEL is not set; "
                "pass model=... explicitly or set LLM_MODEL in the strategy's env file"
            )
        try:
            return ChatOpenAI(
                model=model_id,
                base_url=credentials.base_url,
                api_key=credentials.api_key,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise AgentError(f"could not build chat model {model_id!r}: {exc}") from exc

    def __getitem__(self, name: str) -> AgentHandle:
        return self._agents[name]

    def __contains__(self, name: str) -> bool:
        return name in self._agents

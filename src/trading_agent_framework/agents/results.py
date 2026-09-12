"""Pure result types for the agent framework, and the LangChain message-list parser.

Deliberately duck-typed instead of importing `langchain_core.messages`: an assistant message is
anything with a `tool_calls` attribute (even an empty list), a tool-result message is anything
with a `tool_call_id` attribute. Neither attribute is unique to this framework's use of LangChain,
but the combination reliably distinguishes `AIMessage` / `ToolMessage` / everything else without
this module depending on `langchain_core` at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: str


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    output: str
    tool_calls: list[ToolCallRecord]


def parse_agent_messages(messages: Sequence[Any]) -> AgentRunResult:
    """Turn a `create_agent(...).invoke(...)["messages"]` list into a lean `AgentRunResult`."""
    tool_results: dict[str, str] = {}
    for message in messages:
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None:
            tool_results[tool_call_id] = _as_text(message.content)

    tool_calls: list[ToolCallRecord] = []
    output = ""
    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if calls is None:
            continue
        for call in calls:
            tool_calls.append(
                ToolCallRecord(
                    name=call["name"],
                    args=dict(call["args"]),
                    result=tool_results.get(call["id"], ""),
                )
            )
        text = getattr(message, "text", None)
        output = text if isinstance(text, str) else _as_text(message.content)

    return AgentRunResult(output=output, tool_calls=tool_calls)


def _as_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)

"""Pure result types for the agent framework, and the LangChain message-list parser.

Deliberately duck-typed instead of importing `langchain_core.messages`: an assistant message is
anything whose `type` is `"ai"` (its `tool_calls`, if any, are collected separately), a tool-result
message is anything with a `tool_call_id` attribute. This distinguishes `AIMessage` / `ToolMessage` /
everything else without this module depending on `langchain_core` at all.
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
    """Turn a `create_agent(...).invoke(...)["messages"]` list into a lean `AgentRunResult`.

    `output` is every non-empty `AIMessage` text, in order, separated by `---`.
    """
    # First pass: index each tool's result by the id of the call that produced it.
    tool_results: dict[str, str] = {}
    for message in messages:
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None:
            tool_results[tool_call_id] = _as_text(message.content)

    # Second pass: pair each requested tool call with its result, and gather the AI texts.
    tool_calls: list[ToolCallRecord] = []
    ai_texts: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                ToolCallRecord(
                    name=call["name"],
                    args=dict(call["args"]),
                    result=tool_results.get(call["id"], ""),
                )
            )
        if getattr(message, "type", None) == "ai":
            text = getattr(message, "text", None)
            text = (text if isinstance(text, str) else _as_text(message.content)).strip()
            if text:
                ai_texts.append(text)

    return AgentRunResult(output="\n\n---\n\n".join(ai_texts), tool_calls=tool_calls)


def _as_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)

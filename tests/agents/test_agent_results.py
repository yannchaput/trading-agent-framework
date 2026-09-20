from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord, parse_agent_messages


def _human(content: str) -> SimpleNamespace:
    return SimpleNamespace(type="human", content=content)


def _ai(content: str, tool_calls: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="ai", content=content, tool_calls=tool_calls or [])


def _tool_result(tool_call_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool", content=content, tool_call_id=tool_call_id)


def test_text_only_reply_has_no_tool_calls() -> None:
    messages = [_human("hi"), _ai("hello there")]

    result = parse_agent_messages(messages)

    assert result == AgentRunResult(output="hello there", tool_calls=[])


def test_a_single_tool_call_is_captured_with_its_result() -> None:
    messages = [
        _human("remember this"),
        _ai("", tool_calls=[{"name": "remember", "args": {"text": "note"}, "id": "call_1"}]),
        _tool_result("call_1", '{"id": "memory_abc", "kind": "memory", "status": "active"}'),
        _ai("Remembered."),
    ]

    result = parse_agent_messages(messages)

    assert result == AgentRunResult(
        output="Remembered.",
        tool_calls=[
            ToolCallRecord(
                name="remember",
                args={"text": "note"},
                result='{"id": "memory_abc", "kind": "memory", "status": "active"}',
            )
        ],
    )


def test_multiple_tool_calls_are_captured_in_order() -> None:
    messages = [
        _human("do two things"),
        _ai(
            "",
            tool_calls=[
                {"name": "first", "args": {"x": 1}, "id": "call_1"},
                {"name": "second", "args": {"y": 2}, "id": "call_2"},
            ],
        ),
        _tool_result("call_1", "1"),
        _tool_result("call_2", "2"),
        _ai("Done."),
    ]

    result = parse_agent_messages(messages)

    assert [call.name for call in result.tool_calls] == ["first", "second"]
    assert [call.result for call in result.tool_calls] == ["1", "2"]
    assert result.output == "Done."


def test_a_tool_call_with_no_matching_tool_message_gets_empty_result() -> None:
    messages = [
        _human("go"),
        _ai("", tool_calls=[{"name": "orphan", "args": {}, "id": "call_missing"}]),
        _ai("Done anyway."),
    ]

    result = parse_agent_messages(messages)

    assert result.tool_calls == [ToolCallRecord(name="orphan", args={}, result="")]


def test_every_ai_message_becomes_output_separated_by_a_rule() -> None:
    messages = [_human("go"), _ai("thinking..."), _ai("final answer")]

    result = parse_agent_messages(messages)

    assert result.output == "thinking...\n\n---\n\nfinal answer"


def test_empty_and_whitespace_only_ai_messages_are_skipped() -> None:
    messages = [_human("go"), _ai("  "), _ai("only reply"), _ai("")]

    result = parse_agent_messages(messages)

    assert result.output == "only reply"


def test_tool_messages_never_leak_into_output() -> None:
    messages = [
        _human("go"),
        _ai("", tool_calls=[{"name": "remember", "args": {}, "id": "call_1"}]),
        _tool_result("call_1", '{"id": "memory_abc"}'),
    ]

    result = parse_agent_messages(messages)

    assert result.output == ""


def test_an_ai_message_without_a_tool_calls_attribute_is_still_captured() -> None:
    messages = [
        _human("go"),
        _ai("first", tool_calls=[{"name": "remember", "args": {}, "id": "call_1"}]),
        _tool_result("call_1", '{"id": "memory_abc"}'),
        SimpleNamespace(type="ai", content="last"),
    ]

    result = parse_agent_messages(messages)

    assert result.output == "first\n\n---\n\nlast"
    assert [call.name for call in result.tool_calls] == ["remember"]


def test_no_ai_messages_gives_empty_output_and_no_tool_calls() -> None:
    result = parse_agent_messages([_human("hello")])

    assert result == AgentRunResult(output="", tool_calls=[])


def _ai_with_list_content(list_content: list[dict[str, Any]], text: str) -> SimpleNamespace:
    return SimpleNamespace(type="ai", content=list_content, tool_calls=[], text=text)


def test_list_content_uses_the_text_property_not_the_raw_content() -> None:
    messages = [
        _human("describe this"),
        _ai_with_list_content([{"type": "text", "text": "hello"}], text="hello"),
    ]

    result = parse_agent_messages(messages)

    assert result.output == "hello"

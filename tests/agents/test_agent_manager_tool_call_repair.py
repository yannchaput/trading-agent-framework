"""A local LLM sometimes writes out what should have been a tool call as plain JSON text in the

message content instead of using the model API's real tool-calling channel (observed with
`remember_decision` in `news_binary` after a long run of grounding-gate retries: the final
`AIMessage` had empty `tool_calls` and content like `{"name": "remember_decision", "arguments":
{...}}`, sometimes with a trailing `</tool_call>` tag or a stray `}`). LangChain treats that as the
agent's final answer and never executes the tool, so the call is silently lost.

`AgentManager` repairs this generically for every agent it builds: if a model turn ends with no real
tool call but content that parses as `{"name": <a tool this agent has>, "arguments": {...}}`, it is
rewritten into a real tool call before the graph continues, so the tool actually runs.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager, _is_glm_model, _parse_glm_tool_call


def _credentials() -> LLMCredentials:
    return LLMCredentials(base_url="http://localhost:8000/v1", api_key="key", default_model="qwen3-8b")


def _manager() -> AgentManager:
    return AgentManager(_credentials)


def _fake_model(messages: list[AIMessage]) -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(messages=iter(messages))


_recorded: list[str] = []


def remember_decision(text: str) -> dict[str, str]:
    """Record a decision."""
    _recorded.append(text)
    return {"id": "decision_abc123", "status": "recorded"}


def setup_function() -> None:
    _recorded.clear()


def test_a_pseudo_tool_call_written_as_plain_text_is_repaired_and_actually_runs() -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model(
            [
                AIMessage(content='{"name": "remember_decision", "arguments": {"text": "KEEP risk_on"}}'),
                AIMessage(content="Recorded."),
            ]
        ),
    )

    result = handle.run("decide")

    assert _recorded == ["KEEP risk_on"]
    assert [(c.name, c.args) for c in result.tool_calls] == [("remember_decision", {"text": "KEEP risk_on"})]
    assert '"error"' not in result.tool_calls[0].result


def test_trailing_junk_after_the_json_does_not_block_the_repair() -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model(
            [
                AIMessage(content='{"name": "remember_decision", "arguments": {"text": "KEEP risk_on"}}\n</tool_call>'),
                AIMessage(content="Recorded."),
            ]
        ),
    )

    handle.run("decide")

    assert _recorded == ["KEEP risk_on"]


def test_a_normal_real_tool_call_is_unaffected() -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model(
            [
                AIMessage(content="", tool_calls=[ToolCall(name="remember_decision", args={"text": "KEEP risk_on"}, id="call_1")]),
                AIMessage(content="Recorded."),
            ]
        ),
    )

    result = handle.run("decide")

    assert _recorded == ["KEEP risk_on"]
    assert [c.name for c in result.tool_calls] == ["remember_decision"]


def test_json_looking_text_naming_an_unknown_tool_is_left_as_plain_text() -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model([AIMessage(content='{"name": "not_a_real_tool", "arguments": {"text": "x"}}')]),
    )

    result = handle.run("decide")

    assert _recorded == []
    assert result.tool_calls == []
    assert result.output == '{"name": "not_a_real_tool", "arguments": {"text": "x"}}'


def test_ordinary_prose_final_answers_pass_through_untouched() -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model(
            [
                AIMessage(content="", tool_calls=[ToolCall(name="remember_decision", args={"text": "KEEP risk_on"}, id="call_1")]),
                AIMessage(content="KEEP current regime 'risk_on'."),
            ]
        ),
    )

    result = handle.run("decide")

    assert result.output == "KEEP current regime 'risk_on'."


def test_the_repair_is_logged_as_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    manager = _manager()
    handle = manager.create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_fake_model(
            [
                AIMessage(content='{"name": "remember_decision", "arguments": {"text": "KEEP risk_on"}}'),
                AIMessage(content="Recorded."),
            ]
        ),
    )

    with caplog.at_level(logging.WARNING):
        handle.run("decide")

    assert "repair" in caplog.text.lower()
    assert "remember_decision" in caplog.text


# --- GLM's native tool-call format, leaked into the content ---------------------------------------
# glm-4.7-flash sometimes writes its own template instead of a JSON call:
# `remember_decision<arg_key>text</arg_key><arg_value>...</arg_value></tool_call>`, often glued to prose
# (observed in news_binary live/paper/backtest logs). Only repaired when the model is a GLM model.

_GLM_CONTENT = (
    "The regime stays defensive. 本次未完成切换。remember_decision<arg_key>text</arg_key><arg_value>No regime flip; remain in SHV.</arg_value>"
    "<arg_key>symbol</arg_key><arg_value>SHV</arg_value><arg_key>action</arg_key><arg_value>hold</arg_value></tool_call>"
)


class _GlmFakeModel(FakeToolCallingChatModel):
    model_name: str = "glm-4.7-flash"


class _QwenFakeModel(FakeToolCallingChatModel):
    model_name: str = "qwen3-8b"


def remember_decision_with_extras(text: str, symbol: str | None = None, action: str | None = None) -> dict[str, str]:
    """Record a decision."""
    _recorded.append(text)
    return {"id": "decision_abc123", "status": "recorded"}


def test_the_glm_parser_reads_the_tool_name_glued_to_prose_and_every_argument() -> None:
    assert _parse_glm_tool_call(_GLM_CONTENT, {"remember_decision"}) == (
        "remember_decision",
        {"text": "No regime flip; remain in SHV.", "symbol": "SHV", "action": "hold"},
    )


def test_the_glm_parser_strips_whitespace_around_keys_and_values() -> None:
    content = "<tool_call>remember_decision\n<arg_key> text </arg_key>\n<arg_value>\nKEEP\n</arg_value>\n</tool_call>"

    assert _parse_glm_tool_call(content, {"remember_decision"}) == ("remember_decision", {"text": "KEEP"})


@pytest.mark.parametrize(
    "content",
    [
        # Starts mid-call: no tool name before the first <arg_key> (observed in a backtest log).
        "Trade plan: No trade.</arg_value><arg_key>text</arg_key><arg_value>DEFENSIVE</arg_value></tool_call>",
        "not_a_real_tool<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>",
        "remember_decision without any argument tags",
        "remember_decision<arg_key>text</arg_key> but the value tag never comes",
    ],
)
def test_the_glm_parser_rejects_what_it_cannot_read_unambiguously(content: str) -> None:
    assert _parse_glm_tool_call(content, {"remember_decision"}) is None


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [("glm-4.7-flash", True), ("zai-org/GLM-4.7-Flash", True), ("qwen3-8b", False), ("", False), (None, False)],
)
def test_glm_models_are_recognized_by_name(model_name: str | None, expected: bool) -> None:
    assert _is_glm_model(model_name) is expected


def test_a_glm_model_s_leaked_tool_call_is_repaired_and_actually_runs() -> None:
    handle = _manager().create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision_with_extras],
        model=_GlmFakeModel(messages=iter([AIMessage(content=_GLM_CONTENT.replace("remember_decision<", "remember_decision_with_extras<")), AIMessage(content="Recorded.")])),
    )

    result = handle.run("decide")

    assert _recorded == ["No regime flip; remain in SHV."]
    assert [c.name for c in result.tool_calls] == ["remember_decision_with_extras"]


def test_glm_tags_from_a_non_glm_model_are_left_as_plain_text() -> None:
    content = "remember_decision<arg_key>text</arg_key><arg_value>KEEP</arg_value></tool_call>"
    handle = _manager().create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_QwenFakeModel(messages=iter([AIMessage(content=content)])),
    )

    result = handle.run("decide")

    assert _recorded == []
    assert result.output == content


def test_a_glm_model_still_gets_the_json_repair() -> None:
    handle = _manager().create(
        name="trader",
        system_prompt="x",
        tools=[remember_decision],
        model=_GlmFakeModel(messages=iter([AIMessage(content='{"name": "remember_decision", "arguments": {"text": "KEEP"}}'), AIMessage(content="Recorded.")])),
    )

    handle.run("decide")

    assert _recorded == ["KEEP"]

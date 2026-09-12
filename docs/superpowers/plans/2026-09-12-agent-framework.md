# Agent Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give strategies a `strategy.agents` registry that creates and runs LangChain `create_agent`-based
"simple agents" (lumibot's `initialize()`-creates / `on_trading_iteration()`-runs pattern), with tools wired
in explicitly by the caller and nothing auto-included.

**Architecture:** New `agents/` package mirroring `memory/`'s pure/I-O split: `config.py` (pure, env-driven
`LLMCredentials`), `results.py` (pure, duck-typed LangChain-message-list parser into a lean
`AgentRunResult`), `manager.py` (`AgentManager`/`AgentHandle`, the only module that imports `langchain`/
`langchain_openai`, and only inside method bodies so strategies that never touch `.agents` never pay for it).
`Strategy.agents` is a new lazy property, same shape as `.memory`/`.indicators`.

**Tech Stack:** `langchain>=1.0,<2.0` (`create_agent`), `langchain-openai` (`ChatOpenAI` against any
OpenAI-compatible server — vLLM/llama.cpp/LM Studio), Python 3.14, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-agent-framework-design.md`

## Global Constraints

- `langchain` / `langchain_openai` / `langchain_core` are imported **only inside method bodies** in
  `agents/manager.py` — never at module level anywhere in `src/`. (Spec §3.1, §3.2.)
- No builtin tools, no MCP servers, no builtin skills, no `strategy.parameters` auto-population, no replay
  cache. Tools are passed in explicitly by the caller only. (Spec §1, §3.2.)
- Every exception from an SDK/library call must be wrapped in a `TradingFrameworkError` subclass before it
  reaches calling code — `AgentError` for anything from `langchain`/`langchain_openai`/`ChatOpenAI`;
  `ConfigurationError` for this framework's own config/usage checks (missing env var, no model given).
  (Spec §3.2; project-wide rule in `CLAUDE.md`.)
- Tests: no network, no `MagicMock` — hand-written fakes only (`tests/fakes.py`). (Spec §7; project-wide
  rule in `CLAUDE.md`.)
- `ToolCallRecord.result` is a `str` (the raw `ToolMessage.content`), never re-parsed into a Python object.
  (Spec §5.2 Amendment 2.)
- `AgentManager`/`AgentHandle` do not know `Strategy`. (Spec §3.2.)

---

### Task 1: `LLMCredentials` (env-driven LLM config)

**Files:**
- Create: `src/trading_agent_framework/agents/config.py`
- Test: `tests/agents/test_agent_config.py`
- Modify: `env/.env.example` (append LLM connection template)
- Modify: `pyproject.toml` (add `langchain`, `langchain-openai` to `dependencies`)

**Interfaces:**
- Consumes: `trading_agent_framework.utils.errors.ConfigurationError` (already exists).
- Produces: `LLMCredentials(base_url: str, api_key: str, default_model: str | None = None)` — frozen
  dataclass with slots; `LLMCredentials.from_env(env: Mapping[str, str] | None = None) -> LLMCredentials`
  classmethod. Later tasks (`AgentManager.__init__`) take a zero-arg `credentials_source:
  Callable[[], LLMCredentials]` and will call `LLMCredentials.from_env` (no args) through it.

- [ ] **Step 1: Add the new dependencies to `pyproject.toml`**

Open `pyproject.toml` and change the `dependencies` list from:

```toml
dependencies = [
    "alpaca-py>=0.44.0,<0.45",
    "pandas-ta-classic>=0.6.52,<0.7",
    "python-dotenv>=1.2",
]
```

to:

```toml
dependencies = [
    "alpaca-py>=0.44.0,<0.45",
    "langchain>=1.0,<2.0",
    "langchain-openai>=1.0",
    "pandas-ta-classic>=0.6.52,<0.7",
    "python-dotenv>=1.2",
]
```

Then run `uv sync` to install them.

Run: `uv sync`
Expected: completes without error; `uv run python -c "import langchain, langchain_openai"` succeeds.

- [ ] **Step 2: Append the LLM template to `env/.env.example`**

The file currently ends with:

```
# Only set this to false once you are certain you want to trade live.
ALPACA_IS_PAPER=true
```

Append (with a blank line before it):

```

# LLM connection for the agent framework (Strategy.agents / create_agent). Works
# against any OpenAI-compatible server -- vLLM, llama.cpp, LM Studio, or real
# OpenAI.
LLM_BASE_URL=http://localhost:8000/v1

# Default model id used when Strategy.agents.create(...) is called without an
# explicit `model=`. Individual agents may override this per-call.
LLM_MODEL=qwen3-8b

# Most local OpenAI-compatible servers ignore this value, but the OpenAI SDK
# requires a non-blank string to construct the client. Use any placeholder.
LLM_API_KEY=not-needed-for-local
```

- [ ] **Step 3: Write the failing tests**

Create `tests/agents/test_agent_config.py`:

```python
from __future__ import annotations

import pytest

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.utils.errors import ConfigurationError


def test_from_env_reads_base_url_api_key_and_default_model() -> None:
    env = {
        "LLM_BASE_URL": "http://localhost:8000/v1",
        "LLM_API_KEY": "key",
        "LLM_MODEL": "qwen3-8b",
    }

    credentials = LLMCredentials.from_env(env)

    assert credentials.base_url == "http://localhost:8000/v1"
    assert credentials.api_key == "key"
    assert credentials.default_model == "qwen3-8b"


def test_from_env_default_model_is_optional() -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_API_KEY": "key"}

    credentials = LLMCredentials.from_env(env)

    assert credentials.default_model is None


@pytest.mark.parametrize("blank_model", ["", "   "])
def test_from_env_blank_default_model_normalizes_to_none(blank_model: str) -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_API_KEY": "key", "LLM_MODEL": blank_model}

    credentials = LLMCredentials.from_env(env)

    assert credentials.default_model is None


@pytest.mark.parametrize("base_url_value", [None, "", "   "])
def test_from_env_missing_or_blank_base_url_raises(base_url_value: str | None) -> None:
    env = {"LLM_API_KEY": "key"}
    if base_url_value is not None:
        env["LLM_BASE_URL"] = base_url_value

    with pytest.raises(ConfigurationError):
        LLMCredentials.from_env(env)


@pytest.mark.parametrize("api_key_value", [None, "", "   "])
def test_from_env_missing_or_blank_api_key_raises(api_key_value: str | None) -> None:
    env = {"LLM_BASE_URL": "http://localhost:8000/v1"}
    if api_key_value is not None:
        env["LLM_API_KEY"] = api_key_value

    with pytest.raises(ConfigurationError):
        LLMCredentials.from_env(env)


def test_from_env_defaults_to_os_environ_when_no_mapping_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    credentials = LLMCredentials.from_env()

    assert credentials.base_url == "http://localhost:9000/v1"
    assert credentials.api_key == "env-key"


def test_repr_does_not_leak_the_api_key() -> None:
    credentials = LLMCredentials(base_url="http://localhost:8000/v1", api_key="supersecretkey")

    text = repr(credentials)

    assert "supersecretkey" not in text
    assert "LLMCredentials" in text
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_agent_config.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'trading_agent_framework.agents'`.

- [ ] **Step 5: Implement `agents/config.py`**

Create `src/trading_agent_framework/agents/config.py`:

```python
"""Env-driven LLM connection credentials for the agent framework (pure, no I/O)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_agent_framework.utils.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class LLMCredentials:
    base_url: str
    api_key: str = field(repr=False)
    default_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMCredentials:
        source = env if env is not None else os.environ

        base_url = source.get("LLM_BASE_URL")
        if not base_url or not base_url.strip():
            raise ConfigurationError("Missing or blank LLM_BASE_URL environment variable")

        api_key = source.get("LLM_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError("Missing or blank LLM_API_KEY environment variable")

        default_model = source.get("LLM_MODEL")
        if default_model is not None and not default_model.strip():
            default_model = None

        return cls(base_url=base_url, api_key=api_key, default_model=default_model)
```

Create `src/trading_agent_framework/agents/__init__.py` (just enough for the test import to resolve the
package; it is extended in Task 4):

```python
"""LangChain agent creation/execution for strategies (`strategy.agents`)."""

from trading_agent_framework.agents.config import LLMCredentials

__all__ = ["LLMCredentials"]
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/test_agent_config.py -v`
Expected: PASS (12 tests — two of the test functions are parametrized).

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/agents/ tests/agents/`
Expected: no errors.

```bash
git add pyproject.toml uv.lock env/.env.example \
    src/trading_agent_framework/agents/config.py \
    src/trading_agent_framework/agents/__init__.py \
    tests/agents/test_agent_config.py
git commit -m "Task 1: add langchain deps and env-driven LLMCredentials"
```

---

### Task 2: `AgentRunResult` / `ToolCallRecord` and the message-list parser

**Files:**
- Create: `src/trading_agent_framework/agents/results.py`
- Test: `tests/agents/test_agent_results.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `ToolCallRecord(name: str, args: dict[str, Any], result: str)` — frozen dataclass, slots.
  - `AgentRunResult(output: str, tool_calls: list[ToolCallRecord])` — frozen dataclass, slots.
  - `parse_agent_messages(messages: Sequence[Any]) -> AgentRunResult` — pure function. Task 3's
    `AgentHandle.run()` calls this on `agent.invoke(...)["messages"]`.

This module is duck-typed and imports nothing from `langchain`/`langchain_core` (not even under
`TYPE_CHECKING`): it identifies message roles purely by attribute presence —
`getattr(message, "tool_call_id", None) is not None` for a tool-result message, and
`getattr(message, "tool_calls", None) is not None` for an assistant message (verified against real
`langchain_core.messages.{HumanMessage,AIMessage,ToolMessage}` objects: `HumanMessage` has neither
attribute; `AIMessage` always has `.tool_calls`, even `[]`, but never `.tool_call_id`; `ToolMessage` always
has `.tool_call_id` but never `.tool_calls`). This keeps `agents/results.py` genuinely stdlib-only.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_agent_results.py`:

```python
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord, parse_agent_messages


def _human(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _ai(content: str, tool_calls: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _tool_result(tool_call_id: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_call_id=tool_call_id)


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


def test_only_the_last_ai_message_content_becomes_output() -> None:
    messages = [_human("go"), _ai("thinking..."), _ai("final answer")]

    result = parse_agent_messages(messages)

    assert result.output == "final answer"


def test_no_ai_messages_gives_empty_output_and_no_tool_calls() -> None:
    result = parse_agent_messages([_human("hello")])

    assert result == AgentRunResult(output="", tool_calls=[])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_agent_results.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'trading_agent_framework.agents.results'`.

- [ ] **Step 3: Implement `agents/results.py`**

Create `src/trading_agent_framework/agents/results.py`:

```python
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
        output = _as_text(message.content)

    return AgentRunResult(output=output, tool_calls=tool_calls)


def _as_text(content: object) -> str:
    return content if isinstance(content, str) else str(content)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/test_agent_results.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/agents/results.py tests/agents/test_agent_results.py`
Expected: no errors.

```bash
git add src/trading_agent_framework/agents/results.py tests/agents/test_agent_results.py
git commit -m "Task 2: add AgentRunResult/ToolCallRecord and the message-list parser"
```

---

### Task 3: `AgentManager` / `AgentHandle`

**Files:**
- Create: `src/trading_agent_framework/agents/manager.py`
- Modify: `src/trading_agent_framework/utils/errors.py` (add `AgentError`)
- Modify: `tests/fakes.py` (add `FakeToolCallingChatModel`)
- Test: `tests/agents/test_agent_manager.py`

**Interfaces:**
- Consumes:
  - `LLMCredentials` / `LLMCredentials.from_env` (Task 1, `agents/config.py`).
  - `AgentRunResult`, `parse_agent_messages` (Task 2, `agents/results.py`).
  - `ConfigurationError` (existing, `utils/errors.py`).
- Produces:
  - `AgentError(TradingFrameworkError)` in `utils/errors.py`.
  - `AgentHandle(name: str, agent: Any)` with `.name: str` and
    `.run(task_prompt: str, *, context: Mapping[str, Any] | None = None) -> AgentRunResult`.
  - `AgentManager(credentials_source: Callable[[], LLMCredentials])` with:
    - `.create(*, name: str, system_prompt: str, model: str | BaseChatModel | None = None,
      tools: Sequence[Callable[..., Any] | BaseTool] | None = None,
      timeout_seconds: float | None = None) -> AgentHandle`
    - `.__getitem__(name: str) -> AgentHandle`
    - `.__contains__(name: str) -> bool`
  - Task 4's `Strategy.agents` property constructs `AgentManager(LLMCredentials.from_env)`.

- [ ] **Step 1: Add `AgentError` to `utils/errors.py`**

Open `src/trading_agent_framework/utils/errors.py`. It currently ends with:

```python
class MemoryValidationError(MemoryStoreError, ValueError):
    """Raised when a memory write is rejected (bad text/kind/tags, unknown or non-open thesis)."""
```

Append:

```python


class AgentError(TradingFrameworkError):
    """Raised when building or running a LangChain agent fails (never a raw SDK exception)."""
```

- [ ] **Step 2: Add `FakeToolCallingChatModel` to `tests/fakes.py`**

`langchain_core`'s own `GenericFakeChatModel` returns a scripted sequence of messages but raises
`NotImplementedError` from `bind_tools()` — and `create_agent` always calls `model.bind_tools(...)` when the
agent has any tools. Add a one-line override.

Add this import near the top of `tests/fakes.py`, alongside the other third-party imports (after the
`alpaca.trading.requests` import block, before the `trading_agent_framework` imports):

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
```

Add this class at the end of the file, after `memory_rows`:

```python


class FakeToolCallingChatModel(GenericFakeChatModel):
    """`GenericFakeChatModel` with `bind_tools` stubbed out.

    The base class doesn't implement tool binding (it raises `NotImplementedError`), but
    `create_agent` always calls `model.bind_tools(...)` when the agent has tools. This fake
    scripts its responses directly via `messages`, so it doesn't need real tool-schema binding --
    returning `self` unchanged is enough.
    """

    def bind_tools(self, tools: object, **kwargs: object) -> "FakeToolCallingChatModel":
        return self
```

- [ ] **Step 3: Write the failing tests**

Create `tests/agents/test_agent_manager.py`:

```python
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.utils.errors import AgentError, ConfigurationError


def _credentials() -> LLMCredentials:
    return LLMCredentials(base_url="http://localhost:8000/v1", api_key="key", default_model="qwen3-8b")


def _fake_model(messages: list[AIMessage]) -> FakeToolCallingChatModel:
    return FakeToolCallingChatModel(messages=iter(messages))


def _manager() -> AgentManager:
    return AgentManager(_credentials)


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_create_returns_a_handle_and_registers_it_by_name() -> None:
    manager = _manager()

    handle = manager.create(name="analyst", system_prompt="be helpful", model=_fake_model([AIMessage("hi")]))

    assert isinstance(handle, AgentHandle)
    assert handle.name == "analyst"
    assert manager["analyst"] is handle
    assert "analyst" in manager
    assert "someone_else" not in manager


def test_create_with_duplicate_name_raises_value_error() -> None:
    manager = _manager()
    manager.create(name="analyst", system_prompt="x", model=_fake_model([AIMessage("hi")]))

    with pytest.raises(ValueError, match="analyst"):
        manager.create(name="analyst", system_prompt="y", model=_fake_model([AIMessage("hi")]))


def test_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        _manager()["missing"]


def test_create_with_no_model_and_no_llm_model_env_raises_configuration_error() -> None:
    manager = AgentManager(lambda: LLMCredentials(base_url="http://x", api_key="k", default_model=None))

    with pytest.raises(ConfigurationError):
        manager.create(name="analyst", system_prompt="x")


def test_run_returns_a_text_only_result() -> None:
    manager = _manager()
    handle = manager.create(
        name="analyst", system_prompt="be helpful", model=_fake_model([AIMessage("The answer is 42.")])
    )

    result = handle.run("What is the answer?")

    assert result == AgentRunResult(output="The answer is 42.", tool_calls=[])


def test_run_with_a_tool_call_round_trips_through_the_tool() -> None:
    manager = _manager()
    model = _fake_model(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="add", args={"a": 1, "b": 2}, id="call_1")]),
            AIMessage(content="The answer is 3."),
        ]
    )
    handle = manager.create(name="calculator", system_prompt="you do math", model=model, tools=[add])

    result = handle.run("what is 1+2?")

    assert result == AgentRunResult(
        output="The answer is 3.",
        tool_calls=[ToolCallRecord(name="add", args={"a": 1, "b": 2}, result="3")],
    )


def test_run_with_context_appends_it_to_the_prompt() -> None:
    manager = _manager()
    captured: dict[str, Any] = {}

    class RecordingModel(FakeToolCallingChatModel):
        def _generate(self, messages: list[Any], **kwargs: Any) -> Any:
            captured["last_human_content"] = messages[-1].content
            return super()._generate(messages, **kwargs)

    handle = manager.create(
        name="analyst", system_prompt="x", model=RecordingModel(messages=iter([AIMessage("ok")]))
    )

    handle.run("Decide.", context={"symbol": "SPY"})

    assert "Decide." in captured["last_human_content"]
    assert "SPY" in captured["last_human_content"]


def test_a_tool_exception_surfaces_as_agent_error() -> None:
    def broken(text: str) -> str:
        """Always fails."""
        raise ValueError("boom")

    manager = _manager()
    model = _fake_model(
        [
            AIMessage(content="", tool_calls=[ToolCall(name="broken", args={"text": "x"}, id="call_1")]),
            AIMessage(content="unreachable"),
        ]
    )
    handle = manager.create(name="analyst", system_prompt="x", model=model, tools=[broken])

    with pytest.raises(AgentError, match="analyst"):
        handle.run("go")


def test_create_with_a_chat_model_instance_bypasses_credentials() -> None:
    manager = AgentManager(lambda: (_ for _ in ()).throw(AssertionError("credentials should not be read")))

    handle = manager.create(name="analyst", system_prompt="x", model=_fake_model([AIMessage("ok")]))

    assert handle.run("go") == AgentRunResult(output="ok", tool_calls=[])
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_agent_manager.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'trading_agent_framework.agents.manager'`.

- [ ] **Step 5: Implement `agents/manager.py`**

Create `src/trading_agent_framework/agents/manager.py`:

```python
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
        model: "str | BaseChatModel | None" = None,
        tools: "Sequence[Callable[..., Any] | BaseTool] | None" = None,
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

    def _resolve_model(self, model: "str | BaseChatModel | None", timeout_seconds: float | None) -> Any:
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/test_agent_manager.py -v`
Expected: PASS (9 tests).

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, no regressions (existing suite unaffected — `agents/manager.py` is new and only imported by
its own tests so far).

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/agents/ tests/agents/ tests/fakes.py src/trading_agent_framework/utils/errors.py`
Expected: no errors.

```bash
git add src/trading_agent_framework/agents/manager.py \
    src/trading_agent_framework/utils/errors.py \
    tests/fakes.py tests/agents/test_agent_manager.py
git commit -m "Task 3: add AgentManager/AgentHandle and AgentError"
```

---

### Task 4: `Strategy.agents` wiring, package exports, and docs

**Files:**
- Modify: `src/trading_agent_framework/agents/__init__.py`
- Modify: `src/trading_agent_framework/core/strategy.py`
- Test: `tests/core/test_strategy_agents.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `AgentManager` (Task 3, `agents/manager.py`), `LLMCredentials` (Task 1, `agents/config.py`).
- Produces: `Strategy.agents` property (`-> AgentManager`), lazy and memoized per instance, following the
  same shape as `Strategy.memory` / `Strategy.indicators`.

- [ ] **Step 1: Widen `agents/__init__.py`'s exports**

Replace the contents of `src/trading_agent_framework/agents/__init__.py` (written in Task 1) with:

```python
"""LangChain agent creation/execution for strategies (`strategy.agents`).

Plain eager imports here are fine (unlike `brokers/__init__.py`'s lazy `__getattr__`): nothing in
`config.py`, `results.py` or `manager.py` imports `langchain` at module level, so importing this
package -- or `trading_agent_framework.core.strategy`, which imports it -- never pulls LangChain
into a strategy that doesn't create an agent.
"""

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord

__all__ = [
    "AgentHandle",
    "AgentManager",
    "AgentRunResult",
    "LLMCredentials",
    "ToolCallRecord",
]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/test_strategy_agents.py`:

```python
from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents import AgentManager
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


def _strategy(tmp_path: Path, name: str = "momentum") -> Strategy:
    return Strategy(
        FakeBroker(FakeClock(_START), strategy_name=name),
        mode=TradingMode.PAPER,
        project_root=tmp_path,
    )


def test_agents_is_lazy_and_returns_an_agent_manager(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    manager = strategy.agents

    assert isinstance(manager, AgentManager)


def test_agents_is_memoized_across_accesses(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    assert strategy.agents is strategy.agents


def test_agents_is_independent_per_strategy_instance(tmp_path: Path) -> None:
    first = _strategy(tmp_path, name="momentum")
    second = _strategy(tmp_path, name="momentum")

    assert first.agents is not second.agents
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/core/test_strategy_agents.py -v`
Expected: FAIL — `AttributeError: 'Strategy' object has no attribute 'agents'`.

- [ ] **Step 4: Wire `Strategy.agents`**

Open `src/trading_agent_framework/core/strategy.py`.

Add these two imports alongside the existing `trading_agent_framework.memory.store` import:

```python
from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager
```

In `Strategy.__init__`, next to the existing:

```python
        self._memory: MemoryStore | None = None
        self._memory_mode: TradingMode | None = None
```

add:

```python
        self._agents: AgentManager | None = None
```

Immediately after the `memory` property (which ends with `return self._memory`), add:

```python

    @property
    def agents(self) -> AgentManager:
        """This strategy's LLM agents (lumibot's `strategy.agents`); built on first use."""
        if self._agents is None:
            self._agents = AgentManager(LLMCredentials.from_env)
        return self._agents
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/core/test_strategy_agents.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, no regressions.

- [ ] **Step 7: Update `CLAUDE.md`**

In the `## Architecture` bullet list, insert a new bullet immediately after the `memory/` bullet (before the
`log.py` bullet):

```markdown
- `agents/` -- LangChain agent creation/execution (lumibot's `strategy.agents`): `config.py` (**pure**:
  `LLMCredentials.from_env`, mirrors `AlpacaCredentials`), `results.py` (**pure**: `AgentRunResult`/
  `ToolCallRecord`, and a duck-typed parser from a LangChain message list), `manager.py`
  (`AgentManager`/`AgentHandle`, the only place that imports `langchain`/`langchain_openai`, and only
  inside method bodies). No builtin tools, no MCP/skills: the caller passes `tools=[...]` explicitly (e.g.
  `memory_tools(self.memory)`).
```

In the `## Key patterns / gotchas` list, insert two new bullets immediately after the
`**Backtesting memory is wiped**` bullet (at the end of the list):

```markdown
- **`agents/` defers every LangChain import to inside method bodies** (never at module level), mirroring
  `core/indicators.py`'s deferred `pandas_ta_classic` import -- a strategy that never calls
  `self.agents.create(...)` never pays for LangChain in memory. `AgentManager` doesn't know `Strategy`
  either; it takes a `credentials_source` callable (`LLMCredentials.from_env`).
- **`AgentRunResult.tool_calls[i].result` is a string, not the tool's return value.** LangChain always
  stringifies (JSON-encodes) a tool's return value into `ToolMessage.content` before it reaches agent code,
  so the original Python object (e.g. a memory tool's `{id, kind, status}` dict) isn't recoverable there --
  read it from the tool's own return value directly if you need it as data, not from the agent's result.
```

- [ ] **Step 8: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/agents/ src/trading_agent_framework/core/strategy.py tests/core/test_strategy_agents.py`
Expected: no errors.

```bash
git add src/trading_agent_framework/agents/__init__.py \
    src/trading_agent_framework/core/strategy.py \
    tests/core/test_strategy_agents.py \
    CLAUDE.md
git commit -m "Task 4: wire Strategy.agents and document the agent framework"
```

---

## Self-Review

**Spec coverage:**
- §3.1 modules (`config.py`, `results.py`, `manager.py`, `__init__.py`) → Tasks 1-4.
- §3.2 boundaries (no `Strategy` knowledge, build-once, no tool wrapping, error wrapping, no cross-run state)
  → Task 3 (`AgentManager`/`AgentHandle` implementation and its tests); "no hidden state" is structural
  (`run()` always builds a fresh `{"messages": [...]}` call, no checkpointer passed to `create_agent`).
- §4 configuration (`LLMCredentials`, `.env.example`) → Task 1.
- §5.1 creation semantics (duplicate name, model resolution, tools passthrough) → Task 3.
- §5.2 running semantics (context, error wrapping, message parsing) → Task 3 (`AgentHandle.run`) + Task 2
  (`parse_agent_messages`).
- §5.3 result dataclasses → Task 2.
- §6 strategy wiring → Task 4.
- §7 testing (fake model, no network) → Task 3 (`FakeToolCallingChatModel`) and its test file.
- §8 documentation → Task 4, Step 7.
- §9 amendments (no tool wrapping, `result: str`, no-I/O model construction, fake needs `bind_tools`) → all
  reflected directly in the Task 3/4 code and tests above (nothing left to separately implement).
- Out-of-scope items (new domain tools, multi-agent orchestration, backtesting replay cache) → correctly not
  represented by any task.

**Placeholder scan:** no TBD/TODO, no "add appropriate error handling" — every step shows the actual code or
test to write.

**Type consistency:** `LLMCredentials` (Task 1) is used identically in Tasks 3 and 4.
`AgentRunResult`/`ToolCallRecord` (Task 2) fields (`output`, `tool_calls`, `name`, `args`, `result`) are used
identically in Task 3's tests and implementation. `AgentManager.__init__(credentials_source)` (Task 3) is
called identically in Task 4 (`AgentManager(LLMCredentials.from_env)`). `AgentHandle.name`/`.run(...)` (Task
3) match Task 4's expectations (not directly exercised in Task 4's tests, but no conflicting shape is
introduced). `AgentError`/`ConfigurationError` usage matches `utils/errors.py`'s existing exception
hierarchy.

---

**Plan complete and saved to `docs/superpowers/plans/2026-09-12-agent-framework.md`.**

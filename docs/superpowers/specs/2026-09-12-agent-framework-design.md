# Agent framework — design

- **Date:** 2026-09-12
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md` ("Agentic framework" bullet); user's own follow-up constraints
  (below); lumibot reference `lumibot/components/agents/manager.py` (`AgentManager`, `AgentHandle`,
  `AgentRunResult`) and `lumibot/strategies/examples/agent_momentum_allocator.py`.

## 1. Context

Lumibot lets a strategy create named LLM agents (Google ADK runtime) in `initialize()` and run them in
`on_trading_iteration()`, wired to tools via `@agent_tool` and a large builtin-tool/skill/MCP surface. This
spec ports the *agent creation and execution* mechanism — not a tool catalog — to this framework, per the
user's explicit constraints:

- The agent is created in `initialize()`.
- The agent is run in `on_trading_iteration()`.
- Built with **LangChain** (not LangGraph directly) — "only simple agents". `langchain`'s `create_agent()`
  (1.0 LTS) is the current, non-deprecated way to build a single-purpose tool-calling agent and is used
  under the hood; strategy/framework code never touches `langgraph` directly.
- Tools are **not** wired in by default (unlike lumibot's `include_builtin_tools=True`). The user passes an
  explicit `tools=[...]` list.

**Not covered by this spec** (explicitly out of scope):
- Building any new domain tool (market data, order placement, SEC/fundamentals). That is the separate
  "Tools" TODO item. This spec only makes existing tools (e.g. `memory_tools()`) pluggable into an agent.
- Multi-agent orchestration / chains between agents. The user asked for "only simple agents"; nothing here
  prevents a strategy from creating several independent named agents and calling them in sequence itself,
  but the framework provides no orchestration primitive beyond that.
- Backtesting-specific behavior (replay cache, deterministic/cost-free reruns). `run_backtesting()` already
  raises `NotImplementedError`; this is deferred to when the backtesting subsystem itself is designed.
- MCP servers, builtin skills, `strategy.parameters` auto-population, HITL middleware — lumibot features not
  requested and not needed for a lean single-agent-per-call use case.

## 2. Decisions taken during brainstorming

| Topic | Decision |
|---|---|
| LLM backend | OpenAI-compatible HTTP server (vLLM / llama.cpp / LM Studio), reached via `langchain-openai`'s `ChatOpenAI` with a custom `base_url`. Also works unchanged against real OpenAI if ever needed. |
| Agent API shape | Named registry, matching lumibot: `strategy.agents.create(name=..., ...)` then `strategy.agents["name"].run(...)`. |
| Run result | Lean `AgentRunResult(output: str, tool_calls: list[ToolCallRecord])` — enough to log/audit, without lumibot's event/usage/cache/timing metadata. |
| Model config | Env-driven, following the existing per-strategy-per-mode env file convention (`AlpacaCredentials` pattern): `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. |
| Dependency import cost | `langchain` / `langchain_openai` are imported lazily inside `agents/manager.py` methods only — never at module import time — so `import trading_agent_framework.core.strategy` stays light for strategies that never touch `.agents` (same trick `core/indicators.py` uses for `pandas_ta_classic`). |

## 3. Architecture

### 3.1 Modules

New top-level package `src/trading_agent_framework/agents/`, following the repo's pure / I/O split (same
shape as `memory/`):

| Module | Role | Depends on |
|---|---|---|
| `agents/config.py` | **Pure**, no I/O: `LLMCredentials.from_env()` (reads `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`), same shape as `config.env.AlpacaCredentials`. | `config/env.py` |
| `agents/results.py` | **Pure**: `ToolCallRecord`, `AgentRunResult` dataclasses, and the pure function that turns a LangChain message list into an `AgentRunResult`. | stdlib only |
| `agents/manager.py` | `AgentManager` (the `strategy.agents` object) and `AgentHandle`. The only module that imports `langchain` / `langchain_core` / `langchain_openai`, and only inside method bodies. | `config.py`, `results.py`, `utils/errors.py` |
| `agents/__init__.py` | Lazy `__getattr__` re-export of `AgentManager`, `AgentHandle`, `AgentRunResult`, `ToolCallRecord`, `LLMCredentials` — same pattern as `brokers/__init__.py`. Importing the package does not import `langchain`. | — |

Plus:
- `utils/errors.py`: new `AgentError(TradingFrameworkError)`.
- `core/strategy.py`: lazy `Strategy.agents` property (§7).
- `env/.env.example`: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` template entries.
- `pyproject.toml`: `langchain>=1.0,<2.0`, `langchain-openai` added to `dependencies`.

### 3.2 Boundaries

- **`AgentManager` does not know `Strategy`.** It is constructed with just a name→handle dict and the
  strategy's logger is *not* threaded through — logging the result is the strategy code's job (see the
  `agent_momentum_allocator.py` reference: `self.log_message(f"[agent] {result.summary}")`), not something
  the framework does automatically. This mirrors `MemoryStore`'s independence from `Strategy` and keeps the
  framework's job to "build and run an agent", nothing more.
- **`AgentHandle` builds its LangChain agent once, at `create()` time**, not on every `run()` — the model
  and tool list are fixed for the handle's lifetime. Re-creating a same-named agent is a caller error
  (`ValueError`, lumibot parity), not a way to reconfigure one.
- **Tool wrapping is automatic but non-magical.** Any callable in `tools=[...]` that is not already a
  LangChain `BaseTool` is wrapped with `langchain_core.tools.tool(fn)` before being handed to
  `create_agent`. This makes `memory_tools(strategy.memory)` (plain typed functions with one-line
  docstrings) usable as-is; a callable with no docstring or missing type hints fails at `create()` time with
  whatever error `langchain_core.tools.tool()` raises — not swallowed or special-cased.
- **Error wrapping.** Any exception raised while building the chat model or LangChain agent in `create()`
  (e.g. `ChatOpenAI` construction failure, a tool `langchain_core.tools.tool()` can't wrap), or while
  invoking the agent / parsing its response in `run()` (`httpx` connection errors, malformed provider
  responses, langchain internal errors), is caught and re-raised as `AgentError`, per the repo's "never let
  a raw SDK exception escape" rule. `AgentManager.create()`'s own programming errors that are this
  framework's own checks, not an SDK failure — duplicate `name`, missing model with no `LLM_MODEL` — raise
  plain `ValueError` / `ConfigurationError` instead: those are caller mistakes, not integration failures,
  matching how `MemoryValidationError` vs. plain `ValueError` are split in the memory layer.
- **No hidden state across runs.** Each `run()` call is a fresh `{"messages": [...]}` invocation — no
  `checkpointer`, no conversation memory across calls (lumibot doesn't have this either; if a strategy wants
  the agent to see prior context it passes it explicitly via `context=` or by having the agent call
  `search_memory`).

## 4. Configuration

### 4.1 `LLMCredentials`

```python
@dataclass(frozen=True, slots=True)
class LLMCredentials:
    base_url: str
    api_key: str = field(repr=False)
    default_model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMCredentials:
        ...
```

- Loaded the same way `AlpacaCredentials.from_env()` is: from whatever env file `load_strategy_env()`
  already loaded into `os.environ` for this strategy/mode — no new resolution logic.
- `LLM_BASE_URL` is required; missing/blank raises `ConfigurationError` (same wording style as
  `AlpacaCredentials`).
- `LLM_API_KEY` is required but may be any non-blank placeholder string — most local OpenAI-compatible
  servers (vLLM, llama.cpp, LM Studio) don't check it, but `ChatOpenAI` requires a non-empty value to
  construct. The `.env.example` comment says so explicitly.
- `LLM_MODEL` is optional; it's the default model id used when `AgentManager.create()` is called without an
  explicit `model=`. Missing `LLM_MODEL` with no `model=` argument raises `ConfigurationError` at `create()`
  time (not at `LLMCredentials.from_env()` time, since a model-instance caller never needs it).

### 4.2 `.env.example` additions

```
# LLM connection for the agent framework (create_agent). Works against any
# OpenAI-compatible server -- vLLM, llama.cpp, LM Studio, or real OpenAI.
LLM_BASE_URL=http://localhost:8000/v1

# Default model id used when Strategy.agents.create(...) is called without an
# explicit `model=`. Individual agents may override this per-call.
LLM_MODEL=qwen3-8b

# Most local OpenAI-compatible servers ignore this value, but the OpenAI SDK
# requires a non-blank string to construct the client. Use any placeholder.
LLM_API_KEY=not-needed-for-local
```

## 5. `AgentManager` / `AgentHandle`

### 5.1 Creation

```python
class AgentManager:
    def __init__(self, credentials_source: Callable[[], LLMCredentials]) -> None: ...

    def create(
        self,
        *,
        name: str,
        system_prompt: str,
        model: str | BaseChatModel | None = None,
        tools: Sequence[Callable[..., Any] | BaseTool] | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentHandle: ...

    def __getitem__(self, name: str) -> AgentHandle: ...
    def __contains__(self, name: str) -> bool: ...
```

- `credentials_source` is a zero-arg callable the strategy passes in (`LLMCredentials.from_env`), not
  resolved eagerly — so constructing `AgentManager` (which happens lazily on first `strategy.agents` access,
  §7) does no I/O until `.create()` actually needs a model string resolved.
- `name in self._agents` → `ValueError(f"Agent with name {name!r} already exists.")` (lumibot's exact
  wording, since it's a reasonable message and keeps parity).
- `model`:
  - `None` → use `credentials.default_model`; `ConfigurationError` if that's also `None`.
  - `str` → build `ChatOpenAI(base_url=credentials.base_url, api_key=credentials.api_key, model=model, timeout=timeout_seconds)`.
  - `BaseChatModel` instance → used as-is (escape hatch: a caller who wants Anthropic, a different provider,
    or custom `ChatOpenAI` kwargs the credentials shape doesn't expose, builds their own model and skips
    `LLMCredentials` entirely). `timeout_seconds` is ignored in this case — it's the caller's model to
    configure.
- `tools`: each element that is not already a `BaseTool` is wrapped with `langchain_core.tools.tool(fn)`.
  Order is preserved (no dedup, no builtin additions).
- Builds the LangChain agent immediately (`create_agent(model=..., tools=wrapped_tools, system_prompt=system_prompt)`)
  and stores it on the returned `AgentHandle`. Any exception here (bad model id format, `ChatOpenAI`
  rejecting the credentials, `langchain_core.tools.tool()` unable to wrap a tool) is caught and re-raised as
  `AgentError` — see §3.2.

### 5.2 Running

```python
class AgentHandle:
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None) -> AgentRunResult: ...
```

- Builds the human message: `task_prompt` alone, or `f"{task_prompt}\n\nContext:\n{context}"` when
  `context` is given (simple, no templating engine — this is a lean framework, not a prompt-management
  system).
- Calls `self._agent.invoke({"messages": [{"role": "user", "content": message}]})`.
- Parses the returned message list (`result["messages"]`) into an `AgentRunResult`:
  - `output` = the content of the last `AIMessage`.
  - `tool_calls` = one `ToolCallRecord(name, args, result)` per `AIMessage.tool_calls` entry, matched to its
    corresponding `ToolMessage` by `tool_call_id`, in call order. A tool call with no matching `ToolMessage`
    (defensive case — shouldn't happen with a well-behaved `create_agent` loop, but the parser must not
    crash on it) gets `result=None`.
- Any exception from `.invoke(...)` (model construction already happened in `create()`, §5.1) is caught and
  re-raised as `AgentError(f"agent {self.name!r} failed: {exc}") from exc`.

### 5.3 `AgentRunResult` / `ToolCallRecord`

```python
@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: Any

@dataclass(frozen=True, slots=True)
class AgentRunResult:
    output: str
    tool_calls: list[ToolCallRecord]
```

Pure dataclasses in `agents/results.py`; no LangChain types leak into their fields (`args`/`result` are
whatever plain JSON-compatible values the tool call carried — `dict`/`list`/`str`/`int`/`float`/`bool`/`None`
— since every existing tool, e.g. `memory_tools()`, returns exactly that).

## 6. Strategy wiring

```python
@property
def agents(self) -> AgentManager:
    """This strategy's LLM agents (lumibot's `strategy.agents`); built on first use."""
    if self._agents is None:
        self._agents = AgentManager(LLMCredentials.from_env)
    return self._agents
```

- Lazy, like `.memory` / `.indicators`: a strategy that never creates an agent never imports `langchain`.
- One `AgentManager` per `Strategy` instance for the instance's lifetime — not re-keyed by trading mode
  (unlike `.memory`), since which LLM to talk to doesn't change with paper vs. live vs. backtesting the way
  the memory DB path does. A backtest run constructs a fresh `Strategy`, so it gets a fresh `AgentManager`
  too.

Usage in a strategy subclass:

```python
from trading_agent_framework.memory.tools import memory_tools

class MyStrategy(Strategy):
    def initialize(self):
        self.agents.create(
            name="news_agent",
            model="qwen3-8b",
            system_prompt="You are a cautious equities analyst...",
            tools=memory_tools(self.memory),
        )

    def on_trading_iteration(self):
        result = self.agents["news_agent"].run("Decide today's trade for SPY.")
        self.log_info(f"[news_agent] {result.output}")
        for call in result.tool_calls:
            self.log_debug(f"[news_agent] tool {call.name}({call.args}) -> {call.result}")
```

## 7. Testing

pytest, no network, no `MagicMock` (repo convention). LangChain ships a hand-scriptable fake chat model
(`langchain_core.language_models.fake_chat_models.GenericFakeChatModel`) that returns a pre-loaded sequence
of messages, including `AIMessage`s with `tool_calls` — used instead of a hand-rolled fake, since it already
satisfies "no network" and produces real LangChain message objects (closer to production than a hand-rolled
stub would be, and there is nothing repo-specific to fake here the way there is for the Alpaca SDK).

- `tests/agents/test_agent_config.py` — `LLMCredentials.from_env`: required `LLM_BASE_URL`/`LLM_API_KEY`,
  optional `LLM_MODEL`, blank-string rejection, `ConfigurationError` messages.
- `tests/agents/test_agent_results.py` — message-list → `AgentRunResult` parsing: plain text-only reply, one
  tool call, multiple tool calls in order, a tool call with no matching `ToolMessage` (defensive case).
- `tests/agents/test_agent_manager.py` — `create()`: duplicate name raises, missing model with no
  `LLM_MODEL` raises, a plain-callable tool gets wrapped and is actually invocable, a `BaseTool` instance
  passed through unchanged, an already-built `BaseChatModel` bypasses `LLMCredentials` entirely.
  `__getitem__` / `__contains__` on unknown name. `run()`: happy path against `GenericFakeChatModel` (text
  only, and with a scripted tool call), and a model/tool exception surfaces as `AgentError`.
- `tests/core/test_strategy_agents.py` — `.agents` is lazy and memoized per instance (mirrors
  `test_strategy_memory.py`'s shape for `.memory`).

No smoke script: nothing here talks to a real broker, and testing against a real local LLM server is a
manual, environment-specific check the user runs themselves (documented in `CLAUDE.md`, not scripted).

## 8. Documentation

- `CLAUDE.md`: `agents/` bullet in the architecture list; a "Key patterns" gotcha noting `langchain` is
  deferred-imported (never at module level) so strategies that don't use agents stay light, matching the
  existing `pandas_ta_classic` gotcha's spirit.
- `TODO.md`: not edited by the implementation (carries the user's own uncommitted changes, same call as the
  memory-layer spec) — the user strikes "Agentic framework - langchain foundation" themselves.

## 9. Amendments made while planning

None yet — this section is a placeholder for changes discovered during implementation planning or coding,
following the convention set by the memory-layer spec (§10 there).

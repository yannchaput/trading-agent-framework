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
| `agents/__init__.py` | Plain eager re-export of `AgentManager`, `AgentHandle`, `AgentRunResult`, `ToolCallRecord`, `LLMCredentials` — same style as `memory/__init__.py`. This is enough (not `brokers/__init__.py`'s lazy `__getattr__`) because `manager.py` itself never imports `langchain` at module level; importing the package still doesn't import `langchain`. | — |

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
- **No tool wrapping needed.** `create_agent(tools=[...])` accepts plain callables directly (verified against
  `langchain` 1.4.0's actual signature: `tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]]`) —
  it wraps them internally. `memory_tools(strategy.memory)` (plain typed functions with one-line docstrings,
  no `@tool` decorator) is passed straight through with zero glue code; verified end-to-end against a fake
  tool-calling model. A callable with no docstring or missing type hints fails inside `create_agent` itself
  with whatever error LangChain's own tool conversion raises — not swallowed or special-cased by this
  framework.
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
    `timeout_seconds` defaults to `None`, which `ChatOpenAI` treats as **no timeout at all** (verified: not
    the OpenAI SDK's usual 600s default — `timeout=None` disables it) — deliberately not given a finite
    default here, because a reasoning-capable local model (this framework's actual target, e.g.
    `Qwen3-30B-A3B-Thinking`) can legitimately take minutes per call, and a framework-imposed default would
    silently fail exactly the deployments this project is for. The cost is explicit, not hidden: per
    `CLAUDE.md`'s "Strategy code runs on one thread" rule, a hung or OOM'd local server blocks
    `on_trading_iteration()` (and therefore order-event hook dispatch) indefinitely — callers with a
    less-patient model or a flakier server should pass `timeout_seconds` explicitly.
  - `BaseChatModel` instance → used as-is (escape hatch: a caller who wants Anthropic, a different provider,
    or custom `ChatOpenAI` kwargs the credentials shape doesn't expose, builds their own model and skips
    `LLMCredentials` entirely). `timeout_seconds` is ignored in this case — it's the caller's model to
    configure; `create()`'s docstring says so.
- `tools`: passed straight through to `create_agent(tools=...)` — no wrapping, no dedup, no builtin
  additions (§3.2).
- Builds the LangChain agent immediately (`create_agent(model=..., tools=tools, system_prompt=system_prompt)`)
  and stores it on the returned `AgentHandle`. `ChatOpenAI` construction does no I/O (verified: no network
  call happens until the first real request), so failures here are mostly `create_agent`'s own tool
  conversion errors (e.g. an untyped/undocumented tool) or pydantic validation errors from the model
  constructor (e.g. a bad `timeout_seconds` type) — caught and re-raised as `AgentError`, same as any
  `run()`-time failure (§3.2).

### 5.2 Running

```python
class AgentHandle:
    def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None) -> AgentRunResult: ...
```

- Builds the human message: `task_prompt` alone, or `f"{task_prompt}\n\nContext:\n{context}"` when
  `context` is given (simple, no templating engine — this is a lean framework, not a prompt-management
  system).
- Calls `self._agent.invoke({"messages": [{"role": "user", "content": message}]})`, then parses the returned
  message list (`result["messages"]`) into an `AgentRunResult` — both steps inside the same `try/except`, so
  a malformed-but-successful `invoke()` result (an unexpected message shape, an incompatible LangChain
  version) is wrapped the same way a connection failure is. §3.2 is authoritative here: "invoking the agent
  **/ parsing its response**... is caught and re-raised as `AgentError`" — the parse step is not exempt.
  - `output` = a message's own rendered text, taken from its `.text` attribute when that's a plain `str`
    (the langchain-core 1.x `BaseMessage.text` property, which correctly extracts text from multimodal/
    block-structured `content` — e.g. an Anthropic-style `content=[{"type": "text", ...}, ...]` list, which a
    reasoning-capable local model can also produce) and falling back to `_as_text(message.content)`
    otherwise (plain `str()` for anything that isn't already a `str`). Using `message.content` directly
    would render a list-content message as a Python list repr instead of its actual text — verified
    empirically against a `content=[{"type": "text", "text": "hello"}]` message. This keeps `results.py`
    duck-typed (`getattr(message, "text", None)`, no `langchain_core` import) while still handling the
    `BaseChatModel`-instance escape hatch (§5.1) correctly, since that path is exactly where non-string
    content shows up.
  - `tool_calls` = one `ToolCallRecord(name, args, result)` per `AIMessage.tool_calls` entry, matched to its
    corresponding `ToolMessage` by `tool_call_id`, in call order. `result` is `ToolMessage.content` **as a
    string, verbatim** — LangChain's tool-execution node always stringifies a tool's return value into
    `ToolMessage.content` (JSON-encoding a dict/list, plain `str()` otherwise; verified empirically), so the
    original Python object a tool returned (e.g. `memory_tools()`'s `{"id": ..., "kind": ..., "status": ...}`)
    is not recoverable as a Python object here — only its rendered text. This is sufficient for the stated
    purpose (§5.3: logging/audit), and reparsing JSON speculatively would be guessing at a shape this
    framework doesn't control. A tool call with no matching `ToolMessage` (defensive case — shouldn't happen
    with a well-behaved `create_agent` loop, but the parser must not crash on it) gets `result=""`.

### 5.3 `AgentRunResult` / `ToolCallRecord`

```python
@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: str

@dataclass(frozen=True, slots=True)
class AgentRunResult:
    output: str
    tool_calls: list[ToolCallRecord]
```

Pure dataclasses in `agents/results.py`; no LangChain types leak into their fields. `args` is the tool-call
arguments dict LangChain already parses onto `AIMessage.tool_calls` (plain JSON-compatible values); `result`
is always a `str` — the tool's `ToolMessage.content`, verbatim (§5.2).

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

pytest, no network, no `MagicMock` (repo convention). LangChain ships a hand-scriptable fake chat model,
`langchain_core.language_models.fake_chat_models.GenericFakeChatModel`, that returns a pre-loaded sequence of
messages (including `AIMessage`s with `tool_calls`) — used instead of a hand-rolled fake, since it already
satisfies "no network" and produces real LangChain message objects (closer to production than a hand-rolled
stub would be, and there is nothing repo-specific to fake here the way there is for the Alpaca SDK).
`GenericFakeChatModel.bind_tools()` raises `NotImplementedError` (verified) — `create_agent` always calls
`bind_tools()` on the model, so tests use a `FakeToolCallingChatModel(GenericFakeChatModel)` subclass in
`tests/fakes.py` that overrides `bind_tools(self, tools, **kwargs)` to `return self` unchanged (the fake
scripts its responses directly; it doesn't need real tool-schema binding).

- `tests/agents/test_agent_config.py` — `LLMCredentials.from_env`: required `LLM_BASE_URL`/`LLM_API_KEY`,
  optional `LLM_MODEL`, blank-string rejection, `ConfigurationError` messages.
- `tests/agents/test_agent_results.py` — message-list → `AgentRunResult` parsing: plain text-only reply, one
  tool call, multiple tool calls in order (`result` is the raw `ToolMessage.content` string, including a
  JSON-object payload left un-parsed), a tool call with no matching `ToolMessage` (defensive case, `result=""`).
- `tests/agents/test_agent_manager.py` — `create()`: duplicate name raises `ValueError`, missing model with
  no `LLM_MODEL` raises `ConfigurationError`, a plain-callable tool (e.g. one of `memory_tools()`) is
  accepted and actually invocable through the agent, an already-built `BaseChatModel` bypasses
  `LLMCredentials` entirely. `__getitem__` / `__contains__` on an unknown name. `run()`: happy path against
  `FakeToolCallingChatModel` (text only, and with a scripted tool call whose result round-trips through
  `AgentRunResult`), and a tool exception surfaces as `AgentError` (verified: an unhandled tool exception
  propagates out of `agent.invoke(...)` as the tool's own exception type, so `run()` must catch broadly).
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

These override the sections they name. All verified by running real `langchain==1.4.0` /
`langchain-openai==1.6.2` code in an ephemeral `uv run --with` environment during plan-writing, not assumed.

1. **§3.2 / §5.1 — no manual tool wrapping.** `create_agent(tools=[...])`'s actual signature accepts
   `Sequence[BaseTool | Callable[..., Any] | dict[str, Any]]` and wraps plain callables itself. Verified
   `memory_tools(store)`'s plain functions (no `@tool` decorator) work unmodified as `create_agent` tools,
   end to end through a scripted tool call. `AgentManager.create()` does no wrapping of its own.
2. **§5.2 / §5.3 — `ToolCallRecord.result` is `str`, not `Any`.** LangChain's tool-execution node always
   converts a tool's return value into `ToolMessage.content` as a string (JSON-encoding dicts/lists,
   `str()` otherwise) before it ever reaches agent code — verified with a tool returning a dict. There is no
   point in the pipeline where the original Python object is recoverable, so `ToolCallRecord.result: str`
   holds that rendered text verbatim instead of speculatively re-parsing it.
3. **§5.1 — `ChatOpenAI` construction does no I/O.** Verified: `ChatOpenAI(model=..., base_url=..., api_key=..., timeout=...)`
   builds successfully offline; no network call happens until the first real request. Failures surfaced by
   `create()` are therefore `create_agent`'s tool-conversion errors or pydantic validation errors on the
   model constructor, not connectivity errors — those only ever appear from `run()`'s `.invoke()` call.
4. **§7 — the test fake needs `bind_tools` overridden.** `GenericFakeChatModel.bind_tools()` raises
   `NotImplementedError` (it's an abstract stub on `BaseChatModel`), and `create_agent` unconditionally
   calls `model.bind_tools(...)` when the agent has any tools. Tests use a one-line
   `FakeToolCallingChatModel(GenericFakeChatModel)` subclass that overrides `bind_tools` to `return self`.

## 10. Amendments made during the final whole-branch review

Found by the final reviewer (Opus) after all 4 implementation tasks passed their own task-scoped reviews;
these are genuine spec/implementation gaps a per-task lens can't see, not planning-time corrections.

1. **§3.2 vs §5.2 — the spec contradicted itself, and the implementation followed the narrower clause.**
   §3.2 said parsing the response is wrapped in `AgentError`; §5.2 said only `.invoke(...)` is. Resolved in
   favor of §3.2 (the stricter, "never let a raw exception escape" reading) — §5.2 above now says both steps
   share one `try/except`. The exposure was small (`create_agent`'s output-state shape is type-guaranteed,
   and malformed model tool calls land in `AIMessage.invalid_tool_calls`, never `tool_calls`), but leaving
   any daylight between "invoke" and "parse" in a public framework method contradicted the project's own
   error-wrapping rule for no real benefit.
2. **§5.2 — `output` must use `.text`, not raw `.content`.** `AIMessage.content` can be a list of content
   blocks (Anthropic-style multimodal/reasoning output; a local reasoning model such as
   `Qwen3-30B-A3B-Thinking` can produce the same shape) rather than a plain string. The original duck-typed
   `_as_text(message.content)` rendered that case as a Python list repr instead of the model's actual text —
   verified against `AIMessage(content=[{"type": "text", "text": "hello"}])`. Fixed by preferring
   `message.text` (a `str` property on langchain-core 1.x's `BaseMessage` that already extracts the text
   correctly) and falling back to `_as_text(message.content)` only when `.text` isn't a plain `str`. Still no
   `langchain_core` import in `results.py` — `.text` is read via `getattr`, matching the file's existing
   duck-typing style.
3. **§5.1 — the "no default timeout" tradeoff is now explicit, not silent.** `timeout_seconds=None` means
   `ChatOpenAy` disables its request timeout entirely (verified: not the OpenAI SDK's usual 600s). Given this
   framework's actual target — slow, reasoning-capable local models — a framework-imposed finite default
   would silently break exactly the deployments it's for, so no default was added. What changed: §5.1 above,
   `create()`'s docstring, and a `CLAUDE.md` gotcha now say so explicitly, including the concrete cost (a
   hung/OOM'd local server blocks `on_trading_iteration()`, and therefore order-event hook dispatch,
   indefinitely — `CLAUDE.md`'s own "Strategy code runs on one thread" rule).
4. **§7 — the `memory_tools()` integration test from the original test list was substituted, silently.**
   §7 asked for "a plain-callable tool (e.g. one of `memory_tools()`)"; the committed test used a
   locally-defined `add(a, b)` instead. Both prove the same mechanism, but only `memory_tools()` proves the
   claim this spec actually rests weight on (§3.2: "`memory_tools(strategy.memory)`... is passed straight
   through with zero glue code") has regression coverage — a future change to a memory tool's signature that
   LangChain's schema inference can't handle would otherwise surface only at runtime, not in the suite. A
   `memory_tools()`-based test was added alongside the existing `add`-based ones (both are useful: `add`
   is a minimal walking-skeleton test, `memory_tools()` is the load-bearing integration proof).
5. **Not amended — `tests/agents/__init__.py` deleted, no spec change needed.** Task 1 created this
   unrequested empty file (deferred as a harmless Minor at the time); the final review found it actually
   registers a top-level `agents` module in `sys.modules` during test collection (verified), a latent
   shadowing hazard against the real `trading_agent_framework.agents` package. Removed — pytest still
   collects and runs `tests/agents/*` without it, matching every other test directory in the project.

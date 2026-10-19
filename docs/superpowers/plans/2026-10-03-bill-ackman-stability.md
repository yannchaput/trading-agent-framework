# Bill Ackman Stability Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the Bill Ackman agents from trading on noise: code enforces the two-fail exit and a re-entry cooldown, the short seller gets memory, a required thesis and a tool budget, sampling gets a temperature, and the researcher is widened and un-anchored.

**Architecture:** Two generic, opt-in framework options in `agents/manager.py` (`temperature` at `create()`, `exempt_tools` at `create()` + `tool_budget` at `run()`), then Ackman changes from the pure layers outward: `handoff.py` validators, `hysteresis.py` cooldowns, `parameters.py`, `state.py` v2, `pipeline.py` wiring, `prompts.py`. No change to `screen/`, `rebalancer.py`, `portfolio.py`.

**Tech Stack:** Python 3.14, `uv`, pytest, LangChain 1.4 (`create_agent`, `wrap_tool_call`), `langchain_openai.ChatOpenAI`.

**Spec:** `docs/superpowers/specs/2026-10-03-bill-ackman-stability-design.md`

## Global Constraints

- Run everything with `uv run` (`uv run pytest ...`, `uv run ruff check`). Never pip.
- `agents/manager.py` imports `langchain`/`langchain_openai`/`langchain_core` only inside method bodies, never at module level.
- `strategies/bill_ackman/handoff.py` has NO `from __future__ import annotations` (the agent layer reads real annotations). Keep it that way.
- Tests never touch the network and use hand-written fakes (`tests/fakes.py`, the fakes in `tests/strategies/bill_ackman/test_ackman_pipeline.py`), not `MagicMock`.
- Framework options are opt-in: `temperature=None`, `exempt_tools=None`, `tool_budget=None` must leave every other strategy's behaviour unchanged.
- The framework never names a strategy's tool: the budget message is built only from `{budget}`, `{blocked_tool}` (the refused call) and `{exempt_tools}` (the caller's list).
- Concern vocabulary, exactly: `debt`, `margin`, `competition`, `management`, `accounting`, `valuation`.
- New defaults, exactly: `research_top_n = 8`, `reentry_cooldown_reviews = 4`, `agent_temperature = 0.3`, short seller `tool_budget = 2 * len(to_judge)`, `STATE_VERSION = 2`.
- Prompts must never state the review cadence (`tests/strategies/bill_ackman/test_ackman_prompts.py` guards "each day|every day|daily|yesterday|today").
- Each task ends with `uv run pytest` (whole suite) and `uv run ruff check` green, then one commit whose message ends with `(Task N)` and the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Several tool calls in ONE model turn that cross the budget: only the first `tool_budget` non-exempt calls may run, the rest get the budget error (Task 2 test).
2. A paper/live state file written by the old code (version 1): it must load as an empty state with a warning, never crash or half-load (Task 6 test).
3. The trader submits an EMPTY portfolio while a holding is required: refused like any other missing required holding (Task 3 test).
4. The forced retry of the short-seller stage: the second attempt gets the same `tool_budget` as the first, not `None` (Task 8 test).
5. An abandoned review must not advance or start any cooldown (Task 7 test).

---

### Task 1: `temperature` option on `AgentManager.create`

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (`AgentManager.__init__`, `telemetry_summary`, `create`, `_resolve_model`)
- Test: `tests/agents/test_agent_manager_temperature.py` (create)

**Interfaces:**
- Produces: `AgentManager.create(*, name, system_prompt, model=None, tools=None, timeout_seconds=None, temperature: float | None = None) -> AgentHandle`; `AgentManager._resolve_model(model, timeout_seconds, temperature)`; `telemetry_summary()` entries gain key `"temperature"` (the chat model's `temperature` attribute, `None` when unset).

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_agent_manager_temperature.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from langchain_core.messages import AIMessage
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentManager

T0 = datetime(2026, 1, 5, 21, 0, tzinfo=UTC)


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


class _WarmModel(FakeToolCallingChatModel):
    temperature: float | None = 0.3


def test_a_temperature_reaches_the_chat_model() -> None:
    assert _manager()._resolve_model(None, None, 0.3).temperature == 0.3


def test_without_a_temperature_nothing_is_sent_and_the_server_default_applies() -> None:
    chat_model = _manager()._resolve_model(None, None, None)

    assert chat_model.temperature is None
    assert "temperature" not in chat_model._default_params


def test_a_prebuilt_model_is_used_as_is_whatever_the_temperature() -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))

    assert _manager()._resolve_model(model, None, 0.9) is model


def test_the_telemetry_summary_records_each_agents_temperature() -> None:
    manager = _manager()
    manager.enable_telemetry(now=lambda: T0)
    manager.create(name="warm", system_prompt="x", model=_WarmModel(messages=iter([AIMessage(content="ok")]))).run("go")
    manager.create(name="plain", system_prompt="x", model=FakeToolCallingChatModel(messages=iter([AIMessage(content="ok")]))).run("go")

    summary = manager.telemetry_summary()

    assert summary["warm"]["temperature"] == 0.3
    assert summary["plain"]["temperature"] is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/agents/test_agent_manager_temperature.py -v`
Expected: FAIL (`_resolve_model() takes 3 positional arguments but 4 were given`, `KeyError: 'temperature'`).

- [ ] **Step 3: Implement**

In `AgentManager.__init__`, add after `self._calls`:

```python
        self._temperatures: dict[str, float | None] = {}
```

Replace `telemetry_summary`:

```python
    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        """Per-agent totals of the calls recorded so far, each with the agent's sampling `temperature` (empty when telemetry is off or nothing ran)."""
        summary = summarize(self._calls)
        for agent, totals in summary.items():
            totals["temperature"] = self._temperatures.get(agent)
        return summary
```

In `create`, add the parameter `temperature: float | None = None,` after `timeout_seconds`, add this paragraph to the docstring after the `timeout_seconds` paragraph:

```
        `temperature` defaults to `None` -- nothing is sent, so the server's own default applies (vLLM: the
        model's `generation_config.json`). Like `timeout_seconds`, it only applies when `model` is a string;
        a pre-built `BaseChatModel` keeps its own sampling settings. The telemetry summary records the
        temperature the chat model actually has.
```

change `chat_model = self._resolve_model(model, timeout_seconds)` to `chat_model = self._resolve_model(model, timeout_seconds, temperature)`, and after `self._agents[name] = handle` add:

```python
        self._temperatures[name] = getattr(chat_model, "temperature", None)
```

In `_resolve_model`, change the signature to `def _resolve_model(self, model: str | BaseChatModel | None, timeout_seconds: float | None, temperature: float | None = None) -> Any:` and the `ChatOpenAI(...)` call to:

```python
            sampling = {} if temperature is None else {"temperature": temperature}
            return ChatOpenAI(
                model=model_id,
                base_url=credentials.base_url,
                api_key=SecretStr(credentials.api_key),
                timeout=timeout_seconds,
                **sampling,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/ tests/core/test_strategy_agent_telemetry.py tests/backtesting/test_runner_agent_telemetry.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/agents/manager.py tests/agents/test_agent_manager_temperature.py
git commit -m "feat: agents.create takes an optional sampling temperature (Task 1)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: per-run tool budget (`exempt_tools` + `tool_budget`)

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (new `_ToolBudget`, `_tool_budget_message`; `AgentHandle.__init__`/`run`; new `AgentManager._tool_budget_middleware`; `create`)
- Modify: `docs/superpowers/specs/2026-10-03-bill-ackman-stability-design.md` (one sentence, see Step 5)
- Test: `tests/agents/test_agent_manager_tool_budget.py` (create)

**Interfaces:**
- Consumes: `AgentManager.create(..., temperature=...)` from Task 1.
- Produces: `AgentManager.create(..., exempt_tools: Sequence[str] | None = None)`; `AgentHandle.run(task_prompt, *, context=None, run_id=None, force_tool=None, tool_budget: int | None = None) -> AgentRunResult`; module function `_tool_budget_message(budget: int, blocked_tool: str, exempt_tools: Sequence[str]) -> str`.

The middleware is installed on every agent (like `_forced_tool_choice_middleware`) and is a no-op when a run has no budget; that is the "opt-in" guarantee.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_agent_manager_tool_budget.py`:

```python
from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.fakes import FakeToolCallingChatModel

from trading_agent_framework.agents.config import LLMCredentials
from trading_agent_framework.agents.manager import AgentHandle, AgentManager, _tool_budget_message


def _manager() -> AgentManager:
    return AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b"))


def _calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(content="", tool_calls=[ToolCall(name=name, args=args, id=f"call_{index}_{name}") for index, (name, args) in enumerate(calls)])


def _agent(seen: list[str], *messages: AIMessage, exempt_tools: list[str] | None = None) -> AgentHandle:
    def lookup(symbol: str) -> str:
        """Look up a symbol."""
        seen.append(symbol)
        return f"{symbol} ok"

    def submit(answer: str) -> dict[str, str]:
        """Submit the answer."""
        seen.append(f"submit:{answer}")
        return {"status": "recorded"}

    model = FakeToolCallingChatModel(messages=iter(list(messages)))
    return _manager().create(name="seller", system_prompt="x", model=model, tools=[lookup, submit], exempt_tools=exempt_tools)


def test_calls_past_the_budget_are_refused_and_never_run() -> None:
    seen: list[str] = []
    handle = _agent(
        seen,
        _calls(("lookup", {"symbol": "A"})),
        _calls(("lookup", {"symbol": "B"})),
        _calls(("lookup", {"symbol": "C"})),
        _calls(("submit", {"answer": "x"})),
        AIMessage(content="done"),
        exempt_tools=["submit"],
    )

    result = handle.run("go", tool_budget=2)

    assert seen == ["A", "B", "submit:x"]  # C was refused, the exempt submit still ran
    refused = [call for call in result.tool_calls if call.name == "lookup"][2]
    assert json.loads(refused.result) == {"error": "tool budget of 2 calls spent; lookup was not run. Finish now: call submit."}


def test_several_calls_in_one_turn_count_one_by_one() -> None:
    seen: list[str] = []
    handle = _agent(seen, _calls(("lookup", {"symbol": "A"}), ("lookup", {"symbol": "B"}), ("lookup", {"symbol": "C"})), AIMessage(content="done"))

    result = handle.run("go", tool_budget=2)

    assert sorted(seen) == ["A", "B"]
    assert sum("tool budget of 2 calls spent" in call.result for call in result.tool_calls) == 1


def test_without_a_budget_nothing_is_refused() -> None:
    seen: list[str] = []
    handle = _agent(seen, _calls(("lookup", {"symbol": "A"})), _calls(("lookup", {"symbol": "B"})), _calls(("lookup", {"symbol": "C"})), AIMessage(content="done"))

    handle.run("go")

    assert seen == ["A", "B", "C"]


def test_the_budget_is_per_run() -> None:
    seen: list[str] = []
    handle = _agent(
        seen,
        _calls(("lookup", {"symbol": "A"})),
        _calls(("lookup", {"symbol": "B"})),
        AIMessage(content="first run done"),
        _calls(("lookup", {"symbol": "C"})),
        AIMessage(content="second run done"),
    )

    handle.run("go", tool_budget=1)
    handle.run("go again")  # no budget: a previous run's budget must not linger

    assert seen == ["A", "C"]


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="tool_budget"):
        _agent([], AIMessage(content="done")).run("go", tool_budget=-1)


def test_the_message_names_only_the_callers_tools() -> None:
    assert _tool_budget_message(12, "get_bars", ["submit_a", "submit_b"]) == "tool budget of 12 calls spent; get_bars was not run. Finish now: call submit_a or submit_b."
    assert _tool_budget_message(3, "get_bars", []) == "tool budget of 3 calls spent; get_bars was not run. Answer now without calling another tool."
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/agents/test_agent_manager_tool_budget.py -v`
Expected: FAIL with `ImportError: cannot import name '_tool_budget_message'`.

- [ ] **Step 3: Implement**

In `manager.py`, add above `class AgentHandle`:

```python
class _ToolBudget:
    """One agent's tool-call budget for the current run: set by `AgentHandle.run`, enforced by `_tool_budget_middleware`."""

    def __init__(self) -> None:
        self.limit: int | None = None
        self.used = 0


def _tool_budget_message(budget: int, blocked_tool: str, exempt_tools: Sequence[str]) -> str:
    """The error a call past the budget gets instead of running; names only the caller's own tools."""
    spent = f"tool budget of {budget} calls spent; {blocked_tool} was not run."
    if exempt_tools:
        return f"{spent} Finish now: call {' or '.join(exempt_tools)}."
    return f"{spent} Answer now without calling another tool."
```

Change `AgentHandle.__init__` to:

```python
    def __init__(self, name: str, agent: Any, budget: _ToolBudget | None = None) -> None:
        self.name = name
        self._agent = agent
        self._budget = budget or _ToolBudget()
```

Change `AgentHandle.run` to take `tool_budget: int | None = None` (after `force_tool`), append to its docstring:

```
        `tool_budget`, when given, caps this run's tool calls: past it, every tool but the agent's
        `exempt_tools` (see `AgentManager.create`) returns an `{"error": ...}` without running, so the
        model finishes with what it has instead of overflowing its context. Each run starts a fresh count.
```

and make its body:

```python
        if tool_budget is not None and tool_budget < 0:
            raise ValueError(f"tool_budget must be at least 0, got {tool_budget}")
        message = task_prompt if context is None else f"{task_prompt}\n\nContext:\n{context}"
        run_id = run_id or uuid.uuid4().hex
        self._budget.limit, self._budget.used = tool_budget, 0
        try:
            with agent_call_context(run_id=run_id, force_tool=force_tool):
                raw_result = self._agent.invoke({"messages": [{"role": "user", "content": message}]})
            logger.log_debug(f"Model returned this raw messages: {str(raw_result)}")
            return parse_agent_messages(raw_result["messages"])
        except Exception as exc:
            raise AgentError(f"agent {self.name!r} failed: {exc}") from exc
        finally:
            self._budget.limit = None
```

Add this method to `AgentManager`, after `_forced_tool_choice_middleware`:

```python
    def _tool_budget_middleware(self, budget: _ToolBudget, exempt_tools: Sequence[str]) -> Any:
        """Middleware for every agent: past `AgentHandle.run(tool_budget=N)` calls in a run, refuse every tool but `exempt_tools`.

        A no-op when the run has no budget (the default). A refused call never runs: its result is an
        `{"error": ...}` the model can act on, the same feedback shape as a tool's own refusal. Exempt
        calls are neither counted nor refused.
        """
        from langchain.agents.middleware import wrap_tool_call
        from langchain_core.messages import ToolMessage

        exempt = tuple(exempt_tools)

        @wrap_tool_call
        def enforce_tool_budget(request: Any, handler: Callable[[Any], Any]) -> Any:
            name = request.tool_call["name"]
            if budget.limit is None or name in exempt:
                return handler(request)
            if budget.used >= budget.limit:
                logger.log_warning(f"tool budget of {budget.limit} calls spent: {name} was refused")
                error = {"error": _tool_budget_message(budget.limit, name, exempt)}
                return ToolMessage(content=json.dumps(error), tool_call_id=request.tool_call["id"], name=name)
            budget.used += 1
            return handler(request)

        return enforce_tool_budget
```

In `create`, add the parameter `exempt_tools: Sequence[str] | None = None,` after `temperature`, add to the docstring:

```
        `exempt_tools` names the tools a per-run `tool_budget` (see `AgentHandle.run`) never refuses,
        typically the tool the agent must end with; without a budget it changes nothing.
```

and change the body so the budget is shared by the middleware and the handle:

```python
        chat_model = self._resolve_model(model, timeout_seconds, temperature)
        budget = _ToolBudget()
        try:
            from langchain.agents import create_agent

            glm_format = _is_glm_model(getattr(chat_model, "model_name", None))
            middleware = [
                self._tool_call_repair_middleware(glm_format=glm_format),
                self._forced_tool_choice_middleware(),
                self._tool_budget_middleware(budget, exempt_tools or ()),
            ]
            if self._telemetry_now is not None:
                middleware.append(self._telemetry_middleware(name))
            agent = create_agent(model=chat_model, tools=list(tools or []), system_prompt=system_prompt, middleware=middleware)
        except Exception as exc:
            raise AgentError(f"failed to create agent {name!r}: {exc}") from exc

        handle = AgentHandle(name, agent, budget)
```

(keep the two lines after it: `self._agents[name] = handle`, `self._temperatures[...]`, `return handle`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/ -v`
Expected: PASS. If `test_several_calls_in_one_turn_count_one_by_one` shows more than 2 lookups ran, LangGraph ran the calls concurrently: guard the check-and-increment with a `threading.Lock` stored on `_ToolBudget` (`self.lock = threading.Lock()`, `with budget.lock:` around the `if budget.used >= ...` / `budget.used += 1` block, calling `handler` outside the lock).

- [ ] **Step 5: Align the spec sentence**

In `docs/superpowers/specs/2026-10-03-bill-ackman-stability-design.md`, replace `Both
  options default to \`None\`, which adds no middleware, so other strategies are unchanged.` with `Both
  options default to \`None\`: the middleware is installed on every agent but does nothing without a budget, so
  other strategies are unchanged.`

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/agents/manager.py tests/agents/test_agent_manager_tool_budget.py docs/superpowers/specs/2026-10-03-bill-ackman-stability-design.md
git commit -m "feat: agents get an optional per-run tool budget with exempt tools (Task 2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: hand-off validators (concern, what_changed, required holdings)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/handoff.py`
- Modify: `tests/strategies/bill_ackman/test_ackman_handoff.py` (`_verdict` helper and the valid-verdicts test, new tests)
- Modify: `tests/strategies/bill_ackman/test_ackman_pipeline.py` (`judges` helper only)

**Interfaces:**
- Produces: `CONCERNS: tuple[str, ...]`; `Verdict(symbol, verdict, reason, concern: str | None = None, what_changed: str | None = None)`; `validate_verdicts(raw, *, expected, reason_max_chars, previous: Mapping[str, str] | None = None)`; `validate_portfolio(..., reason_max_chars, required: Collection[str] = ())`; `HandoffRecorder.expect_verdicts(symbols, previous: Mapping[str, str] | None = None)`; `HandoffRecorder.expect_portfolio(allowed, required: Sequence[str] = ())`.

- [ ] **Step 1: Write the failing tests**

In `tests/strategies/bill_ackman/test_ackman_handoff.py`, replace the `_verdict` helper and `test_valid_verdicts_cover_exactly_the_asked_symbols` with:

```python
def _verdict(symbol: str, verdict: Any = "survive", reason: str = "debt is fine", **extra: Any) -> dict[str, Any]:
    return {"symbol": symbol, "verdict": verdict, "reason": reason, **extra}


def test_valid_verdicts_cover_exactly_the_asked_symbols() -> None:
    verdicts = validate_verdicts([_verdict("aaa", " FAIL ", concern=" Debt "), _verdict("BBB")], expected=["AAA", "BBB"], reason_max_chars=300)

    assert verdicts == [Verdict("AAA", "fail", "debt is fine", concern="debt"), Verdict("BBB", "survive", "debt is fine")]
```

Add, after `test_invalid_verdicts_are_refused`:

```python
@pytest.mark.parametrize("concern", [None, "", "lawsuit", 3])
def test_a_fail_needs_a_concern_from_the_list(concern: Any) -> None:
    with pytest.raises(HandoffError, match="needs a concern: one of debt, margin, competition, management, accounting, valuation"):
        validate_verdicts([_verdict("AAA", "fail", concern=concern)], expected=["AAA"], reason_max_chars=300)


def test_a_survive_drops_any_concern() -> None:
    (verdict,) = validate_verdicts([_verdict("AAA", concern="debt")], expected=["AAA"], reason_max_chars=300)

    assert verdict.concern is None


def test_a_flip_needs_what_changed() -> None:
    with pytest.raises(HandoffError, match="AAA was 'survive' at the previous review: say in what_changed what changed since then"):
        validate_verdicts([_verdict("AAA", "fail", concern="debt")], expected=["AAA"], reason_max_chars=300, previous={"AAA": "survive"})


def test_a_flip_with_what_changed_is_accepted_and_kept() -> None:
    (verdict,) = validate_verdicts(
        [_verdict("AAA", "fail", concern="margin", what_changed=" margin fell 8 points in the new 10-K ")], expected=["AAA"], reason_max_chars=300, previous={"AAA": "survive"}
    )

    assert verdict.what_changed == "margin fell 8 points in the new 10-K"


def test_an_unchanged_verdict_or_a_new_symbol_needs_no_what_changed() -> None:
    verdicts = validate_verdicts([_verdict("AAA", "fail", concern="debt"), _verdict("BBB")], expected=["AAA", "BBB"], reason_max_chars=300, previous={"AAA": "fail"})

    assert [verdict.what_changed for verdict in verdicts] == [None, None]


def test_what_changed_is_length_capped_like_a_reason() -> None:
    with pytest.raises(HandoffError, match="what_changed for AAA is 11 characters: keep it under 10"):
        validate_verdicts([_verdict("AAA", reason="short", what_changed="x" * 11)], expected=["AAA"], reason_max_chars=10, previous={"AAA": "fail"})
```

Add, after `test_with_nothing_allowed_the_refusal_says_to_submit_an_empty_list`:

```python
def test_a_required_holding_must_stay_in_the_portfolio() -> None:
    with pytest.raises(HandoffError, match="CCC failed once and is kept until a second consecutive fail: include it with a weight of at least 0.05"):
        validate_portfolio([_position("AAA", 0.3)], allowed=["AAA", "CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def test_an_empty_portfolio_is_refused_while_a_holding_is_required() -> None:
    with pytest.raises(HandoffError, match="CCC failed once"):
        validate_portfolio([], allowed=["CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300)


def test_a_required_holding_may_be_shrunk_to_the_minimum_weight() -> None:
    positions = validate_portfolio(
        [_position("AAA", 0.3), _position("CCC", 0.05)], allowed=["AAA", "CCC"], required=["CCC"], max_positions=5, min_weight=0.05, max_weight=0.35, max_total_weight=0.98, reason_max_chars=300
    )

    assert [position.symbol for position in positions] == ["AAA", "CCC"]
```

Add, after `test_arming_a_stage_clears_the_previous_submission_and_error`:

```python
def test_the_recorder_checks_flips_and_required_holdings_against_what_it_was_armed_with() -> None:
    recorder = _recorder()
    tools = submit_tools(recorder)

    recorder.expect_verdicts(["AAA"], previous={"AAA": "survive"})
    assert "what_changed" in tools["submit_verdicts"]([_verdict("AAA", "fail", concern="debt")])["error"]
    assert tools["submit_verdicts"]([_verdict("AAA", "fail", concern="debt", what_changed="debt doubled")]) == {"status": "recorded"}

    recorder.expect_portfolio(["AAA", "CCC"], required=["CCC"])
    assert "CCC failed once" in tools["submit_portfolio"]([_position("AAA", 0.3)])["error"]
    assert tools["submit_portfolio"]([_position("AAA", 0.3), _position("CCC", 0.1)]) == {"status": "recorded"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_handoff.py -v`
Expected: FAIL (`Verdict.__init__() got an unexpected keyword argument 'concern'`, `unexpected keyword argument 'previous'` / `'required'`).

- [ ] **Step 3: Implement in `handoff.py`**

Below `VERDICTS = (SURVIVE, FAIL)` add:

```python
CONCERNS = ("debt", "margin", "competition", "management", "accounting", "valuation")
```

Replace the `Verdict` dataclass:

```python
@dataclass(frozen=True, slots=True)
class Verdict:
    symbol: str
    verdict: str
    reason: str
    concern: str | None = None  # one of CONCERNS for a short seller's fail; None for a survive or a code verdict
    what_changed: str | None = None  # required when the verdict differs from the previous review's
```

Add after `_reason`:

```python
def _optional_text(item: Mapping[str, Any], key: str, symbol: str, max_chars: int) -> str | None:
    value = item.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise HandoffError(f"the {key} for {symbol} must be text")
    text = value.strip()
    if len(text) > max_chars:
        raise HandoffError(f"the {key} for {symbol} is {len(text)} characters: keep it under {max_chars}")
    return text


def _concern(item: Mapping[str, Any], symbol: str, verdict: str) -> str | None:
    if verdict != FAIL:
        return None
    value = item.get("concern")
    concern = value.strip().lower() if isinstance(value, str) else None
    if concern not in CONCERNS:
        raise HandoffError(f"the fail for {symbol} needs a concern: one of {', '.join(CONCERNS)}")
    return concern
```

Replace `validate_verdicts`:

```python
def validate_verdicts(raw: Any, *, expected: Collection[str], reason_max_chars: int, previous: Mapping[str, str] | None = None) -> list[Verdict]:
    """Exactly one verdict (`survive` or `fail`) per symbol in `expected`, and no other symbol.

    A `fail` names its `concern` (one of `CONCERNS`); a verdict that differs from `previous` (symbol -> the last
    review's verdict) says `what_changed`.
    """
    items = _items(raw, "verdicts")
    wanted = set(expected)
    before = previous or {}
    seen: set[str] = set()
    verdicts = []
    for index, item in enumerate(items, start=1):
        symbol = _symbol(item, index)
        if symbol not in wanted:
            raise HandoffError(f"{symbol} was not asked about: judge only {', '.join(sorted(wanted))}")
        if symbol in seen:
            raise HandoffError(f"{symbol} appears twice")
        seen.add(symbol)
        verdict = item.get("verdict")
        verdict = verdict.strip().lower() if isinstance(verdict, str) else verdict
        if verdict not in VERDICTS:
            raise HandoffError(f"the verdict for {symbol} must be 'survive' or 'fail'")
        reason = _reason(item, symbol, reason_max_chars)
        concern = _concern(item, symbol, verdict)
        what_changed = _optional_text(item, "what_changed", symbol, reason_max_chars)
        if before.get(symbol, verdict) != verdict and what_changed is None:
            raise HandoffError(f"{symbol} was '{before[symbol]}' at the previous review: say in what_changed what changed since then")
        verdicts.append(Verdict(symbol, verdict, reason, concern, what_changed))
    missing = sorted(wanted - seen)
    if missing:
        raise HandoffError(f"no verdict for {', '.join(missing)}: give one verdict per symbol")
    return verdicts
```

In `validate_portfolio`, add the keyword parameter `required: Collection[str] = (),` after `reason_max_chars: int,`, add `Every symbol in \`required\` must be held.` to its docstring, and insert before `total = ...`:

```python
    held = {position.symbol for position in positions}
    for symbol in required:
        if symbol not in held:
            raise HandoffError(f"{symbol} failed once and is kept until a second consecutive fail: include it with a weight of at least {min_weight}")
```

In `HandoffRecorder`, replace `expect_verdicts` and `expect_portfolio`:

```python
def expect_verdicts(self, symbols: Sequence[str], previous: Mapping[str, str] | None = None) -> None:
    self._arm(VERDICTS_STAGE, {"expected": list(symbols), "previous": dict(previous or {})})


def expect_portfolio(self, allowed: Sequence[str], required: Sequence[str] = ()) -> None:
    self._arm(PORTFOLIO, {"allowed": list(allowed), "required": list(required)})
```

In `submit`, pass `previous=self._context["previous"]` to `validate_verdicts` and `required=self._context["required"]` to `validate_portfolio`.

In `submit_tools`, change the `submit_verdicts` docstring to (one line):

```python
        """Submit one verdict per symbol you were asked about: objects with symbol, verdict (survive or fail), reason, concern (for a fail) and what_changed (when the verdict changed)."""
```

- [ ] **Step 4: Keep the pipeline tests' fake short seller valid**

In `tests/strategies/bill_ackman/test_ackman_pipeline.py`, replace `judges`:

```python
def judges(**verdicts: str) -> Step:
    return lambda tools, ctx: tools["submit_verdicts"](
        [
            {"symbol": symbol, "verdict": verdict, "reason": f"{verdict} reason", "what_changed": "new facts", **({"concern": "debt"} if verdict == "fail" else {})}
            for symbol, verdict in verdicts.items()
        ]
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/handoff.py tests/strategies/bill_ackman/test_ackman_handoff.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "feat: bill_ackman fails need a concern, flips say what changed, pending fails stay held (Task 3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: re-entry cooldowns (pure)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/hysteresis.py`
- Test: `tests/strategies/bill_ackman/test_ackman_hysteresis.py`

**Interfaces:**
- Produces: `advance_cooldowns(cooldowns: Mapping[str, int], *, forced_exits: Sequence[str], reviews: int) -> dict[str, int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/strategies/bill_ackman/test_ackman_hysteresis.py` (and add `advance_cooldowns` to its import from `hysteresis`):

```python
def test_a_forced_exit_starts_a_cooldown() -> None:
    assert advance_cooldowns({}, forced_exits=["HLT"], reviews=4) == {"HLT": 4}


def test_each_completed_review_counts_every_cooldown_down_and_drops_it_at_zero() -> None:
    assert advance_cooldowns({"A": 3, "B": 1}, forced_exits=[], reviews=4) == {"A": 2}


def test_a_forced_exit_keeps_the_symbol_out_of_the_next_reviews_exactly() -> None:
    cooldowns = advance_cooldowns({}, forced_exits=["HLT"], reviews=4)  # forced out at review R
    for _ in range(4):  # reviews R+1 .. R+4 start with HLT still cooling down
        assert "HLT" in cooldowns
        cooldowns = advance_cooldowns(cooldowns, forced_exits=[], reviews=4)
    assert "HLT" not in cooldowns  # back at review R+5


def test_a_symbol_forced_out_again_restarts_its_cooldown() -> None:
    assert advance_cooldowns({"HLT": 2}, forced_exits=["HLT"], reviews=4) == {"HLT": 4}


def test_zero_reviews_disables_the_cooldown() -> None:
    assert advance_cooldowns({}, forced_exits=["HLT"], reviews=0) == {}


def test_the_input_cooldowns_are_not_modified() -> None:
    before = {"A": 2}

    advance_cooldowns(before, forced_exits=["B"], reviews=4)

    assert before == {"A": 2}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_hysteresis.py -v`
Expected: FAIL with `ImportError: cannot import name 'advance_cooldowns'`.

- [ ] **Step 3: Implement**

Append to `hysteresis.py`, and add to its module docstring the sentence: `A forced exit then stays out of the candidates and the allowed set for a few completed reviews (\`advance_cooldowns\`).`

```python
def advance_cooldowns(cooldowns: Mapping[str, int], *, forced_exits: Sequence[str], reviews: int) -> dict[str, int]:
    """The cooldowns after a completed review: each one counts down (dropped at 0), then each forced exit starts at `reviews`.

    The pipeline keeps a symbol with a cooldown out of the researcher's candidates and the trader's allowed set, so
    a forced exit at review R stays out from R+1 through R+`reviews`. `reviews` = 0 disables cooldowns.
    """
    after = {symbol: left - 1 for symbol, left in cooldowns.items() if left > 1}
    if reviews > 0:
        after.update(dict.fromkeys(forced_exits, reviews))
    return after
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_hysteresis.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/hysteresis.py tests/strategies/bill_ackman/test_ackman_hysteresis.py
git commit -m "feat: bill_ackman re-entry cooldowns after a forced exit (Task 4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: parameters and agent wiring (top 8, cooldown, temperature, exempt submit)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/parameters.py`
- Modify: `src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py` (`initialize`)
- Test: `tests/strategies/bill_ackman/test_ackman_params.py`, `tests/strategies/bill_ackman/test_ackman_strategy.py`

**Interfaces:**
- Consumes: `AgentManager.create(..., temperature=, exempt_tools=)` (Tasks 1-2).
- Produces: `AckmanParams.research_top_n = 8`, `AckmanParams.reentry_cooldown_reviews: int = 4`, `AckmanParams.agent_temperature: float | None = 0.3`.

- [ ] **Step 1: Write the failing tests**

In `test_ackman_params.py`, change the line `assert (params.research_top_n, params.max_positions) == (5, 5)` to `assert (params.research_top_n, params.max_positions) == (8, 5)` and add below it `assert (params.reentry_cooldown_reviews, params.agent_temperature) == (4, 0.3)`. Add these entries to the `test_an_out_of_range_value_is_refused` parametrize list:

```python
({"reentry_cooldown_reviews": -1},)
({"agent_temperature": -0.1},)
({"agent_temperature": 2.01},)
({"agent_temperature": math.nan},)
```

and append:

```python
def test_a_cooldown_of_zero_and_the_temperature_bounds_are_accepted() -> None:
    AckmanParams(reentry_cooldown_reviews=0, agent_temperature=0.0)
    AckmanParams(agent_temperature=2.0)
    AckmanParams(agent_temperature=None)
```

Append to `test_ackman_strategy.py`:

```python
def test_every_agent_is_sampled_at_the_configured_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=AckmanParams(agent_temperature=0.2))

    strategy.initialize()

    assert [created["temperature"] for created in agents.created] == [0.2, 0.2, 0.2]


def test_only_the_short_seller_has_a_budget_exempt_tool_its_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    assert [created.get("exempt_tools") for created in agents.created] == [None, ["submit_verdicts"], None]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_strategy.py -v`
Expected: FAIL (`AckmanParams.__init__() got an unexpected keyword argument 'reentry_cooldown_reviews'`, `KeyError: 'temperature'`).

- [ ] **Step 3: Implement**

In `parameters.py`, change `research_top_n: int = 5  # ideas the researcher submits` to `research_top_n: int = 8  # ideas the researcher submits`, and add after `forced_exit_fails`:

```python
    reentry_cooldown_reviews: int = 4  # completed reviews a forced exit stays out of the candidates and the allowed set (0 disables)
    agent_temperature: float | None = 0.3  # sampling temperature of the three agents; None leaves the LLM server's default
```

In `__post_init__`'s `problems` dict, add:

```python
            "reentry_cooldown_reviews must be at least 0": self.reentry_cooldown_reviews < 0,
            "agent_temperature must be None or in [0, 2]": self.agent_temperature is not None and not (math.isfinite(self.agent_temperature) and 0 <= self.agent_temperature <= 2),
```

In `agent_bill_ackman.py`'s `initialize`, replace the three `self.agents.create(...)` calls with:

```python
            temperature = self.settings.agent_temperature
            self.agents.create(
                name="researcher",
                system_prompt=RESEARCHER_SYSTEM,
                tools=[*fundamentals_tools(self), *market_data_tools(self), submit["submit_ranking"]],
                temperature=temperature,
            )
            self.agents.create(
                name="short_seller",
                system_prompt=SHORT_SELLER_SYSTEM,
                tools=[*fundamentals_tools(self), *news_tools(self), *market_data_tools(self), submit["submit_verdicts"]],
                temperature=temperature,
                exempt_tools=["submit_verdicts"],  # its per-review tool budget never blocks the submission
            )
            self.agents.create(name="trader", system_prompt=TRADER_SYSTEM, tools=[submit["submit_portfolio"]], temperature=temperature)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/parameters.py src/trading_agent_framework/strategies/bill_ackman/agent_bill_ackman.py tests/strategies/bill_ackman/test_ackman_params.py tests/strategies/bill_ackman/test_ackman_strategy.py
git commit -m "feat: bill_ackman ranks 8 ideas, samples at 0.3 and exempts submit_verdicts from the budget (Task 5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: state file v2 (verdict records, cooldowns)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/state.py`
- Modify: `src/trading_agent_framework/strategies/bill_ackman/pipeline.py` (the `ReviewState(...)` built in step 10 of `_review` only)
- Test: `tests/strategies/bill_ackman/test_ackman_state.py`, `tests/strategies/bill_ackman/test_ackman_pipeline.py` (one assertion)

**Interfaces:**
- Consumes: `Verdict.concern` (Task 3).
- Produces: `STATE_VERSION = 2`; `ReviewState.last_verdicts: dict[str, dict[str, Any]]` (symbol -> `{"verdict", "reason", "concern", "date"}`); `ReviewState.cooldowns: dict[str, int]` (last field, default `{}`).

- [ ] **Step 1: Write the failing tests**

In `test_ackman_state.py`, replace `_state()`:

```python
def _state() -> ReviewState:
    return ReviewState(
        last_review="2026-09-14",
        fail_counts={"HLT": 1},
        last_ranking=["AAA", "BBB"],
        last_verdicts={
            "AAA": {"verdict": "survive", "reason": "cash rich", "concern": None, "date": "2026-09-14"},
            "HLT": {"verdict": "fail", "reason": "debt doubled", "concern": "debt", "date": "2026-09-14"},
        },
        abandoned_streak=2,
        cooldowns={"OLD": 3},
    )
```

In `test_the_empty_state_has_no_history`, change the assertion to:

```python
    assert (state.last_review, state.fail_counts, state.last_ranking, state.last_verdicts, state.abandoned_streak, state.cooldowns) == (None, {}, [], {}, 0, {})
```

In the parametrize list above `test_a_corrupt_or_wrong_shaped_file_is_an_empty_state_with_a_warning`: change `json.dumps({"version": 2})` to `json.dumps({"version": 3})`, change every other `"version": 1` to `"version": 2`, replace the `last_verdicts` entry with the three entries below, and add the four entries after them:

```python
(json.dumps({"version": 2, "last_verdicts": {"AAA": "survive"}}),)  # the version-1 shape
(json.dumps({"version": 2, "last_verdicts": {"AAA": {"verdict": "maybe", "reason": "x", "concern": None, "date": "2026-09-14"}}}),)
(json.dumps({"version": 2, "last_verdicts": {"AAA": {"verdict": "fail", "reason": "x", "concern": None}}}),)  # no date
(json.dumps({"version": 2, "last_verdicts": {"AAA": {"verdict": "fail", "reason": 1, "concern": None, "date": "2026-09-14"}}}),)
(json.dumps({"version": 2, "cooldowns": {"OLD": 0}}),)
(json.dumps({"version": 2, "cooldowns": {"OLD": True}}),)
(json.dumps({"version": 2, "cooldowns": []}),)
```

Append:

```python
def test_a_file_written_by_the_previous_version_is_an_empty_state_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "last_review": "2026-09-07", "fail_counts": {"HLT": 1}, "last_ranking": [], "last_verdicts": {"HLT": "fail"}, "abandoned_streak": 0}))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == ReviewState()

    assert "not a valid state" in caplog.text
```

In `test_ackman_pipeline.py`, in `test_a_review_runs_the_three_agents_in_order_and_places_the_orders`, replace `assert state.last_verdicts == {"AAA": "survive", "BBB": "fail"}` with:

```python
    assert state.last_verdicts == {
        "AAA": {"verdict": "survive", "reason": "survive reason", "concern": None, "date": "2026-09-14"},
        "BBB": {"verdict": "fail", "reason": "fail reason", "concern": "debt", "date": "2026-09-14"},
    }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_state.py tests/strategies/bill_ackman/test_ackman_pipeline.py -v`
Expected: FAIL (`ReviewState.__init__() got an unexpected keyword argument 'cooldowns'`).

- [ ] **Step 3: Implement in `state.py`**

Change the module docstring's first paragraph to say the store keeps "the fail counters, the re-entry cooldowns, the previous review's ranking and verdicts (with their reasons), and the abandoned-review streak". Set `STATE_VERSION = 2`. Replace `ReviewState`:

```python
@dataclass(frozen=True, slots=True)
class ReviewState:
    last_review: str | None = None  # ISO date of the last completed review
    fail_counts: dict[str, int] = field(default_factory=dict)  # holding -> consecutive fails (always >= 1)
    last_ranking: list[str] = field(default_factory=list)
    last_verdicts: dict[str, dict[str, Any]] = field(default_factory=dict)  # symbol -> {verdict, reason, concern, date}
    abandoned_streak: int = 0  # abandoned reviews in a row
    cooldowns: dict[str, int] = field(default_factory=dict)  # symbol -> completed reviews it stays out (always >= 1)
```

Add above `_valid`:

```python
_VERDICT_KEYS = {"verdict", "reason", "concern", "date"}


def _valid_counts(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool) and v >= 1 for k, v in value.items())


def _valid_verdict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == _VERDICT_KEYS
        and value["verdict"] in ("survive", "fail")
        and isinstance(value["reason"], str)
        and (value["concern"] is None or isinstance(value["concern"], str))
        and isinstance(value["date"], str)
    )
```

Replace `_valid`:

```python
def _valid(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("version") != STATE_VERSION:
        return False
    last_review = raw.get("last_review")
    ranking, verdicts = raw.get("last_ranking", []), raw.get("last_verdicts", {})
    streak = raw.get("abandoned_streak", 0)
    return (
        (last_review is None or isinstance(last_review, str))
        and _valid_counts(raw.get("fail_counts", {}))
        and _valid_counts(raw.get("cooldowns", {}))
        and isinstance(ranking, list)
        and all(isinstance(symbol, str) for symbol in ranking)
        and isinstance(verdicts, dict)
        and all(isinstance(k, str) and _valid_verdict(v) for k, v in verdicts.items())
        and isinstance(streak, int)
        and not isinstance(streak, bool)
        and streak >= 0
    )
```

In `StateStore.load`, change the `last_verdicts=` line to `last_verdicts={symbol: dict(record) for symbol, record in raw.get("last_verdicts", {}).items()},` and add `cooldowns=dict(raw.get("cooldowns", {})),` after `abandoned_streak=...`.

- [ ] **Step 4: Save the new shape from the pipeline**

In `pipeline.py` `_review`, step 10, replace the `ReviewState(...)` argument list with:

```python
ReviewState(
    last_review=now.date().isoformat(),
    fail_counts=outcome.fail_counts,
    last_ranking=ranked,
    last_verdicts={symbol: {"verdict": verdict.verdict, "reason": verdict.reason, "concern": verdict.concern, "date": now.date().isoformat()} for symbol, verdict in verdicts.items()},
    abandoned_streak=0,
    cooldowns=dict(state.cooldowns),
)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/state.py src/trading_agent_framework/strategies/bill_ackman/pipeline.py tests/strategies/bill_ackman/test_ackman_state.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "feat: bill_ackman state v2 keeps verdict reasons and cooldowns (Task 6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: pipeline enforces required holdings and cooldowns (D1)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/pipeline.py` (`_review`)
- Test: `tests/strategies/bill_ackman/test_ackman_pipeline.py`

**Interfaces:**
- Consumes: `HandoffRecorder.expect_portfolio(allowed, required=)` (Task 3), `advance_cooldowns` (Task 4), `AckmanParams.reentry_cooldown_reviews` (Task 5), `ReviewState.cooldowns` (Task 6).
- Produces: trader context key `"required"`; `reviews.jsonl` keys `"required"` and `"cooldowns"` (the cooldowns after the review); `"candidates"` in the log lists only the candidates the researcher saw.

- [ ] **Step 1: Write the failing tests**

Append to `test_ackman_pipeline.py` in the "holdings, hysteresis and forced exits" section:

```python
def test_a_pending_fail_the_trader_drops_is_refused_and_the_holding_is_kept(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100})
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="fail")]
    h.trader.steps = [holds(AAA=0.3), holds(AAA=0.3, HHH=0.2)]

    h.run()

    assert h.trader.calls[0]["context"]["required"] == ["HHH"]
    assert h.trader.calls[1]["force_tool"] == "submit_portfolio"
    assert "HHH failed once and is kept until a second consecutive fail" in h.trader.calls[1]["task"]
    assert ("HHH", "sell", 60.0) in h.orders  # shrunk from 50% to 20%, not sold out
    (line,) = h.log_lines()
    assert line["required"] == ["HHH"]


def test_a_forced_exit_starts_its_cooldown(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=ReviewState(fail_counts={"HHH": 1}))
    h.short_seller.steps = [judges(HHH="fail")]

    h.run()

    assert ("HHH", "sell", 100.0) in h.orders
    assert h.store.load().cooldowns == {"HHH": 4}
    assert h.log_lines()[0]["cooldowns"] == {"HHH": 4}


def test_a_symbol_cooling_down_is_hidden_from_the_researcher_and_its_cooldown_counts_down(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA", 1), _candidate("BBB", 2)]), state=ReviewState(cooldowns={"BBB": 2}))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive")]
    h.trader.steps = [holds(AAA=0.3)]

    h.run()

    assert [sheet["symbol"] for sheet in h.researcher.calls[0]["context"]["candidates"]] == ["AAA"]
    assert h.store.load().cooldowns == {"BBB": 1}
    (line,) = h.log_lines()
    assert [c["symbol"] for c in line["candidates"]] == ["AAA"]


def test_an_abandoned_review_leaves_the_cooldowns_alone(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([_candidate("AAA")]), state=ReviewState(cooldowns={"BBB": 2}))
    h.researcher.steps = [does_nothing, does_nothing]

    outcome = h.run()

    assert outcome.completed is False
    assert h.store.load().cooldowns == {"BBB": 2}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_pipeline.py -v -k "pending_fail or cooldown"`
Expected: FAIL (`KeyError: 'required'`, cooldowns stay `{}`).

- [ ] **Step 3: Implement in `pipeline.py`**

Import `advance_cooldowns` next to `apply_verdicts`: `from trading_agent_framework.strategies.bill_ackman.hysteresis import advance_cooldowns, apply_verdicts`.

In `_review`, replace `candidates = universe_result.candidates` with:

```python
        cooling = set(state.cooldowns)  # recent forced exits: kept out of the candidates and the allowed set
        candidates = [candidate for candidate in universe_result.candidates if candidate.symbol not in cooling]
```

After the `outcome = apply_verdicts(...)` block (step 7), add:

```python
        allowed = [symbol for symbol in outcome.allowed if symbol not in cooling]
        required = [symbol for symbol in outcome.pending if symbol in allowed]  # first fails: kept until a second one
```

In step 8, replace `if outcome.allowed:` with `if allowed:`, `self._recorder.expect_portfolio(outcome.allowed)` with `self._recorder.expect_portfolio(allowed, required=required)`, `for symbol in outcome.allowed` (in the context comprehension) with `for symbol in allowed`, and add `"required": required,` to the context dict right after the `"allowed": [...]` entry.

In step 10, compute the new cooldowns before saving:

```python
        cooldowns = advance_cooldowns(state.cooldowns, forced_exits=outcome.forced_exits, reviews=params.reentry_cooldown_reviews)
```

use `cooldowns=cooldowns,` in the `ReviewState(...)` (replacing `cooldowns=dict(state.cooldowns),`), and in the log record replace `"allowed": outcome.allowed,` with:

```python
                "allowed": allowed,
                "required": required,
                "cooldowns": cooldowns,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/pipeline.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "feat: bill_ackman keeps pending fails and cools forced exits down (Task 7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: pipeline gives the short seller memory and a budget, the researcher no anchor (D2, D4)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/pipeline.py` (`AgentLike`, `_review`, `_run_stage`)
- Test: `tests/strategies/bill_ackman/test_ackman_pipeline.py` (`FakeAgent.run`, one existing assertion, new tests)

**Interfaces:**
- Consumes: `AgentHandle.run(..., tool_budget=)` (Task 2), `HandoffRecorder.expect_verdicts(symbols, previous=)` (Task 3), `ReviewState.last_verdicts` records (Task 6).
- Produces: short-seller context item key `"previous_verdict"` (the stored record or `None`); researcher context without `"previous_ranking"`; `_run_stage(agent_name, tool, task, context, tool_budget: int | None = None)`.

- [ ] **Step 1: Write the failing tests**

In `test_ackman_pipeline.py`, replace `FakeAgent.run`:

```python
def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
    self.calls.append({"task": task_prompt, "context": context, "run_id": run_id, "force_tool": force_tool, "tool_budget": tool_budget})
    if self.steps:
        self.steps.pop(0)(self.tools, context)
    return AgentRunResult(output="done", tool_calls=list(self.tool_calls))
```

In `test_the_agents_get_the_context_the_design_promises`, replace `assert researcher["previous_ranking"] == ["OLD"]` with `assert "previous_ranking" not in researcher  # no anchoring on the last ranking`.

Add a new section at the end of the file:

```python
# --- short-seller memory and budget, researcher anchoring ---------------------------------------------

_SURVIVED = {"verdict": "survive", "reason": "fine", "concern": None, "date": "2026-09-07"}


def test_the_short_seller_sees_its_previous_verdict_and_gets_two_tool_calls_per_name(tmp_path: Path) -> None:
    screen = FakeScreen([_candidate("AAA", 1)], holdings=[_candidate("HHH")])
    h = _harness(tmp_path, screen, held={"HHH": 100}, state=ReviewState(last_verdicts={"HHH": _SURVIVED}))
    h.researcher.steps = [ranks("AAA")]
    h.short_seller.steps = [judges(AAA="survive", HHH="survive")]
    h.trader.steps = [holds(AAA=0.3, HHH=0.3)]

    h.run()

    to_judge = {entry["fact_sheet"]["symbol"]: entry for entry in h.short_seller.calls[0]["context"]["to_judge"]}
    assert to_judge["HHH"]["previous_verdict"] == _SURVIVED
    assert to_judge["AAA"]["previous_verdict"] is None
    assert h.short_seller.calls[0]["tool_budget"] == 4
    assert h.researcher.calls[0]["tool_budget"] is None and h.trader.calls[0]["tool_budget"] is None


def test_the_forced_retry_keeps_the_short_sellers_budget(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100})
    h.short_seller.steps = [does_nothing, judges(HHH="survive")]
    h.trader.steps = [holds(HHH=0.3)]

    h.run()

    assert [call["tool_budget"] for call in h.short_seller.calls] == [2, 2]


def test_a_flip_without_what_changed_is_refused_and_corrected(tmp_path: Path) -> None:
    h = _harness(tmp_path, FakeScreen([], holdings=[_candidate("HHH")]), held={"HHH": 100}, state=ReviewState(last_verdicts={"HHH": _SURVIVED}))
    bare_fail: Step = lambda tools, ctx: tools["submit_verdicts"]([{"symbol": "HHH", "verdict": "fail", "reason": "debt up", "concern": "debt"}])  # noqa: E731
    h.short_seller.steps = [bare_fail, judges(HHH="fail")]
    h.trader.steps = [holds(HHH=0.2)]

    h.run()

    assert "HHH was 'survive' at the previous review" in h.short_seller.calls[1]["task"]
    verdict = h.log_lines()[0]["verdicts"][0]
    assert (verdict["verdict"], verdict["concern"], verdict["what_changed"]) == ("fail", "debt", "new facts")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_pipeline.py -v`
Expected: FAIL (`KeyError: 'previous_verdict'`, `tool_budget` is `None`, `previous_ranking` still present).

- [ ] **Step 3: Implement in `pipeline.py`**

Change `AgentLike.run` to:

```python
def run(self, task_prompt: str, *, context: Mapping[str, Any] | None = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult: ...
```

In step 3 (researcher), delete the line `"previous_ranking": state.last_ranking,` from the context.

In step 6 (short seller), replace the block from `self._recorder.expect_verdicts(to_judge)` to the `_run_stage` call with:

```python
            previous = {symbol: state.last_verdicts[symbol] for symbol in to_judge if symbol in state.last_verdicts}
            self._recorder.expect_verdicts(to_judge, previous={symbol: record["verdict"] for symbol, record in previous.items()})
            context = {
                "current_datetime": now.isoformat(),
                "to_judge": [
                    {
                        "fact_sheet": sheets[symbol],
                        "researcher_reason": researcher_reasons.get(symbol),
                        "held": symbol in holdings,
                        "current_weight": weights.get(symbol, 0.0),
                        "fail_count": state.fail_counts.get(symbol, 0),
                        "previous_verdict": previous.get(symbol),
                    }
                    for symbol in to_judge
                ],
            }
            # One or two checks per name; past the budget only submit_verdicts runs (it is exempt), so a long
            # review set cannot overflow the model's context.
            self._run_stage("short_seller", "submit_verdicts", SHORT_SELLER_TASK, context, tool_budget=2 * len(to_judge))
```

Change `_run_stage`'s signature to `def _run_stage(self, agent_name: str, tool: str, task: str, context: dict[str, Any], tool_budget: int | None = None) -> None:`, add `\`tool_budget\` caps each attempt's research tool calls (see \`AgentHandle.run\`).` to its docstring, and change the agent call to:

```python
                result = agent.run(prompt, context=context, run_id=run_id, force_tool=force_tool, tool_budget=tool_budget)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/pipeline.py tests/strategies/bill_ackman/test_ackman_pipeline.py
git commit -m "feat: bill_ackman short seller remembers its verdicts and has a tool budget; researcher unanchored (Task 8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: prompts

**Files:**
- Modify: `src/trading_agent_framework/strategies/bill_ackman/prompts.py`
- Test: `tests/strategies/bill_ackman/test_ackman_prompts.py`

**Interfaces:**
- Consumes: `handoff.CONCERNS` (Task 3), context keys `required` (Task 7) and `previous_verdict` (Task 8).

- [ ] **Step 1: Write the failing tests**

Append to `test_ackman_prompts.py` (add `from trading_agent_framework.strategies.bill_ackman.handoff import CONCERNS`):

```python
def test_the_researcher_is_not_told_to_prefer_holdings_or_shown_the_last_ranking() -> None:
    assert "Prefer a stock we already hold" not in prompts.RESEARCHER_SYSTEM
    assert "previous review's ranking" not in prompts.RESEARCHER_SYSTEM


def test_the_researcher_is_warned_about_value_traps_and_concentration() -> None:
    assert "price_return_12m" in prompts.RESEARCHER_SYSTEM and "check why" in prompts.RESEARCHER_SYSTEM
    assert "different businesses and industries" in prompts.RESEARCHER_SYSTEM


def test_the_short_seller_knows_the_concerns_its_memory_and_the_news_rule() -> None:
    for concern in CONCERNS:
        assert concern in prompts.SHORT_SELLER_SYSTEM
    for phrase in ("previous_verdict", "what_changed", "A news event alone", "cheaper, not riskier"):
        assert phrase in prompts.SHORT_SELLER_SYSTEM


def test_the_trader_must_keep_required_holdings_and_is_no_longer_told_to_let_go() -> None:
    assert "let go" not in prompts.TRADER_SYSTEM
    assert "required" in prompts.TRADER_SYSTEM and "at least at the minimum weight" in prompts.TRADER_SYSTEM
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bill_ackman/test_ackman_prompts.py -v`
Expected: FAIL on the four new tests.

- [ ] **Step 3: Replace the three system prompts in `prompts.py`**

```python
RESEARCHER_SYSTEM = (
    "You are the researcher of a concentrated, long-only stock portfolio in the style of Bill Ackman: own just a few simple, "
    "high-quality companies and put real money behind them. At each review you receive fact sheets for the companies a quantitative "
    "screen selected and the stocks currently held.\n\n"
    "Your job: find the simple, predictable companies that make lots of cash and trade at a good price, and rank the best "
    "ones, best first.\n"
    "Read the fact sheet first; its numbers are computed by code, so do not recompute them. fcf_yield is free cash flow over "
    "market cap (higher is cheaper). fcf_margin_5y is the five-year average free cash flow over revenue. operating_margin_stdev "
    "is the volatility of the operating margin (lower is more predictable). revenue_cagr_5y is the growth rate. "
    "net_debt_to_operating_income is a debt multiple (lower is safer, negative means net cash); if debt_reported is false the "
    "debt figure is missing, so be suspicious of it. price_return_12m is the share price change over a year.\n"
    "A high fcf_yield after a large fall in price_return_12m can mean the market sees a problem the numbers do not show yet: "
    "check why the price fell before ranking it high; a cheap price alone is not a reason. When quality is similar, spread "
    "your ideas across different businesses and industries.\n"
    "Use the research tools only to check a specific claim, never to browse. Judge a stock we hold by the same standard as a "
    "new idea. You do not trade and have no order tool: a separate trader decides what to hold.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_ranking exactly once, with your ranked ideas: for each, the symbol (from the candidates "
    "only) and one short reason. If the tool returns an error, read it and call it again with a corrected argument. Once it "
    "returns status recorded, reply with one line and call no other tool."
)

SHORT_SELLER_SYSTEM = (
    "You are a short seller. Attack each idea you are given: too much debt, weak management, strong rivals, or a price that is "
    "too high. Say which ideas survive.\n\n"
    "For each symbol, survive means your attack failed: the company still looks simple, cash-generative, not over-indebted "
    "and not overpriced. fail means your attack succeeded and a prudent investor should not hold it. Start from the fact "
    "sheet (its numbers are computed by code; do not recompute them), then test a specific concern with the filing, news and "
    "price tools: heavy or rising debt, a falling or erratic margin, a new competitor, an accounting problem, a price far "
    "above what the cash flow supports. One or two checks per name is enough; your tool calls are limited.\n"
    "A news event alone (an outage, a downgrade, a price-target cut, a lawsuit headline) is not a broken thesis: fail a "
    "company only when its debt, margin, competition, management, accounting or valuation changed in a way the fact sheet or "
    "a filing supports. A fallen price with intact cash flow makes a company cheaper, not riskier.\n"
    "A symbol you judged at the last review shows your previous_verdict: start from it and change it only when something "
    "material changed since then. A holding shows a fail_count: a stock that has already failed once and fails again is "
    "sold, so do not fail a holding lightly.\n"
    "Sources can be wrong or stale: do not repeat a figure from a news item as fact when the fact sheet or a filing says "
    "otherwise.\n"
    "Judge every symbol in to_judge and no other symbol.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_verdicts exactly once, with one object per symbol: the symbol, the verdict (survive or "
    "fail) and one short reason. A fail also needs concern: one of debt, margin, competition, management, accounting or "
    "valuation. When a verdict differs from your previous_verdict, also give what_changed: one short sentence on what changed "
    "since then. If the tool returns an error, read it and call it again with a corrected argument. Once it returns status "
    "recorded, reply with one line and call no other tool."
)

TRADER_SYSTEM = (
    "You are the trader of a concentrated, long-only stock portfolio. Hold the few ideas that survived, with more money in "
    "the best ones.\n\n"
    "You choose only from the allowed list in the context. Each name there has its research rank, the short seller's verdict "
    "and reason, a pending_fail_count (a holding that failed once), its current_weight and its fact sheet. The names in "
    "required failed once while held: each must stay in your portfolio at least at the minimum weight; you may reduce it, "
    "and code sells it if it fails again. Names in forced_exits are sold by code and are not allowed.\n"
    "Size each position as a fraction of portfolio value, for example 0.25, within the minimum and maximum weight given in "
    "the constraints; the weights together must not exceed the maximum total. Do not hold more positions than allowed. Leave "
    "money unallocated when fewer names deserve it: code parks it in short-term Treasuries (SHV), so cash is never a reason "
    "to hold a weak stock. An empty list means hold nothing. Prefer changing little: keep a current holding near its current "
    "weight unless the ranking or the verdict gives a reason to change it. You do not place orders; code does.\n\n"
    f"{_ENGLISH}\n\n"
    "End your run by calling submit_portfolio exactly once, with one object per stock to hold: the symbol, the weight and one "
    "short reason. If the tool returns an error, read it and call it again with a corrected argument. Once it returns status "
    "recorded, reply with one line and call no other tool."
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/strategies/bill_ackman/ -v`
Expected: PASS (including `test_no_prompt_hardcodes_the_review_cadence` and `test_every_prompt_requires_english_and_names_its_submit_tool`).

- [ ] **Step 5: Full suite, lint, commit**

```bash
uv run pytest && uv run ruff check
git add src/trading_agent_framework/strategies/bill_ackman/prompts.py tests/strategies/bill_ackman/test_ackman_prompts.py
git commit -m "feat: bill_ackman prompts drop the anchoring and the let-go rule, add the thesis rules (Task 9)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the notes**

1. In the `strategies/` architecture bullet, change "`StateStore` keeps the fail counters, the last ranking and verdicts and the abandoned streak" to "`StateStore` keeps the fail counters, the re-entry cooldowns, the last ranking, the last verdicts (with reason, concern and date) and the abandoned streak (`STATE_VERSION` 2: an older file loads as an empty state)".
2. In "**The Ackman agents never trade, and only their submit tools are trusted.**", append: "A short seller's `fail` must name a `concern` (`handoff.CONCERNS`), and a verdict that differs from its `previous_verdict` (shown in its context) must say `what_changed`. The short seller runs with `tool_budget = 2 x len(to_judge)` and `exempt_tools=["submit_verdicts"]`; all three agents run at `AckmanParams.agent_temperature` (0.3). The researcher ranks `research_top_n` (8) ideas and is not shown its previous ranking."
3. In "**Ackman hysteresis and open orders.**", replace "code sells it whatever the trader submits (it is not in the trader's allowed set; its counter is kept so a failed sell is forced again)" with "code sells it whatever the trader submits (it is not in the trader's allowed set; its counter is kept so a failed sell is forced again). A holding at its first fail is `required`: `submit_portfolio` refuses a portfolio without it at `min_weight` or more (the trader may shrink it, not drop it; its old prompt told it to let go of a failed name, and it dropped every pending fail). A forced exit then starts a re-entry cooldown (`reentry_cooldown_reviews`, 4 completed reviews; `hysteresis.advance_cooldowns`): the symbol is removed from the researcher's candidates and the allowed set, and an abandoned review does not count down".
4. After the "**`Strategy.agents.create(..., timeout_seconds=None)`**" bullet, add a bullet: "**`agents.create(temperature=None, exempt_tools=None)` and `AgentHandle.run(tool_budget=None)` are opt-in.** `temperature=None` sends nothing (the server default applies); the telemetry summary records each agent's temperature. A per-run `tool_budget` is enforced by a `wrap_tool_call` middleware installed on every agent (a no-op without a budget): past the budget every tool but `exempt_tools` returns `{\"error\": \"tool budget of N calls spent; <tool> was not run. ...\"}` without running. The message is built only from the caller's own tool names."

- [ ] **Step 2: Lint, full suite, commit**

```bash
uv run pytest && uv run ruff check
git add CLAUDE.md
git commit -m "docs: bill_ackman stability fixes in CLAUDE.md (Task 10)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

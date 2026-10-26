# bull_bear thinking budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every agent turn can carry vLLM's `thinking_token_budget`, so a turn that thinks too long is cut by the server and still submits; bull_bear sets one budget for the researcher/bull/bear and one for the judge, below raised output caps.

**Architecture:** The agent layer (`agents/manager.py`) gains an opt-in `thinking_token_budget` on `AgentManager.create`, sent by a new `wrap_model_call` middleware in the request body (`extra_body`), with the same refusal fallback as the reasoning effort, a WARNING when a call's reasoning reaches the budget, and a WARNING at creation when the budget is not below the output cap. bull_bear adds two parameters and raises its two caps.

**Tech Stack:** Python 3.14, LangChain `create_agent` middleware, `langchain_openai.ChatOpenAI`, vLLM 0.30 (`thinking_token_budget`), pytest with hand-written fakes.

**Spec:** `docs/superpowers/specs/2026-10-26-bull-bear-thinking-budget-design.md`

## Global Constraints

- Work on branch `feat/bull-bear-thinking-budget` (from `main`); never touch the user's uncommitted `TODO.md`.
- Hand-written fakes (`tests/fakes.py`, `RecordingToolCallingChatModel`), no `MagicMock`, no network in tests.
- `agents/manager.py` stays the only module importing `langchain`/`langchain_openai`, and only inside method bodies.
- TDD: failing test first. Run `uv run pytest` and `uv run ruff check` before each commit.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Values: `thinking_token_budget` 28672, `judge_thinking_token_budget` 36864, `max_output_tokens` 36864 (was 32768), `judge_max_output_tokens` 45056 (was 40960).
- Detection margin: a call's reasoning is "at the budget" when `reasoning_tokens >= budget - max(64, budget // 100)`.
- Log wording (polish allowed, keep the facts): `[judge] thinking budget of 36864 tokens reached (36855 reasoning tokens): the server ended the thinking`; `[judge] thinking budget of 45056 tokens is not below the output cap of 45056: the cap stops the turn first and the budget never fires`.

## Review Focus

1. An `extra_body` already set on the chat model (`ChatOpenAI(extra_body=...)`) must survive: the budget is merged into it, not replacing it. Test in Task 1.
2. A refusal resend keeps the output cap and the reasoning effort (the budget middleware sits between them). Test in Task 1.
3. A response with no reported reasoning tokens (or no AI message) logs nothing and never raises. Test in Task 2.
4. A forced retry (`_run_stage`'s second run through the same agent) gets the budget too: it is per agent, not per run. Covered by Task 1's "every model call of the run" test plus Task 3's wiring test.
5. `thinking_token_budget=None` with a cap set, and a budget with no cap: no warning, nothing sent / budget sent. Test in Task 1.

---

### Task 1: `create(thinking_token_budget=...)` sends the budget, with refusal fallback, cap warning and telemetry

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (`create`, `__init__` state, `telemetry_summary`, new `_thinking_budget_middleware`)
- Modify: `src/trading_agent_framework/agents/CLAUDE.md` (the `create()` options paragraph and the middleware order)
- Test: `tests/agents/test_agent_manager_thinking_budget.py` (new; mirror `tests/agents/test_agent_manager_reasoning_effort.py`: `_manager`, `_with_model`, `_RefusingModel`, `_lookup`)

**Interfaces:**
- Produces: `AgentManager.create(..., thinking_token_budget: int | None = None, ...) -> AgentHandle`. Raises `ValueError("thinking_token_budget must be at least 1, got N")` for N < 1. `telemetry_summary()[agent]["thinking_token_budget"]: int | None`.
- The budget reaches the model as `model_settings["extra_body"] = {**existing, "thinking_token_budget": N}`, where `existing` is the request's own `model_settings.get("extra_body")` merged over `getattr(request.model, "extra_body", None)` (both may be `None`).

- [ ] **Step 1: Write the failing tests** (helper `_budgets(model) = [(call.get("extra_body") or {}).get("thinking_token_budget") for call in model.calls]`; refusal text `_BUDGET_REFUSAL = "Error code: 400 - {'message': '`thinking_token_budget` must be a non-negative integer or -1 for unlimited.', 'code': 400}"`)

```python
def test_every_model_call_of_the_run_is_sent_the_budget(...)  # two calls (tool call, then "done"), budget 28672
    assert _budgets(model) == [28672, 28672]

def test_without_a_budget_nothing_is_sent(...)
    assert "extra_body" not in model.calls[0] or "thinking_token_budget" not in model.calls[0]["extra_body"]

def test_a_prebuilt_model_gets_the_budget()  # create(model=prebuilt, thinking_token_budget=500)
    assert _budgets(model) == [500]

def test_the_budget_is_merged_into_an_existing_extra_body(...)  # fake model with extra_body={"top_k": 20}
    assert model.calls[0]["extra_body"] == {"top_k": 20, "thinking_token_budget": 500}

def test_a_budget_below_one_is_refused()
    with pytest.raises(ValueError, match="thinking_token_budget must be at least 1"):
        _manager(None).create(name="a", system_prompt="x", thinking_token_budget=0)

def test_a_refusal_naming_thinking_token_budget_is_resent_without_it(...)  # caplog
    assert _budgets(model) == [500, None]; assert result.output == "done"
    # exactly one WARNING, naming thinking_token_budget=500

def test_after_a_refusal_no_agent_of_the_manager_sends_it(...)  # as the effort test: refused agent and a second agent
    assert _budgets(refused) == [500, None, None]; assert _budgets(other) == [None]

def test_the_resend_keeps_the_output_cap_and_the_effort(...)  # effort XHIGH, max_output_tokens=4096, budget 500
    assert [(c.get("max_tokens"), c.get("reasoning_effort"), _budget(c)) for c in model.calls] == [(4096, "xhigh", 500), (4096, "xhigh", None)]

def test_any_other_failure_is_not_retried(...)  # ValueError("connection reset") -> AgentError, one call

def test_a_budget_not_below_the_cap_is_warned_about_at_creation(caplog)  # cap 4096, budget 4096
    # one WARNING containing "thinking budget of 4096 tokens is not below the output cap of 4096"; the agent is created

def test_no_cap_warning_when_the_budget_is_below_the_cap_or_either_is_unset(caplog)  # (4096, 4095), (None, 500), (4096, None)
    assert no WARNING records

def test_the_telemetry_summary_records_each_agents_budget(...)  # agent with 500, agent without
    assert summary["with"]["thinking_token_budget"] == 500 and summary["without"]["thinking_token_budget"] is None

def test_the_telemetry_summary_shows_no_budget_after_a_refusal(...)
    assert manager.telemetry_summary()["a"]["thinking_token_budget"] is None

def test_a_real_chat_openai_puts_the_budget_in_the_request_body() -> None:
    chat = ChatOpenAI(model="qwen3-8b", base_url="http://localhost:1/v1", api_key=SecretStr("k"), extra_body={"top_k": 20})
    bound = chat.bind(extra_body={"top_k": 20, "thinking_token_budget": 500})
    payload = chat._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    assert payload["extra_body"] == {"top_k": 20, "thinking_token_budget": 500}
    assert "thinking_token_budget" not in payload  # never a top-level OpenAI client argument
```

For the merge test, give the fake model an `extra_body` attribute (subclass `RecordingToolCallingChatModel` with `extra_body: dict[str, Any] | None = None`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/test_agent_manager_thinking_budget.py -v`
Expected: FAIL (`create()` got an unexpected keyword argument `thinking_token_budget`).

- [ ] **Step 3: Implement in `agents/manager.py`**

- `__init__`: `self._thinking_budgets: dict[str, int | None] = {}` and `self._thinking_budget_refused = False`.
- `create(...)`: validate like `max_output_tokens`; when both are set and `thinking_token_budget >= max_output_tokens`, `logger.log_warning(...)` (wording in Global Constraints); register `self._thinking_budgets[name]`.
- `_thinking_budget_middleware(self, budget: int) -> Any`: a `wrap_model_call` shaped like `_reasoning_effort_middleware`: skip when `self._thinking_budget_refused`; send via `request.override(model_settings={**request.model_settings, "extra_body": merged})`; on an exception whose text contains `"thinking_token_budget"`, set the refused flag, log one WARNING `the server refused thinking_token_budget=N: sent without it for the rest of the run (...)`, and return `handler(request)`; any other exception propagates.
- Stack order in `create()`: `..., caller middleware, output cap, thinking budget, reasoning effort, compaction, telemetry` (insert the budget between the cap and the effort).
- `telemetry_summary()`: `totals["thinking_token_budget"] = None if self._thinking_budget_refused else self._thinking_budgets.get(agent)`.
- Docstrings of `create` and `telemetry_summary`: add the option, in the style of the effort's paragraph.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/ -v` then `uv run pytest -q` and `uv run ruff check`
Expected: all pass.

- [ ] **Step 5: Update `agents/CLAUDE.md`**: in the `agents.create(...)` paragraph, add `thinking_token_budget` (opt-in; vLLM field sent in `extra_body`, merged with the model's own; refusal fallback and run-wide drop like the effort; also sent to a pre-built model; a budget not below the cap is warned about; telemetry records it) and add it to the "Order in `create()`" sentence between the output cap and the reasoning effort.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/manager.py src/trading_agent_framework/agents/CLAUDE.md tests/agents/test_agent_manager_thinking_budget.py
git commit -m "feat(agents): opt-in thinking_token_budget per agent, sent in the request body (Task 1)"
```

### Task 2: a WARNING when a call's reasoning reaches the budget

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py` (`_thinking_budget_middleware`; it needs the agent name: `_thinking_budget_middleware(self, agent_name: str, budget: int)`)
- Modify: `src/trading_agent_framework/agents/CLAUDE.md` (one sentence in the paragraph Task 1 wrote)
- Test: `tests/agents/test_agent_manager_thinking_budget.py`

**Interfaces:**
- Consumes: Task 1's middleware and `usage_from_message` (`agents/telemetry.py`); pick the response's last AI message the way `_record_call` does (`getattr(response, "result", None) or [response]`, last message with `type == "ai"`); factor that pick into one small helper both use rather than copying it.

- [ ] **Step 1: Write the failing tests** (AIMessages carry `usage_metadata={"input_tokens": 10, "output_tokens": N, "total_tokens": N + 10, "output_token_details": {"reasoning": R}}`)

```python
def test_reasoning_at_the_budget_is_logged(...)  # budget 1000, R = 936 (1000 - max(64, 10))
    # one WARNING containing "thinking budget of 1000 tokens reached (936 reasoning tokens)" and the agent name

def test_reasoning_just_outside_the_margin_is_not_logged(...)  # budget 1000, R = 935
    assert no WARNING

def test_a_large_budget_uses_a_one_percent_margin(...)  # budget 36864: R = 36496 logs, R = 36495 does not

def test_no_reported_reasoning_logs_nothing(...)  # AIMessage without usage_metadata; and one with usage but no output_token_details
    assert no WARNING and the run's output is unchanged

def test_after_a_refusal_nothing_is_logged_even_at_the_old_budget(...)  # refused budget 500, then a call with R = 500
    assert only the refusal WARNING
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/agents/test_agent_manager_thinking_budget.py -v`
Expected: the new tests FAIL (no WARNING logged).

- [ ] **Step 3: Implement**: after `handler(...)` returns with the budget sent, read the AI message's reasoning tokens; when not `None` and `>= budget - max(64, budget // 100)`, `logger.log_warning(f"[{agent_name}] thinking budget of {budget} tokens reached ({reasoning} reasoning tokens): the server ended the thinking")`. Never raise from this check.

- [ ] **Step 4: Run the tests**: `uv run pytest tests/agents/ -v`, `uv run pytest -q`, `uv run ruff check`. Expected: all pass.

- [ ] **Step 5: `agents/CLAUDE.md`**: add that a call whose reported reasoning comes within `max(64, 1%)` of the budget logs a WARNING, and why the margin (a cut turn reports slightly fewer reasoning tokens than the budget: 99 for 100, 491 for 500, measured 2026-10-26).

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/manager.py src/trading_agent_framework/agents/CLAUDE.md tests/agents/test_agent_manager_thinking_budget.py
git commit -m "feat(agents): log a WARNING when a call's reasoning reaches its thinking budget (Task 2)"
```

### Task 3: bull_bear sets the budgets and raises the caps

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py` (two new fields, two new values, comments, validation)
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` (`strategy.parameters` record near line 79; the four `self.agents.create(...)` calls near lines 116-139)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the "Every agent's turns are capped" paragraph)
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `tests/strategies/bull_bear/test_bull_bear_strategy.py`

**Interfaces:**
- Consumes: Task 1's `create(thinking_token_budget=...)`.
- Produces: `BullBearParams.thinking_token_budget: int | None = 28672`, `BullBearParams.judge_thinking_token_budget: int | None = 36864`, `max_output_tokens = 36864`, `judge_max_output_tokens = 45056`; `strategy.parameters` gains both budget keys.

- [ ] **Step 1: Write the failing tests**

```python
# test_bull_bear_params.py (update the two existing cap asserts, add the budgets)
assert params.max_output_tokens == 36864 and params.judge_max_output_tokens == 45056
assert params.thinking_token_budget == 28672 and params.judge_thinking_token_budget == 36864
assert params.max_output_tokens - params.thinking_token_budget == 8192
assert params.judge_max_output_tokens - params.judge_thinking_token_budget == 8192
# validation: BullBearParams(thinking_token_budget=0) and BullBearParams(judge_thinking_token_budget=0) raise with
# "thinking_token_budget must be None or at least 1" / "judge_thinking_token_budget must be None or at least 1"
# (use the file's existing pattern for the cap validation tests)

# test_bull_bear_strategy.py
def test_each_agent_gets_its_thinking_budget_next_to_its_cap(tmp_path)  # BullBearParams(thinking_token_budget=7000, judge_thinking_token_budget=15000, max_output_tokens=9000, judge_max_output_tokens=20000)
    budgets = {c["name"]: c["thinking_token_budget"] for c in agents.created}
    assert budgets == {"researcher": 7000, "bull": 7000, "bear": 7000, "judge": 15000}
    assert (strategy.parameters["thinking_token_budget"], strategy.parameters["judge_thinking_token_budget"]) == (7000, 15000)

def test_a_judge_budget_of_none_means_no_budget_not_the_shared_one(tmp_path)  # judge_thinking_token_budget=None
    assert budgets["judge"] is None and budgets["bull"] == 28672
# and update the existing `caps["bull"] == 32768` assert to 36864
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_strategy.py -v`
Expected: FAIL (no `thinking_token_budget` field; old cap values).

- [ ] **Step 3: Implement**

- `parameters.py`: the four values; validation messages as in the tests, next to the caps' own. **Comments beside the four parameters** (the user's request; replace the current cap comments), covering: the budget is a safety net set high on purpose, a healthy turn should never reach it, a budget below normal use would cut good reasoning; each cap is its budget + 8,192, the room after a cut for the submission (one run per review covers every stock, up to about 25: the judge's reason up to 300 characters per stock plus `what_changed` up to 300 on each flip, about 15,000 characters or 4-5k tokens at most; the bull's and bear's argument up to 300 per stock; the rest is margin for text written after the cut); the cap is the backstop for an answer that runs on and a server that ignores the budget; the window check (prompts peaked near 20k in the 10-25 run; 20k + 45,056 is about 65k of the 81,920 window); `judge_thinking_token_budget=None` means no budget for the judge, not the shared one.
- `agent_bull_bear.py`: record both budgets in `strategy.parameters` next to the caps; pass `thinking_token_budget=self.settings.thinking_token_budget` to the researcher, bull and bear, and `thinking_token_budget=self.settings.judge_thinking_token_budget` to the judge.

- [ ] **Step 4: Run the tests**: `uv run pytest -q`, `uv run ruff check`. Expected: all pass.

- [ ] **Step 5: `bull_bear/CLAUDE.md`**: in "Every agent's turns are capped", state the three layers (thinking budget, then the output cap as backstop, then compaction), the new values (budgets 28672 / 36864, caps 36864 / 45056, both recorded in `settings.json`), that a cut turn still submits instead of a forced retry, the `None` rule for the judge's budget, and that compaction's summary call gets neither the budget nor the cap.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/parameters.py src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_strategy.py
git commit -m "feat(bull_bear): thinking budgets 28672 / 36864 under caps raised to 36864 / 45056 (Task 3)"
```

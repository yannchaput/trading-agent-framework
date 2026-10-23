# LLM Reasoning Effort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An optional `LLM_REASONING_EFFORT` env variable (`low`/`medium`/`high`/`xhigh`) that every agent of a strategy sends as `"reasoning_effort"` in the request body, falling back to sending nothing if the server refuses it, and recorded in the run's telemetry.

**Architecture:** `agents/config.py` parses the variable into a `ReasoningEffort` `StrEnum` on `LLMCredentials`. `AgentManager.create()` adds a `wrap_model_call` middleware (built like `output_cap_middleware`) that puts `reasoning_effort` into `model_settings`. A refusal mentioning `reasoning_effort` is resent once without it, and from then on the manager sends it to no agent. `telemetry_summary()` records the effort per agent next to `temperature`, and the dashboard's Parameters tab shows it.

**Tech Stack:** Python 3.14, LangChain `create_agent` middleware, `langchain_openai.ChatOpenAI` (merges `model_settings` kwargs into the request payload), pytest with the hand-written fakes in `tests/fakes.py`.

**Spec:** no separate file. The design was approved in chat on 2026-10-23 and is summarized here:
- The user's case: Qwen3.8-27B gives better results with `reasoning_effort=xhigh`, sent as a field of the request payload.
- There is one effort for every agent of a strategy, set in the strategy's env file next to `LLM_MODEL`.
- Allowed values are an enum: `low`, `medium`, `high`, `xhigh`. Any other value is a `ConfigurationError`. Case and surrounding spaces are ignored. Blank or missing means nothing is sent.
- If the server refuses the field, the strategy logs one warning, resends that call without it, and sends it on no later call of the run.
- The compaction summary call does not get it: it bypasses every `wrap_model_call` middleware.
- The telemetry summary records the effort actually sent, per agent, like `temperature`. The dashboard shows it under the agent's Model row.

## Global Constraints

- Env variable name: `LLM_REASONING_EFFORT`. Wire field name: `reasoning_effort`. Values, in this order: `low`, `medium`, `high`, `xhigh`.
- Error message, exactly: `LLM_REASONING_EFFORT must be one of low, medium, high, xhigh, got {raw!r}`.
- `agents/config.py` stays pure (no I/O). Every LangChain import in `agents/manager.py` stays inside method bodies.
- Like `temperature` and `timeout_seconds`, the effort applies only to a model `AgentManager` builds from `LLMCredentials` (`model` is `None` or a string). A pre-built `BaseChatModel` passed as `model` gets none.
- Middleware order in `create()`, outermost first: tool-call repair, forced tool choice, tool budget, final tools, the caller's middleware, output cap, **reasoning effort**, compaction, telemetry.
- Dashboard label `Reasoning effort`. A `None` value reads `server default`. A run whose agent dict has no `reasoning_effort` key gets no row.
- Commit messages follow the repo's style (`feat(agents): ...`) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **The real vLLM refusal text.** Detection is the substring `reasoning_effort` in `str(exc)`. A `openai.BadRequestError`-style message (`Error code: 400 - {'object': 'error', 'message': "1 validation error ... reasoning_effort ..."}`) must trigger the fallback. Pinned in Task 2 by `test_a_refusal_naming_reasoning_effort_is_resent_without_it`.
2. **A second agent after a refusal.** After one agent's call was refused, another agent of the same manager must not send the field at all. Its first call must not be refused again. Pinned in Task 2 by `test_after_a_refusal_no_agent_of_the_manager_sends_it`.
3. **Effort combined with the output cap.** A call refused for `reasoning_effort` must be resent with its `max_tokens` intact. Pinned in Task 2 by `test_the_resend_keeps_the_output_cap`.
4. **Messy env values.** `" XHIGH "` must read as `xhigh`. `"extreme"` and `"none"` must be refused at startup, not sent. Pinned in Task 1.
5. **Pre-built models.** A strategy that passes its own `BaseChatModel` must not get the field, even with `LLM_REASONING_EFFORT` set. Pinned in Task 2 by `test_a_prebuilt_model_gets_no_reasoning_effort`.

---

### Task 1: `LLM_REASONING_EFFORT` in `LLMCredentials`

**Files:**
- Modify: `src/trading_agent_framework/agents/config.py`
- Modify: `env/.env.example` (after the `LLM_MAX_INPUT_TOKENS` block, line 59)
- Modify: `README.md` (the `## ✏️ Environment variables` table, line 64)
- Modify: `src/trading_agent_framework/agents/CLAUDE.md` (the `config.py` description in Architecture)
- Test: `tests/agents/test_agent_config.py`

**Interfaces:**
- Produces: `class ReasoningEffort(StrEnum)` in `agents/config.py`, with members `LOW = "low"`, `MEDIUM = "medium"`, `HIGH = "high"`, `XHIGH = "xhigh"`. `LLMCredentials.reasoning_effort: ReasoningEffort | None = None`, a new last field.

- [ ] **Step 1: Write the failing tests** in `tests/agents/test_agent_config.py`

```python
def test_from_env_reasoning_effort_is_optional() -> None:
    assert LLMCredentials.from_env({"LLM_BASE_URL": "http://localhost:8000/v1"}).reasoning_effort is None


@pytest.mark.parametrize(("value", "expected"), [("xhigh", ReasoningEffort.XHIGH), (" XHIGH ", ReasoningEffort.XHIGH), ("low", ReasoningEffort.LOW), ("Medium", ReasoningEffort.MEDIUM), ("high", ReasoningEffort.HIGH), ("", None), ("   ", None)])
def test_from_env_reads_reasoning_effort(value: str, expected: ReasoningEffort | None) -> None:
    credentials = LLMCredentials.from_env({"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_REASONING_EFFORT": value})
    assert credentials.reasoning_effort == expected


@pytest.mark.parametrize("value", ["extreme", "none", "minimal", "x-high"])
def test_from_env_refuses_an_unknown_reasoning_effort(value: str) -> None:
    with pytest.raises(ConfigurationError, match=r"LLM_REASONING_EFFORT must be one of low, medium, high, xhigh, got"):
        LLMCredentials.from_env({"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_REASONING_EFFORT": value})
```

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/agents/test_agent_config.py -v`
Expected: collection error, `ImportError: cannot import name 'ReasoningEffort'`.

- [ ] **Step 3: Implement** `ReasoningEffort` and `_reasoning_effort(source: Mapping[str, str]) -> ReasoningEffort | None` in `agents/config.py`

Model the parser on `_max_input_tokens`: strip, then lowercase, and treat blank as `None`. Build the message's value list from the enum (`", ".join(effort.value for effort in ReasoningEffort)`). Pass the parsed value into `from_env`'s `cls(...)`. Give the field a one-line comment like `max_input_tokens` has.

- [ ] **Step 4: Run the tests and check they pass**

Run: `uv run pytest tests/agents/test_agent_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Document the variable**
  - `env/.env.example`: add a commented, blank `LLM_REASONING_EFFORT=` block after `LLM_MAX_INPUT_TOKENS`. It should say: optional; one of low/medium/high/xhigh; sent as `reasoning_effort` with every agent call; blank or missing sends nothing; a server that refuses it gets one warning and the run carries on without it.
  - `README.md` env table: add the row `| `LLM_REASONING_EFFORT` | every agent's model calls (`reasoning_effort` in the request) | optional: `low`, `medium`, `high` or `xhigh`; blank sends nothing |`.
  - `agents/CLAUDE.md` Architecture, `config.py` description: after the `LLM_MAX_INPUT_TOKENS` clause, add "`LLM_REASONING_EFFORT`, optional, a `ReasoningEffort` (`low`/`medium`/`high`/`xhigh`, case-insensitive) or a `ConfigurationError`".

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/config.py tests/agents/test_agent_config.py env/.env.example README.md src/trading_agent_framework/agents/CLAUDE.md
git commit -m "feat(agents): LLM_REASONING_EFFORT env variable (low/medium/high/xhigh)"
```

---

### Task 2: Send `reasoning_effort` with every agent call, with fallback and telemetry

**Files:**
- Modify: `src/trading_agent_framework/agents/manager.py`: `__init__` (line 196), `telemetry_summary` (line 215), `create` (lines 380-461), and a new `_reasoning_effort_middleware` next to the other `_..._middleware` methods
- Modify: `src/trading_agent_framework/agents/CLAUDE.md` (the `agents.create(temperature=None, ...)` gotcha and its middleware-order sentence)
- Test: `tests/agents/test_agent_manager_reasoning_effort.py` (new)

**Interfaces:**
- Consumes: `ReasoningEffort` and `LLMCredentials.reasoning_effort` from Task 1.
- Produces:
  - `telemetry_summary()[agent]["reasoning_effort"]` is a `str | None`: the effort's value, or `None` when none was configured, the model was pre-built, or the server refused it during the run. Task 3 reads this key from `settings.json["agents"]`.
  - `AgentManager._reasoning_effort_middleware(self, effort: ReasoningEffort) -> Any`.
  - `AgentManager._reasoning_effort_refused: bool`, starting `False`.
  - `AgentManager._reasoning_efforts: dict[str, ReasoningEffort | None]`.

- [ ] **Step 1: Write the failing tests** in `tests/agents/test_agent_manager_reasoning_effort.py`

The setup is as follows:
- `_manager(effort)` returns `AgentManager(lambda: LLMCredentials(base_url="http://localhost:8000/v1", api_key="k", default_model="qwen3-8b", reasoning_effort=effort))`.
- `_with_model(manager, model, monkeypatch)` does `monkeypatch.setattr(manager, "_resolve_model", lambda *args, **kwargs: model)`. That makes `create(model=None)` take the credentials path while running the fake.
- Reuse the `_lookup` tool, the `_lookup_call()` helper and a `_RefusingModel` like `tests/agents/test_agent_output_cap.py`'s, copied into this file.
- `_EFFORT_REFUSAL = "Error code: 400 - {'object': 'error', 'message': \"1 validation error for ChatCompletionRequest\\nreasoning_effort\\n  Input should be 'low', 'medium' or 'high'\", 'type': 'BadRequestError', 'code': 400}"`.

Tests (name: assertion):
- `test_every_model_call_of_the_run_is_sent_the_effort`: with `XHIGH`, a 2-turn run (`_lookup_call()`, then `"done"`) gives `[call.get("reasoning_effort") for call in model.calls] == ["xhigh", "xhigh"]`.
- `test_without_an_effort_nothing_is_sent`: with `None`, `"reasoning_effort" not in model.calls[0]`.
- `test_a_prebuilt_model_gets_no_reasoning_effort`: with `XHIGH` and `create(model=fake)` (no monkeypatch), `"reasoning_effort" not in model.calls[0]`.
- `test_a_refusal_naming_reasoning_effort_is_resent_without_it`: `_RefusingModel(error=ValueError(_EFFORT_REFUSAL))`. Expect `[call.get("reasoning_effort") for call in model.calls] == ["xhigh", None]`, `result.output == "done"`, and exactly one WARNING record in `caplog` containing `reasoning_effort=xhigh`.
- `test_after_a_refusal_no_agent_of_the_manager_sends_it`: after the refused run above, a second run of the same agent and a run of a second agent created on the same manager send no `reasoning_effort` key at all. No second warning is logged.
- `test_the_resend_keeps_the_output_cap`: with `max_output_tokens=4096` and the refusal, `[(c.get("max_tokens"), c.get("reasoning_effort")) for c in model.calls] == [(4096, "xhigh"), (4096, None)]`.
- `test_any_other_failure_is_not_retried`: `ValueError("connection reset")` raises `AgentError` matching `connection reset`, and `len(model.calls) == 1`.
- `test_the_telemetry_summary_records_each_agents_effort`: with `enable_telemetry(now=lambda: T0)`, an agent on the credentials path reads `"xhigh"`, and an agent with a pre-built model reads `None`.
- `test_the_telemetry_summary_shows_no_effort_after_a_refusal`: after a refused-then-resent run, `summary["a"]["reasoning_effort"] is None`.

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/agents/test_agent_manager_reasoning_effort.py -v`
Expected: FAIL. The send tests fail because `reasoning_effort` is absent from calls; the summary tests fail with `KeyError: 'reasoning_effort'`.

- [ ] **Step 3: Implement** in `agents/manager.py`
  - `__init__`: add `self._reasoning_efforts = {}` and `self._reasoning_effort_refused = False`.
  - `create()`: set `effort = None if model is not None and not isinstance(model, str) else self._credentials_source().reasoning_effort`. Append `self._reasoning_effort_middleware(effort)` to `stack` right after the output cap, when `effort is not None`. Record `self._reasoning_efforts[name] = effort` next to `self._temperatures[name]`. Add a docstring paragraph after the `max_output_tokens` one: the effort comes from `LLM_REASONING_EFFORT` only for a model built here; it is refused once, then dropped for the whole manager; the compaction summary call never gets it. Update the `middleware` paragraph's list of inner middleware to name the reasoning effort too.
  - `_reasoning_effort_middleware(effort)`: a `wrap_model_call` (import inside the method). When `self._reasoning_effort_refused` is set, it calls `handler(request)` unchanged. Otherwise it sends `request.override(model_settings={**request.model_settings, "reasoning_effort": effort.value})`. On an exception whose `str()` contains `"reasoning_effort"`, it sets the flag, logs `logger.log_warning(f"the server refused reasoning_effort={effort.value}: sent without it for the rest of the run ({exc})")` and returns `handler(request)`. Any other exception is re-raised. A second refusal of the resend propagates.
  - `telemetry_summary()`: set `totals["reasoning_effort"]` to `None` when the flag is set or the agent's effort is `None`, and to the effort's `.value` otherwise. Extend the docstring to mention it.

- [ ] **Step 4: Run the tests and check they pass**

Run: `uv run pytest tests/agents/ -v`
Expected: all PASS, including the existing output-cap, temperature and telemetry tests.

- [ ] **Step 5: Document the behaviour** in `agents/CLAUDE.md`. In the `agents.create(temperature=None, ...)` gotcha, after the `max_output_tokens` sentences, add one sentence covering:
  - `LLM_REASONING_EFFORT` sends `reasoning_effort` with every agent turn of a model built from `LLMCredentials` (a pre-built model gets none).
  - A refusal naming `reasoning_effort` is resent once without it, with one WARNING, and the manager drops it for every later call.
  - vLLM usually ignores the field silently for a model that doesn't use it, so the fallback only catches an explicit refusal.
  - The compaction summary call never gets it.
  - The telemetry summary records it per agent, `None` after a refusal.

  Update the "Order in `create()`" sentence to the order in Global Constraints.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/manager.py src/trading_agent_framework/agents/CLAUDE.md tests/agents/test_agent_manager_reasoning_effort.py
git commit -m "feat(agents): send LLM_REASONING_EFFORT with every agent call, drop it on refusal"
```

---

### Task 3: Show the effort in the dashboard's Parameters tab

**Files:**
- Modify: `src/trading_agent_framework/dashboard/reader.py` (`_agent_rows`, lines 158-183)
- Test: `tests/dashboard/test_reader.py` (next to the temperature tests, line 515)

**Interfaces:**
- Consumes: `settings.json["agents"][agent]["reasoning_effort"]: str | None` (Task 2).

- [ ] **Step 1: Write the failing tests** in `tests/dashboard/test_reader.py`, modelled on the three temperature tests:
  - `test_load_parameters_shows_the_agents_reasoning_effort_under_its_temperature`: with `{**_TRADER, "temperature": 0.3, "reasoning_effort": "xhigh"}`, the row after `("Model", "Temperature", "0.3")` is `("Model", "Reasoning effort", "xhigh")`.
  - `test_load_parameters_shows_a_none_reasoning_effort_as_the_server_default`: with `"reasoning_effort": None`, `("Model", "Reasoning effort", "server default")` is in the rows.
  - `test_load_parameters_has_no_reasoning_effort_row_when_settings_predate_it`: with `_TRADER`, there is no `Reasoning effort` label.

- [ ] **Step 2: Run them and check they fail**

Run: `uv run pytest tests/dashboard/test_reader.py -k reasoning_effort -v`
Expected: the first two FAIL (no such row). The third PASSES already.

- [ ] **Step 3: Implement.** In `_agent_rows`, right after the temperature row and under the same "absent in runs recorded before ..." guard, add `add("Model", "Reasoning effort", "server default" if effort is None else str(effort))`. Update the docstring's "Model (with its temperature)" to "Model (with its temperature and reasoning effort)".

- [ ] **Step 4: Run the whole suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: all tests pass, and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/dashboard/reader.py tests/dashboard/test_reader.py
git commit -m "feat(dashboard): show each agent's reasoning effort in the Parameters tab"
```

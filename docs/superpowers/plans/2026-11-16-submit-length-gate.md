# Submit Length Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** sector_pods agents count their texts with a core `count_characters` tool, submit tools refuse over-long texts (naming each count) a bounded number of times, and the last attempt is trimmed at a hard cap.

**Architecture:** Strategy-agnostic plumbing in `agents/tools/text.py` (`trim_text` moved there, `TextLimit`, `LengthGate`, `count_characters`, `count_step`); `SubmitSlot` owns an optional gate and resets it on `arm`. sector_pods submit tools hand their labelled texts to the gate before the slot's validator, which now checks the hard caps.

**Tech Stack:** Python 3.14, LangChain tools (plain functions), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-16-submit-length-gate-design.md`

## Global Constraints

- Nothing strategy-specific in `agents/`: no `weight`, `score`, `sector_view`, `objection`, `overrides`, `reason` as names, arguments or messages in `agents/tools/text.py` or the `SubmitSlot` diff.
- Values: `reason_max_chars` 200 / `reason_hard_max_chars` 300; `summary_max_chars` 400 / `summary_hard_max_chars` 700; `max_submit_attempts` 3; `max_agent_steps` 60.
- Only length refusals use an attempt. `max_attempts == 1` never refuses (trim only).
- The prompt never quotes the hard caps; the refusal never says which attempt it is or that the last one is accepted.
- Trim WARNING keeps today's text: `[sector_pods] <tool>: <label> trimmed from <n> to <m> characters (limit <hard>)`.
- `agents/tools/text.py` must not use `from __future__ import annotations`: LangChain reads `count_characters`' signature at runtime (same rule as `sector_pods/handoff.py`). `agents/graph.py` keeps it.
- Per the user's CLAUDE.md: before every `git commit`, run `/ponytail:ponytail-review` on the staged changes, fix every "Must fix", re-run the gate. Docstrings (Google style) on every new public symbol. No `# noqa` / `# type: ignore` / `pragma: no cover` added. Run `uv sync --extra dashboard` once before the first test run (plain sync skips ~290 tests).
- Spec deviation (decided while planning): `LengthGate.review` takes a **sequence** of `(label, value, limit)` and returns a list, not a mapping: two risk objections on the same symbols share a label, so a mapping would drop one. Step 0 updates the spec line.

## Review Focus

1. Two texts with the same label (two objections on `XLK, IGV`; a PM listing a symbol twice): both are measured, refused or trimmed separately and written back to the right entry. Test in Task 1 and Task 3.
2. A PM refused for a weight cap between two over-long submissions: the weight refusal uses no attempt, so the third submit is still the first trim-eligible one only after two length refusals. Test in Task 3.
3. The forced retry inside one `AgentNode` call shares the attempts (one `arm` per node call); the next week's run starts fresh. Test in Task 2.
4. A missing or non-string text (`reason: None`, an entry that is not an object): the gate passes it through untouched and the validator returns its usual message; no crash in the label builder. Test in Task 1 and Task 3.
5. Boundaries: a text exactly at `stated` is not refused; exactly at `hard` is not trimmed; surrounding whitespace is not counted (both `count_characters` and the gate strip). Test in Task 1.

---

### Task 0: Branch and commit the spec

- [ ] **Step 1:** `git switch -c feat/submit-length-gate` (the user's uncommitted `sector_pods/strategy.py` window change travels in the working tree: never stage it).
- [ ] **Step 2:** In the spec, replace the `review(texts: Mapping[...])` line with the sequence form from Global Constraints.
- [ ] **Step 3:** `git add docs/superpowers/specs/2026-11-16-submit-length-gate-design.md docs/superpowers/plans/2026-11-16-submit-length-gate.md && git commit -m "docs: submit length gate design and plan"`

### Task 1: `agents/tools/text.py`

**Files:**
- Create: `src/trading_agent_framework/agents/tools/text.py`
- Modify: `src/trading_agent_framework/agents/tools/__init__.py` (eager export of `text_tools`, `LengthGate`, `TextLimit`, `count_step`, `trim_text`; add to `__all__`)
- Modify: `src/trading_agent_framework/strategies/common/handoff.py` (delete `trim_text`, update the module docstring), `strategies/bull_bear/pipeline.py:33`, `strategies/sector_pods/handoff.py:11` (import from `agents.tools.text`)
- Modify: `tests/strategies/common/test_common_handoff.py` (move the `# --- trim_text` block, lines 111-135, out), `tests/strategies/bull_bear/test_bull_bear_handoff.py:9`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py:29` (imports)
- Test: `tests/agents/tools/test_text_tools.py`

**Interfaces:**
- Produces:
  - `trim_text(text: str, *, max_chars: int, min_chars: int) -> str` (moved verbatim)
  - `@dataclass(frozen=True) TextLimit(stated: int, hard: int)`; `__post_init__` raises `ValueError` unless `1 <= stated <= hard`
  - `text_tools() -> list[Callable[..., dict[str, Any]]]` returning `[count_characters]`; `count_characters(texts: list[str]) -> dict[str, list[int]]` returns `{"counts": [len(str(t).strip()) for t in texts]}`; docstring: `Count the characters of each text. Returns counts: one number per text, in the order given.`
  - `class LengthGate(max_attempts: int, *, name: str, log_tag: str)`: `ValueError` if `max_attempts < 1`; `reset() -> None`; `review(texts: Sequence[tuple[str, Any, TextLimit]]) -> dict[str, str] | list[Any]` (an `{"error": ...}` dict, or the values in order: stripped strings, trimmed where needed; non-strings unchanged). Logs through `ColorLogger(logging.getLogger(__name__), log_tag)`.
  - `count_step(submit_tool: str, texts: str) -> str`

- [ ] **Step 1: Write the failing tests** (move the 5 `trim_text` tests here unchanged, then add):

```python
GATE_LIMIT = TextLimit(stated=10, hard=20)

def gate(attempts=3): return LengthGate(attempts, name="submit_x", log_tag="strat")

def test_count_characters_returns_stripped_lengths_in_order():
    (count,) = text_tools()
    assert count.__name__ == "count_characters"
    assert count(texts=["  abc ", "", "x" * 401]) == {"counts": [3, 0, 401]}

def test_an_over_long_text_is_refused_naming_each_count_and_stated_limit():
    result = gate().review([("a", "x" * 11, GATE_LIMIT), ("b", "ok", GATE_LIMIT), ("c", "y" * 15, GATE_LIMIT)])
    assert result == {"error": "too long: a has 11 characters (limit 10); c has 15 characters (limit 10). "
                               "Count them with count_characters, shorten those texts and call submit_x again with the whole submission."}

def test_the_last_attempt_is_accepted_trimmed_at_the_hard_cap(caplog):
    g = gate(3)
    texts = [("a", "One two. " * 4, GATE_LIMIT), ("b", "x" * 15, GATE_LIMIT)]   # 35 and 15 chars
    assert "error" in g.review(texts) and "error" in g.review(texts)
    kept = g.review(texts)
    assert kept == [trim_text("One two. " * 4, max_chars=20, min_chars=5), "x" * 15]   # b: over stated, under hard: kept whole
    assert "[strat] submit_x: a trimmed from 35 to 17 characters (limit 20)" in caplog.text

def test_only_length_refusals_use_an_attempt():   # within limits never counts
    g = gate(2)
    assert g.review([("a", "short", GATE_LIMIT)]) == ["short"]
    assert "error" in g.review([("a", "x" * 11, GATE_LIMIT)])
    assert g.review([("a", "x" * 11, GATE_LIMIT)]) == ["x" * 11]

def test_reset_gives_the_attempts_back(): ...   # 2 refusals, reset(), refused again
def test_one_attempt_never_refuses(): assert gate(1).review([("a", "x" * 25, GATE_LIMIT)]) == ["x" * 19 + "…"]
def test_boundaries_are_inclusive_and_whitespace_is_not_counted():
    assert gate().review([("a", "  " + "x" * 10 + "  ", GATE_LIMIT)]) == ["x" * 10]
def test_non_strings_pass_through_untouched(): assert gate().review([("a", None, GATE_LIMIT), ("b", 3, GATE_LIMIT)]) == [None, 3]
def test_same_labels_are_kept_apart(): # two "a" entries, one over: one refusal line; after attempts, both values back in order
def test_invalid_limits_and_attempts_raise(): # TextLimit(0, 5), TextLimit(10, 9), LengthGate(0, ...) -> ValueError
def test_count_step_names_the_tool_and_the_texts():
    step = count_step("submit_x", "every `note` (10 characters at most)")
    assert "`count_characters`" in step and "`submit_x`" in step and "every `note` (10 characters at most)" in step
```

- [ ] **Step 2:** `uv run pytest tests/agents/tools/test_text_tools.py -q` → FAIL (ImportError).
- [ ] **Step 3: Implement** `text.py`. Refusal rule: `over = [(label, len) for string values whose stripped length > limit.stated]`; if `over` and `self._refusals < max_attempts - 1`: increment, `log_info(f"{name}: refused for length: {listed}")`, return the error; else return each string stripped and, above `limit.hard`, `trim_text(..., max_chars=hard, min_chars=stated // 2)` with the WARNING. `count_step` copy (from the spike):

```text
Length check: before you call `{submit_tool}`, call `count_characters` once with {texts}, in that order.
It returns the length of each text in characters. `{submit_tool}` refuses a text over its limit.
If a count is over its limit, shorten that text and count it again. Submit only texts whose last count is within the limit, unchanged.
```

- [ ] **Step 4:** `uv run pytest tests/agents/tools/ tests/strategies/common/ tests/strategies/bull_bear/ -q` → PASS.
- [ ] **Step 5:** `grep -rn "strategies.common.handoff import.*trim_text" src tests` → empty. `grep -niE "weight|score|sector_view|objection|overrides|reason" src/trading_agent_framework/agents/tools/text.py` → empty.
- [ ] **Step 6:** Add `agents/tools/text.py` to `agents/CLAUDE.md`'s `agents/tools/` bullet (one sentence: the gate rule, `trim_text` moved here, no strategy vocabulary) and fix the `trim_text` mention in `strategies/common/CLAUDE.md`. Ponytail review, commit `feat(agents): count_characters tool and LengthGate for submit texts`.

### Task 2: `SubmitSlot` owns an optional gate

**Files:** Modify `src/trading_agent_framework/agents/graph.py` (`SubmitSlot`); Test `tests/agents/test_agent_graph.py`.

**Interfaces:**
- Consumes: `LengthGate.reset()` (Task 1). Import it under `TYPE_CHECKING` only (`graph.py` stays import-light).
- Produces: `SubmitSlot(validate, gate: LengthGate | None = None)`; read-only property `gate -> LengthGate | None`; `arm()` calls `gate.reset()` when set.

- [ ] **Step 1: Failing tests:**

```python
def test_arming_a_slot_resets_its_gate():
    gate = LengthGate(2, name="submit", log_tag="t"); slot = SubmitSlot(lambda p, s: None, gate=gate)
    assert "error" in gate.review([("a", "x" * 11, TextLimit(10, 20))])
    slot.arm({})
    assert "error" in gate.review([("a", "x" * 11, TextLimit(10, 20))])
    assert slot.gate is gate and SubmitSlot(lambda p, s: None).gate is None

def test_the_forced_retry_shares_the_gate_attempts():
    # AgentNode with a fake agent whose first run refuses once through the gate and records nothing,
    # second (forced) run reviews again: accepted (attempts 2), so the node writes the value; arm ran once.
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** `uv run pytest tests/agents/ -q` → PASS. **Step 5:** ponytail review, commit `feat(agents): SubmitSlot resets its length gate on arm`.

### Task 3: sector_pods params and submit tools through the gate

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/params.py`, `handoff.py`
- Test: `tests/strategies/sector_pods/test_sector_pods_params.py`, `test_sector_pods_handoff.py`

**Interfaces:**
- Consumes: `LengthGate`, `TextLimit` (Task 1), `SubmitSlot.gate` (Task 2).
- Produces: `SectorPodsParams.reason_hard_max_chars: int = 300`, `summary_hard_max_chars: int = 700`, `max_submit_attempts: int = 3`, `max_agent_steps: int = 60`; properties `reason_limit -> TextLimit` and `summary_limit -> TextLimit`. Submit tool factories keep their signatures `pod_submit_tool(slot, params)` etc.; with `slot.gate is None` texts go to the validator unreviewed.

- [ ] **Step 1: Failing tests.** Params: defaults above; `ConfigurationError` naming `reason_hard_max_chars must be at least reason_max_chars` (and the summary one) and `max_submit_attempts must be at least 1`. Handoff (replace the four `..._trimmed...` tests, lines 202-268; build slots as `SubmitSlot(validate_pod(TECH, PARAMS), gate=LengthGate(PARAMS.max_submit_attempts, name=POD_SUBMIT, log_tag="sector_pods"))`):

```python
def test_an_over_long_pod_text_is_refused_with_its_count_then_trimmed_on_the_third_attempt(caplog):
    ...
    # 1st, 2nd: {"error": "too long: reason for XLK has 263 characters (limit 200); sector_view has ... (limit 400). ..."}, slot.value is None
    # 3rd: {"status": "recorded"}; reason <= 300 and sector_view <= 700, cut at a sentence; WARNING "[sector_pods] submit_pod_view: sector_view trimmed from"


def test_a_text_between_the_stated_and_hard_limits_is_recorded_whole_on_the_last_attempt(): ...  # 450-char view kept as is (validator checks 700)
def test_two_objections_on_the_same_symbols_are_reviewed_apart(): ...  # Review Focus 1
def test_a_weight_refusal_uses_no_length_attempt(): ...  # Review Focus 2: long texts refused, then valid texts + 30% weight refused, then long texts refused again (2nd length refusal), then trimmed
def test_a_missing_reason_still_gets_the_validators_message(): ...  # Review Focus 4: reason None -> the existing "must be a non-empty string" error, not a gate error
def test_without_a_gate_texts_reach_the_validator_unreviewed(): ...  # 800-char view -> the validator's length error against 700
```

- [ ] **Step 2:** run → FAIL. **Step 3: Implement.** Validators: every `_text(..., params.reason_max_chars)` / `summary_max_chars` becomes the hard cap. Delete `_trimmed`, `_trimmed_reasons`, the `trim_text` import. One private helper `_reviewed(slot, entries, entry_key, long_fields) -> dict | tuple[list, dict]` builds the labels (`reason for <symbol>` or `reason for <a, b>` via the existing `_subject`, then each long field name), calls `slot.gate.review`, and writes the values back into copies of the entries (an entry that is not a dict is left as is, contributing no text). Add the params, properties and validation.
- [ ] **Step 4:** `uv run pytest tests/strategies/sector_pods/ -q` → PASS. **Step 5:** ponytail review, commit `feat(sector_pods): refuse over-long texts up to max_submit_attempts, then trim at the hard cap`.

### Task 4: wire the gate and the counter into the agents and prompts

**Files:**
- Modify: `src/trading_agent_framework/strategies/sector_pods/strategy.py` (`_node`, `_graph_parts`), `prompts.py`, `strategies/sector_pods/CLAUDE.md`
- Test: `tests/strategies/sector_pods/test_sector_pods_strategy.py`, `test_sector_pods_prompts.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: every agent's tools end with `*text_tools()`; `exempt_tools == [submit, "count_characters", *exempt]`; each `SubmitSlot` gets `LengthGate(params.max_submit_attempts, name=submit, log_tag="sector_pods")` (build the slot inside `_node` from a validator argument so the gate's name is the submit tool's).

- [ ] **Step 1: Failing tests.** In `test_each_pod_gets_its_own_news_and_macro_tools` (line 93): `exempt_tools` become `["submit_pod_view", "count_characters"]` and `["submit_target_weights", "count_characters", "remember_lesson"]`; every created agent's tool names contain `count_characters`; each node's `slot.gate` is a `LengthGate`. Prompts: each of `pod_system`, `risk_system`, `pm_system` contains `count_characters` and its submit tool name, and none contains `"300"` or `"700"`.
- [ ] **Step 2:** run → FAIL. **Step 3: Implement.** Prompts append `count_step(SUBMIT, texts)` right after each submit step, with texts: pod `every \`reason\`, then \`sector_view\``; risk `every \`reason\`, then \`summary\``; PM `every \`reason\`, then \`overrides\`, then \`summary\``.
- [ ] **Step 4:** `uv run pytest tests/strategies/sector_pods/ -q` → PASS.
- [ ] **Step 5:** `strategies/sector_pods/CLAUDE.md`: replace the "An over-long free text is trimmed, never refused" passage with: count before submit, refusal naming each count up to `max_submit_attempts - 1` times (length refusals only), last attempt trimmed at the hard caps (300/700), the run `2026-11-16_085725` (726 trims) and the spike numbers, `max_agent_steps` 60. Ponytail review, commit `feat(sector_pods): agents count their texts before submitting`.

### Task 5: cleanup and gate

- [ ] **Step 1:** `/simplify` on `git diff main...` (aim: what bull_bear will reuse stays in `agents/tools/text.py`; no duplicate label/write-back logic across the three submit tools).
- [ ] **Step 2:** `uv run ruff check && uv run ruff format --check && uv run pyright src/trading_agent_framework/agents src/trading_agent_framework/strategies/sector_pods && uv run pytest --cov -q && uv run interrogate -v` → exit 0, coverage >= 80%, interrogate >= 80%. Report the numbers.
- [ ] **Step 3:** Global Constraints grep on `text.py` and `git diff main -- src/trading_agent_framework/agents/graph.py` → no strategy words.
- [ ] **Step 4:** Ponytail review, commit any cleanup `refactor: simplify the submit length gate wiring`. No push, no PR unless asked. Tell the user the follow-up: a sector_pods backtest (expect trims near 0, no `loop:` abandons).

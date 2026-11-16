# bull_bear Length Gate and One Submit Recorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** bull_bear's four agents count their texts and submit through the shared `LengthGate`, and every strategy records submissions through one `SubmitRecorder` in `agents/` (`SubmitSlot` deleted).

**Architecture:** `HandoffError` and `SubmitRecorder` move to `agents/submit.py`; the recorder gains an optional gate per stage, reset on `_arm`, and a `submit_texts` entry point that runs the gate after its guards. A single-stage `NodeRecorder` replaces `SubmitSlot` for LangGraph nodes. sector_pods' `_submit` logic becomes `review_texts` in `agents/tools/text.py`; bull_bear's and sector_pods' submit tools both call it.

**Tech Stack:** Python 3.14, LangChain tools (plain functions), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-16-bull-bear-length-gate-design.md`

## Global Constraints

- No strategy vocabulary in `agents/submit.py` or `agents/tools/text.py`: `grep -niE "argument|reason|note|verdict|weight|score|sector_view|objection" ` on both returns nothing (field names arrive as arguments).
- No `from __future__ import annotations` in `agents/submit.py`, `agents/tools/text.py` or any `handoff.py` (tool schemas are read at runtime). `agents/graph.py` keeps it.
- No re-export left behind: `HandoffError` / `SubmitRecorder` are imported from `trading_agent_framework.agents.submit` everywhere, `strategies/common/handoff.py` included.
- bill_ackman, congress_trades, insider_trades: import changes only. `InsiderRecorder.submit(stage, raw)` keeps its signature (so `submit_texts` is a new method, not a `submit` keyword).
- bull_bear values: `note_max_chars` 500 / `note_hard_max_chars` 800; `argument_max_chars` 300 / `argument_hard_max_chars` 450; `reason_max_chars` 300 / `reason_hard_max_chars` 450 (also for `what_changed`); `max_submit_attempts` 3; `max_agent_steps` stays 60.
- Gate `log_tag` "bull_bear", gate `name` = the stage's tool name. Prompts quote stated limits only, never the hard caps; no prompt contains `held`, `weight_pct` or `forced`.
- The forced retry shares the gate's attempts with the first run: neither `_run_stage` nor `AgentNode` re-arms between attempts.
- A gate refusal goes through `_fail`, so it lands in `last_error`.
- Per the user's CLAUDE.md: `uv sync --extra dashboard` once before the first test run; Google docstrings on every new public symbol; no `# noqa` / `# type: ignore` / `pragma: no cover` added; before every `git commit` run `/ponytail:ponytail-review` on the staged changes, fix every "Must fix", re-run the gate, list the "Should fix" to the user.

## Review Focus

1. A submit call while the stage is not armed, or after a recording, spends no gate attempt (guards run before the gate). Test in Task 1.
2. The judge's optional `what_changed` absent or `None` on most verdicts: passes the gate untouched, only present ones are measured and labelled `what_changed for <SYM>`. Test in Task 2 and Task 4.
3. A researcher run that ends after two length refusals: the forced retry's first over-long note is recorded cut at 800, and the retry prompt quotes the length refusal (`last_error`). Test in Task 5.
4. Malformed submissions (`cases` not a list, an entry not a dict, `argument: 42`): the gate passes them through and the validator gives its usual message, no crash in label building. Test in Task 2 and Task 4.
5. A text the gate accepted between stated and hard (e.g. a 420-character argument on the last attempt) is recorded, not refused by `required_text`. Test in Task 4.

---

### Task 0: Commit the spec and plan

- [ ] **Step 1:** On branch `feat/bull-bear-length-gate` (already created): `git add docs/superpowers/specs/2026-11-16-bull-bear-length-gate-design.md docs/superpowers/plans/2026-11-16-bull-bear-length-gate.md && git commit -m "docs: bull_bear length gate and one submit recorder: design and plan"`

### Task 1: `agents/submit.py`: the recorder moves and gains gates

**Files:**
- Create: `src/trading_agent_framework/agents/submit.py`
- Modify: `src/trading_agent_framework/strategies/common/handoff.py` (delete `HandoffError`, `SubmitRecorder`; import `HandoffError` from `agents.submit`; module docstring)
- Modify imports: `strategies/{bill_ackman,bull_bear,congress_trades,insider_trades}/handoff.py`, and every test importing either name (`tests/strategies/{bill_ackman,bull_bear,congress_trades,insider_trades,common}/…`)
- Test: `tests/agents/test_submit.py` (move the `SubmitRecorder` tests out of `tests/strategies/common/test_common_handoff.py`, then add)

**Interfaces:**
- Produces (in `agents/submit.py`):
  - `class HandoffError(ValueError)` (moved verbatim)
  - `class SubmitRecorder`: as today plus `__init__(self, gates: Mapping[str, LengthGate] | None = None)`; `_arm` also calls `reset()` on the armed stage's gate; property `gate -> LengthGate | None` (the armed stage's); `_tool_name(stage: str) -> str` used by `submit`'s not-expected message and `missing_tool` (default `self.TOOL_NAMES[stage]`, `.get` semantics kept for `missing_tool`);
    `submit_texts(self, stage: str, review: Callable[[LengthGate], dict[str, str] | tuple[Any, ...]], *args: Any) -> dict[str, Any]`: same guards as `submit` (not expected, already recorded) first; then, if the stage has a gate, `out = review(gate)`: a dict is returned via `self._fail(out["error"])`, a tuple replaces `args`; then `return self.submit(stage, *args)`.
  - `class NodeRecorder(SubmitRecorder)`: `__init__(self, tool: str, validate: Callable[[Any, Mapping[str, Any]], None], gate: LengthGate | None = None)` (validate raises `HandoffError`); `tool: str` attribute; `arm(self, state: Mapping[str, Any]) -> None` arms stage `tool` with the state as context; `_validate` calls `validate(payload, context)` and returns the payload; `_tool_name` returns `tool`.

- [ ] **Step 1: Write the failing tests** in `tests/agents/test_submit.py`:

```python
def test_arming_a_stage_resets_only_its_gate() -> None:
    # gates {"a": gate_a, "b": gate_b}; spend a refusal on each; _arm("a", {}) -> gate_a refuses again from 0, gate_b still spent; recorder.gate is gate_a

def test_submit_texts_spends_no_attempt_when_the_stage_is_not_armed_or_already_recorded() -> None:
    # review callback counts its calls; submit_texts on an unarmed stage -> error "... is not expected at this point", calls == 0
    # after a recorded submit, submit_texts -> {"error": "already recorded for this run"}, calls == 0

def test_a_gate_refusal_is_returned_and_kept_in_last_error() -> None:
    # review returns {"error": "too long: x"} -> result == {"error": "too long: x"}, recorder.last_error == "too long: x", not submitted

def test_submit_texts_records_the_reviewed_args() -> None:
    # review returns ("trimmed",) -> submission == "trimmed"

def test_submit_texts_without_a_gate_submits_the_args_unchanged() -> None: ...

def test_node_recorder_validates_against_the_armed_state() -> None:
    # NodeRecorder("submit_x", lambda p, s: None if p in s["allowed"] else raise HandoffError("not allowed")); arm({"allowed": [1]}); submit("submit_x", 2) -> error "not allowed"; submit("submit_x", 1) -> recorded, submission == 1, missing_tool == "submit_x"
```

- [ ] **Step 2:** `uv run pytest tests/agents/test_submit.py -v` → FAIL (`ModuleNotFoundError: trading_agent_framework.agents.submit`).
- [ ] **Step 3:** Implement `agents/submit.py` per Interfaces; move the code from `common/handoff.py`; update every import (src and tests).
- [ ] **Step 4:** `uv run pytest tests/agents tests/strategies -q` → all pass; `grep -rn "common.handoff import.*\(HandoffError\|SubmitRecorder\)" src tests` → empty.
- [ ] **Step 5:** Review (`/ponytail:ponytail-review`), then commit: `refactor(agents): move SubmitRecorder and HandoffError to agents/submit.py, with optional length gates`.

### Task 2: `review_texts` in `agents/tools/text.py`

**Files:**
- Modify: `src/trading_agent_framework/agents/tools/text.py`, `agents/tools/__init__.py` (export `review_texts`)
- Test: `tests/agents/tools/test_text_tools.py`

**Interfaces:**
- Produces: `review_texts(gate: LengthGate, entries: Any, fields: Mapping[str, TextLimit], subject: Callable[[Mapping[str, Any]], str], texts: Mapping[str, tuple[Any, TextLimit]] | None = None) -> dict[str, str] | tuple[Any, dict[str, Any]]`.
  Builds one `(f"{field} for {subject(entry)}", entry[field], limit)` triple per dict entry and per field present in it with a non-`None` value (entry order, then field order), then the top-level `texts` as `(label, value, limit)`; calls `gate.review` once. Returns the gate's error dict, or `(entries, top)`: `entries` a new list with each reviewed field written back into a copy of its entry (a non-list `entries` and non-dict items returned untouched), `top` = `{label: reviewed value}` in `texts` order.

- [ ] **Step 1: Write the failing tests:**

```python
def test_review_texts_labels_every_field_of_every_entry_then_the_top_level_texts() -> None:
    # gate max_attempts=2, TextLimit(10, 20); entries [{"s": "A", "f": "x"*11, "g": "ok"}], fields {"f": L, "g": L}, subject=lambda e: e["s"], texts {"view": ("y"*12, L)}
    # -> {"error": "too long: f for A has 11 characters (limit 10); view has 12 characters (limit 10). ..."}

def test_review_texts_skips_an_absent_or_none_field() -> None:
    # entries [{"s": "A", "f": "ok"}, {"s": "B", "f": "ok", "g": None}] -> accepted; returned entries equal the input; no label "g for ..." ever built (subject called only for dicts)

def test_review_texts_writes_trimmed_values_back_to_the_right_entry() -> None:
    # max_attempts=1, two entries with the same subject "A, B" and texts of 25 and 5 chars -> first entry trimmed to <= 20, second unchanged, input list not mutated

def test_review_texts_passes_malformed_entries_through() -> None:
    # entries "not a list" -> ("not a list", {}); entries [1, {"s": "A", "f": 42}] -> ([1, {"s": "A", "f": 42}], {})
```

- [ ] **Step 2:** `uv run pytest tests/agents/tools/test_text_tools.py -v` → FAIL (`ImportError: review_texts`).
- [ ] **Step 3:** Implement `review_texts` per Interfaces.
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Review, commit: `feat(agents): review_texts puts an entry list's text fields and top-level texts through a length gate`.

### Task 3: sector_pods on `NodeRecorder`; `SubmitSlot` deleted

**Files:**
- Modify: `src/trading_agent_framework/agents/graph.py` (delete `SubmitSlot`; `AgentNode.slot: SubmitSlot` → `recorder: NodeRecorder`; `__call__` uses `recorder.arm(state)`, `recorder.submitted`, `recorder.submission`; module docstring)
- Modify: `strategies/sector_pods/handoff.py` (validators raise `HandoffError`, return `None`; `_text` deleted in favour of `common/handoff.py`'s `required_text` / `optional_text`; `Validator` type `Callable[[Any, Mapping[str, Any]], None]`; `_submit` → `recorder.submit_texts(tool, review, payload)` with `review_texts(gate, entries, {"reason": params.reason_limit}, _subject, {label: (value, params.summary_limit)})`; tool factories take `NodeRecorder`), `strategies/sector_pods/agent_sector_pods.py:147` (`NodeRecorder(submit, validator, gate=LengthGate(...))`), and any `slot.value` reader in `strategies/sector_pods/graph.py`
- Test: `tests/agents/test_agent_graph.py`, `tests/strategies/sector_pods/test_sector_pods_handoff.py`, `tests/strategies/sector_pods/test_sector_pods_graph.py`

**Interfaces:**
- Consumes: `NodeRecorder`, `submit_texts` (Task 1), `review_texts` (Task 2).
- Produces: `AgentNode(name, agent, recorder: NodeRecorder, submit_tool, task, reads, write, tool_budget=None, max_steps=None)`.
- `check_weights` keeps returning `str | None` (the execute node calls it); `validate_pm` raises its message.

- [ ] **Step 1: Update the tests first:** replace `SubmitSlot` with `NodeRecorder` in the three files (validators now raise; assertions on `{"error": ...}` payloads keep their messages except where `_text`'s wording changes to `required_text`'s: `"<who> has no <field>: give one short sentence"`, `"the <field> for <who> is <n> characters: keep it under <limit>"`); keep `test_the_forced_retry_shares_the_gate_attempts_and_the_next_run_starts_fresh` (now on `NodeRecorder`) and add:

```python
def test_a_gate_refusal_in_a_sector_pods_tool_is_the_recorders_last_error() -> None:
    # pod tool, an over-long sector_view on attempt 1 -> recorder.last_error starts with "too long: sector_view has"
```

- [ ] **Step 2:** `uv run pytest tests/agents/test_agent_graph.py tests/strategies/sector_pods -q` → FAIL (`ImportError: NodeRecorder` from tests once updated / `SubmitSlot` still used).
- [ ] **Step 3:** Implement per Files/Interfaces; delete `SubmitSlot`.
- [ ] **Step 4:** Same command → PASS; `grep -rn "SubmitSlot" src tests` → empty.
- [ ] **Step 5:** Review, commit: `refactor(sector_pods): record through NodeRecorder and review_texts; drop SubmitSlot`.

### Task 4: bull_bear parameters and hand-off on the gate

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py`, `strategies/bull_bear/handoff.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`, `tests/strategies/bull_bear/test_bull_bear_params.py`

**Interfaces:**
- Produces in `BullBearParams`: `argument_hard_max_chars: int = 450`, `reason_hard_max_chars: int = 450`, `max_submit_attempts: int = 3`; validation messages `"argument_hard_max_chars must be at least argument_max_chars"`, `"reason_hard_max_chars must be at least reason_max_chars"`, `"max_submit_attempts must be at least 1"`; properties `note_limit`, `argument_limit`, `reason_limit -> TextLimit`.
- `HandoffRecorder.__init__(params)` passes `gates={stage: LengthGate(params.max_submit_attempts, name=tool, log_tag="bull_bear") for stage, tool in TOOL_NAMES.items()}`.
- Submit tools call `recorder.submit_texts(stage, review, *args)`: note → `texts={f"note for {recorder.note_symbol}": (note, params.note_limit)}`, review returns `(symbol, top[label])`; bull / bear → `fields={"argument": params.argument_limit}`; judge → `fields={"reason": params.reason_limit, "what_changed": params.reason_limit}`; `subject = lambda entry: str(entry.get("symbol"))`. `submit_tools(recorder: HandoffRecorder, params: BullBearParams)` (it needs the limits).
- Validators check the HARD caps: `validate_note(symbol, note, *, expected, max_chars)` (`ask_chars` gone; over `max_chars` is a plain `HandoffError`), `validate_bull_cases` / `validate_bear_cases(max_chars=params.argument_hard_max_chars)`, `validate_verdicts(max_chars=params.reason_hard_max_chars)`.
- Deleted: `NoteTooLong`, `_refused_note`, `refused_note`, the `_arm` override, the NOTE-stage `try/except` in `_validate`.

- [ ] **Step 1: Write the failing tests** (delete the tests for the "same text", `refused_note`, "asked length" message and `test_a_trimmed_note_is_always_accepted_by_the_recorder`; rewrite the two hard-limit note tests on the gate):

```python
def test_an_over_long_note_is_refused_with_its_count_then_recorded_trimmed_at_800_on_the_third_attempt() -> None:
    # 900-char note: attempts 1-2 -> error contains "note for AAPL has 900 characters (limit 500)"; attempt 3 -> recorded, len(submission.note) <= 800

def test_an_over_long_argument_names_the_stock_and_counts() -> None:
    # bull cases, one argument of 320 -> error contains "argument for AAPL has 320 characters (limit 300)"

def test_an_argument_between_stated_and_hard_is_recorded_on_the_last_attempt() -> None:
    # 420-char argument sent 3 times -> third recorded with the 420 chars unchanged

def test_the_judge_gate_measures_reason_and_only_the_what_changed_given() -> None:
    # two verdicts, one with what_changed of 310, one without -> error lists "what_changed for X has 310 characters (limit 300)" and no label for the other

def test_a_refusal_for_another_rule_spends_no_attempt() -> None:
    # a weak "in" refused twice, then an over-long reason: still refused for length (attempts unspent)

def test_arming_the_next_stock_gives_the_note_gate_its_attempts_back() -> None: ...

def test_hard_caps_below_their_stated_limit_or_zero_attempts_are_rejected() -> None:  # parameters
```

- [ ] **Step 2:** `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py tests/strategies/bull_bear/test_bull_bear_params.py -v` → FAIL.
- [ ] **Step 3:** Implement per Interfaces; update `submit_tools`' caller in `agent_bull_bear.py` to pass `self.settings`.
- [ ] **Step 4:** Same command → PASS.
- [ ] **Step 5:** Review, commit: `feat(bull_bear): put notes, arguments and verdict reasons through the length gate`.

### Task 5: bull_bear pipeline, agents and prompts

**Files:**
- Modify: `strategies/bull_bear/pipeline.py` (`_research_notes`: drop the refused-note branch, `trimmed`, the `trim_text` import; `record["notes"] = {"written": ..., "failed": ...}`; `_run_stage`: drop `skip_retry` and its docstring lines; module docstring line 6)
- Modify: `strategies/bull_bear/agent_bull_bear.py` (every agent: `tools=[..., *text_tools(), submit[...]]`, `exempt_tools` includes `"count_characters"`: researcher `["submit_note", "count_characters", "get_income_statement", "get_balance_sheet"]`, bull/bear/judge `[tool, "count_characters"]`)
- Modify: `strategies/bull_bear/prompts.py` (`researcher_task`, `bull_task`, `bear_task`, `judge_task` end with `count_step(...)`: `"submit_note", "`note`"`; `"submit_bull_case", "every `argument`"`; `"submit_bear_case", "every `argument`"`; `"submit_verdicts", "every `reason`, then every `what_changed` you give"`)
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py`, `test_bull_bear_prompts.py`, `test_bull_bear_strategy.py`

**Interfaces:**
- Consumes: `count_step`, `text_tools` (existing), Task 4's recorder.

- [ ] **Step 1: Write the failing tests** (delete the refused-note-kept tests):

```python
def test_a_researcher_run_ending_after_its_length_refusals_gets_the_forced_retry_whose_long_note_is_recorded_trimmed() -> None:
    # fake researcher: run 1 submits a 900-char note twice then stops; run 2 (force_tool="submit_note") submits it again
    # -> note recorded, len <= 800; the retry prompt contains "note for AAPL has 900 characters"; record["notes"] == {"written": 1, "failed": []}

def test_every_task_prompt_asks_for_the_count_and_quotes_only_the_stated_limits() -> None:
    # each of the 4 tasks contains "count_characters" and its submit tool; contains "500"/"300"; never "800" or "450"

def test_every_agent_has_count_characters_outside_its_budget() -> None:
    # fake AgentManager records create() kwargs: "count_characters" in each agent's tool names and exempt_tools
```

- [ ] **Step 2:** `uv run pytest tests/strategies/bull_bear -q` → FAIL.
- [ ] **Step 3:** Implement per Files.
- [ ] **Step 4:** `uv run pytest tests/strategies/bull_bear -q` → PASS (the no-`held`/`weight_pct`/`forced` prompt tests included); `grep -rn "refused_note\|NoteTooLong\|skip_retry\|trimmed" src/trading_agent_framework/strategies/bull_bear` → empty.
- [ ] **Step 5:** Review, commit: `feat(bull_bear): count characters before every submit; drop the pipeline's refused-note trim`.

### Task 6: docs and the quality gate

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the note-limit gotcha → the gate rule with the five limits, `max_submit_attempts`, shared retry attempts; remove every "refused note kept trimmed" clause, `notes.trimmed`, and the note-length loop in "normal noise"), `agents/CLAUDE.md` (`graph.py` bullet: `NodeRecorder` instead of `SubmitSlot`; `text.py`: `review_texts`; new `submit.py` bullet), `strategies/common/CLAUDE.md` (`handoff.py` no longer holds `HandoffError`/`SubmitRecorder`), `strategies/sector_pods/CLAUDE.md` (any `SubmitSlot` mention)

- [ ] **Step 1:** Edit the docs; `grep -rn "SubmitSlot\|refused_note\|NoteTooLong" src docs/superpowers/specs/2026-11-16-bull-bear-length-gate-design.md --include=*.md` → only the spec's own history lines.
- [ ] **Step 2:** `/simplify` on the branch diff, then the gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%), `uv run pyright` on touched files, and the two greps from Global Constraints. Report the numbers.
- [ ] **Step 3:** Review, commit: `docs: bull_bear length gate and the single submit recorder`.

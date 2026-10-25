# bull_bear Qwen 3.8 Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three changes:
- a researcher never loses its research to a too-long note;
- the judge gets its own, larger token cap and a smaller memory budget;
- the judge treats `max_in` as a ceiling, not a target.

**Architecture:**
- The `HandoffRecorder` remembers the last note refused for length and tells the model plainly how to shorten it.
- A pure `trim_note` cuts that note under the hard limit.
- The pipeline uses the trimmed note instead of the forced retry or "research unavailable".
- The judge's cap and memory budget are `BullBearParams` fields wired in `agent_bull_bear.py`.
- The ceiling is a prompt change only.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-25-bull-bear-qwen38-robustness-design.md`

## Global Constraints

- Run everything through `uv` (`uv run pytest`, `uv run ruff check`). Never call pip or python directly.
- `handoff.py` keeps NO `from __future__ import annotations`: tool schemas are built from real annotations.
- Tests use the hand-written fakes already in `tests/strategies/bull_bear/` (`FakeAgent`, `_harness`). No `MagicMock`, no network.
- Code never rewords agent text. Cutting a refused note short is the only edit allowed.
- The ceiling (`max_book_size` 10), `validate_verdicts`' max_in check, the cooldown and holdings-blindness are unchanged.
- No in-run correction with thinking off, and no new retry.
- Commit messages carry the task number (e.g. `feat(bull_bear): ... (Task 1)`) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- After each task: `uv run pytest tests/strategies/bull_bear/` and `uv run ruff check` both clean.

## Review Focus

- **A later refusal for another reason (wrong symbol, empty note)** after a length refusal: the length-refused note is still the one kept (Task 1 test).
- **A refused note from one stock must never become another stock's note.** Arming a stage clears it (Task 1 test). A later stock that loops with no refusal of its own gets "research unavailable" (Task 3 test).
- **Decimals in notes** (`$1.58B`, `+8.7%`): a `.` not followed by whitespace is not a sentence end (Task 2 test).
- **A trimmed note always fits:** at most `note_hard_max_chars`, `…` included. It would be accepted by `validate_note` (Task 2 test).
- **`judge_max_output_tokens=None`** means the judge is created with no cap. It does not fall back to `max_output_tokens` (Task 4 test).

---

### Task 1: Note refusal message and the recorder's refused note

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py` (`validate_note`, `HandoffRecorder`)
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`

**Interfaces:**
- Produces:
  - `class NoteTooLong(HandoffError)`, raised by `validate_note` for a note longer than `max_chars`. Attribute `note: str` holds the stripped refused text.
  - `validate_note(symbol, note, *, expected: str, max_chars: int, ask_chars: int) -> Note`. `max_chars` is the hard limit (600); `ask_chars` is the length asked for (500).
  - `HandoffRecorder.refused_note -> str | None` (property): the most recent note refused for length since the stage was armed. Cleared by every `expect_*`. Not cleared by a later refusal for another reason.

- [ ] **Step 1: Write the failing tests** (replace `test_an_empty_or_long_note_is_refused`)

```python
def test_an_empty_note_is_refused() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert "empty" in tools["submit_note"]("AAA", "   ")["error"]


def test_a_long_note_is_refused_with_the_asked_length_and_how_to_shorten_it() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")

    error = tools["submit_note"]("AAA", "x" * 662)["error"]

    assert error == "the note for AAA is 662 characters; a note is at most 500: rewrite it to about 450 characters by dropping your least important facts, not by trimming words"
    assert "600" not in error  # the hard limit stays hidden from the model


def test_the_same_long_note_sent_again_is_called_out() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("AAA", "x" * 662)

    again = tools["submit_note"]("AAA", "  " + "x" * 662 + " ")["error"]  # same text once stripped
    shorter = tools["submit_note"]("AAA", "x" * 650)["error"]

    assert again.startswith("this is the same text you sent before: the note for AAA is 662 characters")
    assert not shorter.startswith("this is the same text")


def test_the_recorder_keeps_the_last_note_refused_for_length_until_the_next_stage() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert recorder.refused_note is None

    tools["submit_note"]("AAA", "x" * 700)
    tools["submit_note"]("AAA", " " + "y" * 650 + " ")
    tools["submit_note"]("BBB", "z" * 10)  # refused for its symbol: the length-refused note stays

    assert recorder.refused_note == "y" * 650
    recorder.expect_note("BBB")
    assert recorder.refused_note is None


def test_a_note_refused_for_another_reason_leaves_no_refused_note() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    tools["submit_note"]("BBB", "x" * 700)
    tools["submit_note"]("AAA", "  ")
    assert recorder.refused_note is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -v`
Expected: the new tests FAIL. The long-note message is still `keep it under 600`, and `refused_note` does not exist.

- [ ] **Step 3: Implement**

- In `validate_note`, the long-note branch raises `NoteTooLong` with the message above. The target is `int(ask_chars * 0.9)`.
- `HandoffRecorder.submit` passes `ask_chars=params.note_max_chars`.
- On `NoteTooLong`, if `exc.note == self._refused_note`, prefix `this is the same text you sent before: `. Then set `self._refused_note = exc.note`.
- `_arm` resets `_refused_note` to `None`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass.

- [ ] **Step 5: Update docs and commit**

In `strategies/bull_bear/CLAUDE.md`, gotcha "A note is asked for at `note_max_chars` (500)...": add that the refusal names 500 and asks for about 450 by dropping facts, never quotes 600, and calls out an identical resubmission. Cite the 10-23 run: Qwen 3.8 re-sent `601, 601, 601...`.

```bash
git add src/trading_agent_framework/strategies/bull_bear/handoff.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_handoff.py
git commit -m "feat(bull_bear): a refused note says how to shorten it and is kept by the recorder (Task 1)"
```

---

### Task 2: `trim_note`, the pure cut

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py` (validation section)
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`

**Interfaces:**
- Produces: `trim_note(text: str, *, max_chars: int, min_chars: int) -> str`. The pipeline calls it with `max_chars=note_hard_max_chars` (600) and `min_chars=note_max_chars // 2` (250).

- [ ] **Step 1: Write the failing tests**

```python
def test_a_note_within_the_limit_is_only_stripped() -> None:
    assert trim_note("  A fact. Another.  ", max_chars=600, min_chars=250) == "A fact. Another."


def test_a_long_note_keeps_the_most_whole_sentences_that_fit() -> None:
    text = "a" * 300 + ". " + "b" * 250 + "! " + "c" * 100 + "."
    assert trim_note(text, max_chars=600, min_chars=250) == "a" * 300 + ". " + "b" * 250 + "!"


def test_a_decimal_point_is_not_a_sentence_end() -> None:
    text = "Revenue " + "r" * 280 + ". Q3 EPS $1.58 vs $1.47 est " + "e" * 300 + "."
    assert trim_note(text, max_chars=600, min_chars=250) == "Revenue " + "r" * 280 + "."


def test_with_no_long_enough_sentence_the_note_is_cut_at_a_word_with_an_ellipsis() -> None:
    text = "Short. " + "word " * 200
    trimmed = trim_note(text, max_chars=600, min_chars=250)
    assert len(trimmed) <= 600 and trimmed.endswith("word…")


def test_a_note_with_no_space_is_cut_hard_with_an_ellipsis() -> None:
    assert trim_note("x" * 700, max_chars=600, min_chars=250) == "x" * 599 + "…"


@pytest.mark.parametrize("length", [601, 650, 1200])
def test_a_trimmed_note_is_always_accepted_by_the_recorder(length: int) -> None:
    text = ("Fact 2025-10-27: revenue $1.58B. " * 60)[:length]
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert tools["submit_note"]("AAA", trim_note(text, max_chars=600, min_chars=250)) == {"status": "recorded"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -k trim -v`
Expected: FAIL with `ImportError` (`trim_note`).

- [ ] **Step 3: Implement `trim_note`** (rules from the spec)

- Strip the text. Return it if it fits.
- A sentence end is a `.`, `!` or `?` that is followed by whitespace, or is the last character. Keep the longest prefix ending at a sentence end with length `<= max_chars`.
- If that prefix (stripped) has at least `min_chars` characters, return it.
- Otherwise cut at the last whitespace within the first `max_chars - 1` characters, `rstrip`, and append `…`.
- With no whitespace there, return `text[: max_chars - 1] + "…"`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/handoff.py tests/strategies/bull_bear/test_bull_bear_handoff.py
git commit -m "feat(bull_bear): trim_note cuts a refused note at a sentence under the hard limit (Task 2)"
```

---

### Task 3: The pipeline keeps the refused note, trimmed

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (`_run_stage`, step 4 of `_review`)
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Consumes:
  - `HandoffRecorder.refused_note` (Task 1);
  - `trim_note(text, *, max_chars, min_chars)` (Task 2).
- Produces:
  - `_run_stage(..., skip_retry: Callable[[], bool] | None = None)`. It is called after a first attempt that recorded nothing. `True` raises `ReviewAbandoned(agent_name, error)` with no forced retry.
  - `reviews.jsonl` `notes` = `{"written": int, "failed": [...], "trimmed": [...]}`. Trimmed stocks are counted in `written`.

- [ ] **Step 1: Write the failing tests**

All three use the same `long_note` script. It submits a note made of whole sentences, `LONG` (700 characters), and returns. Assert on `trim_note(LONG, max_chars=600, min_chars=250)`.

```python
LONG = ("2025-10-27: revenue rose 12% y/y to $1.58B. " * 16)[:700]


def long_note(tools: dict[str, Callable[..., dict[str, Any]]], ctx: Any) -> None:
    tools["submit_note"](ctx["fact_sheet"]["symbol"], LONG)


def _notes_seen_by_the_bull(h: Harness) -> dict[str, str]:
    return {s["fact_sheet"]["symbol"]: s["note"] for s in h.agents["bull"].calls[0]["context"]["stocks"]}


def test_a_note_refused_for_length_is_kept_trimmed_without_a_forced_retry(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    h = _harness(tmp_path)
    h.agents["researcher"].overrides = [long_note]  # S00's run ends on a refused note

    with caplog.at_level(logging.INFO):
        assert h.run().completed

    assert len(h.agents["researcher"].calls) == 15  # no forced retry for S00
    assert _notes_seen_by_the_bull(h)["S00"] == trim_note(LONG, max_chars=600, min_chars=250)
    assert h.log_lines()[0]["notes"] == {"written": 15, "failed": [], "trimmed": ["S00"]}
    messages = [re.sub(r"\x1b\[[0-9;]*m", "", r.getMessage()) for r in caplog.records if r.levelno == logging.WARNING]
    assert any(f"[researcher] note for S00 trimmed from 700 to {len(trim_note(LONG, max_chars=600, min_chars=250))} characters" in m for m in messages)
    assert not any("no note for S00" in m for m in messages)


def test_a_researcher_that_loops_after_a_refused_note_keeps_it_trimmed(tmp_path: Path) -> None:
    def long_then_loop(tools, ctx):  # noqa: ANN001, ANN202
        long_note(tools, ctx)
        _loops(tools, ctx)

    h = _harness(tmp_path)
    h.agents["researcher"].overrides = [long_then_loop, _loops]  # S00 loops after a refusal; S01 loops with none

    assert h.run().completed

    seen = _notes_seen_by_the_bull(h)
    assert seen["S00"] == trim_note(LONG, max_chars=600, min_chars=250)
    assert seen["S01"] == UNAVAILABLE_NOTE  # S00's refused note never leaks into S01
    assert h.log_lines()[0]["notes"] == {"written": 14, "failed": ["S01"], "trimmed": ["S00"]}


def test_a_note_refused_for_length_on_the_forced_retry_is_kept_trimmed(tmp_path: Path) -> None:
    h = _harness(tmp_path)
    h.agents["researcher"].overrides = [does_nothing, long_note]  # no refusal first: the forced retry runs, and is refused

    assert h.run().completed

    assert h.agents["researcher"].calls[1]["force_tool"] == "submit_note"
    assert _notes_seen_by_the_bull(h)["S00"] == trim_note(LONG, max_chars=600, min_chars=250)
```

Also update the expected `notes` record in two tests:
- `test_the_review_log_records_the_whole_review_as_one_line` expects `{"written": 15, "failed": [], "trimmed": []}`;
- `test_a_stock_whose_research_fails_gets_research_unavailable_and_the_review_goes_on` expects `{"written": 14, "failed": ["S03"], "trimmed": []}`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_pipeline.py -v -k "refused or trimmed or review_log or research_fails"`
Expected: FAIL. The forced retry runs, the note is `research unavailable`, and `trimmed` is missing.

- [ ] **Step 3: Implement**

- `_run_stage` gains `skip_retry` (see Interfaces). Check it after attempt 0's warnings are logged, before building the retry prompt.
- The researcher loop passes `skip_retry=lambda: self._recorder.refused_note is not None`.
- In its `except ReviewAbandoned` (`AgentLooped` included):
  - With `self._recorder.refused_note`: trim it (`params.note_hard_max_chars`, `params.note_max_chars // 2`), use it as the note, add the symbol to `trimmed`, and log the WARNING `[researcher] note for {symbol} trimmed from {n} to {m} characters: the last refused note is kept with its research`.
  - Otherwise: keep today's `no note for` warning and `UNAVAILABLE_NOTE`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass, including `test_the_forced_retry_repeats_the_task_and_only_the_researcher_is_told_its_research_is_gone` (no refusal there, so the retry still runs).

- [ ] **Step 5: Update docs and commit**

In `strategies/bull_bear/CLAUDE.md`:
- "a failed run gives that stock the note research unavailable": unless a note was refused for length. That note is then kept, trimmed, in place of the forced retry or the loop's "research unavailable".
- The `reviews.jsonl` description gains `notes.trimmed`.
- The loop gotcha ("for the researcher it costs only that stock its note"): same exception.

```bash
git add src/trading_agent_framework/strategies/bull_bear/pipeline.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_pipeline.py
git commit -m "feat(bull_bear): a note refused for length is kept trimmed, not retried from the fact sheet (Task 3)"
```

---

### Task 4: The judge's own token cap and a memory budget of 3

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py:76` (parameters) and `:126-135` (judge creation)
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py`, `tests/strategies/bull_bear/test_bull_bear_strategy.py`

**Interfaces:**
- Produces:
  - `BullBearParams.judge_max_output_tokens: int | None = 16384`, validated with the message `judge_max_output_tokens must be None or at least 1`;
  - `BullBearParams.judge_memory_budget = 3`;
  - `strategy.parameters["judge_max_output_tokens"]`.

- [ ] **Step 1: Write the failing tests**

In `test_bull_bear_params.py`:
- In `test_the_defaults_are_the_specs_and_valid`, assert `params.judge_memory_budget == 3` and `params.judge_max_output_tokens == 16384`.
- Add `({"judge_max_output_tokens": 0}, "judge_max_output_tokens must be None or at least 1")` to the invalid-value cases.

In `test_bull_bear_strategy.py`, replace `test_every_agent_gets_the_output_cap_of_the_settings`:

```python
def test_the_judge_gets_its_own_output_cap_and_the_others_share_one(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(max_output_tokens=9000, judge_max_output_tokens=20000))

    strategy.initialize()

    caps = {created["name"]: created["max_output_tokens"] for created in agents.created}
    assert caps == {"researcher": 9000, "bull": 9000, "bear": 9000, "judge": 20000}
    assert (strategy.parameters["max_output_tokens"], strategy.parameters["judge_max_output_tokens"]) == (9000, 20000)


def test_a_judge_cap_of_none_means_no_cap_not_the_shared_one(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(judge_max_output_tokens=None))

    strategy.initialize()

    caps = {created["name"]: created["max_output_tokens"] for created in agents.created}
    assert caps["judge"] is None and caps["bull"] == 12288
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_strategy.py -v`
Expected: FAIL. `judge_max_output_tokens` is unknown and the budget is still 6.

- [ ] **Step 3: Implement**

- Add the field next to `max_output_tokens`, with a comment. Cite the 10-23 run: 14 judge calls spent 12,288 on reasoning; the judge's prompt reaches 20.5k, 90th percentile 14.8k, of the 32k window; above that, `output_cap`'s fallback sends uncapped.
- Add the validation entry.
- Set `judge_memory_budget = 3`, with a comment: the 3.8 judge made 107 memory calls, each kept in its context.
- In the `max_agent_steps` comment, the worst normal run becomes the judge with its 3 memory calls, a limit of 15.
- `agent_bull_bear.py`:
  - the judge's `create` uses `max_output_tokens=self.settings.judge_max_output_tokens`;
  - `self.parameters` adds `"judge_max_output_tokens"`;
  - the comment above `cap` says the judge has its own.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass. `test_the_judge_runs_with_its_memory_budget_and_its_lessons_are_logged` passes its own budget, so it is unaffected.

- [ ] **Step 5: Update docs and commit**

In `strategies/bull_bear/CLAUDE.md`:
- `judge_memory_budget` 6 becomes 3;
- "All four agents get `BullBearParams.max_output_tokens` (12288...)": the researcher, bull and bear get it, and the judge gets `judge_max_output_tokens` (16384, also in `settings.json`);
- the `max_agent_steps` worst normal run (limit 15).

```bash
git add src/trading_agent_framework/strategies/bull_bear/parameters.py src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_strategy.py
git commit -m "feat(bull_bear): the judge gets its own output cap (16384) and a memory budget of 3 (Task 4)"
```

---

### Task 5: The judge prompt: `max_in` is a ceiling, not a target

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py:88-93` (`JUDGE_SYSTEM`)
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:** none (prompt text only).

- [ ] **Step 1: Write the failing test** (replace `test_a_stronger_candidate_needing_a_place_is_a_new_material_fact_for_the_previous_verdict`)

```python
def test_max_in_is_a_ceiling_not_a_target_and_no_stock_is_put_out_for_room_alone() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())

    assert "max_in, given in the constraints, is a ceiling, not a target" in text
    assert "put a stock in only when its own case earns a place" in text
    assert "Putting fewer than max_in in is normal, and you may put none in." in text
    assert "Only when more stocks earn a place than max_in allows" in text
    assert "A stronger candidate" not in text and "needs a place under max_in" not in text
    assert "change it only for a new, material fact" in text  # the previous_verdict rule stays
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -v`
Expected: the new test FAILS on the ceiling sentence.

- [ ] **Step 3: Implement**

Replace *"Put at most max_in stocks in, given in the constraints: when more deserve it, keep the strongest cases in and give the others out with your reason. You may put none in. "* with this exact copy:

> max_in, given in the constraints, is a ceiling, not a target: put a stock in only when its own case earns a place. Putting fewer than max_in in is normal, and you may put none in. Only when more stocks earn a place than max_in allows, keep the strongest cases in and put the others out with your reason. 

Delete *"A stronger candidate that needs a place under max_in is a new, material fact: put the weaker stock out and say so in what_changed. "*

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass. This includes `test_no_prompt_explains_held_or_weight`: the new text must not contain `held`.

- [ ] **Step 5: Update docs and commit**

In `strategies/bull_bear/CLAUDE.md`, the judge's description "at most `max_book_size` (10) `in`" gains: "a ceiling, not a target (10-23 run: Qwen 3.8 filled the book in 39 of 55 reviews and put holdings out for room)".

```bash
git add src/trading_agent_framework/strategies/bull_bear/prompts.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_prompts.py
git commit -m "feat(bull_bear): the judge is told max_in is a ceiling, not a target (Task 5)"
```

---

## After the last task

Run the full suite with `uv run pytest` and `uv run ruff check`, then the whole-branch review. The spec's success criteria are checked on the next 1Y backtest on Qwen 3.8 (`medium`). That run is the user's, by hand; it is not part of this plan.

# bull_bear note hard limit 800 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise `BullBearParams.note_hard_max_chars` from 700 to 800 so fewer researcher notes are refused for length.

**Architecture:** One parameter value. `HandoffRecorder` (the refusal) and `BullBearPipeline._research_notes` (the `trim_note` fallback) already read `note_hard_max_chars`, so no other source line changes. The tests whose inputs were chosen to sit between 700 and 800 move above 800.

**Tech Stack:** Python 3.14, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-13-bull-bear-note-hard-limit-800-design.md`

## Global Constraints

- `note_max_chars` stays 500 (the ask). Only `note_hard_max_chars` changes, to 800.
- The prompt, `researcher_task`, `retry_prompt` and the refusal message are unchanged: the refusal names 500 and about 450, never the hard limit.
- No change under `src/` other than `strategies/bull_bear/parameters.py` and `strategies/bull_bear/CLAUDE.md`.
- Work on branch `fix/bull-bear-note-hard-limit-800` off `main`; no push, no merge unless asked.

## Review Focus

- A note of exactly 800 characters is recorded and one of 801 is refused (the boundary moved; pinned in Step 1).
- The refusal never quotes the hard limit: the test checks the actual `note_hard_max_chars`, not the literal "700" (Step 3).
- The trimmed fallback note fits the new limit: `trim_note(..., max_chars=800)` output is always recorded (Step 3).
- A note between 701 and 800 characters, refused before, is now recorded without a refusal (Step 1, the 762-character case).
- The judge's context grows by at most 100 characters per stock; not testable cheaply, checked by the window estimate in the spec (about 69k of 81,920 tokens).

---

### Task 1: Raise the note hard limit to 800

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py:67` (the `note_hard_max_chars` default and its comment)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the "A note is asked for at `note_max_chars` (500) but refused only above `note_hard_max_chars` (700)" paragraph)
- Modify (spec, commit with the change): `docs/superpowers/specs/2026-11-13-bull-bear-note-hard-limit-800-design.md`
- Test: `tests/strategies/bull_bear/test_bull_bear_params.py:24`
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py:35-110, 341-346`
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py:120, 614-646`

**Interfaces:**
- Consumes: `BullBearParams.note_hard_max_chars: int`, `trim_note(text, *, max_chars, min_chars) -> str`, the `submit_note(symbol, note)` tool from `submit_tools(recorder)`.
- Produces: `BullBearParams().note_hard_max_chars == 800`.

- [ ] **Step 0: Branch**

Run: `git checkout -b fix/bull-bear-note-hard-limit-800`
Expected: `Switched to a new branch 'fix/bull-bear-note-hard-limit-800'`

- [ ] **Step 1: Write the failing tests for the new limit**

In `test_bull_bear_params.py::test_the_defaults_are_the_specs_and_valid`:

```python
assert (params.note_max_chars, params.note_hard_max_chars) == (500, 800)  # the researcher is asked for 500, a note up to 800 is kept
```

In `test_bull_bear_handoff.py`, rewrite `test_a_note_up_to_the_hard_limit_is_recorded_though_the_researcher_is_asked_for_less` and add one test:

```python
assert params.note_max_chars < params.note_hard_max_chars == 800
assert tools["submit_note"]("AAA", "x" * 800) == {"status": "recorded"}


def test_a_note_one_character_over_the_hard_limit_is_refused_and_one_refused_under_700_before_is_now_recorded() -> None:
    recorder, tools = _tools()
    recorder.expect_note("AAA")
    assert "801 characters" in tools["submit_note"]("AAA", "x" * 801)["error"]
    assert tools["submit_note"]("AAA", "x" * 762) == {"status": "recorded"}  # refused while the limit was 700
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_handoff.py -q`
Expected: 3 FAILED, each on the 700 default (`(500, 700) == (500, 800)`, `700 == 800`, the 762-character note refused).

- [ ] **Step 3: Set `note_hard_max_chars: int = 800` in `parameters.py`, then move the tests whose inputs assumed 700**

The comment keeps the 10-31 figures and adds the 2026-11-13 HyperQwen ones (14% of researcher runs refused, refused drafts median 741, 6 of 61 left at 800).

Then, so the existing tests keep testing what their names say (inputs move above the new limit, assertions keep their meaning):

- `test_a_long_note_is_refused_with_the_asked_length_and_how_to_shorten_it`: 762 -> 862 in the call and in the expected message (`"the note for AAA is 862 characters; a note is at most 500: rewrite it to about 450 characters by dropping your least important facts, not by trimming words"`); replace `assert "700" not in error` with `assert str(BullBearParams().note_hard_max_chars) not in error`.
- `test_the_same_long_note_sent_again_is_called_out`: 762 -> 862 (three places, message included), 750 -> 850.
- `test_the_recorder_keeps_the_last_note_refused_for_length_until_the_next_stage`: 800 -> 900, 750 -> 850 (both places).
- `test_a_note_refused_for_another_reason_leaves_no_refused_note`: unchanged (refused for its symbol).
- `test_a_trimmed_note_is_always_accepted_by_the_recorder`: parametrize `[801, 850, 1200]`, `max_chars=800`.
- `test_bull_bear_pipeline.py`: `LONG = ("2025-10-27: revenue rose 12% y/y to $1.58B. " * 22)[:920]  # whole sentences, 920 characters: over the hard limit (800)` (920, not 900: `[:900]` ends on a space and the recorder strips it to 899); every `trim_note(LONG, max_chars=700, min_chars=250)` -> `max_chars=800`; the warning check `trimmed from 800 to` -> `trimmed from 920 to`.

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all pass (399 + 1 new = 400).

- [ ] **Step 5: Update `bull_bear/CLAUDE.md`**

In the note paragraph: `note_hard_max_chars` (700) -> (800); after "The limit went from 600 to 700 after the 2026-10-31 2Y run: ..." add one sentence: it went from 700 to 800 on 2026-11-13 for HyperQwen (Qwen3.8 W4A16), whose notes run longer (median 563 against 461 on vLLM): 14% of researcher runs had a length refusal at 700, the refused drafts had a median of 741, a refused run cost 1.6 more model calls and 22 s, and at 800 6 of the 61 refusals remain (replay of the 2026-11-13 run).

- [ ] **Step 6: Quality gate**

Run: `uv run ruff check && uv run ruff format --check && uv run pytest -q && uv run interrogate -v src | tail -3`
Expected: ruff exits 0, formatting clean, every test passes, docstring coverage >= 80%.

- [ ] **Step 7: Review, then commit**

Stage the changes, run `/ponytail:ponytail-review` on the staged diff, fix every "Must fix", re-run Step 6, list the "Should fix" to the user. Then:

```bash
git add src/trading_agent_framework/strategies/bull_bear/parameters.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_handoff.py tests/strategies/bull_bear/test_bull_bear_pipeline.py docs/superpowers/specs/2026-11-13-bull-bear-note-hard-limit-800-design.md docs/superpowers/plans/2026-11-13-bull-bear-note-hard-limit-800.md
git commit -m "fix(bull_bear): raise the note hard limit from 700 to 800 characters"
```

The commit message body states the replay figures and ends with the attribution line from the session.

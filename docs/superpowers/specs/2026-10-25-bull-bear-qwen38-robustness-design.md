# bull_bear: robustness for Qwen 3.8 (note refusals, judge token cap, max_in as a ceiling)

Date: 2026-10-25 · Amends `2026-10-11-bull-bear-debate-realignment-design.md` (kept as history).

## Why

The 1Y backtest on Qwen3.8-27B (`logs/bull_bear/backtesting/2026-10-23_231712_backtesting`, `LLM_REASONING_EFFORT=medium`)
made +42%. The baseline on Qwen3.6-27B (`2026-10-19_234111_backtesting`) made +66%, and six 3.6 runs on the same period
made +62% to +91%. No code bug was found. The per-stock decision quality (`uv run bull-bear-scores`) cannot be told
apart between the two models. The whole gap opened in May-June 2026, mostly on MU. Three model behaviours made the 3.8
run fragile; this design fixes each one.

1. **The researcher loses notes to refusals.** A note above `note_hard_max_chars` (600) is refused with
   `the note for PLTR is 662 characters: keep it under 600`. Qwen 3.6 cut 50-100 characters once and it went
   through (13 of 14 runs). Qwen 3.8 shaves a few characters, then re-sends the same text:
   - PLTR: `[769, 699, 662, 662, 662, 662, 662, 662, 662, 662, 662, OK]`
   - ABVX: `[616, 610, 601, 601, 601, 601, 601, 601, 601, OK]`

   In total: 81 refusals vs 16, and 20 runs hit `max_agent_steps` (60) and gave their stock "research unavailable".
   A loop is never retried, so `retry_prompt` is never seen. The only correction the model gets is the tool's error
   string. A run that stops after a refusal (no loop) gets the forced retry. That retry is a fresh run, so the note is
   rewritten from the fact sheet alone: the news and filings the first run found are thrown away.
2. **The judge hits the 12,288 output-token cap.** 14 judge calls on 3.8 (more than 10% of judge calls) spent the whole
   cap on reasoning; on 3.6, 6 runs ended this way. Every such run was saved by the forced retry, which decides in a
   fresh run that has lost its memory lookups. Bull and bear peaked at 8.9k and 10.7k. The judge's prompt reaches
   20.5k tokens (90th percentile 14.8k) of the 32,768 window. The 3.8 judge made 107 memory calls (3.6: 34), and each
   result stays in its context.
3. **The judge treats `max_in` (10) as a quota.** Full books: 39 of 55 reviews on 3.8, 18 of 55 on 3.6. `out`
   verdicts citing the cap: 57 vs 14 (27 vs 8 on stocks already held). The prompt invites it: *"A stronger candidate
   that needs a place under max_in is a new, material fact: put the weaker stock out and say so in what_changed."* On
   2026-05-05 the 3.8 judge put MU out "for the slot" (bull high, bear low). The cooldown then kept MU out of its
   May-June run; MU alone accounts for $1.66k of the $2.4k gap.

## Goals

- No researcher run loses the research it found because its note was too long.
- Qwen 3.8 shortens a refused note properly on the next attempt, instead of re-sending it.
- The judge has room to think: fewer runs end at the token cap.
- The judge puts a stock in only on its own merits; 10 is a ceiling, not a target.

## Non-goals

- The cooldown rules. Every sell of a holding still starts the 4-review cooldown, including a cut made for room; the
  user chose this for simplicity, to be revisited after the backtest.
- The ceiling itself: `max_book_size` stays 10. Removing it was considered and dropped.
- Reasoning effort per agent (a TODO item), a larger vLLM window, and a thinking-off retry.
- The judge stays blind to holdings: no prompt says what is held or that an `out` sells.

## Anti-goals

- Code must never reword agent text. Cutting a refused note short is the only edit allowed.
- No new forced retry or in-run correction with thinking off (tried and reverted on 10-19; see `bull_bear/CLAUDE.md`).

## Decisions (binding)

### 1. Researcher notes

**A. The refusal message.** For a note longer than `note_hard_max_chars`, `validate_note`'s error:

- Gives the note's length and the requested maximum, `note_max_chars` (500). It never mentions 600, so the
  "ask 500, refuse above 600" margin stays hidden from the model.
- Asks for a rewrite to about 90% of `note_max_chars` (450) by dropping the least important facts, not by trimming
  words.

Wording (the builder may polish it, but keep the numbers and the instruction):
`the note for PLTR is 662 characters; a note is at most 500: rewrite it to about 450 characters by dropping your least important facts, not by trimming words`.

If the refused note is the same text (after strip) as the previous note refused for length in this stage, the error
starts with `this is the same text you sent before: ` and then gives the message above.

**B. Keep the last refused note, trimmed.** The recorder remembers the most recent note refused for length in the
armed stage (its full text). When a researcher run ends with no recorded note and such a refused note exists, the
pipeline uses that note, trimmed, as the stock's note. This replaces both:

- the "research unavailable" note after a loop (`AgentLooped`), and
- the forced retry after a run that stopped without a recorded note.

The rules:

- Trimming (pure function): keep the longest prefix of at most `note_hard_max_chars` characters that ends at a sentence
  end (`.`, `!` or `?` followed by whitespace or the end of the text). If that prefix is shorter than half of
  `note_max_chars`, cut instead at the last whitespace within the limit and end with `…` (still within the limit).
  Then strip.
- Only a refusal for length counts. A refusal for a wrong symbol or an empty note leaves no refused note.
- With no refused note, behaviour is unchanged: a loop gives "research unavailable", and a run that ended without a
  note gets the forced retry.
- The bull, bear and judge stages are unchanged.
- Logging: one WARNING per trimmed note,
  `[researcher] note for PLTR trimmed from 662 to 581 characters: the last refused note is kept with its research`.
  The run's own loop error, or the refused-submission warning, is still logged as today.
- `reviews.jsonl`: `notes` gains `trimmed`, the list of symbols whose note was trimmed. These are counted in `written`
  and not in `failed`.

### 2. The judge's token cap and memory budget

- New `BullBearParams.judge_max_output_tokens: int | None = 16384`, used for the judge instead of
  `max_output_tokens`. It is validated like `max_output_tokens` (None or at least 1). Researcher, bull and bear keep
  `max_output_tokens` (12288).
- `strategy.parameters` (and so `settings.json`) records `judge_max_output_tokens` next to `max_output_tokens`.
- When prompt plus cap exceeds the window, `output_cap_middleware`'s fallback (sent once without the cap) handles it as
  today. No change in `agents/`.
- `judge_memory_budget`: 6 → 3.

### 3. The judge prompt: `max_in` is a ceiling

In `JUDGE_SYSTEM`:

- Replace *"Put at most max_in stocks in, given in the constraints: when more deserve it, keep the strongest cases in
  and give the others out with your reason. You may put none in."* with a version that says:
  - `max_in` (in the constraints) is a ceiling, not a target.
  - Put a stock in only when its own case earns a place; fewer than `max_in` in is normal, and none is allowed.
  - Only when more stocks earn a place than `max_in` allows, keep the strongest cases in and put the others out with
    your reason.
- Delete *"A stronger candidate that needs a place under max_in is a new, material fact: put the weaker stock out and
  say so in what_changed."* The rest of the `previous_verdict` / `what_changed` rule stays.

The `max_in` validation in `validate_verdicts`, the constraints block and the cooldown are unchanged.

## Constraints

- Follow the repo's conventions (root and `strategies/bull_bear/CLAUDE.md`):
  - hand-written fakes, not mocks; no network in tests;
  - `handoff.py` keeps no `from __future__ import annotations`;
  - TDD, numbered tasks, a review-fix commit after each task.
- Update `strategies/bull_bear/CLAUDE.md` where it states:
  - the note limits and refusal (the gotcha "A note is asked for at 500 but refused only above 600");
  - "a failed run gives that stock the note research unavailable";
  - the judge's memory budget (6) and the output cap (12288 for all four agents);
  - the `notes` record in `reviews.jsonl`.
- Update the `max_agent_steps` comment in `parameters.py` if it still cites 6 memory calls as the worst run.
- Existing tests that pin the old values (`judge_memory_budget == 6`, the refusal text, the forced retry of a
  researcher run after a length refusal) change with the behaviour.

## Success criteria (the next 1Y backtest on Qwen 3.8, `medium`)

- `reviews.jsonl`: `notes.failed` near 0. Notes lost to length show up in `trimmed`, not `failed`.
- Researcher refusals per run: no long runs of the same length (e.g. `601, 601, 601`).
- Judge runs ending at the cap: fewer than 10.
- Full books (10) well below 39 of 55, and few `out` verdicts citing the cap.
- Returns are not a criterion of this change on their own: they will be compared, not promised.

## Calls made while writing this up (not discussed with the user)

- A run with no length refusal still gets the forced retry. If the retry's own note is then refused for length,
  that note is kept, trimmed (it is the most recent note refused for length in the armed stage).
- `judge_max_output_tokens=None` means no cap for the judge. It does not fall back to `max_output_tokens`.
- For a stock whose note was trimmed, the `[researcher] no note for X` warning is replaced by the trimmed warning.
  The run's own loop error or refusal warning stays.
- Trimming edge cases:
  - A sentence end is a `.`, `!` or `?` followed by whitespace in the full text, or the last character of the text.
  - With no whitespace to cut at, cut at the limit minus one character and add `…`.

## Left to the builder

- Where the refused-note memory lives (recorder attribute or method) and how the pipeline reads it.
- Exact phrasing of the prompt and error text, within the decisions above.
- Task breakdown and order.

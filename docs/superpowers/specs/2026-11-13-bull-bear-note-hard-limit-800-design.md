# bull_bear: raise the note hard limit from 700 to 800 characters

Short design (small change).

## Intent

Cut the model turns and time that note-length refusals cost the researcher. On HyperQwen (Qwen3.8 27B W4A16), notes run
longer than on vLLM (median accepted note 563 characters against 461; 80% over the 500 asked against 30%; 36% over 600
against 5%), so more first drafts land just above the 700-character hard limit and are refused.

Measured in the 2026-11-13 budget A/B replay (424 researcher runs over the 106 tasks of
`logs/bull_bear/backtesting/2026-11-13_194705_backtesting`):

- 61 of 424 runs (14%) had at least one length refusal.
- A refused run took 4.89 model calls against 3.33, and 98.5 s against 76.2 s: about 4% of the researcher's total time.
- The refused drafts are near misses: median 741 characters, interquartile range 712 to 768, longest 1,088.
- At a hard limit of 800, 6 of the 61 refusals would remain; at 900, 3; at 1,000, 1.

## Goals, non-goals, anti-goals

- Goal: fewer length refusals, so fewer researcher turns and less time per review.
- Non-goal: shorter notes, or notes closer to 500. The ask stays 500.
- Anti-goal: changing what the model is asked for. "Widen the limit, never the ask": the model aims at the ask
  (median first draft about 495 on vLLM), so raising the ask would move the drafts up with it.

## Approaches

1. **Raise `note_hard_max_chars` from 700 to 800. Chosen.** One parameter; the prompt, the refusal message (which names
   500 and never the hard limit) and the trim fallback stay as they are. It removes about 90% of the refusals and still
   refuses real outliers. Cost: up to 100 more characters per note in the bull, bear and judge contexts, about 2,500
   characters (under 1k tokens) at 25 stocks; the judge's window check keeps a wide margin (about 69k of 81,920 tokens).
2. Never refuse, trim at a sentence (`trim_note`). Removes every refusal turn, but for the 14% of long notes code
   drops the last facts where today the model chooses which to drop. A bigger behaviour change for about 0.6% more time
   saved.
3. Ask for a number of facts instead of characters. A prompt change with an uncertain effect on the notes; would need
   its own A/B.

Dropped: a character-counting tool (adds a model turn to every note to save about 1.6 calls on 14% of them) and a
token-based limit through vLLM's `/tokenize` (the model cannot count tokens either; ties the validator to a vLLM-only
HTTP endpoint).

## Design

- `strategies/bull_bear/parameters.py`: `note_hard_max_chars: int = 800`; its comment records the 2026-11-13 HyperQwen
  numbers next to the 10-31 ones.
- Nothing else in `src/` changes: `HandoffRecorder` (refusal) and `pipeline._research_notes` (`trim_note` fallback)
  already read `note_hard_max_chars`.
- Tests: the defaults test asserts `(500, 800)`. Tests that hard-code 700 as "the hard limit" read
  `note_hard_max_chars` or use 800 instead. The pipeline's `LONG` note (now exactly 800 characters, "over the hard
  limit") becomes longer than 800 so it is still refused. The test that the refusal never quotes the hard limit keeps
  checking the actual limit.
- `strategies/bull_bear/CLAUDE.md`: the note paragraph says 800, and records why it went from 700 to 800 (HyperQwen
  drafts, 14% of runs refused, refused drafts median 741).

## Left to the builder

Exact wording of the comment and the CLAUDE.md sentence; whether a hard-coded 700 in a test becomes 800 or reads the
parameter (prefer the parameter where the test is about "the hard limit").

## Not verified

The effect on a full backtest (time per review, notes trimmed) is projected from the replay, not measured; a later
HyperQwen run on the same window will show it. Runs before this change used 700, so their note lengths are not
comparable at the top end.

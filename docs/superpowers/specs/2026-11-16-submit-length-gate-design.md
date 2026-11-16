# Submit length gate: count, refuse a bounded number of times, then trim

Date: 2026-11-16. Status: proposed. Reference strategy: sector_pods (bull_bear adopts it later, separately).

## Why

Backtest `logs/sector_pods/backtesting/2026-11-16_085725_backtesting` (107 weeks) logged 726 trims: 48-79% of every
`sector_view`, risk `summary`, PM `summary` and `overrides`, 9-14% of the financials, consumer and risk reasons. The
models aim at the stated limit and overshoot by up to 1.7x (they cannot count characters); `trim_text` then cuts at a
sentence, so a 410-character view loses ~100 characters, usually its conclusion. Refusing instead (run 2026-11-11)
looped until `max_agent_steps`, because a refused model guesses how much to cut.

Spike (2026-11-16, 84 replayed runs of this backtest, qwen3.8-27b, scratchpad `spike_count.py`):

| Arm (28 runs each) | First submit refused | Final submit still over | Long text median | Time per run |
|---|---|---|---|---|
| refuse once, then accept | 64% | 29% | 376 | 114 s |
| count tool + prompt step | 0 | 0 | 308 | 120 s |
| same, first count inflated x1.3 | 0 | 0 | 302 | 141 s |

With the count, the agent writes under the limit from the start; when told a count was over (20 cases), it shortened
and recounted every time. A refusal that names each text's count is still needed as a backstop, and a trim as the last.

## Goals

- Agents count their texts with a core `count_characters` tool before submitting.
- A submit tool refuses over-long texts, naming each text's character count and stated limit, at most
  `max_submit_attempts - 1` times per node run; the last attempt is accepted, trimmed at a hard cap.
- Only length refusals use an attempt. A refusal for anything else (a weight cap, a missing score) does not.
- The mechanism lives in `agents/` as strategy-agnostic plumbing; sector_pods is the first user.

## Non-goals

- Migrating bull_bear (its notes, arguments, verdicts). It reuses this later, in its own change; that change drops its
  pipeline-level `refused_note` trim.
- Changing `reviews.jsonl`, the graph, the tool budgets or what the agents write.

## Anti-goals

- No strategy vocabulary in `agents/`: no `weights`, `scores`, `sector_view`, `reason`... as names, arguments or
  messages. `agents/` only sees strings, `TextLimit`s and opaque payloads.
- No loop: whatever the model does, the gate stops refusing after the configured attempts.

## Approaches considered

- **A. A length gate the submit tool calls (picked).** Framework object, strategy wrapper hands it labelled texts.
  Independent of how the submission is recorded; one line of wiring per submit tool.
- B. Built into `SubmitSlot`'s validation: least wiring, but couples length logic with the slot and its validator.
- C. A middleware rewriting submit tool arguments from field paths: hidden behaviour, fiddly nested paths, hard to test.

## Design

### Framework: `agents/tools/text.py` (new), exported from `agents/tools/__init__.py`

- `count_characters(texts: list[str]) -> {"counts": [int, ...]}`: the stripped length of each text, in order. Returns
  only the counts: the agent decides. Built by a factory `text_tools() -> [count_characters]` like the other modules.
  A strategy lists it in its agents' `exempt_tools` (outside the research budget).
- `TextLimit(stated: int, hard: int)`, frozen: `stated` is what the prompt quotes and the refusal threshold; `hard`
  (>= stated) is the last attempt's trim length. The hard cap is never quoted to the model.
- `LengthGate(max_attempts: int, *, name: str, log_tag: str)` (`name` is the submit tool's name, used in messages):
  `reset()`; `review(texts: Sequence[tuple[str, Any, TextLimit]])` takes `(label, text, limit)` triples (a sequence, not
  a mapping: two objections on the same symbols share a label) and returns either `{"error": ...}` or the values in
  order, stripped. Labels are the strategy's strings.
  - A non-string value is passed through untouched (the strategy's validator refuses it).
  - Some text over its `stated` and fewer than `max_attempts - 1` length refusals since `reset`: count one refusal,
    log INFO, return the error. Message: every over-long text, `<label> has <n> characters (limit <stated>)`, then
    `count them with count_characters, shorten those texts and submit the whole submission again`.
  - Otherwise: accept; a text over `hard` is cut with `trim_text(max_chars=hard, min_chars=stated // 2)` and logged
    WARNING `<log_tag> <name>: <label> trimmed from <n> to <m> characters (limit <hard>)` (today's format).
  - `max_attempts == 1` never refuses (today's trim-only behaviour).
- `count_step(submit_tool: str, fields: str) -> str`: the prompt paragraph (spike wording): count every text with
  `count_characters` before calling `<submit_tool>`, shorten and recount any text over its limit, submit only texts
  whose last count fits, unchanged.
- `trim_text` moves here from `strategies/common/handoff.py` (`agents/` must not import `strategies/`); its importers
  (bull_bear pipeline, sector_pods handoff, their tests) import it from `agents/tools/text.py`. No alias left behind.

### Framework: `agents/graph.py`

`SubmitSlot(validate, gate: LengthGate | None = None)`: `arm()` also calls `gate.reset()`, so a forced retry within the
same node run shares the attempts and each new run starts fresh. The slot never calls `review` (the submit tool does).
Exposes `gate` for the tool factories.

### Strategy: sector_pods

- `params.py`: `reason_hard_max_chars` 300, `summary_hard_max_chars` 700, `max_submit_attempts` 3, `max_agent_steps`
  40 -> 60 (bull_bear's value; a worst-case PM run with 3 attempts needs ~35). Validation: each hard >= its stated,
  `max_submit_attempts` >= 1.
- `handoff.py`: each submit tool builds `{label: (text, limit)}` (`reason for XLK`, `reason for XLK, IGV` for an
  objection, `sector_view`, `summary`, `overrides`), calls `slot.gate.review`, returns its error, else writes the
  texts back into the payload and calls `slot.submit`. `_trimmed` / `_trimmed_reasons` go. The validators check
  lengths against the hard caps (else a 450-character text the gate accepted would be refused).
- `strategy.py`: each `SubmitSlot` gets `LengthGate(params.max_submit_attempts, log_tag="[sector_pods]")`; every agent
  gets `text_tools()` and `count_characters` in `exempt_tools`.
- `prompts.py`: the three system prompts end their submit step with `count_step(...)`; they keep quoting the stated
  limits only.
- `CLAUDE.md` (sector_pods and agents): replace "an over-long free text is trimmed, never refused" with the gate rule,
  the spike numbers and the run it came from.

## Acceptance

- Tests: gate (refuse with counts, attempts counted on length only, last attempt trims at hard and passes texts
  between stated and hard, `max_attempts=1`, non-strings, reset), `count_characters`, `SubmitSlot.arm` resets the gate,
  sector_pods submit tools end to end through the slot, params validation, prompts quote the step and the stated limits.
- `grep -rniE "weight|score|sector_view|objection|overrides" src/trading_agent_framework/agents/tools/text.py` and
  the `SubmitSlot` diff: empty.
- Quality gate green (ruff, format, pytest, coverage >= 80%, interrogate >= 80%, pyright on touched files); `/simplify`
  on the diff after green, aiming at what bull_bear will reuse.
- Follow-up (by the user): a sector_pods backtest; expect trims near 0 and no `loop:` abandons.

## Calls made by Claude (check these)

- The refusal does not say which attempt it is, nor that the last one is accepted (a model told so may stop shortening).
- Refusals are logged at INFO, trims stay WARNING.
- `count_characters` takes a list (one call per run instead of up to 17 for the PM).
- `trim_text` moves to `agents/tools/text.py` rather than being duplicated.

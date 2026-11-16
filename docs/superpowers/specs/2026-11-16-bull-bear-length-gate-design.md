# bull_bear length gate, and one submit recorder for every strategy

Date: 2026-11-16. Status: proposed. Follows `2026-11-16-submit-length-gate-design.md` (sector_pods, merged in #20), whose
non-goal "migrating bull_bear" this change carries out.

## Why

sector_pods now counts its texts with `count_characters` and puts them through a `LengthGate` at submit: refused with
each count while attempts remain, then accepted, cut at a hard cap (spike: 0 over-long first submits with the count,
against 64% without). bull_bear still has its own, older length handling:

- the researcher's note is asked at `note_max_chars` (500) and refused above `note_hard_max_chars` (800) with a custom
  message (`NoteTooLong`, "this is the same text you sent before"); a run that ends after such a refusal keeps the
  refused note, trimmed by the pipeline (`HandoffRecorder.refused_note`, `skip_retry`, `trim_text` in `pipeline.py`);
- the bull's and bear's `argument` and the judge's `reason` / `what_changed` have no hard cap: over 300 characters the
  whole submission is refused, with no attempt limit and no trim.

The user wants bull_bear on the same policy, through the same `agents/` code, and the two strategies to share more:
in particular one submit recorder instead of `SubmitRecorder` (strategies/common, 4 strategies) and `SubmitSlot`
(agents/graph.py, sector_pods).

## Goals

- bull_bear's four agents count their texts with `count_characters` and submit through a `LengthGate`: researcher
  `note`; bull and bear `argument` per stock; judge `reason` and `what_changed` per stock.
- bull_bear's own length machinery goes: `NoteTooLong`, the "same text" message, `refused_note`, `skip_retry` and the
  pipeline's `trim_text` call. The gate's last-attempt trim replaces it.
- One recorder class, `SubmitRecorder`, in `agents/`; `SubmitSlot` is deleted. sector_pods' graph nodes and the four
  stage-based strategies (bill_ackman, bull_bear, congress_trades, insider_trades) all use it.
- The "put an entry list's text fields and some top-level texts through the gate, then write them back" logic that
  sector_pods has in `handoff._submit` moves to `agents/` and both strategies call it.

## Non-goals

- Order-event logging hooks (copied between bull_bear and sector_pods): out of scope, by the user's choice.
- Length gates for bill_ackman, congress_trades and insider_trades: they only change imports.
- Changing what the agents write, the graph, tool budgets, `reviews.jsonl` keys other than `notes.trimmed`, or the
  stated limits.

## Anti-goals

- No strategy vocabulary in `agents/` (same rule and grep as the sector_pods spec): field names such as `reason` or
  `argument` are arguments passed in, never literals in `agents/`.
- No behaviour change for bill_ackman, congress_trades, insider_trades.
- No loop: the gate stops refusing after `max_submit_attempts - 1` length refusals per armed stage.

## Approaches considered (recorder merge)

- **A. One `SubmitRecorder` in `agents/`, the existing stage-based class, with optional gates (picked).** sector_pods
  nodes get a single-stage recorder. Smallest change for the four working strategies; keeps the stage machinery
  insider_trades relies on (it re-arms a stage within a run).
- B. Keep both classes, `SubmitRecorder` routing each stage to a `SubmitSlot`: two classes remain, no real merge.
- C. Only `SubmitSlot`; every strategy builds one slot per stage: rewrites four strategies' `expect_*`/`_validate`;
  insider's in-run stage switch has no natural home.

Semantic difference resolved: `SubmitSlot` kept the LAST valid submission, `SubmitRecorder` keeps the FIRST and refuses
later ones ("already recorded"). Every sector_pods agent has `final_tools=[submit]` (`agent_sector_pods.py`), so a
recorded submission ends the run, and `AgentNode` retries only when nothing was recorded. The one remaining case, two
valid submit calls requested in the same model turn, now records the first instead of the second. Accepted.

## Design

### `agents/`: the recorder

- `HandoffError` and `SubmitRecorder` move from `strategies/common/handoff.py` to a new `agents/submit.py`
  (`agents/` must not import `strategies/`). Every importer imports them from there; no re-export left behind in
  `common/handoff.py` (the house rule from the `trim_text` move). `common/handoff.py` keeps its readers, imports
  `HandoffError` from `agents/` for its own use.
- `SubmitRecorder.__init__(gates: Mapping[str, LengthGate] | None = None)`: an optional gate per stage. `_arm(stage, ...)`
  resets that stage's gate; a `gate` property returns the armed stage's gate or `None`. Recorders without gates behave
  exactly as today.
- A gate review must not spend an attempt on a call the recorder would refuse anyway (stage not armed, already
  recorded). The builder picks the mechanism (e.g. the submit tool asks the recorder, or `submit` takes the texts).
- A gate refusal is recorded in `last_error` like any other refusal, so bull_bear's forced-retry prompt names the
  length refusal (today sector_pods' `_submit` returns it before the slot sees it).
- The forced retry SHARES the first run's attempts: neither `_run_stage` nor `AgentNode` re-arms between the two runs,
  as the sector_pods spec already decided ("a forced retry within the same node run shares the attempts"). So a retry
  after spent length refusals has its first over-long submit accepted, cut at the hard cap. No change to either
  retry loop; the step maths (one gate's worth of submits per armed stage) holds.
- Single-stage recorder for graph nodes (replaces `SubmitSlot`): built from a tool name, a validator
  `validate(payload, state) -> None` that raises `HandoffError` (the recorder keeps the payload), and an optional gate; `arm(state)` arms its one
  stage with the graph state as context; `submission` / `submitted` as on the base class. `AgentNode` takes it instead
  of a slot (`slot` field renamed `recorder`; reads `submitted`/`submission`, not `value`).

### `agents/tools/text.py`: the shared gate helper

A function generalising sector_pods' `_submit` (name the builder's call), with no field names of its own:
given the gate, a list of entries (dicts; non-dicts and a non-list left untouched for the validator), a mapping
`field -> TextLimit` for the entry text fields to check (several fields per entry: the judge has `reason` and
`what_changed`; an absent / `None` optional field passes through), a function giving each entry's label subject
(`AAPL`; sector_pods' `XLK, IGV` for an objection), and labelled top-level texts `label -> (value, TextLimit)`.
Returns the gate's `{"error": ...}` or the entries and top-level texts with the reviewed values written back. Labels:
`<field> for <subject>` and the top-level label as given (today's sector_pods labels unchanged).

### bull_bear

- `parameters.py`: `argument_hard_max_chars` 450, `reason_hard_max_chars` 450 (used for `what_changed` too: the same
  ratio as sector_pods' reason 200 -> 300), `max_submit_attempts` 3; keep `note_max_chars` 500 / `note_hard_max_chars`
  800 (now the note's `TextLimit`). Validation: each hard >= its stated, attempts >= 1. Recorded in `settings.json`
  like the other parameters. `max_agent_steps` stays 60: the worst researcher run (7 searches, 2 statements, up to 3
  counts, 3 submits = 15 calls) needs 3 x 15 + 5 = 50.
- `handoff.py`: `HandoffRecorder` gets a `LengthGate` per stage (name = the stage's tool, `log_tag` "bull_bear").
  Each submit tool puts its texts through the shared helper before `recorder.submit`. `NoteTooLong`, `_refused_note`,
  `refused_note` and the `_arm` override go. The validators check lengths against the HARD caps (a 400-character
  argument the gate accepted on the last attempt must not then be refused by `required_text`); `validate_note` loses
  `ask_chars`.
- `pipeline.py`: `_research_notes` loses the refused-note branch and `trimmed`; `_run_stage` loses `skip_retry`. A
  researcher run that ends with no recorded note gets the normal forced retry, then "research unavailable".
  `reviews.jsonl` `notes` becomes `{written, failed}` (no reader uses `trimmed`: grep of src/ and scripts/).
- `agent_bull_bear.py`: every agent gets `text_tools()`, `count_characters` in `exempt_tools` (the researcher's budget
  counts `search_news` only, as now).
- `prompts.py`: each task ends with `count_step(<submit tool>, <texts in order>)`: researcher `` `note` ``; bull and
  bear `` every `argument` ``; judge `` every `reason`, then every `what_changed` you give ``. Prompts keep quoting
  the stated limits only, and still contain no `held`, `weight_pct` or `forced`.
- `CLAUDE.md` (bull_bear): replace the note-limit paragraph and every "refused note kept trimmed" mention with the gate
  rule; the "normal noise" list loses the researcher note-length loop.

### sector_pods

- Each node gets the single-stage recorder (with its gate) instead of `SubmitSlot`; the validators raise `HandoffError`
  instead of returning a message, and use `common/handoff.py`'s `required_text` / `optional_text` in place of its own
  `_text` (messages change wording, not meaning). `check_weights` keeps returning `str | None` (the execute node calls
  it too); the PM validator raises its message.
- `handoff._submit` becomes a call to the shared helper; `_subject` stays as the label function.

### Other strategies

bill_ackman, congress_trades, insider_trades: import `HandoffError` / `SubmitRecorder` from `agents/submit.py`. Nothing
else.

## Acceptance

- Tests: recorder (gate reset on arm per stage, gate property, no attempt spent when not armed or already recorded,
  single-stage `arm(state)`), shared helper (several fields per entry, optional field absent, top-level texts, malformed
  entries untouched), `AgentNode` with the new recorder, bull_bear submit tools end to end (refuse with counts, last
  attempt trims at hard and records), parameters validation, prompts quote the count step and the stated limits only,
  pipeline: a researcher run ending after its length refusals gets the forced retry, whose first over-long note is
  recorded cut at 800 (shared attempts), and the retry prompt names the length refusal. Existing tests of the four
  stage-based strategies pass with only import changes.
- `grep -rn "SubmitSlot\|refused_note\|NoteTooLong\|skip_retry" src tests`: empty. `grep -rniE
  "argument|reason|note|verdict|weight|score" src/trading_agent_framework/agents/submit.py` and the helper: empty.
- Quality gate green (ruff, format, pytest incl. `uv sync --extra dashboard`, coverage >= 80%, interrogate >= 80%,
  pyright on touched files); `/simplify` after green.
- Follow-up (by the user): a bull_bear backtest; expect no `run stopped as a loop` from note length and few trims.

## Calls made by Claude (check these)

- Hard caps 450 for argument and reason/what_changed (1.5x, as sector_pods); `max_submit_attempts` 3.
- A researcher whose run ends right after a length refusal no longer keeps its refused note: forced retry, then
  "research unavailable". The spike saw no loops once the agent could count.
- `notes.trimmed` dropped from `reviews.jsonl`; trims are logged as WARNINGs by the gate.
- sector_pods' `_text` replaced by the common readers (wording of its refusals changes).
- New module `agents/submit.py` for the recorder; `AgentNode.slot` renamed `recorder`.
- The forced retry shares the gate's attempts with the first run (as in sector_pods today), and gate refusals land in
  `last_error`.

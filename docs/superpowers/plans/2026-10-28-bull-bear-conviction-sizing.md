# bull_bear conviction sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bull_bear judge rates every stock of the debate `strong`, `standard` or `weak`, then puts in only strong or standard ones (a weak stock is always out); code sizes each position by its strength alone (14% / 9%, scaled down above 98%, the rest parked in SHV as today).

**Architecture:** `submit_verdicts` gains a validated `strength`; the judge prompt explains it as a balance of the bull's and the bear's cases. A replay of the new judge on two logged runs gates the rest. Then a pure `strength_weights` replaces `capped_inverse_volatility` in the pipeline, the state (version 3) remembers each stock's strength so the judge sees it next time, the logs carry it, and `bull-bear-scores` scores it.

**Tech Stack:** Python 3.14, `uv`, pytest, ruff; LangGraph agents through `AgentManager`; vLLM server (Qwen 3.8) for the replay only.

**Spec:** `docs/superpowers/specs/2026-10-28-bull-bear-conviction-sizing-design.md`

## Global Constraints

- Work on branch `feat/bull-bear-conviction-sizing` (created in Task 1); never push; the user squash-merges into `main` later.
- Never stage or edit `TODO.md` (the user's uncommitted file): `git add` only this plan's files, by path.
- Commit messages: `<type>(bull_bear): <what> (Task N)`, ending with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- `handoff.py` has NO `from __future__ import annotations` (tool schemas come from real annotations).
- Tests use hand-written fakes, never `MagicMock`; the test suite never touches the network.
- Strengths: exactly `strong`, `standard`, `weak` (lower case after normalisation). Weights: `strong_weight` 0.14, `standard_weight` 0.09 (no weak weight: a weak stock is never put in). SHV parking stays as it is; no new SHV work.
- Code never reads the bull's conviction or the bear's risk to size; the judge's strength is the only sizing input; no volatility term.
- No prompt may contain `held` or `weight_pct` (existing test), nor the weight numbers.
- Verification for every task: `uv run pytest tests/strategies/bull_bear/ -q` and `uv run ruff check` clean; the last task runs the whole suite.

## Review Focus

1. A judge that writes `"Strong"` or `" weak "` is normalised (like `verdict`); one that leaves a stock without a strength, or puts a weak stock `in`, is refused with the stock named: Task 4 tests.
2. The paper account's existing version-2 state file after deployment: verdicts, cooldowns and history load, and the next review shows no previous strength: Task 8 test.
3. A book of 10 strong stocks (sum 1.40) is scaled to exactly 98%, 9.8% each, and a book of 2 standard stocks invests 18% and parks the rest: Task 7 tests (sizing and pipeline).
4. An old `reviews.jsonl` (no strengths) through `bull-bear-scores` prints empty strength blocks and exits 0: Task 9 test.
5. A version-3 state file whose entry has an unknown or missing strength, or a weak `in`, is an invalid state (empty, with the usual warning), never a crash: Task 8 test.

---

### Task 1: The judge's verdict carries a validated strength

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py` (`Verdict`, `Verdicts`, `validate_verdicts`, `submit_verdicts` docstring)
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py:327-329` (record: `strength_counts`)
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`, `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Produces: `STRENGTHS = ("strong", "standard", "weak")` in `handoff.py`; `Verdict.strength: str | None = None` (last field, after `what_changed`); `Verdicts.strengths -> dict[str, str]` (symbol -> strength of every `in`, submission order). `asdict(verdict)` in `reviews.jsonl` therefore carries `strength` (null on `out`).

- [ ] **Step 1: Create the branch**

Run: `git switch -c feat/bull-bear-conviction-sizing`
Expected: `Switched to a new branch`; `git status` shows only ` M TODO.md`.

- [ ] **Step 2: Write the failing handoff tests** (in `test_bull_bear_handoff.py`, using the file's existing helpers)

```python
def test_every_in_carries_a_strength_and_every_out_none() -> None:
    v = validate_verdicts([{"symbol": "AAA", "verdict": "in", "reason": "r", "strength": " Strong "},
                           {"symbol": "BBB", "verdict": "out", "reason": "r"}], expected=["AAA", "BBB"], max_in=10, max_chars=300)
    assert [x.strength for x in v.verdicts] == ["strong", None]
    assert v.strengths == {"AAA": "strong"}

@pytest.mark.parametrize(("item", "message"), [
    ({"symbol": "AAA", "verdict": "in", "reason": "r"}, "the strength for AAA must be one of strong, standard, weak"),
    ({"symbol": "AAA", "verdict": "in", "reason": "r", "strength": "high"}, "the strength for AAA must be one of strong, standard, weak"),
    ({"symbol": "AAA", "verdict": "out", "reason": "r", "strength": "weak"}, "AAA is out: give a strength only to a stock you put in"),
])
def test_a_missing_unknown_or_misplaced_strength_is_refused(item: dict, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        validate_verdicts([item], expected=["AAA"], max_in=10, max_chars=300)

def test_a_changed_strength_needs_no_what_changed() -> None:
    v = validate_verdicts([{"symbol": "AAA", "verdict": "in", "reason": "r", "strength": "weak"}], expected=["AAA"], max_in=10, max_chars=300, previous={"AAA": "in"})
    assert v.verdicts[0].what_changed is None
```

Also update the existing verdict tests in this file (`test_a_verdict_for_every_stock_is_recorded_with_the_stocks_put_in`, `test_verdicts_and_symbols_are_normalised_like_the_debaters_cases`, `test_a_changed_verdict_*`, `test_the_first_valid_submission_is_final`, ...) so every `in` item carries `"strength": "standard"`.

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -q`
Expected: FAIL (`unexpected keyword` / `no attribute strengths` / missing refusal).

- [ ] **Step 4: Implement** in `handoff.py`: `validate_verdicts` reads `strength` with `_one_of(item, "strength", symbol, STRENGTHS)` for an `in`; for an `out` with a non-empty `strength` it raises `HandoffError(f"{symbol} is out: give a strength only to a stock you put in")`. `submit_verdicts`' docstring becomes: "Submit one verdict per stock: objects with symbol, verdict (in or out), one short reason, strength (strong, standard or weak) for every stock you put in, and what_changed when the verdict differs from your previous_verdict."

- [ ] **Step 5: Keep the pipeline green and log the counts**

In `test_bull_bear_pipeline.py`, the `verdicts()` script helper becomes `verdicts(chosen: Collection[str] | None = None, strengths: Mapping[str, str] | None = None) -> Script`: each `in` item gets `"strength": (strengths or {}).get(symbol, "standard")`. Every hand-written `submit_verdicts` call in the file gets a strength on its `in` items. In `test_the_review_log_records_the_whole_review_as_one_line`, the first verdict now equals `{..., "what_changed": None, "strength": "standard", "previous_verdict": None}` and add `assert line["strength_counts"] == {"standard": 10}`.
In `pipeline.py`, `record.update(...)` gains `strength_counts=dict(Counter(verdict.strength for verdict in verdicts.verdicts if verdict.strength))`.

- [ ] **Step 6: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/handoff.py src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/bull_bear/test_bull_bear_handoff.py tests/strategies/bull_bear/test_bull_bear_pipeline.py
git commit -m "feat(bull_bear): the judge gives every stock it puts in a strength (Task 1)"
```

---

### Task 2: The judge's prompt explains the strengths as a balance

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`JUDGE_SYSTEM`)
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:**
- Consumes: `STRENGTHS` names from Task 1 (the prompt spells them out; do not import).

- [ ] **Step 1: Write the failing tests**

```python
def test_the_judge_rates_each_in_by_the_balance_of_both_cases() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())
    for word in ("strong", "standard", "weak", "strength", "clearly outweighs", "every stock carries some risk"):
        assert word in text
    assert "bull" in text and "bear" in text
    assert "no material risk" not in text and "risk-free" in text  # "a strong case does not need a risk-free stock"

def test_the_judge_is_told_strength_sets_the_size_but_not_the_numbers() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())
    assert "largest position" in text and "stays in cash" in text
    assert not any(n in text for n in ("14", "0.14", "9%", "0.09", "4%", "0.04"))

def test_the_judge_submits_a_strength_for_every_stock_it_puts_in() -> None:
    assert "strength (strong, standard or weak) for every stock you put in" in " ".join(prompts.JUDGE_SYSTEM.split())
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -q`
Expected: the three new tests FAIL.

- [ ] **Step 3: Implement** — add to `JUDGE_SYSTEM`, after the `max_in` sentences, a passage carrying these points (wording yours, phrases in the tests verbatim): for every stock put in, give a strength, weighing the bull's conviction against the bear's risk; `strong` when the bull's case clearly outweighs the bear's, `standard` when it outweighs it, `weak` when the stock only narrowly earns a place; every stock carries some risk, so a strong case does not need a risk-free stock; code sizes each position by its strength alone: a strong case gets the largest position, a weak one a small position, and money the weak cases do not use stays in cash. The closing submit sentence names `strength (strong, standard or weak) for every stock you put in`. Keep every existing sentence (the existing prompt tests must still pass) and never write `held`.

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/prompts.py tests/strategies/bull_bear/test_bull_bear_prompts.py
git commit -m "feat(bull_bear): the judge's prompt explains the three strengths as a balance (Task 2)"
```

---

### Task 3: Replay gate (no product code; STOP for the user's go)

**Files (session scratchpad, never committed):**
- Create: `<scratchpad>/strength_replay.py` (runs the judge), `<scratchpad>/strength_score.py` (scores it)
- Reads: `logs/bull_bear/backtesting/2026-10-26_232151_backtesting/backtest.log` and `.../2026-10-27_195100_backtesting/backtest.log` (+ their `reviews.jsonl`), `<scratchpad>/exit_opens.pkl` (Yahoo opens/closes cache), `<scratchpad>/bull_build.py` (`extract(log) -> {date: judge context}`), `<scratchpad>/jr_replay.py` (how to build the judge with `AgentManager`, memory tools and the recorder).

`<scratchpad>` is `/tmp/claude-1000/-mnt-data-projets-trading-agent-framework/3082b479-e5eb-4357-ab00-2e55affcb022/scratchpad`.

- [ ] **Step 1: Check the server** — `curl -s localhost:8010/v1/models` lists `qwen3.8-27b-awq`. If not, stop and ask the user.

- [ ] **Step 2: Write `strength_replay.py`** — for each run (`38-1026`, `38-NEW`) and each judge context from `extract()`: build the judge exactly as `jr_replay.py` does (credentials `http://localhost:8010/v1`, model `qwen3.8-27b-awq`, `max_input_tokens` 81920, effort `medium`; empty temporary `MemoryStore`; `JUDGE_SYSTEM` and `judge_task(params.reason_max_chars)` imported from the branch; `params.judge_max_output_tokens`, `params.judge_thinking_token_budget`, `params.agent_temperature`), arm `expect_verdicts` with the context's symbols, `max_in` and `previous` (each stock's logged `previous_verdict["verdict"]`), run with the context as logged. Wrap `submit_verdicts` to count rejected calls. Append one JSON line per (run, date) to `strength_replay.jsonl`: `run, date, secs, submitted, rejections, last_error, verdicts {symbol: verdict}, strengths {symbol: strength}`; skip (run, date) pairs already in the file (resumable).

- [ ] **Step 3: Smoke-run one date** — run the script limited to the first date of `38-NEW`; expect one line with `submitted: true` and a strength on every `in`.

- [ ] **Step 4: Full run in the background** (about 108 judge calls, roughly a night): `uv run --project /mnt/data/projets/trading-agent-framework python strength_replay.py` from the scratchpad, `run_in_background`. Wait for its completion notification; do not poll.

- [ ] **Step 5: Write and run `strength_score.py`** — forward 20-session excess per candidate exactly as `bull_build.py` computes it (open of the first session after the review to the open 20 sessions later, minus the mean of that review's candidates). Print, pooled over both runs and per run: mean excess and n of `strong`, `standard`, `weak` among the replayed `in` stocks; the mean of per-review strong - weak spreads (reviews missing either group skipped, counted) and strong - standard, each with a Newey-West t at 3 lags (`agent_scores.newey_west_t`); the pooled group-mean gap strong - weak; the share of each strength; the share of verdicts that differ from the logged ones; rejections and failed runs.

- [ ] **Step 6: Apply the bar and STOP** — pass when (pooled) strong > standard > weak in mean excess AND mean per-review strong - weak >= +3.0 pts. A judge that almost never says `strong` (under about 10% of `in`) is a wording failure whatever the returns. Report the figures to the user and wait for their go.
  - On a miss: Task 3b (one revision of the strength wording in `JUDGE_SYSTEM`, written from what the replay showed, with its test updated; commit `fix(bull_bear): revise the judge's strength wording after the replay (Task 3b)`), then rerun Steps 4-6 into `strength_replay_2.jsonl`, and say in the report that this pass is tuned on the same runs. A second miss ends the work: report and do not continue to Task 4.

---

> **Revision 1 (2026-10-29): Tasks 1-3 are DONE and stay as history.** Task 1 (`41b5a36`) and Task 2 (`5951789`, `ac8a345`) built a strength only on an `in` and a weak position; Task 3's replay (strong +1.61 / standard -0.01 / weak -4.94, per-review strong - weak +2.23 vs the +3.0 bar) missed its bar and the user chose to proceed with changes (spec "Revision 1"): the judge rates EVERY debated stock strong/standard/weak and puts in only strong/standard ones (weak is always out); two weights (14% / 9%); SHV parking untouched. Tasks 4-5 rework Tasks 1-2; Task 6 starts a non-gating second replay; Tasks 7-9 replace the old sizing/state/scorer tasks.

---

### Task 4: Every verdict carries a strength, and a weak stock cannot be put in

Reworks Task 1's handoff (committed at `41b5a36`).

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/handoff.py` (`Verdict`, `Verdicts`, `validate_verdicts`, `submit_verdicts` docstring)
- Modify (if needed): `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (`strength_counts`)
- Test: `tests/strategies/bull_bear/test_bull_bear_handoff.py`, `test_bull_bear_pipeline.py`, `test_bull_bear_backtest.py`

**Interfaces:**
- Produces: `STRENGTHS = ("strong", "standard", "weak")` (unchanged); `Verdict.strength` is set on EVERY verdict by `validate_verdicts` (one of `STRENGTHS`, normalised like `verdict`); `Verdicts.strengths -> dict[str, str]` is now symbol -> strength of EVERY verdict, in submission order; `Verdicts.chosen` unchanged. `asdict(verdict)` in `reviews.jsonl` carries `strength` on every verdict; `strength_counts` counts all verdicts.

- [ ] **Step 1: Update the failing handoff tests** (in `test_bull_bear_handoff.py`)

```python
def test_every_verdict_carries_a_strength() -> None:
    v = validate_verdicts([{"symbol": "AAA", "verdict": "in", "reason": "r", "strength": " Strong "},
                           {"symbol": "BBB", "verdict": "out", "reason": "r", "strength": "weak"}], expected=["AAA", "BBB"], max_in=10, max_chars=300)
    assert [x.strength for x in v.verdicts] == ["strong", "weak"]
    assert v.strengths == {"AAA": "strong", "BBB": "weak"}

@pytest.mark.parametrize(("item", "message"), [
    ({"symbol": "AAA", "verdict": "in", "reason": "r"}, "the strength for AAA must be one of strong, standard, weak"),
    ({"symbol": "AAA", "verdict": "out", "reason": "r"}, "the strength for AAA must be one of strong, standard, weak"),
    ({"symbol": "AAA", "verdict": "in", "reason": "r", "strength": "high"}, "the strength for AAA must be one of strong, standard, weak"),
    ({"symbol": "AAA", "verdict": "in", "reason": "r", "strength": "weak"}, "AAA is weak: a weak stock goes out"),
])
def test_a_missing_unknown_or_weak_in_strength_is_refused(item: dict, message: str) -> None:
    with pytest.raises(HandoffError, match=message):
        validate_verdicts([item], expected=["AAA"], max_in=10, max_chars=300)

def test_a_strong_or_standard_stock_may_still_be_out_and_a_changed_strength_needs_no_what_changed() -> None:
    v = validate_verdicts([{"symbol": "AAA", "verdict": "out", "reason": "r", "strength": "strong"},
                           {"symbol": "BBB", "verdict": "in", "reason": "r", "strength": "standard"}], expected=["AAA", "BBB"], max_in=10, max_chars=300, previous={"BBB": "in"})
    assert v.chosen == ("BBB",) and v.verdicts[1].what_changed is None
```
Adapt the file's other verdict tests: every item (`in` AND `out`) carries a strength, `"standard"` unless the test is about strengths. Replace Task 1's tests that required "AAA is out: give a strength only to a stock you put in".

- [ ] **Step 2: Run them to verify they fail** — `uv run pytest tests/strategies/bull_bear/test_bull_bear_handoff.py -q`; expected FAIL.

- [ ] **Step 3: Implement** in `handoff.py`: `validate_verdicts` reads `strength` with `_one_of(item, "strength", symbol, STRENGTHS)` for every item; an `in` whose strength is `weak` raises `HandoffError(f"{symbol} is weak: a weak stock goes out, put it out or rate it standard or strong")`; the "give a strength only to a stock you put in" check and its message go. `Verdicts.strengths` covers all verdicts. `submit_verdicts`' docstring becomes: "Submit one verdict per stock: objects with symbol, strength (strong, standard or weak), verdict (in or out; a weak stock is always out), one short reason, and what_changed when the verdict differs from your previous_verdict." (keep `# noqa: E501` if over the line limit as today).

- [ ] **Step 4: Keep the pipeline and backtest tests green** — in `test_bull_bear_pipeline.py` the `verdicts(chosen, strengths)` helper gives EVERY item (`out` too) a strength (default `"standard"`, and an `out` default `"weak"`); every hand-written `submit_verdicts` call and the `test_bull_bear_backtest.py` judge script give every item a strength, `out` items `"weak"`. `test_the_review_log_records_the_whole_review_as_one_line`: every verdict now carries `strength`; assert `line["strength_counts"] == {"standard": 10, "weak": <number of outs>}`. In `pipeline.py` `strength_counts` counts every verdict's strength (it already skips `None`; check it).

- [ ] **Step 5: Run the bull_bear tests** — `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`; expected all pass.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/handoff.py src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/bull_bear/test_bull_bear_handoff.py tests/strategies/bull_bear/test_bull_bear_pipeline.py tests/strategies/bull_bear/test_bull_bear_backtest.py
git commit -m "feat(bull_bear): the judge rates every stock and a weak stock is always out (Task 4)"
```

---

### Task 5: The judge's prompt rates the whole debate, then selects among strong and standard

Reworks Task 2's prompt (committed at `5951789`, `ac8a345`).

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`JUDGE_SYSTEM`)
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:**
- Consumes: the strengths and the weak-is-out rule from Task 4 (the prompt spells them out; do not import).

- [ ] **Step 1: Update the failing tests**
  - `test_the_judge_rates_each_in_by_the_balance_of_both_cases` becomes: the text contains "rate every stock", "strong when the bull's case clearly outweighs the bear's", "standard when it outweighs it", "every stock carries some risk" and "risk-free" (use the real wording you write; each phrase must fail without the new passage), and not "no material risk".
  - New: the prompt says the weak ones are left out: it contains "weak" together with a sentence that a weak stock goes out and that you then choose which stocks to put in only among the strong and standard ones (assert both phrases of your wording).
  - Keep the two Task-2-fix tests (a change of strength alone needs no new fact and no what_changed; the evidence rules hold for the strength as for the verdict) adapted to the new wording.
  - `test_the_judge_is_told_strength_sets_the_size_but_not_the_numbers`: assert "largest position" and no weight numbers (`14`, `0.14`, `9%`, `0.09`); drop the "stays in cash" assertion and assert the prompt does not mention "cash" or "SHV" in the strength passage.
  - `test_the_judge_submits_a_strength_for_every_stock_it_puts_in` becomes: the closing submit sentence contains "a strength (strong, standard or weak) for every stock".

- [ ] **Step 2: Run them to verify they fail** — `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -q`; expected the changed tests FAIL.

- [ ] **Step 3: Implement** — rewrite the Task 2 strength passage in `JUDGE_SYSTEM` (wording yours, the phrases in the tests verbatim; the user's rule: balance, never absolutes such as "no material risk"): first rate EVERY stock in stocks strong, standard or weak by weighing the bull's conviction against the bear's risk (strong when the bull's case clearly outweighs the bear's, standard when it outweighs it, weak when the stock only narrowly earns a place or does not); every stock carries some risk, so a strong case does not need a risk-free stock; only once every stock is rated, choose the stocks to put in, only among the strong and standard ones, and put every weak stock out; `max_in` stays a ceiling; the stronger the case the larger the position code gives it (strong gets the largest position); no numbers, nothing about cash or SHV, nothing about how many of each strength. Keep the two Task-2-fix sentences (strength changes need no `what_changed`; the evidence rules hold for the strength) and every other existing sentence; never write `held`. The closing submit sentence names the symbol, "a strength (strong, standard or weak) for every stock", the verdict (in or out; a weak stock is always out), one short reason, and what_changed when the verdict differs from the previous_verdict.

- [ ] **Step 4: Run the bull_bear tests** — `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`; expected all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/prompts.py tests/strategies/bull_bear/test_bull_bear_prompts.py
git commit -m "feat(bull_bear): the judge's prompt rates the whole debate then selects among strong and standard (Task 5)"
```

---

### Task 6: Second replay in the background (non-gating; scratchpad only, no commit)

**Files (session scratchpad, never committed):** modify copies `<scratchpad>/strength_replay.py` -> `strength_replay2.py` and `strength_score.py` -> `strength_score2.py` (the originals stay as the record of the first replay).

`<scratchpad>` is `/tmp/claude-1000/-mnt-data-projets-trading-agent-framework/3082b479-e5eb-4357-ab00-2e55affcb022/scratchpad`.

- [ ] **Step 1: Replay script** — `strength_replay2.py` is `strength_replay.py` writing `strength_replay2.jsonl`, recording `strengths` for EVERY verdict (the branch's `Verdicts.strengths` now covers all), plus `rejections` and `last_error` as before. It imports `JUDGE_SYSTEM`, `judge_task` and the handoff from the branch code at the moment it starts, so start it only after Task 5 is committed.
- [ ] **Step 2: Smoke-run one date** (first date of `38-NEW`, output `strength_smoke2.jsonl`): `submitted: true`, a strength on every stock, every weak stock out.
- [ ] **Step 3: Scorer** — `strength_score2.py <jsonl>` prints: the share of each strength among all debated stocks and among those put in; the mean 20-session excess (same definition as the first scorer) by strength over ALL debated stocks and over the stocks put in, pooled and per run, with n; per-review strong - weak (all debated stocks) and strong - standard (among those put in, and among all) with Newey-West t (3 lags) and the skipped-review count; the number of weak stocks the judge put in (must be 0); how many logged `in` stocks are now out; rejections and failed runs. Prove it runs on the smoke line.
- [ ] **Step 4: Start the full run** in the background (`run_in_background`, about 5 hours); it does not gate Tasks 7-9, which touch other files and do not affect a running process. The controller scores it when it ends and reports with the final report.

---

### Task 7: Strength weights size the book

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/parameters.py` (weights, validation, `weights_by_strength` property)
- Modify: `src/trading_agent_framework/strategies/bull_bear/sizing.py` (replace `capped_inverse_volatility`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (sizing step, `# 8.` block), `agent_bull_bear.py` (mirror the weights into `self.parameters`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the sizing sentence)
- Test: `tests/strategies/bull_bear/test_bull_bear_sizing.py` (rewrite), `test_bull_bear_params.py`, `test_bull_bear_pipeline.py`, `test_bull_bear_strategy.py`

**Interfaces:**
- Consumes: `Verdicts.strengths` (Task 4), now strength of every verdict (only the book's stocks are read; none is weak).
- Produces: `BullBearParams.strong_weight / standard_weight: float` and `BullBearParams.weights_by_strength -> dict[str, float]` (`{"strong": .., "standard": ..}`); `sizing.strength_weights(strengths: Mapping[str, str], *, weights: Mapping[str, float], total: float) -> dict[str, float]`. `min_weight`, `max_weight` and `capped_inverse_volatility` are gone. There is no weak weight.

- [ ] **Step 1: Write the failing sizing tests** (replace the file's contents)

```python
W = {"strong": 0.14, "standard": 0.09}

def test_each_stock_gets_its_strengths_weight_in_input_order() -> None:
    assert strength_weights({"B": "standard", "A": "strong"}, weights=W, total=0.98) == pytest.approx({"B": 0.09, "A": 0.14})
    assert list(strength_weights({"B": "standard", "A": "strong"}, weights=W, total=0.98)) == ["B", "A"]

def test_ten_strong_stocks_are_scaled_to_the_total_ratios_kept() -> None:
    w = strength_weights({f"S{i}": "strong" for i in range(10)}, weights=W, total=0.98)
    assert sum(w.values()) == pytest.approx(0.98) and all(x == pytest.approx(0.098) for x in w.values())

def test_seven_strong_three_standard_scale_by_one_factor() -> None:
    w = strength_weights({**{f"S{i}": "strong" for i in range(7)}, **{f"T{i}": "standard" for i in range(3)}}, weights=W, total=0.98)
    assert w["T0"] == pytest.approx(0.09 * 0.98 / 1.25) and w["S0"] == pytest.approx(0.14 * 0.98 / 1.25)

def test_a_book_under_the_total_is_not_scaled_up() -> None:
    assert sum(strength_weights({"A": "standard", "B": "standard"}, weights=W, total=0.98).values()) == pytest.approx(0.18)

def test_no_stock_is_an_empty_book_and_an_unknown_or_weak_strength_is_an_error() -> None:
    assert strength_weights({}, weights=W, total=0.98) == {}
    for bad in ("high", "weak"):
        with pytest.raises(ValueError, match=bad):
            strength_weights({"A": bad}, weights=W, total=0.98)
```

- [ ] **Step 2: Write the failing params tests** — `test_the_defaults_are_the_specs_and_valid` asserts `(params.strong_weight, params.standard_weight, params.cash_buffer, params.rebalance_band) == (0.14, 0.09, 0.02, 0.03)` and `params.weights_by_strength == {"strong": 0.14, "standard": 0.09}`; `test_an_invalid_value_is_refused_with_a_reason` drops its `min_weight`/`max_book_size x min_weight` cases and gains:
  - `({"standard_weight": 0.14}, "0 < standard_weight < strong_weight <= 1")`
  - `({"strong_weight": 1.2}, "0 < standard_weight < strong_weight <= 1")`
  - `({"standard_weight": 0.035}, "a standard stock in a full book of strong ones must stay above rebalance_band")` (0.035 x 0.98 / 1.295 = 2.65%)
  - `({"strong_weight": math.nan}, "must be finite")`
  and a test that `min_weight`, `max_weight` and `weak_weight` are not fields (`dataclasses.fields`).

- [ ] **Step 3: Write the failing pipeline tests** (in `test_bull_bear_pipeline.py`; the judge scripts rate every stock, outs `weak`)
  - `test_fewer_than_five_stocks_put_in_leave_the_rest_parked` becomes `test_strength_weights_leave_the_rest_parked`: judge puts S03 `strong`, S14 `standard` in -> `line["targets"] == approx({"S03": 0.14, "S14": 0.09, "SHV": 0.75})`.
  - `test_the_review_log_records_the_whole_review_as_one_line`: 10 `standard` -> every stock 0.09, sum 0.90, `SHV` 0.08.
  - `test_a_held_stock_whose_strength_falls_is_sold_down`: held S00 = 28 shares (1,400 = 14% at 50), judge keeps S00 `in` as `standard` -> an S00 sell order.
  - `test_a_held_stock_rated_weak_is_put_out_and_sold`: held S00, judge rates it `weak` and `out` (what_changed given) -> S00 sold, in cooldown.
  - `test_a_scaled_strength_change_inside_the_band_is_not_traded`: held S00..S09 at 19.6 shares each (9.8%), judge keeps all ten in, S00..S06 `strong`, S07..S09 `standard` (targets 7.06%, a 2.74-pt move) -> no order for S07, S08, S09.
  - Fix the other weight assertions that assumed inverse volatility (`test_max_volatility_filters_new_entries...`: S03 now has its strength's weight; `test_a_broker_error_during_the_rebalance...`: the stock targets sum to 0.90).

- [ ] **Step 4: Run them to verify they fail** — `uv run pytest tests/strategies/bull_bear/test_bull_bear_sizing.py tests/strategies/bull_bear/test_bull_bear_params.py tests/strategies/bull_bear/test_bull_bear_pipeline.py -q`; expected the new and changed tests FAIL.

- [ ] **Step 5: Implement**
  - `sizing.py`: `strength_weights` — each weight is `weights[strength]` (`ValueError` naming an unknown or weightless strength); when the sum exceeds `total`, every weight is multiplied by `total / sum`; keys keep input order. Module docstring rewritten to say why (the judge's strength alone sizes; the rest is parked by `target_portfolio` as before).
  - `parameters.py`: the two fields replace `min_weight`/`max_weight` (comment: ten standard make 90%); the floats tuple includes them; the two removed checks are replaced by `"0 < standard_weight < strong_weight <= 1"` and `"a standard stock in a full book of strong ones must stay above rebalance_band (it would never be bought)"`, failing when `min(standard, standard * investable / ((max_book_size - 1) * strong + standard)) <= rebalance_band`; property `weights_by_strength`. `max_book_size`'s comment drops "below 5 the rest of the money is parked".
  - `pipeline.py` step 8: `target_weights = strength_weights({symbol: verdicts.strengths[symbol] for symbol in book}, weights=params.weights_by_strength, total=params.investable)`; the comment says strength alone, the rest parked. The volatility lookup goes.
  - `agent_bull_bear.py`: `self.parameters` gains `"strong_weight"` and `"standard_weight"`; add the matching assertion to the existing `settings.json`/parameters test in `test_bull_bear_strategy.py`.
  - `CLAUDE.md`: replace "sizes it by inverse volatility within [4%, 20%] summing to `min(98%, n x 20%)` (`sizing.capped_inverse_volatility`; below 5 stocks the rest is parked in SHV, zero `in` parks everything but the cash buffer)" with the strength sizing: weights 14/9% by the judge's strength, scaled down by one factor above 98%, the rest parked in SHV as before; resizing on a strength change is the rebalancer's band, best effort (a scaled book can miss a change under 3 pts, at no cost in exposure).

- [ ] **Step 6: Run the bull_bear tests** — `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`; expected all pass.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/{parameters,sizing,pipeline,agent_bull_bear}.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_{sizing,params,pipeline,strategy}.py
git commit -m "feat(bull_bear): size each position by the judge's strength alone (Task 7)"
```

---

### Task 8: The state remembers each strength; the judge sees it; the log line names it

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/state.py` (`STATE_VERSION` 3, validation)
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (judge log line, `decided`, where the judge's `previous_verdict` is built)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (state and judge sentences)
- Test: `tests/strategies/bull_bear/test_bull_bear_state.py`, `test_bull_bear_pipeline.py`

**Interfaces:**
- Consumes: `STRENGTHS`, `Verdict.strength` on every verdict (Task 4).
- Produces: `state.verdicts[symbol]` = `{"verdict", "reason", "date", "strength"}` for EVERY verdict (an `in` is never weak); the judge's `previous_verdict` therefore shows the previous strength of every stock judged before.

- [ ] **Step 1: Write the failing tests**
  - state: `test_a_version_3_state_with_strengths_loads_back` (an `in` entry `"strength": "standard"` and an `out` entry `"strength": "weak"` round-trip); `test_a_version_2_state_loads_without_strengths` (a v2 file with verdicts, cooldowns and history loads them all unchanged, no warning); `test_a_missing_unknown_or_weak_in_strength_is_an_invalid_state` (parametrized on a v3 file: an entry without `strength`, `"strength": "high"`, and `{"verdict": "in", ..., "strength": "weak"}` -> empty state with the "not a valid state" warning); the saved file has `"version": 3`.
  - pipeline: `test_the_judge_sees_its_previous_strength` (state `verdicts={"S00": {"verdict": "in", "reason": "x", "date": "2026-09-29", "strength": "strong"}, "S01": {"verdict": "out", ..., "strength": "weak"}}` -> the judge's context for S00 and S01 has `previous_verdict` carrying those strengths; after the review the store holds each stock's new strength); `test_a_strength_change_alone_needs_no_what_changed` (S00 previously `in` strong, now `in` standard -> the review completes on the first judge run); `test_the_judge_log_line_names_each_strength_put_in` (`"[judge] in: S00 (standard), ..."` in caplog); `test_an_execution_abandonment_stores_the_strengths` (extend the existing execution-abandonment test: the stored verdicts carry their strength).

- [ ] **Step 2: Run them to verify they fail** — `uv run pytest tests/strategies/bull_bear/test_bull_bear_state.py tests/strategies/bull_bear/test_bull_bear_pipeline.py -q`; expected the new tests FAIL.

- [ ] **Step 3: Implement**
  - `state.py`: `STATE_VERSION = 3`; `_valid` accepts version 2 (entries exactly `{verdict, reason, date}`) and version 3 (the same plus a required `strength` in `STRENGTHS`, and never `weak` on an `in`); everything else unchanged (version 1 path kept). Module docstring: add the strength and the v2 load.
  - `pipeline.py`: `decided` adds `"strength": verdict.strength` to every verdict; the judge's context shows the previous strength next to the previous verdict when the state has one; the judge line becomes `[judge] in: S00 (strong), S03 (standard); out: ...`.
  - `CLAUDE.md`: state `STATE_VERSION` 3 (a version-2 file loads without strengths); the judge rates every stock of the debate strong / standard / weak (the bull's conviction against the bear's risk), puts in only strong and standard ones, sees its previous strength, and a strength change needs no `what_changed`; `reviews.jsonl` gains `strength` on every verdict and `strength_counts`.

- [ ] **Step 4: Run the bull_bear tests** — `uv run pytest tests/strategies/bull_bear/ -q && uv run ruff check`; expected all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/{state,pipeline}.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_{state,pipeline}.py
git commit -m "feat(bull_bear): remember each stock's strength and show it to the judge (Task 8)"
```

---

### Task 9: `bull-bear-scores` scores the judge's strengths

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/batch/agent_scores.py` (`STRENGTHS`, `strength_calls`, two `_AGENTS` entries)
- Modify: `src/trading_agent_framework/strategies/bull_bear/batch/scores_cli.py` (`_TITLES`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the `batch/` sentence)
- Test: `tests/strategies/bull_bear/test_bull_bear_agent_scores.py`, `test_bull_bear_scores_cli.py`

**Interfaces:**
- Produces: `strength_calls(review) -> dict[str, str]` (symbol -> strength of EVERY verdict with a known strength, `in` or `out`); `score_agents` returns, after bull, bear, judge, `AgentScore("strength", "strong - weak", ...)` and `AgentScore("strength_standard", "strong - standard", ...)`, both over groups `("strong", "standard", "weak")`.

- [ ] **Step 1: Write the failing tests**
  - `test_the_judges_strengths_are_scored_strong_against_weak_and_standard`: two reviews where `strong` stocks beat `weak` ones by a known excess -> the `strength` spread mean equals it; groups count every rated stock (`in` and `out`); `strength_standard` spread likewise.
  - `test_a_run_without_strengths_gives_empty_strength_scores`: an old-format review (verdicts without `strength`) -> both strength agents have `n == 0` groups and a spread with `mean is None`.
  - CLI: `test_the_report_prints_the_strength_blocks` (titles `JUDGE strength, strong - weak` and `JUDGE strength, strong - standard` in the output) and an old-format run exits 0 and prints `- (0 rev)` in those blocks.

- [ ] **Step 2: Run them to verify they fail** — `uv run pytest tests/strategies/bull_bear/test_bull_bear_agent_scores.py tests/strategies/bull_bear/test_bull_bear_scores_cli.py -q`; expected FAIL.

- [ ] **Step 3: Implement** the two `_AGENTS` entries (`("strength", STRENGTHS, ("strong", "weak"), strength_calls)`, `("strength_standard", STRENGTHS, ("strong", "standard"), strength_calls)`) and the two `_TITLES`; update docstrings that say "bull, bear, judge" and the `CLAUDE.md` sentence on `score_agents`.

- [ ] **Step 4: Run the whole suite and lint** — `uv run pytest -q && uv run ruff check`; expected all pass, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/batch/{agent_scores,scores_cli}.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_{agent_scores,scores_cli}.py
git commit -m "feat(bull_bear): score the judge's strengths against forward returns (Task 9)"
```

After Task 9: score the second replay when it ends and report to the user together (backtest count and window are theirs to decide); no merge without their go.

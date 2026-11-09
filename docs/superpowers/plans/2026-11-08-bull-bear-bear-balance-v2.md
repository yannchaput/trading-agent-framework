# bull_bear bear balance v2 replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-test v1's committed bear/judge change on 230 fresh reviews of the 5Y run with six outcome-based keep/drop tests, and write the KEEP/DROP report.

**Architecture:** One throwaway script, `scripts/replays/bull_bear_bear_balance_v2.py`, that loads v1's script (`scripts/replays/bull_bear_bear_balance.py`) by file path and reuses its log parsing, fields, `replay_review` and `_Replay`. v2 adds a judge-only replay (arms J-old, J-new), its own samples, its own six tests and its own data files. Pure parts are unit-tested in a test file next to the script; the replay itself runs once, after a pre-registration commit.

**Tech Stack:** Python 3.14, uv, pytest, the project's `AgentManager` against the local Qwen 3.8 27B AWQ server (as v1).

**Spec:** `docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-v2-design.md` (v1 spec and report it builds on: `docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-design.md`, `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance.md`).

## Global Constraints

- No change under `src/` or `tests/`. v1's files (`scripts/replays/bull_bear_bear_balance.py`, `docs/superpowers/replays/data/replay_*.json*`) are read, never modified.
- The new prompts must equal those at commit `407a310` byte for byte (`BULL_SYSTEM`, `BEAR_SYSTEM`, `JUDGE_SYSTEM`, `bull_task`, `bear_task`, `judge_task`); arm A / J-old use v1's `OLD_PROMPTS_REV` (`d852eac~1`).
- Run log: `logs/bull_bear/backtesting/2026-11-05_012455_backtesting` in the MAIN checkout (v1's `RUN_DIR`), read-only.
- Fresh pool: 262 reviews minus v1's 24 sample reviews and 8 anti-goal reviews (both read from v1's committed `docs/superpowers/replays/data/replay_plan.json`) = 230.
- Full sample: 36 reviews, `random.Random(20261109).sample(sorted(pool), 36)`.
- Judge sample: pool reviews with at least one logged stock with bull `medium`/`high` and bear `high`, then 60 of them, `random.Random(20261110).sample(sorted(eligible), 60)`.
- Tests and thresholds (spec table): 1 high share 4-15% and low share B <= A + 15 points; 2 J-new out rate < 75% and >= 15 points below J-old, min 30 stocks; 3 J-new out rate on `runway_months` < 12 >= 80%, min 10; 4 mean 20d excess of B high < B medium, min 20 high; 5 B rates high >= half of `runway_months` < 12, min 10; 6 v1's flip test (empty group = 0). Below a minimum: not testable, does not pass.
- Outcomes: 20-session excess from `strategies.bull_bear.batch.agent_scores.forward_returns` with opens from `batch.scores_cli.fetch_opens`, never compounded.
- No `# noqa`, no per-file ignore, no config change. ruff (line length 200, `D` rules apply outside `tests/`) must pass on both new files, so test functions carry docstrings.
- The automated suite stays network-free: the new test file never calls the model, SEC or Yahoo.
- Before each commit, run `/ponytail:ponytail-review` on the staged changes (user rule); fix its "Must fix" items, list its "Should fix" ones. If the skill is not installed in the session, say so in the report.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (or the implementer's model).

## Review Focus

1. A review that failed in one arm of a pair (A/B, or J-old/J-new) must be excluded from both arms of that pair, and counted in the report. Test: `test_valid_pairs_drop_a_review_failed_in_either_arm` (Task 1).
2. A stock with no `runway_months` (absent) is not in the runway groups of tests 3 and 5. Test: `test_runway_group_skips_absent_runway` (Task 1).
3. A float boundary (0.85 - 0.70 = 0.15) must pass test 2's "at least 15 points" (v1 hit the same with 0.65 - 0.5). Test: `test_veto_passes_at_exactly_15_points` (Task 1).
4. A stock without a 20d excess (no open) is left out of test 4's means and the flip means, not counted as 0. Test: `test_high_means_risk_ignores_missing_excess` (Task 1).
5. A results file with a v2 line for a (review, arm) already there is skipped on resume, and a J arm line is never confused with a full line. Test: `test_done_keys_separate_full_and_judge_lines` (Task 1).

---

### Task 1: The pure parts and their tests

**Files:**
- Create: `scripts/replays/bull_bear_bear_balance_v2.py` (pure section only in this task)
- Test: `scripts/replays/test_bull_bear_bear_balance_v2.py` (run by path; `testpaths` stays `tests`)

**Interfaces:**
- Consumes: v1 module, loaded by `importlib.util.spec_from_file_location("bull_bear_bear_balance", Path(__file__).with_name("bull_bear_bear_balance.py"))` into a module-level `v1` (no `sys.path` edit, so no E402). Uses `v1.with_new_fields(sheet, fields) -> dict`, `v1.share(part, whole) -> float | None`.
- Produces (all in `bull_bear_bear_balance_v2.py`):
  - constants `FULL_SEED = 20261109`, `FULL_SIZE = 36`, `JUDGE_SEED = 20261110`, `JUDGE_SIZE = 60`, `POOL_EXPECTED = 230`, `FULL_ARMS = ("A", "B")`, `JUDGE_ARMS = ("J-old", "J-new")`, `MIN_VETO = 30`, `MIN_RUNWAY = 10`, `MIN_HIGH = 20`, `NEW_PROMPTS_REV = "407a310"`.
  - `fresh_pool(days: Sequence[str], v1_plan: Mapping[str, Any]) -> list[str]`: sorted days minus `v1_plan["sample"][*]["date"]` minus `v1_plan["anti_goal"]["extra_reviews"]`.
  - `full_sample(pool: Sequence[str]) -> list[str]`: `sorted(random.Random(FULL_SEED).sample(sorted(pool), FULL_SIZE))`; `ValueError` if the pool has fewer than 36.
  - `judge_eligible(record: Mapping[str, Any]) -> bool`: some symbol with `record["bull"]` conviction in (`medium`, `high`) and `record["bear"]` risk `high`.
  - `judge_sample(pool: Sequence[str], records: Mapping[str, Mapping[str, Any]]) -> list[str]`: `sorted(random.Random(JUDGE_SEED).sample(sorted(eligible pool days), JUDGE_SIZE))`; `ValueError` if fewer than 60 are eligible.
  - `judge_context(arm: str, day: str, judge_ctx: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]`: J-old returns a copy equal to `judge_ctx`; J-new replaces each stock's `fact_sheet` with `v1.with_new_fields(sheet, fields[day][symbol])` and leaves every other key (`note`, `bull`, `bear`, `previous_verdict`, `constraints`, `current_datetime`) as logged.
  - `valid_pairs(lines: Sequence[Mapping[str, Any]], days: Sequence[str], arms: tuple[str, str], kind: str) -> tuple[list[str], dict[str, list[str]]]`: (days where both arms have an `ok` line of that kind, {arm: days failed}).
  - `done_key(line: Mapping[str, Any]) -> tuple[str, str, str]`: `(date, arm, kind)`, kinds `"full"` and `"judge"`.
  - `runway_symbols(sheets: Sequence[Mapping[str, Any]]) -> set[str]`: symbols whose `runway_months` is present and < 12.
  - `keep_drop_tests_v2(*, bear: Mapping[str, Sequence[str]], veto: Mapping[str, tuple[int, int]], runway_out_new: tuple[int, int], excess_by_level_b: Mapping[str, Sequence[float]], runway_high_b: tuple[int, int], flips: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]`: six dicts with keys `test` (1-6), `name`, their numbers, `testable` (where a minimum applies), `pass`, `note` (why not testable, else None). `veto` maps `"J-old"`/`"J-new"` to (outs, stocks); `excess_by_level_b` maps `"high"`/`"medium"` to the 20d excesses present; `runway_high_b` is (rated high, stocks); `flips` uses v1's keys `"B_in_A_out"`, `"A_in_B_out"`.
  - `decision(tests: Sequence[Mapping[str, Any]], complete: bool) -> str`: `"NONE: ..."` when incomplete, `"KEEP: all six tests pass"`, else `"DROP: test N (name) fails; ..."` (v1's wording).
  - Test names: 1 `the bear still argues`, 2 `no veto`, 3 `the judge still removes real risks`, 4 `high means risk`, 5 `real risks still flagged`, 6 `no harm in the flips`.

- [ ] **Step 1: Write the failing tests** in `scripts/replays/test_bull_bear_bear_balance_v2.py`, importing the script with `importlib` by path the same way (module docstring + one-line docstring per test). Cases, exact values:

```python
def test_fresh_pool_removes_v1_reviews():
    days = ["d1", "d2", "d3", "d4"]
    plan = {"sample": [{"index": 0, "date": "d1"}], "anti_goal": {"extra_reviews": ["d3"]}}
    assert v2.fresh_pool(days, plan) == ["d2", "d4"]


def test_full_sample_is_the_fixed_seed_draw():
    pool = [f"2022-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)] + [f"2023-{m:02d}-01" for m in range(1, 13)]  # 36
    pool += ["2024-01-01", "2024-02-01"]  # 38
    assert v2.full_sample(pool) == sorted(random.Random(20261109).sample(sorted(pool), 36))
    with pytest.raises(ValueError):
        v2.full_sample(pool[:35])


def test_judge_sample_keeps_only_eligible_reviews():
    # 61 eligible (bull medium + bear high on X), 5 not (bear medium)
    records = {f"e{i:02d}": {"bull": [{"symbol": "X", "conviction": "medium"}], "bear": [{"symbol": "X", "risk": "high"}]} for i in range(61)}
    records |= {f"n{i}": {"bull": [{"symbol": "X", "conviction": "high"}], "bear": [{"symbol": "X", "risk": "medium"}]} for i in range(5)}
    chosen = v2.judge_sample(sorted(records), records)
    assert len(chosen) == 60 and all(day.startswith("e") for day in chosen)
    assert chosen == sorted(random.Random(20261110).sample(sorted(d for d in records if d.startswith("e")), 60))
    with pytest.raises(ValueError):
        v2.judge_sample([f"e{i:02d}" for i in range(59)], records)


def test_judge_eligible_needs_bull_medium_or_high_and_bear_high_on_the_same_stock():
    assert not v2.judge_eligible({"bull": [{"symbol": "X", "conviction": "low"}, {"symbol": "Y", "conviction": "high"}], "bear": [{"symbol": "X", "risk": "high"}, {"symbol": "Y", "risk": "low"}]})


def test_judge_context_old_is_logged_and_new_adds_fields_only():
    ctx = {
        "current_datetime": "2024-01-02T09:00:00-05:00",
        "constraints": {"max_in": 10},
        "stocks": [
            {
                "fact_sheet": {"symbol": "X", "operating_margin_pct": 5.0, "operating_margin_prior_year_pct": 20.0},
                "note": "n",
                "bull": {"conviction": "high", "argument": "a"},
                "bear": {"risk": "high", "concern": "fundamentals", "argument": "b"},
                "previous_verdict": {"verdict": "in"},
            }
        ],
    }
    fields = {"2024-01-02": {"X": {"operating_margin_change_pts": -15.0, "days_since_filed": 10, "return_since_filed_pct": -3.0}}}
    assert v2.judge_context("J-old", "2024-01-02", ctx, fields) == ctx
    new = v2.judge_context("J-new", "2024-01-02", ctx, fields)
    sheet = new["stocks"][0]["fact_sheet"]
    assert list(sheet) == ["symbol", "operating_margin_pct", "operating_margin_prior_year_pct", "operating_margin_change_pts", "days_since_filed", "return_since_filed_pct"]
    assert {k: v for k, v in new["stocks"][0].items() if k != "fact_sheet"} == {k: v for k, v in ctx["stocks"][0].items() if k != "fact_sheet"}
    assert ctx["stocks"][0]["fact_sheet"].get("days_since_filed") is None  # the logged context is not mutated


def test_valid_pairs_drop_a_review_failed_in_either_arm():
    lines = [
        {"date": "d1", "arm": "J-old", "kind": "judge", "ok": True},
        {"date": "d1", "arm": "J-new", "kind": "judge", "ok": False},
        {"date": "d2", "arm": "J-old", "kind": "judge", "ok": True},
        {"date": "d2", "arm": "J-new", "kind": "judge", "ok": True},
        {"date": "d2", "arm": "A", "kind": "full", "ok": True},
    ]
    assert v2.valid_pairs(lines, ["d1", "d2", "d3"], ("J-old", "J-new"), "judge") == (["d2"], {"J-old": [], "J-new": ["d1"]})


def test_done_keys_separate_full_and_judge_lines():
    assert v2.done_key({"date": "d1", "arm": "A", "kind": "full"}) != v2.done_key({"date": "d1", "arm": "J-old", "kind": "judge"})


def test_runway_group_skips_absent_runway():
    assert v2.runway_symbols([{"symbol": "A", "runway_months": 11.9}, {"symbol": "B", "runway_months": 12}, {"symbol": "C"}]) == {"A"}
```

Plus `keep_drop_tests_v2` cases, built from one helper `base_inputs()` that passes every test (bear B: 5 high, 50 medium, 45 low; A: 10 high, 45 medium, 45 low; veto J-old (27, 30), J-new (21, 30); runway_out_new (8, 10); excess high 20 x -0.05, medium 20 x -0.01; runway_high_b (5, 10); flips B_in_A_out [0.02], A_in_B_out [0.01]) and one change per case:

| test function | change to `base_inputs()` | expect |
|---|---|---|
| `test_all_six_pass_on_base_inputs` | none | every `pass` True; `decision(tests, True) == "KEEP: all six tests pass"` |
| `test_bear_still_argues_fails_below_4_percent` | B: 3 high, 52 medium, 45 low | test 1 False |
| `test_veto_fails_at_75_percent` | J-new (23, 30) (76.7%) | test 2 False |
| `test_veto_needs_15_points_below_old` | J-old (25, 30), J-new (22, 30) (10 points) | test 2 False |
| `test_veto_passes_at_exactly_15_points` | J-old (34, 40), J-new (28, 40) (85% vs 70%) | test 2 True |
| `test_veto_not_testable_under_30_stocks` | J-old (26, 29), J-new (10, 29) | test 2 `testable` False, `pass` False, `note` names 30 |
| `test_runway_out_fails_below_80_percent` | runway_out_new (7, 10) | test 3 False |
| `test_runway_out_not_testable_under_10` | runway_out_new (9, 9) | test 3 not testable, False |
| `test_high_means_risk_fails_when_not_worse` | excess high 20 x -0.01 (equal to medium) | test 4 False |
| `test_high_means_risk_not_testable_under_20` | excess high 19 x -0.05 | test 4 not testable, False |
| `test_high_means_risk_ignores_missing_excess` | excess high: 20 x -0.05 (the caller passes only present values; assert `n_high == 20` in the result) | test 4 True, `n_high` 20 |
| `test_real_risks_flagged_fails_below_half` | runway_high_b (4, 10) | test 5 False |
| `test_real_risks_flagged_not_testable_under_10` | runway_high_b (9, 9) | test 5 not testable, False |
| `test_flips_empty_group_counts_as_zero` | flips B_in_A_out [], A_in_B_out [-0.01] | test 6 True (0 >= -0.01) |
| `test_decision_lists_each_failure` | runway_high_b (4, 10) and J-new (23, 30) | `decision(...) == "DROP: test 2 (no veto) fails; test 5 (real risks still flagged) fails"` |
| `test_decision_none_when_incomplete` | none, `complete=False` | starts with `"NONE"` |

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest scripts/replays/test_bull_bear_bear_balance_v2.py -q`
Expected: FAIL (module file missing or attributes missing).

- [ ] **Step 3: Implement the pure section** of `scripts/replays/bull_bear_bear_balance_v2.py`: module docstring (what v2 is, the spec path, the subcommands Task 2 adds), the `v1` loader, the constants and functions above. Test 1 reproduces v1's test 1 logic (`round(low_b - low_a, 12) <= 0.15`); test 2 uses `round(old - new, 12) >= 0.15`; percent thresholds compare shares from `v1.share`.

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest scripts/replays/test_bull_bear_bear_balance_v2.py -q && uv run ruff check scripts/replays/ && uv run ruff format --check scripts/replays/`
Expected: all pass, ruff clean. Also `uv run pytest -q` (the main suite, unchanged) still passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/replays/bull_bear_bear_balance_v2.py scripts/replays/test_bull_bear_bear_balance_v2.py
git commit -m "feat(bull_bear): v2 replay pure parts: fresh samples, judge-only context, six keep/drop tests"
```

### Task 2: Prepare, the two runners, score, and the pre-registration

**Files:**
- Modify: `scripts/replays/bull_bear_bear_balance_v2.py` (I/O section and CLI)
- Create (by `prepare`): `docs/superpowers/replays/data/replay_v2_plan.json`
- Results (by the runners): `docs/superpowers/replays/data/replay_v2_results.jsonl`; report (by `score`): `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance-v2.md`

**Interfaces:**
- Consumes: Task 1's functions; from v1: `parse_log()`, `read_reviews()`, `load_prompts(rev)`, `replay_review(arm, entry, log, fields, prompts)`, `_Replay`, `_server_up`, `_load_env`, `ServerDown`, `_calls`, `FIELDS_FILE` (read only; it holds the four fields for all 262 reviews), `PLAN_FILE` (v1's plan, read only), `OLD_PROMPTS_REV`, `STAGES`, `HORIZON`.
- Produces: subcommands `prepare`, `run [--arms A,B] [--reviews D,...]`, `judge [--arms J-old,J-new] [--reviews D,...]`, `score [--out PATH]`; result lines with `kind` `"full"` (from `v1.replay_review`, arm `A`/`B`) or `"judge"`.

- [ ] **Step 1: `prepare`** (no agent call): parse the log and records; check `load_prompts(None)` equals `load_prompts(NEW_PROMPTS_REV)` on the six names in Global Constraints (SystemExit otherwise); pool via `fresh_pool` with v1's `PLAN_FILE`, SystemExit unless `len(pool) == POOL_EXPECTED`; `full_sample`, `judge_sample`; write `replay_v2_plan.json` with `created`, the git SHA of HEAD, `pool_size`, `full_sample`, `judge_eligible_count`, `judge_sample`, and the six thresholds copied as data. Print both lists and the eligible count.

Run: `uv run python scripts/replays/bull_bear_bear_balance_v2.py prepare`
Expected: `pool 230`, `full sample 36`, `judge eligible 116` (from the spec; a different number means the eligibility rule disagrees with the spec's count: stop and report, do not adjust), `judge sample 60`.

- [ ] **Step 2: `run`**: for each full-sample day and arm in `--arms` (default A,B), not already done per `done_key`, call `v1.replay_review(arm, {"date": day, "index": None}, log, fields, prompts[arm])` with `prompts = {"A": load_prompts(OLD_PROMPTS_REV), "B": load_prompts(None)}` and append the line to the v2 results file. Same server-up checks, ServerDown handling and progress print as v1's `_replay_all`, but on v2's own results file (do not reassign v1's module globals).

- [ ] **Step 3: `judge`**: `replay_judge(arm: str, day: str, log, fields, prompts) -> dict`: a `v1._Replay(arm, day, prompts)` with `create(("judge",), review_time)`, `recorder.expect_verdicts(symbols, max_in=params.max_book_size, previous={symbol: pv["verdict"]})` from the logged `previous_verdict`s, then `replay.stage("judge", "submit_verdicts", prompts.judge_task(params.reason_max_chars), judge_context(arm, day, judge_ctx, fields), tool_budget=params.judge_memory_budget)`. Line keys as v1's judge part: `date`, `arm`, `kind: "judge"`, `ok`, `failed_stage`, `stocks`, `verdicts`, `judge_prompt_tokens`, `judge_max_input_tokens`, `stages`, `started`, `seconds`. J-old uses `load_prompts(OLD_PROMPTS_REV)`, J-new `load_prompts(None)`.

- [ ] **Step 4: `score`**: from the v2 results and the logged records:
  - valid days per pair with `valid_pairs`; outcomes with `forward_returns` over the union of valid full and judge days;
  - test 1 from A/B bear levels over valid full days (every debate-set stock);
  - test 2 group: valid judge days, stocks with logged bull medium/high and logged bear high; (outs, stocks) per J arm;
  - test 3: J-new (outs, stocks) on `runway_symbols` of the logged judge fact sheets, valid judge days;
  - test 4: B's high and medium 20d excesses present, valid full days;
  - test 5: B's (rated high, stocks) on `runway_symbols`, valid full days;
  - test 6: v1's flip groups between A and B verdicts, valid full days;
  - `decision(tests, complete)` with `complete` false while any planned (day, arm) line is missing;
  - the report: decision line, coverage (failed per arm, missing lines), the six-test table (numbers, minimum, PASS/FAIL/NOT TESTABLE), then the spec's non-gating sections (B vs A and A vs logged differences; J-old vs logged; J flips with their 20d excess; bear level tables with deciles for A, B, logged; bull x bear out-rate tables for A, B, logged, J-old, J-new; v1's anti-goal rule on the full sample with B's ratings; largest judge prompt and timings per arm).
  Write to `REPORT_FILE_V2` or `--out`.

- [ ] **Step 5: Gate and pre-registration commit** (before any agent call):

Run: `uv run pytest scripts/replays/test_bull_bear_bear_balance_v2.py -q && uv run ruff check scripts/replays/ && uv run ruff format --check scripts/replays/ && uv run pyright scripts/replays/bull_bear_bear_balance_v2.py`
Expected: pass (pyright: no new errors beyond v1's script's own, which this task does not touch).

```bash
git add scripts/replays/bull_bear_bear_balance_v2.py docs/superpowers/replays/data/replay_v2_plan.json
git commit -m "docs(bull_bear): pre-register the v2 bear-balance replay (fresh samples, six tests)"
```

### Task 3: The replay and the report (controller-run, long)

**Files:**
- Create: `docs/superpowers/replays/data/replay_v2_results.jsonl`, `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance-v2.md`

- [ ] **Step 1: Smoke one review per arm** (after the pre-registration commit):

Run: `uv run python scripts/replays/bull_bear_bear_balance_v2.py run --reviews <first full-sample day>` then `... judge --reviews <first judge-sample day>`, then `... score --out <scratchpad>/v2_smoke.md`.
Expected: four `ok=True` lines; J-new's printed fact sheets show the four fields, J-old's none; the smoke report says `NONE: ... incomplete`. The smoke lines stay (resume skips them).

- [ ] **Step 2: Full run in the background** (about 16 h): `run && judge && score`. Poll by line count; no rule or threshold changes while it runs.

- [ ] **Step 3: Read the report**; check coverage (failed reviews, missing lines) and the decision line. Commit:

```bash
git add docs/superpowers/replays/
git commit -m "docs(bull_bear): v2 bear-balance replay report, decision <KEEP|DROP>"
```

- [ ] **Step 4: Report to the user**: the decision, the six numbers, the coverage, and the spec's next step (KEEP: choose a backtest window with the user, ideally one the agents have not seen; DROP: neither branch merged). No merge, no push.

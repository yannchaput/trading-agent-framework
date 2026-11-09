# Bull_bear bear balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the bull_bear bear rate "high" as a balance built on four coded fact-sheet fields, stop the judge treating "high" as a veto, and validate the change with a replay against a keep/drop rule fixed in advance.

**Architecture:** Two fields (`operating_margin_change_pts`, `debt_growth_pct`) come from the SEC figures in `fundamentals.fundamentals_fields`; two (`days_since_filed`, `return_since_filed_pct`) need dated closes, so `DailySeries` gains `dates`, a pure `fact_sheet.since_filing` computes them, and the pipeline merges them into each stock's fundamentals before `fact_sheet_row`. The prompts change in `prompts.py` only. The replay is a throwaway script that rebuilds each stage's context from the 5Y run's log and runs two arms.

**Tech Stack:** Python 3.14, uv, pytest, ruff, LangChain agents through `AgentManager`, vLLM (Qwen 3.8 27B AWQ on `http://localhost:8010`).

**Spec:** `docs/superpowers/specs/2026-11-08-bull-bear-bear-balance-design.md`

## Global Constraints

- Work in the worktree `.claude/worktrees/bull-bear-bear-balance`, branch `worktree-bull-bear-bear-balance`. No push, no merge (the user merges as one squash commit).
- No prompt (system or task) may contain `held`, `weight_pct` or `forced` (`test_no_prompt_explains_held_or_weight`, `test_no_prompt_mentions_forced_exits`).
- The automated tests never touch the network; the replay script is the only thing that calls the model, SEC or Yahoo.
- No look-ahead: fundamentals stay the filings dated before the review day (the existing `SecFundamentals` cutoff); closes stay the completed bars `market_data._series` already cuts; `return_since_filed_pct` starts from the last close dated strictly before `fundamentals_filed`.
- Thresholds, verbatim from the spec: revenue_growth_pct -10 or below; operating_margin_change_pts -10 or below when operating_margin_prior_year_pct is -50 or above; debt_growth_pct +50 or above while free cash flow is negative; runway_months below 12; return_since_filed_pct +20 or above means "already weighed".
- A figure that cannot be computed is absent from the sheet (never `None`, never 0).
- `uv run pytest` and `uv run ruff check` pass at the end of every task.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A filing dated on a weekend or holiday: the base close is the last session before it, never a later one (Task 3 test).
2. A `DailySeries` built without dates (older fakes, a source that cannot date its rows): `days_since_filed` still appears, `return_since_filed_pct` is absent, no crash (Task 3 and Task 4 tests).
3. A malformed or missing `fundamentals_filed` (a sheet with `fundamentals_unavailable`): no new field, no crash (Task 3 test).
4. A prior-year debt of 0 or below, or a missing prior margin: the change field is absent, never a division error or a fake 0 (Task 1 tests).
5. The review day in market time: an `as_of` near midnight UTC still counts days in America/New_York (Task 4 uses `MARKET_TZ`; Task 3 takes a plain `date`).

---

### Task 1: Margin change and debt growth in the SEC fields

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/fundamentals.py` (`fundamentals_fields`, around lines 112-127)
- Test: `tests/strategies/bull_bear/test_bull_bear_fundamentals.py`

**Interfaces:**
- Produces: `fundamentals_fields(...)` output may now carry `operating_margin_change_pts: float` (right after `operating_margin_prior_year_pct`) and `debt_growth_pct: float` (right after `total_debt_prior_year_bn`).

- [ ] **Step 1: Write the failing tests**

```python
def test_the_margin_change_and_debt_growth_are_computed_from_the_shown_figures():
    f = fundamentals_fields(MU_INCOME, MU_BALANCE, stale=False)
    assert f["operating_margin_pct"] == 67.5 and f["operating_margin_prior_year_pct"] == 24.8
    assert f["operating_margin_change_pts"] == 42.7  # the difference of the two rounded margins the agents read
    assert f["debt_growth_pct"] == -17.2  # 9.56 / 11.54 - 1
    keys = list(f)
    assert keys.index("operating_margin_change_pts") == keys.index("operating_margin_prior_year_pct") + 1
    assert keys.index("debt_growth_pct") == keys.index("total_debt_prior_year_bn") + 1


def test_no_prior_margin_or_debt_leaves_the_change_fields_out():
    income = {"revenue": _flow(10e9, "2025-11-28", "2026-02-26"), "operating_income": _flow(1e9, "2025-11-28", "2026-02-26")}
    balance = {"debt": {"value": 2e9, "unit": "USD", "filed": "2026-03-25", "end": "2026-02-26"}}
    f = fundamentals_fields(income, balance, stale=False)
    assert "operating_margin_change_pts" not in f and "debt_growth_pct" not in f


def test_a_prior_debt_of_zero_gives_no_debt_growth():
    balance = {"debt": {"value": 2e9, "unit": "USD", "filed": "2026-03-25", "end": "2026-02-26", "prior_year": {"value": 0.0, "end": "2025-02-27"}}}
    assert "debt_growth_pct" not in fundamentals_fields(MU_INCOME, balance, stale=False)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_fundamentals.py -k "change or debt_growth" -v`
Expected: FAIL with `KeyError: 'operating_margin_change_pts'`.

- [ ] **Step 3: Implement**

In `fundamentals_fields`: `operating_margin_change_pts = round(margin - prior_margin, 1) + 0.0` from the two already-rounded margins, only when both exist; `debt_growth_pct = _pct(debt / debt_before - 1)` only when `debt` is not None and `debt_before > 0`. Insert each right after its prior-year field so the dict order matches the test.

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_fundamentals.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/fundamentals.py tests/strategies/bull_bear/test_bull_bear_fundamentals.py
git commit -m "feat(bull_bear): margin change and debt growth on the fact sheet"
```

### Task 2: Dated daily series

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/market_data.py` (`DailySeries`, `_series`)
- Test: `tests/strategies/bull_bear/test_bull_bear_market_data.py`

**Interfaces:**
- Produces: `DailySeries.dates: list[date] | None = None` (last field; session dates in market time, same rows as `closes`). `_series` always fills it.

- [ ] **Step 1: Write the failing test**

```python
def test_the_series_carries_the_session_date_of_every_close() -> None:
    source = FakeSource([{"AAA": _bars("AAA", sessions=306)}])  # today's bar + 305 completed sessions

    (series,) = _yahoo(source, [], []).load(["AAA"], held=[]).values()

    assert series.dates is not None and len(series.dates) == len(series.closes) == 300
    assert series.dates[-1] == date(2026, 10, 5) and series.closes[-1] == 404.0  # today's (TODAY) bar is dropped with its date
    assert series.dates == sorted(series.dates)


def test_the_gate_series_are_dated_too() -> None:
    # same setup as test_the_gate_reads_each_symbol_skips_failures_and_drops_todays_bar, one symbol
    ...
    assert series.dates is not None and series.dates[-1] == date(2026, 10, 5)
```

The second test copies the gate setup of `test_the_gate_reads_each_symbol_skips_failures_and_drops_todays_bar` for one good symbol; its assertion is the one above.

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_market_data.py -k session_date -v`
Expected: FAIL with `AttributeError` on `dates`.

- [ ] **Step 3: Implement**

Add the field (default `None`, so every existing `DailySeries(...)` call keeps working) and fill it in `_series` from the cut frame's index: `pd.DatetimeIndex(frame.index).tz_convert(MARKET_TZ).date` when the index is tz-aware, `.date` otherwise, as a list of `date`.

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_market_data.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/market_data.py tests/strategies/bull_bear/test_bull_bear_market_data.py
git commit -m "feat(bull_bear): daily series carry their session dates"
```

### Task 3: Days and return since the filing (pure)

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/fact_sheet.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`

**Interfaces:**
- Produces: `since_filing(filed: str | None, dates: Sequence[date] | None, closes: Sequence[float], review_day: date) -> dict[str, Any]` returning `{"days_since_filed": int, "return_since_filed_pct": float}`, each key present only when computable.

- [ ] **Step 1: Write the failing tests**

```python
D = [date(2026, 3, 19), date(2026, 3, 20), date(2026, 3, 23), date(2026, 3, 24), date(2026, 3, 25), date(2026, 3, 26)]
C = [100.0, 101.0, 102.0, 104.0, 90.0, 117.0]


def test_since_filing_counts_calendar_days_and_starts_the_return_the_close_before_the_filing_day():
    out = since_filing("2026-03-25", D, C, date(2026, 4, 14))
    assert out == {"days_since_filed": 20, "return_since_filed_pct": 12.5}  # 117 / 104 - 1: the filing day's own move counts


def test_a_weekend_filing_starts_from_the_last_session_before_it():
    assert since_filing("2026-03-22", D, C, date(2026, 4, 14))["return_since_filed_pct"] == 15.8  # a Sunday: 117 / 101 (Fri 03-20) - 1


def test_a_filing_older_than_the_history_has_no_return():
    assert since_filing("2026-03-19", D, C, date(2026, 4, 14)) == {"days_since_filed": 26}


def test_undated_closes_give_only_the_days():
    assert since_filing("2026-03-25", None, C, date(2026, 4, 14)) == {"days_since_filed": 20}


def test_no_or_a_malformed_filing_date_gives_nothing():
    assert since_filing(None, D, C, date(2026, 4, 14)) == {}
    assert since_filing("not a date", D, C, date(2026, 4, 14)) == {}
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_fact_sheet.py -k since_filing -v` (and the other four by name)
Expected: FAIL with `ImportError: cannot import name 'since_filing'`.

- [ ] **Step 3: Implement `since_filing` in `fact_sheet.py`**

Parse `filed[:10]` with `date.fromisoformat` (any `ValueError`/`TypeError` → `{}`). Base = the last `closes[i]` whose `dates[i] < filed` (`bisect_left` on `dates`); no base, a base `<= 0`, `dates is None` or a length mismatch → no return key. Return value through the module's `_pct` (one decimal). `fact_sheet_row` is unchanged.

- [ ] **Step 4: Run the file**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_fact_sheet.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/fact_sheet.py tests/strategies/bull_bear/test_bull_bear_fact_sheet.py
git commit -m "feat(bull_bear): days and return since the latest filing"
```

### Task 4: The pipeline puts the filing fields on every sheet

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (`_rank`, `_review` step 3)
- Test: `tests/strategies/bull_bear/test_bull_bear_pipeline.py`

**Interfaces:**
- Consumes: `DailySeries.dates` (Task 2), `since_filing` (Task 3).
- Produces: every fact sheet whose fundamentals carry `fundamentals_filed` also carries `days_since_filed`, and `return_since_filed_pct` when the stock's dated closes reach back before the filing.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_sheets_carry_the_days_and_return_since_the_filing(tmp_path: Path) -> None:
    # dated series: 300 sessions ending 2026-10-05 (the harness clock is 2026-10-06 12:00 ET)
    h = _harness(tmp_path, series=_dated_series(), fundamentals_of=lambda symbol, as_of: {"fundamentals_filed": "2026-09-15", "revenue_growth_pct": 5.0})
    assert h.run().completed
    sheet = h.agents["bear"].calls[0]["context"]["stocks"][0]["fact_sheet"]
    assert sheet["days_since_filed"] == 21
    assert "return_since_filed_pct" in sheet
    judge_sheet = h.agents["judge"].calls[0]["context"]["stocks"][0]["fact_sheet"]
    assert judge_sheet["days_since_filed"] == 21


def test_undated_series_still_give_the_days_since_the_filing(tmp_path: Path) -> None:
    h = _harness(tmp_path, fundamentals_of=lambda symbol, as_of: {"fundamentals_filed": "2026-09-15"})
    assert h.run().completed
    sheet = h.agents["bull"].calls[0]["context"]["stocks"][0]["fact_sheet"]
    assert sheet["days_since_filed"] == 21 and "return_since_filed_pct" not in sheet
```

`_dated_series()` is a new test helper: `_default_series()` with `dates` set to the 300 weekday sessions ending 2026-10-05. If `_harness` takes its series under another keyword, use that one.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_pipeline.py -k since_the_filing -v`
Expected: FAIL with `KeyError: 'days_since_filed'`.

- [ ] **Step 3: Implement**

`_rank` keeps the loaded `series` on the pipeline (`self._daily: dict[str, DailySeries]`, reset to `{}` at the start of every `_rank`). Step 3 of `_review` builds each stock's fundamentals as `{**fundamentals, **since_filing(fundamentals.get("fundamentals_filed"), daily.dates, daily.closes, as_of.astimezone(MARKET_TZ).date())}` with `daily = self._daily[stock.symbol]`, then calls `fact_sheet_row` as before. A sheet with `fundamentals_unavailable` has no `fundamentals_filed` and gains nothing.

- [ ] **Step 4: Run the bull_bear tests**

Run: `uv run pytest tests/strategies/bull_bear -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/pipeline.py tests/strategies/bull_bear/test_bull_bear_pipeline.py
git commit -m "feat(bull_bear): the fact sheet says how the stock moved since its filing"
```

### Task 5: The prompts and the strategy notes

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`_FIGURES`, `BEAR_SYSTEM`, `JUDGE_SYSTEM`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md`
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:**
- Consumes: the four field names from Tasks 1-4.

- [ ] **Step 1: Update and add the tests**

- `test_every_agent_reads_the_fundamentals_fields_and_the_sources_rule`: add `"operating_margin_change_pts"`, `"debt_growth_pct"`, `"days_since_filed"`, `"return_since_filed_pct"` to `fields`.
- Replace `test_the_bear_rates_high_only_on_fundamental_evidence` with:

```python
def test_the_bear_rates_high_only_on_a_large_unweighed_deterioration_that_outweighs_the_strengths() -> None:
    text = " ".join(prompts.BEAR_SYSTEM.split())
    assert "Base each case on the stock's fundamentals and financial health; news only supports it." in text
    rule = text[text.index("Rate a stock high only when all three hold") :][:1400]
    for piece in (
        "revenue_growth_pct of -10 or below",
        "operating_margin_change_pts of -10 or below when operating_margin_prior_year_pct is -50 or above",
        "debt_growth_pct of 50 or above while free cash flow is negative",
        "runway_months below 12",
        "return_since_filed_pct is 20 or above",
        "medium, not high",
        "clearly outweighs",
    ):
        assert piece in rule, piece
    for fact in ("a guidance cut", "a share issuance", "going-concern doubts", "a covenant breach", "a restatement", "an impairment"):
        assert fact in rule, fact
    assert "the company itself reports or files" in rule
    assert "rate it medium and name it" in text
    assert "When the fact sheet's fundamentals are unavailable or stale, the company's own reported facts in the note are your evidence." in text
    assert "A fallen price with intact cash flow makes a company cheaper, not riskier." in text
    assert "a decline in revenue or operating margin, a fall in free cash flow or a rise in debt" not in text  # the old any-decline list
```

- Replace `test_the_judge_puts_a_stock_out_on_a_fundamental_risk_not_on_news_from_others` with:

```python
def test_the_judge_weighs_the_bears_evidence_not_its_level() -> None:
    text = " ".join(prompts.JUDGE_SYSTEM.split())
    assert "The bear's risk level is its argument, not a verdict" in text
    assert "weigh a high risk the way you weigh a medium one" in text
    assert "return_since_filed_pct of 20 or above" in text
    assert "Put a stock out when the bear names a risk shown in its fundamentals" not in text
    assert "Neither is news from others, already priced or not, whatever consequence the bear argues for it." in text
    assert "lasting consequence" not in text and "already reacted to" not in text
    assert "not a reason for out by itself" in text
    assert "When the fact sheet's fundamentals are unavailable or stale, the company's own reported facts in the note are the evidence." in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -v`
Expected: the three tests above FAIL; every other prompt test still PASSES.

- [ ] **Step 3: Rewrite the prompt text**

- `_FIGURES`: one sentence each for the four fields, after the debt / current-assets sentences: `operating_margin_change_pts` (the margin less the prior year's, in percentage points), `debt_growth_pct` (total debt against a year earlier), `days_since_filed` (calendar days since `fundamentals_filed`), `return_since_filed_pct` (the stock's return from the last close before `fundamentals_filed` to the last close: how the market moved since it saw those figures).
- `BEAR_SYSTEM`: replace the sentence that starts "Rate a stock high only on evidence in its fundamentals" (through "...the company's own reported facts in the note are your evidence.") with the spec's section B rule, opening with "Rate a stock high only when all three hold" and carrying every test string above. Keep the news-from-others sentence, the unavailable-or-stale sentence and the fallen-price sentence verbatim; keep the momentum_exhaustion, "Use none with low..." and "do not give every stock the same rating" sentences unchanged.
- `JUDGE_SYSTEM`: replace "Put a stock out when the bear names a risk shown in its fundamentals (...) that outweighs the bull's case." with the spec's section C content, carrying the test strings. Everything else unchanged.
- No absolutes ("no risk", "always", "never" applied to a stock's risk), per the spec.

- [ ] **Step 4: Run the prompt tests, then everything**

Run: `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -v && uv run pytest -q && uv run ruff check`
Expected: all PASS, ruff clean.

- [ ] **Step 5: Update `bull_bear/CLAUDE.md`**

In the architecture paragraph: replace the bear's old high rule ("The bear names a concern and rates a stock HIGH only on evidence in its fundamentals: a decline in revenue ...") with the new three-part rule and its thresholds, add the four fields to the fact-sheet description (what each is and that code computes them because "large" and "recent" in words were applied inconsistently and the move since a filing cannot be computed by the model), and add the judge's "the bear's level is an argument, not a verdict". Two lines of why: the 5Y run's high was not predictive (20d excess +0.2%, highest best-decile rate 13.9%) while the judge put out 97% of it. Update the window estimate sentence with the measured figures from the spec's constraints (judge largest call 19,031 tokens).

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/bull_bear/prompts.py src/trading_agent_framework/strategies/bull_bear/CLAUDE.md tests/strategies/bull_bear/test_bull_bear_prompts.py
git commit -m "feat(bull_bear): the bear rates high as a balance and the judge weighs evidence, not the level"
```

### Task 6: The replay and its report

**Files:**
- Create: `scripts/replays/bull_bear_bear_balance.py` (throwaway, kept for the record; not a repo command)
- Create: `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance.md` (the report)

**Interfaces:**
- Consumes: `fundamentals_fields` / `SecFundamentals` (Task 1), `since_filing` (Task 3), the new prompts (Task 5); the old prompts are read with `git show 5bcab74:src/trading_agent_framework/strategies/bull_bear/prompts.py` and exec'd into a namespace, so both arms run in one process; `HandoffRecorder` and `submit_tools` from `handoff.py`; `bull_task`, `bear_task`, `judge_task`; `AgentManager`; `agent_scores.forward_returns` and `scores_cli.fetch_opens`.
- Run directory: `logs/bull_bear/backtesting/2026-11-05_012455_backtesting` in the MAIN checkout (`/mnt/data/projets/trading-agent-framework/logs/...`; logs are not in the worktree). SEC cache: the main checkout's `cache/sec`.

- [ ] **Step 1: Inputs and the fixed lists, printed before any agent runs**

The script's `prepare` stage:
1. Parses `backtest.log`: every `HumanMessage(content=...)` that starts with "Make the bull case", "Make the bear case" or "Give your verdict" (Python reprs: `ast.literal_eval`), the context after `Context:\n` (`ast.literal_eval` again), keyed by `current_datetime[:10]` and stage. Asserts 262 of each.
2. Sample: the records of `reviews.jsonl` at indices `[5 + 11 * k for k in range(24)]`.
3. Arm-B fields for every candidate of every review (all 262: the anti-goal list needs them): `SecFundamentals` over the main checkout's SEC cache as of the review's `current_datetime`, then `fundamentals_fields`' `operating_margin_change_pts` / `debt_growth_pct`, plus `since_filing(...)` on `YahooDailyBars` dated closes cut to the sessions before the review day. Cache the result as `replay_fields.json` next to the report's data.
4. The anti-goal list from all 262 reviews, by the spec's rule (up to 12, `random.Random(20261108).sample` over the qualifying list sorted by review date then symbol).
5. Prints the sample dates and the anti-goal list, writes both to `replay_plan.json`, and stops. Commit that file before Step 2 (it is the pre-registration).

Run: `uv run python scripts/replays/bull_bear_bear_balance.py prepare`
Expected: 24 dates, the anti-goal list (or fewer than 6, which the report must flag), no agent call.

- [ ] **Step 2: Dry run on one review**

`run --arms A,B --reviews <first sample date>`: for each arm, a fresh `HandoffRecorder`, bull then bear on the logged bull/bear context (arm B: each `fact_sheet` gets the four fields), then the judge on the logged judge context with `fact_sheet`, `bull` and `bear` replaced by that arm's and `previous_verdict` as logged. Settings from `BullBearParams()` (thinking budgets, output caps, temperature, `max_agent_steps`); the judge gets `memory_tools` over an empty `MemoryStore` in a temp directory. A stage with no valid submission gets one retry with the submit tool forced; a second failure marks that review failed for that arm (excluded from both arms in the tests, and counted in the report). Output: one JSON line per review and arm in `replay_results.jsonl` (bull, bear, verdicts, the judge's measured prompt tokens, seconds per stage).

Run: `uv run python scripts/replays/bull_bear_bear_balance.py run --arms A,B --reviews 2021-10-26` (index 5, the first sample review; the last is 2026-09-08)
Expected: two result lines; the arm-B bear context shows the four fields on stocks with fundamentals.

- [ ] **Step 3: The full replay (background, several hours)**

Run: `uv run python scripts/replays/bull_bear_bear_balance.py run --arms A,B` then `uv run python scripts/replays/bull_bear_bear_balance.py anti-goal` (arm-B bear only, on the anti-goal reviews outside the sample).
Expected: 48 result lines (24 x 2) and one bear line per extra anti-goal review; failures counted, not fatal.

- [ ] **Step 4: Score and decide**

`score` computes, from `replay_results.jsonl`, the logged run and the 20-session excess of `agent_scores.forward_returns` (opens from `scores_cli.fetch_opens`), each of the spec's five keep/drop tests with its number and PASS/FAIL, plus the non-gating tables (arm B's bear level table in the spec's format, the bull x bear out-rate table for A and B, the measured largest judge prompt). Writes the report with the decision line `KEEP` or `DROP` and why.

Run: `uv run python scripts/replays/bull_bear_bear_balance.py score`
Expected: `docs/superpowers/replays/2026-11-08-bull-bear-bear-balance.md` written; the decision follows the rule mechanically.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check scripts/replays/bull_bear_bear_balance.py
git add scripts/replays/bull_bear_bear_balance.py docs/superpowers/replays/
git commit -m "docs(bull_bear): replay of the balance-based bear against the logged 5Y run"
```

Then stop and report the decision to the user. On DROP the branch is not merged; on KEEP the backtest is planned with the user.

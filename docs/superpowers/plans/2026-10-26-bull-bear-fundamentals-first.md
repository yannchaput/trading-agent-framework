# bull_bear: fundamentals first, news as support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every bull_bear fact sheet carries the stock's latest SEC fundamentals and a "decline before the last week" return, and the four prompts make fundamentals (not third-party news) the ground for a high bear risk and a judge `out`.

**Architecture:** The framework's SEC client gets a 1-day default cache age (`fundamentals/`). A new pure-plus-injected-client module `strategies/bull_bear/fundamentals.py` reduces `sec.income_statement`/`sec.balance_sheet` to compact fact-sheet fields; `ReviewPipeline` takes a `fundamentals_of(symbol, as_of)` callable (injected like `sector_of`) and merges its result into `fact_sheet_row`. Prompts change in `prompts.py` only.

**Tech Stack:** Python 3.14, `uv`, pytest with hand-written fakes, ruff.

**Spec:** `docs/superpowers/specs/2026-10-26-bull-bear-fundamentals-first-design.md`

## Global Constraints

- No wall-clock time in data code: SEC figures are read as of `strategy.clock.now()` (root `CLAUDE.md`).
- Tests never touch the network; hand-written fakes, no `MagicMock` (root `CLAUDE.md`).
- `fact_sheet.py` stays pure; SEC I/O only through an injected `SecEdgarClient`-shaped object.
- Agents must not recompute fact-sheet numbers; percentages one decimal; money in billions, 3 decimals, reporting currency labelled, never converted.
- An SEC failure for one stock never abandons a review.
- Framework SEC freshness default: **1 day** by the strategy clock; an explicit `max_age_days` passed by a caller (earnings_drift's and insider_trades' `0`) is kept; bill_ackman's screen is untouched.
- No prompt may contain the word `held` (existing test `test_no_prompt_explains_held_or_weight`) or a review cadence word.
- Valuation fields are out of scope.
- TDD; one commit per task (tests + code + docs); commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never stage `TODO.md`.

## Review Focus

1. A debate stock SEC does not know (no CIK) → its fact sheet says `fundamentals_unavailable`, the review goes on (Task 2 test, Task 3 pipeline test).
2. Revenue at or below zero, or a missing prior year → no `ZeroDivisionError`; the dependent fields are omitted (Task 2 tests).
3. The injected `fundamentals_of` raising an unexpected exception → one WARNING, `fundamentals_unavailable`, review continues (Task 3 test).
4. An IFRS filer reporting in EUR → `currency: "EUR"`, amounts not converted (Task 2 test).
5. A cached SEC file 2 days old in paper/live is refetched; one 12 hours old is not; a past `as_of` (backtest) never refetches; an explicit `max_age_days=0` still refetches (Task 1 tests).

---

### Task 1: SEC cache freshness as a framework default

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/freshness.py` (constant)
- Modify: `src/trading_agent_framework/fundamentals/edgar_client.py` (getter defaults, `default_client`)
- Modify: `src/trading_agent_framework/agents/tools/fundamentals.py` (drop `MAX_PAYLOAD_AGE_DAYS` and `_default_client`)
- Modify: `src/trading_agent_framework/fundamentals/CLAUDE.md`, `src/trading_agent_framework/agents/CLAUDE.md` (any mention of the 30-day tool age)
- Test: `tests/fundamentals/test_edgar_client.py`, `tests/agents/tools/test_fundamentals_tools.py`

**Interfaces:**
- Produces: `freshness.DEFAULT_MAX_AGE_DAYS: int = 1`; `SecEdgarClient.get_company_facts_payload(cik, *, as_of=None, max_age_days=DEFAULT_MAX_AGE_DAYS)` and `get_submissions_payload(...)` with the same default (staleness still applies only when `as_of` is given); `edgar_client.default_client(project_root: Path) -> SecEdgarClient` (raises `ConfigurationError("Missing or blank SEC_EDGAR_USER_AGENT environment variable")` as `_default_client` does today, cache at `project_root / "cache" / "sec"`).

- [ ] **Step 1: Write the failing tests** in `tests/fundamentals/test_edgar_client.py` (reuse the file's existing cached-file helpers; set a file's mtime with `os.utime`):
  - `test_by_default_a_cached_payload_older_than_one_day_is_refetched`: cached file mtime = now − 2 days, `get_company_facts_payload(cik, as_of=datetime.now(UTC))` with no `max_age_days` → the fake transport is called and the file rewritten.
  - `test_by_default_a_cached_payload_younger_than_one_day_is_served`: mtime = now − 12 hours → no fetch.
  - `test_by_default_a_past_as_of_never_refetches`: mtime = now, `as_of=datetime(2025, 1, 2, tzinfo=UTC)` (a backtest date) → served, no fetch.
  - `test_an_explicit_zero_max_age_still_refetches`: mtime = now − 1 minute, `max_age_days=0`, `as_of=now` → fetched.
  - `test_default_client_requires_the_user_agent(monkeypatch, tmp_path)`: blank env → `ConfigurationError` matching `SEC_EDGAR_USER_AGENT`.
  - Same default for `get_submissions_payload` (one test).
- [ ] **Step 2:** In `tests/agents/tools/test_fundamentals_tools.py`, change the freshness assertion(s) so the tools pass `as_of` and **no** `max_age_days` (the fake records the default it receives; make the fake's default `max_age_days` a sentinel and assert the sentinel is recorded). Run `uv run pytest tests/fundamentals/test_edgar_client.py tests/agents/tools/test_fundamentals_tools.py -q` → the new tests FAIL.
- [ ] **Step 3: Implement.** Add the constant (with a one-line comment: why 1 day, and that backtests are unaffected); default the two getters to it; move `_default_client` into `edgar_client.default_client(project_root)` and have `fundamentals_tools` call `default_client(strategy.project_root)`; the tools stop passing `max_age_days`. Update `freshness.py`'s module docstring ("live refreshes monthly" → the 1-day default) and `get_json`'s docstring.
- [ ] **Step 4:** `uv run pytest tests/fundamentals tests/agents/tools tests/strategies/earnings_drift tests/strategies/insider_trades tests/strategies/bill_ackman -q` → PASS; `uv run ruff check` clean.
- [ ] **Step 5:** Update `fundamentals/CLAUDE.md` ("the drill-down tools pass 30", "live refreshes monthly") and any 30-day mention in `agents/CLAUDE.md`: one framework default of 1 day in `freshness.DEFAULT_MAX_AGE_DAYS`, explicit values kept, bill_ackman's screen and backtests unaffected.
- [ ] **Step 6: Commit** `feat(fundamentals): SEC cache freshness of 1 day as a framework default`.

### Task 2: Fundamentals fields for the fact sheet

**Files:**
- Create: `src/trading_agent_framework/strategies/bull_bear/fundamentals.py`
- Test: `tests/strategies/bull_bear/test_bull_bear_fundamentals.py`

**Interfaces:**
- Consumes: `sec.income_statement(payload, *, as_of)`, `sec.balance_sheet(payload, *, as_of)`, `sec.is_stale_statement(values, *, as_of)` (values carry `value`, `unit`, `filed`, `form`, `start` (flows), `end`, optional `prior_year: {value, start?, end}`; income keys used: `revenue`, `operating_income`, `free_cash_flow`; balance key: `debt`); `SecEdgarClient.ticker_to_cik`, `get_company_facts_payload(cik, *, as_of)` (Task 1 default age); `FundamentalsNotFoundError`, `FundamentalsError`.
- Produces:
  - `fundamentals_fields(income: Mapping[str, Mapping[str, Any]], balance: Mapping[str, Mapping[str, Any]], *, stale: bool) -> dict[str, Any]` (pure)
  - `unavailable_fundamentals(reason: str) -> dict[str, Any]` → `{"fundamentals_unavailable": reason}`
  - `class SecFundamentals: __init__(self, client) ; __call__(self, symbol: str, as_of: datetime) -> dict[str, Any]`
  - Field names (exact): `fundamentals_period_end`, `fundamentals_filed` (ISO dates `YYYY-MM-DD`), `period_months`, `revenue_growth_pct`, `operating_margin_pct`, `operating_margin_prior_year_pct`, `free_cash_flow_bn`, `free_cash_flow_prior_year_bn`, `free_cash_flow_months`, `total_debt_bn`, `total_debt_prior_year_bn`, `currency`, `fundamentals_stale` (only when true), `fundamentals_unavailable` (only when no figures).

Rules the tests pin: a figure that cannot be computed is **omitted** (not `None`); growth needs a prior revenue > 0; a margin needs its revenue > 0; months = round(days / 30.44) from `start`/`end`; period end, filed and `period_months` come from `revenue` (else the first income value that has them); `currency` is the revenue's unit, else the first money field's; percentages one decimal, billions 3 decimals; `income` and `balance` both empty → `unavailable_fundamentals("no SEC figures")`.

- [ ] **Step 1: Write the failing tests:**

```python
def _flow(value, start, end, prior=None, unit="USD", filed="2026-03-25", form="10-Q"):
    row = {"value": value, "unit": unit, "filed": filed, "form": form, "start": start, "end": end}
    if prior is not None:
        row["prior_year"] = {"value": prior[0], "start": prior[1], "end": prior[2]}
    return row

MU_INCOME = {
    "revenue": _flow(23.86e9, "2025-11-28", "2026-02-26", (8.05e9, "2024-11-29", "2025-02-27")),
    "operating_income": _flow(16.1e9, "2025-11-28", "2026-02-26", (2.0e9, "2024-11-29", "2025-02-27")),
    "free_cash_flow": _flow(8.54e9, "2025-08-29", "2026-02-26", (-0.075e9, "2024-08-30", "2025-02-27")),
}
MU_BALANCE = {"debt": {"value": 9.56e9, "unit": "USD", "filed": "2026-03-25", "end": "2026-02-26", "prior_year": {"value": 11.54e9, "end": "2025-02-27"}}}

def test_the_fields_compare_the_latest_period_with_a_year_earlier():
    f = fundamentals_fields(MU_INCOME, MU_BALANCE, stale=False)
    assert f["revenue_growth_pct"] == 196.4
    assert f["operating_margin_pct"] == 67.5 and f["operating_margin_prior_year_pct"] == 24.8
    assert f["free_cash_flow_bn"] == 8.54 and f["free_cash_flow_prior_year_bn"] == -0.075
    assert f["total_debt_bn"] == 9.56 and f["total_debt_prior_year_bn"] == 11.54
    assert f["period_months"] == 3 and f["free_cash_flow_months"] == 6
    assert f["fundamentals_period_end"] == "2026-02-26" and f["fundamentals_filed"] == "2026-03-25"
    assert f["currency"] == "USD" and "fundamentals_stale" not in f and "fundamentals_unavailable" not in f
```

  Plus: `test_no_prior_year_omits_growth_and_the_prior_margin`; `test_revenue_at_or_below_zero_omits_the_margins_and_growth` (revenue 0 and prior revenue 0, no exception); `test_an_ifrs_filer_keeps_its_currency` (unit `"EUR"` → `currency == "EUR"`, amounts unchanged); `test_stale_figures_are_flagged`; `test_no_figures_at_all_is_unavailable`; `SecFundamentals` with a hand-written fake client: `test_an_unknown_ticker_is_unavailable_not_an_error` (`ticker_to_cik` raises `FundamentalsNotFoundError` → `{"fundamentals_unavailable": "not an SEC filer"}`), `test_a_failed_sec_request_is_unavailable` (`FundamentalsError` → reason `"SEC request failed"`), `test_the_figures_are_read_as_of_the_given_time` (the fake records the `as_of` passed to `get_company_facts_payload` and no `max_age_days`; a payload with one fact filed after `as_of` does not show).
- [ ] **Step 2:** `uv run pytest tests/strategies/bull_bear/test_bull_bear_fundamentals.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** `fundamentals.py` (module docstring: pure reduction plus a thin SEC source; why the fact sheet carries fundamentals). `SecFundamentals.__call__` catches `FundamentalsNotFoundError` first, then `FundamentalsError`; stale = `sec.is_stale_statement(income, as_of=as_of)`.
- [ ] **Step 4:** run the file → PASS; `uv run ruff check` clean.
- [ ] **Step 5: Commit** `feat(bull_bear): fundamentals fields for the fact sheet from SEC`.

### Task 3: Fact sheet, pipeline and strategy wiring

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/fact_sheet.py`
- Modify: `src/trading_agent_framework/strategies/bull_bear/pipeline.py` (constructor, step 3 "Fact sheet")
- Modify: `src/trading_agent_framework/strategies/bull_bear/agent_bull_bear.py` (kwarg, one shared SEC client)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (fact sheet section)
- Test: `tests/strategies/bull_bear/test_bull_bear_fact_sheet.py`, `test_bull_bear_pipeline.py`, `test_bull_bear_strategy.py`, `test_bull_bear_backtest.py`

**Interfaces:**
- Consumes: Task 2's `SecFundamentals`, `unavailable_fundamentals`; Task 1's `default_client(project_root)`.
- Produces: `fact_sheet_row(stock: DebateStock, *, sector: str, fundamentals: Mapping[str, Any]) -> dict[str, Any]` (adds `return_1m_skip_1w_pct` and merges `fundamentals` after `sector`); `ReviewPipeline(..., fundamentals_of: Callable[[str, datetime], Mapping[str, Any]], ...)`; `BullBearStrategy(..., fundamentals_of: Callable[[str, datetime], Mapping[str, Any]] | None = None)`.

- [ ] **Step 1: Write the failing tests:**
  - fact sheet: `test_the_return_before_the_last_week_leaves_out_the_last_5_sessions` — with `closes = [100.0 + i for i in range(300)]`, `sheet["return_1m_skip_1w_pct"] == 4.2` (`(394 - 378) / 378`, i.e. `compute_return_from_prices(closes, 16, 5)`); `test_the_fundamentals_are_merged_into_the_row` (`fundamentals={"revenue_growth_pct": 12.0}` appears; `held` still absent). Update existing calls to pass `fundamentals={}`.
  - pipeline (existing fixture at `test_bull_bear_pipeline.py:177` passes `sector_of`): add `fundamentals_of`; `test_the_fact_sheet_carries_each_stocks_fundamentals_as_of_the_clock` (a recording fake returns `{"revenue_growth_pct": 5.0}`; the researcher's context fact sheet contains it; the recorded `as_of` equals the strategy clock's now); `test_a_failing_fundamentals_source_costs_that_stock_its_figures_not_the_review` (the fake raises `RuntimeError` for one symbol → that fact sheet has `fundamentals_unavailable`, a WARNING naming the symbol is logged, the review completes).
  - strategy/backtest: pass `fundamentals_of=lambda symbol, as_of: {}` where `sector_of` is injected today; `test_without_an_injected_source_the_strategy_reads_sec` (no `fundamentals_of` → the pipeline's source is a `SecFundamentals`; read it from the built pipeline's attribute).
- [ ] **Step 2:** `uv run pytest tests/strategies/bull_bear -q` → new tests FAIL.
- [ ] **Step 3: Implement.** `fact_sheet_row` adds `"return_1m_skip_1w_pct": _pct(compute_return_from_prices(row.closes, 16, _WEEK))` next to `return_1w_pct` (name the 16 as a module constant with a comment: the 16 sessions of the month before the last week). Pipeline step 3: `now = self._strategy.clock.now()`; per stock, `fundamentals_of(symbol, now)`, any `Exception` → `log_warning(f"[bull_bear] fundamentals for {symbol} unavailable: {exc}")` and `unavailable_fundamentals("error")`. Strategy: build one client with `default_client(self.project_root)` in `initialize` (inside the existing `try`, so a missing user agent stays a `FatalStrategyError`), pass it to `fundamentals_tools(self, client=client)` and, unless `fundamentals_of` was injected, `SecFundamentals(client)` to the pipeline.
- [ ] **Step 4:** `uv run pytest tests/strategies/bull_bear -q` → PASS; full `uv run pytest -q` → PASS; `uv run ruff check` clean.
- [ ] **Step 5:** `bull_bear/CLAUDE.md`: the fact sheet's new fields, where they come from (SEC as of the clock, framework 1-day cache), `fundamentals_unavailable` on any failure (never abandons), `return_1m_skip_1w_pct`.
- [ ] **Step 6: Commit** `feat(bull_bear): fundamentals and the return before the last week in the fact sheet`.

### Task 4: Prompts — fundamentals first

**Files:**
- Modify: `src/trading_agent_framework/strategies/bull_bear/prompts.py` (`_FACT_SHEET`, `researcher_system`, `BULL_SYSTEM`, `BEAR_SYSTEM`, `JUDGE_SYSTEM`)
- Modify: `src/trading_agent_framework/strategies/bull_bear/CLAUDE.md` (the rules)
- Test: `tests/strategies/bull_bear/test_bull_bear_prompts.py`

**Interfaces:** Consumes Task 2/3's field names (exact).

Sentences the tests pin (exact text, place them where they read naturally; the rest of the wording is free within the spec):
- `_FACT_SHEET`: names every Task 2 field and `return_1m_skip_1w_pct`; contains `"Sources can be wrong or stale: do not repeat a figure from a news item as fact when the fact sheet or a filing says otherwise."`
- Researcher (both variants): `"The fact sheet already gives the latest SEC figures against a year earlier"`; asks for the company's own reported facts with dates (`"what the company itself reported or filed"`); keeps `prior_year` (its tools' values) and "Do not repeat the fact sheet's numbers"; the no-news variant still has no "news".
- Bull: `"Base each case on the stock's fundamentals and financial health"`.
- Bear:
  - `"Base each case on the stock's fundamentals and financial health; news only supports it."`
  - `"Rate a stock high only on evidence in its fundamentals"` followed by the fact-sheet declines (revenue, operating margin, free cash flow, debt) and the company-reported facts (results, a guidance cut, a share issuance, going-concern doubts, a covenant breach, a restatement, an impairment).
  - `"News from others is never a high risk by itself, whatever consequence you argue"` with the examples (a competitor's technology, an analyst action, a lawsuit headline, speculation about the future).
  - `"When the fact sheet's fundamentals are unavailable or stale, the company's own reported facts in the note are your evidence."`
  - `"A fallen price with intact cash flow makes a company cheaper, not riskier."`
  - `"Use momentum_exhaustion only for a decline that ran for weeks: a drawdown_from_52w_high_pct of -15 or below together with a return_1m_skip_1w_pct of -10 or below"`; keeps `"still rising is never momentum_exhaustion"`, the strong-trend / 200-day / three-month sentence and the concern list.
  - Removed: `"negative one-month return"`, `"lasting consequence"`, `"already reacted to"`.
- Judge: `"Put a stock out when the bear names a risk shown in its fundamentals"`; `"Neither is news from others, already priced or not, whatever consequence the bear argues for it."`; removed: `"lasting consequence"`; everything else (max_in, previous_verdict, recent_verdicts, memory) unchanged.

- [ ] **Step 1: Write the failing tests.** New: `test_every_agent_reads_the_fundamentals_fields_and_the_sources_rule`, `test_the_bear_rates_high_only_on_fundamental_evidence`, `test_the_bear_treats_news_from_others_as_support_only`, `test_momentum_exhaustion_needs_a_decline_that_ran_for_weeks`, `test_the_bull_argues_from_fundamentals`, `test_the_judge_puts_out_only_on_a_fundamental_risk`, `test_the_researcher_reads_the_figures_and_looks_for_what_explains_them`. Update the tests that pin removed text: `test_momentum_exhaustion_needs_a_reversal_already_under_way` (now the new test), `test_the_bear_may_not_rate_news_the_price_already_absorbed_high_by_itself` (assert `"lasting consequence"` and `"already reacted to"` are absent from BEAR and JUDGE), line 84's `"already reacted to"` in the judge test.
- [ ] **Step 2:** `uv run pytest tests/strategies/bull_bear/test_bull_bear_prompts.py -q` → FAIL.
- [ ] **Step 3: Implement** the prompt changes; update the module docstring if it describes the fact sheet.
- [ ] **Step 4:** `uv run pytest -q` → PASS; `uv run ruff check` clean.
- [ ] **Step 5:** `bull_bear/CLAUDE.md`: fundamentals first (bull, bear, judge), what counts as evidence for a high (fact sheet declines, company-reported facts; not third-party news), the new `momentum_exhaustion` test with MU 2026-03-31 as the example, the sources rule.
- [ ] **Step 6: Commit** `feat(bull_bear): fundamentals first, news as support, in the four prompts`.

---

## After the final review: validation (no repo changes)

Run by the controller, not an implementer; results reported to the user before any backtest.

1. **Replay** (throwaway script in the session scratchpad, modelled on the 2026-10-26 `bb_replay.py` / `jr_replay*.py`): for the 10-23 run's 2026-03-31 review, rebuild each debate stock's fact sheet with the new `fact_sheet_row` (closes from the backtest data as of that date; fundamentals with `SecFundamentals` and `as_of` = the review time), keep the saved researcher notes, run bear ×3 and bull ×3 at the current settings, then the judge on their cases. Pass: MU never high and never `momentum_exhaustion`; the judge puts MU in.
2. **Anti-goal check:** find, in the saved runs' reviews, stocks whose new fact sheet shows a revenue or operating margin decline (or whose note has a company-reported deterioration); replay the bear ×3 on such a review: the stock rated high in most runs. None found → say so.
3. Report the bear's number of highs per run on identical input (old spread: 2 to 8 of 12).
4. **1Y backtest, Qwen 3.8**, once the user asks for it: compare with the 3.6 runs (+61% to +91%) and the 3.8 runs (+42%, +44.6%); check the log for "too long to summarize".

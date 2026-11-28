# Earnings and profitability entry gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test two entry gates (earnings reaction, gross profitability) against the ungated rank-only book under bull_bear's book rules, development then a one-shot holdout, rule A (higher CAGR, Sharpe no lower).

**Architecture:** One research script, `scripts/bull_bear/entry_gates.py`, importing `rank_only_stress` (data, simulator) and `score_variants` (`ranked`, `book_picks`, `window_stats`, `passes`, `table`) as siblings, plus `earnings_drift.events` and `fundamentals` from `src/`. Pure signal functions checked by `self_check()`; a separate `--fetch` step downloads SEC data into `cache/sec/`. No `src/` change.

**Tech Stack:** Python 3.14, pandas, `SecEdgarClient`, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-11-28-earnings-profitability-gates-design.md`

## Global Constraints

- Gates exactly as the spec: `earnings` blocks a reaction < 0 (3-session window close(r+1)/close(r-2) - 1 minus SPY's; usable from the session after r+1; older than 126 sessions = missing); `profitability` blocks the bottom third of the day's eligible stocks with a value (10-K gross profit / assets, fiscal periods of 350-380 days, older than 18 months = missing). Missing = allowed. No tuning.
- Held stocks are never blocked; ranks (and the keep-to-35 rule) come from the full ungated ranking; the gate only removes fill candidates. Fewer than 10 passers: the book holds fewer, rest in SHV.
- Book: `score_variants.ranked(..., "today")` order, `book_picks` rules (keep to rank 35, fill to 10, 9.8%, weekly, 5 bps, no cooldown), `rank_only_stress.simulate` unchanged.
- Windows: `earnings` dev 2005-01-01..2015-12-31, holdout 2016-01-01..2026-11-24; `profitability` dev 2012-01-01..2018-12-31, holdout 2019-01-01..2026-11-24. Fresh book at 1.0; compared with the ungated book on the same window.
- Holdout runs only for a dev passer, in the same invocation. Kept only if it passes both. No rule change after a result is seen.
- SEC user agent from `SEC_EDGAR_USER_AGENT` (loaded with `load_dotenv("env/.env.earnings_drift_baseline.backtesting")`), never written into the repo. Cache under `cache/sec/` and `cache/stress_2005/` only.

## Review Focus

- A later 10-K repeats last year's figures with a later `filed` date: the value used is always that of the latest period end filed strictly before the trade day, never a row filed on or after it.
- A 10-K's 3-month Q4 rows or 9-month rows: excluded by the 350-380-day rule, never read as a fiscal year.
- Zero, negative or missing assets: missing (allowed), never `inf` or a crash.
- An event whose r+1 session is the trade day or later is not yet usable (look-ahead).
- A held stock with a blocked signal stays held while ranked within 35, and the ungated and gated books give it the same rank.

---

### Task 1: Pure signals, gates and the blocked fill rule

**Files:**
- Create: `scripts/bull_bear/entry_gates.py`
- Modify: `scripts/bull_bear/score_variants.py` (`book_picks` gets `blocked`)

**Interfaces:**
- Produces:
  - `score_variants.book_picks(ranked, held, top_n=..., keep_rank=..., blocked: frozenset[str] = frozenset()) -> list[str]`: unchanged when `blocked` is empty; new names in `blocked` are skipped; holdings never.
  - `reaction(closes: dict[date, float], spy: dict[date, float], sessions: list[date], r: date) -> tuple[date, float] | None`: (usable-from session, market-adjusted 3-session return); None when r-2, r+1 or the session after r+1 is not in `sessions`, or a close is missing.
  - `earnings_value(reactions: list[tuple[date, float]], day: date, sessions: list[date]) -> float | None`: the latest reaction with usable-from <= `day`; None if none, or if more than 126 sessions lie between its usable-from and `day`.
  - `Row = tuple[date, date, date, float]` (start, end, filed, value); `annual_rows(gaap: dict, tag: str) -> list[Row]`: USD rows of form 10-K with a 350-380-day period. `instant_rows(gaap, "Assets") -> list[tuple[date, date, float]]` (end, filed, value) from 10-K.
  - `gross_profit_rows(gaap: dict) -> list[Row]`: `GrossProfit` rows; for a period end with none, revenue (`Revenues`, else `RevenueFromContractWithCustomerExcludingAssessedTax`) minus cost (`CostOfRevenue`, else `CostOfGoodsAndServicesSold`) for rows with the same start, end and filed date.
  - `gpa_value(gross: list[Row], assets: list[tuple[date, date, float]], day: date) -> float | None`: among rows filed strictly before `day`, the latest period end E; gross = the latest-filed gross row for E, assets = the latest-filed assets row at E (filed before `day`); None if no assets, assets <= 0, or the gross row was filed more than 548 days before `day`.
  - `blocked(gate: str, values: dict[str, float | None]) -> frozenset[str]`: `earnings`: value < 0; `profitability`: the first `len(known) // 3` of the known values sorted by (value, symbol).
  - `self_check() -> None`

- [ ] **Step 1: Write `self_check()` with hand-worked asserts**, functions as stubs raising `NotImplementedError`:
  - `book_picks`: with `blocked=frozenset()` the existing score_variants asserts still hold; `book_picks(["A","B","C"], set(), top_n=2, blocked=frozenset({"A"})) == ["B","C"]`; a held `A` ranked 1 and blocked is kept: `book_picks(["A","B","C"], {"A"}, top_n=2, blocked=frozenset({"A","B"})) == ["A","C"]`; all blocked: `book_picks(["A","B"], set(), top_n=2, blocked=frozenset({"A","B"})) == []`.
  - `reaction`: sessions d1..d6, stock closes 100, 100, 100, 110, 110, 110 and SPY flat 100, r = d3 -> `(d5, 0.10)` (close(d4)/close(d1) - 1 = 0.10); SPY 100 -> 105 over d1..d4 gives `0.10 - 0.05`; r = d5 -> None (no session after r+1 = d6); a missing stock close at d4 -> None.
  - `earnings_value`: reactions `[(d5, -0.02), (d40, 0.03)]` -> `0.03` on d40, `-0.02` on d39, None on d4; a reaction usable from session 0 is None on session 127 and its value on session 126.
  - `annual_rows`/`gross_profit_rows`: a 10-K with a 365-day `GrossProfit` row and a 91-day row returns only the first; a period with only `Revenues` 100 and `CostOfRevenue` 60 on the same start, end and filed gives 40.
  - `gpa_value`: FY2010 gross 40 filed 2011-02-15, assets 200 at 2010-12-31 filed 2011-02-15, the same period restated to gross 50 in a 10-K filed 2012-02-15 (assets 250 at 2011-12-31 filed 2012-02-15, gross 2011 = 60): on 2011-02-15 -> None (filed not strictly before); on 2011-06-01 -> `0.2`; on 2012-03-01 -> `60/250`; on 2013-09-01 -> None (548 days); assets 0 -> None.
  - `blocked`: `earnings` on `{"A": -0.01, "B": 0.0, "C": None}` == `{"A"}`; `profitability` on six known values and one None blocks the two lowest; ties broken by symbol.
- [ ] **Step 2:** `uv run python scripts/bull_bear/entry_gates.py --self-check`. Expected: `NotImplementedError`.
- [ ] **Step 3:** Implement. `self_check()` also calls `score_variants.self_check()`.
- [ ] **Step 4:** Rerun: `self-check ok`, exit 0; `uv run python scripts/bull_bear/score_variants.py --self-check` ok; `uv run ruff check scripts/bull_bear && uv run ruff format --check scripts/bull_bear` exit 0; `uv run pyright scripts/bull_bear` 0 errors.
- [ ] **Step 5:** On branch `research/earnings-momentum` (never `main`): `git add -f` the spec, this plan, `scripts/bull_bear/entry_gates.py`, `scripts/bull_bear/score_variants.py`; commit `research(bull_bear): entry gates, pure signals and self-check`.

### Task 2: SEC fetch and per-symbol signal series

**Files:**
- Modify: `scripts/bull_bear/entry_gates.py`

**Interfaces:**
- Consumes: Task 1; `SecEdgarClient(user_agent, Path("cache/sec"))` (`ticker_to_cik`, `get_submissions_payload(cik, max_age_days=None)`, `get_submissions_page(name)`, `get_company_facts_payload(cik, max_age_days=None)`), `fundamentals.sec.older_pages(recent, date(2004, 8, 1))`, `earnings_drift.events.earnings_events`, `events.reaction_date`; `rank_only_stress.load_data`.
- Produces:
  - `fetch(symbols: list[str]) -> None` (`--fetch`): for every symbol, CIK, submissions with older pages, company facts; a failure (unmapped, HTTP) is printed and counted, never raised. Prints a progress line every 100 symbols and the final counts (mapped, submissions, facts, failures).
  - `signals(symbols, frames, sessions) -> dict[str, dict[str, object]]`: per symbol `{"reactions": list[tuple[date, float]], "gross": list[Row], "assets": list[...]}` from the cache only (a symbol not cached gets empty lists). Saved once to `cache/stress_2005/entry_gates_signals.pkl` and reused when present.

- [ ] **Step 1:** Add to `self_check()`: `signals` on a fake submissions payload (two 8-K item 2.02 rows, one 8-K/A, one 10-Q) and synthetic frames returns two reactions with the expected usable-from dates. Run `--self-check`: fails.
- [ ] **Step 2:** Implement `fetch` and `signals`; let `signals` take its payload reader as a parameter so the self-check passes fakes.
- [ ] **Step 3:** `--self-check` ok, lint/format/pyright as in Task 1.
- [ ] **Step 4:** Run `uv run python scripts/bull_bear/entry_gates.py --fetch` (background; SEC is rate-limited, expect tens of minutes). Expected: final counts with >= 2,400 mapped. Record the counts.
- [ ] **Step 5:** Commit `research(bull_bear): entry gates, SEC fetch and signal series`.

### Task 3: Gated book, both windows, report, verdict

**Files:**
- Modify: `scripts/bull_bear/entry_gates.py`

**Interfaces:**
- Consumes: Tasks 1-2; `score_variants.ranked`, `window_stats`, `passes`, `table`; `rank_only_stress.simulate`, `month_universe`, `SLEEVE`, `WEIGHTS[0]`, `TOP_N`.
- Produces:
  - `run(gate: str | None, first: date, last: date, data: tuple, series: dict) -> tuple[pd.Series, dict[str, float]]`: `gate=None` is the ungated book. Each rebalance: `order = ranked(day, held, ..., "today")`, values for every name in `order` (`earnings_value` / `gpa_value`), `blocked(gate, values)`, `book_picks(order, held, blocked=...)`. Stats: `window_stats` plus `turnover`, `held` (mean names in the book), `blocked` (share of the non-held names in `order[:35]` that are blocked), `missing` (share of `order` with no value).
  - `missing_by_year(...)` printed per gate: share of names with no value per calendar year.
  - `main()`: per gate, dev ungated vs gated on the gate's dev window, table, pass; holdout only for a passer, table; verdict line `<gate>: KEPT | FAILED development | FAILED holdout`. Saves `cache/stress_2005/entry_gates_equity.csv`.
- [ ] **Step 1:** Add to `self_check()`: on synthetic frames where the rank-1 stock has a negative reaction and is not held, the `earnings` book skips it and the ungated book holds it; where it is held, both books hold it. Run `--self-check`: fails.
- [ ] **Step 2:** Implement. Cache the ungated `ranked` order per day so the two books of a window share it.
- [ ] **Step 3:** `--self-check` ok, lint/format/pyright.
- [ ] **Step 4:** Run `uv run python scripts/bull_bear/entry_gates.py` once (background). Sanity: the ungated 2005-2015 book must give CAGR 0.5% and Sharpe 0.20 (score-variant step 2) within 0.1 point; if not, stop and report.
- [ ] **Step 5:** Append `## Results` to the spec (tables, missing-by-year, verdicts, caveats); write `docs/superpowers/replays/2026-11-28-earnings-profitability-gates.md` and a memory note; `TODO.md` only if the user asks. Commit `research(bull_bear): entry gates results`.

### Final passes (after the last task)

- [ ] `/simplify`, code-simplifier, stage, `/ponytail:ponytail-review` on the staged diff; fix every "Must fix".
- [ ] Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest --cov`, `uv run interrogate -v`; report the numbers (`interrogate` crashes on a clean `main`).
- [ ] No push, PR or merge without the user's go-ahead.

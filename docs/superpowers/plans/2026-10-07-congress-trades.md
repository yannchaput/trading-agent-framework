# congress_trades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new strategy `congress_trades`: a team of three agents (research → portfolio → trading) that rebuilds what
Nancy Pelosi owns today from the House Clerk's newest yearly report plus the trade reports (PTRs) since, turns it
into a target mix weighted by her holdings' value bands, and trades to that mix and checks every order filled. The
bot ticks once a day but trades only when she has filed something new. Paper, live and backtesting.

**Architecture:** Pure `congress/{ptr,annual,holdings}.py` (parsing, bands, reconstruction, weights); I/O
`congress/clerk_client.py` (the only Clerk/`httpx`/`pypdf` code, disk cache) and `congress/source.py` (the filings
known before today, parsed, and the new-filing diff); research tools in `agents/tools/congress.py` (clock-gated);
`strategies/congress_trades/` following `bill_ackman`: `HandoffRecorder` with three submit tools, a `CongressPipeline`
that short-circuits on "nothing new", a guarded `TradeDesk` (the only order code), a `StateStore`.

**Tech Stack:** Python 3.14, httpx, pypdf (new), pytest, `uv`, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-10-07-congress-trades-design.md` (read it first: it fixes the "owns today"
rule, the daily trigger, and each agent's validation).

**Patterns to copy (read before the task that uses them):** `strategies/bill_ackman/handoff.py` (recorder, submit
tools, validators), `pipeline.py` (`_run_stage`, abandonment, `ReviewLog`), `state.py` (atomic JSON store),
`parameters.py` (self-validating dataclass), `agent_bill_ackman.py` (wiring, `FatalStrategyError` on
`ConfigurationError`), `rebalancer.py` (open-order accounting, the smaller-of-buying-power-and-cash sizing);
`fundamentals/edgar_client.py` (cache, rate limit, error wrapping); `agents/tools/news.py` (tool bound to the
strategy, clock gate, `RunMemo`).

## Global Constraints

- Run everything through `uv` (`uv run pytest`, `uv run ruff check`, `uv run pyright`); never pip. Add the dependency
  with `uv add pypdf`.
- The automated tests never touch the network: `httpx.MockTransport`, hand-written fakes (`tests/fakes.py`), no
  `MagicMock`. `scripts/tests/` is the only place that touches the real Clerk.
- Money is `Decimal`; no new float boundary except the JSON/LangChain boundary of the handoff weights (as in
  `bill_ackman/handoff.py`). Pure modules do no I/O and read no clock.
- **A filing is known only when its filing DATE is strictly before `clock.now()`'s market date.** No
  `datetime.now()` / `time.sleep` in strategy, tool or desk code (`strategy.clock`, `strategy.sleep`); the client's
  rate limit is the one allowed `time.sleep`, as in `SecEdgarClient`.
- No raw `httpx`/`pypdf` exception escapes: wrap in `CongressDataError`.
- Files whose functions are tools (`handoff.py`, `desk.py` tool closures, `agents/tools/congress.py`) have no
  `from __future__ import annotations` (the agent layer reads real annotations).
- Tests that count runs pin their own `sleeptime`; the prompts never state the cadence.
- Commit after every task; each commit message ends with the `Co-Authored-By:` trailer of the model executing
  the task. Line length and style follow the surrounding code (ruff config in `pyproject.toml`).
- Out of scope: other politicians at once, Senate, options, shorting, paid APIs, a parking instrument.

## Review Focus

- **No look-ahead**: a filing dated today or later never reaches an agent or the reconstruction (Task 5 source test,
  Task 6 tool test), and the diff "new filings" uses the same known set.
- **"Nothing new" costs nothing and trades nothing**: no agent is run, no order sent, state unchanged (Task 10).
- **Pending trade is not a daily trade**: only the trading stage re-runs, at most `max_trade_retries` times, never
  re-sends an order that is already open (Tasks 9-10).
- **The tier rule is code, not prompt**: a higher-tier holding with a smaller weight is rejected by `submit_target` (Task 8).
- **A changed layout must not read as "no holdings/trades"** (Tasks 2-3 raise `CongressDataError`).
- **PTR trades dated on or before the yearly report's Dec 31 are ignored** (already inside the report) while PTRs
  filed before the report but trading after Dec 31 ARE applied (Task 4).
- **The desk cannot over-buy, short, or touch positions this strategy does not own** (Task 9).
- **The unverified layouts**: the checkpoint at the end of Task 5 re-bases Tasks 2-3 on real text; nothing ships before it.

---

## File map

| File | Change |
|---|---|
| `pyproject.toml`, `uv.lock` | `pypdf` |
| `src/trading_agent_framework/utils/errors.py` | `CongressDataError` |
| `src/trading_agent_framework/congress/{__init__,ptr,annual,holdings,clerk_client,source}.py` | new |
| `src/trading_agent_framework/agents/tools/congress.py`, `agents/tools/__init__.py` | research tools, lazy export |
| `src/trading_agent_framework/strategies/congress_trades/{__init__,parameters,state,handoff,desk,prompts,pipeline,agent_congress_trades}.py` | new |
| `src/trading_agent_framework/main.py` | builder + registry |
| `scripts/tests/smoke_congress_trades.py` | new, manual |
| `env/.env.example`, `README.md`, `CLAUDE.md` | `CONGRESS_USER_AGENT`, strategy section, architecture + gotchas |
| `tests/congress/`, `tests/agents/tools/test_congress_tools.py`, `tests/strategies/congress_trades/`, `tests/test_main.py` | new / extended |

---

## Task 1: Plumbing

- [ ] **Step 1:** `uv add pypdf`; confirm `uv run python -c "import pypdf"` (pure Python, no system binary).
- [ ] **Step 2:** `utils/errors.py`, after `FundamentalsNotFoundError`:

```python
class CongressDataError(TradingFrameworkError):
    """Raised when a congressional-disclosure lookup or parse fails (never a raw httpx/pypdf exception)."""
```

- [ ] **Step 3:** `env/.env.example`: `CONGRESS_USER_AGENT=` beside `SEC_EDGAR_USER_AGENT`, same comment style
  ("`<app or project name> <contact email>`"); one README env line mirroring the SEC one.
- [ ] **Step 4:** `uv run pytest -q`, `uv run ruff check` green; commit `chore: pypdf, CongressDataError, CONGRESS_USER_AGENT`.

---

## Task 2: `congress/ptr.py` -- index, member match, PTR rows (pure)

The year index `<YYYY>FD.xml` has `<Member>` elements: `Prefix, Last, First, Suffix, FilingType, StateDst, Year,
FilingDate (M/D/YYYY), DocID`. `P` = PTR, `O` = annual report (verified on the real index 2026-10-07: `C` is a candidate report, `A` an amendment, both ignored); the index of year Y lists the report of reporting year Y, filed the next May, and the PDF folder is the Year field.
PTR text, whitespace collapsed, is a run of rows like
`SP NVIDIA Corporation - Common Stock (NVDA) [ST] P 01/14/2025 01/14/2025 $250,001 - $500,000`
(owner code optional: `SP` spouse, `JT` joint, `DC` dependent; sides `P`, `S`, `S (partial)`, `E`).

- [ ] **Step 1: Failing tests** (`tests/congress/test_ptr.py`, synthetic fixture `tests/congress/fixtures/ptr_synthetic.txt`):
  - `test_parse_index_returns_ptr_annual_and_amendment_refs_with_kind` (other types dropped; `FilingRef.kind` in
    `{"ptr","annual"}`, `filed` a `date`, `year` the reporting year, `"1/5/2025"` parses).
  - `test_member_match_is_exact_on_normalized_first_and_last` ("Nancy Pelosi" matches; honorific/suffix ignored,
    case-insensitive; not `Paul Pelosi`, not `Pelosi-Smith`).
  - `test_malformed_index_raises_congress_data_error` (not XML; a `Member` with no `DocID`).
  - `test_parses_a_purchase_with_owner_and_range`, `test_sale_and_partial_sale_are_sells`,
    `test_options_bonds_exchanges_are_counted_not_parsed`, `test_open_ended_top_band_uses_the_floor_for_both_bounds`,
    `test_a_row_split_across_lines_parses`, `test_ticker_with_dot_or_dash_parses` (`BRK.B`, `BF-B`).
  - `test_text_with_type_tags_but_no_parseable_row_raises`, `test_text_with_no_type_tag_raises`,
    `test_blank_text_is_an_image_only_filing` (returns `None`).
- [ ] **Step 2:** run, expect FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement:**

```python
@dataclass(frozen=True, slots=True)
class FilingRef:
    doc_id: str; member: str; kind: str; filed: date; year: int   # kind: "ptr" | "annual"

@dataclass(frozen=True, slots=True)
class Transaction:
    doc_id: str; owner: str; ticker: str; asset_name: str; side: str     # owner: self|spouse|joint|dependent; side: buy|sell|sell_partial
    transaction_date: date; notification_date: date
    amount_low: Decimal; amount_high: Decimal; filed: date

@dataclass(frozen=True, slots=True)
class PtrParse:
    transactions: list[Transaction]; skipped_non_stock: int

def normalize_name(name: str) -> str: ...
def parse_index(xml_text: str) -> list[FilingRef]: ...                  # xml.etree.ElementTree; raises CongressDataError
def filings_for(refs, politician: str) -> list[FilingRef]: ...
def parse_ptr(text: str, ref: FilingRef) -> PtrParse | None: ...         # None = image-only
```

  Collapse whitespace, `finditer` one compiled stock-row pattern, count every `[XX]` tag and report
  `skipped_non_stock = tags - parsed`. `E` rows are dropped and counted. `side` keeps `sell` and `sell_partial`
  apart (Task 4 needs the difference).
- [ ] **Step 4:** PASS; ruff; commit `feat(congress): filing index and PTR parsing`.

---

## Task 3: `congress/annual.py` -- value bands, tiers, yearly report (pure)

Yearly report Schedule A rows (assumed shape): `SP Apple Inc. - Common Stock (AAPL) [ST] $5,000,001 - $25,000,000 Dividends $...`.
Bands (index = tier): `$1,001-$15,000`, `$15,001-$50,000`, `$50,001-$100,000`, `$100,001-$250,000`,
`$250,001-$500,000`, `$500,001-$1,000,000`, `$1,000,001-$5,000,000`, `$5,000,001-$25,000,000`,
`$25,000,001-$50,000,000`, `Over $50,000,000`; spouse/dependent assets may use `Over $1,000,000` (tier of the
$1,000,001-$5,000,000 band; flagged `open_ended`).

- [ ] **Step 1: Failing tests** (`tests/congress/test_annual.py`, fixture `annual_synthetic.txt`):
  - `test_bands_are_ordered_and_tier_of_a_value_is_the_band_that_contains_it` (including the exact edges, `$5,000,001` and `$25,000,000`).
  - `test_parse_annual_assets_reads_stock_rows_with_owner_and_value_band`.
  - `test_other_asset_tags_options_and_no_ticker_rows_are_counted` (`skipped_non_stock`).
  - `test_over_one_million_spouse_band_maps_to_the_one_to_five_million_tier`.
  - `test_period_end_is_december_31_of_the_reporting_year`.
  - `test_text_with_no_parseable_asset_and_no_other_tag_raises`; `test_blank_text_is_image_only`.
- [ ] **Step 2-3: Implement** `VALUE_BANDS: tuple[Band, ...]`, `tier_of(value: Decimal) -> int`,
  `AssetHolding(doc_id, owner, ticker, asset_name, value_low, value_high, tier)`, `parse_annual(text, ref) -> AnnualParse | None`,
  `period_end(ref) -> date`.
- [ ] **Step 4:** PASS; ruff; commit `feat(congress): yearly report parsing and value tiers`.

---

## Task 4: `congress/holdings.py` -- what she owns today (pure)

- [ ] **Step 1: Failing tests** (`tests/congress/test_holdings.py`):
  - `test_annual_assets_alone_give_holdings_at_the_band_midpoint`.
  - `test_ptr_buy_after_period_end_adds_the_range_midpoint`; `test_ptr_buy_dated_on_or_before_period_end_is_ignored`
    (disclosure lag: already inside the report); `test_ptr_filed_before_the_annual_report_but_traded_after_period_end_is_applied`.
  - `test_full_sale_sets_the_holding_to_zero_and_removes_it`; `test_partial_sale_subtracts_the_midpoint_with_floor_zero`.
  - `test_a_ptr_buy_of_a_ticker_not_in_the_annual_report_creates_a_holding`.
  - `test_tier_is_recomputed_from_the_estimated_value` (a buy lifts a holding from the $1M-$5M to the $5M-$25M tier).
  - `test_two_owners_of_the_same_ticker_add_up`.
  - `test_baseline_weights_are_proportional_to_midpoints_capped_and_sum_at_most_max_total` and
    `test_baseline_weight_cap_is_not_redistributed`.
  - `test_filings_dated_today_or_later_are_not_the_callers_input` is a SOURCE test (Task 5); here `reconstruct` takes the already-filtered inputs and has no date logic besides `period_end`.
- [ ] **Step 2-3: Implement:**

```python
@dataclass(frozen=True, slots=True)
class Holding:
    ticker: str; asset_name: str; value_low: Decimal; value_high: Decimal; tier: int; sources: tuple[str, ...]   # DocIDs

def reconstruct(assets: Sequence[AssetHolding], transactions: Sequence[Transaction], *, period_end: date) -> list[Holding]: ...
def baseline_weights(holdings: Sequence[Holding], *, max_total: Decimal, max_position: Decimal) -> dict[str, Decimal]: ...
```

  Transactions are applied in (transaction_date, doc_id) order. Weights quantized `ROUND_DOWN` to 4 places so the
  sum can never exceed `max_total`. A holding whose value reaches 0 is removed.
- [ ] **Step 4:** PASS; ruff; commit `feat(congress): reconstruct holdings and baseline weights`.

---

## Task 5: `ClerkClient`, `CongressSource`, smoke script, and the format CHECKPOINT

**Files:** `congress/clerk_client.py`, `congress/source.py`, `scripts/tests/smoke_congress_trades.py`, tests.

- [ ] **Step 1: Client tests** (`httpx.MockTransport`, `tmp_path`; style of `tests/fundamentals/test_edgar_client.py`):
  blank user agent -> `ConfigurationError`; the year index is read from the zip's `<YYYY>FD.xml` and cached; a past
  year's index fetched after that year is never refetched; the current year's index is refetched after a day **by
  `as_of`** (stale = `mtime.year <= year and is_stale(mtime, as_of, 1)`) and not within a day; PDF text (yearly
  `.../financial-pdfs/{year}/{doc_id}.pdf`, PTR `.../ptr-pdfs/{year}/{doc_id}.pdf`) is cached forever; an
  image-only PDF returns `""` and is cached; 500 / 404 / truncated zip / corrupt PDF -> `CongressDataError`; a stale
  index that cannot refresh is served with a warning; atomic cache writes.
- [ ] **Step 2: Implement** `ClerkClient(user_agent, cache_dir, *, min_request_interval_seconds=1.0, transport=None)`:
  `year_index_xml(year, as_of) -> str`, `filing_text(ref) -> str`. `pypdf` imported inside `filing_text` only.
- [ ] **Step 3: Source tests** (`FakeClerk`):
  - `test_known_filings_exclude_anything_filed_today_or_later` (the day before is included, `as_of`'s date is not).
  - `test_newest_annual_report_is_the_base_and_older_ones_are_ignored` (an amendment filed later wins).
  - `test_ptrs_are_those_with_transactions_after_the_annual_period_end` is NOT here (that is `holdings`); the source
    returns every known PTR filed after `period_end - 60 days` (covers the 45-day lag) and `holdings.reconstruct` filters by transaction date.
  - `test_no_annual_report_known_gives_an_error_not_an_empty_book` (`CongressDataError`).
  - `test_new_filings_is_known_doc_ids_minus_processed`.
  - `test_counts_image_only_filings_and_skipped_rows`; `test_a_filing_that_raises_aborts_the_lookup`.
  - `test_spans_two_calendar_years_and_loads_both_indexes`.
- [ ] **Step 4: Implement:**

```python
@dataclass(frozen=True, slots=True)
class KnownFilings:
    annual: AnnualParse; ptrs: list[PtrParse]; refs: list[FilingRef]     # refs: every known annual+PTR of the politician
    unparsed_filings: int; skipped_non_stock: int

class CongressSource:
    def __init__(self, client, politician: str) -> None: ...
    def known(self, as_of: datetime) -> KnownFilings: ...                      # filed strictly before as_of's market date
    def new_since(self, known: KnownFilings, processed: Collection[str]) -> list[FilingRef]: ...
```

- [ ] **Step 5: Smoke script** `scripts/tests/smoke_congress_trades.py` (read-only; header and messaging style of
  `smoke_quality_screen.py`): needs `CONGRESS_USER_AGENT`; lists the politician's newest annual report and the PTRs
  since, parses them, prints the reconstructed holdings with tiers and baseline weights and every skipped-row count,
  and writes each filing's raw extracted text to `<root>/cache/house_clerk/smoke/`.
- [ ] **Step 6:** PASS; ruff; pyright; commit `feat(congress): Clerk client, known-filings source and smoke script`.
- [ ] **Step 7 (CHECKPOINT -- needs a machine that can reach disclosures-clerk.house.gov): re-base on real text.**
  Run the smoke script. Copy one real yearly report's text and one real PTR's text into
  `tests/congress/fixtures/{annual_real,ptr_real}.txt` (trim to a few rows; strip nothing that a parser depends
  on), keep the synthetic fixtures, and fix `ptr.py` / `annual.py` until both sets pass. Confirm: the index
  `FilingType` codes (`P`/`O`; done 2026-10-07), `Year` meaning for an annual, the annual PDF URL, and that Pelosi's PDFs have a text
  layer. Record what differed in the commit message and in the spec's Risks section. **Tasks 6+ do not depend on
  the text layout and may proceed meanwhile, but nothing ships before this step.**

---

## Task 6: Research tools

**Files:** `agents/tools/congress.py`, `agents/tools/__init__.py` (lazy export), `tests/agents/tools/test_congress_tools.py`,
extend `tests/agents/tools/test_agents_tools_lazy_imports.py`.

- [ ] **Step 1: Failing tests** (fake source; `Strategy(FakeBroker(FakeClock(now)))`, frozen 2026-09-14 10:00 ET):
  - `test_list_filings_returns_only_filings_filed_before_today` (a source that ignores its bound is still filtered by the tool).
  - `test_list_filings_rows_are_lean` (`doc_id, kind, filed, year`; newest first).
  - `test_read_filing_refuses_a_doc_id_filed_today_or_later` and `..._an_unknown_doc_id` (`{"error": ...}`).
  - `test_read_filing_returns_stock_rows_for_a_ptr_and_for_the_annual_report` (lean keys; amounts as `int`; ISO dates).
  - `test_read_filing_reports_skipped_rows_and_image_only` ; `test_source_failure_is_an_error_payload`.
  - `test_identical_calls_in_one_run_are_reused` (`RunMemo`); one-line docstrings.
- [ ] **Step 2: Implement** `congress_research_tools(strategy, source) -> list[Callable[..., dict[str, Any]]]` with
  `list_filings()` and `read_filing(doc_id: str)`; the cutoff is `strategy.clock.now()` (the research-tool rule in CLAUDE.md).
- [ ] **Step 3:** lazy export beside `news_tools` / `fundamentals_tools` (it pulls `httpx` via the source).
- [ ] **Step 4:** PASS; ruff; pyright; commit `feat(agents): congress research tools`.

---

## Task 7: `CongressParams` and `StateStore`

**Files:** `strategies/congress_trades/{__init__,parameters,state}.py`, tests.

- [ ] **Step 1: Failing tests:** `CongressParams` defaults (`politician "Nancy Pelosi"`, `max_holdings 20`, `max_positions 15`,
  `max_total_weight 0.95`, `max_position_weight 0.15`, `min_weight 0.01`, `rebalance_band 0.01`, `min_trade_pct 0.005`,
  `max_trade_retries 3`, `max_consecutive_abandoned 3`, `reason_max_chars 300`, `agent_temperature 0.3`,
  `order_wait_seconds 60`); validation errors (blank politician, weights not in (0,1], `min_weight > max_position_weight`,
  `max_positions x min_weight` over `max_total_weight`, non-finite). `StateStore`: round-trip; atomic write; a missing, corrupt or
  wrong-`STATE_VERSION` file is an empty state; `wipe()`; `PendingTrade(target, days, orders_open)` round-trips; no
  method ever raises on an I/O problem. State fields: `processed: list[str]`, `holdings`, `target`, `traded: list[str]`,
  `pending_trade`, `abandoned_streak`, `last_run`. `RunLog` appends one JSON line per run to `runs.jsonl` in the run directory.
- [ ] **Step 2-3: Implement** (copy the shape of `bill_ackman/parameters.py` and `state.py`; path
  `data/congress_trades_state_<mode>.json`).
- [ ] **Step 4:** PASS; ruff; commit `feat(congress_trades): params and state`.

---

## Task 8: `handoff.py` -- three submit tools and the recorder

- [ ] **Step 1: Failing tests** (`tests/strategies/congress_trades/test_handoff.py`):
  - `submit_holdings`: ok; ticker not in any known filing; unreasoned deviation from the baseline; non-positive value; duplicate; more than `max_holdings`; a baseline holding silently omitted (must be listed with `drop: true` and a reason); reason too long.
  - `submit_target`: ok; ticker not in holdings; weight below `min_weight` / above `max_position_weight`; sum above `max_total_weight`;
    **`test_a_higher_tier_holding_with_a_smaller_weight_is_rejected`** and the equal-weight / capped-tie case accepted; more than `max_positions`;
    a dropped holding needs a reason; empty list allowed (hold cash).
  - `submit_trade_report`: ok when every desk order is final and named; rejects an order still working; rejects an order missing from the report;
    accepts a cancelled/rejected order only with a reason.
  - Recorder: wrong stage -> `{"error"}`; second valid submission -> `{"error": "already recorded..."}`; `last_error` kept.
- [ ] **Step 2-3: Implement** `HandoffRecorder(params)` with `expect_holdings(known_tickers, baseline)`,
  `expect_target(holdings)`, `expect_report(desk)`; `submit_tools(recorder)` returns the three closures
  (`submit_holdings`, `submit_target`, `submit_trade_report`). Validators are pure functions raising `HandoffError`.
- [ ] **Step 4:** PASS; ruff; pyright; commit `feat(congress_trades): handoff tools`.

---

## Task 9: `TradeDesk`

**Read first:** `bill_ackman/rebalancer.py` (open orders, sizing) and `earnings_drift/desk.py` (guarded tools, `traded` symbols, re-entrant lock).

- [ ] **Step 1: Failing tests** (`FakeBroker` + a `BacktestBroker` for the fill-timing ones):
  - `place_order` buys: refused for a symbol not in the target; refused beyond target weight (held + open buys + this order, within `rebalance_band`); refused above the smaller of `buying_power` and `cash` + same-run sell proceeds; accepted otherwise; fractional quantities floored.
  - sells: refused for a position not owned (not a target ticker and not in `state.traded`); refused above held minus open sells; a sell of a non-target owned position accepted; **sells submitted after a buy this run are refused** ("sells first").
  - never short / never margin (`buying_power` alone never used).
  - `traded` records the symbol BEFORE `submit_order` (an order the client raised on may still have reached the broker).
  - `check_orders`: reports status/filled/avg price of each desk order; waits with `order_wait_seconds`; a timeout reports `working`, not an error.
  - **`test_check_orders_in_a_daily_backtest`** pins what actually happens: if `wait_for_orders_execution` advances the
    simulated clock to the next bar the orders come back filled; otherwise they come back `working` and the audit
    path owns it. Write the test for the real behaviour found, and note it in the spec (see the spec's Backtesting section).
  - `audit(target)`: returns shortfalls when final positions differ from target by more than the band; orders already open are not shortfalls to re-send.
  - One re-entrant lock (tools may run on parallel LangGraph threads).
- [ ] **Step 2-3: Implement** `TradeDesk(strategy, params, state)` with `tools() -> [place_order, check_orders]` (closures with
  one-line docstrings and real annotations), `orders` (this run's), `audit(target) -> list[Shortfall]`, `reset()`.
- [ ] **Step 4:** PASS; ruff; pyright; commit `feat(congress_trades): trade desk`.

---

## Task 10: `CongressPipeline` and prompts

**Read first:** `bill_ackman/pipeline.py` (`_run_stage`, `ReviewAbandoned`, `retry_prompt`, outcome/streak).

- [ ] **Step 1: Failing tests** (a `FakeAgents` dict scripting each agent's submit call through the recorder; a `FakeSource`):
  - `test_nothing_new_and_nothing_pending_runs_no_agent_and_sends_no_order` (state unchanged, log line "nothing new", `completed=True`).
  - `test_first_run_treats_every_known_filing_as_new_and_builds_the_portfolio`.
  - `test_a_new_ptr_runs_research_portfolio_trading_in_order_and_updates_processed` (the research context carries `new_filings` and `baseline`).
  - `test_research_context_contains_only_filings_known_before_today`.
  - `test_processed_is_updated_only_when_the_whole_run_completes` (an abandoned research/portfolio stage leaves `processed` so tomorrow retries it; `abandoned_streak` up).
  - `test_stage_without_a_valid_submission_is_forced_once_then_abandons` and `..._backtest_raises_fatal_at_max_consecutive_abandoned`.
  - `test_trading_stage_with_unfilled_orders_sets_pending_trade` and `test_pending_trade_reruns_only_trading_up_to_max_trade_retries`
    (no research, no portfolio, stored target used; orders already open are not re-sent; gives up and logs after the limit; a quiet day with nothing pending still trades nothing).
  - `test_a_new_filing_while_a_trade_is_pending_replaces_the_pending_trade` (fresh research/portfolio wins).
  - `test_broker_error_in_trading_abandons_with_the_orders_already_sent_logged`.
  - `test_run_log_line_records_new_filings_holdings_target_orders_and_shortfalls`.
- [ ] **Step 2: Implement** `CongressPipeline.run() -> RunOutcome(completed, abandoned_streak)`; prompts in `prompts.py`:
  - Research system prompt: reconstruct only from known filings; trust the baseline arithmetic, resolve ambiguities, justify every deviation; stocks only; end with `submit_holdings`.
  - Portfolio system prompt: bigger value tier gets a share at least as big as a smaller tier; start from `baseline_weight`; drop only untradable names with a reason; end with `submit_target`.
  - Trading system prompt: sells first, then buys; size with the smaller of `buying_power` and `cash` + same-run sell proceeds; never short or use margin; `place_order` errors are final for that order (read, fix once, or skip); after the last order call `check_orders` until every order is final or the wait times out; end with `submit_trade_report` naming every order.
  - No cadence wording anywhere.
- [ ] **Step 3:** PASS; ruff; pyright; commit `feat(congress_trades): pipeline and prompts`.

---

## Task 11: Strategy, backtest wiring, registry

- [ ] **Step 1: Read** how `YahooBacktestData` handles an asset that was not in `preload_assets`. If it loads
  lazily on `bars()`, preload only the benchmark; if not, `run_backtesting` pre-scans the tickers of every
  annual+PTR filed in `[start - 1 year, end]` (one `CongressSource` lookup) and passes them as `preload_assets`. Add a test for whichever is true.
- [ ] **Step 2: Failing tests** (`tests/strategies/congress_trades/test_congress_strategy.py`; `_FakeHandle` style from `tests/strategies/test_news_builtin.py`):
  - `test_initialize_creates_three_agents_with_only_their_own_tools` (research: list/read/market data/`submit_holdings`; portfolio: `submit_target` only; trading: account tools, `get_last_price`, `place_order`, `check_orders`, `submit_trade_report`; none has memory/indicators).
  - `test_sleeptime_is_daily_and_iteration_starts_at_ten`.
  - `test_missing_congress_user_agent_or_llm_model_makes_the_strategy_refuse_to_start` (`ConfigurationError` -> `FatalStrategyError`, like `bill_ackman`).
  - `test_backtest_wipes_state_and_paper_keeps_it`.
  - `test_backtest_aborts_after_three_abandoned_runs_but_paper_carries_on`.
  - `test_a_quiet_day_makes_zero_agent_calls`.
  - `test_run_backtesting_defaults` (daily timestep, two-year window, SPY benchmark, budget, `agent_telemetry=True`).
- [ ] **Step 3: Implement** `CongressTradesStrategy` (shape of `agent_bill_ackman.py`): `initialize` builds the source, state,
  recorder, desk, three agents (`temperature=params.agent_temperature`; the portfolio agent and the research agent have no order tool) and the
  pipeline; `on_trading_iteration` runs it and raises `FatalStrategyError` per the abandonment rule; `_run_log_path()` -> `runs.jsonl`.
- [ ] **Step 4: Registry.** `main.py`: `_build_congress_trades(broker, mode)` returns the strategy (no universe file); register
  `"congress_trades"`. `tests/test_main.py`: the registry set gains it; a builder test as for `news_binary`.
- [ ] **Step 5:** `uv run pytest`, ruff, pyright; commit `feat(congress_trades): strategy and registry`.

---

## Task 12: Docs and final verification

- [ ] **Step 1: README** strategy section (what it does, the filing lag, the "trades only on a new filing" rule, env file
  `env/.env.congress_trades.<mode>`: `LLM_*`, `CONGRESS_USER_AGENT`, `ALPACA_DATA_*`, broker keys in paper/live).
- [ ] **Step 2: CLAUDE.md:** `congress/` and `agents/tools/congress.py` in Architecture (pure modules, `ClerkClient` the only Clerk/`httpx`/`pypdf` code,
  `CongressSource`); `congress_trades` in the `strategies/` bullet; the smoke command in Commands; gotchas: filings keyed on FILING date strictly before
  today (disclosure lag leaks the future otherwise); the "owns today" rule (annual base + PTRs with transaction date after its Dec 31); nothing-new
  is a code short-circuit (no LLM) and pending-trade re-runs the trading stage only; the trading agent HAS order tools, bounded by `TradeDesk`;
  estimates come from bands (options excluded); the real-layout verification date from the Task 5 checkpoint.
- [ ] **Step 3: Full run:** `uv run pytest`, `uv run ruff check`, `uv run pyright` clean.
- [ ] **Step 4 (needs network + an LLM):** smoke script, then `uv run agent congress_trades backtesting`; in
  `logs/congress_trades/backtesting/<run>/` confirm orders appear only on days after a new filing, no tool call returned a filing dated on/after its tick, and
  `settings.json` has per-agent telemetry totals.
- [ ] **Step 5:** commit `docs: congress_trades in README and CLAUDE.md`, then the review-fix commit the repo cadence expects.

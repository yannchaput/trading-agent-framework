# congress_trades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new strategy `congress_trades`: an LLM agent reads newly filed stock transactions of a configurable list of
House members (default Nancy Pelosi) from the House Clerk's Periodic Transaction Reports (PTRs), through a
clock-gated `search_congress_trades` tool, and trades the disclosed tickers weighted by the disclosed dollar
range. It runs in paper, live and backtesting.

**Architecture:** A pure `congress/ptr.py` (index parsing, name matching, transaction parsing, amount bands,
weights), an I/O `congress/clerk_client.py` (the only Clerk/`httpx`/`pypdf` code, disk-cached), a thin
`congress/source.py` that joins them into "trades filed in a window, known at `as_of`", the tool in
`agents/tools/congress.py` (cutoff from `strategy.clock.now()`), and `strategies/congress_trades/`
(`NewsBinaryStrategy`-shaped: `PrebuiltTools.all(self)` + the congress tool, inline prompt, daily tick).

**Tech Stack:** Python 3.14, httpx, pypdf (new), pytest, `uv`, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-10-07-congress-trades-design.md`

## Global Constraints

- Run everything through `uv` (`uv run pytest`, `uv run ruff check`, `uv run pyright`); never pip. Add the dependency
  with `uv add pypdf`.
- The automated tests never touch the network: fake `httpx.MockTransport`, hand-written fakes (`tests/fakes.py`),
  no `MagicMock`. `scripts/tests/` is the only place that touches the real Clerk.
- Money is `Decimal`; no new float boundary. Pure modules (`ptr.py`) do no I/O and read no clock.
- **A filing is known only when its filing DATE is strictly before `clock.now()`'s date** (market time). Trades
  are keyed on the filing date, never on the transaction date. No `datetime.now()` / `time.sleep` in
  strategy, tool or source code (`strategy.clock`, `strategy.sleep`); the client's rate limit is the one allowed
  `time.sleep`, as in `SecEdgarClient`.
- No raw `httpx`/`pypdf` exception escapes: wrap in `CongressDataError` (`utils/errors.py`).
- `agents/tools/congress.py` and `congress/ptr.py` have no `from __future__ import annotations` where the agent layer
  reads annotations (the tool function); follow `memory/tools.py`.
- Commit after every task; each commit message ends with the `Co-Authored-By:` trailer of the model executing
  the task. Line length and style follow the surrounding code (ruff config in `pyproject.toml`).
- Out of scope: Senate disclosures, options, shorting, paid APIs (Quiver), a code-enforcing desk, a decision retry
  turn like `news_binary`'s.

## Review Focus

- **No look-ahead.** A filing dated today or later never reaches the agent (Task 7 tool test with a frozen
  clock; Task 6 source test), and weights are computed from the same known set in a backtest and live.
- **A changed layout must not read as "no trades"**: a filing with text but no parseable row raises
  `CongressDataError`; only an image-only (no text) filing is skipped and counted (Task 3).
- **The unverified layout.** Task 1 produces a real fixture; Tasks 2-3 are written against a synthetic one and the
  checkpoint at the end of Task 3 re-bases them on the real text. Do not merge Tasks 4+ on a parser the
  checkpoint has not confirmed.
- **Weights bounded**: sum of ticker weights <= `max_total_weight`, each <= `max_position_weight`, sells carry none.
- A member name match is exact on normalized first+last name: "Nancy Pelosi" never matches "Pelosi, Paul"
  or "Nancy Pelosi-Smith" (Task 2).

---

## File map

| File | Change |
|---|---|
| `pyproject.toml`, `uv.lock` | `pypdf` dependency |
| `src/trading_agent_framework/utils/errors.py` | `CongressDataError` |
| `src/trading_agent_framework/congress/__init__.py` | new package |
| `src/trading_agent_framework/congress/ptr.py` | new, pure |
| `src/trading_agent_framework/congress/clerk_client.py` | new, I/O |
| `src/trading_agent_framework/congress/source.py` | new, `CongressSource` |
| `src/trading_agent_framework/agents/tools/congress.py` | new, `congress_trades_tools` |
| `src/trading_agent_framework/agents/tools/__init__.py` | lazy-export `congress_trades_tools` |
| `src/trading_agent_framework/strategies/congress_trades/{__init__,parameters,agent_congress_trades}.py` | new |
| `src/trading_agent_framework/main.py` | builder + `AGENT_STRATEGIES["congress_trades"]` |
| `scripts/tests/smoke_congress_trades.py` | new, manual |
| `env/.env.example`, `README.md`, `CLAUDE.md` | `CONGRESS_USER_AGENT`, strategy section, architecture + gotcha |
| `tests/congress/{test_ptr,test_clerk_client,test_source}.py`, `tests/agents/tools/test_congress_tools.py`, `tests/strategies/congress_trades/test_congress_strategy.py`, `tests/test_main.py` | new / extended |

---

## Task 1: Plumbing and the real-format smoke script

**Files:** `pyproject.toml`, `utils/errors.py`, `scripts/tests/smoke_congress_trades.py` (created last, in Task 6,
once `ClerkClient` exists; this task only prepares what it needs).

- [ ] **Step 1: Dependency.** `uv add pypdf`. Confirm `uv run python -c "import pypdf"` works (pure Python, no
  system binary).
- [ ] **Step 2: Error type.** In `utils/errors.py`, after `FundamentalsNotFoundError`:

```python
class CongressDataError(TradingFrameworkError):
    """Raised when a congressional-disclosure lookup or parse fails (never a raw httpx/pypdf exception)."""
```

- [ ] **Step 3: Env.** `env/.env.example`: add `CONGRESS_USER_AGENT=` next to `SEC_EDGAR_USER_AGENT` with the same
  "`<app or project name> <contact email>`" comment. README env section: one line mirroring the SEC one.
- [ ] **Step 4: Run `uv run pytest -q` and `uv run ruff check`** (unchanged green), commit
  `chore: pypdf dependency, CongressDataError, CONGRESS_USER_AGENT env`.

---

## Task 2: `ptr.py` -- index parsing and name matching (pure)

**Files:** `congress/__init__.py`, `congress/ptr.py`, `tests/congress/test_ptr.py`.

The year index `<YYYY>FD.xml` has `<Member>` elements with children `Prefix, Last, First, Suffix, FilingType,
StateDst, Year, FilingDate (M/D/YYYY), DocID`. PTRs are `FilingType == "P"`.

- [ ] **Step 1: Write failing tests** in `tests/congress/test_ptr.py`:
  - `test_parse_index_keeps_only_ptrs` (a `C` annual report and an `X` extension are dropped; `FilingRef.filed` is a
    `date`, `doc_id` a string, `year` an int).
  - `test_filed_date_parses_month_first_without_padding` (`"1/5/2025"`).
  - `test_name_match_is_exact_on_normalized_first_and_last` ("Nancy Pelosi" matches `Last=Pelosi, First=Nancy`; also
    `Prefix=Hon.` and a `Suffix` are ignored; does NOT match `First=Paul`, `Last=Pelosi-Smith`, nor a
    case/whitespace variant mismatch -> case-insensitive match is True).
  - `test_filings_for_returns_only_the_named_members` (two members requested).
  - `test_malformed_index_raises_congress_data_error` (not XML; a `Member` with no `DocID`).
- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** (`xml.etree.ElementTree`, which is stdlib and safe for this trusted-origin file; if
  the project later parses untrusted XML switch to `defusedxml`):

```python
@dataclass(frozen=True, slots=True)
class FilingRef:
    doc_id: str
    member: str          # "First Last", normalized display name
    filed: date
    year: int

def normalize_name(name: str) -> str: ...        # lower, strip honorifics ("hon.", "mr.", "mrs.", "ms.", "dr."), collapse spaces
def parse_index(xml_text: str) -> list[FilingRef]: ...   # PTRs only; raises CongressDataError on malformed input
def filings_for(refs: Iterable[FilingRef], politicians: Sequence[str]) -> list[FilingRef]: ...
```

  `member` is built from `First` + `Last` (no prefix/suffix); `filings_for` compares `normalize_name(ref.member)`
  to `normalize_name(requested)`.
- [ ] **Step 4: Run tests, expect PASS; `uv run ruff check`; commit** `feat(congress): PTR index parsing and member matching`.

---

## Task 3: `ptr.py` -- transaction parsing (pure) and the real-format checkpoint

**Files:** `congress/ptr.py`, `tests/congress/test_ptr.py`, `tests/congress/fixtures/ptr_sample.txt`.

PTR text, once whitespace is collapsed, is a run of rows like
`SP NVIDIA Corporation - Common Stock (NVDA) [ST] P 01/14/2025 01/14/2025 $250,001 - $500,000`
(owner code optional: `SP` spouse, `JT` joint, `DC` dependent child, none = filer; a row may add `S (partial)`,
`F S:` / `D:` description lines after it). **This is the assumed layout, not a verified one.**

- [ ] **Step 1: Fixture.** Write `tests/congress/fixtures/ptr_sample.txt` by hand in the shape above: a header
  block, a stock purchase (spouse), a stock sale, `S (partial)`, an option row (`[OP]`), a bond (`[GS]`, no ticker),
  an exchange (`E`), a `Over $50,000,000` band, a row split across two lines, and a trailing "Filing ID" footer.
- [ ] **Step 2: Failing tests:**
  - `test_parses_a_purchase_with_owner_and_range` (owner `spouse`, side `buy`, `Decimal` bounds, both dates).
  - `test_sale_and_partial_sale_are_sells`.
  - `test_options_bonds_and_exchanges_are_counted_not_parsed` (`ParseResult.skipped_non_stock` equals their count).
  - `test_open_ended_top_band_uses_the_floor_for_both_bounds`.
  - `test_a_row_split_across_lines_parses`.
  - `test_text_with_no_parseable_stock_row_and_no_other_row_raises` (a changed layout is an error).
  - `test_empty_text_is_an_image_only_filing` (`parse_transactions("")` returns `None`).
  - `test_ticker_with_dot_or_dash_parses` (`BRK.B`, `BF-B`).
- [ ] **Step 3: Implement:**

```python
@dataclass(frozen=True, slots=True)
class Transaction:
    doc_id: str; member: str; owner: str          # "self" | "spouse" | "joint" | "dependent"
    ticker: str; asset_name: str; side: str       # "buy" | "sell"
    transaction_date: date; notification_date: date
    amount_low: Decimal; amount_high: Decimal; filed: date

@dataclass(frozen=True, slots=True)
class ParseResult:
    transactions: list[Transaction]
    skipped_non_stock: int

def parse_transactions(text: str, ref: FilingRef) -> ParseResult | None: ...   # None = image-only (blank text)
```

  Collapse whitespace first, then `finditer` one compiled pattern for stock rows; count every `[XX]` type tag and
  report `skipped_non_stock = tags - parsed stock rows`. Side `P` -> buy, `S` / `S (partial)` -> sell, `E`
  dropped (counted). Non-blank text with no type tag at all raises `CongressDataError`; tags present but zero stock
  rows is fine (a filing of bonds only).
- [ ] **Step 4: Tests PASS; ruff; commit** `feat(congress): parse PTR transactions`.
- [ ] **Step 5 (CHECKPOINT, needs a machine that can reach disclosures-clerk.house.gov): re-base on real text.**
  After Task 6 creates the smoke script, run it, save one real Pelosi PTR's extracted text over
  `ptr_sample.txt` (keep the synthetic cases as `ptr_synthetic.txt`), and fix `parse_transactions` until both
  fixtures pass. Record in the commit message what differed. Tasks 4+ do not depend on the text layout and may
  proceed meanwhile, but nothing ships before this step.

---

## Task 4: `ptr.py` -- suggested weights (pure)

**Files:** `congress/ptr.py`, `tests/congress/test_ptr.py`.

- [ ] **Step 1: Failing tests:**
  - `test_weights_are_proportional_to_range_midpoints_and_sum_to_max_total` (two buys, 3:1 midpoints, caps not binding
    -> 0.675 / 0.225 with `max_total=0.9`).
  - `test_each_weight_is_capped_at_max_position` (one buy, `max_position=0.15` -> 0.15, total below max).
  - `test_buys_of_the_same_ticker_add_up` (one weight per ticker).
  - `test_sells_carry_no_weight_and_do_not_dilute_buys`.
  - `test_no_buys_gives_an_empty_mapping`.
- [ ] **Step 2: Implement:**

```python
def suggested_weights(transactions: Iterable[Transaction], *, max_total: Decimal, max_position: Decimal) -> dict[str, Decimal]:
    """Ticker -> portfolio weight: its share of the window's buy midpoints times max_total, capped at max_position."""
```

  Quantize to 4 places with `ROUND_DOWN` so the sum can never exceed `max_total`. The cap is not redistributed
  (spare weight stays in cash, deliberately).
- [ ] **Step 3: PASS; ruff; commit** `feat(congress): suggested position weights`.

---

## Task 5: `CongressParams`

**Files:** `strategies/congress_trades/{__init__,parameters}.py`, `tests/strategies/congress_trades/test_congress_params.py`.

- [ ] **Step 1: Failing tests:** defaults (`politicians == ("Nancy Pelosi",)`, `lookback_days == 45`,
  `max_total_weight == Decimal("0.9")`, `max_position_weight == Decimal("0.15")`, `result_limit == 20`);
  `__post_init__` rejects an empty politician list, `lookback_days < 1`, a weight outside (0, 1], and
  `max_position_weight > max_total_weight` with `ConfigurationError`.
- [ ] **Step 2: Implement** as a frozen dataclass validating itself in `__post_init__` (like `ScreenParams` /
  `DriftParams`; copy their style).
- [ ] **Step 3: PASS; ruff; commit** `feat(congress_trades): CongressParams`.

---

## Task 6: `ClerkClient`, `CongressSource` and the smoke script

**Files:** `congress/clerk_client.py`, `congress/source.py`, `scripts/tests/smoke_congress_trades.py`,
`tests/congress/test_clerk_client.py`, `tests/congress/test_source.py`.

- [ ] **Step 1: Client tests** (`httpx.MockTransport`, `tmp_path` cache; copy the fixture style of
  `tests/fundamentals/test_edgar_client.py`):
  - `test_blank_user_agent_raises_configuration_error`.
  - `test_year_index_is_fetched_from_the_zip_and_cached` (a zip built in memory with `<YYYY>FD.xml`; second call makes no request).
  - `test_past_year_index_fetched_after_that_year_is_never_refetched` (`mtime.year > year`, backtest `as_of`).
  - `test_current_year_index_is_refetched_after_a_day_by_as_of` and `..._not_within_a_day`
    (rule: stale = `mtime.year <= year and freshness.is_stale(mtime, as_of, 1)`; `as_of` is the caller's clock, never the wall clock).
  - `test_ptr_text_is_cached_forever` (filings are immutable; amendments are new DocIDs).
  - `test_image_only_pdf_returns_empty_text_and_is_cached` (a PDF with no text layer; build a minimal one with `pypdf.PdfWriter().add_blank_page`).
  - `test_http_and_pdf_failures_are_wrapped_in_congress_data_error` (500, truncated zip, corrupt PDF; a 404 PDF too).
  - `test_stale_index_that_cannot_refresh_is_served_with_a_warning` (same degrade rule as `SecEdgarClient.get_json`).
- [ ] **Step 2: Implement** `ClerkClient(user_agent, cache_dir, *, min_request_interval_seconds=1.0, transport=None)`:
  `year_index_xml(year, as_of) -> str`, `ptr_text(ref) -> str`. URLs:
  `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip` and
  `https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf`. Cache layout
  `cache_dir/index/{year}FD.xml` and `cache_dir/ptr/{year}/{doc_id}.txt`; atomic writes (write to a temp name, then
  `replace`) so an interrupted download is not a cache hit. `pypdf` imported inside `ptr_text` only.
- [ ] **Step 3: Source tests** (`FakeClerk` returning a canned index + texts):
  - `test_trades_between_dates_known_before_as_of` -- window `[start, end)` over FILING date, `end` exclusive; a
    filing dated on `as_of`'s date is excluded, the day before is included.
  - `test_spans_two_years_and_loads_both_indexes`.
  - `test_counts_image_only_filings_and_skipped_rows` (`unparsed_filings`, `skipped_non_stock`).
  - `test_only_requested_members` and `test_newest_first_order` (ties by `doc_id`).
  - `test_a_filing_that_raises_aborts_the_lookup` (a layout change surfaces, not "no trades").
- [ ] **Step 4: Implement:**

```python
@dataclass(frozen=True, slots=True)
class TradesResult:
    transactions: list[Transaction]      # newest filing first
    unparsed_filings: int
    skipped_non_stock: int

class CongressSource:
    def __init__(self, client: ClerkClientLike) -> None: ...
    def trades(self, politicians: Sequence[str], *, start: date, end: date, as_of: datetime) -> TradesResult:
        """Stock transactions filed in [start, end), end exclusive. Caller passes end = as_of's market date."""
```

- [ ] **Step 5: Smoke script** `scripts/tests/smoke_congress_trades.py` (read-only; follow `smoke_quality_screen.py`'s
  header and `SystemExit` messaging): reads `CONGRESS_USER_AGENT`, builds the client in `<root>/cache/house_clerk`,
  prints Pelosi's last 5 PTRs (filing date, DocID), parses the newest, prints the parsed rows plus
  `skipped_non_stock` / image-only status, and writes the newest filing's raw extracted text to
  `<root>/cache/house_clerk/last_smoke_ptr.txt` for Task 3's checkpoint. Add it to the CLAUDE.md commands list in Task 10.
- [ ] **Step 6: PASS; ruff; pyright; commit** `feat(congress): Clerk client, trades source and smoke script`.

---

## Task 7: `search_congress_trades` tool

**Files:** `agents/tools/congress.py`, `agents/tools/__init__.py` (lazy map entry), `tests/agents/tools/test_congress_tools.py`,
extend `tests/agents/tools/test_agents_tools_lazy_imports.py`.

- [ ] **Step 1: Failing tests** (fake source records its call arguments; `Strategy(FakeBroker(FakeClock(now)))`):
  - `test_window_ends_the_day_before_the_clock_date` (frozen 2026-09-14 10:00 ET; the source is asked for
    `end == date(2026, 9, 14)` exclusive, `start == end - days`, `as_of == clock.now()`).
  - `test_never_returns_a_transaction_filed_today_or_later` (fake source ignoring its bounds; the tool still filters by `filed < today`).
  - `test_rows_are_lean_and_carry_the_ticker_weight` (keys exactly
    `ticker, side, member, owner, filed, traded, amount_low, amount_high, suggested_weight`; amounts as `int`,
    dates ISO; sells have `suggested_weight` `None`).
  - `test_weights_use_the_whole_window_not_the_row_cap` (more buys than `result_limit`: weights still sum correctly).
  - `test_ticker_and_politician_filters`; a `politician` not in `params.politicians` -> `{"error": ...}` (the agent cannot widen the list).
  - `test_days_is_clamped_to_1_and_lookback_days`.
  - `test_source_failure_returns_an_error_payload` (`CongressDataError` -> `{"error": ...}`, not raised).
  - `test_identical_search_in_one_run_is_reused` (`RunMemo`, like `search_news`).
  - `test_tool_has_a_one_line_docstring` and the payload has `count`, `unparsed_filings`, `skipped_non_stock`.
- [ ] **Step 2: Implement** `congress_trades_tools(strategy, source, params) -> list[Callable[..., dict[str, Any]]]`
  (no `from __future__ import annotations`). One tool, `search_congress_trades(days: int = 45, ticker: str = "", politician: str = "")`.
  `today = strategy.clock.now().astimezone(MARKET_TZ).date()`. Compute weights over the window's buys
  (`ptr.suggested_weights`) before applying the ticker/politician filters and `result_limit`, so the weight
  of a name does not depend on how the agent filtered. Amounts serialized as `int` (tokens), weights as `float`
  rounded to 4 places (the JSON boundary for LangChain; the computation stays `Decimal`).
- [ ] **Step 3: Lazy export:** add `congress_trades_tools` to `agents/tools/__init__.py`'s `TYPE_CHECKING` import,
  `__all__` and `_LAZY` map (it imports `congress.source`, which pulls `httpx`); extend the lazy-imports test the
  same way as `news_tools` / `fundamentals_tools`.
- [ ] **Step 4: PASS; ruff; pyright; commit** `feat(agents): search_congress_trades tool`.

---

## Task 8: `CongressTradesStrategy`

**Files:** `strategies/congress_trades/agent_congress_trades.py`, `__init__.py`,
`tests/strategies/congress_trades/test_congress_strategy.py`.

- [ ] **Step 1: Read first** (no code yet): `strategies/news_builtin/agent_news_binary.py` (shape to copy: `_FakeHandle`
  test style in `tests/strategies/test_news_builtin.py`) and `strategies/earnings_drift/agent_earnings_drift.py`'s
  `run_backtesting` (daily timestep, data source and how dynamically chosen symbols are loaded in a backtest).
  Mirror the latter's data source and `preload_assets` handling: this strategy's tickers are unknown up front.
  If a symbol the agent trades has no preloaded data and the data source cannot load it lazily, STOP and add a
  short design note to the spec instead of improvising.
- [ ] **Step 2: Failing tests:**
  - `test_initialize_creates_one_agent_with_prebuilt_and_congress_tools` (tool names include `search_congress_trades`,
    `submit_order`, `remember_decision`; no `search_news`).
  - `test_sleeptime_is_daily_and_iteration_starts_at_ten` (`sleeptime == "1D"`, `iteration_start_time == time(10, 0)`).
  - `test_each_tick_runs_the_agent_once_with_datetime_and_portfolio_context`.
  - `test_missing_congress_user_agent_makes_the_strategy_refuse_to_start` (`ConfigurationError` in `initialize` ->
    `FatalStrategyError`, like `bill_ackman`; no network).
  - `test_backtest_aborts_after_three_consecutive_agent_failures` and `test_paper_mode_does_not_abort`
    (`MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS` reused from `agent_news_binary`; import it, do not redefine).
  - `test_warns_when_no_decision_was_recorded` (a log warning; **no retry turn**).
  - `test_system_prompt_states_the_sizing_and_sell_rules` (contains the `suggested_weight`, "SMALLER of
    `buying_power`", "never short", and "nothing new filed -> do nothing" rules).
- [ ] **Step 3: Implement.** Class attributes: `AGENT_NAME = "congress_trades_trader"`, `TASK_PROMPT`, `parameters`
  (`backtesting_start/end`: two years, `benchmark_symbol "SPY"`, `warmup_trading_days`, `budget`), constructor takes
  `settings: CongressParams = CongressParams()` and an optional `source` (tests inject a fake; default built in
  `initialize` from `CONGRESS_USER_AGENT` + `find_project_root() / "cache" / "house_clerk"`).
  Reuse `NewsBinaryStrategy._portfolio_snapshot`'s logic: **extract it to a shared helper**
  (`strategies/news_builtin/portfolio.py::portfolio_snapshot(strategy)`) and call it from both strategies, rather
  than copy it; `_track_regime` stays in `news_binary`. All of `tests/strategies/test_news_builtin.py` must stay green unmodified.
  System prompt, in this order: allowed instruments (any US common stock the filings name; never short, margin,
  options, FOREX); English-only; workflow (1. `search_memory`; 2. `search_congress_trades` with the default lookback;
  3. for each ticker with a buy and not held: buy at `suggested_weight` x `portfolio_value` using the smaller-of
  sizing rule from news_binary's prompt; 4. for a held ticker with a disclosed sale: sell it; ignore sells of names
  not held; 5. nothing new since the last recorded decision -> trade nothing, still `remember_decision` once; 6. always `get_positions` and
  `get_account_balance` before trading; sells before buys; honour `{"error"}` returns; `remember_decision` exactly once, then stop).
  "New" means a `filed` date later than the newest filing in the last `remember_decision` (the prompt tells the agent
  to put that date in its decision text).
- [ ] **Step 4: PASS; existing news_binary tests PASS; ruff; pyright; commit** `feat(congress_trades): strategy`.

---

## Task 9: Registry, builder and docs

**Files:** `main.py`, `tests/test_main.py`, `README.md`, `env/.env.example`.

- [ ] **Step 1: Failing tests:** `test_registry_lists_the_strategies` gains `"congress_trades"`;
  `test_congress_trades_builder_returns_the_strategy` (`main_module._build_congress_trades(broker, TradingMode.BACKTESTING)`
  is a `CongressTradesStrategy` with `is_backtesting`; no universe file is needed).
- [ ] **Step 2: Implement** `_build_congress_trades(broker, mode) -> Strategy | None` returning
  `CongressTradesStrategy(broker=broker, mode=mode)`; register it.
- [ ] **Step 3: README** section next to the other strategies: what it does, the 45-day lag caveat, env file
  `env/.env.congress_trades.<mode>` with `LLM_*`, `CONGRESS_USER_AGENT`, `ALPACA_DATA_*` (backtest bars) and the broker keys in
  paper/live.
- [ ] **Step 4: `uv run pytest`, ruff, pyright; commit** `feat(congress_trades): register the strategy`.

---

## Task 10: CLAUDE.md, final verification

**Files:** `CLAUDE.md`.

- [ ] **Step 1: CLAUDE.md:** add `congress/` and `agents/tools/congress.py` to Architecture (pure `ptr.py`,
  `ClerkClient` the only module importing `httpx`/`pypdf` for the Clerk, `CongressSource`); add
  `congress_trades` to the `strategies/` bullet; add the `smoke_congress_trades.py` command; add one gotcha:
  **congress trades are keyed on the FILING date** (strictly before `clock.now()`'s date; disclosure lags the trade by up to 45
  days, so trade-date keying leaks the future), `suggested_weight` is advisory, image-only filings are skipped and
  counted, and the Clerk layout was verified against a real filing on `<date>` (fill in at the Task 3 checkpoint).
- [ ] **Step 2: Full run:** `uv run pytest`, `uv run ruff check`, `uv run pyright` all clean.
- [ ] **Step 3 (needs network + an LLM):** `uv run python scripts/tests/smoke_congress_trades.py`, then
  `uv run agent congress_trades backtesting`; confirm in `logs/congress_trades/backtesting/<run>/` that no tool call
  returned a row filed on or after its tick's date, and that `settings.json` agent totals exist.
- [ ] **Step 4: Commit** `docs: congress_trades in CLAUDE.md`, then the review-fix commit the repo cadence expects.

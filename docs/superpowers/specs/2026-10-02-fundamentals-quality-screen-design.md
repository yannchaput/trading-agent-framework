# Fundamentals quality screen

Date: 2026-10-02 · Branch: `feature/fundamentals-quality-screen` · First of two specs for the Bill Ackman
portfolio strategy (the second covers the three-agent strategy that consumes this screen)

> **Note (moved):** the screen described here now lives in `src/trading_agent_framework/strategies/bill_ackman/screen/`
> (`quality.py`, `screen.py`, `splits.py`, `annual_store.py`, and `annual_figures.py`, which holds the annual-figures
> reduction that used to be in `fundamentals/sec.py`). `fundamentals/` keeps the SEC client and the generic SEC
> translation. See `docs/superpowers/specs/2026-10-02-bill-ackman-strategy-design.md` §1.1.

## Problem

The Bill Ackman strategy (lumibot's
[example](https://lumibot.lumiwealth.com/agents_example_bill_ackman_portfolio_ai_trading_bot.html)) has a
researcher agent pick "simple, predictable companies that make lots of cash and trade at a good price". The
lumibot example gives it a fixed list of 10 tickers. Here the researcher picks from the cross_momentum
universe file (1,200 symbols), which no LLM can read company by company. Code must first cut the universe to
a short list of candidates that match the criteria, with the numbers that justify each one.

Nothing in the framework does that today: `fundamentals/` serves one company at a time to an agent tool, has
no cash-flow fields, and its cache never expires.

## Goal

A code-only screen, with no LLM and no strategy knowledge, that answers: *of these symbols, which were
simple, predictable, cash-generative, lightly indebted and reasonably priced on date D, using only what was
public on D?*

```python
screen = QualityScreen(store, splits, params=ScreenParams())
result = screen.run(symbols, as_of=strategy.clock.now(), price_of=strategy.get_last_price)
result.candidates   # ranked list[Candidate], best first, at most params.top_n
result.rejections   # dict[str, str]: symbol -> reason
```

Success: the test suite and `ruff check` pass; the manual smoke script runs the real screen on about 20
symbols and prints a ranked list with a reason for every rejected symbol; a second run the same day makes no
SEC request.

## Non-goals

- **No strategy, no agents, no fact-sheet wording.** The second spec turns `Candidate` into what agents read.
- **No batch command.** Data is fetched lazily on first use.
- **No quarterly or trailing-twelve-month figures.** Annual (10-K) figures only: quarterly cash-flow values in
  XBRL are year-to-date cumulatives and error-prone to unwind.
- **No IFRS support.** A 20-F filer with no `us-gaap` facts is rejected as `no_data`.
- **No fix for the agent drill-down tools' cache.** `get_income_statement` and the other fundamentals tools
  keep reading the never-expiring raw cache. The second spec handles that, where those tools are used.
- **No dashboard view.**

## 1. Modules

All under `fundamentals/`.

| Module | Kind | Responsibility |
|---|---|---|
| `sec.py` (extended) | pure | Cash-flow, debt and share-count tag lists; `annual_figures(payload)` reduces a company-facts payload to the rows the screen needs; `parse_sic(submissions_payload)`. |
| `quality.py` (new) | pure | `ScreenParams`, `Candidate`, `ScreenResult`, the gates, the score and the ranking. |
| `splits.py` (new) | pure + I/O | Pure `restate_shares(...)`; `SplitHistory`, the only module that imports `yfinance` for splits (lazily). |
| `annual_store.py` (new) | I/O | `AnnualFiguresStore`: lazy per-symbol fetch through `SecEdgarClient`, reduced on-disk cache, freshness. |
| `screen.py` (new) | wiring | `QualityScreen.run(...)`: asks the store and the split history for data, calls the pure functions, logs the summary. |
| `edgar_client.py` (extended) | I/O | An uncached fetch path, so the store can read a company-facts payload without writing 4 MB to disk. |

`quality.py` and the additions to `sec.py` follow the same purity rule as `brokers/alpaca/orders.py`: no I/O,
no clock, no state. The screen never receives a `Strategy` or a broker; `as_of` and `price_of` are passed in,
as `MemoryStore` takes `now`.

## 2. Data

### 2.1 What is kept per company

`annual_figures(payload)` returns, and the store writes to `cache/sec/annual/CIK<10 digits>.json`:

```json
{
  "cik": "0000320193",
  "schema": 2,
  "fetched_at": "2026-10-02T14:03:11+00:00",
  "status": "ok",
  "sic": 3571,
  "flows": [{"field": "revenue", "start": "2024-09-29", "end": "2025-09-27", "value": 416161000000, "filed": "2025-10-31"}],
  "balances": [{"field": "debt", "end": "2025-09-27", "value": 90678000000, "filed": "2025-10-31"}],
  "shares": [{"end": "2026-07-17", "value": 14594180000, "filed": "2026-07-31", "kind": "cover"}]
}
```

- **`flows`**: revenue, operating income, operating cash flow, capex. Only facts from a 10-K or 10-K/A whose
  period lasts 350 to 380 days.
- **`balances`**: debt and cash. Only facts from a 10-K or 10-K/A.
- **`shares`**: share counts from 10-K and 10-Q filings. Each row is tagged `"kind": "cover"` (the `dei`
  cover-page count) or `"kind": "weighted"` (the `us-gaap` weighted-average diluted count); cover rows come
  first. A 10-Q reports a 3-month and a year-to-date weighted count under one (period end, filing): the
  shortest period is kept, whatever the payload's row order.
- Every version of a figure is kept (a later 10-K restates earlier years), each with its own `filed` date, so
  the as-of selection in §3.1 can be exact.
- `sic` is absent until the sector gate first asks for it (§3.2).
- `status` is `"ok"` or `"absent"` (§5).
- `schema` is `annual_store.SCHEMA_VERSION`. It must be bumped whenever the tag lists or the row shape in
  `sec.py` change: a record fetched today is fresh for every past date in a backtest (§2.4), so without a bump
  an old cache would hide a tag fix. A record with a missing or different `schema` is a cache miss (§5).

A reduced file is a few KB. The raw payload (about 4 MB) is never written by the store.

### 2.2 Tags

Each field tries its tags in order and uses the first one present for a given period and filing, except revenue (below).

| Field | Tags |
|---|---|
| revenue | the LARGEST value within one filing among `RevenueFromContractWithCustomerExcludingAssessedTax`, `Revenues`, `SalesRevenueNet`, `RegulatedAndUnregulatedOperatingRevenue` (utilities such as NEE) and `RevenueFromContractWithCustomerIncludingAssessedTax`; see below |
| operating_income | `OperatingIncomeLoss` |
| operating_cash_flow | `NetCashProvidedByUsedInOperatingActivities`, `NetCashProvidedByUsedInOperatingActivitiesContinuingOperations` |
| capex | `PaymentsToAcquirePropertyPlantAndEquipment`, `PaymentsToAcquireProductiveAssets` |
| debt | within one filing the first that exists: `LongTermDebt`; `LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities`; `LongTermDebtNoncurrent` + `LongTermDebtCurrent`; `LongTermDebtAndCapitalLeaseObligations` + `LongTermDebtAndCapitalLeaseObligationsCurrent` (the current part counts as 0 when absent); `DebtLongtermAndShorttermCombinedAmount`. Commercial paper and short-term borrowings are not added (known limitation: filers disagree on whether the long-term tags include them) |
| cash | existing `BALANCE_SHEET_TAGS["cash"]` |
| shares | `dei:EntityCommonStockSharesOutstanding` and `us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding`, both kept |

**Revenue takes the largest value, not the first tag.** A filer can tag only part of its top line with the
ASC 606 tag: URI (equipment rentals sit outside it) reports `RevenueFromContractWithCustomerExcludingAssessedTax`
= 3.7 B and `Revenues` = 16.1 B for FY2025 in one filing, so the first-tag rule gave margins four times too
high. Checked on real payloads (URI and the 19 other smoke symbols, every filing and year): the tags differ
for URI (15 of 17 filing-years with two tags), COST (`SalesRevenueNet`, net sales without memberships, below
`Revenues`) and NEE (the contract tag below `RegulatedAndUnregulatedOperatingRevenue`), and in every one of
those the larger value is the total; everywhere else only one tag exists or the tags agree, so those
symbols are unchanged. Within one (period end, filing) the largest value wins, so two tags never blend;
across filings every version is still kept. The agent tools' `INCOME_STATEMENT_TAGS` are untouched.

`shares` is the one field that keeps every tag's rows instead of the first tag per period: the two counts
have different dates, and a multi-class company's per-class cover-page counts are dimensioned facts that the
company-facts API omits, so its only usable count is the weighted average. The cover-page count lives in the
`dei` namespace; the existing code reads only `us-gaap`.

### 2.3 Fiscal years

A fiscal year is identified by its period **end date**, never by XBRL's `fy` field: each 10-K repeats three
years of figures, all tagged with the filing's own `fy` (verified on Apple's payload).

### 2.4 Freshness

Two rules, one per kind of data.

**The annual store (SEC figures) uses the `as_of` rule.** `fetched_at` is stamped from an injected
`wall_clock` callable (the real fetch time). A file is stale when

```
fetched_at < as_of - params.max_age_days        (default 30)
```

`as_of` is the caller's clock, so one rule covers every mode: in a backtest `as_of` is simulated and a file
fetched today is fresh for every past date; in paper/live the file is refetched monthly. No wall-clock value
is compared against data, and the data cutoff itself is always `as_of`. This is right because the figures are
point-in-time: every version of a figure carries its own `filed` date, so one fetch answers every past date.

**Split history uses the wall clock, not `as_of`.** Split history is not point-in-time data: it must match
the basis of today's split-adjusted prices, a wall-clock fact. An entry is stale when

```
fetched_at < wall_clock() - params.split_max_age_days        (default 1)
```

with `wall_clock` the injected timezone-aware callable (UTC `now` by default). Judging it by `as_of` would let
a backtest keep a pre-split entry forever, while the prices fetched today are already adjusted for the newer
split: the restated market cap would miss that split and the name would rank first. The cost is one extra
Yahoo split fetch per survivor per day, including in backtests.

For both, a stale entry is refetched on the next `run`. If that refetch fails on a transport error, the stale
entry is used and a warning is logged.

### 2.5 Cost

The first run over 1,200 symbols downloads about 5 GB once (10 to 20 minutes at the client's rate limit) and
leaves a few MB on disk. Later runs read the reduced files once per process and keep them in memory.

## 3. The screen

### 3.1 As-of selection

A row is **known** on `as_of` when its `filed` date is strictly before `as_of`'s date. SEC gives a filing
date without a time, and annual reports are often filed after the close, so a row filed today is treated as
known from tomorrow. (The existing agent tools use `filed <= as_of`; the screen is stricter on purpose.)

The screen uses, per field and period end, the latest known version. The **window** is the 5 most recent
fiscal years that have a known revenue row. Balance figures are those whose `end` equals the latest fiscal
year's end.

The **share count** is the known entry with the latest `end`, then the latest `filed`, the cover-page count
winning a full tie. `counted_on` is the day the count is true on, which is what §4 restates from: the row's
`end` for a cover-page count, its `filed` date for a weighted-average count. ASC 260 restates a weighted
average for splits that happen after the period end but before the report is issued, so the count already
reflects every split up to its filing date; applying a split dated between `end` and `filed` would count it
twice.

### 3.2 Gates

Gates run in this order. The first one that fails is the recorded reason.

| # | Reason | Rule | Parameter (default) |
|---|---|---|---|
| 1 | `no_data` | No CIK, status `absent`, or no known flows as of the date. | |
| 2 | `stale_filing` | Latest fiscal year ended more than N months before `as_of`. | `max_filing_age_months` (18) |
| 3 | `insufficient_history` | Fewer than N consecutive fiscal years with revenue (above zero), operating income, operating cash flow and capex. Consecutive means each year's end is 350 to 380 days after the previous one. Also: implausible figures, i.e. operating income or free cash flow above revenue in any window year (a partial revenue tag or a tagging error), checked before the loss gates. | `years` (5) |
| 4 | `operating_loss` | Operating income <= 0 in any year of the window. | |
| 5 | `negative_fcf` | Free cash flow (operating cash flow minus capex) <= 0 in any year. | |
| 6 | `shrinking_revenue` | Fewer than N of the yearly revenue changes are increases, or the latest year's revenue is below the first year's. | `min_growth_years` (3 of 4) |
| 7 | `debt_unknown` | The latest fiscal year has no debt figure while an earlier year of the window has one. | |
| 8 | `too_much_debt` | Net debt (debt minus cash) above a multiple of the latest year's operating income. Net debt <= 0 passes. | `max_net_debt_to_operating_income` (4.0) |
| 9 | `excluded_sector` | SIC code inside an excluded range. A missing SIC code passes. | `excluded_sic_ranges` ((4900, 4999), (6000, 6799)) |
| 10 | `duplicate_listing` | Another symbol of the same company (same CIK) is already accepted. The first one in input order that passes every gate wins. | |
| 11 | `no_price` | `price_of(symbol)` returns `None` or a price <= 0, or raises a framework error, or there is no share count. | |
| 12 | `no_split_data` | The split lookup failed (§4), or the restated market cap is not finite and above zero (a corrupt split ratio). | |

Notes:

- **Missing debt.** A company with no debt figure in any year of its window is treated as debt-free and its
  candidate carries `debt_reported=False`. Missing cash counts as 0. A company whose latest year has no debt
  figure while an earlier year does is rejected as `debt_unknown`: a changed or missing debt tag for the
  latest year must not read as zero debt.
- **Order of the first gates.** With no known revenue row at all, `assess` returns `insufficient_history`
  before `stale_filing`: there is no fiscal year to judge staleness by. (No known flows of any kind is still
  `no_data`.) The implausible-figures case (operating income or free cash flow above revenue) is reported as
  `insufficient_history` too, ahead of the loss gates.
- **Sector.** The SIC code comes from SEC's submissions payload. It costs one request per company, so the gate
  runs after the numeric gates, only on their survivors, and the code is saved in the reduced file. The ranges
  exclude utilities (4900–4999) and finance, insurance and real estate (6000–6799).
- **Duplicate listings.** The universe holds both GOOGL and GOOG; without this gate one company could take
  two of the candidate slots.
- **Price-dependent gates last.** Gates 1 to 10 need no price, so `price_of` is called only for their survivors.

### 3.3 Metrics and score

For each survivor:

| Metric | Definition |
|---|---|
| `market_cap` | restated share count (§4) x price |
| `fcf_yield` | latest year's free cash flow / `market_cap` |
| `fcf_margin` | mean over the window of (free cash flow / revenue) |
| `operating_margin` | latest year's operating income / revenue |
| `operating_margin_stdev` | population standard deviation of the yearly operating margins |
| `revenue_growth` | compound annual growth rate from the first to the last year of the window |
| `net_debt_to_operating_income` | net debt / latest operating income (negative when net cash) |

Each of the three scored metrics becomes a percentile rank among the survivors: `(rank - 1) / (n - 1)`,
ascending, average rank for ties, and `1.0` when there is a single survivor.

```
score = 0.4 * pct(fcf_yield) + 0.3 * pct(fcf_margin) + 0.3 * (1 - pct(operating_margin_stdev))
```

The weights are `ScreenParams.weights`. Candidates are sorted by score descending, then market cap descending,
then symbol, and the first `top_n` (default 15) are returned with `rank` starting at 1.

### 3.4 Types

```python
@dataclass(frozen=True, slots=True)
class ScreenParams:
    years: int = 5
    min_growth_years: int = 3
    max_filing_age_months: int = 18
    max_net_debt_to_operating_income: float = 4.0
    excluded_sic_ranges: tuple[tuple[int, int], ...] = ((4900, 4999), (6000, 6799))
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)   # fcf_yield, fcf_margin, margin stability
    top_n: int = 15
    max_age_days: int = 30                  # annual SEC figures, by as_of (§2.4)
    split_max_age_days: int = 1             # split history, by the wall clock (§2.4)
    max_fetch_failure_ratio: float = 0.2
    hollow_min_sample: int = 5              # symbols at the SIC/split gate before its failure ratio applies (§5)
    # __post_init__ raises ValueError for out-of-range values (years < 2, a ratio outside [0, 1], ...)

@dataclass(frozen=True, slots=True)
class Candidate:
    symbol: str
    rank: int
    score: float
    sic: int | None
    market_cap: Decimal
    fcf_yield: float
    fcf_margin: float
    operating_margin: float
    operating_margin_stdev: float
    revenue_growth: float
    net_debt_to_operating_income: float
    debt_reported: bool
    fiscal_year_end: date
    filed: date            # filing date of the latest fiscal year's figures

@dataclass(frozen=True, slots=True)
class ScreenResult:
    candidates: list[Candidate]
    rejections: dict[str, str]
```

**Number types.** Statement figures stay integers, as SEC reports them. Price and `market_cap` are `Decimal`.
Ratios and the score are floats: they are dimensionless ranking inputs, not money, so this is not a fourth
float boundary in the sense of CLAUDE.md.

## 4. Splits

Bar prices are split-adjusted to today (Yahoo `auto_adjust=True`, Alpaca adjusted bars), but a share count is
as reported on its date. Without correction, a 2-for-1 split after the count halves market cap and doubles
`fcf_yield`.

- `restate_shares(shares, counted_on, splits)` (pure) multiplies the count by the ratio of every split dated
  after `counted_on`. Splits dated on or before `counted_on` are ignored. Splits after `as_of` are applied
  too, on purpose: the price being multiplied is already adjusted for them, so this undoes a price adjustment
  and leaks no information.
- `SplitHistory.splits(symbol, max_age_days=...)` returns `[(date, ratio), ...]` from yfinance, imported
  lazily. Results are cached in `cache/splits.json` with a `fetched_at` per symbol and the wall-clock freshness
  rule of §2.4 (never `as_of`). It is called only for symbols that reach gate 12.
- **Documented limit:** `price_of` must return a price adjusted up to today (Alpaca/Yahoo adjusted bars, the
  last trade). A frozen older price snapshot, adjusted only up to some earlier date, overstates the market
  cap of a name that split since (the restated count includes the newer split, the price does not), which
  demotes that name in the ranking rather than promoting it.
- A malformed cache entry (bad or naive `fetched_at`, bad split list, a ratio that is not finite and above
  zero) is a cache miss, validated once on load; a failed cache write logs a warning and the fetched value
  stays in memory. A Yahoo row whose ratio is not finite and above zero is not a split and is ignored.
- An empty history is a valid answer. A failed lookup rejects the symbol as `no_split_data`: a silently wrong
  valuation is the failure this module exists to prevent.
- `yfinance` moves from the `backtesting-yahoo` extra to the core dependencies (the extra keeps its entry).

## 5. Errors

- **Per-symbol problems are rejections, never exceptions.**
- **Real absences are cached.** HTTP 404 on company facts: the reduced file is written with
  `status: "absent"` and the symbol is not requested again until the file is stale. A ticker with no CIK
  needs no file: the ticker map is already cached by the client, so the lookup costs no request.
- **Transport errors are not cached.** A network failure or SEC's throttle page (HTML with a 2xx status)
  leaves no file, so the next run retries. The symbol is rejected as `no_data` for this run.
- **A hollow screen raises.** Three rules, each raising `FundamentalsError` that names the gate and the counts:
  1. *SEC figures:* more than `max_fetch_failure_ratio` (20%) of the symbols passed to `run` failed on
     transport errors (no minimum sample).
  2. *SIC lookup:* more than that share of the symbols that reached the sector gate (those that passed the
     numeric gates) failed their SIC lookup.
  3. *Split lookup:* more than that share of the symbols that reached the split gate failed their lookup
     (no cached copy: `splits()` raised), e.g. Yahoo down.

  Rules 2 and 3 apply only when at least `ScreenParams.hollow_min_sample` (5) symbols reached the gate, so a
  handful of dotted tickers (BRK.B) cannot abort a small screen. A symbol whose SIC lookup failed is still
  rejected `no_data`, but counts in rule 2 only, not in rule 1. Without rules 2 and 3 a Yahoo outage would
  return an empty ranking dressed as a result.
- **Configuration.** A missing `SEC_EDGAR_USER_AGENT` raises `ConfigurationError` when the client is built,
  as today.
- **A corrupt reduced file** is treated as a cache miss and refetched, as `SecEdgarClient.get_json` already
  does for its own cache. So is a file of the wrong shape (bad or naive `fetched_at`, wrong CIK, missing
  lists) and one whose `schema` is missing or not the current `SCHEMA_VERSION` (§2.1): a tag-list fix reaches
  every cache by a version bump.
- **Logging.** Each run logs one summary line: symbols in, candidates out, and the count per rejection reason.

`edgar_client.py` needs to tell a 404 from a transport error. `FundamentalsError` gains a subclass
`FundamentalsNotFoundError`, raised on HTTP 404; existing callers that catch `FundamentalsError` are
unaffected.

## 6. Testing

The automated suite stays off the network and uses hand-written fixtures, not mocks.

- **`tests/fundamentals/test_sec.py`** (extended): `annual_figures` on a payload where each 10-K repeats three
  years; every restated version is kept; quarterly and non-10-K facts are dropped; the tag fallbacks of §2.2,
  including the two-part debt sum and the `dei` share count; `parse_sic`.
- **`tests/fundamentals/test_quality.py`**: as-of selection (a restatement filed after `as_of` is ignored);
  one test per gate reason, each on a company that passes every earlier gate; gate order; the three scored
  metrics; percentile ranks with ties and with a single survivor; the sort and its tie-breaks; `top_n`.
- **`tests/fundamentals/test_splits.py`**: `restate_shares` with no split, one split, several splits, and a
  split dated on or before `counted_on`; `SplitHistory` with an injected fetch function: caching, freshness,
  empty history, failed lookup.
- **`tests/fundamentals/test_annual_store.py`**: `httpx.MockTransport` and a temp cache directory: lazy fetch;
  the reduced file is written and the raw payload is not; a second call makes no request; the freshness rule
  in both directions (simulated past `as_of`, and `as_of` beyond `max_age_days`); a 404 is cached as
  `absent`; a transport error is not cached; a stale file survives a failed refetch; a corrupt file is
  refetched; the SIC code is fetched once and saved.
- **`tests/fundamentals/test_screen.py`**: `QualityScreen.run` with a fake store and fake split history:
  `price_of` is called only for survivors of gates 1 to 10; a duplicate listing; the hollow-screen error at the threshold; the
  summary log line.
- **`scripts/tests/smoke_quality_screen.py`** (manual, real SEC and yfinance): runs the screen on about 20
  symbols and prints the candidates and the rejections.

## 7. Known limits

- **The universe file is a current snapshot**, so a backtest screens today's survivors. `cross_momentum` has
  the same bias.
- **SIC codes are current, not point-in-time.** A company that changed industry is classified by its present
  code on every backtest date.
- **The gates are stricter than Ackman's real book.** From memory of their figures (not verified), several
  Pershing Square holdings would fail: HLT and QSR on debt, UBER on its loss years. The screen finds
  companies matching the stated criteria, not his holdings.
- **Capex tags vary.** A company reporting capital expenditure under a tag outside §2.2 is rejected as
  `insufficient_history`.

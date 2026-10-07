# congress_trades: an LLM agent that follows Nancy Pelosi's disclosed stock trades

Date: 2026-10-07 · Source: port of lumibot's "Nancy Pelosi trading bot" agent example (the page was not readable
from the build sandbox; the design below is from the brief, not from that page's code)

## Goal

A new strategy, `congress_trades`, in which an LLM agent reads newly disclosed stock transactions of a configurable
list of members of Congress (default: Nancy Pelosi) and trades their tickers, weighting each position by the
size of the disclosed trade. It runs in paper, live and backtesting.

Success: `uv run pytest` and `uv run ruff check` pass; `uv run agent congress_trades backtesting` runs a
multi-year window with no look-ahead; the smoke script parses a real Clerk filing.

## Decisions taken in brainstorming (binding)

1. **Data: the House Clerk's Periodic Transaction Reports (PTRs)**, the official free source. No API key.
2. **The decision-maker is an LLM agent** with a congress-trades tool, like `news_binary` (not a code copier).
3. **Sizing follows the disclosed amount.** Filings give dollar ranges only (e.g. $250,001-$500,000); the tool
   turns the range midpoint into a suggested weight (section 3).
4. **Several politicians from the start**, a configurable list; stock transactions only (no options, bonds, funds).

## Decisions taken by the designer (change if wrong)

- **As-of rule: a filing is known only when its filing date is strictly before `clock.now()`'s date** (the Clerk gives
  a date, no time; same rule as the quality screen). Trades are keyed by FILING date, never transaction date:
  disclosure lags the trade by up to 45 days, and keying on the trade date would leak the future in a backtest.
- **Long-only.** A disclosed sale is acted on only for a position this account holds; no shorts.
- **Spouse trades count** (Paul Pelosi's trades are most of the disclosed volume); the payload carries `owner`
  (`self`, `spouse`, `joint`, `dependent`) so the agent can see it.
- **Daily cadence**, `sleeptime = "1D"`, `iteration_start_time = 10:00` ET. Filings appear at most once a day, so a
  faster tick buys nothing.
- **Sizing is guidance in the tool payload and the prompt, not a code guardrail** (no `Desk`). If runs show the
  agent ignoring it, add a desk like `earnings_drift/desk.py` in a follow-up.

## 1. Layout

```
fundamentals-style data client
  congress/__init__.py
  congress/ptr.py            PURE: filing index parsing, PTR text -> Transaction, amount ranges, weights
  congress/clerk_client.py   ClerkClient: the only module importing httpx for the Clerk; cache <root>/cache/house_clerk/
agents/tools/congress.py     congress_trades_tools(strategy) -> [search_congress_trades]   (clock-gated)
strategies/congress_trades/
  __init__.py
  agent_congress_trades.py   CongressTradesStrategy (system prompt inline, like news_binary)
  parameters.py              CongressParams (politicians, lookback, caps)
main.py                      AGENT_STRATEGIES["congress_trades"]
scripts/tests/smoke_congress_trades.py   manual smoke test against the real Clerk (read-only)
tests/congress/, tests/agents/tools/test_congress.py, tests/strategies/congress_trades/
```

`ptr.py` is pure (no I/O, no clock), like `fundamentals/sec.py`. `clerk_client.py` is the only network code.

## 2. Data layer

**Index.** The Clerk publishes one zip per year, `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/<YYYY>FD.zip`,
with `<YYYY>FD.xml`: one row per filing (last name, first name, filing type, filing date, DocID). PTRs have filing
type `P`. `ptr.parse_index(xml)` returns `FilingRef(doc_id, member, filed: date, year)` for PTRs only; the
member filter is a normalized-name match (`"Nancy Pelosi"` matches last `Pelosi` + first `Nancy`, honorifics and
suffixes ignored).

**Filings.** Each PTR is a PDF at `.../ptr-pdfs/<YYYY>/<DocID>.pdf`. `ClerkClient.fetch_pdf_text(ref)` downloads and
extracts the text (a PDF text library, one new dependency, chosen in the plan; the choice must not need a system
binary). Image-only (scanned) filings carry no text: they are skipped with a logged warning, and counted in the
payload's `unparsed_filings`.

**Transactions.** `ptr.parse_transactions(text, ref)` returns `Transaction(doc_id, member, owner, asset_name,
ticker, side, transaction_date, notification_date, amount_low, amount_high, filed)`:

- Keep only asset type `[ST]` (stock) with a ticker in parentheses, e.g. `NVIDIA Corporation - Common Stock (NVDA) [ST]`.
  Options `[OP]`, funds, bonds and rows with no ticker are dropped and counted (`skipped_non_stock`).
- `side`: `P` -> `buy`; `S` and `S (partial)` -> `sell`; `E` (exchange) is dropped.
- `amount_low`/`amount_high` are `Decimal` from the range bands the form uses ($1,001-$15,000 ... Over $50,000,000;
  an open-ended top band gets `amount_high = amount_low`).
- Money is `Decimal`; no new float boundary.
- A row that does not parse is skipped and counted, never raised, but a filing where NO row parses raises
  `FundamentalsError`-style `CongressDataError` (a changed layout must not read as "no trades").

**Caching.** The year index and each PDF's extracted text are cached under `<project_root>/cache/house_clerk/`.
A filing is immutable once filed (amendments are new DocIDs), so a PDF never expires; the CURRENT year's index is
refreshed when its `fetched_at` is older than 1 day by `as_of` (`fundamentals/freshness.is_stale`, same rule as the
SEC client: a backtest never refetches an existing file). Rate limit and a descriptive `User-Agent` from
`CONGRESS_USER_AGENT` (env, required, like `SEC_EDGAR_USER_AGENT`).

## 3. The agent tool

`search_congress_trades(days: int = 30, ticker: str = "", politician: str = "")` in `agents/tools/congress.py`,
bound to the `Strategy` so it takes its cutoff from `strategy.clock.now()` (the research-tool rule in CLAUDE.md).

- Returns transactions whose `filed` date is in `[today - days, today)` (strictly before today), newest first,
  capped (default 20) and lean: `{ticker, side, member, owner, filed, traded, amount_low, amount_high, suggested_weight}`.
  Plus `{"count", "unparsed_filings", "skipped_non_stock"}`. Errors come back as `{"error": ...}`.
- `suggested_weight` is computed in code (`ptr.suggested_weight`): the transaction's range midpoint divided by the
  sum of the midpoints of all buys in the window by the same members, times `CongressParams.max_total_weight`
  (default 0.9), each capped at `max_position_weight` (default 0.15). So the weights sum to at most 0.9 across the
  names disclosed in the window and a single big trade cannot take the whole book. Sells carry no weight.
- Window sums use only filings known at `now`, so the weights are identical in a backtest and live.
- Same-run identical calls reuse the first result (`RunMemo`, like `search_news`).

## 4. The strategy

`CongressTradesStrategy` mirrors `NewsBinaryStrategy`: `PrebuiltTools.all(self)` + `congress_trades_tools(self)`,
inline system prompt, a `portfolio` snapshot in every run's context, `FatalStrategyError` after 3 consecutive
agent failures in a backtest, and the single corrective `remember_decision` follow-up. The prompt tells the agent to:

1. Call `search_congress_trades` (default lookback `CongressParams.lookback_days`, 45 = the legal filing window).
2. Buy tickers with a recent disclosed purchase it does not already hold, at `suggested_weight` of the portfolio,
   sized against the smaller of `buying_power` and `cash` + same-run sell proceeds (the existing rule).
3. Sell a held ticker when a politician disclosed a sale of it; ignore sells for names it does not hold.
4. Take no action when nothing new was filed (no churn), and record the decision with `remember_decision`.

The universe is whatever tickers the filings contain, so tickers the broker cannot trade are refused by the
broker and come back as `{"error": ...}`.

Benchmark: SPY (`benchmark_symbol` default). Backtest window: the predefined decade/5-year window is too long
for the free Clerk history of electronic PTRs; the default is `PredefinedWindow` covering 2 years, adjustable.

## 5. Testing

- `tests/congress/test_ptr.py`: index parsing, name matching, row parsing for P / S / S (partial) / options /
  no-ticker / amount bands, an unparseable filing raises, weights sum to at most `max_total_weight` and respect the cap.
- `tests/congress/test_clerk_client.py`: fake `httpx` transport; cache hit does not refetch; backtest `as_of`
  never refetches; current-year index refetched after a day.
- `tests/agents/tools/test_congress.py`: the tool never returns a filing dated today or later (frozen strategy
  clock), error payloads, `RunMemo`.
- `tests/strategies/congress_trades/`: a fake agent run through the executor in backtesting.
- No test touches the network. `scripts/tests/smoke_congress_trades.py` runs against the real Clerk.

## 6. Risks and open points

- **Filing layout is unverified.** The build sandbox could not reach `disclosures-clerk.house.gov`. The first
  plan task is to run the smoke script on a machine with access, capture one real PTR's extracted text as a
  fixture, and fix `ptr.py` against it before any other code depends on it.
- **Disclosure lag** (up to 45 days) makes this a slow-signal strategy; the backtest, not the idea, decides
  whether it adds anything. A known cost of keying on the filing date.
- **Dollar ranges are coarse**, so weights are approximate by construction.
- **Scanned pre-electronic filings** are unreadable and skipped.
- **Sizing is advisory** (see designer decisions).

## Out of scope

Senate disclosures, options and futures, shorting, paid APIs (Quiver), real-time alerts, a desk with code guardrails.

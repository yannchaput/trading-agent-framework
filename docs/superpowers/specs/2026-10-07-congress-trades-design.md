# congress_trades: a three-agent team that mirrors Nancy Pelosi's disclosed stock holdings

Date: 2026-10-07 · Source: port of lumibot's "Nancy Pelosi trading bot" agent example (the page was not readable
from the build sandbox; the design below is from the brief, not from that page's code). Supersedes the first
version of this spec (single agent, search tool): the brief changed to a researcher → portfolio → trader pipeline.

## Goal

A new strategy, `congress_trades`, that once a day checks the House Clerk's disclosures for Nancy Pelosi and
**trades only when she has filed something new**. Three agents hand structured results to each other:

1. **Research agent** finds her newest yearly report (Financial Disclosure, FD) and every Periodic Transaction
   Report (PTR) filed since, and works out what she owns today.
2. **Portfolio agent** turns those holdings into a target mix: the bigger her holding's value band, the bigger
   its share of the account.
3. **Trading agent** places the buys and sells to match the mix and checks that every order filled.

It runs in paper, live and backtesting. Success: `uv run pytest`, `uv run ruff check` and `uv run pyright` pass;
`uv run agent congress_trades backtesting` runs a multi-year window with no look-ahead and trades on filing days only;
the smoke script parses a real yearly report and a real PTR.

## What "owns today" means (binding)

- **Reports known before today.** A filing counts only when its FILING date is strictly before `clock.now()`'s
  market date (the Clerk gives a date, no time; same rule as the quality screen). Applies to the yearly report
  and to every PTR.
- **Base = the newest yearly report known** (index types `C` annual and `A` amendment; the newest by filing
  date wins). It lists each asset held on the report's period end (Dec 31 of its reporting year) as a VALUE BAND
  (e.g. $5,000,001 - $25,000,000), never shares.
- **Then every PTR trade made after that Dec 31.** Deviation from the brief's wording ("every trade report filed
  since" the yearly report): a yearly report is filed months after its period end (Pelosi files in summer), so
  PTRs filed between Dec 31 and the yearly report's filing date describe trades the report does not include.
  The rule is therefore: PTRs known before today whose TRANSACTION date is after the report's period end. A PTR trade
  dated on or before the period end is already inside the report (disclosure lag) and is ignored.
- **Estimate per ticker** (`congress/holdings.py`, pure, `Decimal`): start at the band midpoint; a buy adds the
  range midpoint; a partial sale subtracts it (floor 0); a full sale (`S`) sets it to 0. Result: an estimated
  value range and a TIER (the index of the yearly-report value band the estimate falls in, 0 = smallest).
- Stocks only (`[ST]` with a ticker). Options, funds, bonds and rows with no ticker are dropped and counted in
  the handoff (much of her real exposure is call options, which a long-only stock bot cannot copy; this is a
  stated limit, not a bug). Spouse and joint assets count (`owner` is kept). One politician (a `politician`
  setting, default "Nancy Pelosi"); several at once is out of scope.

## Daily check, trade only on news (binding)

State file `data/congress_trades_state_<mode>.json` (`StateStore`, wiped at the start of a backtest only, like
`bill_ackman`) keeps `processed`: the DocIDs of the yearly report and PTRs the last COMPLETED run was built
from. Each daily tick (`sleeptime = "1D"`, `iteration_start_time = 10:00` ET):

1. Code lists the filings known before today. **New = known DocIDs not in `processed`.**
2. **Nothing new, nothing pending → the research stage answers "nothing new"**: logged as the research result,
   no agent is run (zero LLM calls, deterministic), no order is sent.
3. Something new → research → portfolio → trading.
4. **Pending trade.** If the last run finished its trading stage with orders unfilled, state holds
   `pending_trade` (the target and a day counter) and the next ticks re-run ONLY the trading stage against the
   stored target, at most `max_trade_retries` (3) times, then give up and log it. This is what "check that every
   order filled" means across days; it never re-runs research or portfolio and never trades on a quiet day with
   nothing pending.
5. The first run (empty state) treats every known filing as new, so it builds the portfolio once.

Design choice flagged for review: step 2 is a code short-circuit, not an agent that answers "nothing new". The
observable behaviour is the same and it saves a model call per quiet day.

## Agents and hand-offs (same pattern as `bill_ackman`)

Each agent ends its run by calling ONE submit tool, validated by a `HandoffRecorder` armed by the pipeline; an
invalid submission returns `{"error": ...}` so the model can correct itself; the first valid one is final; free
text from an agent is logged and ignored. A stage with no valid submission after one forced retry
(`force_tool=<submit tool>`) abandons the run: nothing is traded, state does not move, `abandoned_streak` goes
up, and a backtest raises `FatalStrategyError` at `max_consecutive_abandoned` (3). All three agents run at
`agent_temperature` 0.3. `handoff.py` has no `from __future__ import annotations`.

**Research agent.** Tools: `list_filings()` and `read_filing(doc_id)` (lean parsed rows; the clock gate is in the
tool: it refuses a DocID filed on or after today), market-data tools for a ticker sanity check, and
`submit_holdings`. Its context carries `new_filings` and a code-computed `baseline` (the reconstruction above).
Code does the arithmetic; the agent resolves what code cannot (an ambiguous partial sale, a renamed or merged
ticker, an unparsed row) and may drop or adjust a holding, always with a reason. Validation: every ticker must
appear in a known filing; value estimates must be positive; a holding that differs from the baseline needs a
reason; at most `max_holdings` (20).

**Portfolio agent.** Tools: `submit_target` only. Context: the holdings with tier and estimated range, a
code-computed `baseline_weight` per ticker (share of estimated midpoints × `max_total_weight`, each capped at
`max_position_weight`), and the current portfolio weights. Validation (the brief's rule, enforced by code):
tickers ⊆ holdings; each weight in [`min_weight`, `max_position_weight`]; sum ≤ `max_total_weight`; **a holding
in a higher tier never gets a smaller weight than one in a lower tier**; at most `max_positions` names; a held
name may be dropped only with a reason (e.g. not tradable). Defaults: `max_total_weight` 0.95, `max_position_weight` 0.15,
`min_weight` 0.01.

**Trading agent.** Tools: `get_positions` and `get_account_balance` (the existing account tools),
`get_last_price`, and the desk's `place_order(symbol, side, quantity)`, `check_orders()` and
`submit_trade_report`. `TradeDesk` is the only order code and enforces, per call, with `{"error": ...}` on a
violation: a buy only for a target ticker and not beyond its target weight (held + open buys + this order, within the
rebalance band); a sell only of a position this strategy owns (a target ticker or one in `state.traded`; a shared
account's other positions are left alone) and never above held minus open sells; sells before buys; buys sized
against the SMALLER of `buying_power` and `cash` + the proceeds of the sells submitted this run (the repo rule,
never `buying_power` alone); no shorts, no margin. `check_orders()` waits (`strategy.wait_for_orders_execution`,
with a timeout) and returns each order's status, filled quantity and average price.
`submit_trade_report(orders)` is accepted only when EVERY desk order is in a final status and the report names
each one; an order that ended cancelled or rejected is accepted with a reason. The pipeline then audits in code:
final positions against the target (tolerance: the rebalance band); any shortfall sets `pending_trade`.

## Layout

```
congress/__init__.py
congress/ptr.py             PURE: filing index (types C/A/P), member match, PTR rows -> Transaction, amount bands
congress/annual.py          PURE: yearly report text -> AssetHolding (value band), value-band table and tiers
congress/holdings.py        PURE: reconstruct(annual assets, PTR transactions, period_end) -> Holding (range, tier); baseline weights
congress/clerk_client.py    ClerkClient: the only Clerk/httpx/pypdf code, disk cache <root>/cache/house_clerk/
congress/source.py          CongressSource: known-before-today filings, parsed; the new-filing diff
agents/tools/congress.py    research tools list_filings / read_filing, bound to the Strategy (clock gate)
strategies/congress_trades/
  parameters.py             CongressParams
  handoff.py                HandoffRecorder + submit_holdings / submit_target / submit_trade_report validation
  desk.py                   TradeDesk: guarded place_order, check_orders; the only order code
  state.py                  StateStore (processed, last holdings/target, pending_trade, traded, abandoned_streak), RunLog
  pipeline.py               CongressPipeline: nothing-new short-circuit, three stages, audit
  prompts.py                the three system prompts and task prompts
  agent_congress_trades.py  CongressTradesStrategy
main.py                     AGENT_STRATEGIES["congress_trades"]
scripts/tests/smoke_congress_trades.py
```

The pure modules do no I/O and read no clock. `clerk_client.py` is the only network code (`CONGRESS_USER_AGENT`
required, like `SEC_EDGAR_USER_AGENT`); yearly and PTR PDFs are immutable once filed, so their extracted text is
cached forever; a year's index is refetched when its `fetched_at` is older than 1 day by `as_of`
(`fundamentals/freshness.is_stale`) and the file was fetched no later than that year, so a backtest never
refetches an existing index. Image-only (scanned) PDFs have no text: skipped with a warning and counted. A
filing with text but no parseable row raises `CongressDataError`: a changed layout must not read as "no trades".

## Backtesting

Daily bars (`timestep="day"`) like `bill_ackman`; the tickers are not known up front, so the data source must
load them lazily (the plan's Task 9 checks how `YahooBacktestData` handles an asset that was not preloaded and
adds a pre-scan of the window's tickers to `preload_assets` if it does not). A backtest fills an order on the next
bar, so `check_orders()` in a daily backtest finds orders unfilled at the 10:00 tick; the plan verifies how
`wait_for_orders_execution` advances the simulated clock and, if it cannot cover a next-session fill, the
audit/`pending_trade` path (next tick re-checks, never re-sends what is open) is the designed behaviour. Orders
still open count toward a position when sizing (the `Rebalancer` rule). Default window: two years.

## Risks and open points

- **Both filing layouts are unverified** (yearly report Schedule A and PTR rows). The build sandbox cannot reach
  `disclosures-clerk.house.gov`. The first plan tasks end in a checkpoint: run the smoke script where the site is
  reachable, store real extracted text as fixtures, fix the parsers. Nothing ships before it.
- **Estimates are coarse by construction** (bands and ranges, not shares), and exclude options.
- **Disclosure lag** (PTRs up to 45 days, the yearly report months) makes this a slow signal; the backtest decides.
- **The trading agent has order tools.** Unlike `bill_ackman`, per the brief; the desk's guardrails make a bad
  order impossible rather than unlikely, and an order the agent forgets is caught by the audit.

## Out of scope

Other politicians at once, Senate disclosures, options and futures, shorting, paid APIs, a parking instrument
(unallocated money stays cash).

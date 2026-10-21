# insider_trades: follow opportunistic insiders' open-market trades (Form 4)

Date: 2026-10-21 · Source: port of lumibot's "Insider trading bot" agent example
(https://lumibot.lumiwealth.com/agents_example_insider_trading_bot.html), reshaped. The reference's edge claim is
Cohen, Malloy and Pomorski (2012, Journal of Finance): copying insiders' **non-routine** (opportunistic) trades earned
about 0.82% a month over the market. Unlike the lumibot example, this strategy applies the study's routine /
opportunistic filter.

## Intent

A long-only book over a short watchlist that leans toward stocks whose company insiders (CEOs, CFOs, directors) buy
with their own money and away from stocks they sell heavily, counting only insiders whose trades carry information
(opportunistic, not routine). It is a **baseline**: pure insider-following, no market-timing overlay. A regime-gated
variant is a separate, later strategy.

The LLM is a real decision maker here (two agents), driven by a local model (e.g. Qwen3-8B), so code does the
bookkeeping and lays out facts; the agents verify and decide within limits code enforces.

## Goals / success

- **Technical:** `uv run agent insider_trades backtesting` runs the default one-year window
  (`PredefinedWindow.YEAR`, 2025-09-29 to 2026-10-16) end to end with **no look-ahead** and clean hand-offs between
  the two agents (every run ends in a valid submit, or is abandoned and logged). `uv run pytest`, `uv run ruff check`
  and `uv run pyright` pass.
- **Functional:** the user judges resilience during bearish periods (e.g. Oct-Nov 2025) from that backtest against
  SPY. This is measured, not engineered: nothing in this design reacts to the market.
- Runs in `live` (Tastytrade), `paper` (Alpaca paper) and `backtesting` (Yahoo daily bars).

## Non-goals / anti-goals

- **No use of the market regime** (`strategy.regime`) to size or gate. Explicitly deferred by the user.
- No shorting, no margin, no options, no parking instrument (unallocated money stays cash).
- No full-universe screening yet: the watchlist is a parameter; a filter will replace it later. Do not hard-code
  the ten tickers anywhere but the parameter default.
- Anti-goal: a weight the agent "made up" with no insider fact behind it. Tilts must trace to opportunistic
  activity in a note.
- Anti-goal: a backtest that sees a Form 4 before its filing date.
- Do not change `congress_trades` behaviour. Reusing its patterns is expected; whether shared code is extracted into
  `strategies/common/` or copied and adapted is the builder's call, but congress_trades' tests must keep passing
  unchanged.

## Constraints (from the repo)

All root and nested `CLAUDE.md` rules apply; the ones that bite here: money is `Decimal`; no wall-clock time in
strategy, executor or data-tool code (every cutoff from `strategy.clock.now()`); research tools gate on the clock in
the tool; SEC access goes only through `fundamentals/edgar_client.py` (`SEC_EDGAR_USER_AGENT`); buys are sized
against the SMALLER of `buying_power` and cash + this run's sell proceeds; the automated tests never touch the
network (hand-written fakes); `FatalStrategyError` to abort a backtest.

## Watchlist and broker (decided)

- Default watchlist (lumibot's): AAPL, MSFT, JPM, BAC, XOM, CVX, PFE, INTC, F, KO.
- **Live broker: Tastytrade** (`BROKER=tastytrade` in `env/.env.insider_trades.live`). Cash account (the broker
  refuses a margin account live). Fractional quantities work only on market orders of eligible stocks, so the desk
  sends market orders only.
- **Paper: Alpaca paper** (`BROKER=alpaca` in `env/.env.insider_trades.paper`) -- Tastytrade has no usable paper
  account here (the sandbox clears positions daily and strategies refuse it). *Builder-facing note, my call:* the
  strategy code is broker-agnostic; only env files differ.
- **Backtests: `YahooBacktestData`**, daily bars, `BROKER=tastytrade` in the backtest env file (picks the $0
  commission fee schedule). Benchmark SPY. Budget: builder's call (congress_trades uses 10,000).

## The insider data (decided)

**Source:** each watchlist company's SEC submissions (Form 4 filings appear in the issuer's filing list) and each
filing's Form 4 XML. Only form type `4`. *My call:* `4/A` amendments and Form 5 are ignored (as congress_trades
ignores amendments); log their count.

**Transactions kept:** non-derivative table rows with transaction code **P** (open-market purchase) or **S**
(open-market sale). Everything else (A awards, M option exercises, G gifts, F tax withholding, derivative rows) is
dropped and counted.

**Who counts (my call):** officers and directors only, matching the brief ("CEOs, CFOs, and directors"). A reporting
owner who is only a 10% owner or "other" is dropped and counted (on this watchlist a fund's block sales would
otherwise dominate a stock's signal). **Joint filings (my call):** a filing listing several reporting owners
describes ONE set of rows; its dollars are counted once, attributed to the first listed owner who is an officer or
director (the others are named in the note as co-filers), never once per owner.

**Per transaction, parsed:** issuer ticker, reporting owner(s) (CIK + name), role (director / officer with title / 10% owner / other, from the relationship block), transaction
date, code, shares, price, dollars (= shares x price, `Decimal`), shares owned after the transaction (so code can
compute the fraction of the holding sold), the 10b5-1 flag (the `aff10b5One` checkbox, present on filings since
2023) and the footnotes' text (the researcher may read a plan mention there), plus the filing's accession number
and **filing date**.

**Point-in-time rule (binding):** a filing is known only when its filing date is **strictly before** today's market
date by `strategy.clock.now()` (the congress_trades rule). The source applies it, and the research tool applies it
again to whatever the source returns.

**History depth:** classification needs each insider's P/S trades in the three calendar years before the year of
the trade being classified, so the source must reach at least 3 full calendar years before the backtest start
(back to 2022-01-01 for the default window). The submissions JSON's `recent` block is not enough for active filers:
the older pages listed in `filings.files` must be read too (the existing `sec.parse_filings` reads `recent` only).
SEC caching: filing XML is immutable (cache forever); the submissions payload is refreshed by the strategy clock's
`max_age_days` rule like the other SEC callers (a backtest loads with the run's end as `as_of`, as earnings_drift
does). The first backtest downloads a few thousand small XML files at SEC's rate limit; that is accepted.

*Builder-facing gotcha:* a Form 4's `primaryDocument` in the submissions JSON usually points at the XSL-rendered
HTML (`xslF345X05/<name>.xml`); the raw XML is the same file name at the accession folder's root. Verify against a
real filing in the smoke script before trusting the parser (the build sandbox may not reach SEC; same checkpoint
as congress_trades: store real XML as fixtures).

## Routine vs opportunistic (decided: code labels, the researcher verifies)

Code computes each insider's label per watchlist company, following the study:

- Look at the insider's P/S **transaction dates** (from filings known before today) in the three calendar years
  before the year of the trade being labelled.
- **Routine:** they traded in the same calendar month in each of those three years.
- **Opportunistic:** they traded at least once in each of those three years, with no such repeated month.
- **Unclassified:** missing a year of trades (new insiders, rare traders). Their trades do not move weights by
  default; counted and shown.
- *My call:* a trade with the 10b5-1 flag set is **routine** regardless of history (a pre-scheduled plan trade
  carries no timing information).

**Labels are per trade (my call), summarised per insider.** The history label depends on the trade's year (a window
crossing January 1 compares the two sides against different 3-year histories) and the 10b5-1 flag is per trade, so
one insider can have routine and opportunistic dollars in the same window. The note therefore reports, per
insider, dollars bought and sold **split by label**; only opportunistic dollars feed the stock's signal. An override
by the researcher applies to one insider's in-window trades of one label (e.g. "this insider's opportunistic sales
are really plan sales"); it never changes a 10b5-1-flagged trade into opportunistic.

The label comes with its evidence, laid out for the researcher: the insider's trade months per year for the three
years and the rule that fired. The researcher may override a label (routine <-> opportunistic, unclassified ->
either) **only with a reason** (e.g. a plan mentioned in a footnote, an insider newly promoted to CEO, a filing the
parser misread). An override without a reason is refused by the submit tool.

## The signal window

A transaction counts toward a stock's signal for **30 calendar days after its filing date** (lumibot's window; the
study rebalances monthly). After that it expires and the stock drifts back toward an even weight on the next run.

## A stock's signal class (decided by the user)

Code puts each watchlist stock in one of three classes from its in-window trades and their **final** labels (after
the researcher's overrides). The class drives `submit_target`'s ordering rule and is shown to the trader.

- **Buying:** at least one opportunistic purchase (code P) in the window that is not negligible. There is **no
  netting against sales**: an opportunistic buy is by definition a break from the insider's habit, and that is the
  signal. A purchase is negligible, and ignored, when it is below `min_buy_holding_fraction` of the insider's
  holding (shares bought / shares owned after the purchase). *My reading of the user's "negligible once scaled to
  the portfolio" as the insider's own holding, symmetric with the selling rule; default 0.01; configurable.*
- **Selling:** opportunistic dollars sold exceed opportunistic dollars bought in the window, **and** at least one
  opportunistic seller sold at least `min_sell_holding_fraction` of their holding (shares sold / shares owned
  before the sale). Default 0.10; configurable.
- **None:** neither.
- *My call:* a stock that meets both rules is **buying** (opportunistic buys are the study's stronger signal); the
  note shows both facts.

## Daily check, act only on change (decided)

`sleeptime = "1D"`, `iteration_start_time = 10:00` ET (congress_trades' timing). Each tick:

1. Code refreshes the watchlist's Form 4s known before today.
2. **Change =** a newly known filing with at least one P or S row on a watchlist stock, **or** an in-window filing
   that expired since the last completed run. *My call:* any new P/S filing triggers, routine or not, because the
   researcher must get the chance to verify a label; a filing with no P/S rows never triggers.
3. **No change and nothing pending -> no agent runs, no order** (a "nothing_new" line in `runs.jsonl`).
4. **Change ->** researcher (once per changed stock) -> trader (once).
5. **First run** (empty state): every stock is "changed"; the trader builds the initial book (even weights when no
   opportunistic signal is in window).
6. **Pending trade** exactly as congress_trades: a trading stage that ends with orders still working, or positions
   off target by more than the band, stores `pending_trade`; later ticks re-run only the trading stage against the
   stored target, at most `max_trade_retries` (3); a new change replaces it. Open orders are never re-sent.

State lives in `data/insider_trades_state_<mode>.json` (processed accession numbers, the latest note per stock, the
current target, `traded`, `pending_trade`, abandoned streak), wiped at the start of a backtest only.

## Agents and hand-offs (decided: two agents)

Same hand-off pattern as congress_trades: each agent ends by calling one submit tool, validated by a recorder; an
invalid submission returns `{"error": ...}` for the model to correct; one forced retry; a stage with no valid
submission abandons the run (nothing traded, state unchanged, streak +1; a backtest raises `FatalStrategyError` at
`max_consecutive_abandoned`, 3). Free text is logged and ignored. Temperature 0.3 (congress default).

### 1. Researcher (no order tools)

One run **per changed stock** (*my call:* keeps each context small for a local model).

- **Tools:** an insider-filings tool (clock-gated) that returns the stock's parsed P/S transactions in the signal
  window with code's label and evidence per insider, and a way to read one filing's parsed detail (footnotes
  included) by accession number. Exact tool split is the builder's call; payloads must stay lean.
- **Output: `submit_note`** -- one note for the stock: for each insider active in the window: name, role, dollars
  bought and sold split by label, fraction of their holding bought / sold, final labels, and why (code's rule, or the override reason); plus
  a one-line summary of the stock's net opportunistic buying / selling. Validation: insiders and accession numbers
  must exist in the tool's data (never invented); dollar figures must match code's (the agent does not recompute);
  overrides need a reason.
- Stocks with no change keep their last note while it has in-window activity; a stock with nothing in window gets a
  code-written "no opportunistic activity" note (no agent run).

### 2. Trader / portfolio manager (order tools through the desk)

One run per change, reading every stock's current note. **One agent, one run, two submits in order (my call,
following the user's "one trader/portfolio agent"):** the desk refuses every order until a valid `submit_target` is
recorded, and `submit_target` is accepted once. The run is complete when `submit_trade_report` is valid. A forced
retry forces whichever submit is still missing. A valid target with no valid trade report keeps the target and,
if any order was sent, leaves a `pending_trade` (congress_trades' rule); with no target, the run is abandoned. A
pending-trade retry runs the same agent with the stored target in its context and without `submit_target` among
its tools.

- **Context:** all ten notes, the current positions and weights, the previous target, the limits.
- **Decides weights:** even by default (`max_total_weight` / number of stocks = 9.5% with 10), tilted toward net
  opportunistic buying and away from net opportunistic selling.
- **`submit_target` (validated by code):** every watchlist stock gets a weight in **[3%, 20%]** (never dropped);
  the sum is at most **95%**; **ordering rule** (classes from "A stock's signal class"): a *buying* stock never gets
  less than a *none* stock, and a *selling* stock never more; **with every stock in *none*, weights must be even**
  (within a rounding tolerance). Each weight that differs from even needs
  a short reason naming the note fact behind it.
- **Then trades** through the desk (congress_trades' `TradeDesk` rules): sells before buys; buys bounded by the
  submitted target plus the band and by the SMALLER of `buying_power` and cash + this run's sell proceeds; sells
  only of positions this strategy ordered; no shorts, no margin; market orders only; `check_orders`; ends with
  **`submit_trade_report`** naming every order. The pipeline then audits positions against the target.

## Parameters (`InsiderParams`, frozen dataclass, validated in `__post_init__`)

Decided values: `watchlist` (the ten tickers), `signal_window_days` 30, `history_years` 3, `max_total_weight` 0.95,
`min_weight` 0.03, `max_position_weight` 0.20, `min_buy_holding_fraction` 0.01, `min_sell_holding_fraction` 0.10,
`max_trade_retries` 3, `max_consecutive_abandoned` 3,
`agent_temperature` 0.3. Builder's call (congress defaults are fine): `rebalance_band`, `min_trade_pct`,
`reason_max_chars`, `order_wait_seconds`. Validation must include `len(watchlist) x min_weight <= max_total_weight`.

## Registration and files

- `Strategies.INSIDER_TRADES` in `utils/strategy_factory.py`; strategy directory `strategies/insider_trades/` with
  its own `CLAUDE.md`, and a line in `strategies/CLAUDE.md` and the root `CLAUDE.md`'s strategy list.
- Env files `env/.env.insider_trades.{live,paper,backtesting}` as above (documented in README's env table only if
  a new variable is introduced; none is planned).
- Manual smoke script `scripts/tests/smoke_insider_trades.py` (read-only, real SEC): fetch one watchlist
  company's Form 4s over 3+ years, parse a real filing, print a few labels with evidence. Add it to the root
  `CLAUDE.md` commands list.

## Left to the builder

Module layout inside `strategies/insider_trades/`; extract-vs-copy of congress_trades' desk/handoff/state code;
the exact tool split and payload shapes for the researcher; prompt wording (English only, written for a local
model, congress_trades' style); the budget; the parser's handling of malformed or missing fields (a filing that has
XML but no parseable table must raise, not read as "no trades", congress_trades' rule).

## Risks

- **Large caps have few opportunistic insider buys.** The lumibot run saw none in three weeks. A year may still
  look nearly equal-weight; that is a valid baseline result, not a bug.
- **Routine sells by 10b5-1 plans are frequent** in large caps, so "any new P/S filing triggers" may run the agents
  most days. Acceptable for a local model; revisit if backtests are too slow.
- **Unverified XML details** (primaryDocument path, `aff10b5One` presence, multi-owner filings) until the smoke
  script runs against real SEC data.

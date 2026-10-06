# earnings_drift: post-earnings announcement drift with an agent that trades

Date: 2026-10-05 · Source: `prompts/Intraday_next_strategies.md` §1 (PEAD / earnings momentum)

## Problem

The intraday continuation strategies tested so far (ORB, news_binary, vwap_pullback) buy a move already under
way and have shown no edge after costs (vwap_pullback 5Y: +0.007R per trade over 1,578 trades). Post-earnings
announcement drift is a documented anomaly with a 1–10 day horizon that fits this framework's constraints
(long only, cash account, overnight allowed) and does not depend on intraday timing.

## Goal

A new strategy `strategies/earnings_drift/`, registered in `main.py` as `"earnings_drift"`, runnable in `live`,
`paper` and `backtesting`:

```text
earnings 8-K (item 2.02) → positive EPS surprise → strong abnormal reaction that holds, on high volume
→ agent decides BUY + trailing stop → hold 1–10 sessions → exit on the trail, the agent's sell, or max hold
```

Success: the test suite and `ruff check` pass; a 1-year backtest runs end to end in agent mode and in baseline
mode; a 5-year baseline backtest runs. The strategy is judged on expectancy per trade after fees and slippage,
agent mode against baseline mode over the same window, and both against SPY on the same (survivorship-biased)
universe.

## Decisions taken in brainstorming (binding)

1. **Code gates, agent decides, baseline measures.** Code finds earnings events and applies hard gates; the
   agent makes the final buy/skip call on the gated candidates, sets the trailing stop and manages holdings. A
   code-only baseline mode (`agent_enabled=False`) takes every gated candidate with a fixed trail, so the
   agent's value is measurable (bill_ackman's LLM layers added ~0 return for 2x turnover).
2. **The agent places its own orders**, through strategy-specific order tools (`buy`, `set_trailing_stop`,
   `sell`). The order mechanics (stop on fill, cancel-and-wait before a sell) live in code behind those tools.
3. **Code guardrails, each logging a warning when it fires:** max holding period, stop backstop, order tool
   limits, and the baseline mode itself (one startup warning).
4. **Daily cadence:** one decision cycle per session, right after the close (`after_market_closes`), on the
   completed reaction day; orders fill at the next open in both backtest and live. Day bars only. (Amended
   2026-10-05 while planning: a 09:30 tick made every backtest entry fill one session late, see §2.1.)
5. **Event detection:** SEC 8-K item `2.02` filings (exact `acceptanceDateTime`); the surprise comes from the
   Benzinga "EPS ... Estimate" headline in Alpaca news, parsed in code.
6. **Trailing stops are tighten-only** once placed.
7. **Backtest data source:** `AlpacaBacktestData` (SIP).

## Non-goals

- Intraday entries or exits beyond the broker's own trailing stop.
- Shorts, margin.
- Gating or sizing on `strategy.regime` (a separate design decision, per CLAUDE.md).
- A sector relative-strength gate (needs a sector ETF per name; a later experiment).
- A dashboard page: the Backtesting tab already reads the run directory; `trades.jsonl` and
  `decisions.jsonl` are for offline analysis.
- Memory tools for the agent: each run's context already carries every holding's thesis.
- A point-in-time universe (dropped on 2026-10-04 for every strategy).

## 1. Layout

```
strategies/earnings_drift/
  __init__.py
  agent_earnings_drift.py   EarningsDriftStrategy: lifecycle hooks, agent creation, run_backtesting defaults
  parameters.py             DriftParams (pure, validated)
  events.py                 SEC submissions → EarningsEvent, reaction session (pure)
  event_source.py           SecEdgarClient wiring, paginated submissions (I/O)
  surprise.py               Benzinga headline → Surprise (pure)
  reaction.py               reaction-day features from daily bars (pure)
  screening.py              hard gates (pure)
  fact_sheet.py             Candidate → lean dict for the agent context (pure)
  scanner.py                events → news → bars → features → gates → candidates (wiring)
  desk.py                   the only order code: buy / trail / sell, stop on fill, guardrails
  tools.py                  agent tools over the desk and the decision log
  book.py                   open trades + failure streaks state file; trades.jsonl, decisions.jsonl (I/O)
  prompts.py                system prompt
```

## 2. Daily flow

`sleeptime = "1D"`; `minutes_after_closing = 0`. The work happens in `after_market_closes`, not at the open.

1. **`after_market_closes`** (16:00 in backtests; paper/live first wait `live_bar_delay_seconds` (300) with
   `strategy.sleep`, so the day's bar is final):
   1. `scanner.prepare(today)`: in paper/live it first refreshes every universe name's SEC submissions (about
      1,200 requests, a few minutes; backtests read the cache once, at the first cycle). It then builds the
      candidates from **today's** reactions (§3).
   2. `desk.begin_session(today, candidates, trading_dates)`, then `desk.reconcile()`: settle what hooks may have
      missed, then the max-hold exits and the stop backstop (§6).
   3. If there are candidates or holdings: one agent run with the context of §5. Otherwise no agent call. In
      baseline mode: `desk.baseline_entries()` instead (§7).
   4. `desk.ensure_stops()`: the stop backstop again, right after the run.
   5. Every candidate without a `buy` or `skip` is written to `decisions.jsonl` as `undecided`; the state file is saved.
2. **`on_trading_iteration`** (09:30, `sleeptime = "1D"`): raises a pending `FatalStrategyError` recorded by the
   previous cycle (§5.4, §3.6); the executor swallows every exception from the other hooks, and only
   `on_trading_iteration` may end a run. In paper/live it then runs `desk.recover()` (settle the fills and ends
   whose hooks were lost, then the stop backstop, then save; an exception is logged with its type and never
   escapes): an entry that filled at the open while the process was down (`sync_open_orders` adopts only open
   orders) gets its stop at once instead of after the close. Never in a backtest, which delivers every fill.
3. **`on_filled_order` / `on_canceled_order`** forward to the desk: a filled buy gets its trailing stop; a filled
   stop or sell closes the trade into `trades.jsonl`.

### 2.1 When the buy happens

The agent calls `buy` right after the close of the reaction session R. `Desk.buy` submits a market BUY,
`time_in_force="day"`, which fills at the open of R+1 in both modes (an after-close release on day D reacts on
D+1 = R and is bought at the open of D+2).

- **Why after the close:** `BacktestBroker` skips the bar that is forming when an order arrives ("nothing fills
  within the submitting bar", `backtesting/broker.py`). On daily bars an order sent at 09:30 skips that whole day
  and fills at the next open, one session later than live. An order sent at the exact close of a bar skips
  nothing. Live, Alpaca queues a DAY market order sent after the close for the next open.
- **Paper/live:** the entry fills at the R+1 open; the executor dispatches the fill hook while it waits for the
  close, and the hook places the stop seconds after the open. `Desk.buy` never waits for a fill.
- **Backtest:** the entry fills at the R+1 open plus slippage; the clock passes that bar's close in one jump, so
  the hook places the stop at the R+1 close and it trails from R+2. **The entry session has no stop in a
  backtest** (§11).
- A fix in the shared broker (an order sent at or before a session's open fills at that open) would be more
  correct for every daily strategy but changes their results; it is out of scope, for its own spec.

## 3. Events and candidates

### 3.1 Earnings events (`events.py`, pure)

`earnings_events(payloads, symbol) -> list[EarningsEvent]` over the submissions payload's `filings.recent`
plus every older page listed in `filings.files`: form exactly `8-K` (amendments `8-K/A` are ignored) whose `items` contain
`2.02`. `EarningsEvent(symbol, accepted_at, accession_number, primary_document)`, `accepted_at` from
`acceptanceDateTime` (aware, UTC).

**Reaction session** = the first session whose **close** is strictly after `accepted_at`: a release before the
open reacts that day, one during the session reacts that same day, one after the close reacts the next session.
`reaction_date(accepted_at, trading_dates) -> date | None` is pure over the trading dates (the benchmark's daily
bar dates) and takes every close as 16:00 ET (early closes: §11).

Two 2.02 filings for one symbol with the same reaction session collapse to the earliest.

### 3.2 Event source (`event_source.py`)

Wraps `SecEdgarClient` (`ticker_to_cik`, `get_submissions_payload`, plus the older pages
`submissions/CIK..-submissions-NNN.json` fetched through `get_json` and cached under `cache/sec/submissions/`).
Freshness follows the existing `as_of`/`max_age_days` rule with `max_age_days=0`: a file fetched after `as_of` is
served, an older one is fetched again. Paper/live load with `as_of` = now, so they refetch at every cycle; a
backtest loads once with `as_of` = the run's end (§11), so a file fetched before the end is refreshed. The parsed
events are kept in memory: a backtest loads them once, paper/live once per cycle. An older page is fetched only when its `filingTo` reaches the backtest
first cycle's date minus `event_lookback_days`. A symbol with no CIK or a failed fetch is skipped with reason `no_sec_data`.

### 3.3 Surprise (`surprise.py`, pure)

`parse_surprise(headline) -> Surprise | None` for Benzinga's form

```text
[CORRECTION: ]<Company> Q<n>|FY [Adj. ]EPS $<a> Beats|Misses|In-Line With $<e> Estimate[, Sales $<s> Beat|Miss|In-Line With $<t> Estimate]
```

Negative values as `$(0.12)` or `$-0.12`; sales units `K`/`M`/`B`. `Surprise(eps_actual, eps_estimate,
eps_surprise_pct, eps_beat, sales_actual, sales_estimate, sales_surprise_pct, sales_beat)` as `Decimal`; the
sales fields are `None` when absent. `eps_beat` is computed from the numbers (`actual > estimate`), never from
the verb. Headlines without an estimate ("Up From ...") return `None`.

`pick_surprise(articles, accepted_at)` considers articles created from `accepted_at − 2h` to the reaction
session's close; a `CORRECTION:` headline wins over the original; otherwise the earliest parsable one.

### 3.4 Reaction features (`reaction.py`, pure)

From the stock's and SPY's daily bars up to and including the reaction session R (`prev` = session before R):

| Feature | Definition |
|---|---|
| `gap_pct` | open_R / close_prev − 1 |
| `return_pct` | close_R / close_prev − 1 |
| `abnormal_pct` | return_pct − SPY's return_pct on R |
| `hold_ratio` | (close_R − close_prev) / (high_R − close_prev); `None` when high_R ≤ close_prev |
| `close_location` | (close_R − low_R) / (high_R − low_R); 0.5 when high = low |
| `rel_volume` | volume_R / mean volume of the 20 sessions before R |
| `dollar_volume_20d` | mean close × volume of the 20 sessions before R |
| `runup_20d_pct`, `runup_60d_pct` | close_prev / close 20 (60) sessions earlier − 1 |
| `atr14_pct` | 14-session ATR up to R / close_R |
| `reaction_low` | low_R |

### 3.5 Gates (`screening.py`, pure)

| Gate | Default (`DriftParams`) | Reject reason |
|---|---|---|
| Surprise parsed | — | `no_surprise_data` |
| EPS beat | `eps_actual > eps_estimate` | `eps_miss` |
| Abnormal return | `abnormal_pct ≥ min_abnormal_pct` (0.03) | `weak_reaction` |
| Holds the gain | `hold_ratio ≥ min_hold_ratio` (0.5) and `close_location ≥ min_close_location` (0.5) | `faded` |
| Relative volume | `rel_volume ≥ min_rel_volume` (2.0) | `low_volume` |
| Liquidity | `close_R ≥ min_price` (10) and `dollar_volume_20d ≥ min_dollar_volume` (20,000,000) | `illiquid` |
| Not held | no open trade, position or pending order in the symbol | `already_held` |
| Data | missing bars for R, prev or the 20-session baseline | `no_bars` |

Checked in this order, the first failure being the reason: `already_held`, `no_surprise_data`, `eps_miss`,
`no_bars`, `weak_reaction`, `faded`, `low_volume`, `illiquid`. Only events whose reaction session is **today's**
(the session that just closed) become candidates, so an event is considered once.

### 3.6 Scanner (`scanner.py`)

`prepare(today)`: the trading dates from the benchmark's daily bars (up to today); events from §3.2 for the
universe, kept when their reaction date is today; one `NewsProvider.get_news([event.symbol],
start=accepted_at − surprise_lookback_hours (2 h), end=min(clock.now(), accepted_at + surprise_window_hours (24 h)),
limit=news_limit)` per event; daily bars for those names and SPY through
`strategy.get_historical_prices_for_assets` (so backtests go through `_source_bars`); features; gates. Logs one
summary line: events, candidates, reject counts by reason. A failed news query rejects only its symbol as
`no_news`. One symbol and a narrow window per query because the provider answers the newest `limit` articles
first: a query shared by several symbols, or reaching far past the release, cuts the oldest articles, and the
Benzinga EPS headline is among the first after the release (busy names were rejected as `no_surprise_data`).

**Hollow scan:** if SEC fails for more than half of the universe (with at least 20 failures), the session has no
candidates, an error is logged and the events are not kept (the next cycle loads them again);
`hollow_scan_streak` increments (reset by a sound scan). At
`max_consecutive_hollow_scans` (3) a backtest records a pending fatal reason, raised as `FatalStrategyError` by the
next morning's `on_trading_iteration` (§2).

## 4. Fact sheet (`fact_sheet.py`, pure)

One lean dict per candidate:

```text
symbol, reported_at (ET, minute), timing: before_open | during_session | after_close
eps: {actual, estimate, surprise_pct, result: BEAT|MISS|IN-LINE}
sales: {actual, estimate, surprise_pct, result} | null
reaction: {gap_pct, return_pct, abnormal_pct, hold_ratio, close_location, rel_volume}
context: {runup_20d_pct, runup_60d_pct, atr14_pct, close, reaction_low}
max_quantity
filing: {accession_number}            (readable with get_filing_document)
headlines: up to 5 headlines (title + time) from the surprise window
```

Percentages rounded to 0.1, prices to the cent. BEAT/MISS and signs come from code.

## 5. Agent

One agent, `"drift"`, created in `initialize` at `agent_temperature` (0.3).

**Context of a run:** `date`, `candidates` (fact sheets), `holdings` (§5.2), `balances` (portfolio value, cash,
buying power, free slots = `max_positions` − open trades).

### 5.1 Tools

| Tool | Role |
|---|---|
| `buy(symbol, quantity, trail_percent, reason)` | Desk entry (§6.1) |
| `set_trailing_stop(symbol, trail_percent, reason)` | Desk trail change, tighten-only (§6.2) |
| `sell(symbol, reason)` | Desk exit (§6.3) |
| `skip(symbol, reason)` | Records a pass on a candidate in `decisions.jsonl` |
| `search_news` | existing, clock-gated |
| `get_filings`, `get_filing_document` | from `fundamentals_tools`, clock-gated (the press release, guidance) |
| `market_data_tools` | existing (historical prices, last price) |

`tool_budget = tool_budget_per_item (4) × (candidates + holdings)`; `buy`, `set_trailing_stop`, `sell` and
`skip` are exempt. Tool docstrings are one line each (token budget). The tools module has no
`from __future__ import annotations` (the agent layer reads real annotations).

### 5.2 Holdings in the context

Per open trade: `symbol, entry_date, entry_price, last_close, pnl_pct, sessions_held, trail_percent,
stop_status (working | missing | backstop), thesis, reaction_low`. Positions sold by the max-hold guardrail
in this cycle's `reconcile` are not listed. `max_holding_sessions` is never shown or stated in the prompt (a
stated cadence changes behaviour; bill_ackman).

### 5.3 Prompt (`prompts.py`)

- Judge each candidate: buy when the surprise looks real (EPS beat with sales confirming; guidance held or raised
  per the 8-K press release) and the reaction is strong and held; skip one-off beats (tax, buyback, items),
  guidance cuts, or a stretched run-up. Every candidate gets `buy` or `skip`.
- Size up to `max_quantity`; less for lower conviction.
- Trail: about 2–3× `atr14_pct`, within the tool's bounds.
- Holdings: sell when the thesis breaks (for example a close under `reaction_low`); otherwise hold; tighten the
  trail as gains build. Doing nothing is holding.
- The values in the context are facts computed by code; do not recompute BEAT/MISS.

### 5.4 Agent failures

An `AgentError`, or a run that raised, drops the session's candidates (no late entries) and leaves holdings to
their stops and the guardrails; `agent_failure_streak` increments (reset by a successful run). At
`max_consecutive_agent_failures` (3) a backtest records a pending fatal reason, which the next morning's
`on_trading_iteration` raises as `FatalStrategyError` (§2); paper/live log and carry on.

## 6. Desk (`desk.py`)

The only module that submits or cancels orders. Keeps `Trade(symbol, event accession, entry_order_id,
stop_order_id, trail_percent, thesis, opened_session, entry_price, quantity, state: pending|open|closed)` in the
book.

### 6.1 `buy(symbol, quantity, trail_percent, reason)`

Validation (each failure returns `{"error": ...}` and logs a warning naming the guardrail `order_limits`):
the symbol is one of today's candidates; not already bought this session; quantity is a positive integer
`≤ max_quantity(symbol)` recomputed at call time; open + pending trades `< max_positions`;
`min_trail_percent ≤ trail_percent ≤ max_trail_percent` (3, 15).

`max_quantity(symbol) = floor(min(portfolio_value / max_positions, available) / reaction close)` with
`available = min(buying_power, cash + proceeds of sells submitted this cycle − buys already submitted this cycle)`
(CLAUDE.md's sizing rule): the buys are deducted from the cash side only, since `buying_power` already nets every
pending order (the backtest broker's projection, Alpaca's open orders).

Then a market BUY (`day`), `Trade` recorded as `pending`, a `buy` line in `decisions.jsonl`. It never waits for
the fill (it comes at the next open, §2.1). Returns the lean order.

### 6.2 `set_trailing_stop(symbol, trail_percent, reason)`

The symbol must have an open trade; `trail_percent` within bounds and `≤` the current trail (tighten-only;
widening is an `order_limits` error). Cancel the working stop and wait for the cancel (`cancel_wait_seconds`,
the desk's `_expected_cancels` so its hook is not read as an outside cancel); place the new stop. If the new stop
is refused, place the old trail again and return `{"error": ...}`. If the old stop filled during the cancel, the
trade is closed: `{"status": "already_closed"}`.

### 6.3 `sell(symbol, reason)`

Open trade required. Cancel the stop and wait; market SELL of the held quantity (`day`); exit reason
`agent_sell`. A stop that filled during the cancel returns `{"status": "already_closed"}`.

### 6.4 Order hooks

- Filled buy: `Trade` → `open`; SELL `TRAIL`, `trail_percent`, `gtc`, for the filled quantity. A partial fill is
  protected for what filled; the remainder expires with the day order (reconcile cancels a stale entry).
- Filled stop or sell: trade closed, one line in `trades.jsonl`: `symbol, event, entry/exit time and price,
  quantity, pnl, return_pct, r_multiple` (return over the entry trail), `sessions_held`,
  `exit_reason: trail | agent_sell | max_hold | backstop_sell`, `thesis`, `agent_enabled`.
- Cancelled entry: trade dropped (or kept for the filled part).
- Stop cancelled from outside the strategy: placed again at once, with a warning (`stop_backstop`).

### 6.5 Guardrails

All log `log_warning("guardrail <name>: ...")`.

| Guardrail | When | Action |
|---|---|---|
| `max_hold` | `reconcile()`: `sessions_held ≥ max_holding_sessions` (10). `sessions_held` counts the sessions from the entry fill's session through today, both included, so the sell queued at the 10th close fills at the 11th open | cancel the stop, wait, market sell, exit reason `max_hold` |
| `stop_backstop` | `reconcile()` and `ensure_stops()`: an open trade with no working stop (errored, cancelled outside, missed hook) | place a stop at the trade's trail, or `default_trail_percent` (8) if none; holdings show `stop_status: backstop` next run |
| `order_limits` | every desk tool call | `{"error": ...}` to the agent |
| `orphan` | `reconcile()`: a position with no `Trade` (restart, manual buy) | adopted as an open trade at `default_trail_percent`, `opened_session` = today, stop placed. Only symbols this strategy traded are adopted in a shared account: an orphan in a symbol never traded by this strategy is left alone and logged once |
| `baseline` | startup with `agent_enabled=False` | one warning: agent disabled, baseline mode |

A stop that cannot be placed at all (refused twice) is replaced by an immediate market sell, exit reason
`backstop_sell` (a position is never left unprotected; vwap_pullback's rule).

## 7. Baseline mode

`agent_enabled=False`: no agent is created (no LLM configuration required). Each tick, `baseline_entries` buys
the candidates in descending `abnormal_pct` order, up to the free slots, each at `max_quantity` and
`default_trail_percent`, through the same `Desk.buy` path (`reason="baseline"`). Exits: trail and max hold only.
Everything else (reconcile, guardrails, logs) is identical.

## 8. State and logs (`book.py`)

`data/{strategy name}_state_<mode>.json` (`earnings_drift_state_<mode>.json`, and a distinct
`earnings_drift_baseline_state_<mode>.json` for the baseline, so both can run on one account without managing each
other's trades): `{version: 1, trades: [...open/pending...], agent_failure_streak, hollow_scan_streak}`. Wiped at `initialize` in backtesting only; paper/live survive restarts and `reconcile`
squares them with the broker (a trade whose position is gone is closed with reason `unknown`). A malformed or
other-version file loads as empty state with a warning.

Run directory (`logs/earnings_drift/<mode>/<run_id>/`): `trades.jsonl` (§6.4) and `decisions.jsonl` — one line
per candidate per session: `date, symbol, gate features, decision: buy|skip|undecided|baseline, quantity,
trail_percent, reason`, plus one line per rejected event with its reason. No run id (outside a runner)
disables both logs.

## 9. Parameters (`parameters.py`)

`DriftParams` (frozen, `__post_init__` validation: positive values, `min_trail ≤ default_trail ≤ max_trail`,
fractions in (0, 1]):

| Field | Default |
|---|---|
| `agent_enabled` | True |
| `max_positions` | 8 |
| `max_holding_sessions` | 10 |
| `min_trail_percent`, `default_trail_percent`, `max_trail_percent` | 3, 8, 15 |
| `min_abnormal_pct`, `min_hold_ratio`, `min_close_location`, `min_rel_volume` | 0.03, 0.5, 0.5, 2.0 |
| `min_price`, `min_dollar_volume` | 10, 20,000,000 |
| `live_volume_share` | 0.03 (paper/live scale `min_dollar_volume` by it, §11) |
| `volume_baseline_sessions` | 20 |
| `surprise_lookback_hours`, `surprise_window_hours` | 2, 24 |
| `bars_lookback_sessions` | 75 |
| `news_limit` | 50 |
| `event_lookback_days` | 10 |
| `live_bar_delay_seconds`, `cancel_wait_seconds` | 300, 30 |
| `tool_budget_per_item` | 4 |
| `agent_temperature` | 0.3 |
| `max_consecutive_agent_failures`, `max_consecutive_hollow_scans` | 3, 3 |
| `sec_hollow_fraction`, `sec_hollow_min_failures` | 0.5, 20 |

## 10. Configuration and backtest

- `main.py`: `_build_earnings_drift(broker, mode)` loads the cross_momentum universe file (as vwap_pullback
  does). Env file `env/.env.earnings_drift.<mode>`: the usual Alpaca groups, `SEC_EDGAR_USER_AGENT`, the LLM
  variables (agent mode). No new variable.
- `parameters`: `backtesting_start/end` from `PredefinedWindow.YEAR`, `benchmark_symbol` SPY,
  `warmup_trading_days` 283, `budget` 10000, `slippage` 0.0005.
- `run_backtesting` defaults: `data_source=AlpacaBacktestData` (SIP), `timestep="day"`, `preload_assets` =
  universe + SPY, `agent_telemetry=True`, the news source defaulting as for every strategy. The 5-year baseline
  runs as its own registered strategy, `earnings_drift_baseline` (`DriftParams(agent_enabled=False)`,
  `PredefinedWindow.SEMI_DECADE`, its own logs and its own state file `data/earnings_drift_baseline_state_<mode>.json`
  (§8); env file `env/.env.earnings_drift_baseline.<mode>`, else `env/.env`).
- `ConfigurationError` in `initialize` (no `SEC_EDGAR_USER_AGENT`; no model in agent mode) becomes
  `FatalStrategyError`: the strategy refuses to start.

## 11. Known limits

- **Entry session unprotected in backtests.** The stop goes on after the entry bar's close; a daily bar cannot
  order the low against the entry. Bias in either direction.
- **Early closes.** `reaction_date` takes every close as 16:00 ET: a release between an early close (13:00) and
  16:00 is mapped to that day, scanned before it was filed, and missed. A few days a year, rarely with earnings.
- **Live bars are IEX, backtest bars SIP.** Relative volume compares IEX with IEX live, SIP with SIP in backtests
  (a ratio, scale free). Dollar volume is not: IEX is ~2-3% of consolidated volume, so paper/live evaluate the gates
  with `min_dollar_volume × live_volume_share` (0.03; the scanner's `volume_share`, 1.0 in backtests). The share is
  an average: thresholds may still need retuning for live.
- **SEC freshness in backtests.** A backtest loads the events once; it uses the run's end as the cache's `as_of`
  (a bare-midnight end means through that day, so the next midnight), so a submissions file fetched before the end
  is fetched again and no 8-K filed late in the window is missing. Every other clock use stays the simulated now.
- **After-hours order changes, live.** `set_trailing_stop` and `sell` cancel and place orders after the close;
  the paper smoke run must confirm Alpaca accepts a GTC trailing stop and a DAY market sell sent after hours. The
  same must be confirmed for IBKR on a paper account (IB Gateway) before running it there: a DAY MKT order and a GTC
  TRAIL order sent after the close.
- **Trailing stop on daily bars.** `fills.evaluate_trailing_stop` checks the level from previous bars before the
  bar's high may raise it, ties against the trader: pessimistic against live, where Alpaca trails tick by tick.
- **Survivorship bias.** Today's universe; absolute returns are upper bounds; compare modes on the same universe.
- **Benzinga coverage.** Events without a parsable headline are rejected (`no_surprise_data`); the smoke script
  measures that share before the first backtest.
- **Today's fee rates** over past periods (as every backtest).
- **8-K 2.02 is not only earnings** (rarely, pre-announcements or restatements): without a parsable surprise
  headline they are rejected anyway.

## 12. Testing (no network, hand-written fakes)

- Pure: `events` (recent + paginated pages; before-open, during-session, after-close releases; holiday; non-2.02
  8-K; duplicate events), `surprise` (Beats/Misses/In-Line, negative EPS, K/M/B sales, `Adj. EPS`, FY, no sales
  part, correction preferred, "Up From" ignored), `reaction` (each formula, degenerate bars), `screening` (each
  gate's boundary and order), `fact_sheet`, `parameters` validation.
- Desk (backtest broker, fakes): buy → fill → stop; partial fill;
  every `order_limits` error; tighten-only; refused new stop re-places the old; sell cancels and waits; stop
  filled during the cancel; max hold; backstop; outside cancel; orphan adoption; stop refused twice → market
  sell; a warning per guardrail.
- Tools: `skip` and `undecided` lines; budget exemption.
- Book: round trip, wipe in backtesting, malformed file.
- Strategy: a short end-to-end backtest on a fake data source with a scripted fake agent; the same in baseline
  mode; the 3-failures `FatalStrategyError`; hollow-scan streak.
- Manual smoke script `scripts/tests/smoke_earnings_events.py` (read-only, real SEC + Alpaca news): a few names
  over a past quarter; prints events, reaction sessions, parsed surprises and the share of events with a parsed
  headline.

## 13. Docs

CLAUDE.md: architecture entry for `strategies/earnings_drift/` and a gotcha paragraph (reaction-session rule,
agent-placed orders behind the desk, guardrails and their warnings, tighten-only, the unprotected entry session
in backtests, baseline mode). README command list if strategies are listed there.

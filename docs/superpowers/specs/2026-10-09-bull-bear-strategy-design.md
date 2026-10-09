# bull_bear: a researcher → bull ∥ bear → judge debate over the cross_momentum shortlist

Date: 2026-10-09 · Source: port of lumibot's "Bull vs Bear AI Stock Trading Bot" agent example
(https://lumibot.lumiwealth.com/agents_example_bull_vs_bear_ai_stock_trading_bot.html). The original debates a
fixed list of 13 large caps every day and lets the judge trade; this port debates 15 stocks shortlisted by
cross_momentum's own ranking, once a week, and code places every order.

## Goal

A new strategy, `bull_bear`, that every Tuesday at 12:00 ET:

1. ranks the cross_momentum universe with cross_momentum's own momentum score and filters, and shortlists the
   **top 15** (plus holdings still ranked 16–35);
2. has a **researcher** write one dated-facts note per shortlisted stock (news + SEC fundamentals);
3. has a **bull** and a **bear** argue every stock from the same evidence, independently;
4. has a **judge** pick 5–10 winners;
5. sizes the winners by inverse volatility in code and rebalances.

Purpose: find out whether an LLM debate adds value on top of a momentum shortlist. The comparison is against
cross_momentum runs; **no baseline variant** is built (decision 2026-10-09).

Runs in paper, live and backtesting. Success: `uv run pytest`, `uv run ruff check` and `uv run pyright` pass;
cross_momentum and bill_ackman behave exactly as before the code moves; `uv run agent bull_bear backtesting` runs a
multi-week window, writes `reviews.jsonl` and trades on Tuesdays only.

## Decisions (binding)

| Topic | Decision |
|---|---|
| Shortlist | Top 15 of cross_momentum's score after cross_momentum's filters, same code (`strategies/common/scoring.py`) |
| Cadence | Weekly, Tuesday, `iteration_start_time` 12:00 ET (`sleeptime = "1D"`, non-Tuesdays return at once) |
| Who decides what | The judge picks stocks; code sizes and trades. No agent has an order tool |
| Holdings | Kept in the debate while ranked ≤ 35; ranked > 35 or unranked = forced exit by code |
| Evidence | Code fact sheet + one researcher note per stock; bull, bear and judge have no research tools |
| Overlays | None (no vol targeting, breadth or GLD/IEF sleeve). Possible follow-up spec if the debate shows an edge |
| Baseline | None |

## Package layout

### `strategies/common/` (new; code shared between strategies, moved without behaviour change)

- `scoring.py` (**pure**) — moved out of `CrossMomentumStrategy._compute_indicators_for_ticker` /
  `compute_target_portfolio` and `cross_momentum/utils.py`: `compute_return_from_prices`, `annualized_volatility`,
  `momentum_score`, `apply_filters`, plus
  - `score_stock(symbol, df, params) -> MomentumRow | None`: `df` is the completed-session daily frame (the caller
    drops today's bar with `completed_bars`); returns `None` when history is short (`min_trading_days`), a return
    is missing, or `apply_filters` fails. `MomentumRow` (frozen): `symbol, price, ret_12_1m, ret_6_1m, ret_3m,
    volatility, avg_dollar_volume, trading_days, score, closes` (closes kept for the fact sheet and cross_momentum's
    breadth/risk code).
  - `rank(rows) -> list[RankedRow]`: sorted by score, descending, ranks 1..n.
  - `params` is cross_momentum's `CONFIG` (the keys it already reads). `CrossMomentumStrategy` calls these instead
    of its own method body; its ATR and `close_series` extras stay in cross_momentum.
- `rebalancer.py` — bill_ackman's `Rebalancer` and `PlacedOrder`, moved. Its parameter type becomes a
  `RebalanceParams` protocol (`parking_symbol`, `rebalance_band`, `min_trade_pct`, `cash_buffer`), satisfied by
  `AckmanParams` and `BullBearParams`. Docstring gains the cash invariant (see "Orders").
- `portfolio.py` — `TargetPortfolio` and `target_portfolio`, moved from bill_ackman.
- `yahoo_daily_bars.py` — `YahooDailyBars`, moved from cross_momentum.
- `sector_provider.py` — `SectorProvider`, moved from cross_momentum.

bill_ackman and cross_momentum import from `common/`; no re-export shims are left behind.

### `strategies/bull_bear/` (new)

| Module | Role |
|---|---|
| `agent_bull_bear.py` | `BullBearStrategy`: `initialize`, `on_trading_iteration`, backtest preload |
| `parameters.py` | `BullBearParams` (frozen, validated in `__post_init__`) |
| `debate_set.py` | **pure**: ranked rows + holdings → debate set and forced exits |
| `fact_sheet.py` | **pure**: one row per debate-set stock |
| `sizing.py` | **pure**: `capped_inverse_volatility` |
| `handoff.py` | the submit tools and `HandoffRecorder` (validators pure); no `from __future__ import annotations` |
| `prompts.py` | the four system prompts and the per-stage user prompts |
| `pipeline.py` | `ReviewPipeline`: one review, data → research → bull → bear → judge → rebalance |
| `state.py` | `StateStore` (`data/bull_bear_state_<mode>.json`) and `ReviewLog` (`reviews.jsonl`) |

Registered as `"bull_bear"` in `utils/strategy_factory.py`'s `Strategies`; `build_strategy` returns `None` with a
warning when the universe file is missing (same rule as the other universe strategies).

## Parameters (`BullBearParams`)

| Name | Default | Meaning |
|---|---|---|
| `shortlist_size` | 15 | top-ranked stocks debated |
| `retention_rank` | 35 | a holding ranked worse is a forced exit |
| `min_picks` / `max_picks` | 5 / 10 | judge's pick count |
| `min_weight` / `max_weight` | 0.04 / 0.20 | per-stock weight bounds |
| `cash_buffer` | 0.02 | never invested |
| `rebalance_band` | 0.03 | drift (fraction of portfolio value) that triggers a trade |
| `min_trade_pct` | 0.005 | smallest order |
| `parking_symbol` | `"SHV"` | where unspent money goes |
| `research_tool_budget` | 3 | tool calls per researcher run (`submit_note` exempt) |
| `note_max_chars` / `argument_max_chars` / `reason_max_chars` | 500 / 300 / 300 | hand-off text limits |
| `max_consecutive_abandoned` | 3 | abandoned reviews in a row that abort a backtest |
| `agent_temperature` | 0.3 | all four agents; `None` = server default |
| `rebalance_time` | `"12:00"` | ET, validated with `parse_rebalance_time` |
| `rebalance_weekday` | 1 | Tuesday |
| `yahoo_retry_delays` | (60, 180) | seconds, through `Strategy.sleep` |
| `min_yahoo_coverage` | 0.5 | share of the universe a Yahoo batch must cover |

Validation: `min_weight > rebalance_band` (the rebalancer buys a new stock only past the band);
`max_picks × min_weight ≤ 1 − cash_buffer`; `min_picks × max_weight ≥ 1 − cash_buffer` (so the book is always
fully invested); `1 ≤ min_picks ≤ max_picks ≤ shortlist_size`; `shortlist_size < retention_rank`; finite values,
fractions in range, `parking_symbol` not blank.

Momentum parameters (`w_12m`, `w_6m`, `w_3m`, `skip_days`, `volatility_window`, `min_price`,
`min_dollar_volume`, `max_volatility`, `min_trading_days`) are **read from cross_momentum's `CONFIG`**, not
duplicated, so the shortlist is cross_momentum's ranking by construction.

## One review (`ReviewPipeline.run`)

### 1. Data and ranking

- Universe: `load_cross_momentum_universe()` (`data/universe/us_stock_universe.json`, ~1,200 symbols). Using
  today's file in a backtest carries survivorship bias, the same caveat as cross_momentum.
- Paper/live: one `YahooDailyBars` batch for the universe only (300 completed bars; a holding outside the universe
  is never requested, so it is unranked and a forced exit, as in cross_momentum; prices for orders come from
  `get_last_price`, not from this batch). The batch is
  **unusable** when the download fails, it covers < `min_yahoo_coverage` of the universe, or it misses a held
  universe stock. Unusable → fetched again after each of `yahoo_retry_delays`; still unusable → review abandoned
  (stage `data`). Same rules and reasons as cross_momentum (IEX volume would gut the dollar-volume filter).
- Backtest: daily bars through `get_historical_prices` (the `_source_bars` gate), Yahoo data source by default.
  `run_backtesting` preloads the universe + SHV (as `CrossMomentumStrategy._backtest_preload_assets`).
- Every frame goes through `completed_bars(df, market_date)` then `score_stock`; survivors are `rank`ed.

### 2. Debate set (`debate_set.build`, pure)

Input: ranked rows, current holdings (excluding SHV). Output: `DebateSet(stocks, forced_exits)`.

- `stocks` = the top `shortlist_size` by rank, plus every holding with `shortlist_size < rank ≤ retention_rank`,
  tagged `held`. Size 15–25.
- `forced_exits` = holdings that are unranked (no data, failed a filter, left the universe) or ranked
  `> retention_rank`, each with its reason (`unranked` / `rank N`). They never reach the agents.

### 3. Fact sheet (`fact_sheet.rows`, pure)

One row per debate-set stock: `symbol, rank, score, ret_12_1m, ret_6_1m, ret_3m, ret_1m, volatility,
drawdown_from_52w_high, pct_vs_sma200, sector, held, weight` (current weight counting open orders, from
`Rebalancer.current_weights`; 0 when not held). `ret_1m`, the drawdown and the SMA200 distance are computed from
`MomentumRow.closes`. `sector` from `SectorProvider` (yfinance, not point-in-time; accepted, sectors rarely change);
`"UNKNOWN"` on failure. Numbers rounded for the prompt (returns and weights to 1 decimal %).

### 4. Researcher — one run per stock

- Tools: `only(news_tools(self) + fundamentals_tools(self), ["search_news", "get_income_statement",
  "get_balance_sheet"])` + `submit_note`. All clock-gated (no look-ahead). `tool_budget = research_tool_budget`,
  `exempt_tools=["submit_note"]`.
- Input: that stock's fact-sheet row. Output: `submit_note(symbol, note)` — dated facts only (news, latest
  revenue/earnings trend, debt), no opinion, ≤ `note_max_chars`. The recorder refuses another symbol, an empty
  note or an over-long one.
- A run with no valid note after the forced retry (or raising `AgentError`) gives the note
  `"research unavailable"`; the review continues. The count of failed notes goes to `reviews.jsonl`.
- A `ConfigurationError` raised by the news or SEC layer propagates (it means the strategy is misconfigured).

### 5. Bull, then bear (independent, same evidence)

- Input to both: the full fact sheet + every note. No tools but their submit tool. The bear never sees the bull's
  case (run sequentially, not in parallel: same result as lumibot's `run_together`, no threads in strategy code).
- `submit_bull_case(cases=[{symbol, conviction, argument}])`, `conviction ∈ {low, medium, high}`.
- `submit_bear_case(cases=[{symbol, risk, concern, argument}])`, `risk ∈ {low, medium, high}`, `concern ∈
  {valuation, momentum_exhaustion, earnings, fundamentals, news_event, sector, none}`.
- Both must cover every debate-set symbol exactly once; unknown or duplicate symbols, bad enums and over-long
  arguments are refused with `{"error": ...}`. No quota forces the bear to flag anything; the conviction and risk
  distributions are logged per review (to spot a rubber-stamp, as seen with Ackman's short seller).

### 6. Judge

- Input: fact sheet, notes, both cases, current holdings with weights.
- `submit_picks(picks=[{symbol, reason}], drops=[{symbol, reason}])`:
  - `min_picks ≤ len(picks) ≤ max_picks`, unique, all in the debate set;
  - `drops` = exactly the held debate-set stocks not picked, each with a reason;
  - reasons ≤ `reason_max_chars`.
- The pick order is kept: it is the buy order.

### Hand-off rules (all stages, same as bill_ackman)

`HandoffRecorder` is armed per stage with what that stage may submit; the first valid submission is final; free
text (including the last message) is logged and ignored. A stage with no valid submission is run once more with
`force_tool=<its submit tool>` and a prompt quoting the last tool error. Prompts never state the weekly cadence.

### 7. Sizing (`sizing.capped_inverse_volatility`, pure)

`weights ∝ 1 / volatility` over the picks, then water-filled to `[min_weight, max_weight]` summing to
`1 − cash_buffer`: clamp, redistribute the excess or deficit over the unclamped names, repeat until stable. Both
bounds hold exactly (unlike cross_momentum's `inverse_volatility_weights`, which renormalises after capping). A
non-positive or missing volatility is a programming error (`score_stock` never returns one) and raises.
`target_portfolio(weights, cash_buffer=...)` builds the `TargetPortfolio`; parking weight is ~0 by construction.

### 8. Orders (`common.rebalancer.Rebalancer`, moved unchanged)

`rebalance(target, forced_exits)`: sells first (forced exits, drops, trims beyond the band), then buys in pick
order, then SHV. Buys are sized against `min(buying_power, cash + this run's sell proceeds + earlier open orders'
net credit) − buffer`.

**Cash invariant (no double count of fills, cf. congress_trades `275c2f3`, earnings_drift `c652426`).** The
account is read **once, before any order and before positions and open orders**; `cash` is never read again in
the call. Sell proceeds are added to that pre-sell snapshot, so a live sell that fills within seconds is counted
once. `buying_power` is re-read after the sells but only inside `min()`. A refused buy replaces `available` with
the broker's figure, never adds to it. Earlier open orders count only their unfilled part. bull_bear calls
`rebalance` **once per review**; there is no second trading pass. Pinned by two new tests (below).

## State, logs, lifecycle

- `StateStore` → `data/bull_bear_state_<mode>.json`, `STATE_VERSION = 1` (another version loads as empty):
  `last_completed_review` (market date), `abandoned_streak`, `last_picks` (symbol, reason, date). Wiped in
  `initialize` in backtesting mode only.
- A Tuesday whose review already completed is not run again (restart after 12:00). Unlike cross_momentum, which
  reruns on purpose: an LLM review is too costly to repeat.
- `ReviewLog` → `reviews.jsonl` in the run directory, one line per review: date, debate set (symbol, rank, held),
  forced exits with reasons, notes written/failed, bull conviction and bear risk counts, picks and drops with
  reasons, target weights, orders placed, outcome (`completed` | `abandoned` + `stage` + `reason`).
- `initialize`: builds the four agents (`self.agents.create(..., temperature=agent_temperature)`) and the state
  store; a `ConfigurationError` (no LLM, no `SEC_EDGAR_USER_AGENT`, no news credentials) becomes
  `FatalStrategyError`.
- `on_trading_iteration`: returns unless the market date is a `rebalance_weekday` and that date's review has not
  completed; then runs the pipeline. Raises `FatalStrategyError` from here when a backtest reaches
  `max_consecutive_abandoned`.

## Abandoning a review

A review is abandoned — **no order at all, forced exits included**, nothing in the state changes except
`abandoned_streak + 1`, the reason goes to `reviews.jsonl` — when:

- the data is unusable after the retries (stage `data`);
- the bull, the bear or the judge has no valid submission after the forced retry, or raises `AgentError`
  (stage `bull` / `bear` / `judge`);
- a `BrokerError` / `BacktestError` is raised before the rebalancer starts (stage `broker`), or during it (stage
  `execution`, with `Rebalancer.placed` logged: those orders are already out).

A completed review resets `abandoned_streak` to 0. Backtest: `FatalStrategyError` at `max_consecutive_abandoned`
in a row; paper/live log and carry on.

## Testing

Hand-written fakes, no network, no `MagicMock`.

- **Moves first, behaviour unchanged.** Before moving: a test pinning `score_stock`'s output to cross_momentum's
  current per-ticker result on a fixed frame. After: existing cross_momentum and bill_ackman tests pass with
  imports changed only; rebalancer tests move to `tests/strategies/common/`.
- **Rebalancer cash regression** (fake broker that fills every order at submission and moves `cash` at once):
  a rotation (sell A, buy B) spends at most pre-sell cash + A's proceeds − buffer; two buys in a row, the second
  is sized on what the first left, not on the original cash minus the first's cost again.
- **Pure:** `capped_inverse_volatility` (bounds hold, sum = 0.98, 5 equal vols → 0.196 each, a very volatile pick
  floored at 0.04); `debate_set.build` (top 15, holdings 16–35 added and tagged, > 35 and unranked forced out with
  reasons, SHV ignored); fact-sheet fields; `BullBearParams` validation.
- **Hand-offs:** coverage exactly once, unknown/duplicate symbol, bad enum, over-long text, pick count bounds,
  `drops` exactly the unpicked holdings, first valid submission final.
- **Pipeline** (scripted fake agents): completed review end to end; one note failing → `"research unavailable"`;
  bull / bear / judge failing → forced retry → abandoned with no orders (forced exits included); unusable Yahoo
  batch with retries then abandon; `BrokerError` before and during the rebalance; a completed Tuesday not rerun;
  3 abandoned in a backtest → `FatalStrategyError`.
- **Strategy:** registered in `Strategies`; non-Tuesday does nothing; `ConfigurationError` in `initialize` →
  `FatalStrategyError`.
- **Backtest:** like `test_ackman_backtest.py`, fake agents and fake data source over two Tuesdays: orders filled,
  `reviews.jsonl` written.

## Docs

- `CLAUDE.md`: architecture entries for `strategies/common/` and `strategies/bull_bear/`; update the cross_momentum
  and bill_ackman entries for the moved modules.
- `README.md`: env file `env/.env.bull_bear.<mode>` (LLM, `SEC_EDGAR_USER_AGENT`, `ALPACA_NEWS_*`, broker keys) and
  `uv run agent bull_bear <mode>`.

## Risks and open points

- **Cost and duration:** 15–25 researcher runs + 3 debate runs per week. A multi-year backtest on the local 27B
  model is long; start with a few months.
- **Rubber-stamp / herd:** the bull may rate everything high and the bear everything low (momentum names look
  good). Watch the logged distributions before adding quotas.
- **News in backtests** comes from Alpaca's news API (Benzinga); thin names may have no news, so notes lean on SEC
  figures.
- **Survivorship bias** from today's universe file, as in cross_momentum.

## Out of scope

- Risk overlays (vol targeting, breadth, GLD/IEF sleeve) — a follow-up spec if the debate shows an edge.
- A code-only baseline variant.
- Rebuttal rounds between bull and bear.
- Agent memory tools.

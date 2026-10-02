# Bill Ackman portfolio strategy: three agents over the quality screen

Date: 2026-10-02 · Branch: `feature/bill-ackman-strategy` · Second of two specs. The first,
`2026-10-02-fundamentals-quality-screen-design.md`, built the screen this strategy consumes (merged).

## Problem

lumibot's
[Bill Ackman Portfolio AI Trading Bot](https://lumibot.lumiwealth.com/agents_example_bill_ackman_portfolio_ai_trading_bot.html)
is a three-agent daily strategy. A researcher ranks the 5 best simple, cash-rich, fairly priced companies of a fixed
10-ticker universe. A short seller attacks each idea. A trader holds the 3 to 5 that survive, with more money in the
best ones, and sells a stock that no longer survives. Each agent's prompt is two sentences, each hands free text to the
next, and the trader sizes and places its own orders. On a local model that shape loses a day's decision whenever one
agent rambles, mis-sizes an order or narrates a trade it never placed (all observed on `news_binary`).

This framework has no such strategy. The first spec built the part that does not need an LLM: a code screen that cuts
the 1,200-symbol universe file to at most 15 candidates with the numbers that justify each one. This spec builds the
strategy on top of it.

## Goal

`BillAckmanStrategy`, registered as `"bill_ackman"`, runnable in `live`, `paper` and `backtesting`:

```
uv run agent bill_ackman backtesting
```

One review per trading session, in this fixed order: **screen (code) → researcher → short seller → trader → rebalancer
(code)**. The agents decide; code validates, remembers and places every order.

Success:
- The test suite and `ruff check` pass, and the moved screen passes the same tests it passed before the move.
- A backtest over the default window completes, writes `reviews.jsonl` and the usual run report, and no order is placed
  by an agent.
- A day on which an agent fails to submit a valid result ends with the book unchanged and a logged reason, never with a
  half-executed rebalance.

## Decisions taken in brainstorming (binding)

| Topic | Decision |
|---|---|
| Purpose | A hardened strategy for paper/live on the local model, faithful to the three roles, not to the two-sentence prompts |
| Execution | The trader's only output is a target portfolio through one tool call; a code rebalancer places the orders |
| Cadence | A full review every session, as on the page |
| Candidates | The quality screen over the universe file, top 15 |
| Company data | A code-built fact sheet per company in the context, plus drill-down tools |
| Hand-off | Each agent ends with one structured submit tool; code validates it and passes on only that data |
| Stability | Code hysteresis (2 consecutive fails force an exit) plus yesterday's ranking in the context; no memory tools |
| Review set | The researcher's top 5 plus every current holding |
| Short book | Unallocated money is parked in SHV, by code |
| Backtest cost | Short default window (`PredefinedWindow.BI_MONTH`, about 2 months), the full daily review with no shortcuts |
| Screen location | The screen moves to `strategies/bill_ackman/screen/`; the SEC client stays in `fundamentals/` |

## Non-goals

- **No orchestrator agent**, no LangGraph, no supervisor. The strategy code calls the three agents in order.
- **No agent has an order tool.** The page's `allow_trading=False/True` flag is replaced by the stricter rule that none
  of the three can trade.
- **No shorting, no margin, no options.** A long-only book of at most 5 stocks plus SHV.
- **No skipping of unchanged days** and no weekly cadence in the agents (rejected for backtest honesty).
- **No agent memory tools** (`search_memory`, theses). Continuity is the code state of §6.
- **No dashboard view** of `reviews.jsonl`. The file is written so a later view can read it.
- **No change to the cross-momentum rebalancer.** Two small pure order helpers it owns are moved to `utils/helpers.py`
  (§8); its behaviour is untouched.

## 1. Layout, and moving the screen

Everything strategy-specific lives in `src/trading_agent_framework/strategies/bill_ackman/`:

| Module | Kind | Job |
|---|---|---|
| `__init__.py` | | exports `BillAckmanStrategy` |
| `agent_bill_ackman.py` | strategy | `BillAckmanStrategy`: creates the agents, one review per tick, `run_backtesting()` |
| `pipeline.py` | orchestration | `ReviewPipeline`: one review from screen to target portfolio, retries, abandonment |
| `fact_sheet.py` | pure | a screen `Candidate` plus price facts to the compact dict the agents read |
| `handoff.py` | pure + tool factories | `HandoffRecorder`, the validation rules and the three submit tools |
| `hysteresis.py` | pure | fail counters, forced exits, the allowed set for the trader |
| `portfolio.py` | pure | the trader's weights to target weights, with the SHV remainder |
| `rebalancer.py` | order code | the only module that places orders |
| `state.py` | I/O | persisted counters and last review; the `reviews.jsonl` writer |
| `parameters.py` | data | `AckmanParams`, one frozen dataclass of thresholds |
| `prompts.py` | data | the three system prompts |
| `screen/` | moved | `quality.py`, `screen.py`, `splits.py`, `annual_store.py`, and new `annual_figures.py` |

### 1.1 The move (a pure move, no behaviour change)

The first task of the plan, in its own commit.

- **To `strategies/bill_ackman/screen/`**: `fundamentals/quality.py`, `screen.py`, `splits.py`, `annual_store.py`.
- **New `screen/annual_figures.py`**: the screen-only code taken out of `fundamentals/sec.py` --
  `annual_figures`, `parse_sic`, `ANNUAL_FLOW_TAGS`, `MIN_FISCAL_YEAR_DAYS`, `MAX_FISCAL_YEAR_DAYS` and their private
  helpers (`_tag_rows`, `_is_fiscal_year`, `_slim`, `_first_source_per_period`, `_largest_per_period`,
  `_debt_sources` and the like).
- **Stays in `fundamentals/`**: `edgar_client.py`, `freshness.py` (the generic client uses it, §7), and `sec.py`
  without the code above. `sec._parse_dt` becomes the public `sec.parse_dt` so `annual_figures.py` can import it.
- **`fundamentals/__init__.py`** exports only `SecEdgarClient`. The screen exports (`Candidate`, `QualityScreen`,
  `ScreenParams`, `ScreenResult`, `build_quality_screen`) move to `strategies/bill_ackman/screen/__init__.py`.
- **Tests**: `tests/fundamentals/` keeps the client, `sec.py` and `freshness` tests. `test_quality.py`, `test_screen.py`,
  `test_splits.py`, `test_annual_store.py`, `test_screen_integration.py`, the part of `test_sec.py` that covers
  `annual_figures`/`parse_sic`, and `annual_fixtures.py` move to `tests/strategies/bill_ackman/screen/`.
- **`scripts/tests/smoke_quality_screen.py`** keeps its path and gets new imports.
- **Docs**: the `fundamentals/` bullet of `CLAUDE.md` is split in two, and the first spec gets a one-line note that the
  screen now lives in the strategy.
- **Evidence the move changed nothing**: the same number of tests pass, and the smoke run prints the same candidates.

One small API addition in the moved code: `QualityScreen.run(..., top_n: int | None = None)`, where a value overrides
`params.top_n` for that call (used in §2 step 2 to screen the holdings without truncation). `None` keeps today's
behaviour.

## 2. The daily review

`BillAckmanStrategy.on_trading_iteration` runs `ReviewPipeline.run()` once per session (`sleeptime "1D"`). Steps:

1. **Screen the universe (code).** `screen.run(universe, as_of=<market-local now>, price_of=self.get_last_price)`.
   `universe` is `load_cross_momentum_universe()`; `as_of` is `self.clock.now().astimezone(MARKET_TZ)` (the screen reads
   `as_of.date()`). Result: at most 15 `Candidate`s.
2. **Screen the holdings (code).** The strategy's holdings are every position except SHV. `screen.run(holdings, ...,
   top_n=len(holdings))` gives each holding its metrics, or a rejection reason. A holding the universe screen did not
   return is thereby still described. A strategy account is assumed dedicated, as for `cross_momentum`.
3. **Researcher.** Context: the 15 fact sheets, the holdings (symbol, weight) and yesterday's ranking. It calls
   `submit_ranking`.
4. **Review set (code).** The ranked symbols plus the holdings, de-duplicated.
5. **Screen verdicts (code).** A holding the holdings screen rejected for a **quality** reason (`insufficient_history`,
   `stale_filing`, `operating_loss`, `negative_fcf`, `shrinking_revenue`, `debt_unknown`, `too_much_debt`,
   `excluded_sector`) gets the verdict `fail` from code, with the reason `screen: <reason>`, and is not sent to the short
   seller. A holding rejected for a **data** reason (`no_data`, `no_price`, `no_split_data`, `duplicate_listing`) is still
   sent, with a reduced fact sheet (price facts and `screen_unavailable: <reason>`), and the short seller judges it from
   the tools.
6. **Short seller.** Context: the fact sheets of the review set it must judge, the researcher's reasons, and each
   holding's fail counter. It calls `submit_verdicts`.
7. **Hysteresis (code, §4).** Fail counters update, forced exits are decided, the trader's allowed set is built.
8. **Trader.** Context: the allowed names (survivors in research order, then holdings with a pending fail, each with its
   counter), the forced exits (not allowed), current weights and the constraints. It calls `submit_portfolio`.
9. **Targets and execution (code, §4-§5).** Target weights, then the rebalancer places the orders.
10. **Record (code, §6).** The review is appended to `reviews.jsonl`, the counters are persisted.

An abandoned review (§9) skips steps 7 to 9: nothing is traded, counters are not updated.

## 3. Hand-off contracts (`handoff.py`)

A `HandoffRecorder` holds, for the current review, what each submit tool must validate against and the one accepted
submission. Before each agent runs, the pipeline arms the recorder for that stage (for example, `expect_ranking(
candidates)`). The tools are closures over the recorder. **The module that defines them has no
`from __future__ import annotations`**: the agent layer reads real annotations (as `memory/tools.py` does). Each tool
has a one-line docstring, because both the schema and the docstring reach the LLM on every call.

A tool returns `{"status": "recorded"}` for a valid submission. For an invalid one it returns `{"error": "<what is
wrong and what is allowed>"}` and records nothing, so the model can correct it in the same run. The first valid
submission is final: a second call returns `{"error": "already recorded for this review"}`.

| Tool | Argument | Valid when |
|---|---|---|
| `submit_ranking` | `ideas: list[{symbol, reason}]` | 1 to `research_top_n` (5) items, or all of them if fewer candidates; unique symbols, all among the screen's candidates; `reason` non-empty and at most 300 characters |
| `submit_verdicts` | `verdicts: list[{symbol, verdict, reason}]` | exactly one item per symbol the short seller was asked to judge, no others; `verdict` is `survive` or `fail`; `reason` as above |
| `submit_portfolio` | `positions: list[{symbol, weight, reason}]` | at most `max_positions` (5); unique symbols, all in the allowed set; each `weight` between `min_weight` (0.05) and `max_weight` (0.35); the weights sum to at most `1 - cash_buffer` (0.98); `reason` as above. An empty list is valid and means everything goes to SHV |

`weight` is a fraction of portfolio value. Symbols are upper-cased and trimmed before validation. A symbol the agent
invented is an error naming the allowed list.

## 4. Hysteresis and target portfolio

### 4.1 Fail counters (`hysteresis.py`, pure)

State: `fail_counts: dict[str, int]`, one entry per holding that has failed and not yet recovered.

For every **holding** in the review set with verdict `v`:
- `survive`: the counter resets (the entry is removed).
- `fail`: the counter goes up by one.

A holding whose new counter is at least `forced_exit_fails` (2) is a **forced exit**: code sells it in full whatever
the trader submits, and it is not in the trader's allowed set. A symbol that is no longer held loses its counter.

A new candidate that fails the attack is simply not allowed; it has no counter.

### 4.2 The trader's allowed set

`allowed = {ranked symbols with verdict survive} ∪ {holdings with verdict survive} ∪ {holdings with verdict fail and
counter < forced_exit_fails}`. The last group is a **pending fail**: kept eligible, flagged to the trader with its
counter. The trader may still drop a pending or surviving holding by leaving it out; forced exits are the only sells
code imposes beyond what the trader leaves out.

### 4.3 Target weights (`portfolio.py`, pure)

Input: the trader's validated positions and the cash buffer. Output: a `TargetPortfolio`: a weight per stock, plus the
SHV weight `1 - cash_buffer - sum(stock weights)`. The cash buffer stays as cash for fees and fill drift. Nothing is
rescaled: a weight the trader chose is the weight targeted. A review with no survivors and no holdings targets 98% SHV.

## 5. The rebalancer (`rebalancer.py`)

The only module that places orders. It follows the proven shape of `cross_momentum.rebalance` and the two `CLAUDE.md`
rules for cash accounts, without importing from that strategy.

1. **Current weights** come from positions times last price over portfolio value; a price lookup that fails values the
   position at 0 and logs a warning rather than aborting after sells are submitted.
2. **Sells first.** Forced exits and every held stock not in the targets, in full. A held target stock above its target
   by more than `rebalance_band` (5 points of portfolio value): the excess. SHV above its target by more than the band:
   the excess, and also whatever is needed to fund the buys.
3. **Buys.** Each target stock below its target by more than the band, up to the cash available, which is the **smaller
   of `buying_power` and cash plus the estimated proceeds of the sells submitted in this run, minus the cash buffer**
   (never `buying_power` alone: on a margin account it exceeds cash; never `min(cash, buying_power)`: raw cash ignores
   the sell credit).
4. **SHV last.** The money the stock buys leave goes to SHV, up to its target, if it is below target by more than the
   band.
5. **Quantities** use `fractional_qty`; an order that would be smaller than `min_trade_pct` (0.5%) of portfolio value is
   skipped. A target weight is never below `min_weight`, which is at least the band, so a chosen stock not yet held
   (current weight 0) always exceeds the band and is bought.
6. **Errors.** A rejected order is logged and does not stop the others. A buy rejected for buying power resyncs the
   available cash from the broker's reported value (`parse_insufficient_buying_power`), as `cross_momentum` does.

`rebalancer.py` returns the list of submitted orders (symbol, side, quantity) for the log. It imports no LangChain.

## 6. State and the review log (`state.py`)

- **State file**: `<project_root>/data/bill_ackman_state_<mode>.json`:
  `{"version": 1, "last_review": "YYYY-MM-DD", "fail_counts": {...}, "last_ranking": [...], "last_verdicts": {...},
  "abandoned_streak": 0}`. It is written atomically after each completed review (temporary file in the same directory,
  then replace). A corrupt or missing file is an empty state with a logged warning.
- **Backtests start clean**: `run_backtesting` deletes the state file first, so one run cannot leak into the next
  (same rule as `cross_momentum`'s history file and the agent memory). Paper and live state is never wiped, so the
  counters survive a restart.
- **Review log**: `reviews.jsonl` in this run's directory (`logs/<strategy>/<mode>/<run_id>/`, as `vwap_pullback` writes
  `trades.jsonl`; no log when there is no run id). One line per review: date; run id; the candidates (symbol, rank,
  score); the ranking; the review set; each verdict with its source (`llm` or `screen`); the counters before and after;
  the forced exits; the allowed set; the submitted portfolio; the targets including SHV; the orders; and `abandoned`
  with the stage and the error when applicable. Decimals as strings.

## 7. The drill-down tools' cache (the gap spec 1 left)

`agents/tools/fundamentals.py` reads raw SEC payloads that `SecEdgarClient.get_json` caches forever, so in live an agent
can read financials that miss the latest 10-Q. The fix, in the generic client:

- `get_company_facts_payload`, `get_submissions_payload` and `get_json` gain optional `as_of: datetime | None = None`
  and `max_age_days: int | None = None`. With both given, a cached file is stale when its modification time is more
  than `max_age_days` before `as_of` (the existing `freshness.is_stale` rule); a stale file is refetched and rewritten,
  and if that refetch fails the stale file is used with a warning. Without them the behaviour is exactly today's, so no
  other caller changes.
- `fundamentals_tools` passes `strategy.clock.now()` and 30 days.
- A backtest never refetches an existing file (its date is today and `as_of` is in the past); live refreshes monthly.

The filing-text cache (`get_text`) is immutable per accession number and is left as is.

## 8. Shared helpers

The rebalancer needs `fractional_qty` and `parse_insufficient_buying_power`, which live in
`strategies/cross_momentum/utils.py`. A strategy must not import another strategy's utilities, so both move, unchanged
and with their tests, to `utils/helpers.py` (which `cross_momentum` already imports from), and `cross_momentum/utils.py`
re-imports them under the same names so its tests and call sites keep working. This is the only edit to
`cross_momentum`, and it is a move.

The portfolio snapshot that `news_binary` builds for its agent (balances and positions with `pct_of_portfolio`) is useful
to the trader too. It is **not** extracted here: the trader's context carries current weights computed by
`rebalancer`'s own weight function (§5.1). Extraction would touch `news_binary` and is a separate refactor.

## 9. Failure handling

- **Invalid submission**: the tool returns an error the model can read (§3).
- **No valid submission after an agent run**: one forced retry, `AgentHandle.run(..., run_id=<same id>, force_tool=
  "submit_<stage>")`, with a corrective prompt that quotes the last tool error. The retry reuses the run id so the stage
  stays one logical run.
- **Still none**: the review is **abandoned**: nothing is traded, counters are untouched, the stage and error are
  logged and written to `reviews.jsonl`, `abandoned_streak` goes up. A successful review resets it.
- **Backtests**: `abandoned_streak >= max_consecutive_abandoned` (3) raises `FatalStrategyError` (the executor reports
  it and writes no report), as `news_binary` does after 3 failed runs. Paper and live log an error and carry on.
- **Screen failures**: a `FundamentalsError` from either screen run (a hollow screen: SEC or Yahoo down) abandons the
  review the same way. A `ConfigurationError` (missing `SEC_EDGAR_USER_AGENT`) is raised from `initialize` as a
  `FatalStrategyError`: the strategy refuses to start rather than fail daily.
- **An `AgentError` from a run** counts as no submission for that stage (the retry applies once).
- **Agent hand-off is the only LLM output the strategy trusts.** Free text from an agent, including its final message,
  is logged and otherwise ignored.

## 10. Configuration and backtest setup

`AckmanParams` (`parameters.py`, a frozen dataclass; every value below is a default):

| Parameter | Default | Meaning |
|---|---|---|
| `screen` | `ScreenParams()` | the screen's own parameters; `top_n` 15 |
| `research_top_n` | 5 | ideas the researcher submits |
| `max_positions` | 5 | stocks in the target |
| `max_weight` / `min_weight` | 0.35 / 0.05 | per-stock weight bounds; `min_weight >= rebalance_band` |
| `cash_buffer` | 0.02 | share of portfolio value never invested |
| `forced_exit_fails` | 2 | consecutive fails that force an exit |
| `rebalance_band` | 0.05 | drift, in portfolio fraction, that triggers a trade |
| `min_trade_pct` | 0.005 | smallest order, as a fraction of portfolio value |
| `parking_symbol` | `"SHV"` | where unallocated money goes |
| `max_consecutive_abandoned` | 3 | abandoned reviews that abort a backtest |
| `reason_max_chars` | 300 | longest `reason` a submit tool accepts |

`AckmanParams.__post_init__` validates the ranges (positive counts, `0 < min_weight <= max_weight <= 1`,
`min_weight >= rebalance_band`, `0 <= cash_buffer < 1`, `forced_exit_fails >= 1`).

Strategy `parameters` for runs: `benchmark_symbol "SPY"`, `budget 100000`, `warmup_trading_days 10`, and
`backtesting_start`/`backtesting_end` taken from the window factory, as `vwap_pullback` does:
`backtest_window(PredefinedWindow.BI_MONTH)[0]` and `[1]` (`backtesting/time_window.py`; today 2026-07-28 to
2026-09-23, about 41 sessions, so the default moves whenever the factory's dates are updated). The factory's bounds are
already timezone-aware in `MARKET_TZ`. `run_backtesting` passes:

- `data_source=YahooBacktestData`, `timestep "day"`, `sleeptime "1D"`;
- `preload_assets` = the universe, SHV and SPY, so the screen's daily price lookups hit the cache (as `cross_momentum`
  does);
- `agent_telemetry=True`.

Environment: `env/.env.bill_ackman.<mode>` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` optional,
`SEC_EDGAR_USER_AGENT`, plus the broker keys for paper/live). `README.md` line 75 is updated: the user agent is also
needed by this strategy. The first backtest downloads about 5 GB of SEC data once (10 to 20 minutes); later runs reuse
`cache/sec/annual/`.

`main.py` gains `"bill_ackman": _build_bill_ackman`, which loads the universe file (the same message as the other
universe strategies when it is missing) and returns the strategy.

## 11. Agents, tools and prompts

Three agents, created in `initialize` through `self.agents.create(name=..., system_prompt=..., tools=[...])`:

| Agent | Tools |
|---|---|
| `researcher` | `fundamentals_tools(self)`, `market_data_tools(self)`, `submit_ranking` |
| `short_seller` | `fundamentals_tools(self)`, `news_tools(self)`, `market_data_tools(self)`, `submit_verdicts` |
| `trader` | `submit_portfolio` |

No agent gets `trading_tools`, `account_tools`, indicator or memory tools. Every research tool already gates on
`strategy.clock.now()`; with §7 they also refresh in live.

`prompts.py` holds the three system prompts, in English, written to start from the page's sentence and add what a local
model needs: the role and what counts as success; use the fact sheet first and the drill-down tools to check a claim;
write every free-text argument in English; end the run by calling the submit tool exactly once and then reply with one
line. In outline:

- **Researcher**: "Find the simple, predictable companies that make lots of cash and trade at a good price." Rank at most
  5 of the candidates given. Prefer a stable holding to a marginally better new idea unless something changed.
- **Short seller**: "You are a short seller. Attack each idea: too much debt, weak management, strong rivals, or a price
  that is too high." `survive` means the attack failed; judge every name you are given and no other.
- **Trader**: "Hold the few ideas that survived, with more money in the best ones." Choose only from the allowed list,
  respect the weight bounds, weigh a pending fail against its counter, and never invent a symbol.

The exact wording is part of the plan, and is reviewed there.

## 12. Fact sheet (`fact_sheet.py`, pure)

One dict per company, rounded and token-lean (about 8 lines when printed): `symbol`, `sic`, `market_cap`, `fcf_yield`,
`fcf_margin_5y`, `operating_margin`, `operating_margin_stdev`, `revenue_cagr_5y`, `net_debt_to_operating_income`
(with `debt_reported` when false), `fiscal_year_end`, `filed`, `price`, `price_return_12m`. The first nine come from the
screen's `Candidate`; the last two come from the strategy (`get_last_price` and `get_historical_prices`, both clock-gated
in a backtest). A price fact that cannot be computed is `null`. A holding rejected for a data reason carries only the
price facts and `screen_unavailable`.

## 13. Testing

All off the network, with hand-written fakes.

- **Pure modules** (`fact_sheet`, `handoff` validation, `hysteresis`, `portfolio`, `AckmanParams`): unit tests, one per
  rule above, including the boundary values (a weight of exactly 0.05 and 0.35, a total of exactly 0.98, a counter at
  exactly 2).
- **Submit tools**: each valid and invalid case of §3, the first-valid-is-final rule, and symbol normalisation. A test
  also asserts the tool functions' real annotations are readable (the `from __future__` rule).
- **Rebalancer** against `BacktestBroker`: sells before buys; the SHV remainder; the band (a 4-point drift does not trade,
  a 6-point drift does); a forced exit; an order rejected for cash; a failed price lookup; a minimum-size skip.
- **Pipeline** with fake agent handles that call the real submit tools: the happy path; an invalid submission corrected
  in the same run; a missing submission fixed by the forced retry; a missing submission after the retry abandons the
  day with the book untouched; a screen outage abandons the day; 3 abandoned days abort a backtest; a holding failing on
  two consecutive days is sold even if the trader keeps it in its portfolio; a holding that recovers resets its counter.
- **Screen verdicts**: a holding rejected for a quality reason is `fail` by code and never reaches the short seller; a
  holding rejected for a data reason does.
- **State**: round trip, atomic write, corrupt file, wipe at the start of a backtest only.
- **End-to-end**: a backtest over a few sessions with fake agents and a tiny universe, asserting the equity curve exists,
  `reviews.jsonl` has one line per session, and no order came from an agent.
- **The move**: the moved screen tests run unchanged apart from imports. The client freshness additions get tests for
  stale refetch, fresh hit, failed refresh keeping the stale file, and unchanged behaviour without the arguments.

## 14. Smoke script and docs

- `scripts/tests/run_smoke_tests.sh` already runs every `smoke_*.py`, so `smoke_quality_screen.py` is part of the
  aggregate run. Because the runner treats `SKIP:` output as a skip, `smoke_quality_screen.py` is changed to print
  `SKIP: SEC_EDGAR_USER_AGENT is not set` and exit 0 when the variable is missing (it exits 1 today, which would fail the
  aggregate run on a machine without it).
- `scripts/tests/HOWTO.md` gains a "Smoke quality screen" section (what it does, the env file and variable, that it is
  read-only, the first-run cost, the second-run speed).
- `CLAUDE.md`: the architecture bullet for `strategies/` gains `bill_ackman/`; the `fundamentals/` bullet is split; the
  gotchas gain the hand-off rule (agents never trade; only submit tools are trusted), the abandonment rule, and the
  hysteresis rule.

## 15. Delivery order

1. Move the screen out of `fundamentals/` (§1.1) and update the screen smoke script's imports and SKIP behaviour (§14).
2. Client freshness for the drill-down tools (§7).
3. Move the two order helpers to `utils/helpers.py` (§8).
4. Pure modules: `parameters`, `fact_sheet`, `hysteresis`, `portfolio`.
5. `handoff`: the recorder, validation and submit tools.
6. `rebalancer`.
7. `state`.
8. `pipeline`, the strategy class, `main.py` registration, prompts, env and README.
9. The end-to-end backtest test, `HOWTO.md`, `CLAUDE.md`, and one real paper/backtest run to look at (not automated).

## 16. Known limits

- **The universe file is today's snapshot**, so a backtest screens survivors; the first spec's limits (SIC codes not
  point-in-time, debt without commercial paper, IFRS filers excluded) carry over.
- **Recent IPOs never qualify**: the screen needs five years of 10-K data.
- **LLM cost**: a day is three agent runs plus up to three retries. A 2-month backtest on a local 27B model is a matter
  of hours, and a one-year one a day or more.
- **The screen's drift in live**: it reads cached figures for up to 30 days and split history for one day (spec 1), so a
  filing made today can enter the ranking up to a month late.
- **A dedicated account is assumed**: positions the strategy did not open would be reviewed and may be sold.
- **Hysteresis is per holding, not per idea**: a new candidate that fails once is simply not bought, and may be proposed
  again the next day.
- **No dashboard view** of the reviews yet, and no orchestrator to decide about the strategy from outside.

# vwap_pullback_continuation: a pure-code strategy (agents removed)

Date: 2026-10-01 · Branch: `feature/vwap-pullback-pure-code` · Supersedes the agent parts of
`2026-09-29-vwap-pullback-continuation-design.md` (§4 and the agent rows of §8)

## Problem

`vwap_pullback_continuation` runs two LLM agents over a Python scanner. The 2026-09-30..10-01 backtest study
showed they add nothing:

- The entry agent entered 18 of 18 triggered setups and passed on none; a no-LLM run entering every trigger
  gave the same result (window A: +1.48% with agents, +1.56% without).
- The exit agent had no measurable net effect and cost 127 of the 144 LLM minutes of a one-month backtest
  (about 2 h per run, against about 20 min without agents).

The strategy is meant to be run under a future orchestrator agent, which decides about the strategy from
outside. Inside the strategy, an LLM has no role.

## Goal

`strategies/vwap_pullback/` becomes a deterministic, code-only strategy: no LLM call, no LangChain or
LangGraph import, no news fetch. Same strategy name, same class, same registration in `main.py`, runnable in
`live`, `paper` and `backtesting`.

- **Entries**: every triggered setup is tried, through the desk's existing guards.
- **Exits**: the protective stop placed on every fill, and the 15:50 flatten. Nothing else.
- **Unchanged behaviour**: candidate selection (stage 1, stage 2), the setup state machine, sizing and risk
  rules, order handling and its stop invariants.

Success: the test suite and `ruff check` pass, and a backtest on window A (2026-08-25..09-23, $10k) completes
without an LLM and lands close to the earlier no-LLM baseline (+1.56%, 26 trades); differences must be
explained by the entry ordering change of §2.

## Non-goals

- **No orchestrator and no seam for it.** It gets its own spec once its decisions are known.
- **No code exit ladder** (partial profit, trailing stop, break-even, exit below VWAP). The one-month sweeps
  found no exit variant that beat stop + flatten on both windows. The removed code stays in git history
  (`fd32813` for the rule ladder, `main` before this branch for the desk actions).
- **No parameter changes.** Every threshold in `parameters.py` that survives keeps its value.
- **No change to the framework**: `agents/`, `BacktestBroker`'s trailing-stop simulation
  (`fills.evaluate_trailing_stop`) and `agents/tools/news.py` stay as they are, used or not by this strategy.

## Design

### 1. Tick flow

`VwapPullbackStrategy.on_trading_iteration`:

1. Live only: wait `live_bar_delay_seconds` (unchanged).
2. `_ensure_session()`; return when there is no session or it is already flattened (unchanged).
3. First tick of a session: `desk.close_unknown_positions()` (unchanged).
4. `desk.reconcile(now)`: expired or rejected entries and missing stops are settled first.
5. `scanner.scan(session)`: contexts, stage-2 ranking, setups advanced.
6. `desk.enter_triggered()`: the new entry step (§2).

Steps 4-6 replace the LangGraph pass (`classify` → `exit_agent` → `entry_agent`). The other hooks keep their
bodies: `before_market_opens`, `before_market_closes`, `on_filled_order`, `on_canceled_order`.

`initialize` builds the desk and the scanner and, in backtests, still sets `self.clock.max_wait_slice = 60.0`
(a stop must be placed bar by bar after a fill). It creates no agent and no graph.

`run_backtesting` passes `agent_telemetry=False`. The constructor loses `chat_model`.

### 2. Entries: `Desk.enter_triggered()`

One call per tick, after the scan:

1. Return at once when no entry can be accepted: session flattened, outside the entry window, the daily loss
   breaker tripped, or no free slot (the checks `entry_due` makes today).
2. Collect the setups in state `TRIGGERED`.
3. **Order them by stage-2 composite score, best first, symbol as the tie-breaker.** The score acts as the
   filter when triggers outnumber free slots: the best-ranked ones take the slots. It is not a threshold: with
   enough slots every trigger is tried. A triggered symbol that did not pass the stage-2 floor this tick (it is
   tracked only because it is sticky) has no score and goes after every scored one.
4. For each, in that order, call `enter_long(symbol)` until no slot is left.

`enter_long(symbol)` keeps every guard it has today (trigger state, flattened, entry window, breaker, free
slot, exposure, candidate, price and account available, `risk.plan_entry` with its chase guard and R band) and
its marketable limit buy. It loses its `catalyst` and `reason` arguments and the catalyst check. A refusal is
logged as a warning, as today, and the trigger expires by itself on the next bar (it may trigger again).

Score plumbing: `rank_stage2` already computes the composite and the scan discards it. The scan now stores it
in `SessionState.scores: dict[str, float]` (rebuilt every tick, only for symbols passing the floor).
`RankedCandidate.composite` becomes `float | None`, `None` for a sticky symbol outside the floor, in place of
today's `0.0` (which would rank it mid-pack).

This ordering replaces the alphabetical order of the earlier runs, so a tick with more triggers than slots may
pick different trades than the earlier baseline.

Removed with the entry agent: `pass_on_setup`, `awaits_decision`, `entry_due`, `planned_risk`,
`max_passes_per_symbol`.

### 3. Exits

A trade ends in one of two ways: its protective stop fills, or the 15:50 flatten sells it. The stop stays where
it was placed (pullback low minus `stop_buffer_atr` ATRs).

Removed from `Desk`: `take_partial_profit`, `tighten_stop`, `replace_stop_with_trailing`, `exit_position`,
`hold`, `_restored`, `_split`, `exit_review_due`, `mark_reviewed`, `_flags`, `minutes_to_flatten`, `levels`. `_submit_stop`
only ever builds a plain stop order, and `last_close` reads the close of the symbol's last scanned bar directly.

Kept without change in behaviour: `on_order_filled` (stop on every fill, orphan fills sold), `on_order_canceled`,
`_settle_entry`, `_settle_exit`, `_reprotect`, `_protect`, `reconcile`, `_release_stop` (the flatten and
re-protection use it), `_market_sell`, `_cancel`, `flatten_all`, `_sell_unprotected`, `close_unknown_positions`,
`_archive`, `free_slots`, `session_pnl`, `breaker_tripped`, `_has_exposure`, `_pending_sell_proceeds`. The desk's
accounting model (whole unbooked orders; `trade.quantity` drops only when a fill is booked) is not touched.

### 4. What is deleted

| Where | Removed |
|---|---|
| Files | `prompts.py`, `tools.py`, `graph.py`, `news.py` |
| Tests | `test_vwap_graph.py`, `test_vwap_news_prompts.py`, `test_vwap_tools.py`; the exit-action tests of `test_vwap_desk_exits.py` |
| `scanner.py` | `refresh_headlines` and its call |
| `session.py` | `headlines`, `headlines_fetched_at`, `new_headline`, `decided`, `passes` |
| `trades.py` | `Trade.catalyst`, `reason`, `tp1_done`, `stop_kind`, `trail_price`, `last_review_at`, `review_flags`; `trade_flags`, `exit_review_due`, `Trade.unrealised_r` (`unrealised_pnl` stays: the loss breaker reads it) |
| `setups.py` | `health` (it only built the agent's row) |
| `parameters.py` | `exit_review_minutes`, `headlines_per_symbol`, `news_calls_per_run`, `bars_calls_per_run`, `max_passes_per_symbol`, `tp1_fraction_band`, `trail_atr_band`, `ema_length` |
| `features.py` | `Levels`, `latest_levels`, `ema_last`, `bar_atr` (only the exit rows and the trailing stop read them) and their test |
| `agent_vwap_pullback.py` | agent names and tasks, `_classify_node`, `_exit_node`, `_entry_node`, `_run_agent`, the agent-error counter, `MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS`, the `chat_model` argument |
| `pyproject.toml` | the `langgraph` dependency (`graph.py` was its only user); `uv.lock` regenerated by `uv sync` |

`SetupState.TRIGGERED`'s comment and the other docstrings that mention an agent are reworded.

### 5. `trades.jsonl`

Same file, same place (the run directory), one line per closed trade. Fields after the change:

`symbol`, `entered_at`, `closed_at`, `entry_price`, `filled_quantity`, `stop_price`, `r_per_share`,
`realised_pnl`, `realised_r`, `exit_reason`.

Dropped: `catalyst`, `reason`, `tp1_done`, `stop_kind`. Nothing in `src/` reads the file. Older run directories
keep their old lines.

### 6. Error handling

- A failed session preparation still skips the tick and is retried on the next one.
- An exception in `on_trading_iteration` is still logged by the executor and the next tick runs.
- With no agent there is no `AgentError` and no `FatalStrategyError`: a backtest can no longer abort on LLM
  failures, and none of that code remains.
- Entry refusals and order failures keep their current handling inside `Desk`.

### 7. Tests (TDD)

New, written first:

- `enter_triggered`: enters every trigger when slots allow; with more triggers than slots, takes the highest
  scores, symbol breaking ties; an unscored trigger goes last; nothing is submitted when flattened, outside the
  window, with the breaker tripped or with no free slot; one refusal does not stop the following entries.
- `rank_stage2`: a sticky symbol outside the floor gets `composite=None`.
- Scanner: `SessionState.scores` holds this tick's composites and is rebuilt every tick.
- Strategy tick: reconcile, then scan, then entries, in that order; no agent is created; importing and running
  the strategy does not import `langchain` or `langgraph`.
- `Trade.to_json`: exactly the fields of §5.

Kept as the safety net for order handling: `test_vwap_desk_entries.py` (adapted to the new `enter_long`
signature), the reconcile, re-protection, flatten and restart tests, and the features, screening, setups and
risk tests.

### 8. Documentation

- `CLAUDE.md`: the `strategies/` entry and the "vwap_pullback never leaves a position without a stop" gotcha
  describe a code-only strategy (no agents, no graph, no `tools.py`).
- `strategies/vwap_pullback/__init__.py` and `agent_vwap_pullback.py` docstrings.
- The file name `agent_vwap_pullback.py` stays: `cross_momentum`, also without an LLM, uses the same
  `agent_<name>.py` convention.
- `TODO.md`: "Remove entry agent" and "Change agentic architecture" struck through.

### 9. Check on real data

After the implementation, one backtest on window A with the backtesting env file. Reported as measured: return,
trade count, run time, and any difference from the earlier baseline with its cause.

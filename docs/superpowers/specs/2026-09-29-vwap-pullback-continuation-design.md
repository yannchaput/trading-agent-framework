# vwap_pullback_continuation: intraday VWAP pullback continuation, two LangGraph-orchestrated agents

Date: 2026-09-29 · Branch: `feature/vwap-pullback-continuation` · Source brief:
`prompts/Intraday_VWAP_Pullback_Continuation.md`

## Problem

The ORB approach buys a range break and suffers many false breakouts. The brief proposes the opposite
discipline: find stocks already showing abnormal, idiosyncratic strength, wait for the first orderly pullback
toward VWAP, and buy only when buyers visibly return. What matters for CAGR is keeping the rare large intraday
winners, not the win rate.

No strategy in the repo does intraday trading, trailing stops, or runs two cooperating agents; and
`BacktestBroker` cannot simulate a trailing stop (`backtesting/fills.py` raises on `OrderType.TRAIL`).

## Goal

A new strategy package `strategies/vwap_pullback/`, registered as `"vwap_pullback_continuation"` in
`main.py`, runnable in `live`, `paper` and `backtesting`:

- **Strictly intraday**: flat by the close every session.
- **Two LLM agents** (the deployed model is Qwen3.6-27B, local): an **entry agent** judges triggered pullback
  setups (health, catalyst from news) and enters; an **exit agent** manages open trades (partial profit,
  tightening, trailing stop, discretionary exit).
- **A per-tick LangGraph `StateGraph`** orchestrates a deterministic classifier and the two agents; the agents
  run only when state warrants it (a trigger fired, or an open trade had an event).
- **Code, not the model, enforces the safety net**: stop placement and sizing, a protective stop on every
  fill, position/cash caps, a daily circuit breaker, entry windows, the end-of-day flatten.

Judged by backtests (the user's own review of `metrics.json` and the trade log), in particular whether
catalyst-labelled entries outperform `none`.

Out of scope: a deterministic "rules" baseline mode (considered, declined -- possible future work);
multi-day holding; shorts; a dashboard view of the trade log.

## 1. Package layout

`src/trading_agent_framework/strategies/vwap_pullback/`:

| Module | Pure | Responsibility |
|---|---|---|
| `parameters.py` | yes | frozen `VwapPullbackParameters` dataclass: every threshold, window and risk knob (§8) |
| `features.py` | yes | session VWAP, RVOL curve, intraday return, beta-adjusted RS, VWAP distance and move in ATR, cross-sectional z-scores, 5-min resampling, 9-EMA, 5-min ATR |
| `screening.py` | yes | stage 1 (daily) and stage 2 (intraday) candidate selection (§2) |
| `setups.py` | yes | per-candidate state machine (§3) |
| `risk.py` | yes | stop, R, sizing, caps, circuit breaker, time windows (§5) |
| `trades.py` | yes | `Trade` record and `TradeBook` (§5) |
| `news.py` | yes | trim/dedupe headlines into lean rows |
| `prompts.py` | yes | the two system prompts, built from the parameters (§4) |
| `tools.py` | no | the agents' tools (§4) |
| `graph.py` | no | builds the `StateGraph`; its routing functions are pure (§4) |
| `agent_vwap_pullback.py` | no | `VwapPullbackStrategy(Strategy)`: agents, graph, hooks, `run_backtesting` |

Deliberately not used: `PrebuiltTools.all()` (a raw `submit_order`/`cancel_order` would let the model bypass
sizing and stops), cross-session memory (intraday: every session starts fresh), a news grounding gate (§4).

All per-session state (setups, `TradeBook`, RVOL baselines, stage-1 list) lives in `strategy.vars` and is reset
in `before_market_opens`.

## 2. Candidate selection (two stages, Python only)

**Stage 1 -- `before_market_opens`, once per session (`screening.py`).**

- One batched daily-bar fetch (`get_historical_prices_for_assets`) for the universe
  (`load_cross_momentum_universe()`, 1,200 symbols) plus SPY, `stage1_lookback_sessions` (70) sessions.
- Filters: last close ≥ `min_price` ($5); ATR%(14) in `atr_pct_band` ([1.5%, 8%]); 20-day average dollar
  volume in the universe's top `dollar_volume_percentile` (60%) -- a percentile, because Alpaca's IEX volume
  is a small slice of the tape and absolute thresholds would be wrong.
- Rank by mean of z(ATR%) and z(20-day return); keep the top `stage1_size` (150).
- For each survivor also compute: 60-day beta to SPY (daily returns), daily ATR(14).
- Fetch the survivors' minute bars for the previous `rvol_baseline_sessions` (10) sessions once and build
  each one's RVOL baseline: average cumulative volume at each minute-of-session. Sessions are aligned by
  minutes since their open, so early-close sessions contribute only the minutes they have.

**Stage 2 -- every tick from `no_entry_before` (09:45) (`features.py` + `screening.py`).**

For each stage-1 symbol, from today's minute bars (one batched fetch per tick):

| Feature | Definition |
|---|---|
| `ret` | last close ÷ session open − 1 |
| `rs` | `ret − beta × SPY_ret` |
| `rvol` | cumulative volume ÷ baseline cumulative volume at the same minute-of-session |
| `vwap_dist_atr` | (last close − session VWAP) ÷ daily ATR |
| `move_atr` | (session high − session open) ÷ daily ATR |

- Hard floor: `rvol ≥ rvol_min` (1.5), `rs > 0`, last close > VWAP.
- Composite = mean of cross-sectional z(`ret`), z(`rs`), z(`rvol`) over the symbols passing the floor; a
  zero-variance feature contributes 0.
- The top `tracked_size` (30) are tracked. A symbol that has reached `IMPULSE` stays tracked for the rest of
  the session regardless of rank.

## 3. Setup state machine (`setups.py`)

Runs on 5-minute bars resampled from minute bars. Each tick replays **every 5-minute bar closed since the
previous tick**, in order, so no bar is skipped between ticks; replaying N bars in one tick must give the same
result as one bar per tick.

```
WATCH ──impulse──▶ IMPULSE ──retrace──▶ PULLBACK ──resume──▶ TRIGGERED ──entered──▶ IN_TRADE ──▶ DONE
                      │                    │  ▲                  │ (not taken by the next tick → PULLBACK)
                      └────── broken ──────┴──┴──────────────────┴──▶ BROKEN (terminal for the session)
```

- **WATCH → IMPULSE**: `move_atr ≥ impulse_move_atr` (0.8), `rvol ≥ rvol_min`, `rs > 0`, close > VWAP.
  Records `impulse_high` (running max of highs), impulse average bar volume, impulse duration (bars from the
  session open to `impulse_high`).
- **IMPULSE → PULLBACK**: retracement of the impulse leg (session open → `impulse_high`) ≥ `pullback_min_retrace`
  (25%) while close > VWAP. Records `pullback_low` (running min of lows), pullback average bar volume, pullback
  duration.
- **PULLBACK → TRIGGERED** (resumption): the bar's close > the previous bar's high, close > VWAP, and bar volume
  > pullback average volume. `trigger_close` is recorded.
- **TRIGGERED** lasts until the next tick; if no entry was accepted by then it returns to `PULLBACK` (a later
  resumption can re-trigger).
- **→ BROKEN** (any state before `IN_TRADE`), on any of: close < VWAP; retracement > `pullback_max_retrace`
  (61.8%); `rs ≤ 0`; a red bar with body > `bearish_body_atr` (0.5) × daily ATR ÷ 6; a red bar with volume >
  `selling_volume_ratio` (1.5) × impulse average; pullback duration > impulse duration.
- **TRIGGERED → IN_TRADE** as soon as `enter_long` is accepted (the entry order may still be pending). If the
  entry is cancelled with nothing filled (§5), the setup returns to `PULLBACK`; any filled quantity keeps it
  `IN_TRADE`.
- **IN_TRADE → DONE** when the trade closes. One trade per symbol per session.
- Health flags exposed to the agent: `above_vwap`, `vol_ratio` (pullback ÷ impulse average volume),
  `duration_ratio`, `retracement_pct`, `rs_now`, `rvol_now`, composite z, `largest_red_body_atr`.

## 4. Graph, agents and tools

### Graph (`graph.py`)

`build_tick_graph(classify, run_exit, run_entry)` compiles once in `initialize()`; the three callables are
injected so tests use fakes. No checkpointer (cross-tick state is in `strategy.vars`), no `RetryPolicy` (re-running
an LLM node could place an order twice).

State (`TypedDict`): `now`, `setups` (lean rows), `open_trades` (lean rows), `entry_window_open`,
`free_slots`, `exit_review_due` (symbols), `runs: Annotated[list, operator.add]`.

```
START → classify ─┬─(exit due)──▶ exit_agent ─┬─(entry due)──▶ entry_agent ─▶ END
                  ├─(entry due)───────────────┼───────────────▶ entry_agent
                  └─(neither)──▶ END          └─(else)────────▶ END
```

- **classify** (data): advance setups (§3), refresh open trades from positions/orders, fetch headlines
  (`NewsProvider`, gated on `strategy.clock.now()`, headlines only, ≤ `headlines_per_symbol` (3)) for
  `PULLBACK`/`TRIGGERED` setups since the previous close and for open trades since entry, compute windows and
  `free_slots` (pending entries count as taken, pending full exits as freed).
- **exit due**: ≥ 1 open trade with an event since its last review: +1R reached, a 5-min close below VWAP or
  the 9-EMA, a new headline, or `exit_review_minutes` (15) elapsed.
- **entry due**: ≥ 1 `TRIGGERED` setup, entry window open, `free_slots > 0`, circuit breaker not tripped.
- Exit runs before entry so slots it frees are visible the same tick (routing re-reads `free_slots` after the
  exit node).
- Agent nodes call `strategy.agents[name].run(task, context=..., run_id=uuid)` -- repair, forced-tool and
  telemetry middleware apply unchanged. `AgentError` handling mirrors news_binary: log, count consecutive
  failures, raise `FatalStrategyError` after `MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS` (3) in a backtest.

### Entry agent `vwap_entry`

| Tool | Behaviour / code-enforced rules |
|---|---|
| `get_setups()` | `PULLBACK`/`TRIGGERED` rows: health flags, planned stop, R, headlines |
| `get_intraday_bars(symbol, length≤24)` | 5-min OHLCV only |
| `search_news(...)` | existing `news_tools(strategy)`, wrapped in a per-run budget (§4, news) |
| `enter_long(symbol, catalyst, reason)` | refused (`{"error"}`) unless the setup is `TRIGGERED`, the window is open, a slot is free, the circuit breaker is not tripped, no position/order exists in the symbol, the chase guard and R band pass (§5). Code computes size and stop; the agent never passes a quantity or price. `catalyst` ∈ {`earnings`, `guidance`, `analyst`, `contract_or_product`, `sector_or_macro`, `none`} |
| `pass_on_setup(symbol, reason)` | records the decision |

A `TRIGGERED` setup that got neither call is logged and treated as a pass. No retry turn.

### Exit agent `vwap_exit`

| Tool | Behaviour / code-enforced rules |
|---|---|
| `get_open_trades()` | symbol, qty, entry, stop kind/level, R, unrealised R, `tp1_done`, VWAP, 9-EMA (5-min), minutes to flatten, headlines since entry |
| `get_intraday_bars`, `search_news` | as above |
| `take_partial_profit(symbol, fraction)` | fraction in [0.25, 0.5], once per trade |
| `tighten_stop(symbol, stop_price)` | only raises the stop; refused on a trailing stop |
| `replace_stop_with_trailing(symbol, trail_atr)` | `trail_atr` in [0.5, 2.0] × 5-min ATR(14), sent as `trail_price`; refused if its current level (last − trail) is below the current stop |
| `exit_position(symbol, reason)` | market sell of the remaining shares |
| `hold(symbol, reason)` | records the decision; updates `last_review_at` |

Neither agent gets raw order tools. The exit agent only acts on exit orders, the entry agent only on entries.
All tools return lean dicts; failures come back as `{"error": ...}`. Docstrings are one line (token budget).

### News

Code attaches headlines to every setup/trade row (a floor the model cannot skip). Both agents also have
`search_news` for more context, **optional** (no grounding gate -- news_binary's gate caused retry loops) and
**budgeted**: after `news_calls_per_run` (4) calls in one run it returns
`{"error": "news budget for this run is spent; decide with what you have"}`.

### Prompts (`prompts.py`, English only)

- **Entry**: enter only on resumption, never on a VWAP touch; weigh the health flags; read headlines and label
  the catalyst; never enter an M&A target or an offering/dilution headline; with catalyst `none` enter only if
  both RS and RVOL z-scores > `none_catalyst_min_z` (2); call `enter_long` or `pass_on_setup` for every
  `TRIGGERED` setup.
- **Exit**: take TP1 around +1R (25–50%); let the rest run by tightening under VWAP/9-EMA or switching to a
  trailing stop; exit on bearish news, or on a loss of VWAP with heavy volume; don't react to noise inside 1R;
  the code flattens at 15:50 regardless; call exactly one of the action tools or `hold` per open trade.

## 5. Guardrails and order flow

### Sizing (`risk.py`)

- Stop = `pullback_low − stop_buffer_atr` (0.1) × daily ATR. R = `trigger_close − stop`.
- Refused when R is outside `r_band_atr` ([0.15, 1.0]) × daily ATR.
- Quantity = floor(equity × `risk_per_trade` (0.5%) ÷ R), capped by `max_position_pct` (25%) of equity and by
  95% of min(`buying_power`, `cash` + proceeds of pending sells); refused when it rounds to 0.
- At most `max_positions` (4) positions + pending entries.
- **Circuit breaker**: no new entries once the session's realised + open P&L ≤ −`max_daily_loss_pct` (1.5%) of
  the session-open equity.
- **Chase guard**: refused if last price > `trigger_close + chase_guard_r` (0.3) × R.

### Entry

Marketable **limit** buy at last price + `entry_limit_atr` (0.05) × daily ATR, time in force day. If still
open at the next tick, code cancels it and the setup returns to `PULLBACK` (a partial fill keeps its filled
part, see below).

### Protective stop (code only)

- `on_filled_order` for an entry → submit a **STOP sell** for the filled quantity at the stop price; record its
  id in the `Trade`. A partially filled entry that is cancelled gets a stop for the filled quantity.
- **Fail-safe**: if the stop submission fails, market-sell the position immediately and log an error.

### `Trade` / `TradeBook` (`trades.py`)

`Trade`: `symbol`, `entry_order_id`, `qty`, `entry_price`, `stop_order_id`, `stop_kind` (`stop`|`trail`),
`stop_level`, `r_per_share`, `tp1_done`, `catalyst`, `entered_at`, `last_review_at`, `exit_reason`,
`realised_pnl`. `TradeBook` keyed by symbol. Each closed trade is appended as one JSON line to
`trades.jsonl` in the run's log directory.

### Stop hand-off (inside the exit tools)

Alpaca and `BacktestBroker` both refuse a sell above *held − pending sells*, so the stop must go first:

| Tool | Sequence |
|---|---|
| `tighten_stop` | `modify_order(stop, stop_price=new)`; store the returned order's id (Alpaca/backtest return a new order, IBKR the same one) |
| `take_partial_profit` | cancel stop → `wait_for_order_execution(stop, timeout=10s)` → market-sell the fraction → resubmit the stop (same level, or same trail) for the remainder |
| `replace_stop_with_trailing` | cancel stop → wait → submit `TRAIL` sell for the remainder |
| `exit_position` | cancel stop → wait → market-sell the remainder |

If the stop filled before the cancel took effect, the tool returns `{"status": "already_stopped_out"}` and the
trade is closed. The few seconds without a stop during a live hand-off are accepted (a native quantity replace
is Alpaca-only and would break the broker-agnostic seam).

### Session boundaries

- Entries only in [`no_entry_before` (09:45), `no_entry_after` (15:00)] market time.
- `minutes_before_closing = 10`: at 15:50 `before_market_closes` cancels every open order, waits, and
  market-sells every position; the next tick re-checks and retries anything left.
- **Restart**: on the first tick, any position held but absent from the `TradeBook` is closed (not adopted),
  after cancelling its open orders.
- `sleeptime = "5M"` in every mode.

## 6. Framework changes outside the package

1. **`backtesting/fills.py`**: pure `evaluate_trailing_stop(side, bar, *, trail_price, trail_percent,
   reference) -> tuple[FillResult | None, Decimal]`. Sell: level = reference − trail (or reference × (1 −
   pct)); if `bar.low ≤ level` → fill at `min(bar.open, level)` (gap through fills at the open); only
   otherwise does the reference become `max(reference, bar.high)` (pessimistic same-bar ordering). Buy
   mirrors it with a low-water mark. `evaluate_fill` no longer raises for `TRAIL`.
2. **`backtesting/broker.py`**: `_PendingOrder` carries the trailing reference, seeded with the latest close
   (via `_source_bars`) at submission and ratcheted bar by bar in `_process_pending`/`_bars_after`, so it is
   right across multi-bar clock jumps. `_projection` counts a pending `TRAIL` sell as a pending sell.
3. **`core/strategy.py`**: `create_order(..., trail_price=None, trail_percent=None)` selects `OrderType.TRAIL`
   (Alpaca and IBKR already translate it).
4. **`main.py`**: `"vwap_pullback_continuation": _build_vwap_pullback` (returns `None` with a message when the
   universe file is missing, like cross_momentum). Env files `env/.env.vwap_pullback_continuation.<mode>` per
   the README convention.
5. **`CLAUDE.md`**: document the package and the backtest `TRAIL` rule.

## 7. Backtesting configuration and limits

`run_backtesting` override: `data_source=AlpacaBacktestData`, `timestep="minute"`, `warmup_trading_days=75`,
default window one month, benchmark SPY, only SPY preloaded; stage 1 `load()`s the universe's daily series,
candidates' minute bars are fetched lazily. Telemetry on.

Estimated cost: ~30 agent runs per simulated session at ~60 s on the 27B model ≈ 30 min per session, ≈ 10 h per
backtest month.

Known limitations (documented, not fixed):
- Survivorship bias: the universe file is dated 2026-09-25.
- RVOL is built on IEX-only volume.
- Lazily fetched minute bars are not cached across runs (`CachedDataSource` caches `load()` only), so a first
  run fetches every stage-1 symbol's minute history.
- A partial entry fill is unprotected until the entry completes or is cancelled.

## 8. Parameters (`VwapPullbackParameters` defaults)

| Name | Default | | Name | Default |
|---|---|---|---|---|
| `stage1_lookback_sessions` | 70 | | `pullback_max_retrace` | 0.618 |
| `min_price` | 5 | | `bearish_body_atr` | 0.5 |
| `atr_pct_band` | (0.015, 0.08) | | `selling_volume_ratio` | 1.5 |
| `dollar_volume_percentile` | 0.60 | | `stop_buffer_atr` | 0.1 |
| `stage1_size` | 150 | | `r_band_atr` | (0.15, 1.0) |
| `rvol_baseline_sessions` | 10 | | `risk_per_trade` | 0.005 |
| `rvol_min` | 1.5 | | `max_position_pct` | 0.25 |
| `tracked_size` | 30 | | `max_positions` | 4 |
| `impulse_move_atr` | 0.8 | | `max_daily_loss_pct` | 0.015 |
| `pullback_min_retrace` | 0.25 | | `chase_guard_r` | 0.3 |
| `no_entry_before` / `no_entry_after` | 09:45 / 15:00 | | `entry_limit_atr` | 0.05 |
| `exit_review_minutes` | 15 | | `headlines_per_symbol` | 3 |
| `news_calls_per_run` | 4 | | `none_catalyst_min_z` | 2.0 |
| `tp1_fraction_band` | (0.25, 0.5) | | `trail_atr_band` | (0.5, 2.0) |

## Testing (TDD, no network, hand-written fakes)

| Area | Tests |
|---|---|
| `features.py` | VWAP; RVOL minute-of-session alignment incl. early closes; beta-adjusted RS; z-scores with zero variance; 5-min resampling, 9-EMA, ATR |
| `screening.py` | stage-1 filters, percentile volume, ranking; stage-2 floor and ranking; `IMPULSE` symbols stay tracked |
| `setups.py` | table-driven synthetic paths: healthy impulse → pullback → trigger; each broken reason alone; N-bars-in-one-tick = one-bar-per-tick; trigger expiry; one trade per session |
| `risk.py`, `trades.py` | R band, each cap, circuit breaker, chase guard, windows; trade record lifecycle and JSON line |
| `fills.py`, `BacktestBroker` | `TRAIL` touch, gap through, same-bar pessimism, reference ratchet over a multi-bar jump, pending-sell projection; `create_order` trail args |
| `graph.py` | pure routing functions; compiled graph with fake nodes: exit before entry, quiet tick → `END`, `AgentError` → `FatalStrategyError` after 3 in a backtest |
| `tools.py` | against `BacktestBroker` + fake data source: every `enter_long` refusal; each stop hand-off; `already_stopped_out`; `tighten_stop` never loosens; news budget |
| strategy | `on_filled_order` places the stop; stop failure → market sell; 15:50 flatten; unknown position closed on restart; fake agent handles (no LLM) |

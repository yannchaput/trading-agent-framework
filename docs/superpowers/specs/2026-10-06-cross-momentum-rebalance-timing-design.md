# cross_momentum: rebalance at a fixed time on completed sessions, with a date-aligned risk overlay

Date: 2026-10-06 · Source: live incident of 2026-10-06 (two rebalances, same day, opposite exposure)

## Problem

On Tuesday 2026-10-06 the live bot rebalanced at 09:42 ET with the risk overlay at NORMAL (beta 1.35, 20d vol
0.27, 20d corr 0.15, exposure 70% via breadth). A restart at 15:36 ET re-ran the rebalance and read CRITICAL
(beta 2.84, vol 0.39, corr 0.37, exposure 40%): 17 trims, 2 sells and 3 buys, all filled.

Read-only replays on real IEX bars (scripts kept out of the repo) showed:

- With correctly aligned series, every replay gives CRITICAL (beta 2.8-2.9), for the morning's and the
  afternoon's target lists alike, with today's partial bar as of 09:42, as of 15:48, or excluded altogether.
- `portfolio_risk_overlay._build_aligned_returns` aligns series by POSITION (`closes[-min_len:]`), not by date.
  Making about 10 of the 20 target series one day short reproduces the morning reading (beta 1.40, corr 0.17,
  NORMAL); 8 or more shifted series flip the state from CRITICAL to NORMAL. Why half the series were a day short
  at 09:30-09:42 was NOT established (rejected: a frozen clock, the partial bar's values, names without a first
  print).
- Live daily bars include the partial current session (`market_data.sessions_needed`: "+1 covers a partial
  current session"), so the filters depend on the time of day too: DOCN failed the dollar-volume filter at
  $19.3M (min $20M) with today's bar and passes at $21.3M through yesterday.
- A full-universe comparison after the close (2026-10-06, 1,200 symbols) of "with today's bar" against "through
  yesterday": 313 names pass either way with 8 swaps each way, top-20 overlap 16/20 (swaps at ranks 17-24), mean
  rank shift 1.4, breadth 49.8% against 45.4% (same step), overlay CRITICAL in both. Excluding the current day
  changes the book at the margins only; the morning swing came from the misalignment.

Backtests already behave as "decide on the previous completed close": at a 09:30 tick the data gate hides the
forming daily bar, and the orders fill at the next session's open. Live is the mode that departs from it.

## Goal

The Tuesday rebalance decision is the same whatever time the bot runs it on Tuesday: it reads completed
sessions only (Monday's close), its risk overlay aligns series by date, and it runs at a fixed time of day
(12:00 ET) instead of the open.

Success: `uv run pytest`, `uv run ruff check` and `pyright` pass; the 2016-2026 cross_momentum backtest is
essentially unchanged against run `2026-09-29_151101`; a read-only live replay of the scan gives the same
target, filter results and overlay state whether "now" is taken at 09:42 or 15:48 ET.

## Decisions taken in brainstorming (binding)

1. **The decision rests on Monday's completed close.** Every bar dated today (market time) is dropped before
   any computation. Same-day reactions (e.g. TWST's -18.7% on 2026-10-06) wait for the next rebalance.
2. **Restarts are not guarded.** A restart during a Tuesday session after the rebalance time rebalances again,
   as today. With completed bars a re-run reaches the same decision; only the sizing prices move.
3. **The rebalance runs at 12:00 ET**, set by `parameters.py` (`"rebalance_time": "12:00"`).
4. **The scheduler gets the start time (approach A).** A new optional `Strategy.iteration_start_time` moves
   the first tick of every session; the strategy does not sleep inside its iteration and does not tick on an
   interval with a time gate.

## 1. Scheduler start time

### 1.1 `Strategy` (`core/strategy.py`)

New class attribute next to `minutes_before_closing` and the other timing settings:

```python
iteration_start_time: time | None = None  # market time (America/New_York); None = at the open
```

### 1.2 Executor (`core/executor.py`, `_trade`)

- The first tick of a session is `max(session.open, start, now)`, with `start = datetime.combine(session date,
  iteration_start_time, MARKET_TZ)`; `start = session.open` when the attribute is `None` (exactly today's
  behaviour).
- When `start >= stop_at` (`session.close - minutes_before_closing`), the session runs no iteration and the
  executor logs one INFO line naming the start time and the close. (The NYSE's earliest close is 13:00, so
  12:00 never hits this; the rule is defined anyway.)
- An interval sleeptime starts its tick grid at `start` instead of the open.
- `before_market_opens`, `before_starting_trading` and `before_market_closes` keep their times; only
  `on_trading_iteration` moves. The session-based `_iteration_due` logic is unchanged.
- Backtests: the simulated clock waits to `start` the same way. A daily-bar order sent at 12:00 still skips
  the forming bar and fills at the next session's open, as from 09:30 today.

### 1.3 cross_momentum

- `parameters.py`: `"rebalance_time": "12:00"` (ET, `HH:MM`).
- `CrossMomentumStrategy.__init__` sets `self.iteration_start_time = time.fromisoformat(...)`; a malformed
  value raises at construction (`ConfigurationError`).
- The whole iteration (equity sample, rebalance-day check, rebalance) runs at 12:00 ET; a restart after 12:00
  runs at once, before 12:00 waits.

## 2. Completed sessions only, sizing at the live price

### 2.1 Dropping today's bar

- New pure helper in `strategies/cross_momentum/utils.py`: `completed_bars(df, today: date) -> DataFrame`,
  dropping every row whose index date in market time equals `today`. It handles both timestamp conventions:
  live Alpaca daily bars are stamped at midnight market time, backtest (Yahoo) daily bars at the close.
- `on_trading_iteration` only runs between the start time and the close, so a bar dated today is always
  partial there; the rule needs no session-close lookup.
- `today` comes from `self.get_datetime()` in market time.
- `_compute_indicators_for_ticker` fetches 301 daily bars, applies `completed_bars`, keeps the last 300. Every
  value derived from them (price filter, dollar volume, 20d volatility, trading days, the three returns, the
  score, ATR, breadth closes, the overlay closes) reads completed sessions only.
- The overlay's SPY series: same rule (301 bars, drop today, keep 300).
- Backtests: the data gate already hides today's bar, so `completed_bars` is a no-op there and the series are
  the same as before.

### 2.2 Sizing price

- `rebalance()` fetches one current price per target symbol up front with the existing `_price_or_zero`
  (`get_last_price`, the last trade) and uses it everywhere `entry["price"]` was used: target and held values,
  the trim test and quantity, the buy quantity and cost, the `Trimming` / `Buying` / `Holding` log lines.
- A symbol whose price is 0 (failed or empty lookup) gets no buy and no trim this week; one warning is logged
  and the rest of the rebalance proceeds.
- `entry["price"]` (the completed close) still feeds the filters and is kept on the entry.
- Backtests: `get_last_price` is the latest visible close, i.e. Monday's close, the same value as
  `entry["price"]` today.

## 3. Date-aligned risk overlay

### 3.1 Interface (`portfolio_risk_overlay.py`)

`compute_risk_overlay(closes_map: dict[str, pd.Series], target_weights, benchmark_closes: pd.Series | None,
min_obs=40)`: closes are Series indexed by session date (`datetime.date`). The return value is unchanged, plus
an `observations` entry in the metrics dict (the number of aligned returns). The strategy is the only caller.

### 3.2 Alignment (`_build_aligned_returns`)

- Inner-join the target series and the benchmark on their dates, drop rows with a gap, then compute simple
  daily returns from the joined price frame. A date missing from any one series is dropped for all, so every
  return spans the same interval for every symbol and for SPY.
- The "truncate to the shortest length" logic is removed. Weight renormalization is unchanged.
- Fewer than `min_obs` aligned returns: NORMAL (the existing fail-open behaviour, unchanged).
- Without a benchmark, the targets are joined among themselves (vol and corr only), as today.

### 3.3 Strategy side

- `_compute_indicators_for_ticker` also returns `close_series`: the completed closes indexed by
  `bars.index` dates in market time. `target_closes` stores these Series; the SPY series is built the same way.
- `"closes"` stays a list for the score and the breadth helpers, which do not change.
- Before calling the overlay, one WARNING if any target's last bar date differs from SPY's, naming the symbols
  and their dates (evidence for the unexplained shift; it no longer affects the result).
- The `Risk overlay:` INFO line gains `obs=<n>`.

## 4. Testing and verification

### 4.1 Tests (TDD: each written first and seen failing)

- Executor: first iteration at the start time when the bot starts before it; immediate iteration when `now`
  is past the start time (restart); `None` leaves every existing executor test unchanged; a start time at or
  after `stop_at` runs no iteration and logs; an interval sleeptime's grid starts at the start time; the hooks
  keep their times.
- cross_momentum: `completed_bars` under both timestamp conventions, and as a no-op without a bar for today;
  the scan reads completed bars; `rebalance()` sizes with the live price (the fake's last price differs from
  `entry["price"]`) and skips a symbol with no price; a malformed `rebalance_time` fails at construction; the
  existing rebalance tests stay green.
- Overlay (the module has no tests today): series offset by position but carrying correct dates give the
  aligned beta / corr (the 2026-10-06 case); a date missing from one series is dropped for all; fewer than 40
  aligned returns gives NORMAL; the classification thresholds are unchanged.

### 4.2 Verification

- `uv run pytest`, `uv run ruff check`, `pyright`.
- Re-run the 2016-2026 cross_momentum backtest and compare total return, CAGR, Sharpe, max drawdown and trade
  count with run `2026-09-29_151101`; a material difference is investigated as a bug before completion.
- Read-only live replay of the scan (scratch script): identical targets, filter results and overlay state with
  "now" at 09:42 and at 15:48 ET.

### 4.3 Docs

`CLAUDE.md`: `iteration_start_time` in the `core/` executor description; a cross_momentum gotcha covering
completed sessions only, the rebalance at `rebalance_time` (12:00 ET), sizing at the last trade, and the
date-aligned overlay.

## Rollout

- The live bot must be restarted after the merge. A restart on a Tuesday after 12:00 ET rebalances at once.
- The equity history for vol targeting is sampled at 12:00 instead of 09:30 from then on; the first sample
  after the switch spans 2.5 extra hours. The vol leg needs 64 daily samples and has 28, so it is inactive
  either way.

## Out of scope

- The momentum score, the filter thresholds, the overlay and breadth thresholds.
- `risk_diagnostics.py` (backtest-only observation; it already joins on its index).
- Any other strategy (they keep `iteration_start_time = None`).
- A once-per-week rebalance guard (decision 2).
- The root cause of the half-shifted morning series; the warning in §3.3 keeps the evidence.

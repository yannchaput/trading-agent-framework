# Market regime estimate (replaces ADX / RSI / VIX charting)

Date: 2026-10-02. Status: design, awaiting review.

## Goal

Every strategy gets, by default, a daily estimate of the market regime as an int
(`1` bullish, `0` neutral, `-1` bearish), logged and charted. It replaces the
ADX, RSI and VIX lines that `VwapPullbackStrategy` charts today. The dashboard
shows the regime under "Trade Activity", on the same time axis, so each trade
can be read against the regime at that moment.

Success criteria:

- ADX, RSI and VIX no longer appear in vwap_pullback's reporting or on newly
  produced Trades charts.
- A backtest of any strategy produces a `Regime` series in `indicators.parquet`
  from its first session on.
- The Trades tab shows a regime row sharing the portfolio chart's x-axis, with
  green / grey / red shading that also runs behind the portfolio chart.
- The regime is information only. No strategy trades on it, and cross_momentum's
  breadth overlay is untouched.

## Decisions taken in brainstorming

| Question | Decision |
|---|---|
| Regime rule | Trend + volatility on the benchmark's daily bars. No VIX or other feed. |
| Cadence | Once per session, whatever the `sleeptime` (intraday, `1D`, `5D`...). |
| Dashboard form | Step line row (y fixed to -1/0/1) plus faint colored bands behind the portfolio chart. |
| Warmup | `run_backtesting` widens the warmup automatically to what the regime needs. |
| Where it runs | The framework (executor + `Strategy`), not each strategy. |

## Design

### 1. `core/regime.py` (pure)

No I/O, no `Strategy` knowledge, plain float maths on daily closes (not a new
float boundary: it consumes `Bars.df`, the existing float64 seam).

- `RegimeParameters`: frozen dataclass, validates itself in `__post_init__`.
  Defaults: `sma_fast=50`, `sma_slow=200`, `vol_window=20`, `vol_lookback=252`,
  `vol_percentile=0.80`. Exposes `min_bars`, the closes needed:
  `max(sma_slow, vol_window + vol_lookback) + 1` (273 by default).
- `RegimeReading`: frozen dataclass with `regime: int` and the metrics behind it
  (`close`, `sma_fast`, `sma_slow`, `vol`, `vol_threshold`, `stressed`) for the
  log line.
- `classify_regime(closes, params) -> RegimeReading | None`. Returns `None` with
  fewer than `min_bars` closes.
  - **Trend:** `+1` if close > SMA(slow) and SMA(fast) > SMA(slow); `-1` if close
    < SMA(slow) and SMA(fast) < SMA(slow); else `0`.
  - **Volatility cap:** vol is the annualized (`sqrt(252)`) standard deviation of
    daily log returns over `vol_window`. If the latest vol is at or above its own
    `vol_percentile` over the trailing `vol_lookback` days, a `+1` is capped to
    `0`. `0` and `-1` are unchanged: vol can cool a bullish read, never create a
    bearish one.
  - No hysteresis in this version (YAGNI); add it only if the series flips too
    often in real backtests.

### 2. `Strategy` (`core/strategy.py`)

- Class attribute `regime_params: RegimeParameters = RegimeParameters()`,
  overridable per strategy like the other class attributes. The benchmark is the
  existing `benchmark_symbol`.
- `self.regime: int | None`: the latest value, `None` until it can be computed.
  Read-only information for strategy code.
- `_refresh_regime()`:
  1. Reads `regime_params.min_bars` daily bars of the benchmark through
     `get_historical_prices`. In a backtest this goes through
     `BacktestBroker._source_bars`, so only completed sessions are visible and
     there is no look-ahead.
  2. Runs `classify_regime`.
  3. On a reading: sets `self.regime`, writes one `log_info` line (the int and
     its metrics), and calls `add_line("Regime", value, plot_name="Regime")`.
     `add_line` is already a no-op outside backtesting, so paper and live get the
     log line only.
  4. On `None` (too few bars): leaves `self.regime` unchanged and logs a warning
     once per run.
  5. A `BrokerError` / `BacktestError` logs a warning and keeps the previous
     value. A regime failure never skips a session or ends a run.
- No wall-clock time: all timing comes from `strategy.clock` through the bars
  gate, per the repo's no-look-ahead rule.

### 3. Executor (`core/executor.py`)

`_run_session` calls a guarded `strategy._refresh_regime()` once per session,
right after the initial `wait_until(session.open - minutes_before_opening)` and
before the `before_market_opens` hook. It is unconditional, so a live run started
mid-session (where `before_market_opens` is skipped) still gets a regime. A
failure is logged and swallowed, like a hook failure. The executor is otherwise
unchanged.

### 4. Backtest warmup (`Strategy.run_backtesting`)

`warmup_trading_days` is raised to at least `regime_params.min_bars` before the
data source is built and before `run_backtest` loads the benchmark. Without it,
the default warmup of 0 leaves no regime for the first ~270 sessions. The
widening is one `max(...)`, in the same place that already computes
`warmup_start`. Documented in the method's docstring. It only widens the eager
data load's start bound, never the simulated `[start, end]` window, as with the
existing parameter.

Cost: every backtest loads about 273 more trading days of benchmark history.

### 5. vwap_pullback cleanup

- Delete `_chart_indicators`, `_vix_previous_close`, the `_vix` / `_vix_failed`
  attributes, the `VixSeries` import, and the call site in the tick.
- Delete `agents/tools/vix.py` and `tests/agents/tools/test_vix.py` (no other
  users).
- Replace the charting tests in `tests/strategies/vwap_pullback/test_vwap_strategy.py`
  (they assert ADX / RSI / VIX lines) with a test that the strategy adds no
  indicator lines of its own.
- `strategy.indicators.adx` / `.rsi` and the agent indicator tool stay.

### 6. Dashboard

- `trades_chart` (`dashboard/components/charts.py`) takes the `"Regime"` pane out
  of `indicators` and renders it as a dedicated row instead of a generic pane:
  - sub-row under the portfolio chart, sharing its x-axis (same scale);
  - y axis fixed to `[-1.3, 1.3]` with ticks `-1 / 0 / 1` labelled Bearish /
    Neutral / Bullish;
  - a step line (`line_shape="hv"`) holding each value until the next refresh;
  - green / grey / red fills for the row, and the same colors as faint
    `add_vrect` bands behind the portfolio chart (one rect per run of equal
    values, so the figure stays small).
- Colors follow the dark theme tokens already in `components/charts.py`; never
  `plotly_white`.
- `reader.load_indicator_lines` already returns the `"Regime"` pane, so the
  reader needs no change. Old runs with ADX / VIX panes still render them as
  generic panes; a run with no regime data has no regime row.
- Row heights: portfolio 0.5, regime a fixed small share, remaining panes split
  the rest, as the current layout does.

### 7. Tests

- `classify_regime`: table-driven on synthetic series. Strong uptrend gives `+1`;
  downtrend gives `-1`; mixed (close above SMA200 but SMA50 below) gives `0`;
  a bullish series with a late volatility spike gives `0` (the cap); a
  bearish series with the same spike stays `-1`; fewer than `min_bars` gives
  `None`; invalid `RegimeParameters` raise.
- `Strategy._refresh_regime` with a fake broker: sets `regime`, logs, calls
  `add_line("Regime", ...)`; keeps the previous value on `BrokerError`; warns once
  on insufficient bars; adds no line outside backtesting.
- Backtest integration: with the benchmark data from a fake source and
  `warmup_trading_days=0`, `run_backtesting` still produces a regime from the
  first session (the warmup widening), and a bar closing after the cutoff never
  changes the value.
- Executor: `_refresh_regime` runs once per session, also when the run starts
  mid-session, and an exception in it does not stop the run.
- `trades_chart`: the regime row exists and shares the x-axis; its y range and
  colors; the `add_vrect` count equals the number of regime runs; a figure with
  no `"Regime"` pane is unchanged. Update `tests/dashboard/test_trades_chart.py`
  and `tests/dashboard/test_reader.py`, which use ADX / RSI / VIX as sample panes
  (rename the samples).

### 8. Docs

- CLAUDE.md: a gotcha entry (regime computed by default in the executor, backtest
  `add_line` only, warmup widened to `regime_params.min_bars`, strategies must not
  trade on it without a new spec), and the `core/` architecture line.
- `TODO.md`: strike through the Dashboard "Replace VIX, ADX..." item.
- `README.md`: mention only if it currently lists ADX / VIX / RSI charting (to be
  checked while implementing).

## Out of scope

- Using the regime to size or gate any strategy.
- Replacing cross_momentum's breadth overlay or the scorecard's manual Regime tag.
- Hysteresis, a weekly series, or an intraday regime.
- A non-SPY benchmark per strategy beyond what `benchmark_symbol` already allows.

## Risks

- **Thresholds are untuned.** 50/200 SMAs and an 80th-percentile vol cap are
  conventional defaults, not fitted. They are class-level parameters so they can
  change without touching the pure code.
- **Warmup cost** on every backtest (about 273 extra benchmark sessions). Accepted.
- **Alpaca IEX history depth** in paper/live: 273 daily bars is a single
  `get_bars` call, well inside the 200 requests/minute budget, once per session.

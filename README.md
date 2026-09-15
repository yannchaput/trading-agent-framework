# trading-agent-framework
A trading agent harness managing runtime, the broker layer, agent layer, memory etc.

## Environment variable management

This directory holds the environment files that `trading_agent_framework.config.env`
loads credentials and configuration from. **Only two files in this directory are
committed to git: `.env.example` and this `README.md`.** Everything else matching
`env/.env.*` is gitignored (see `/env/*` with `!/env/README.md` and
`!/env/.env.example` in the repository `.gitignore`) because those files carry real
secrets or point at real accounts.

## Naming convention

Strategy- and mode-specific env files follow:

```
env/.env.{strategy_name}.{live|paper|backtesting}
```

For example, a strategy named `momentum` running in paper mode reads from
`env/.env.momentum.paper`. `{strategy_name}` should match the `strategy_name` you
pass to `load_strategy_env`, and the mode suffix must be one of the three values in
`trading_agent_framework.config.env.TRADING_MODES`: `live`, `paper`, `backtesting`.

## Resolution order

`trading_agent_framework.config.env.resolve_env_file` (called by
`load_strategy_env`) looks for the first file that exists, in this order:

1. `env/.env.{strategy_name}.{trading_mode}` -- the strategy- and mode-specific file.
2. `env/.env` -- a shared fallback for all strategies/modes.
3. `.env` at the repository root -- a last-resort fallback.

The first candidate found is loaded (via `python-dotenv`, with `override=True`) and
its path is returned. If none of the three exist, `load_strategy_env` raises
`ConfigurationError`.

## Getting started

Copy `.env.example` to a strategy- and mode-specific file (or to `env/.env` for a
shared default) and fill in real values:

```bash
cp env/.env.example env/.env.momentum.paper
```

Never commit the copy -- it is already covered by the `env/.env.*` gitignore
pattern, so a plain `git add` will not pick it up.

## Backtesting

`Strategy.run_backtesting(start=..., end=...)` runs a strategy against simulated time
and simulated fills -- no network calls to a broker, and (with the default data
source) no Alpaca account needed at all.

```python
from datetime import datetime, timedelta

result = my_strategy.run_backtesting(
    start=datetime.now() - timedelta(days=365),
    end=datetime.now(),
)
print(result.metrics["sharpe_strategy"], result.run_dir)
```

`start`/`end`/`budget`/`benchmark` fall back to the `backtesting_start`/
`backtesting_end`/`budget`/`benchmark_symbol` class attributes when the matching
keyword argument is omitted (`budget` defaults to `Decimal("10000")` and
`benchmark_symbol` to `"SPY"` if neither is set). `run_backtesting` also accepts
`data_source`, `timestep` (`"day"` by default), `commission`, `slippage` and
`risk_free_rate` keyword arguments.

- **Data source**: defaults to `YahooBacktestData(start, end)` -- free daily OHLCV, no
  API key, requires the `backtesting-yahoo` extra (`uv sync --extra backtesting-yahoo`).
  Pass `data_source=AlpacaBacktestData(...)` for feed parity with paper/live trading,
  or wrap either in `backtesting.CachedDataSource(source, cache_dir)` to cache fetched
  bars under `<cache_dir>/<source.name>/` for network-free reruns against the same
  window (useful when iterating on an agent prompt against a fixed period).
- **Fill model**: orders fill against the *next* bar's open (never the bar they were
  submitted on), so a strategy can't trade a price it has already observed. Limit and
  stop orders fill only when the bar's range actually touches the trigger price.
- **Output**: `logs/<strategy>/backtesting/<timestamp>_backtesting/` -- `settings.json`,
  `metrics.json` (Sharpe, Sortino, Calmar, max drawdown, and the rest of the standard
  tearsheet), and three parquet files (`equity.parquet`, `trades.parquet`,
  `indicators.parquet`). `run_backtesting` returns a `BacktestResult` with `run_dir`,
  `settings` and `metrics` attributes pointing at the same data.
- **No look-ahead, structurally**: every price the strategy can see is gated by
  `clock.now()` -- a bar is visible only after it has *closed*. See
  `docs/superpowers/specs/2026-09-12-backtesting-framework-design.md` for the full
  design and its guarantees.

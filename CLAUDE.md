# trading-agent-framework

A lean, from-scratch replacement for a `lumibot`-based trading agent (see
`prompts/MIGRATION_PROMPT.md` for the original brief). Built to stay light
enough that a local GPU-hosted LLM (e.g. Qwen3-8B on a 24GB card) can drive
it without choking on dead code or oversized tool payloads. Supports three
trading modes: `live`, `paper`, `backtesting`.

## Commands

```bash
uv sync                                   # install/update dependencies
uv run agent <strategy_name> <live|paper|backtesting>  # run a strategy (see `Strategies` in `utils/strategy_factory.py`)
uv run dashboard [--benchmark-dir PATH]   # Streamlit dashboard: Backtesting tab (logs/ and memory/ run history) and Models tab (vLLM benchmark results)
uv run batch-universe                     # rebuild the cross_momentum stock universe file
uv run pytest                             # run the full test suite
uv run pytest tests/brokers/alpaca/       # run one directory/file
uv run ruff check                         # lint
uv run python scripts/tests/smoke_alpaca_orders.py     # manual paper-trading smoke test: broker orders
uv run python scripts/tests/smoke_strategy_paper.py    # manual paper-trading smoke test: strategy lifecycle
uv run python scripts/tests/smoke_alpaca_data.py       # manual paper-account smoke test: market data + indicators (read-only)
uv run python scripts/tests/smoke_quality_screen.py    # manual smoke test: fundamentals quality screen against real SEC + Yahoo (read-only)
uv run python scripts/tests/smoke_earnings_events.py    # manual smoke test: earnings_drift events + Benzinga surprises against real SEC + Alpaca news (read-only)
uv run python scripts/tests/smoke_congress_trades.py    # manual smoke test: congress_trades against the real House Clerk (yearly report + PTRs -> holdings, tiers, baseline weights; read-only; needs CONGRESS_USER_AGENT)
uv run python scripts/tests/smoke_ibkr_account.py      # manual paper IB Gateway smoke test: account (read-only)
uv run python scripts/tests/smoke_ibkr_orders.py       # manual paper IB Gateway smoke test: orders
```

Requires Python 3.14 (`.python-version`). Package manager is `uv`, not pip/poetry.

## Architecture

Layered, broker-agnostic by design:

- `entities/` -- pure data: `Order`, `Position`, `Asset`, `Quote`, `Bars`, enums. No I/O, no broker knowledge (`Bars` imports pandas only for type checking).
- `config/env.py` -- strategy/mode env file resolution, `BrokerSettings` (`BROKER`, `BROKER_API_IS_PAPER`), `IbkrSettings`, and `AlpacaCredentials.for_trading/for_news/for_data`.
- `clock.py` -- `MarketClock` ABC + `MarketSession`: the executor's only source of time and waiting (the seam a future backtest clock plugs into).
- `log.py` -- `ColorLogger` (`log_info`/`log_warning`/... with ANSI colours) and `setup_strategy_logging` (lumibot-style `logs/<strategy>/<mode>/<ts>_<mode>/<mode>.log`).

Detail for each area lives in a nested `CLAUDE.md`, loaded when you work under that directory (paths relative to `src/trading_agent_framework/`):

- `brokers/CLAUDE.md` -- `Broker` template, `OrderTracker`, fees, Alpaca and IBKR brokers, market-data and IBKR gotchas.
- `backtesting/CLAUDE.md` -- `BacktestBroker`, data sources, fills, ledger, report, and every backtest-fidelity rule.
- `core/CLAUDE.md` -- `Strategy`, `StrategyExecutor`, timing, the market regime.
- `agents/CLAUDE.md` -- `AgentManager`, telemetry, tool-call repair, and the research-tool clock gate (`agents/tools/`).
- `memory/CLAUDE.md` -- `MemoryStore` and the memory tools.
- `fundamentals/CLAUDE.md` -- SEC EDGAR client.
- `dashboard/CLAUDE.md` -- the Streamlit app.
- `strategies/CLAUDE.md` -- the strategy index; `strategies/common/` shared code; one `CLAUDE.md` per strategy (`bill_ackman`, `bull_bear`, `congress_trades`, `cross_momentum`, `earnings_drift`, `news_builtin`, `vwap_pullback`).

## Key patterns / gotchas

- **Money is `Decimal`** everywhere except three deliberate float boundaries: `orders.py` (Alpaca's SDK wants floats for some request fields; `brokers/ibkr/orders.py` is the same boundary for `ib_async`), `Bars.df` (float64 OHLCV for indicator maths, built only in `market_data._bars_frame`), and the backtesting ledger-to-reporting seam (`backtesting/metrics.py` and `backtesting/report.py` -- vectorbt and JSON/parquet output both need float64; everything upstream of that seam, including `backtesting/broker.py`, `backtesting/fills.py` and `backtesting/ledger.py`, stays exact `Decimal`). Don't add a fourth.
- **Strategy code runs on one thread.** Order hooks (`on_filled_order`, ...) are dispatched by the executor while it waits (between ticks, in `strategy.sleep`, in `wait_for_order_execution`) -- never on the Alpaca stream thread. Broker/stream code must only feed `OrderTracker`; it never calls strategy hooks.
- **An exception in `on_trading_iteration` does NOT stop the run.** The executor logs it, calls `on_bot_crash` and carries on with the next tick (live trading must ride out a bad tick), so a plain `raise` cannot abort a backtest. To end a run on purpose, raise `FatalStrategyError` (`utils/errors.py`) from `on_trading_iteration`: the executor reports it like a failed `initialize` (log, `on_bot_crash`, `on_strategy_end`) and propagates it, so `run_backtest` surfaces a `BacktestError` and writes no report. `NewsBuiltinStrategy` uses this to abort a backtest after 3 consecutive LLM failures. Only `on_trading_iteration` honours this: the other lifecycle hooks (bar `initialize`, which already aborts the run) swallow every exception.
- **No `time.sleep` / `datetime.now` in strategy or executor code** -- go through `strategy.sleep()` / `strategy.clock` so a simulated clock can drive backtests later.
- **ANSI colour codes stay in log files on purpose** (read with VS Code's "ANSI Colors" plugin). Don't strip them and don't switch to `termcolor` (it drops colour when stdout isn't a TTY).
- `strategy.name` is `broker.strategy_name` (it prefixes every `client_order_id` and decides which open orders `sync_open_orders` adopts after a restart).
- Tests use **hand-written fakes** (`tests/fakes.py`) built from real `alpaca.trading.models`/`alpaca.trading.requests` objects, not `MagicMock` -- several tests assert on the exact request object built, which is much cleaner against a fake than a mock.
- The automated test suite **never touches the network**. Scripts in `scripts/tests/` are intentionally excluded from `tests/` and run by hand against real paper-trading credentials.
- `tests/conftest.py` resets package logging after every test; `setup_strategy_logging` disables propagation, which would otherwise starve other tests' `caplog`.
- Env file resolution and naming conventions (`env/.env.{strategy}.{mode}`) are documented in `README.md` -- read that before adding new env-dependent config rather than re-deriving it here.
- **No data tool may use wall-clock time.** A backtest's no-look-ahead guarantee (`backtesting/data/base.py`) holds only as far as the chokepoint it controls -- `BacktestDataSource.bars()`. Any other source of "now" inside a data-fetching tool (news, SEC fundamentals, a future screener) must take its cutoff from `strategy.clock.now()`, exactly as `MemoryStore` already takes `now=self.clock.now`, or it silently leaks future information into a backtest.
- **Credential groups never fall back.** `ALPACA_API_*` (Alpaca trading), `ALPACA_DATA_*` (market data and calendar for both brokers, and `AlpacaBacktestData`), `ALPACA_NEWS_*` (the news tool). `ALPACA_IS_PAPER` is rejected (renamed `BROKER_API_IS_PAPER`).
- **Every strategy gets a market regime by default, and nothing trades on it.** The executor refreshes it once per session (`strategy.regime`: 1 bullish / 0 neutral / -1 bearish, `None` without enough history). Using it to size or gate a strategy is a new design decision. Details and the warmup rule: `core/CLAUDE.md`.
- **Research agent tools (news/macro/fundamentals) gate on `strategy.clock.now()`**, the same rule as the data tools above, never the price-bar chokepoint. Details: `agents/CLAUDE.md`.

## Development workflow

Work proceeds as numbered tasks from a Superpowers implementation plan (see git log: "Task N", "Tasks X-Y"). Each task is typically followed by a review-fix commit before the next task starts. Follow that same cadence for new work in this repo.

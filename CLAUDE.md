# trading-agent-framework

A lean, from-scratch replacement for a `lumibot`-based trading agent (see
`prompts/MIGRATION_PROMPT.md` for the original brief). Built to stay light
enough that a local GPU-hosted LLM (e.g. Qwen3-8B on a 24GB card) can drive
it without choking on dead code or oversized tool payloads. Supports three
trading modes: `live`, `paper`, `backtesting`.

## Commands

```bash
uv sync                                   # install/update dependencies
uv run pytest                             # run the full test suite
uv run pytest tests/brokers/alpaca/       # run one directory/file
uv run ruff check                         # lint
uv run python scripts/tests/smoke_alpaca_orders.py     # manual paper-trading smoke test: broker orders
uv run python scripts/tests/smoke_strategy_paper.py    # manual paper-trading smoke test: strategy lifecycle
uv run python scripts/tests/smoke_alpaca_data.py       # manual paper-account smoke test: market data + indicators (read-only)
```

Requires Python 3.14 (`.python-version`). Package manager is `uv`, not pip/poetry.

## Architecture

Layered, broker-agnostic by design:

- `entities/` -- pure data: `Order`, `Position`, `Asset`, `Quote`, `Bars`, enums. No I/O, no broker knowledge (`Bars` imports pandas only for type checking).
- `brokers/base.py` -- abstract `Broker` template method (`_conform_order` then `_submit_order`), plus `OrderTracker` wiring. Never imports `alpaca`.
- `brokers/tracker.py` -- `OrderTracker`/`SafeList`: thread-safe order state machine (`unprocessed -> new -> (partially_filled ->)* filled`, or `-> canceled/error`). Used by stream and polling threads concurrently.
- `brokers/alpaca/orders.py` -- **pure** translation functions only (status/event maps, price conforming, request building, response parsing). No I/O, no state, no client instances. Together with `account.py`, the only modules allowed to import `alpaca.trading.requests` (`market_data.py` is the only one allowed to import `alpaca.data.requests`).
- `brokers/alpaca/broker.py` -- `AlpacaBroker`: wires the real `TradingClient` and `StockHistoricalDataClient` I/O to the pure modules plus tracker bookkeeping. Contains no translation logic itself.
- `brokers/alpaca/client.py` -- trivial factory functions for building Alpaca SDK clients from credentials, so tests can inject fakes instead.
- `brokers/alpaca/stream.py` -- `AlpacaTradeStream`: trade-update handler and thread lifecycle for the live order stream.
- `config/env.py` -- strategy/mode env file resolution and `AlpacaCredentials`.
- `brokers/__init__.py` -- lazy-imports `AlpacaBroker`/`AlpacaTradeStream` via module `__getattr__` so `import trading_agent_framework.brokers` alone doesn't pull in `alpaca`/pandas.
- `clock.py` -- `MarketClock` ABC + `MarketSession`: the executor's only source of time and waiting (the seam a future backtest clock plugs into).
- `brokers/alpaca/account.py` -- **pure** account and calendar translation (same rules as `orders.py`).
- `brokers/alpaca/market_data.py` -- **pure** market-data translation (same rules as `orders.py`): timesteps (`"minute"`/`"day"` only), the calendar-based bars window, IEX request builders, and bar/trade/quote parsing.
- `brokers/alpaca/clock.py` -- `AlpacaMarketClock`: sessions (early closes included) from Alpaca's calendar, cached ~10 trading days.
- `core/` -- `Strategy` (lumibot hook names/signatures, broker facade, paper/live runners), `StrategyExecutor` (single-threaded session loop), `timing.py` (pure `sleeptime` parsing and tick maths), `events.py` (stream-thread → executor-thread order-event queue), `indicators.py` (`strategy.indicators.<pandas-ta name>(asset, ...)`, no cache).
- `memory/` -- agent memory (lumibot's `strategy.memory`): `records.py` (**pure**: ids, JSON, lean items, search scoring), `store.py` (`MemoryStore`, the only SQLite code: append-only `memory_events`, the `memory_index` projection, `memory_retrievals`; one DB per strategy and mode at `memory/<strategy>/<mode>/memory.sqlite`), `tools.py` (lumibot's 9 memory tools as plain typed functions, plus `agent_call_context` for provenance). No LangChain here: the agent layer wraps the tools.
- `agents/` -- LangChain agent creation/execution (lumibot's `strategy.agents`): `config.py` (**pure**:
  `LLMCredentials.from_env`, mirrors `AlpacaCredentials`), `results.py` (**pure**: `AgentRunResult`/
  `ToolCallRecord`, and a duck-typed parser from a LangChain message list), `manager.py`
  (`AgentManager`/`AgentHandle`, the only place that imports `langchain`/`langchain_openai`, and only
  inside method bodies). No builtin tools, no MCP/skills: the caller passes `tools=[...]` explicitly (e.g.
  `memory_tools(self.memory)`).
- `backtesting/` -- the third trading mode. `clock.py` (`BacktestClock`, simulated time), `broker.py` (`BacktestBroker`, simulated fills via `fills.py`'s pure OHLC rules), `ledger.py` (fills/equity/indicator lines, `Decimal`), `data/` (`BacktestDataSource` ABC, `CachedDataSource`, `YahooBacktestData` default, `AlpacaBacktestData`), `metrics.py` (vectorbt reporting) and `report.py` (writes the run to `logs/<strategy>/backtesting/<ts>_backtesting/`) are the pure-to-I/O layers; `runner.py` orchestrates a run and `Strategy.run_backtesting()` is the public entry point. The executor itself needed no changes -- `MarketClock.max_wait_slice` (`60.0` live, `math.inf` simulated) is the only seam it exposed.
- `log.py` -- `ColorLogger` (`log_info`/`log_warning`/... with ANSI colours) and `setup_strategy_logging` (lumibot-style `logs/<strategy>/<mode>/<ts>_<mode>/<mode>.log`).

## Key patterns / gotchas

- **Money is `Decimal`** everywhere except three deliberate float boundaries: `orders.py` (Alpaca's SDK wants floats for some request fields), `Bars.df` (float64 OHLCV for indicator maths, built only in `market_data._bars_frame`), and the backtesting ledger-to-reporting seam (`backtesting/metrics.py` and `backtesting/report.py` -- vectorbt and JSON/parquet output both need float64; everything upstream of that seam, including `backtesting/broker.py`, `backtesting/fills.py` and `backtesting/ledger.py`, stays exact `Decimal`). Don't add a fourth.
- **Never let a raw SDK/pydantic exception escape.** Wrap broker failures in `BrokerError`/`OrderValidationError`/`OrderEventError` (see `errors.py`). `orders.py`'s `validate_order` wraps pydantic `ValidationError`; `AlpacaBroker._submit_order` wraps client exceptions via `order.set_error(exc)` *before* re-raising (lumibot compatibility contract -- don't reorder this).
- **`orders.py`, `account.py` and `market_data.py` stay pure.** No I/O, no client instances, no state. If you need to call the Alpaca API, that logic belongs in `broker.py`, not here.
- **Strategy code runs on one thread.** Order hooks (`on_filled_order`, ...) are dispatched by the executor while it waits (between ticks, in `strategy.sleep`, in `wait_for_order_execution`) -- never on the Alpaca stream thread. Broker/stream code must only feed `OrderTracker`; it never calls strategy hooks.
- **No `time.sleep` / `datetime.now` in strategy or executor code** -- go through `strategy.sleep()` / `strategy.clock` so a simulated clock can drive backtests later.
- **ANSI colour codes stay in log files on purpose** (read with VS Code's "ANSI Colors" plugin). Don't strip them and don't switch to `termcolor` (it drops colour when stdout isn't a TTY).
- `strategy.name` is `broker.strategy_name` (it prefixes every `client_order_id` and decides which open orders `sync_open_orders` adopts after a restart).
- Tests use **hand-written fakes** (`tests/fakes.py`) built from real `alpaca.trading.models`/`alpaca.trading.requests` objects, not `MagicMock` -- several tests assert on the exact request object built, which is much cleaner against a fake than a mock.
- The automated test suite **never touches the network**. Scripts in `scripts/tests/` are intentionally excluded from `tests/` and run by hand against real paper-trading credentials.
- `tests/conftest.py` resets package logging after every test; `setup_strategy_logging` disables propagation, which would otherwise starve other tests' `caplog`.
- Env file resolution and naming conventions (`env/.env.{strategy}.{mode}`) are documented in `README.md` -- read that before adding new env-dependent config rather than re-deriving it here.
- **Market data:** `get_last_price` is the last *trade* (not the quote midpoint; use `get_quote(...).mid`). Every data request uses the IEX feed; bars are split/dividend-adjusted. `get_bars` makes one `get_calendar` call to start its window at the right session. Nothing is cached: every call hits the API (200 requests/min on the free IEX plan).
- **No data tool may use wall-clock time.** A backtest's no-look-ahead guarantee (`backtesting/data/base.py`) holds only as far as the chokepoint it controls -- `BacktestDataSource.bars()`. Any other source of "now" inside a data-fetching tool (news, SEC fundamentals, a future screener) must take its cutoff from `strategy.clock.now()`, exactly as `MemoryStore` already takes `now=self.clock.now`, or it silently leaks future information into a backtest.
- **`BacktestBroker._source_bars` is the gate** (design spec §5.2): the single place the broker reads a `BacktestDataSource`, and the one defensive check that no returned row closes after the cutoff. Route any new price path through it rather than calling `self._data_source.bars(...)` directly. A source that can't map a bar to a real session must raise `BacktestDataError` rather than fall back to a plausible-looking index -- silently early data is the failure mode this whole subsystem exists to prevent.
- **Backtest `start`/`end` must be timezone-aware** (`MarketSession` enforces tz-aware bounds); `run_backtest` validates this up front and says so, so a naive `datetime.now()` never reaches a comparison deep in the clock.
- **`equity.parquet` and `metrics.json` are built from the SAME session-reduced series** (`runner._session_equity_samples`): exactly one row per trading session, stamped at the session's close. `BacktestBroker.on_advance` samples equity on every clock advance, so the raw ledger is oversampled several-to-one; feeding that to either consumer silently understates annualised metrics and breaks the benchmark join. Don't hand `ledger.equity` straight to `report.write_equity`.
- **Memory tools are token-budgeted.** They return lean payloads (`{id, kind, status}`, or lean search items without metadata) and have one-line docstrings, because both reach the LLM on every call. Weigh the token cost before adding a field. Validation problems come back as `{"error": ...}`; database failures raise `MemoryStoreError`. `tools.py` deliberately has no `from __future__ import annotations` (the agent layer reads real annotations).
- **`MemoryStore` never knows `Strategy`.** It gets time from injected `now` / `wall_clock` callables (the strategy passes `clock.now`) and held positions as `HeldPosition` arguments to `compact_state`. Keep it that way so the store stays testable and backtest-clock friendly.
- **Backtesting memory is wiped** at the first `strategy.memory` access of every backtest run (`fresh=True`), so one run's lessons can't leak into the next. Paper and live memory are never wiped.
- **`agents/` defers every LangChain import to inside method bodies** (never at module level), mirroring
  `core/indicators.py`'s deferred `pandas_ta_classic` import -- a strategy that never calls
  `self.agents.create(...)` never pays for LangChain in memory. `AgentManager` doesn't know `Strategy`
  either; it takes a `credentials_source` callable (`LLMCredentials.from_env`).
- **`AgentRunResult.tool_calls[i].result` is a string, not the tool's return value.** LangChain always
  stringifies (JSON-encodes) a tool's return value into `ToolMessage.content` before it reaches agent code,
  so the original Python object (e.g. a memory tool's `{id, kind, status}` dict) isn't recoverable there --
  read it from the tool's own return value directly if you need it as data, not from the agent's result.
- **`Strategy.agents.create(..., timeout_seconds=None)` (the default) means no request timeout at all.**
  Deliberate: this framework targets slow, reasoning-capable local models that can legitimately take
  minutes per call, and a framework-imposed default would silently break that use case. The cost: per
  "Strategy code runs on one thread" above, a hung or OOM'd local LLM server blocks `on_trading_iteration()`
  -- and therefore order-event hook dispatch -- indefinitely. Pass an explicit `timeout_seconds` if your
  deployment needs one.

## Development workflow

Work proceeds as numbered tasks from a Superpowers implementation plan (see git log: "Task N", "Tasks X-Y"). Each task is typically followed by a review-fix commit before the next task starts. Follow that same cadence for new work in this repo.

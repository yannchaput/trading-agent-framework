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
- `log.py` -- `ColorLogger` (`log_info`/`log_warning`/... with ANSI colours) and `setup_strategy_logging` (lumibot-style `logs/<strategy>/<mode>/<ts>_<mode>/<mode>.log`).

## Key patterns / gotchas

- **Money is `Decimal`** everywhere except two deliberate float boundaries: `orders.py` (Alpaca's SDK wants floats for some request fields) and `Bars.df` (float64 OHLCV for indicator maths, built only in `market_data._bars_frame`). Don't add a third.
- **Never let a raw SDK/pydantic exception escape.** Wrap broker failures in `BrokerError`/`OrderValidationError`/`OrderEventError` (see `errors.py`). `orders.py`'s `validate_order` wraps pydantic `ValidationError`; `AlpacaBroker._submit_order` wraps client exceptions via `order.set_error(exc)` *before* re-raising (lumibot compatibility contract -- don't reorder this).
- **`orders.py`, `account.py` and `market_data.py` stay pure.** No I/O, no client instances, no state. If you need to call the Alpaca API, that logic belongs in `broker.py`, not here.
- **Strategy code runs on one thread.** Order hooks (`on_filled_order`, ...) are dispatched by the executor while it waits (between ticks, in `strategy.sleep`, in `wait_for_order_execution`) -- never on the Alpaca stream thread. Broker/stream code must only feed `OrderTracker`; it never calls strategy hooks.
- **No `time.sleep` / `datetime.now` in strategy or executor code** -- go through `strategy.sleep()` / `strategy.clock` so a simulated clock can drive backtests later.
- **ANSI colour codes stay in log files on purpose** (read with VS Code's "ANSI Colors" plugin). Don't strip them and don't switch to `termcolor` (it drops colour when stdout isn't a TTY).
- `strategy.name` is `broker.strategy_name` (it prefixes every `client_order_id` and decides which open orders `sync_open_orders` adopts after a restart).
- Tests use **hand-written fakes** (`tests/fakes.py`) built from real `alpaca.trading.models`/`alpaca.trading.requests` objects, not `MagicMock` -- several tests assert on the exact request object built, which is much cleaner against a fake than a mock.
- The automated test suite **never touches the network**. Scripts in `scripts/tests/` are intentionally excluded from `tests/` and run by hand against real paper-trading credentials.
- `tests/conftest.py` resets package logging after every test; `setup_strategy_logging` disables propagation, which would otherwise starve other tests' `caplog`.
- Env file resolution and naming conventions (`env/.env.{strategy}.{mode}`) are documented in `env/README.md` -- read that before adding new env-dependent config rather than re-deriving it here.
- **Market data:** `get_last_price` is the last *trade* (not the quote midpoint; use `get_quote(...).mid`). Every data request uses the IEX feed; bars are split/dividend-adjusted. `get_bars` makes one `get_calendar` call to start its window at the right session. Nothing is cached: every call hits the API (200 requests/min on the free IEX plan).

## Development workflow

Work proceeds as numbered tasks from a Superpowers implementation plan (see git log: "Task N", "Tasks X-Y"). Each task is typically followed by a review-fix commit before the next task starts. Follow that same cadence for new work in this repo.

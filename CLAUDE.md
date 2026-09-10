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
uv run python scripts/smoke_alpaca_orders.py  # manual paper-trading smoke test (see below)
```

Requires Python 3.14 (`.python-version`). Package manager is `uv`, not pip/poetry.

## Architecture

Layered, broker-agnostic by design:

- `entities/` -- pure data: `Order`, `Position`, `Asset`, enums. No I/O, no broker knowledge.
- `brokers/base.py` -- abstract `Broker` template method (`_conform_order` then `_submit_order`), plus `OrderTracker` wiring. Never imports `alpaca`.
- `brokers/tracker.py` -- `OrderTracker`/`SafeList`: thread-safe order state machine (`unprocessed -> new -> (partially_filled ->)* filled`, or `-> canceled/error`). Used by stream and polling threads concurrently.
- `brokers/alpaca/orders.py` -- **pure** translation functions only (status/event maps, price conforming, request building, response parsing). No I/O, no state, no client instances. This is the only module allowed to import `alpaca.trading.requests`.
- `brokers/alpaca/broker.py` -- `AlpacaBroker`: wires the real `TradingClient` I/O to `orders.py`'s pure functions plus tracker bookkeeping. Contains no translation logic itself.
- `brokers/alpaca/client.py` -- trivial factory functions for building Alpaca SDK clients from credentials, so tests can inject fakes instead.
- `brokers/alpaca/stream.py` -- `AlpacaTradeStream`: trade-update handler and thread lifecycle for the live order stream.
- `config/env.py` -- strategy/mode env file resolution and `AlpacaCredentials`.
- `brokers/__init__.py` -- lazy-imports `AlpacaBroker`/`AlpacaTradeStream` via module `__getattr__` so `import trading_agent_framework.brokers` alone doesn't pull in `alpaca`/pandas.

## Key patterns / gotchas

- **Money is `Decimal`** everywhere except one deliberate float boundary in `orders.py` (Alpaca's SDK wants floats for some fields) -- don't introduce a second float conversion point.
- **Never let a raw SDK/pydantic exception escape.** Wrap broker failures in `BrokerError`/`OrderValidationError`/`OrderEventError` (see `errors.py`). `orders.py`'s `validate_order` wraps pydantic `ValidationError`; `AlpacaBroker._submit_order` wraps client exceptions via `order.set_error(exc)` *before* re-raising (lumibot compatibility contract -- don't reorder this).
- **`orders.py` stays pure.** No I/O, no client instances, no state. If you need to call the Alpaca API, that logic belongs in `broker.py`, not here.
- Tests use **hand-written fakes** (`tests/fakes.py`) built from real `alpaca.trading.models`/`alpaca.trading.requests` objects, not `MagicMock` -- several tests assert on the exact request object built, which is much cleaner against a fake than a mock.
- The automated test suite **never touches the network**. The one script that does is `scripts/smoke_alpaca_orders.py`, which is intentionally excluded from `tests/` and run by hand against real paper-trading credentials.
- Env file resolution and naming conventions (`env/.env.{strategy}.{mode}`) are documented in `env/README.md` -- read that before adding new env-dependent config rather than re-deriving it here.

## Development workflow

Work proceeds as numbered tasks from a Superpowers implementation plan (see git log: "Task N", "Tasks X-Y"). Each task is typically followed by a review-fix commit before the next task starts. Follow that same cadence for new work in this repo.

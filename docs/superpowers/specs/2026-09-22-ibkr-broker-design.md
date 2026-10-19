# IBKR broker support -- design

Date: 2026-09-22
Status: approved (brainstorming), pending implementation plan

## 1. Goal and scope

Add Interactive Brokers (IBKR) as a second live/paper broker next to Alpaca, selectable per
strategy env file, without changing the memory system, the agent tools' interfaces, or the
backtesting subsystem's data sources.

In scope:

- A broker selector (`BROKER=alpaca|ibkr`) and a generic paper flag (`BROKER_API_IS_PAPER`),
  with a hard rename of `ALPACA_IS_PAPER`.
- Separate, explicit Alpaca credential groups for trading, news and market data.
- A broker factory used by `main.py` for paper/live; backtests no longer build a live broker.
- Extraction of Alpaca market data out of `AlpacaBroker` into a reusable `AlpacaMarketData`.
- `IbkrBroker` (paper and live) over the **TWS API via IB Gateway**, using `ib_async`.

Out of scope (decided):

- **No IBKR backtest data source.** Backtests keep using `YahooBacktestData` or
  `AlpacaBacktestData`, chosen in strategy code as today.
- **No IBKR news.** News always comes from Alpaca (`AlpacaNewsProvider`), in every mode and
  for every broker -- the same stream `news_binary` is backtested against.
- **No IBKR market data.** Prices, quotes, bars and the market calendar always come from
  Alpaca (IEX feed) through `ALPACA_DATA_*`. With `BROKER=ibkr`, IBKR is used for trading,
  account and positions only. No IBKR market-data subscription is required.
- Non-US markets, non-USD accounts, options/futures, bracket/OCO orders, notional orders on IBKR.
- The IBKR Web API / Client Portal Gateway (rejected: browser login + daily re-auth, no
  key/secret for individual accounts).
- Editing the user's real `env/.env.*` files (they hold secrets; the user renames them).

Unchanged: `MemoryStore` and memory tools, agent telemetry, `BacktestBroker`, the executor,
`NewsProvider` protocol, macro/fundamentals tools.

## 2. Env contract

| Variable | Used by | Required when | Default |
|---|---|---|---|
| `BROKER` | broker factory | optional | `alpaca` |
| `BROKER_API_IS_PAPER` | broker factory, `IbkrSettings`, account checks | optional | `true` |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | `AlpacaBroker` trading client and stream | `BROKER=alpaca`, paper/live | -- |
| `IBKR_HOST` | `IbkrBroker` | `BROKER=ibkr` | `127.0.0.1` |
| `IBKR_PORT` | `IbkrBroker` | `BROKER=ibkr` | `4002` if paper, `4001` if live |
| `IBKR_CLIENT_ID` | `IbkrBroker` | `BROKER=ibkr` | `1` |
| `ALPACA_NEWS_API_KEY`, `ALPACA_NEWS_API_SECRET` | `AlpacaNewsProvider` | a strategy uses `news_tools` (any mode) | -- |
| `ALPACA_DATA_API_KEY`, `ALPACA_DATA_API_SECRET` | `AlpacaMarketData`, `AlpacaMarketClock` (via data creds), `AlpacaBacktestData` | paper/live (both brokers); backtests using `AlpacaBacktestData` | -- |
| `ALPACA_DATA_IS_PAPER` | selects the paper/live Alpaca trading endpoint for the **calendar** call only (paper keys are rejected by the live endpoint) | optional | `true` |

Rules:

- Each group is always required by the component that uses it. There is no fallback between
  groups, even with `BROKER=alpaca`; a user with one Alpaca key pair repeats it in each group.
- `ALPACA_IS_PAPER` present in the environment raises
  `ConfigurationError("ALPACA_IS_PAPER was renamed to BROKER_API_IS_PAPER")` from every
  `from_env`/`for_*` constructor below (hard rename, no alias).
- `YahooBacktestData` needs no variables.
- `BROKER` has no meaning in backtesting mode.
- Two IBKR strategies running at the same time must use different `IBKR_CLIENT_ID`s, and a
  strategy must keep the same one across restarts (IBKR only lets the placing client id
  modify/cancel an order).

## 3. Sub-project A -- configuration, factory, entry point

### 3.1 `config/env.py`

- `BrokerKind(StrEnum)`: `ALPACA = "alpaca"`, `IBKR = "ibkr"`.
- `BrokerSettings` (frozen dataclass: `kind`, `is_paper`) with `from_env(env=None)`. Unknown
  `BROKER` value raises `ConfigurationError` listing the valid values. `BROKER_API_IS_PAPER`
  is parsed with the existing `_FALSE_PAPER_VALUES`.
- `AlpacaCredentials` keeps its fields (`api_key`, `api_secret`, `is_paper`). Its generic
  `from_env()` is **removed** and replaced by three named constructors, so no call site can
  read the wrong pair:
  - `for_trading(env=None)`: `ALPACA_API_KEY`/`ALPACA_API_SECRET`, `is_paper` from
    `BROKER_API_IS_PAPER`.
  - `for_news(env=None)`: `ALPACA_NEWS_API_KEY`/`ALPACA_NEWS_API_SECRET`; `is_paper=True`
    (irrelevant to news clients).
  - `for_data(env=None)`: `ALPACA_DATA_API_KEY`/`ALPACA_DATA_API_SECRET`, `is_paper` from
    `ALPACA_DATA_IS_PAPER`.
  - Each error message names exactly the missing variables.
- `IbkrSettings` (frozen dataclass: `host`, `port`, `client_id`, `is_paper`) with
  `from_env(is_paper, env=None)`. Non-integer port/client id raise `ConfigurationError`.
- A shared private `_reject_renamed(source)` performs the `ALPACA_IS_PAPER` check.
- `config/__init__.py` exports the new names.

### 3.2 `brokers/factory.py` (new)

```python
def build_broker(strategy_name: str, *, env: Mapping[str, str] | None = None) -> Broker
```

- Paper/live only. Reads `BrokerSettings.from_env(env)`.
- `alpaca`: `AlpacaBroker.from_credentials(strategy_name, trading=AlpacaCredentials.for_trading(env), data=AlpacaCredentials.for_data(env), news=lambda: AlpacaCredentials.for_news(env))`.
- `ibkr`: `IbkrBroker.from_settings(strategy_name, IbkrSettings.from_env(settings.is_paper, env), data=AlpacaCredentials.for_data(env), news=lambda: AlpacaCredentials.for_news(env))`.
- News credentials are passed as a callable and resolved on first `news_provider()` call, so a
  strategy without `news_tools` (e.g. `cross_momentum`) needs no `ALPACA_NEWS_*`.
- Broker classes are imported inside the function body (keeps `import
  trading_agent_framework.brokers` light, as `brokers/__init__.py` does today).
- Builders are looked up from a module-level mapping so tests can inject fakes.

### 3.3 Call-site migration

- `AlpacaBroker.from_credentials(strategy_name, *, trading, data, news, with_stream=True)`:
  trading client and stream from `trading`; `AlpacaMarketData` from `data`; lazy
  `AlpacaNewsProvider` from `news()`; `is_paper` from `trading.is_paper`;
  `configure_account()` still called on the trading account.
- `AlpacaBacktestData._real_client/_real_trading_client` use `AlpacaCredentials.for_data()`.
- `BacktestBroker.news_provider()` default uses `AlpacaCredentials.for_news()`.
- Smoke scripts under `scripts/tests/` move to the new constructors and messages
  (`BROKER_API_IS_PAPER` instead of `ALPACA_IS_PAPER`).
- Docstrings mentioning `AlpacaCredentials.from_env()` (`core/strategy.py`) are updated.

### 3.4 `main.py` and `PlaceholderBroker`

- `StrategyBuilder = Callable[[Broker, TradingMode], Strategy | None]`.
- Paper/live: `broker = build_broker(strategy_name)`; `ConfigurationError` is printed and exits
  with status 1.
- Backtesting: **no live broker is built** (today `main.py` builds a real `AlpacaBroker`,
  which calls `configure_account()` on the live Alpaca account just to create a placeholder).
  It builds `PlaceholderBroker(strategy_name)` instead.
- `backtesting/placeholder.py` (new): `PlaceholderBroker(Broker)`, `name = "placeholder"`,
  carries `strategy_name` and a `BacktestClock` (so `Strategy.__init__`'s `broker.clock`
  works); every abstract operation raises
  `BrokerError("not available before the backtest starts")`. `run_backtest` replaces it
  with `BacktestBroker` exactly as it replaces today's placeholder.
- `scripts/tests/smoke_backtest.py` uses `PlaceholderBroker`.

### 3.5 `brokers/alpaca/data.py` -- `AlpacaMarketData` (extracted)

- Moves out of `AlpacaBroker` unchanged: `get_last_price`, `get_last_prices`, `get_quote`,
  `get_bars`, `_sessions_before`.
- Constructor: `(data_client, calendar_client, clock: MarketClock)`; `clock.now()` is the
  `end` of `get_bars`.
- `AlpacaMarketData.from_credentials(creds, clock)` builds both clients from `for_data()`
  credentials (`calendar_client` is a `TradingClient`, cast once as `AlpacaBroker` does today).
- `AlpacaBroker` delegates its four market-data methods to it. Existing broker tests must
  pass with only construction changes.
- `market_data.py` (pure) is unchanged.

## 4. Sub-project B -- `IbkrBroker`

### 4.1 Package `brokers/ibkr/`

| Module | Purity | Role |
|---|---|---|
| `orders.py` | pure | `Order` -> `ib_async` `Contract`/`Order`; IBKR status -> `OrderStatus`/`OrderEvent`; `Trade`/`Position`/`Fill` -> entities; quantity conforming and validation. |
| `account.py` | pure | `accountSummary` tags -> `AccountBalances`; paper-account check; cash-account check; base-currency check. |
| `client.py` | I/O | `IbkrConnection`: owns `ib_async.IB` on a dedicated event-loop thread (the only threads/asyncio code in the package). |
| `events.py` | I/O | Order-event handlers on the loop thread; feed `OrderTracker` only. |
| `broker.py` | I/O | `IbkrBroker(Broker)`, `name = "ibkr"`; wiring only, no translation logic. |

Only `orders.py`, `account.py`, `client.py` and `events.py` import `ib_async`, and
`brokers/__init__.py` lazily exports `IbkrBroker` like `AlpacaBroker`. `ib_async` is added to
`pyproject.toml`; an Alpaca-only run never imports it.

### 4.2 `IbkrConnection` (`client.py`)

- Starts a daemon thread running its own asyncio loop and an `IB` instance.
- `connect(host, port, client_id, timeout=15.0)`; `disconnect()` stops the loop and joins the
  thread.
- `call(coro_factory, timeout=30.0)`: submits a coroutine to the loop with
  `asyncio.run_coroutine_threadsafe` and blocks the executor thread for the result. Timeout
  raises `BrokerError("IB Gateway did not answer within Ns")`; any `ib_async` exception is
  wrapped in `BrokerError`. No raw `ib_async` exception escapes.
- Tracks connected state from `IB.disconnectedEvent`. When down, the next `call` reconnects
  (3 tries, short backoff) and then runs the broker's reconcile callback (4.6); if that
  fails, it raises `BrokerError`. No watchdog thread.

### 4.3 Order mapping (`orders.py`)

- Contract: `Stock(symbol, "SMART", "USD")`. Qualified conids are cached per symbol in the
  broker (`qualifyContracts` once per symbol).
- Order types: `MARKET`->`MKT`, `LIMIT`->`LMT`, `STOP`->`STP`, `STOP_LIMIT`->`STP LMT`,
  `TRAIL`->`TRAIL` (`auxPrice` = `trail_price`, or `trailingPercent` = `trail_percent`).
- Time in force: `DAY`, `GTC`, `IOC`, `FOK`, `OPG` map directly. `CLS` becomes order type
  `MOC` (market) or `LOC` (limit).
- `extended_hours` -> `outsideRth`.
- `orderRef` = our `client_order_id` (`{strategy}:{identifier}`).
- `notional` orders raise `OrderValidationError` (no reliable fractional API support; no
  strategy or tool uses notional today).
- Fractional `quantity` is floored to whole shares with a warning; floored to zero raises
  `OrderValidationError`.
- Status map: `PendingSubmit`/`PreSubmitted`/`Submitted` -> `NEW`; `Filled` -> `FILL`;
  `Cancelled`/`ApiCancelled` -> `CANCELED`; `Inactive` -> `ERROR`; `PendingCancel` ->
  `CANCELLING`; anything else -> `UNKNOWN` (logged).

### 4.4 Broker operations (`broker.py`)

- `_submit_order`: stamp `client_order_id` if missing, validate and build the order, qualify
  the contract, `track_unprocessed` **before** `placeOrder` (the event handlers may fire
  first), then place. A synchronous rejection (`errorEvent` for that order id or `Inactive`
  status during submit) calls `order.set_error(exc)` **before** raising `BrokerError`
  (lumibot contract preserved), and untracks the order.
- `order.identifier` stays our own generated id (IBKR's `permId` is often `0` until the first
  status arrives, so it cannot key the tracker at submit time). It is recoverable from
  `orderRef` (`{strategy}:{identifier}`), which is how `sync_open_orders` rebuilds it after a
  restart. `orderId`/`permId` are kept in `order.raw` for cancel/modify.
- `cancel_order`: `cancelOrder` on the cached `Trade`; unknown or foreign-client order raises
  `BrokerError`.
- `modify_order`: re-`placeOrder` with the same `orderId` and new prices (IBKR modifies in
  place). Returns the same `Order` object with updated prices and records a `MODIFIED`
  event; `tracker.mark_replaced` is not called (there is no replacement order). The plan
  must check that `Strategy.modify_order` callers accept a returned order identical to the
  input.
- `pull_positions`, `pull_orders`, `pull_order`: read `ib_async`'s synced cache (open orders
  plus orders completed in this session; IBKR exposes no deeper order history via the API).
  `pull_order` returns `None` for an unknown identifier.
- `close_position(asset, fraction)`: market sell of `floor(position * fraction)` shares;
  `None` when there is no position.
- `close_all_positions(cancel_orders=True)`: `reqGlobalCancel` if requested, then one
  market sell per long position.
- `sync_open_orders`: `reqAllOpenOrders`, keep those whose `orderRef` starts with
  `{strategy_name}:`, adopt untracked ones. Adopted orders from another client id are tracked,
  but `cancel_order`/`modify_order` on them raise
  `BrokerError("order placed by client id N")`.
- `get_account`: `accountSummary` -> `AccountBalances` via `account.py`.
- `configure_account()`: checks only (IBKR cannot set margin/shorting via API):
  - account id starts with `DU` iff `is_paper` -- mismatch raises `ConfigurationError`;
  - account type is not a cash account: raises `ConfigurationError` when live, logs a warning
    when paper;
  - base currency is not `USD`: raises `ConfigurationError`.
- Market data (`get_last_price(s)`, `get_quote`, `get_bars`): delegated to
  `AlpacaMarketData` built from `for_data()`.
- `clock`: `AlpacaMarketClock` over the data credentials' calendar client.
- `news_provider()`: `AlpacaNewsProvider` built lazily from the `news` credentials callable.

### 4.5 Order events (`events.py`)

- Handlers registered on `orderStatusEvent`, `execDetailsEvent` and `errorEvent` by
  `start_stream()`; `stop_stream()` unregisters them and disconnects.
- On the loop thread they only find the order (`tracker.get_tracked_order_by_client_order_id`
  via `orderRef`) and call `tracker.process_trade_event(...)`: `NEW`, `PARTIALLY_FILLED` /
  `FILLED` (with execution price and cumulative filled quantity), `CANCELED`, `ERROR`.
  Never strategy hooks.
- Each execution is keyed by `execId`; an already-applied `execId` is ignored, so replays
  after a reconnect never double-count a fill.
- Informational codes (2104, 2106, 2158 and other "farm connection OK" notices) are logged
  at DEBUG and never mark an order as errored.

### 4.6 Lifecycle and reconnects

- `IbkrBroker.from_settings(...)`: start loop thread -> connect -> `configure_account()` ->
  build `AlpacaMarketData` and `AlpacaMarketClock`. Any failure disconnects and raises
  `BrokerError`/`ConfigurationError`; the strategy does not start.
- IB Gateway restarts daily and sockets can drop. On the next call after a disconnect, the
  connection reconnects and the broker reconciles: `sync_open_orders()` then `reqExecutions`
  for the session, applying only unseen `execId`s.
- A failed reconnect raises `BrokerError`; the executor logs the failed tick and continues
  (existing behaviour), so the bot recovers when the Gateway is back.

## 5. Other touched code

- `agents/tools/trading.py`: the comment/`except` that assume an Alpaca SDK exception is
  generalised (behaviour unchanged: any submit failure becomes `{"error": ...}`).
- `cross_momentum`'s `AlpacaApiRateLimiter` stays (market data is Alpaca under both brokers);
  comment clarified.
- `env/.env.example` rewritten into the groups of section 2, each saying when it is required.
- `README.md`: env contract; IB Gateway setup (API socket port, trusted IP, read-only API off,
  one `IBKR_CLIENT_ID` per strategy, IBC for unattended restarts).
- `CLAUDE.md`: commands for the new smoke scripts; architecture entries for
  `brokers/factory.py`, `brokers/alpaca/data.py`, `brokers/ibkr/*`, `backtesting/placeholder.py`;
  gotchas (credential groups never fall back; IBKR data/news come from Alpaca; client id
  ownership; `ib_async` confined to `brokers/ibkr/`).

## 6. Testing

Automated (no network, hand-written fakes, no `MagicMock`):

- `config/env.py`: defaults, missing variables per group, `ALPACA_IS_PAPER` rejection,
  unknown `BROKER`, IBKR port default by paper flag, invalid integers.
- `build_broker`: dispatch per `BROKER` with injected fake builders; news credentials not
  read until `news_provider()`.
- `PlaceholderBroker`: every operation raises `BrokerError`; `main._run_strategy` in
  backtesting mode builds no network client.
- `AlpacaMarketData`: the existing broker market-data tests, repointed.
- `ibkr/orders.py`, `ibkr/account.py`: pure tests on real `ib_async` objects (`Contract`,
  `Order`, `Trade`, `OrderStatus`, `Fill`, `AccountValue`).
- `FakeIB` in `tests/fakes.py`: records `placeOrder`/`cancelOrder`/`reqGlobalCancel`, serves
  positions/trades/summary, and fires status, execution, error and disconnect events.
- `IbkrBroker` via `FakeIB`: submit, cancel, modify, rejection (`set_error` before raise),
  notional rejection, fractional flooring, `sync_open_orders` adoption, foreign client id,
  reconnect then reconcile without double fills, `configure_account` checks.
- `IbkrConnection`: real event loop with fake coroutines -- result, timeout, exception
  wrapping, shutdown.

Manual (`scripts/tests/`, against a paper IB Gateway):

- `smoke_ibkr_account.py`: read-only -- connect, account checks, balances, positions.
- `smoke_ibkr_orders.py`: refuses unless the account is `DU…`; far-from-market limit order ->
  modify -> cancel; then 1-share market buy and sell; prints tracker events.

## 7. Implementation order

1. Sub-project A: env settings and credentials, `AlpacaMarketData` extraction, call-site
   migration, `PlaceholderBroker`, factory, `main.py`, docs for the env contract.
2. Sub-project B, bottom-up: `ibkr/orders.py`, `ibkr/account.py`, `IbkrConnection`,
   `events.py`, `IbkrBroker`, factory wiring, smoke scripts, docs.

One spec and one plan: B depends on A's `AlpacaMarketData` and factory.

## 8. Known limits and risks

- An IBKR bot needs IB Gateway running and logged in; unattended operation needs IBC (outside
  this repo). A Gateway restart costs at most one failed tick.
- Orders are sized on Alpaca IEX prices while filling on IBKR's consolidated market; small
  price gaps are expected.
- IBKR commissions are real; backtests still default `commission=0` -- strategies targeting
  IBKR should set a commission parameter.
- `pull_orders` sees only the current session's completed orders.
- Whole shares only on IBKR.

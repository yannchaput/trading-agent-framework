# Tastytrade broker -- design spec

Date: 2026-10-15. Status: approved in brainstorming, pending spec review.

## 1. Goal and scope

Add `tastytrade` as a third paper/live broker next to `alpaca` and `ibkr` (`BROKER=tastytrade`),
over Tastytrade's REST API and account streamer (https://developer.tastytrade.com/docs/).

In scope:

- `TastytradeBroker` implementing the full `Broker` interface: orders, positions, account, order
  events. Market data, the market calendar and news come from Alpaca (`ALPACA_DATA_*`,
  `ALPACA_NEWS_*`), exactly as for `IbkrBroker`.
- OAuth2 authentication: a refresh token exchanged for 15-minute access tokens, all settings in the
  env file under `TASTYTRADE_*`.
- A backtest fee schedule for `BROKER=tastytrade`. Backtests never touch Tastytrade: they run on
  `BacktestBroker` with Yahoo/Alpaca bars as today; `BROKER` only picks the fees.
- Refusing to run a strategy against the Tastytrade sandbox.
- Manual sandbox scripts: a one-off dry-run probe, an account smoke test, an order smoke test.

Out of scope (known limits, section 9): trailing stops, a Tastytrade dashboard tab, Tastytrade
market data, client-side trailing-stop emulation.

### 1.1 Facts from the Tastytrade docs that shape the design

- **Auth**: `POST /oauth/token` with `grant_type=refresh_token`, `refresh_token`, `client_secret`
  (`client_id` optional, inferred from the token). Access tokens are JWTs valid 15 minutes; refresh
  tokens never expire and are not rotated. Every request needs `Authorization: Bearer <token>` and a
  `User-Agent: product/version` header (requests without one are rejected).
- **Hosts**: production `https://api.tastyworks.com` / `wss://streamer.tastyworks.com`; sandbox
  `https://api.cert.tastyworks.com` / `wss://streamer.cert.tastyworks.com`. Sandbox and production
  credentials are separate OAuth applications and never work against each other.
- **Sandbox**: no market data (`/market-data` returns 502). Simulated fills ignore the market: market
  orders fill at $1, limit orders under $3 fill immediately, limit orders at $3 or more stay `Live`
  forever. Trades, transactions and positions are cleared every 24 hours.
- **No idempotency**: no idempotency key; order submission is not deduplicated. Orders carry a
  client-chosen `external-identifier`, echoed back. After an uncertain submission, look the
  identifier up before resubmitting.
- **Order types**: `Limit`, `Market`, `Marketable Limit`, `Stop`, `Stop Limit`, `Notional Market`.
  **No trailing stop.** `Market` cannot be `GTC`. Time-in-force: `Day`, `GTC`, `GTD`, `Ext`,
  `GTC Ext`, `IOC` (+ overnight variants).
- **Equities** carry `is-fractional-quantity-eligible` (`GET /instruments/equities/{symbol}`).
- **Statuses**: `Received`, `Routed`, `In Flight`, `Contingent`, `Live`, `Cancel Requested`,
  `Replace Requested`; terminal `Filled`, `Cancelled`, `Rejected`, `Expired`, `Removed`,
  `Partially Removed`. A replace is cancel-and-replace: the original goes `Replace Requested` then
  `Cancelled`; the replacement waits `Contingent` until then.
- **Account streamer**: send `connect` (account numbers + `auth-token`), then `heartbeat` every
  2 s - 1 min. Notifications carry the full object (`type: "Order"`, dasherized keys), not a diff.
- **Account**: `GET /customers/me/accounts` lists accounts with `account-number`, `margin-or-cash`,
  `is-closed`.

## 2. Env contract

| Variable | Required | Meaning |
|---|---|---|
| `BROKER=tastytrade` | yes | selects the broker |
| `BROKER_API_IS_PAPER` | no | `true` (default, fail-safe) = sandbox hosts, `false` = production hosts |
| `TASTYTRADE_CLIENT_SECRET` | yes | OAuth application's client secret |
| `TASTYTRADE_REFRESH_TOKEN` | yes | the grant's refresh token |
| `TASTYTRADE_ACCOUNT_NUMBER` | yes | the account to trade; startup fails if the login doesn't own it |
| `ALPACA_DATA_API_*`, `ALPACA_DATA_IS_PAPER` | yes | prices, bars, calendar (as for IBKR) |
| `ALPACA_NEWS_API_*` | for news tools | news (as for IBKR) |

No `client_id` (inferred). `User-Agent` is a constant `trading-agent-framework/<package version>`.

The manual sandbox scripts read `TASTYTRADE_*` from `env/.env.alpaca.integration-tests` (the shared
integration-tests file; the Alpaca smoke scripts don't read `BROKER`, so they are unaffected).

## 3. Configuration, wiring, refusal, fees

- `config/env.py`:
  - `BrokerKind.TASTYTRADE = "tastytrade"`.
  - `TastytradeSettings` (frozen, slots): `client_secret` and `refresh_token` (`repr=False`),
    `account_number`, `is_paper`; properties `api_url` and `streamer_url` derived from `is_paper`.
    `from_env(is_paper, env=None)`: blank or missing required values raise `ConfigurationError`
    naming the variable.
- `brokers/factory.py`: `_build_tastytrade` -> `TastytradeBroker.from_settings(strategy_name,
  TastytradeSettings.from_env(settings.is_paper, env), data=AlpacaCredentials.for_data(env),
  news=lambda: AlpacaCredentials.for_news(env))`, importing the class inside the builder.
- `brokers/__init__.py`: lazy export `TastytradeBroker`; `import trading_agent_framework.brokers`
  must not import `websockets` or `httpx`-based Tastytrade modules.
- **Sandbox refusal** in `main.py` (`_run_strategy`), for `paper` and `live` runs only: when
  `BrokerSettings.from_env()` gives `kind is TASTYTRADE` and `is_paper`, raise `ConfigurationError`:
  "the Tastytrade sandbox fills every market order at $1 and resets daily; forward-test on Alpaca
  paper (BROKER=alpaca) and use BROKER=tastytrade with BROKER_API_IS_PAPER=false for live trading
  only". The check sits before `build_broker`. `build_broker` itself still builds sandbox brokers
  (the smoke scripts need them).
- `brokers/fees.py`: `COMMISSIONS[BrokerKind.TASTYTRADE] = _tastytrade_commission` returning
  `Decimal(0)` ($0 stock/ETF commission). Regulatory fees (SEC, FINRA TAF, CAT) apply as for every
  broker. Module docstring lists the new broker.
- Dashboard unchanged: Live/Paper tabs keep listing Alpaca strategies only.

## 4. Package `brokers/tastytrade/`

| Module | Role | I/O |
|---|---|---|
| `client.py` | `TastytradeSession`: HTTP, auth, retries, pagination | yes (only HTTP code) |
| `account.py` | balance parsing, startup account check | pure |
| `orders.py` | conform rules, order JSON, status map, order/position parsing, identifier helpers | pure |
| `events.py` | `TastytradeOrderEvents`: order snapshot -> `OrderTracker` | no network |
| `stream.py` | `TastytradeAccountStream`: account-streamer thread | yes (only websocket code) |
| `broker.py` | `TastytradeBroker`: wiring; Alpaca for data, clock, news | via the above |

Pure modules take and return plain JSON dicts (dasherized keys) and our entities; money is parsed
with `Decimal(str)` straight from the API's strings. No float boundary is added: Tastytrade accepts
prices and quantities as strings.

`websockets` becomes a direct dependency in `pyproject.toml` (already installed, 16.1, through
alpaca-py). `httpx` is already direct.

### 4.1 `TastytradeSession` (`client.py`)

- Wraps one `httpx.Client(base_url=settings.api_url, timeout=10s, headers={"User-Agent": ...})`.
- `access_token() -> str`: under a `threading.Lock`, returns the cached token or POSTs
  `/oauth/token` when there is none or it expires within 60 s (expiry from `expires_in`, measured
  with `time.monotonic()`). Shared by REST calls and the streamer thread.
- `get / post / put / delete(path, *, json=None, params=None) -> dict`: adds the bearer token,
  returns the response's `data` object (empty dict for 204).
  - 401: force one token refresh and retry once (any method).
  - GET only: retry on 429, 5xx and `httpx.TransportError`, up to 3 attempts, exponential backoff
    with jitter, honouring `Retry-After`.
  - POST/PUT/DELETE are never retried on 429/5xx/transport errors (no idempotency).
  - An error response raises `TastytradeHTTPError(status_code, "<method> <path>: <status>
    <error.code>: <error.message>")` (a `BrokerError` defined in `client.py`; the broker reads
    `status_code`, e.g. 404 on an unknown symbol). A transport failure, or a 5xx on a write, raises
    `BrokerUncertainError` (a `BrokerError` subclass in `utils/errors.py`) so the broker can tell
    "rejected" from "outcome unknown". Raw `httpx` exceptions never escape.
- `get_all(path, *, params=None) -> list[dict]`: follows `pagination.page-offset` /
  `total-pages`, concatenating `data.items`.
- `close()`.

### 4.2 Account (`account.py`, pure)

- `parse_balances(data) -> AccountBalances`: `cash` <- `cash-balance`, `portfolio_value` <-
  `net-liquidating-value`, `buying_power` <- `equity-buying-power`. Missing or non-numeric ->
  `BrokerError`.
- `check_account(accounts, account_number, *, is_paper) -> list[str]` (warnings), over the
  `items` of `GET /customers/me/accounts`:
  - number not owned -> `ConfigurationError` listing the owned numbers;
  - `is-closed` -> `ConfigurationError`;
  - `margin-or-cash == "Margin"` -> `ConfigurationError` when `not is_paper` ("this framework
    expects a cash account: no margin, no shorting"); a returned warning in the sandbox.

`TastytradeBroker.from_settings` runs the check; any failure closes the session before re-raising.

## 5. Orders

### 5.1 Translation (`orders.py`)

| Ours | Tastytrade |
|---|---|
| `BUY` / `SELL` | leg `action` `Buy to Open` / `Sell to Close` (long-only; a sell with nothing held is rejected by Tastytrade, never opens a short) |
| `MARKET` | `Market` |
| `MARKET` + `notional` | `Notional Market`, `value` (+ `value-effect`), no leg `quantity` |
| `LIMIT` | `Limit`, `price` = `limit_price` |
| `STOP` | `Stop`, `stop-trigger` = `stop_price` |
| `STOP_LIMIT` | `Stop Limit`, `stop-trigger` = `stop_price`, `price` = `stop_limit_price` |
| `TRAIL` | `OrderValidationError` ("Tastytrade has no trailing stops") |
| price effect | `Debit` for buys, `Credit` for sells |
| `DAY` / `GTC` / `IOC` | `Day` / `GTC` / `IOC`; with `extended_hours=True`: `Ext` / `GTC Ext` (`IOC` + extended -> rejected) |
| `OPG`, `CLS`, `FOK`; `MARKET` + `GTC` | `OrderValidationError` |
| prices | strings, `ROUND_HALF_UP` to $0.01 (to $0.0001 below $1) |
| our id | `external-identifier = "{strategy}:{identifier}"`; `source = "trading-agent-framework"` |
| leg | `instrument-type: Equity`, `symbol`, `quantity` as a string |
| symbol | class shares `BRK.B` <-> `BRK/B` (Tastytrade's equity symbology; the probe confirms) |

### 5.2 Conform rules

`conform_order(order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order`, pure,
called from `_conform_order`:

| Order | Eligible stock | Not eligible |
|---|---|---|
| whole quantity | unchanged | unchanged |
| fractional quantity, `MARKET` | decimal quantity sent | floored to whole shares + warning; < 1 share -> `OrderValidationError` |
| fractional quantity, any other type | `OrderValidationError` (Alpaca parity: a floored stop would leave the remainder unprotected) | `OrderValidationError` |
| `notional`, `MARKET` | `Notional Market` | `quantity = floor(notional / last_price)` + warning; < 1 share or no `last_price` -> `OrderValidationError` |
| `notional`, any other type | `OrderValidationError` | `OrderValidationError` |

The broker supplies `fractional_eligible` from `GET /instruments/equities/{symbol}`, cached per
symbol for the broker's life (404 -> `OrderValidationError("Tastytrade does not know ...")`), and
fetches `last_price` (Alpaca) only for a non-eligible notional order. Task 1's probe may tighten
these rules (decimal sells, notional sells, minimum notional); the plan adjusts `orders.py` to its
findings before writing it.

### 5.3 Status map

| Tastytrade | Ours |
|---|---|
| `Received`, `Routed`, `In Flight`, `Contingent` | `SUBMITTED` |
| `Live`, `Replace Requested` | `NEW`, or `PARTIAL_FILL` when the leg has fills |
| `Cancel Requested` | `CANCELLING` |
| `Filled` | `FILL` |
| `Cancelled`, `Removed`, `Partially Removed` | `CANCELED` |
| `Expired` | `EXPIRED` |
| `Rejected` | `ERROR` (message from `reject-reason`) |
| anything else | `UNKNOWN` + warning |

`parse_order(data, strategy_name) -> Order` builds an `Order` from a snapshot (identifier from
`external-identifier` when it carries this strategy's prefix, otherwise the Tastytrade id),
`parse_position(data) -> Position | None` (`Equity` only; `quantity` Decimal, side from
`quantity-direction`, `avg_fill_price` <- `average-open-price`), and `identifier_from_external(ext,
strategy_name)` mirror the IBKR helpers.

### 5.4 Broker operations (`broker.py`)

The broker keeps `_tt_ids: dict[str, str]` (our identifier -> current Tastytrade order id) under a
lock, shared with `events.py`.

- **`_submit_order`**: set `client_order_id` (= `external-identifier`) if missing; build the JSON;
  `tracker.track_unprocessed(order)`; `POST /accounts/{a}/orders`.
  - Success: the response is `{"order": {...}, "buying-power-effect": ..., ...}`; record
    `order.id`, `order.update_raw(...)`, status `SUBMITTED` if still `UNPROCESSED`, then pass the
    returned snapshot through `events.apply` (a sandbox $2 limit is already `Filled` in it; fill ids
    de-duplicate the stream's copy). From here on `TastytradeOrderEvents` owns the status.
  - `BrokerError` (4xx, e.g. 422 buying power): `order.set_error(msg)` **before** re-raising
    (lumibot contract), untrack.
  - `BrokerUncertainError` (timeout, transport, 5xx): search `GET /orders/live`, then today's
    `GET /orders`, for the `external-identifier`. Found -> continue as success. Not found ->
    `set_error`, untrack, raise `BrokerError("submission outcome unknown; not found at Tastytrade;
    not resent")`. Never resubmits.
- **`cancel_order`**: `DELETE /accounts/{a}/orders/{tt_id}`; unknown id -> `BrokerError`.
- **`modify_order`** (limit/stop/stop-limit only; market and notional -> `OrderValidationError`):
  `PUT /accounts/{a}/orders/{tt_id}` with the order JSON **without `legs`** (legs can't change on a
  replace; the community SDK sends it that way), same `external-identifier`, new prices. Before the
  PUT the current id is marked superseded (`events.begin_replace`), so a `Cancelled` for it that
  arrives before the PUT returns is ignored; a certain failure restores it (`events.abort_replace`)
  and the old prices. An uncertain failure looks for a live order with `replaces-order-id == tt_id`
  and the same `external-identifier`; found -> success. Returns **the same `Order`** (as IBKR) with
  prices updated, re-points the id map to the replacement's id (the PUT response is the new order
  itself), fires `OrderEvent.MODIFIED`.
- **`pull_order(identifier)`**: today's and live orders, matched on `external-identifier`.
- **`pull_orders(limit)`**: `GET /orders` newest first, paginated up to `limit`, parsed.
- **`pull_positions`**: `GET /accounts/{a}/positions`, parsed, non-equities dropped.
- **`get_account`**: `GET /accounts/{a}/balances` -> `parse_balances`.
- **`close_position(asset, fraction)`**: market sell of `position.quantity * fraction` through
  `submit_order` (conform handles the fraction); returns `None` with a warning when a non-eligible
  stock rounds below one share.
- **`close_all_positions(cancel_orders)`**: no global cancel endpoint, so cancel every live order
  of the account (same scope as Alpaca and IBKR), then close each position.
- **`sync_open_orders`**: adopt live orders whose `external-identifier` has this strategy's prefix
  and that aren't tracked; register their ids.
- **`reconcile`**: `sync_open_orders()`, then `GET /orders/{tt_id}` for each tracked active order
  through `events.apply` (catches fills/cancels missed while disconnected).
- Market data, clock, news: delegated to `AlpacaMarketData`, `AlpacaMarketClock`,
  `lazy_news_provider`, exactly as `IbkrBroker`.

## 6. Order events and stream

### 6.1 `TastytradeOrderEvents` (`events.py`)

`apply(data: dict) -> bool` takes one order snapshot (streamer or REST):

- Finds our `Order` by Tastytrade id via `_tt_ids`; falls back to `external-identifier` for orders
  adopted after a restart. Unknown -> ignored (another strategy's or a manual order).
- **Fills**: each `legs[0].fills[*]` with an unseen `fill-id` (per-order seen-set) ->
  `process_trade_event(order, PARTIALLY_FILLED | FILLED, price=fill-price,
  filled_quantity=quantity)`; `FILLED` when the order's cumulative fills reach its quantity.
  A notional order has no quantity: its fills are held back until the snapshot says `Filled`, then
  applied with the last one as `FILLED`. Replaying a snapshot is harmless.
- **Status**: a change to `Live` -> `NEW` (from `UNPROCESSED`/`SUBMITTED` only);
  `Cancelled`/`Removed`/`Partially Removed`/`Expired` -> `CANCELED` event (Alpaca maps `expired`
  the same way); `Rejected` -> `ERROR` with `reject-reason`; `Cancel Requested` -> status
  `CANCELLING` (no tracker event).
- **Superseded ids** (replaced by `modify_order`): their fills still count (de-duplicated by
  `fill-id`); their status changes are ignored, so the original's `Cancelled` never cancels the
  modified order.
- Feeds `OrderTracker` only; never calls strategy hooks (threading rule).

### 6.2 `TastytradeAccountStream` (`stream.py`)

- A daemon thread using `websockets.sync.client.connect(settings.streamer_url)`.
- On open: `{"action": "connect", "value": [account_number], "auth-token": session.access_token()}`;
  after `status: ok`, a `heartbeat` (same `auth-token` field, fresh token) every 20 s.
- `type == "Order"` -> `events.apply(data)`; other types ignored (balances and positions are read
  on demand).
- Disconnect, error or an auth failure -> reconnect with backoff 1, 2, 4 ... capped at 30 s, then
  `broker.reconcile()`. Token expiry on an open socket is undocumented: treating a server close as a
  disconnect covers it (fresh token on reconnect); the live smoke run shows which happens.
- `start_stream()` starts it; `stop_stream(timeout)` closes the socket, joins the thread, closes the
  session. Exceptions in the thread are logged, never raised into strategy code.

## 7. Testing

Automated (`tests/brokers/tastytrade/`, no network, hand-written fakes, no `MagicMock`):

- `test_orders.py`: translation table, conform table, TIF/price rules, status map, parsing, on JSON
  fixtures copied from the docs.
- `test_account.py`: balances parsing, account check (not owned, closed, margin live vs sandbox).
- `test_client.py` with `httpx.MockTransport`: token cache and pre-expiry refresh, 401 retry, GET
  backoff on 429/5xx with `Retry-After`, POST never retried, error wrapping, `BrokerUncertainError`,
  pagination.
- `test_events.py`: snapshot sequences -- duplicate fills, partial then full, replaced id's
  `Cancelled` ignored, rejection message, unknown order ignored.
- `test_stream.py`: message handling and reconnect logic with a fake socket (no thread timing
  sleeps: the loop takes injectable connect/sleep functions).
- `test_broker.py` with `FakeTastytradeSession` in `tests/fakes.py`: submit success, 4xx,
  uncertain-found, uncertain-not-found; modify re-pointing; cancel; `close_position` fractions;
  `sync_open_orders`; `reconcile`.
- Existing suites extended: `test_factory.py` (`BROKER=tastytrade`), `test_fees.py`,
  `config` tests (`TastytradeSettings`), a `main.py` refusal test, `test_lazy_imports.py`.

Manual (`scripts/tests/`, sandbox only, `SKIP:` and exit 0 without `TASTYTRADE_*` in
`env/.env.alpaca.integration-tests`; always `TastytradeSettings.from_env(is_paper=True)`, whatever
`BROKER_API_IS_PAPER` says):

- `probe_tastytrade_orders.py` (Task 1, run once by the user; named `probe_*` so
  `run_smoke_tests.sh` doesn't run it): mostly dry-runs. Reports the account record, fractional
  eligibility, class-share symbology (`BRK.B` vs `BRK/B`), quantity decimal precisions, whether a
  decimal `quantity` is accepted on `Market` buys, the minimum notional, the `external-identifier`
  length limit, and the `fee-calculation` of a stock order. Two questions need real sandbox orders,
  all cleaned up: decimal and notional *sells* need a fractional position (a 0.5-share market buy,
  filled at $1, sold back), and the replace semantics need a resting order (a $5 limit buy,
  replaced at $6, then cancelled).
- `smoke_tastytrade_account.py` (read-only): auth, account check, balances, positions.
- `smoke_tastytrade_orders.py`, built on the sandbox fill rules, stream running throughout:
  1. $5 limit buy (rests) -> stream reports `NEW`;
  2. modify to $6 -> `MODIFIED`; the original id's `Cancelled` does not cancel the order;
  3. cancel -> `CANCELED`;
  4. $2 limit buy (fills at once) -> `FILL`, position appears;
  5. market sell to close (fills at $1) -> `FILL`, position gone.
  PASS/FAIL per step.
- `scripts/tests/HOWTO.md`: the three scripts, the daily sandbox reset, how to obtain sandbox
  credentials (sandbox account tool -> OAuth application -> grant -> refresh token).

## 8. Documentation

- `brokers/CLAUDE.md`: architecture lines for `brokers/tastytrade/`; gotchas: no trailing stops;
  events keyed on Tastytrade id (superseded ids' statuses ignored); order writes never retried,
  uncertain submissions looked up by `external-identifier`; sandbox fake fills and the refusal;
  `websockets`/`httpx` stay inside the package.
- Root `CLAUDE.md`: the new manual scripts in Commands; `BROKER=tastytrade` in the credential notes.
- `README.md`: the `TASTYTRADE_*` env contract.
- `TODO.md`: dashboard has no Tastytrade tab.

## 9. Known limits

- **No trailing stops**: `OrderType.TRAIL` is rejected, so `earnings_drift` cannot trade on
  Tastytrade. Client-side emulation would be its own project.
- **No forward-testing on Tastytrade**: the sandbox's fills are synthetic, so strategies never run
  against it; forward-test on Alpaca paper, trade live on Tastytrade.
- **Market data is Alpaca IEX** in live trading, as for IBKR.
- **No dashboard tab** for Tastytrade accounts.
- **No idempotency**: an uncertain submission not found at Tastytrade is reported as an error, not
  resent; the strategy decides on its next tick.
- **Fractional positions can't be protected by stops**: fractional quantities are market-only.

## 10. Implementation order

Each task is followed by a review-fix commit.

1. Sandbox probe script; the user runs it; findings recorded in this spec (section 11).
2. Config (`BrokerKind`, `TastytradeSettings`), `BrokerUncertainError`, fees, `main.py` refusal.
3. `account.py` (pure).
4. `orders.py` (pure), part 1: conform rules and order JSON, adjusted to the probe's findings.
5. `orders.py` (pure), part 2: status map and order/position/fill parsing.
6. `client.py` (`TastytradeSession`, `TastytradeHTTPError`).
7. `events.py`.
8. `stream.py`.
9. `broker.py`, factory wiring, lazy import.
10. Smoke scripts, `HOWTO.md`, docs.

## 11. Probe findings

Recorded by Task 1 from the probe's output.

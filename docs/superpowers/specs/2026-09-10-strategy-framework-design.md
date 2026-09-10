# Strategy framework — design

- **Date:** 2026-09-10
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md`

## 1. Context

The broker layer is complete: entities, the `Broker` ABC, `OrderTracker`, `AlpacaBroker` and
`AlpacaTradeStream`. This spec covers the next layer, the **strategy framework**. It is a lean copy of
lumibot's `Strategy` template:

- `initialize`, and `on_trading_iteration` fired at a custom `sleeptime`;
- the market lifecycle hooks;
- the order-event hooks;
- an API facade for accounting and trading.

The entry points `run_strategy`, `run_paper_trading` and `run_live_trading` come from the user's
`WrappingStrategy` (`lumibot-trading-agent/src/lumibot_trading_agent/support/wrapping_strategy.py`), not
from lumibot itself.

Why a rewrite rather than a port: lumibot's `Strategy` and `StrategyExecutor` together run to about
14,000 lines. They pull in APScheduler, cloud and Discord reporting, options, and dividend handling. The
existing strategies use only a small slice of that. Section 3 lists what is kept.

## 2. Scope

**In scope**
- Live and paper trading, driven by a single-threaded executor.
- The lifecycle and order-event hooks, and the accounting and trading facade.
- Coloured logging to lumibot-style run log files.
- The broker additions this layer needs: account balances, the market calendar, and replace/close orders.

**Out of scope (separate subprojects)**
- **Backtesting.** This spec only provides the `MarketClock` seam that a simulated clock will plug into;
  `run_backtesting` raises `NotImplementedError`.
- **Market data:** `get_last_price(s)`, `get_historical_prices(_for_assets)`, `get_quote`, `get_bars`.
- **Agents and memory:** `self.agents`, variable backup.

## 3. Method inventory (lumibot `Strategy` → this framework)

| Status | Methods |
|---|---|
| ✅ Lifecycle hooks (no-op defaults) | `initialize(**params)`, `on_trading_iteration()`, `before_market_opens()`, `before_starting_trading()`, `before_market_closes()`, `after_market_closes()`, `on_strategy_end()`, `on_bot_crash(error)` (default calls `on_abrupt_closing()`, as lumibot does), `on_abrupt_closing()` |
| ✅ Order-event hooks | `on_new_order(order)`, `on_canceled_order(order)`, `on_partially_filled_order(position, order, price, quantity, multiplier)`, `on_filled_order(position, order, price, quantity, multiplier)` |
| ✅ Config attributes | `name`, `sleeptime = "1M"`, `minutes_before_opening = 60`, `minutes_before_closing = 1`, `minutes_after_closing = 0`, `parameters`, `vars`, `first_iteration`, `is_backtesting`, `trading_mode` |
| ✅ Accounting | `get_cash()`, `get_portfolio_value()`, `cash` / `portfolio_value` properties, `get_positions()`, `get_position(asset)`, `get_orders()`, `get_order(identifier)` |
| ✅ Trading | `create_order(...)`, `submit_order(s)`, `cancel_order(s)`, `cancel_open_orders()`, `modify_order(order, limit_price=None, stop_price=None)`, `sell_all(cancel_open_orders=True)`, `close_position(asset, fraction=1.0)`, `close_positions(assets=None)`, `wait_for_order_execution(order, timeout=None)`, `wait_for_orders_execution(orders, timeout=None)` |
| ✅ Time / logging / control | `get_datetime()`, `sleep(seconds)`, `log_debug/log_info/log_warning/log_error/log_critical(message)`, `stop()` |
| ✅ Runners | `run_strategy()` → `run_paper_trading()` / `run_live_trading()`; `run_backtesting()` raises `NotImplementedError` |
| ⏳ Data subproject | `get_last_price(s)`, `get_historical_prices(_for_assets)`, `get_quote`, `get_bars` |
| ⏳ Backtest subproject | `trace_stats`, `tearsheet_custom_metrics`, `add_line/add_marker/add_ohlc`, `deposit/withdraw/adjust_cash`, `initial_budget` |
| ⏳ Agents / memory subprojects | `self.agents`, `backup_variables_to_db` |
| ❌ Dropped | `log_message`, options/greeks/chains, dividends and splits, `register_cron_callback`, cloud/Discord/`notify`, `on_parameters_updated`/`update_parameters`, `on_closed_market_iteration`, `load_pandas`, `create_asset`, realtime bars, timezone helpers, `get_round_*`, `quote_asset`, `force_start_immediately` (redundant: the first iteration always runs as soon as a session starts, see §5.2) |

## 4. Architecture

```
src/trading_agent_framework/
  clock.py                  NEW  MarketSession + MarketClock ABC
  log.py                    NEW  setup_strategy_logging(), ColorLogger
  entities/account.py       NEW  AccountBalances
  config/env.py             EDIT TradingMode StrEnum; TRADING_MODES derived from it
  brokers/base.py           EDIT clock, is_paper, account/replace/close/sync methods, no-op stream hooks
  brokers/tracker.py        EDIT mark_replaced(old, new)
  brokers/alpaca/orders.py  EDIT replace/close request builders; AlpacaTradingClient Protocol grows
  brokers/alpaca/account.py NEW  pure: parse_account, build_calendar_request, parse_calendar
  brokers/alpaca/clock.py   NEW  AlpacaMarketClock
  brokers/alpaca/broker.py  EDIT implements the new Broker methods; from_credentials wires clock + is_paper
  strategies/__init__.py    NEW  exports Strategy, StrategyExecutor
  strategies/timing.py      NEW  pure: parse_sleeptime, next_tick
  strategies/events.py      NEW  OrderEventQueue
  strategies/executor.py    NEW  StrategyExecutor
  strategies/strategy.py    NEW  Strategy
```

**Layering rules**
- `strategies/`, `clock.py` and `log.py` never import `alpaca`.
- The existing rule "only `orders.py` may import `alpaca.trading.requests`" becomes "only `orders.py`
  and `account.py`". Both are pure translation modules: no I/O and no state. `CLAUDE.md` is updated to match.
- No new dependencies. Colours are hand-written ANSI codes, and time zones come from `zoneinfo`.

### 4.1 `clock.py`

```python
@dataclass(frozen=True, slots=True)
class MarketSession:
    open: datetime   # tz-aware
    close: datetime  # tz-aware

class MarketClock(ABC):
    tz: ZoneInfo                                    # market time zone (America/New_York for Alpaca)
    @abstractmethod
    def now(self) -> datetime: ...                  # tz-aware, in self.tz
    @abstractmethod
    def next_session(self) -> MarketSession: ...    # first session whose close > now()
    @abstractmethod
    def wait(self, seconds: float, wake: threading.Event) -> None: ...
```

- The live `wait` is `wake.wait(seconds)`. A future simulated clock advances its own time and returns
  immediately.
- `AlpacaMarketClock(client)` builds sessions from `TradingClient.get_calendar`. It caches about 10 trading
  days and refetches when the cache runs out.
- Alpaca returns calendar times as naive Eastern Time. `parse_calendar` localises them, and early closes
  come straight from the calendar data.

### 4.2 Broker additions

| `Broker` (ABC) | Alpaca implementation |
|---|---|
| `clock: MarketClock` | `AlpacaMarketClock` built in `from_credentials` |
| `is_paper: bool` | from `AlpacaCredentials.is_paper` |
| `get_account() -> AccountBalances` | `get_account()` → `account.parse_account` (cash, equity → `portfolio_value`, buying_power; all `Decimal`) |
| `modify_order(order, *, limit_price=None, stop_price=None) -> Order` | `replace_order_by_id` → the new order is parsed, then `tracker.mark_replaced(old, new)` |
| `close_position(asset, fraction=1.0) -> Order \| None` | `close_position(symbol, ClosePositionRequest(percentage=fraction·100))` → the order is tracked as unprocessed |
| `close_all_positions(cancel_orders=True) -> list[Order]` | `close_all_positions(cancel_orders=…)` → the returned orders are tracked |
| `sync_open_orders() -> list[Order]` | `pull_orders()` filtered on `client_order_id` starting with `"{strategy_name}:"` → tracked |
| `start_stream()` / `stop_stream()` | concrete no-ops in the ABC; `AlpacaBroker` keeps its existing overrides |

- `OrderTracker.mark_replaced(old, new)` moves `old` to the canceled bucket and tracks `new` as
  unprocessed. It fires no listener: Alpaca's own `replaced` event maps to `MODIFIED`, which the tracker
  already ignores.
- Every new client call wraps SDK exceptions in `BrokerError`.

### 4.3 `strategies/timing.py` (pure)

- `parse_sleeptime(value: int | str) -> SleepTime`:
  - an int means minutes;
  - a string is `<n><unit>`, where the unit is S, M/T, H or D (case-insensitive);
  - `SleepTime` carries either an `interval: timedelta` (S/M/H) or `sessions: int` (D);
  - anything invalid, or ≤ 0, raises `ConfigurationError`.
- `next_tick(anchor, interval, now) -> tuple[datetime, int]` returns the smallest `anchor + k·interval`
  that is strictly later than `now`, plus the number of ticks skipped.

### 4.4 `strategies/events.py`

`OrderEventQueue` is registered as an `OrderTracker` listener.
- The listener runs on the stream thread, inside the tracker lock. It captures `price` and `quantity` from
  `order.transactions[-1]` at that moment (for fill events), puts `(order, event, price, quantity)` on a
  `queue.SimpleQueue`, and sets a shared `threading.Event` called `wake`.
- `drain()` runs on the executor thread and yields the queued items.

## 5. Behaviour

### 5.1 `StrategyExecutor.run()`

1. **Setup**
   - Register the `OrderEventQueue` listener on `broker.tracker`.
   - Call `broker.sync_open_orders()`, then `broker.start_stream()`.
   - Install a SIGTERM handler that raises `KeyboardInterrupt` (only when running on the main thread; the
     previous handler is restored on exit).
2. **Initialize:** call `strategy.initialize(**params)`, passing only the entries of `strategy.parameters`
   that match its signature, as lumibot does. A crash here is fatal: log it with the traceback, call
   `on_bot_crash(e)`, and re-raise.
3. **Session loop.** While not stopped, run each `session = clock.next_session()` as described in §5.2.
4. **Exit**
   - `KeyboardInterrupt` calls `on_abrupt_closing()`.
   - Every exit after a successful `initialize` then calls `on_strategy_end()`.
   - A `finally` block stops the stream, removes the listener and restores the signal handler.

### 5.2 One session

```
pre_open = open − minutes_before_opening
stop_at  = close − minutes_before_closing
if now < pre_open:  wait_until(pre_open)
if now < open:      before_market_opens()           # skipped when the run starts mid-session
wait_until(open);   before_starting_trading()
anchor = max(open, now); tick = anchor
while now < stop_at and not stopped:
    wait_until(tick)
    if now >= stop_at: break
    run on_trading_iteration()                      # first_iteration is True only on the very first call
    sleep = parse_sleeptime(strategy.sleeptime)     # re-read every tick
    if sleep is sessions-based: break               # one iteration this session
    tick, skipped = next_tick(anchor, sleep.interval, now)
    if skipped: log one warning ("iteration overran; skipped N ticks")
before_market_closes()
wait_until(close + minutes_after_closing); after_market_closes()
```

- **`"ND"`:** iterations run in one session out of every N. The lifecycle hooks still run in every session.
- **`stop()`:** sets the stop flag and `wake`. The loop exits at the next check without running the
  remaining lifecycle hooks, then `on_strategy_end()` runs.

### 5.3 Waiting and event dispatch

`_wait_until(deadline)` repeats: drain and dispatch events, then
`clock.wait(min(remaining, 1 s), wake)` and clear `wake`. It stops at the deadline or when stopped.
`Strategy.sleep(seconds)` and `wait_for_order(s)_execution` (which also accepts a predicate and a timeout)
use the same loop. So order hooks run promptly while waiting, and never at the same time as other strategy
code.

Dispatch mapping:

| Event | Action |
|---|---|
| `NEW` | `on_new_order(order)` |
| `CANCELED` | `on_canceled_order(order)` |
| `PARTIALLY_FILLED` | `on_partially_filled_order(get_position(order.asset), order, price, quantity, 1)` |
| `FILLED` | `on_filled_order(get_position(order.asset), order, price, quantity, 1)` |
| `ERROR` | warning log |
| `MODIFIED` | ignored |

### 5.4 Errors

- An exception in `on_trading_iteration` or in an order hook is logged with its traceback, then
  `on_bot_crash(e)` is called and trading continues (lumibot's live behaviour).
- An exception raised inside `on_bot_crash` itself is logged and swallowed.
- Broker methods never let raw SDK exceptions escape (they raise `BrokerError`, per `errors.py`).
- The default `on_bot_crash` calls `on_abrupt_closing()`, as in lumibot. So if `on_abrupt_closing` is
  overridden to `sell_all`, a crash during an iteration will liquidate positions.

### 5.5 Runners

- `run_strategy()` dispatches on `trading_mode`.
- `run_paper_trading()` and `run_live_trading()`:
  1. call `setup_strategy_logging(name, mode)`;
  2. raise `ConfigurationError` on a mode/account mismatch (`run_live_trading` with `broker.is_paper`, or
     the reverse);
  3. log the startup banner: mode, log directory, parameters, time to open/close from `clock.next_session()`,
     cash, positions;
  4. run a `StrategyExecutor`.

## 6. Strategy class

```python
class Strategy:
    sleeptime: int | str = "1M"
    minutes_before_opening: int = 60
    minutes_before_closing: int = 1
    minutes_after_closing: int = 0
    parameters: Mapping[str, Any] = {}          # class defaults, overlaid by the ctor `parameters`

    def __init__(self, broker: Broker, *, name: str | None = None,
                 mode: TradingMode = TradingMode.PAPER,
                 parameters: Mapping[str, Any] | None = None,
                 clock: MarketClock | None = None) -> None: ...
```

- `name` defaults to the class name, and `clock` to `broker.clock`.
- `vars` is a `types.SimpleNamespace`.
- `get_datetime()` returns `clock.now()`.
- `create_order(asset: str | Asset, quantity, side, *, limit_price=None, stop_price=None, time_in_force=TimeInForce.DAY)`
  builds an `entities.Order` for this strategy. A plain symbol string becomes a stock `Asset`.
- Accounting and trading methods are thin delegates to `broker` and `broker.tracker`:
  - `get_orders()` returns the tracked orders;
  - `get_order(id)` checks the tracker first, then falls back to `broker.pull_order`;
  - `cancel_open_orders()` cancels `tracker.get_active_orders()`;
  - `sell_all()` calls `close_all_positions`.
- `wait_for_order(s)_execution` returns once each order is filled, canceled, or in error, or when the
  timeout expires. It needs the trade stream, which the runners always start.
- Hook docstrings stay to one line each, to keep the source small.

## 7. Logging (`log.py`)

- `setup_strategy_logging(strategy_name, mode, *, project_root=None, level=logging.INFO) -> Path`
  - Creates `logs/<strategy>/<mode>/<YYYY-mm-dd_HHMMSS>_<mode>/<mode>.log`, named `live.log`, `paper.log`
    or `backtest.log`. This mirrors `build_logs` together with lumibot's `Trader(logfile=…)`.
  - The project root comes from `config.env.find_project_root`.
  - Attaches a `FileHandler(mode="w", encoding="utf-8")` and a stdout `StreamHandler` to the
    `trading_agent_framework` logger, with `propagate = False`.
  - Uses lumibot's line format: `%(asctime)s | %(levelname)s | %(message)s` for DEBUG/INFO, and
    `%(asctime)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s` for WARNING and above.
  - Is idempotent (it replaces its own handlers on a second call), quiets `urllib3` and `websockets` to
    WARNING, and returns the log file path.
- `ColorLogger(logger, prefix)` provides `log_debug` (grey), `log_info` (blue), `log_warning` (yellow),
  `log_error` and `log_critical` (red).
  - Each method logs `[<prefix>] <ANSI colour>message<ANSI reset>` and returns the plain message.
  - It does not `print()` when a level is disabled; the record is simply dropped.
- **ANSI codes are written to the log files on purpose.** The user reads them with VS Code's "ANSI Colors"
  plugin, and lumibot's `paper.log` has the same codes. `termcolor` is not used because it turns colour off
  when stdout is not a TTY (`nohup`, systemd, cron), which would silently strip colour from the log files
  of background runs.
- `Strategy.log_*` are one-line delegates to the strategy's `ColorLogger`.

## 8. Other decisions

1. A mode/account mismatch between live and paper is a hard error; lumibot only logs it.
2. On startup, open orders are adopted by `client_order_id` prefix, so a restarted strategy can still cancel
   or modify them.
3. `get_cash()` and `get_portfolio_value()` call the Alpaca API on every call, with no cache. Revisit if
   rate limits (200 requests/min) become a problem.
4. `TradingMode` is a `StrEnum` in `config/env.py` (`LIVE = "live"`, `PAPER = "paper"`,
   `BACKTESTING = "backtesting"`). `TRADING_MODES` is derived from it, so existing callers keep working.

## 9. Testing

TDD throughout. Tests never touch the network, and they use hand-written fakes that follow the conventions
of `tests/fakes.py`.

**Fakes (`tests/fakes.py`)**
- `FakeClock`: manual time; `wait` advances time and returns; scripted sessions, including an early close.
- `FakeBroker`: an in-memory `Broker` subclass that records calls.
- Additions to the fake Alpaca client for `get_account`, `get_calendar`, `replace_order_by_id`,
  `close_position` and `close_all_positions`.

**Test files**
- `tests/strategies/test_timing.py`: the sleeptime parsing table (int, S/M/T/H/D, lowercase, invalid,
  ≤ 0) and `next_tick` maths, including skipped-tick counts.
- `tests/strategies/test_executor.py`:
  - hook order across two sessions, and a mid-session start that skips `before_market_opens`;
  - tick timestamps for 5M, 2H, 1D and 2D, and `sleeptime` changed from inside `on_trading_iteration`;
  - an iteration crash calls `on_bot_crash` and trading continues; an `initialize` crash propagates;
  - `stop()` leads to `on_strategy_end`; `KeyboardInterrupt` leads to `on_abrupt_closing` then `on_strategy_end`.
- `tests/strategies/test_events.py`:
  - an event posted from another thread is dispatched on the executor thread with
    `(position, order, price, quantity, 1)`;
  - `wait_for_order_execution` returns on fill and on timeout.
- `tests/strategies/test_strategy.py`: parameter injection into `initialize`, delegation of the facade
  methods, the mode/account guard, and `run_backtesting` raising `NotImplementedError`.
- `tests/brokers/alpaca/`:
  - `parse_account`; `parse_calendar` (Eastern Time made tz-aware, early close);
  - the replace and close request builders;
  - `AlpacaMarketClock.next_session`, including the cache refresh;
  - `AlpacaBroker.modify_order`, `close_*` and `sync_open_orders`;
  - `BrokerError` wrapping.
- `tests/brokers/test_tracker.py`: `mark_replaced`.
- `tests/test_log.py`: directory layout and file name per mode, line format per level, ANSI codes present
  in the file, plain return values, idempotent setup.

## 10. Verification

- `uv run pytest`, `uv run ruff check` and `uv check` all pass cleanly.
- Manual script `scripts/tests/smoke_strategy_paper.py`, following the conventions of
  `smoke_alpaca_orders.py`:
  - runs a tiny strategy against real paper credentials with `sleeptime = "30S"`;
  - uses a script-local always-open clock, so it works outside market hours;
  - logs cash and portfolio value, then submits a far-from-market limit order, modifies it, and cancels
    it (`on_new_order` and `on_canceled_order` must fire);
  - calls `stop()` after three iterations.
- Check afterwards that `logs/smoke/paper/<ts>_paper/paper.log` exists, shows colours in VS Code's ANSI
  Colors plugin, and lists the hooks in the expected order.

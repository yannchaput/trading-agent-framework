# Market data layer — design

- **Date:** 2026-09-11
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md`; scoped by the "Data subproject" line in
  `docs/superpowers/specs/2026-09-10-strategy-framework-design.md` §2/§3.

## 1. Context

The strategy framework (lifecycle hooks, accounting, trading) is complete. This spec covers the market
data methods it deliberately left out: `get_last_price(s)`, `get_quote`, `get_bars` /
`get_historical_prices(_for_assets)`, plus a technical-indicators accessor
(`self.indicators.sma(...)`, per lumibot's `indicators.html`).

Reference: lumibot's `data_sources/alpaca_data.py`, `components/agents/builtins.py` (market-data tool
shapes, for naming only), and `indicators/indicators.py`.

**Not covered by this spec** (explicitly out of scope, per discussion):
- The LangChain `@tool`-wrapped, agent-callable versions of these methods (lumibot's
  `agents_builtin_tools.html` layer). No agent-orchestration layer exists in this repo yet; building
  tool-bindings ahead of it would be designed blind. This spec only adds the underlying `Broker`/`Strategy`
  methods.
- Memory tooling and notification tooling (per the original request).
- Fundamentals/SEC data (`fundamentals.html`) — not requested this round.
- Crypto, options, futures — the entity layer only defines `AssetType.STOCK` today; extending market data
  to other asset classes is a separate future subproject, not a hidden requirement of this one.
- Any caching layer. Every call hits the Alpaca API directly (see §7.3).

## 2. Scope

**In scope**
- `Broker` additions: `get_last_price`, `get_last_prices`, `get_quote`, `get_bars`.
- `Strategy` facade additions: the same four, plus `get_historical_prices` / `get_historical_prices_for_assets`
  as thin single/multi-asset wrappers over `broker.get_bars`.
- `entities/quote.py` (`Quote`) and `entities/bars.py` (`Bars`).
- `core/indicators.py`: a `pandas-ta-classic`-backed `Indicators` accessor (`strategy.indicators`).
- A `strategies/` → `core/` package rename (see §3.1) — unrelated to market data itself, but folded into
  this task since `indicators.py` was going to land in `strategies/` anyway and the rename touches the
  same files this task is already editing.

**Out of scope:** see §1.

## 3. Architecture

### 3.1 Package rename: `strategies/` → `core/`

```
src/trading_agent_framework/strategies/   →   src/trading_agent_framework/core/
tests/strategies/                          →   tests/core/
```

- Every file keeps its name and content unchanged except import paths (`trading_agent_framework.strategies`
  → `trading_agent_framework.core`), touching `core/__init__.py`, `core/strategy.py`, `core/executor.py`,
  the 6 test files, and `scripts/tests/smoke_strategy_paper.py`.
- Class names are unchanged: `Strategy`, `StrategyExecutor`, `OrderEventQueue`, etc. Only the package
  directory and import paths move.
- `CLAUDE.md`'s Architecture section is updated to reference `core/` instead of `strategies/`.

### 3.2 New/changed files

```
src/trading_agent_framework/
  entities/quote.py           NEW   Quote
  entities/bars.py            NEW   Bars
  brokers/base.py             EDIT  + get_last_price(s), get_quote, get_bars (abstract)
  brokers/alpaca/client.py    EDIT  + build_stock_data_client(credentials)
  brokers/alpaca/market_data.py  NEW   pure: timestep/request/response translation
  brokers/alpaca/broker.py    EDIT  implements the new Broker methods; owns the StockHistoricalDataClient
  core/strategy.py            EDIT  + get_last_price(s), get_quote, get_historical_prices(_for_assets),
                                     lazy `indicators` property
  core/indicators.py          NEW   Indicators, IndicatorRow
scripts/tests/
  smoke_alpaca_data.py        NEW   manual smoke script (market data, no automated-suite membership)
pyproject.toml                EDIT  + pandas-ta-classic dependency
CLAUDE.md                     EDIT  strategies/ → core/ throughout; note the new market_data.py pure module
```

**Layering rules (unchanged from the existing convention, extended):**
- "Only `orders.py` and `account.py` may import `alpaca.trading.requests`" becomes "...and
  `market_data.py` may import `alpaca.data.requests`". `market_data.py` is pure: no I/O, no client
  instances, no state — same contract as `orders.py`/`account.py`.
- `core/` never imports `alpaca` (unchanged from `strategies/`'s existing rule).

## 4. Entities

### 4.1 `entities/quote.py`

```python
@dataclass(frozen=True, slots=True)
class Quote:
    asset: Asset
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    timestamp: datetime

    @property
    def mid(self) -> Decimal | None:
        """None unless both bid and ask are present."""
```

Ordinary frozen dataclass — all fields are hashable/comparable, so default `__eq__`/`__hash__` are fine.

### 4.2 `entities/bars.py`

```python
@dataclass(frozen=True, slots=True, eq=False)
class Bars:
    asset: Asset
    timestep: str                 # "day" | "minute"
    df: pd.DataFrame              # tz-aware (America/New_York) DatetimeIndex
                                   # columns: open, high, low, close, volume (float64)
```

`eq=False` is deliberate: a dataclass-generated `__eq__` would compare `df` fields with `==`, and
`DataFrame.__eq__` returns an elementwise DataFrame rather than a bool, so equality comparison between two
`Bars` would raise `ValueError: The truth value of a DataFrame is ambiguous`. Identity-based `__eq__` (the
`object` default) avoids that footgun entirely; nothing in this codebase needs value-equality on `Bars`.

**The float boundary:** `df`'s OHLCV columns are `float64`, not `Decimal` — the second deliberate float
boundary in the codebase alongside `orders.py`'s, and for the same class of reason: `pandas-ta-classic` and
any vectorized indicator math need native float arrays. `Bars` feeds indicators, not order sizing, so this
doesn't touch the "money is Decimal" rule — scalar prices (`get_last_price`, `Quote.bid/ask`) stay `Decimal`.

## 5. `Broker` additions

| `Broker` (ABC) | Alpaca implementation |
|---|---|
| `get_last_price(asset: Asset) -> Decimal \| None` | `StockLatestTradeRequest` → `get_stock_latest_trade` (last **traded** price — see §6.2) |
| `get_last_prices(assets: Sequence[Asset]) -> dict[Asset, Decimal \| None]` | one multi-symbol `StockLatestTradeRequest` call |
| `get_quote(asset: Asset) -> Quote \| None` | `StockLatestQuoteRequest` → `get_stock_latest_quote` (`None` when Alpaca has no quote for the symbol, see §12) |
| `get_bars(assets: Sequence[Asset], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]` | `StockBarsRequest` → `get_stock_bars`, chunked at 150 symbols per call |

No `quote=`/`exchange=` parameters (lumibot carries these for crypto quote-currency and futures-exchange
routing; `AssetType` is stock-only in this codebase and no candidate strategy passes either). This is the
one deliberate simplification vs. lumibot's signatures — adding them back is a five-minute change if a
future crypto/futures subproject needs them, but they'd be dead parameters today.

`get_bars` is the **only** batched primitive — no separate/duplicate broker-level `get_historical_prices`
the way lumibot has both `get_bars` and a mostly-redundant `_get_dataframe_from_api`. `Strategy`'s
`get_historical_prices` / `get_historical_prices_for_assets` (§8) are the single/multi-asset convenience
wrappers, both calling `broker.get_bars` underneath.

Every method wraps Alpaca SDK exceptions in `BrokerError` before they escape — no log-and-return-`None`
swallowing (lumibot's `AlpacaData.get_last_price` does this, which makes a network failure and "no data for
this symbol" indistinguishable). A `None` result means the API legitimately returned nothing (e.g., a
delisted or unknown symbol); a raised `BrokerError` means the call failed.

## 6. Alpaca implementation (`brokers/alpaca/`)

### 6.1 `client.py`

```python
def build_stock_data_client(credentials: AlpacaCredentials) -> StockHistoricalDataClient: ...
```

Same credentials as `build_trading_client` — market data uses the same API key/secret as trading, just a
different client class. `AlpacaBroker.from_credentials` builds both.

### 6.2 Feed and last-price semantics

- Every request explicitly sets `feed=DataFeed.IEX` — explicit rather than relying on Alpaca's server-side
  default (which happens to also be IEX for an unentitled account, but silently changes if the account's
  entitlement ever changes). SIP support, if ever needed, is a one-line change to pass a configured feed
  through instead of the hardcoded constant — not built now (YAGNI).
- `get_last_price` uses the **latest-trade** endpoint, i.e. the actual last executed trade price — not the
  bid/ask midpoint lumibot computes it as (lumibot's `get_last_price` calls its own `get_quote` and
  averages bid/ask, which conflates "last trade" with "current quote mid" and makes `get_last_price` and
  `get_quote` redundant with each other). Here the two are genuinely different pieces of information:
  `get_last_price` → last trade; `get_quote` → current bid/ask (with `.mid` available if a caller wants
  the midpoint explicitly).

### 6.3 `market_data.py` (pure translation)

- `_parse_timestep(value: str) -> TimeFrame`: `"minute"` → `TimeFrame.Minute`, `"day"` → `TimeFrame.Day`;
  anything else raises `ValueError` listing the two supported values. **Only these two timesteps are
  supported in v1** — confirmed against every candidate strategy in the lumibot reference repo
  (`opening_range_breakout` uses `"minute"`, cross-momentum strategies use `"day"`; nothing uses a
  multi-timeframe string like `"5minute"`). Lumibot's multi-format parser/resampler is not ported — add a
  specific timestep the day a real strategy needs one.
- ~~`_bars_start(end, length, timestep)`: a generous calendar-day buffer (`length * 1.6 + 10` days)~~ —
  **superseded during planning (§12.1):** the window start comes from the Alpaca trading calendar.
  `sessions_needed(length, timestep)` is `length + 1` for `"day"` and `ceil(length / 390) + 1` for
  `"minute"` (the `+1` covers a partial current session); `bars_start(end, length, timestep, sessions)`
  returns midnight (market time) of the earliest needed session, so that day's pre-market bars are
  included. The parser still truncates to the last `length` rows.
- `build_bars_request(symbols, length, timestep, end) -> StockBarsRequest`, `build_last_trade_request`,
  `build_latest_quote_request`: thin request builders, `feed=DataFeed.IEX` always set.
- `parse_bars_response(response, assets_by_symbol, timestep, length, include_after_hours) -> dict[Asset, Bars]`:
  parses Alpaca's per-symbol bar frames, and when `timestep == "minute"` and `include_after_hours` is
  `False`, filters to the regular session (9:30–16:00 America/New_York) **before** truncating to the last
  `length` rows (so the caller gets `length` regular-session bars, not `length` raw bars further reduced by
  filtering). No-op filter for `"day"` bars.
- `parse_last_trade_response(response) -> dict[str, Decimal]`, `parse_quote_response(response, asset) -> Quote`:
  straightforward field mapping, `Decimal(str(...))` at the float boundary (matching `orders.py`'s
  existing conversion pattern).

### 6.4 `broker.py`

Holds the `StockHistoricalDataClient`, injected through the constructor (§12.3). Each method is
I/O + `market_data.py` calls + `BrokerError` wrapping — no translation logic of its own, per the existing
"broker.py contains no translation logic itself" rule.

## 7. `Strategy` facade additions (`core/strategy.py`)

```python
def get_last_price(self, asset: Asset | str) -> Decimal | None: ...
def get_last_prices(self, assets: Iterable[Asset | str]) -> dict[Asset, Decimal | None]: ...
def get_quote(self, asset: Asset | str) -> Quote | None: ...

def get_historical_prices(
    self, asset: Asset | str, length: int, timestep: str = "day", *, include_after_hours: bool = True
) -> Bars | None:
    asset = _to_asset(asset)
    return self.broker.get_bars([asset], length, timestep, include_after_hours=include_after_hours).get(asset)

def get_historical_prices_for_assets(
    self, assets: Iterable[Asset | str], length: int, timestep: str = "day", *, include_after_hours: bool = True
) -> dict[Asset, Bars]:
    return self.broker.get_bars(
        [_to_asset(a) for a in assets], length, timestep, include_after_hours=include_after_hours
    )
```

### 7.1 `indicators` property

```python
@property
def indicators(self) -> Indicators:
    if self._indicators is None:
        self._indicators = Indicators(self)
    return self._indicators
```

Lazily constructed once per `Strategy` instance, set to `None` in `__init__`.

### 7.2 Errors

Consistent with the existing `get_cash`/`get_positions` pattern (no try/except there): these facade methods
don't catch `BrokerError` either. It propagates to the strategy author, who decides how to handle a failed
market-data call — same as any other broker call. (Diverges from lumibot's `get_last_price`, which logs and
returns `None` on any exception.)

### 7.3 No caching (v1)

Every call hits the Alpaca API directly, no memoization. Matches the existing "Other decisions" precedent
in the strategy-framework spec (`get_cash`/`get_portfolio_value` also call on every invocation, no cache) —
revisit if the 200 req/min IEX rate limit becomes a real problem for some strategy's call pattern.

## 8. `core/indicators.py`

```python
class IndicatorRow:
    """Attribute-style read-only view over one pandas Series (one row of a
    multi-column indicator result, e.g. bbands -> .BBL_20_2_0)."""

class Indicators:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def __getattr__(self, name: str):
        # returns a bound callable: indicators.sma(asset, length=200, timestep="day", bars=None, **kwargs)
        ...
```

**Dispatch** (`_dispatch(asset, timestep, name, kwargs)`), for a call like `strategy.indicators.sma(asset,
length=200)`:

1. Pop `bars` from `kwargs` if given (how many historical bars to fetch); otherwise default to
   `max(50, int(kwargs.get("length", 50)) * 3)` — enough warm-up for the indicator's own window. Explicit
   and overridable, rather than lumibot's blanket 10,000-bar fallback (which assumes a backtest's already-
   loaded full history; here every call is a live fetch, so a large flat default would be wasteful).
2. Pop `include_after_hours` from `kwargs` if given (default `True`), forwarded to `get_historical_prices`.
3. `bars_obj = self._strategy.get_historical_prices(asset, length=bars_n, timestep=timestep,
   include_after_hours=include_after_hours)`. Returns `None` immediately if `bars_obj is None` or
   `bars_obj.df.empty`.
4. Look up `fn = getattr(pandas_ta_classic, name)` (raises `AttributeError` with a clear message if the
   indicator name doesn't exist, mirroring lumibot's error message).
5. Build `call_args` from whichever of `open/high/low/close/volume` are present in `bars_obj.df.columns`
   (all five, always, since `Bars.df` always has full OHLCV), update with the remaining `kwargs`, call `fn(**call_args)`.
   This is the exact mechanism lumibot's `_compute` uses — verified by reading it directly — not the `.ta`
   DataFrame-accessor style.
6. Take the **last row** of the result (`pd.Series` → last value; `pd.DataFrame` → last row wrapped in
   `IndicatorRow`). No backtest lookahead-safety slicing is needed (no backtesting exists yet) — "last row"
   and "current bar" are the same thing in a live/paper-only system.
7. Normalize `NaN` → `None` (insufficient warm-up data), never fabricate a value.

No memoization — consistent with §7.3. Each call re-fetches and recomputes; acceptable given no caching
exists anywhere else in this layer yet, and can be revisited together if rate limits become a problem.

**Custom indicators:** `strategy.indicators.custom(name, fn, asset, timestep="day", **kwargs)` — same shape
as lumibot's, for a user-supplied `fn(df, **kwargs) -> Series | DataFrame`. No tool-binding involved (§1).

**No `list_indicators`/`get_indicator`/`get_indicators` functions** — those are lumibot's LLM-tool wrappers
around this accessor, out of scope per §1.

## 9. Testing

TDD throughout, no network, following `tests/fakes.py`'s existing conventions.

- `tests/brokers/alpaca/test_market_data.py`: pure-function tests for `parse_timestep`,
  `calendar_lookback_start`/`bars_start` (the calendar-based window, §12.1), request builders, and response
  parsers, using new `make_alpaca_bar`/`make_alpaca_quote`/`make_alpaca_trade` factories in `tests/fakes.py`
  (real `alpaca.data.models` objects, same pattern as `make_alpaca_order`).
  Covers: the after-hours filter-then-truncate ordering, chunking at 150 symbols, unknown-timestep `ValueError`.
- `tests/brokers/alpaca/test_broker_market_data.py`: `AlpacaBroker`'s new methods against a new
  `FakeStockHistoricalDataClient` (same shape as the existing `FakeTradingClient`), including `BrokerError`
  wrapping on a simulated SDK failure.
- `tests/core/test_strategy.py` (moved from `tests/strategies/`, per §3.1): delegation tests for the new
  facade methods against `FakeBroker` (which needs the new abstract methods implemented), plus the
  `get_historical_prices`/`get_historical_prices_for_assets` → `broker.get_bars` wiring.
- `tests/core/test_indicators.py`: `Indicators` against a stub returning a known `Bars.df`, asserting:
  computed values match a hand-computed SMA/RSI, the default-`bars`-lookback heuristic, `NaN` → `None`,
  multi-column `IndicatorRow` attribute access, and the `custom()` path.
- `entities/bars.py`: a small test asserting `Bars() == Bars()` raises (or at minimum does not silently
  succeed) is optional — the `eq=False` choice is really a static-analysis/behavioral note, not something
  requiring its own regression test, but worth a one-line comment where declared.

## 10. Verification

- `uv run pytest`, `uv run ruff check`, `uv check` all pass.
- Manual script `scripts/tests/smoke_alpaca_data.py`, following `smoke_alpaca_orders.py`'s conventions: runs
  against real paper credentials, calls `get_last_price`, `get_quote`, `get_historical_prices` (both
  timesteps), `get_historical_prices_for_assets` on a couple of symbols, and one `indicators.sma(...)` /
  `indicators.bbands(...)` call — logs the results for manual inspection. This is also where the
  calendar-based window (`calendar_lookback_start`/`bars_start`, §12.1) gets its first real-world check
  (confirms the trading-calendar lookup lands on the right session boundaries, and that
  `include_after_hours=False` actually trims the expected rows on real IEX minute data).

## 11. Other decisions

1. `quote=`/`exchange=` parameters from lumibot's signatures are dropped (§5) — stocks-only, unused by any
   candidate strategy, easy to add back if a real need appears.
2. `include_after_hours` is kept (unlike the `quote=`/`exchange=` drop) because `opening_range_breakout`
   actually uses it today.
3. Only `"day"` and `"minute"` timesteps are supported in v1 (§6.3) — matches actual candidate-strategy
   usage; lumibot's multi-timeframe string parser/resampler is not ported.
4. `get_last_price` is the last **traded** price (latest-trade endpoint); `get_quote` is the current
   bid/ask (latest-quote endpoint) with a `.mid` convenience property — a deliberate correctness
   improvement over lumibot's conflation of the two (§6.2).
5. `feed=DataFeed.IEX` is set explicitly on every request rather than omitted (§6.2).
6. No caching anywhere in this layer for v1 (§7.3, §8) — revisit only if the 200 req/min IEX limit becomes
   a real constraint.
7. `Bars` uses `eq=False` to avoid a `DataFrame.__eq__`-ambiguous-truth-value crash on dataclass-generated
   equality (§4.2).
8. **Pre-market warm-up** (e.g. an SMA-20 on 1-minute bars has zero regular-session bars available in the
   first minute after open) is covered by the existing `include_after_hours` flag — no separate mechanism.
   A strategy or indicator call that needs bars to be ready at the open passes `include_after_hours=True`
   (the default) around that time; `Indicators` already forwards `include_after_hours` through to
   `get_historical_prices` (§8), so `strategy.indicators.sma(asset, timestep="minute", length=20)` is
   warmed up with pre-market bars by default without any extra parameter.

## 12. Amendments made while writing the implementation plan

1. **Bars window from the trading calendar** (user decision). The `length * 1.6 + 10` calendar-day
   buffer would pull ~12 days of minute data for a 30-bar request (hundreds of thousands of bars for a
   150-symbol scan). Instead `AlpacaBroker.get_bars` makes one `get_calendar` call (reusing `account.py`'s
   `build_calendar_request`/`parse_calendar`) and starts the window at the Nth-previous session (§6.3).
   The same sessions drive the `include_after_hours=False` filter, so early closes (13:00) are handled
   correctly instead of a hard-coded 9:30–16:00 mask.
2. **`get_quote` returns `Quote | None`**, following §5's own rule that "no data" is `None` and a failed
   call is `BrokerError`.
3. **The data client is injected**, not built lazily: `AlpacaBroker(..., data_client=...)`, built by
   `from_credentials` via `build_stock_data_client`. Data methods raise `BrokerError` when no data client
   was given, like `start_stream` does without a stream.
4. **Public pure-function names** in `market_data.py` (`parse_timestep`, `sessions_needed`, `bars_start`,
   ...), matching `orders.py`'s style (`map_status`, `round_price`), since tests call them directly.
5. **Bars are split- and dividend-adjusted** (`Adjustment.ALL`), mirroring lumibot's default
   (`AlpacaData._auto_adjust = True`), so indicators don't see artificial jumps at splits.
6. **A zero or missing bid/ask becomes `None`** in `Quote` (Alpaca reports an empty side as 0; lumibot
   guards the same way with a truthiness check), so `Quote.mid` is never computed from a fake zero.
7. **Test factories:** `bar_payload`/`make_alpaca_barset`/`make_alpaca_trade`/`make_alpaca_quote` in
   `tests/fakes.py` (alpaca-py builds `BarSet` from raw payload dicts, so a per-bar model factory isn't needed).

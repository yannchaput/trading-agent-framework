# Backtesting framework — design

- **Date:** 2026-09-12
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md` ("Backtesting utility" and "Backtesting logs" bullets); user's
  own follow-up constraints (below); lumibot reference `lumibot/backtesting/backtesting_broker.py`,
  `lumibot/tools/indicators.py`; the dashboard to be migrated,
  `/home/yann/projets/lumibot-trading-agent/src/lumibot_trading_agent/dashboard`
  (`reader.py`, `models.py`, `discovery.py`).

## 1. Context

`Strategy.run_backtesting()` currently raises `NotImplementedError`. This spec designs the third trading
mode, `backtesting`, so that a strategy written for `paper`/`live` runs unchanged against simulated time
and simulated fills, and emits a run directory the (to-be-migrated) Streamlit dashboard can read.

User's explicit constraints:

- Copy lumibot's backtesting capability, not its file layout.
- Use **vectorbt**.
- Keep lumibot's **no-look-ahead rule** for data, indicators, news, everything.
- Do **not** reproduce lumibot's log files; emit whatever the dashboard needs, in whatever format makes the
  dashboard migration easiest.
- Keep the same indicators: **drawdown, Sortino, Sharpe, equity curve, rolling volatility**
  (`MIGRATION_PROMPT.md` additionally asks for **max DD, Calmar, Alpha**).
- Be **broker/provider-agnostic** — IBKR is coming after Alpaca.

### 1.1 Investigation: does lumibot use a backtesting library?

Asked during brainstorming, and answered by reading the installed lumibot 4.5.90 source. **No.** The
simulation engine is entirely hand-written:

| Piece | lumibot's implementation |
|---|---|
| Engine | `lumibot/backtesting/backtesting_broker.py` — 5,024 lines; `class BacktestingBroker(Broker)` subclasses its own *live* broker base class |
| Time cursor | Own `_datetime` cursor on `DataSourceBacktesting`, advanced by `_update_datetime()`; `should_continue()` ends the run |
| Fill model | Hand-rolled in `process_pending_orders()`; per bar it pulls the current OHLCV bar and decides fills against open/high/low/close |
| Session calendar | `pandas_market_calendars` / `exchange_calendars` (calendar data only) |
| **Metrics / tearsheet** | **`quantstats-lumi`** (their fork of `quantstats`), used only in `tools/indicators.py` for post-run reporting |
| Charts | `plotly` / `matplotlib` |

No `vectorbt`, `backtrader`, `zipline`, `pyfolio` or `empyrical` anywhere in its dependencies.

Three conclusions drove the design:

1. **Lumibot's architecture is already the split we want** — hand-written event-driven engine, plus a
   statistics library for reporting. This spec substitutes **vectorbt for quantstats** in that second box.
2. **Nobody vectorizes an agentic backtest**, and neither can we. The strategy's decision at bar *t*
   depends on state produced at bar *t−1* (LLM output, memory writes, positions), while vectorbt's
   `Portfolio.from_signals` needs the whole signal array up front. Precomputing it would *be* look-ahead.
3. **We do not need 5,000 lines.** Lumibot's bulk is options assignment, futures margin, crypto quote
   currencies, smart-limit ladders, six data providers and prefetchers. Scope here is US equities.

### 1.2 Dependency feasibility (verified)

| Package | Finding |
|---|---|
| `vectorbt` 1.1.0 | `requires_python >=3.11,<3.15`; needs `pandas>=3.0.3`, `numpy>=2.4.6`, `numba>=0.66`. Project runs py3.14.3 / pandas 3.0.5 / numpy 2.5.3 → **compatible** |
| `pyarrow` | **Not installed**, and only an optional extra of pandas 3.0 (`pandas[parquet]`). Parquet output therefore adds a real dependency (~40MB wheel) — accepted, reporting-side only |
| `yfinance` 1.7.0 | Pulls 12 transitive deps (`curl_cffi`, `lxml`, `peewee`, `protobuf`, `beautifulsoup4`, …) → **optional extra with a deferred import**, per the `langchain` / `pandas_ta_classic` precedent |

### 1.3 Not covered by this spec

- **Any new data tool** (news, SEC fundamentals, screeners). Those are the separate "Tools" TODO item.
  §5.3 records the hard contract they must honour.
- **LLM call caching / deterministic replay** across backtest reruns. Lumibot has this; it is a real
  cost concern for agentic backtests but is its own subproject. Backtests call the live LLM each run.
- **Agent telemetry** (token counts, latency, cache hits) in `settings.json`. The dashboard's Parameters
  tab renders these when present and degrades gracefully when absent (`reader.load_parameters` returns
  `None` per missing key). The `agents/` layer has no counters today; adding them is a later task.
- **Cache purge / TTL / remote (S3) caching.** The TODO's "Cache management" item. §4.3 builds only the
  local read-through cache the backtester needs.
- **Options, futures, crypto, margin, short selling, corporate actions.** US equities, long-only cash
  account (was "long and short at par" until 2026-09-20; see §6.1).
- **Multi-strategy / portfolio-of-strategies backtests**, parameter sweeps, walk-forward optimisation.
- **The dashboard migration itself.** §6.4 lists exactly what it will need to change; doing it is the
  separate "dashboard" TODO item.

## 2. Decisions taken during brainstorming

| Topic | Decision |
|---|---|
| Engine shape | Event-driven simulation reusing the **existing** `StrategyExecutor`; vectorbt for reporting only |
| vectorbt's role | `Portfolio.from_orders` rebuilds equity/cash/positions from the recorded fill log; the `returns` accessor supplies Sharpe, Sortino, Calmar, max DD, drawdown tables, rolling volatility |
| Data providers | `BacktestDataSource` ABC + read-through parquet cache; **Yahoo is the default**, Alpaca also shipped. Two implementations from day one, so the seam is validated before IBKR arrives |
| Why Yahoo default | Consolidated tape and the **official closing-auction** close, decades of free history. Alpaca's free IEX feed sees only a low-single-digit % of consolidated volume, and its "close" is merely the last IEX print — measurably off the official close in thin names. Alpaca remains the choice when feed parity with paper/live matters |
| Fill model | **Next-bar-open OHLC fills.** Orders submitted during bar *t* are queued and evaluated at *t+1*: market at the open; buy limit iff `low <= limit`; sell limit iff `high >= limit`; stops iff the range crosses. Ties pessimistic |
| Why not same-bar close | A strategy deciding at bar *t* has already observed *t*'s close; filling at that close lets it trade a price it has seen — soft look-ahead, and the classic source of backtest flattery |
| Commission / slippage | Configurable; **both default to 0** (Alpaca equities are genuinely commission-free). Recorded in `settings.json` so every run is interpretable. Unmodelled slippage is the largest remaining optimism — documented, not hidden |
| Sessions | Part of the data-source contract (`sessions()`), not a new calendar dependency. Alpaca returns exact sessions incl. early closes (reusing `brokers/alpaca/clock.py`); Yahoo derives one 9:30–16:00 ET session per daily bar |
| Output layout | Fixed filenames (no timestamped prefixes, no globbing). **Parquet** for the three time series, **JSON** for `settings` / `metrics` / `description` |
| `metrics.json` keys | Flat, keyed **exactly** as the dashboard's `MetricSet` fields, summary tables under `raw.summary_tables` |
| Metrics scope | Full `MetricSet` (~38 fields) including Calmar and Alpha, **plus the benchmark series stored in `equity.parquet`** so benchmark-dependent metrics are reproducible and the dashboard needs no network |
| Indicator lines | `portfolio_value` and `cash` auto-recorded every bar; `strategy.add_line(...)` for custom series (lumibot-compatible signature). `add_marker` deferred |
| Executor change | `MAX_WAIT_SLICE_SECONDS` becomes clock-provided `MarketClock.max_wait_slice` (live 60.0, backtest `inf`) |
| Money invariant | A **third** float boundary is declared at the ledger→reporting seam; CLAUDE.md amended explicitly (§7.2) |

## 3. Architecture

### 3.1 Modules

New package `src/trading_agent_framework/backtesting/`, following the repo's pure / I-O split:

| Module | Role | Purity |
|---|---|---|
| `backtesting/clock.py` | `BacktestClock(MarketClock)`: simulated `now()`, `wait()` advancing time, on-advance callbacks. Sessions are **injected** (resolved by the runner from the data source), so the clock itself performs no I/O | no I/O, no network |
| `backtesting/fills.py` | **Pure**: OHLC fill rules, commission and slippage arithmetic. No I/O, no state, no clients | **pure** |
| `backtesting/broker.py` | `BacktestBroker(Broker)`: cash and position bookkeeping, pending-order queue, `OrderTracker` wiring. Contains no fill *rules* — it calls `fills.py` | stateful, no network |
| `backtesting/ledger.py` | Records fills, per-bar equity samples and indicator lines during the run. The seam between simulation and reporting. `Decimal` throughout | stateful |
| `backtesting/metrics.py` | vectorbt reporting. The only module importing `vectorbt`, and only inside method bodies | float64 boundary |
| `backtesting/report.py` | Serialises a finished run to `settings.json` / `metrics.json` / `*.parquet` | float64 boundary |
| `backtesting/runner.py` | Orchestration: build clock + data source + broker, run the executor, write the report | I/O |
| `backtesting/data/base.py` | **Pure**: `BacktestDataSource` ABC | **pure** |
| `backtesting/data/cache.py` | `CachedDataSource`: read-through parquet cache wrapping *any* source | I/O |
| `backtesting/data/yahoo.py` | `YahooBacktestData`; imports `yfinance` inside method bodies only | I/O |
| `backtesting/data/alpaca.py` | `AlpacaBacktestData`; reuses the pure `brokers/alpaca/market_data.py` translation | I/O |
| `backtesting/__init__.py` | Lazy re-exports via module `__getattr__`, matching `brokers/__init__.py`, so importing the package pulls in neither `vectorbt`/`numba` nor `yfinance` |

Plus `utils/errors.py`: new `BacktestError(TradingFrameworkError)`, and `BacktestDataError(BacktestError)`
for provider/cache failures. Per the existing rule, **no raw SDK or pydantic exception may escape**.

### 3.2 The executor does not change

This is the core of the design. `executor.py` already routes every wait through `MarketClock` and every
session through `clock.next_session()` — the seam was built for this. Simulation therefore needs no new
dispatch code:

```
BacktestClock.wait(seconds)
      │ advances _now
      ├─▶ on_advance callback
      │        │
      │   BacktestBroker.process_pending(bars crossed)
      │        │ fills
      │   OrderTracker  ──▶ OrderEventQueue
      │
      ▼ returns
StrategyExecutor._dispatch_events()  ──▶  strategy.on_filled_order(...)
```

Consequences, all of which preserve existing invariants:

- **"Strategy code runs on one thread" holds.** Fills are produced while the clock advances, but hooks are
  dispatched by the executor when it wakes — exactly as the Alpaca stream thread behaves in live mode.
  Simulation code only ever feeds `OrderTracker`; it never calls a strategy hook.
- **"No `time.sleep` / `datetime.now`" holds** and is now actually exercised.
- `before_market_opens` → `before_starting_trading` → `on_trading_iteration` × N →
  `before_market_closes` → `after_market_closes` fire per simulated session, unchanged.
- `sleeptime` parsing, tick maths and overrun warnings (`core/timing.py`) are reused as-is.

### 3.3 Changes to existing code

Two, both small, both improvements in their own right:

**(a) `MarketClock.max_wait_slice`.** `executor.py`'s module constant `MAX_WAIT_SLICE_SECONDS = 60.0` moves
onto the clock, because only the clock knows whether time is real:

```python
# utils/clock.py
class MarketClock(ABC):
    max_wait_slice: float = 60.0        # live/paper: correct drift (e.g. suspended laptop)

# backtesting/clock.py
class BacktestClock(MarketClock):
    max_wait_slice: float = math.inf    # simulated: jump straight to the deadline
```

```python
# core/executor.py
- clock.wait(min(remaining, MAX_WAIT_SLICE_SECONDS), self._wake)
+ clock.wait(min(remaining, clock.max_wait_slice), self._wake)
```

Live behaviour is byte-identical. Without it, a 10-year daily backtest burns roughly 2M no-op wait slices.
`MAX_WAIT_SLICE_SECONDS` is removed; existing executor tests cover the path.

**(b) `Strategy.run_backtesting()`.** Replaces the `NotImplementedError`. It rebinds `self.broker` and
`self.clock` to the simulated pair, runs the executor, writes the report and returns a `BacktestResult`.
Rebinding (rather than a classmethod factory) keeps the existing `run_strategy()` dispatch intact and
matches the ~15 strategies in the old project that already override `run_backtesting()`.

Also added to `Strategy`: `add_line(...)` (§6.3) and the backtest config class attributes (§6.1).

## 4. Data layer

### 4.1 `BacktestDataSource` (pure ABC)

Provider-agnostic by construction — this is the seam IBKR plugs into later, and the reason
`BacktestBroker` never imports a provider:

```python
class BacktestDataSource(ABC):
    name: ClassVar[str]

    @abstractmethod
    def load(self, assets: Sequence[Asset], start: datetime, end: datetime,
             timestep: str) -> None:
        """Fetch and hold the whole window for these assets. Called once per run."""

    @abstractmethod
    def bars(self, asset: Asset, cutoff: datetime, length: int,
             timestep: str) -> Bars | None:
        """The last `length` bars *closed at or before* `cutoff`, oldest first."""

    @abstractmethod
    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        """Trading sessions in [start, end], early closes included where known."""
```

Concrete sources:

| Source | Bars | Sessions |
|---|---|---|
| `YahooBacktestData` (default) | `yfinance` daily OHLCV, `auto_adjust=True` | One 9:30–16:00 ET session per daily bar |
| `AlpacaBacktestData` | Existing `brokers/alpaca/market_data.py` translation, IEX feed | **Exact**, incl. early closes, via `brokers/alpaca/clock.py`'s calendar |
| `CachedDataSource` | Wraps any source (§4.3) | Delegates |

`YahooBacktestData`'s session derivation models the ~9 half-days a year as full sessions. Irrelevant for
daily strategies (one iteration per session either way); it would matter for minute strategies, for which
Yahoo has no usable history anyway. `AlpacaBacktestData` is exact and is the choice when that matters.

### 4.2 Provider trade-offs (recorded so the default is not re-litigated)

| | Alpaca free (IEX) | Yahoo |
|---|---|---|
| Coverage | IEX prints only — low single-digit % of consolidated volume | Consolidated / primary-exchange OHLCV |
| Daily close | Last IEX trade | **Official closing auction** |
| Intraday history | Minute bars, thin and gappy in illiquid names | Not usable (~60d of 1m, unreliable) |
| Long history | Years | Decades |
| Adjustments | Split/dividend adjusted | Adjusted, **retroactively revised** |
| Stability | Official versioned SDK | Unofficial endpoint, breaks without notice |

For daily strategies — which mark and trade at the close, where the liquidity actually is — Yahoo's
official close is the *more* correct input. Intraday work needs a paid consolidated feed (Alpaca SIP,
Polygon, Databento); neither default serves it well, and that is a known limitation, not an oversight.

### 4.3 Cache

Read-through, wrapping any source, keyed by provider so two providers never collide:

```
cache/backtesting/<provider>/<symbol>_<timestep>_<start>_<end>.parquet
```

- First run fetches and writes; later runs are network-free. This matters most when iterating on an agent
  prompt against a fixed period — the expensive, repeated case.
- A sidecar `_meta.json` per file records `fetched_at`, provider and row count. Yahoo revises its adjusted
  series retroactively, so a cached file is a frozen point-in-time snapshot: **good** for reproducibility,
  but the fetch date must be legible.
- Scope is deliberately minimal: no TTL, no eviction, no remote/S3 backend. That is the TODO's "Cache
  management" item. Deleting the directory is a valid reset.

## 5. The no-look-ahead rule

### 5.1 One chokepoint, one rule

Every price a strategy can see passes through `BacktestDataSource.bars(..., cutoff=clock.now())`, and:

```
visible(bar)  ⟺  bar_end <= cutoff
```

**`bar_end`, not `bar_start`, and never "the bar containing now."** A daily bar dated 2026-01-05 becomes
visible only after 16:00 ET on the 5th; during that session the strategy's latest bar is the 2nd's. This
one distinction is where homemade backtesters leak, so it is stated as a rule and tested directly (§8).

### 5.2 What follows for free

- `get_last_price()` returns the close of the most recent *visible* bar — never the forming bar.
- `get_historical_prices()` / `get_historical_prices_for_assets()` slice `df.loc[:cutoff].tail(length)`.
- **`core/indicators.py` needs no change.** It calls `strategy.get_historical_prices` →
  `broker.get_bars` → data source, so every pandas-ta indicator inherits the gate automatically.
- **`memory/` is already correct**: `MemoryStore` takes `now=self.clock.now`, and backtests open it with
  `fresh=True`, so one run's lessons cannot leak into the next.
- `get_quote()` is synthesised from the visible bar's close (no historical quote data); documented as an
  approximation rather than silently returning a bogus spread.
- A **defensive assertion** in the gate raises `BacktestDataError` if any source ever returns a row past
  the cutoff — cheap insurance against a future provider with a sloppy index.

### 5.3 The limit, stated plainly

The gate cannot police tools that reach outside it. Any future data tool — news, SEC fundamentals, a
web-searching agent — leaks unless it derives its cutoff from `clock.now()`. Both are on the roadmap
(`MIGRATION_PROMPT.md` asks for SEC fundamentals and indicators).

> **Contract for all future data tools: no data tool may use wall-clock time.** Point-in-time cutoff comes
> from `strategy.clock.now()`, exactly as `MemoryStore` already takes `now=`.

Recorded here, and to be added to CLAUDE.md (§7.2), because it is far cheaper to honour up front than to
discover after a suspiciously profitable backtest.

## 6. Simulation and API

### 6.1 `BacktestBroker`

Implements the `Broker` ABC against the ledger instead of a network:

- **Cash and positions** in `Decimal`. `portfolio_value = cash + Σ qty × last visible close`.
- **Pending queue**: `_submit_order` conforms, validates, tracks as `new`, and queues. Nothing fills
  within the submitting bar (§2, next-bar-open).
- **`process_pending(bar)`** applies `fills.py`'s pure rules, then feeds `OrderTracker` only.
- `cancel_order`, `modify_order`, `close_position`, `close_all_positions`, `pull_*` behave as in live mode.
  `sync_open_orders()` returns `[]` (no prior state to adopt). `start_stream`/`stop_stream` are no-ops.
- `is_paper` is `True`; `_run_trading`'s live/paper guard is not on this path, but `run_backtesting()`
  asserts the mode is `BACKTESTING` for symmetry.
- **Long-only cash account (since 2026-09-20).** When an order would fill, a buy whose cost (commission
  included) exceeds cash, or a sell larger than the held quantity, is rejected whole: the order gets an
  error status and an `OrderEvent.ERROR`, and cash, positions and the ledger do not move. The check runs
  at fill time, not at submission, because a rebalance submits its sells and buys together and sizes the
  buys against the sells' proceeds; orders fill in submission order, so the sells free the cash first.
  (Originally short sales were allowed at par and cash could go negative.)

### 6.2 Public API

```python
class Strategy:
    # backtest defaults, overridable per call
    backtesting_start: datetime | None = None
    backtesting_end: datetime | None = None
    budget: Decimal = Decimal("10000")
    benchmark_symbol: str = "SPY"

    def run_backtesting(
        self, *,
        start: datetime | None = None,
        end: datetime | None = None,
        budget: Number | None = None,
        data_source: BacktestDataSource | None = None,   # default: cached Yahoo
        benchmark: str | None = None,
        timestep: str = "day",
        commission: Decimal = Decimal(0),
        slippage: Decimal = Decimal(0),
        risk_free_rate: float = 0.0,
    ) -> BacktestResult: ...
```

`BacktestResult` is a lean frozen dataclass: `run_dir: Path`, `settings: dict`, `metrics: dict`. Series stay
on disk rather than in memory — a minute-resolution run is ~1M rows.

### 6.3 `Strategy.add_line`

```python
def add_line(self, name: str, value: Number, *, color: str | None = None,
             style: str = "solid", plot_name: str = "default_plot") -> None:
    """Record a charted value at the current simulated time. No-op outside backtesting."""
```

Lumibot-compatible signature so old strategy code ports unchanged; unblocks the TODO's "add more
indicators than portfolio value (e.g. SMA 200)". `portfolio_value` and `cash` are recorded automatically
every bar, so the equity chart works with zero strategy code. `add_marker` is deliberately deferred — the
current dashboard draws no markers.

### 6.4 Output contract

```
logs/<strategy>/backtesting/<ts>_backtesting/
  backtesting.log        existing setup_strategy_logging / ColorLogger output
  settings.json          run config, timing, data source, parameters
  metrics.json           flat, keyed by the dashboard's MetricSet fields
  equity.parquet         datetime, portfolio_value, cash, positions_value, return,
                         benchmark_close, benchmark_return
  trades.parquet         time, symbol, side, status, order_type, quantity,
                         filled_quantity, price, trade_cost, trade_slippage,
                         identifier, event_kind
  indicators.parquet     datetime, name, value, color, style, plot_name
  description.json       written by the dashboard; never created or clobbered by a run
```

Directory naming already matches `setup_strategy_logging`. Column names in `trades.parquet` and
`indicators.parquet` deliberately match what `reader.py` already reads.

`settings.json` carries everything the dashboard's `Settings` model expects (`name`,
`backtesting_start`, `backtesting_end`, `budget`, `risk_free_rate`, `backtesting_data_sources`,
`backtest_time_seconds`, `parameters`) plus `mode`, `run_ts`, `timestep`, `sleeptime`, `commission`,
`slippage`, `framework_version`, and a **flat** `benchmark_symbol` string — replacing lumibot's jsonpickle
`{"py/object": "lumibot.entities.asset.Asset", …}` blob. `Settings` has `extra="allow"`, so additions are safe.

`metrics.json` is shaped to drop straight into the existing model:

```json
{
  "sharpe_strategy": 0.62, "sharpe_benchmark": 0.69,
  "sortino_strategy": 0.85, "max_drawdown_strategy": -0.3488,
  "calmar_strategy": 0.53, "alpha": 0.041, "beta": 0.98,
  "volatility_strategy": 0.2827,
  "raw": { "summary_tables": { "eoy_returns_vs_benchmark": [], "drawdowns": [] } }
}
```

`MetricSet.model_validate(json.load(f))` then works directly, which deletes `from_scalars()`,
`_parse_tearsheet_csv()` and `_parse_tearsheet_cell()` — about 120 lines of percent-string parsing removed
rather than ported.

**Constraint (verified):** unlike `Settings`, `MetricSet` does **not** set `extra="allow"`. `metrics.json`
must therefore contain *only* `MetricSet` field names plus `raw` — no stray top-level keys such as a
window size or an engine version. Anything else belongs inside `raw`, or in `settings.json`. The §8
round-trip test enforces this.

**Two improvements over lumibot's output**, both deliberate:

1. The **benchmark series lives in `equity.parquet`**, fetched once at run time. `load_cumulative_returns()`
   loses its live `yf.download()` call: the dashboard works offline and can no longer display numbers that
   disagree with the run that produced them.
2. Fixed filenames — no timestamp-prefixed globbing.

**Exactly two dashboard edits are then unavoidable**, both one-liners, and both belong to the separate
dashboard task: `discovery.py` globs `**/metrics.json` instead of `**/*_tearsheet_metrics.json`, and
`RunRef.from_path()` stops requiring the `agent_` directory prefix. Every other `reader.py` loader either
keeps working (it already calls `pd.read_parquet`) or shrinks.

### 6.5 Metrics

`metrics.py` hands float64 arrays to vectorbt and collects the full `MetricSet`:

- **Requested**: max/avg drawdown, longest DD days, Sortino, Sharpe, equity curve, and annualised
  volatility.
- **Rolling volatility needs no separate output.** The dashboard already derives its rolling charts
  (`rolling_volatility_chart`, `window=126`, verified) from a *daily returns* array. Since `equity.parquet`
  stores `return` and `benchmark_return`, the rolling series is computed by the dashboard on demand —
  `metrics.py` must not persist a redundant rolling file.
- **`MIGRATION_PROMPT.md`**: Calmar, Alpha.
- **Remaining `MetricSet` fields** so the scorecard and side-by-side pages render: total return, CAGR,
  Omega, volatility, Beta, correlation, Treynor, information ratio, R², skew, kurtosis, win-days %,
  win-month %, recovery factor — each with its benchmark twin.
- **Summary tables**: `eoy_returns_vs_benchmark`, `drawdowns`.

Benchmark returns come from the stored series, so Alpha/Beta/correlation/R² are computed once and frozen.
`risk_free_rate` is recorded in `settings.json` and used for Sharpe/Sortino/Treynor.

## 7. Conventions

### 7.1 Deferred imports

`vectorbt` (with `numba`) and `yfinance` are heavy. Both are imported **inside method bodies only**, never
at module level, matching `agents/manager.py`'s `langchain` and `core/indicators.py`'s
`pandas_ta_classic`. `backtesting/__init__.py` uses lazy `__getattr__` re-exports like
`brokers/__init__.py`, so a paper/live strategy that never backtests pays nothing.

`pyproject.toml`: `vectorbt` and `pyarrow` as main dependencies (reporting always runs after a backtest);
`yfinance` under a `[project.optional-dependencies] backtesting-yahoo` extra.

### 7.2 CLAUDE.md amendments required

Three, to be made as part of the implementation:

1. **Money boundary.** The current rule says money is `Decimal` with exactly two float boundaries and
   "Don't add a third." vectorbt needs float64, and parquet/JSON output is float too. Amend to declare a
   third boundary at the **ledger→reporting seam** (`backtesting/metrics.py` and `backtesting/report.py`).
   Everything that can affect a trade — broker, fills, cash, positions, ledger — stays `Decimal` and exact;
   float appears only once results leave the engine.
2. **No-look-ahead contract** for all future data tools (§5.3).
3. **Architecture section**: the new `backtesting/` package and the `max_wait_slice` seam.

## 8. Testing

Per the existing rules: hand-written fakes (never `MagicMock`), and **`tests/` never touches the network**.

| Layer | Test |
|---|---|
| `fills.py` | Table-driven over synthetic OHLC bars: market, limit, stop, stop-limit; gap-through-the-limit; pessimistic tie-breaking; commission and slippage arithmetic. Pure, no I/O |
| Data gate | `FakeDataSource` with known bars: the `bar_end <= cutoff` rule, and specifically that the in-progress bar is **invisible** |
| **No-look-ahead** | Integration: a recording strategy logs every bar timestamp it observes; asserts each closed strictly before the decision that saw it. **This test protects the whole premise** |
| Cache | Round-trip through `CachedDataSource` over a fake source: miss writes, hit reads, no second fetch, `_meta.json` contents |
| `BacktestBroker` | Cash and position accounting across buy/sell/partial/cancel/reject; `Decimal` exactness with no float drift; `OrderTracker` state transitions |
| `BacktestClock` | Session iteration, early closes, `next_session()` → `None` past the end date, `max_wait_slice` |
| Executor integration | Full simulated session: hook order, `first_iteration`, `sleeptime` ticks, fills dispatched on the executor thread |
| `metrics.py` | A hand-computed return series with known Sharpe/Sortino/max-DD, compared against vectorbt's output as golden values |
| `report.py` | Round-trip: write a run, then validate it with the dashboard's own `Settings` / `MetricSet` models — proving the §6.4 contract holds |
| Lazy imports | Extend `tests/brokers/test_lazy_imports.py`: importing `backtesting` must not import `vectorbt`, `numba` or `yfinance` |
| Real data | `scripts/tests/smoke_backtest.py`, run by hand, excluded from `tests/` like the existing smoke scripts |

## 9. Documentation

- `README.md`: a backtesting section — how to run one, the output layout, the cache location, how to
  install the Yahoo extra.
- `CLAUDE.md`: the three amendments in §7.2.
- `TODO.md`: tick "Backtesting (vectorbt)"; leave "Cache management" and "dashboard" open, and note that
  LLM-call replay caching and agent telemetry are now identified follow-ups.

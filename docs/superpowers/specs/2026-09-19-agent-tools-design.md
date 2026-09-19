# Agent tools — design

- **Date:** 2026-09-19
- **Status:** approved design, pending implementation plan
- **Brief:** `prompts/MIGRATION_PROMPT.md` ("Tools" / "Agentic framework" bullets); lumibot reference
  [agents_builtin_tools.html](https://lumibot.lumiwealth.com/agents_builtin_tools.html).

## 1. Context

This is the last item on the MIGRATION checklist in `TODO.md`. `agents/manager.py` can already build
LangChain agents and take a `tools=[...]` sequence of plain typed callables (no `@tool` decoration
needed — LangChain infers the schema from type hints + docstring). `memory/tools.py` already ships the 9
memory tools this way. What's missing is everything else a trading agent needs: order/account/market-data
access, technical indicators, and research tools (news, macro, fundamentals).

Reference implementations read in full or in part while designing:
- `lumibot/components/agents/builtins.py` — `_bind_alpaca_news`, `_bind_get_fred_series`, the
  `_AccountTools`/`_MarketTools`/`_NewsTools`/`_IndicatorTools`/`_FundamentalTools`/`_MacroTools`/
  `_MemoryTools`/`_OrderTools` namespace classes, and `_BuiltinTools.all()`.
- `lumibot/fundamentals/sec.py` — `SECFundamentals`: a hand-rolled SEC EDGAR REST client with
  point-in-time (`as_of`) filtering built into every lookup, plus local JSON/text caching.
- This repo's `memory/tools.py`, `agents/manager.py`, `core/strategy.py`, `core/indicators.py`,
  `brokers/alpaca/market_data.py`, `brokers/alpaca/client.py`, `brokers/alpaca/broker.py`.

**Not covered by this spec** (explicitly out of scope):
- Options, crypto, notifications, DuckDB/docs tools (lumibot has these namespaces; nothing in this
  project's broker layer or strategy roster needs them yet).
- Trailing-stop or notional-sized orders — `Strategy.create_order` doesn't build them today, so
  `trading_tools` wraps what the facade actually supports (quantity-based market/limit/stop/stop-limit).
  Extending `create_order` itself is a separate change.
- SEC cash-flow statements, filing full-text search, and filing-section extraction — lumibot's
  `SECFundamentals` has ~15 methods; this spec ports the 6 the `warren_buffett` strategy actually needs
  (company facts, income statement, balance sheet, filings list, filing document text). The rest can be
  added the same way later.
- Injecting these tools into any specific strategy's `agents.create(...)` call, or writing new
  `agent_*.py` strategy files. This spec delivers the tool factories only.

## 2. Decisions taken during brainstorming

| Topic | Decision |
|---|---|
| Split | `PrebuiltTools.all(strategy)` bundles trading + account + market data + indicators + memory. News/macro/fundamentals are separate factories a strategy wires in explicitly. |
| `PrebuiltTools.all()` scope | Includes market data (`get_last_price`/`get_quote`/`get_bars`) and a generic indicator tool, not just orders/account/memory — an agent needs price/indicator context to decide trades. |
| Module layout | New `agents/tools/` package, one file per domain, mirroring `memory/tools.py`'s factory-function pattern. |
| News provider | Alpaca News via alpaca-py's existing `NewsClient`/`NewsRequest` (same architecture as `market_data.py`, reuses `AlpacaCredentials` — no new dependency, no new env var). |
| Macro provider | FRED via `fredapi`, using `realtime_start`/`realtime_end` vintage parameters (not the naive `fredgraph.csv` endpoint some lumibot example strategies use) so backtests never see post-release data revisions. New env var `FRED_API_KEY`. |
| Fundamentals provider | A trimmed, ported version of lumibot's own `SECFundamentals` (raw REST against `data.sec.gov`, `as_of`-filtered, locally cached) — not a third-party EDGAR SDK, since none give point-in-time filtering for free. New env var `SEC_EDGAR_USER_AGENT`. |
| No-look-ahead | Each of news/macro/fundamentals takes the `Strategy` (not a raw client) so it can pass `strategy.clock.now()` as an explicit cutoff — the tool-level equivalent of `BacktestDataSource`'s chokepoint, since nothing enforces this centrally for non-price data. |
| Backtesting | Full support from day one: all three research tools resolve their as-of cutoff from `strategy.clock.now()`, so `news_builtin`, `m2_liquidity`, and `warren_buffett`-style strategies can backtest against them. |

## 3. Architecture

### 3.1 Modules

New package `src/trading_agent_framework/agents/tools/`:

| Module | Role | Depends on |
|---|---|---|
| `tools/trading.py` | `trading_tools(strategy) -> list[Callable]`: submit/cancel/close orders. | `core.strategy` |
| `tools/account.py` | `account_tools(strategy) -> list[Callable]`: cash/portfolio/buying power, positions. | `core.strategy` |
| `tools/market_data.py` | `market_data_tools(strategy) -> list[Callable]`: last price, quote, bars. | `core.strategy` |
| `tools/indicators.py` | `indicator_tools(strategy) -> list[Callable]`: one generic indicator tool. | `core.strategy`, `core.indicators` |
| `tools/prebuilt.py` | `PrebuiltTools.all(strategy) -> list[Callable]`, combining the four above plus `memory_tools(strategy.memory)`. | the four above, `memory.tools` |
| `tools/news.py` | `news_tools(strategy) -> list[Callable]`: Alpaca news search. | `core.strategy`, `brokers.alpaca` |
| `tools/macro.py` | `macro_tools(strategy) -> list[Callable]`: FRED series lookup. | `core.strategy`, `config.env` |
| `tools/fundamentals.py` | `fundamentals_tools(strategy) -> list[Callable]`: SEC company facts/statements/filings. | `core.strategy`, `fundamentals` |
| `tools/__init__.py` | Exports `PrebuiltTools` and each domain factory. | — |

Plus, in existing packages:

| Module | Change |
|---|---|
| `brokers/alpaca/market_data.py` | Add `build_news_request(...)` and `parse_news(...)` (pure) — it is already "the only module allowed to import `alpaca.data.requests`", and `NewsRequest` lives in that same module, so news joins the existing IEX-market-data family rather than creating a new pure module that would need its own carve-out of that rule. |
| `brokers/alpaca/client.py` | Add `build_news_client(creds) -> NewsClient`. |
| `brokers/alpaca/broker.py` | Add `AlpacaBroker.get_news(...)` I/O method (same try/except-then-`BrokerError` pattern as `get_quote`/`get_bars`). |
| `config/env.py` | Add `FredCredentials.from_env()` (mirrors `AlpacaCredentials`/`LLMCredentials`; `FRED_API_KEY`). |
| `utils/errors.py` | Add `MacroDataError(TradingFrameworkError)`, `FundamentalsError(TradingFrameworkError)`. |

New top-level package `src/trading_agent_framework/fundamentals/` (mirrors `lumibot/fundamentals/`,
kept separate from `brokers/` since it isn't a trading broker and separate from `agents/tools/` since it
has real I/O and caching logic worth unit testing on its own):

| Module | Role |
|---|---|
| `fundamentals/sec.py` | **Pure**: CIK lookup response parsing, the income-statement/balance-sheet tag maps (ported from lumibot), `as_of` candidate filtering and statement-period matching, filing-list parsing, URL building. No I/O. |
| `fundamentals/edgar_client.py` | I/O: `httpx` GET against `data.sec.gov` / `www.sec.gov`, `SEC_EDGAR_USER_AGENT` header, disk cache under `<project_root>/cache/sec/`, a rate-limit sleep between misses (SEC's fair-access policy). Wraps every request exception as `FundamentalsError`. |
| `fundamentals/__init__.py` | Exports `SecEdgarClient` (I/O) and the pure helpers `fundamentals_tools` needs. |

### 3.2 Boundaries

- **Every tool factory takes the `Strategy`, not a raw broker/client**, mirroring `memory_tools(store)`
  but one level up: `Strategy` is the friendliest, already-`Decimal`/`Asset`-typed facade, and it's the
  only thing that exposes `clock.now()` for the no-look-ahead cutoff.
- **Trading/account/market-data/indicator tools call `Strategy` methods only** — never `strategy.broker`
  directly, never new broker methods — except `account_tools`, which needs `strategy.broker.get_account()`
  for `buying_power` (not currently exposed as a `Strategy` property; `cash`/`portfolio_value` are).
- **News stays inside the existing Alpaca broker layer** (`market_data.py`/`client.py`/`broker.py`);
  `tools/news.py` is a thin closure over `strategy.broker.get_news(...)`, same shape as
  `tools/market_data.py` over `strategy.get_last_price(...)`. If `strategy.broker` isn't an `AlpacaBroker`,
  the tool returns `{"error": "news requires an Alpaca broker"}` rather than raising, so a strategy that
  wires this tool in against some future non-Alpaca broker fails softly for the agent instead of crashing
  the run.
- **Macro and fundamentals are broker-independent** — they never touch `strategy.broker`, only
  `strategy.clock.now()`. `macro.py` builds its own `fredapi.Fred` client per call (lazily imported,
  mirroring `core/indicators.py`'s deferred `pandas_ta_classic` import and `agents/manager.py`'s deferred
  `langchain` import — a strategy that never wires in the macro tool never pays for `fredapi`).
  `fundamentals.py` builds one `SecEdgarClient` per `fundamentals_tools(strategy)` call and closes over it.
- **Error wrapping.** Every tool catches its domain's framework exception
  (`BrokerError`/`MacroDataError`/`FundamentalsError`/`OrderValidationError`) and returns
  `{"error": str(exc)}`, exactly like `memory/tools.py`'s `_write` helper — consistent across the whole
  tool surface, and it lets the model self-correct instead of crashing the agent run. Programming errors
  (`TypeError`, etc.) are not caught and propagate, same as memory tools.
- **`mutates_trading = True`** is set on `trading_tools`' order-submitting/canceling/closing functions
  (the same function-attribute convention `memory/tools.py` uses for `remember_decision`). News, macro,
  fundamentals, account, market-data and indicator tools are all read-only and carry no such flag.
- **Token budget.** Every tool returns a lean dict (repo convention from `memory/tools.py`): bars/news/
  fundamentals results are capped (`length`/`limit` parameters, clamped server-side) and only the fields
  an agent plausibly reasons over are included — never a raw SDK/pydantic object, never an entire
  `companyfacts` payload.

### 3.3 No-look-ahead per domain

This is the one property that differs from `memory/tools.py` (whose `now` callable already came from
`strategy.clock.now` via `MemoryStore`) enough to spell out per tool:

- **News:** `tools/news.py` passes `end=strategy.clock.now()` to `NewsRequest` whenever the caller doesn't
  give an explicit `end` (and clamps any caller-given `end` that's after `clock.now()` down to it, mirroring
  lumibot's `lookahead_clamped` behavior) — the same discipline the `news_builtin` strategy's docs describe
  ("validates article timestamps against the simulated datetime").
- **Macro:** `tools/macro.py` always passes `realtime_end=strategy.clock.now().date()` to
  `fredapi.Fred.get_series(...)`, so FRED returns the vintage of each observation as it was known on that
  date — not a later-revised value. `observation_end` is capped the same way so no future observation date
  is returned even if a revision changed it.
- **Fundamentals:** every `fundamentals/sec.py` lookup takes `as_of: datetime` and filters SEC facts/filings
  to `filed <= as_of` (ported verbatim from `SECFundamentals`'s candidate filtering). `tools/fundamentals.py`
  always calls with `as_of=strategy.clock.now()`.

## 4. Tool catalogue

All tools follow `memory/tools.py`'s shape: full type hints, a **one-line docstring** (sent to the model
as the tool description on every call), plain values in and out (`str`/`float`/`int`/`bool`/`list`/`dict`
— no `Decimal`, no entities), errors as `{"error": ...}`.

### 4.1 `trading_tools(strategy)`

| Tool | Parameters | Behavior |
|---|---|---|
| `submit_order` | `symbol: str, quantity: float, side: str, limit_price: float \| None = None, stop_price: float \| None = None, time_in_force: str = "day"` | `strategy.create_order(...)` + `strategy.submit_order(...)`; order type follows from the prices given, exactly as `create_order` already infers it. Returns a lean order dict. `OrderValidationError`/`BrokerError` -> `{"error": ...}`. |
| `cancel_order` | `order_id: str` | `strategy.get_order(order_id)`; `{"error": "unknown order_id"}` if not found, else `strategy.cancel_order(order)` and `{"identifier", "status": "cancel_requested"}`. |
| `cancel_open_orders` | *(none)* | `strategy.cancel_open_orders()`; returns `{"status": "ok"}`. |
| `close_position` | `symbol: str, fraction: float = 1.0` | `strategy.close_position(...)`; lean order dict or `{"status": "no position"}`. |
| `sell_all` | *(none)* | `strategy.sell_all()`; returns `{"orders": [lean order dict, ...]}`. |
| `get_orders` | *(none)* | `{"orders": [lean order dict, ...]}` from `strategy.get_orders()`. |
| `get_order` | `order_id: str` | Lean order dict or `{"error": "unknown order_id"}`. |

`submit_order`, `cancel_order`, `cancel_open_orders`, `close_position`, `sell_all` get
`mutates_trading = True`. Lean order dict: `{"identifier", "symbol", "side", "order_type", "quantity",
"status", "limit_price", "stop_price", "filled_quantity", "avg_fill_price"}` (`None` fields omitted,
`Decimal` -> `float`).

### 4.2 `account_tools(strategy)`

| Tool | Parameters | Behavior |
|---|---|---|
| `get_account_balance` | *(none)* | `{"cash", "portfolio_value", "buying_power"}` (floats) from `strategy.broker.get_account()`. |
| `get_positions` | *(none)* | `{"positions": [lean position dict, ...]}` from `strategy.get_positions()`. |
| `get_position` | `symbol: str` | Lean position dict or `{"position": None}`. |

Lean position dict: `{"symbol", "quantity", "side", "avg_fill_price", "current_price", "market_value",
"unrealized_pnl"}` (`None` fields omitted, `Decimal` -> `float`).

### 4.3 `market_data_tools(strategy)`

| Tool | Parameters | Behavior |
|---|---|---|
| `get_last_price` | `symbol: str` | `{"symbol", "price"}` or `{"error": "no trade data"}`. |
| `get_quote` | `symbol: str` | `{"symbol", "bid", "ask", "mid", "timestamp"}` or `{"error": ...}`. |
| `get_bars` | `symbol: str, length: int = 30, timestep: str = "day"` | `length` clamped to `1..200`. `{"symbol", "timestep", "bars": [{"date", "open", "high", "low", "close", "volume"}, ...]}`, oldest first, from `strategy.get_historical_prices(...)`. |

### 4.4 `indicator_tools(strategy)`

One generic tool rather than one per pandas-ta-classic function (there are hundreds; a fixed per-indicator
tool list would bloat the tool payload for no benefit an LLM can't get from a name string):

| Tool | Parameters | Behavior |
|---|---|---|
| `get_indicator` | `name: str, symbol: str, timestep: str = "day", params: dict[str, Any] \| None = None` | Dispatches to `strategy.indicators.<name>(symbol, timestep, **(params or {}))`. `AttributeError` (unknown indicator name) -> `{"error": ...}`. A scalar result -> `{"indicator", "symbol", "value": <float | None>}`; an `IndicatorRow` (multi-column, e.g. Bollinger Bands) -> `{"indicator", "symbol", "value": <dict>}`. |

`params` is an explicit `dict` argument (not `**kwargs`) so LangChain's type-hint-based schema inference
gives the model a clean, single-object parameter instead of an unconstrained kwargs bag.

### 4.5 `PrebuiltTools.all(strategy)`

```python
class PrebuiltTools:
    @staticmethod
    def all(strategy: Strategy) -> list[Callable[..., Any]]:
        return [
            *memory_tools(strategy.memory),
            *trading_tools(strategy),
            *account_tools(strategy),
            *market_data_tools(strategy),
            *indicator_tools(strategy),
        ]
```

`agents/tools/__init__.py` also exports each domain factory individually (`trading_tools`,
`account_tools`, `market_data_tools`, `indicator_tools`, `news_tools`, `macro_tools`,
`fundamentals_tools`) so a strategy that wants a subset — or wants to compose its own bundle alongside
one research tool — doesn't have to go through `PrebuiltTools.all()`.

### 4.6 `news_tools(strategy)`

| Tool | Parameters | Behavior |
|---|---|---|
| `search_news` | `symbols: str = "", start: str \| None = None, end: str \| None = None, limit: int = 10, include_content: bool = False` | `limit` clamped `1..50`. Builds a `NewsRequest` via `market_data.build_news_request(...)`, calls `strategy.broker.get_news(...)`, parses via `market_data.parse_news(...)`. Returns `{"count", "articles": [{"id", "headline", "summary", "source", "created_at", "symbols", "content": <str, only if include_content>}]}`. `end` defaults to (and is clamped to) `strategy.clock.now()`; `start` defaults to `end - 7 days`. |

Single tool for v1 (headline/summary scan with an optional full-content fetch by re-calling with a
narrower `symbols`/date window), matching what `news_builtin` and `news_sentiment` actually do — not
lumibot's full pagination/page-token surface.

### 4.7 `macro_tools(strategy)`

| Tool | Parameters | Behavior |
|---|---|---|
| `get_fred_series` | `series_id: str, start_date: str \| None = None, limit: int = 60` | `limit` clamped `1..250`. Builds `fredapi.Fred(api_key=FredCredentials.from_env().api_key)`, calls `get_series(series_id, observation_start=start_date, observation_end=cutoff, realtime_end=cutoff)` where `cutoff = strategy.clock.now().date()`. Returns the last `limit` observations: `{"series_id", "as_of", "observations": [{"date", "value"}]}`. `ValueError`/network errors -> `MacroDataError` -> `{"error": ...}`. |

### 4.8 `fundamentals_tools(strategy)`

Backed by one `SecEdgarClient` shared across the tools returned from a single `fundamentals_tools(strategy)`
call (one CIK lookup / cache warms all of them for a given symbol).

| Tool | Parameters | Behavior |
|---|---|---|
| `get_company_facts` | `symbol: str, max_facts: int = 40` | `{"symbol", "cik", "as_of", "facts": {tag: {"value", "unit", "filed", "form"}}}`, `as_of=strategy.clock.now()`. |
| `get_income_statement` | `symbol: str` | `{"symbol", "as_of", "values": {field: {"value", "unit", "filed", "form"}}}` for the tag map in §4 of `fundamentals/sec.py` (revenue, cost_of_revenue, gross_profit, operating_income, net_income, eps_basic, eps_diluted). |
| `get_balance_sheet` | `symbol: str` | Same shape, balance-sheet tag map (assets, current_assets, cash, liabilities, current_liabilities, debt, equity, shares_outstanding). |
| `get_filings` | `symbol: str, form: str \| None = None, limit: int = 10` | `limit` clamped `1..25`. `{"symbol", "as_of", "filings": [{"form", "accession_number", "filing_date", "report_date", "document_url"}]}`, filtered to `filed <= as_of` and optionally to one `form` (e.g. `"10-K"`). |
| `get_filing_document` | `symbol: str, accession_number: str, max_chars: int = 8000` | Fetches and HTML-strips the filing's primary document; `{"symbol", "accession_number", "document_url", "text", "truncated"}`. |

All five raise `FundamentalsError` at the `SecEdgarClient`/`fundamentals/sec.py` boundary on request/parse
failure (unknown ticker, HTTP error, malformed payload); the tool layer catches it and returns
`{"error": ...}`.

## 5. `fundamentals/` package detail

### 5.1 `edgar_client.py` (I/O)

```python
class SecEdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_request_interval_seconds: float = 0.2) -> None: ...
    def get_json(self, url: str, cache_key: tuple[str, ...]) -> dict[str, Any]: ...
    def get_text(self, url: str, cache_key: tuple[str, ...]) -> str: ...
```

- Cache root: `<project_root>/cache/sec/` (new top-level `cache/` dir, gitignored — company-level data,
  not strategy- or mode-scoped, so it doesn't follow the `logs/<strategy>/<mode>` or
  `memory/<strategy>/<mode>` layout).
- `SEC_EDGAR_USER_AGENT` is required (`ConfigurationError` if unset/blank) — SEC's fair-access policy
  blocks generic/missing User-Agents, so no default is guessed (mirrors `AlpacaCredentials`/
  `LLMCredentials`: fail fast with a clear message rather than silently degrade).
- Every `httpx` exception / non-2xx response is wrapped as `FundamentalsError` at this boundary (repo
  rule: no raw library exception escapes).
- **Known limitation, inherited from lumibot:** cache entries never expire. A paper/live run that keeps a
  strategy alive for months could serve a stale `companyfacts` payload for a symbol it cached early on.
  Acceptable for v1 (matches the reference implementation); a TTL or manual cache-bust is a follow-up if
  it bites in practice.

### 5.2 `sec.py` (pure)

Ported from `SECFundamentals`, trimmed to what §4.8 needs:
- `INCOME_STATEMENT_TAGS`, `BALANCE_SHEET_TAGS` (verbatim from lumibot).
- `parse_company_tickers(payload, symbol) -> cik` (the `ticker_to_cik` lookup logic, parsing the fetched
  `company_tickers.json`).
- `filter_facts_as_of(facts, tags, as_of) -> list[candidate]` and `latest_fact(candidates) -> dict | None`
  (the `filed <= as_of` filtering and sort-by-recency logic).
- `match_statement(field_candidates) -> dict[str, dict]` (the anchor/same-period matching that keeps a
  statement's fields from mixing facts pulled from different filings — lumibot's `_statement_anchor`/
  `_best_matching_statement_fact`, kept because dropping it would silently produce wrong statements).
- `parse_filings(submissions_payload, *, form, as_of, limit) -> list[dict]` (the `get_filings` row-building
  loop).
- `filing_url(cik, accession_number, primary_document) -> str`.
- `strip_html(raw) -> str` (kept — filing documents are HTML/XBRL, and an agent needs readable text).

No caching, no `requests`, no strategy/clock knowledge in this module — it takes `as_of: datetime` as a
plain argument, same as `memory/records.py` takes no live clock at all.

## 6. Configuration

### 6.1 New env vars (`env/.env.example` additions)

```
# FRED (Federal Reserve Economic Data) API key for agents/tools/macro.py.
# Get one at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=

# Identity string SEC EDGAR requires on every request (fair-access policy):
# "<app or project name> <contact email>". Requests without one may be blocked.
SEC_EDGAR_USER_AGENT=
```

### 6.2 `config/env.py` addition

```python
@dataclass(frozen=True, slots=True)
class FredCredentials:
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FredCredentials:
        source = env if env is not None else os.environ
        api_key = source.get("FRED_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError("Missing or blank FRED_API_KEY environment variable")
        return cls(api_key=api_key)
```

`SEC_EDGAR_USER_AGENT` is read the same way, inline in `fundamentals_tools(strategy)` (no dataclass needed
for a single string — `SecEdgarClient` takes it as a constructor argument, raising `ConfigurationError` if
blank).

### 6.3 New dependencies (`pyproject.toml`)

```
"fredapi>=0.5,<0.6",
"httpx>=0.27",
```

`fredapi` is lazily imported inside `macro_tools` (never at module level) so a strategy that never wires
in the macro tool doesn't pay for it. `httpx` is a direct import in `fundamentals/edgar_client.py` — small
enough (and useful enough as a general HTTP client for future providers) not to warrant deferring.

## 7. Testing

pytest, hand-written fakes (repo convention, no `MagicMock`), no network in `tests/`.

- `tests/agents/tools/test_trading_tools.py`, `test_account_tools.py`, `test_market_data_tools.py`,
  `test_indicator_tools.py` — each tool's parameter names/docstring/one-liner, lean return shape, error
  dict on the relevant exception, `mutates_trading` only on the five trading tools, built against a fake
  `Strategy`/`Broker` (extends `tests/fakes.py`).
- `tests/agents/tools/test_prebuilt.py` — `PrebuiltTools.all()` returns the concatenation of the five
  factories' outputs, in order, no duplicate tool names.
- `tests/agents/tools/test_news_tools.py` — request building (`end` default/clamp to a fake clock's
  `now()`), response parsing, `{"error": ...}` on a fake broker raising `BrokerError`.
- `tests/agents/tools/test_macro_tools.py` — `get_fred_series` against a fake `fredapi.Fred`-shaped object
  (inject via a factory seam, not a real network call), `realtime_end`/`observation_end` pinned to a fake
  clock, limit clamping, `MacroDataError` -> `{"error": ...}`.
- `tests/fundamentals/test_sec.py` — tag-map filtering, `as_of` exclusion of post-cutoff facts, statement
  anchor/period matching (including the "candidate doesn't match anchor -> omitted" case), filings
  filtering by form and `as_of`, HTML stripping.
- `tests/fundamentals/test_edgar_client.py` — cache hit skips the HTTP call, cache miss writes it,
  `FundamentalsError` wraps a fake transport's raised exception, `ConfigurationError` on blank
  `SEC_EDGAR_USER_AGENT`.
- `tests/agents/tools/test_fundamentals_tools.py` — each tool against a fake `SecEdgarClient`, lean shapes,
  error propagation.

Manual smoke scripts (network, real credentials — excluded from `tests/`, run by hand):
`scripts/tests/smoke_news.py`, `scripts/tests/smoke_macro.py`, `scripts/tests/smoke_fundamentals.py`,
mirroring `scripts/tests/smoke_alpaca_data.py`.

## 8. Documentation

- `CLAUDE.md`: new architecture bullets for `agents/tools/` and `fundamentals/`; a gotcha entry for the
  per-domain no-look-ahead convention (§3.3 of this spec) alongside the existing "No data tool may use
  wall-clock time" bullet, since it's the same invariant applied to a new class of tool.
- `README.md`: document `FRED_API_KEY` and `SEC_EDGAR_USER_AGENT` alongside the existing Alpaca/LLM env
  var docs.
- `TODO.md`: user strikes "Tools" from the MIGRATION list themselves (per the memory spec's precedent —
  this file carries the user's own uncommitted changes and isn't edited by implementation work).

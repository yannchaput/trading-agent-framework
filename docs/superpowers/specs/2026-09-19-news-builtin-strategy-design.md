# News-builtin strategy port — design

- **Date:** 2026-09-19
- **Status:** approved design, pending implementation plan
- **Brief:** port the pasted lumibot strategy `strategies/agent_alpaca_news_builtin.py` to this framework
  as an LLM agent using `PrebuiltTools` + `news_tools`, with backtesting and memory. Env file:
  `env/.env.news_builtin.backtesting`.

## 1. Context

`agent_alpaca_news_builtin.py` is a copy of a lumibot strategy (`WrappingStrategy`, `AgentRuntime`,
`BuiltinTools.news.alpaca_news()`, `self.backtest(YahooDataBacktesting, ...)`) and does not import in this
repo. The agent scans broad-market news (SPY/QQQ/DIA/IWM), reads the most relevant article in full, then
holds SPY or QQQ (bullish) or a defensive ETF (negative/unclear).

It is the first strategy in this repo that drives an LLM agent (`CrossMomentumStrategy` never touches
`self.agents`). Porting it surfaced four framework gaps:

1. **News is unavailable in backtests.** `agents/tools/news.py` hard-requires `strategy.broker` to be an
   `AlpacaBroker` and returns `{"error": "news requires an Alpaca broker"}` under `BacktestBroker`
   (`CLAUDE.md` documents this as "live/paper only by design"). A news strategy that cannot get news in a
   backtest is meaningless, and the hard `AlpacaBroker` check also blocks the planned IBKR broker.
2. **LLM env mismatch.** `LLMCredentials.from_env` requires `LLM_BASE_URL` and `LLM_API_KEY`. The env file
   has `API_BASE_URL=http://localhost:8005//v1/chat/completions` (full endpoint, double slash) and no key.
   `ChatOpenAI` needs the `/v1` base and appends `/chat/completions` itself.
3. **Unbounded article payload.** `market_data.parse_news` returns `content` verbatim, and `search_news`
   allows `limit` up to 50 with `include_content=True`. That can put ~50 full articles in one tool result,
   which a local Qwen3-30B context cannot absorb (this repo's token-budget rule).
4. **`main.py` cannot launch it.** `_run_strategy` is hard-wired to cross-momentum (universe loading,
   `strategy_class(broker=, mode=, universe=)`).

**Out of scope:** LLM-call caching for deterministic backtest reruns and agent telemetry (both already in
`TODO.md`), stripping Qwen `<think>` tags from logs, news pagination, an IBKR broker (only the seam that
makes it possible), and other lumibot strategies.

## 2. Decisions taken during brainstorming

| Topic | Decision |
|---|---|
| News in backtests | Broker-independent: a `NewsProvider` protocol reached through an optional `Broker.news_provider()` hook. Alpaca implements it; a future IBKR broker returns its own. |
| Backtest news source | `BacktestBroker` takes `news_source=`, plumbed from `Strategy.run_backtesting(news_source=...)` exactly like `data_source=`. Defaults lazily to `AlpacaNewsProvider.from_credentials(AlpacaCredentials.from_env())`. |
| Look-ahead gate | Unchanged and stays in the tool: `news_tools` clamps `end` to `strategy.clock.now()`. Providers are dumb and never read the wall clock. |
| Env naming | Standardize on the documented `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`. `API_BASE_URL` is renamed in the env file. No `LLM_PROTOCOL` parameter (YAGNI). |
| Defensive ETF | `SHV`, exposed as a strategy parameter (`defensive_symbol`). The old prompt said only "a defensive ETF", too vague for a local model. |
| Prompts | Rewritten for this framework's tool surface (`search_news` has no `page_token`/`exclude_contentless`), and extended with memory usage. |

## 3. Framework changes

### 3.1 LLM config — `agents/config.py` (pure)

`LLMCredentials.from_env`:

- `LLM_BASE_URL` stays required (still `ConfigurationError` when missing/blank), but the value is
  normalized by a pure helper `normalize_base_url(url)`:
  strip whitespace; drop a trailing `/chat/completions` (and any trailing `/`); collapse `//` in the
  path (never the one after the scheme). `http://localhost:8005//v1/chat/completions` becomes
  `http://localhost:8005/v1`.
- `LLM_API_KEY` becomes **optional**: missing/blank falls back to the module constant
  `PLACEHOLDER_API_KEY = "not-needed"` (the OpenAI SDK requires a non-blank string; vLLM ignores it).
  The field stays `repr=False`.
- `LLM_MODEL` unchanged.

Also: `env/.env.example` documents the optional key and that a full endpoint URL is tolerated;
`env/.env.news_builtin.backtesting` renames `API_BASE_URL` to `LLM_BASE_URL` (value unchanged, the
normalizer handles it). Existing tests that assert `LLM_API_KEY` is required flip to assert the placeholder.

### 3.2 News seam

New `brokers/news.py` (no `alpaca` import, no I/O):

```python
class NewsProvider(Protocol):
    def get_news(
        self, symbols: Sequence[str] = (), *, start: datetime | None = None,
        end: datetime, limit: int = 10, include_content: bool = False,
    ) -> list[dict[str, object]]: ...
```

Same signature `AlpacaBroker.get_news` has today. Lean article dicts, errors as `BrokerError`.

- `brokers/base.py`: `Broker.news_provider(self) -> NewsProvider | None` returns `None` by default (not
  abstract: a broker without news simply has none). `base.py` still never imports `alpaca`.
- New `brokers/alpaca/news.py`: `AlpacaNewsProvider(news_client)` wrapping the existing pure
  `market_data.build_news_request` / `parse_news` (moved call site, not moved logic) and the
  `except Exception -> BrokerError` wrap currently in `AlpacaBroker.get_news`. Classmethod
  `from_credentials(creds)` uses the existing `client.build_news_client`. This module does I/O, so it is
  not "pure" in the `orders.py` sense; it lives beside `broker.py` under the same rule (translation logic
  stays in `market_data.py`).
- `AlpacaBroker.get_news` delegates to its `AlpacaNewsProvider`; `AlpacaBroker.news_provider()` returns it
  (or `None` when built without a news client, replacing `_require_news_client`'s raise for the tool path;
  a direct `get_news` call without a client still raises `BrokerError` as today).
- `BacktestBroker.__init__(..., news_source: NewsProvider | None = None)`. `news_provider()` returns the
  injected source, or lazily builds `AlpacaNewsProvider.from_credentials(AlpacaCredentials.from_env())` on
  first call and memoizes it. A backtest that never calls the news tool never needs credentials for news.
  `backtesting/broker.py` imports the Alpaca provider only inside that method body (keeping
  `test_lazy_imports.py` green).
- `run_backtest(..., news_source=None)` and `Strategy.run_backtesting(..., news_source=None)` pass it to
  the `BacktestBroker` constructor, like `data_source`.

### 3.3 `agents/tools/news.py`

- Replace the `isinstance(strategy.broker, AlpacaBroker)` check with
  `provider = strategy.broker.news_provider()`; `None` returns
  `{"error": "no news provider for this broker"}`. The `AlpacaBroker` import is deleted.
- Clock gating unchanged (`end` clamped to `strategy.clock.now()`, default window 7 days, tz-aware
  inputs required).
- **Token budget (gap 3):**
  - `MAX_CONTENT_LIMIT = 5`: when `include_content=True`, `limit` is clamped to 5 (headline scans keep
    the existing 1..50 clamp).
  - `MAX_CONTENT_CHARS = 6000`: applied in `market_data.parse_news` (pure), which appends a
    `"... [truncated]"` marker when it cuts. About 1.5k tokens per article, so at most ~7.5k per call.
- The agent cannot fetch an article by id, so the prompt tells it to re-query a narrow `start`/`end`
  window around the chosen article's `created_at` with a small `limit` (the same approach the old prompt
  used: "same or narrower window").

### 3.4 `CLAUDE.md`

Rewrite the "news is live/paper-trading only by design" clause: news is provider-based and works in all
three modes; the clock gate still lives in `news_tools`; document the `NewsProvider` seam,
`Broker.news_provider()`, `BacktestBroker(news_source=)`, the content caps, and the `LLMCredentials`
normalization/optional key.

## 4. The strategy

`strategies/news_builtin/` package (mirrors `strategies/cross_momentum/`, and keeps prompts out of the
class file), replacing `strategies/agent_alpaca_news_builtin.py`, which is deleted:

- `prompts.py` — `SYSTEM` and `TASK` string constants (v1), built from the strategy parameters.
- `agent_news_builtin.py` — `NewsBuiltinStrategy(Strategy)`.
- `__init__.py` — exports `NewsBuiltinStrategy` (same shape as cross_momentum's).

### 4.1 Behavior

- `parameters` (class default, overridable via constructor): `symbols = ("SPY", "QQQ")`,
  `defensive_symbol = "SHV"`, `news_symbols = "SPY,QQQ,DIA,IWM"`, `backtest_every_n_iterations = 5`,
  `warmup_trading_days = 0`. Backtest window and money are class attributes
  (`backtesting_start`/`backtesting_end`/`budget`/`benchmark_symbol`), which `Strategy.run_backtesting`
  already reads.
- `initialize()`: `sleeptime = "1D"` in backtest, `"2H"` in paper/live (via `self.is_backtesting`);
  `self.vars.iteration_count = 0`; create agent `news_trader`:
  `self.agents.create(name="news_trader", system_prompt=SYSTEM, tools=[*PrebuiltTools.all(self), *news_tools(self)])`.
  Model comes from `LLM_MODEL` (no `model=`).
- `on_trading_iteration()`: increment the counter; in backtest skip unless it is iteration 1 or a multiple
  of `backtest_every_n_iterations`; otherwise
  `result = self.agents["news_trader"].run(TASK, context={"current_datetime": self.get_datetime().isoformat()})`
  and log `result.output` at info level. Agent failures (`AgentError`) are logged as errors and do not
  crash the run loop (a hung/failed LLM call must not kill a multi-week backtest), except configuration
  errors, which propagate.
- `run_backtesting()`: `super().run_backtesting(data_source=YahooBacktestData, preload_assets=[SPY, QQQ, SHV],
  commission=Decimal("0.001"), warmup_trading_days=...)`. `start`/`end` come from the class attributes and
  are timezone-aware in `MARKET_TZ` (`run_backtest` validates this). No `news_source` argument: the
  default Alpaca provider applies.

### 4.2 Prompts (behavioral contract)

`SYSTEM` states, in this order: the allowed instruments (only SPY, QQQ, or the defensive ETF; never
short, never margin, never USD/FOREX); the workflow (1. `search_memory` for prior decisions/theses;
2. `search_news` scan, `symbols=news_symbols`, `include_content=False`, `limit=30`; 3. pick the most
relevant article and call `search_news` again with a narrow `start`/`end` window around its `created_at`,
`include_content=True`, small `limit`, to read it; 4. compare article timestamps with the provided current
datetime and ignore stale news; 5. decide); the trade rules (bullish means buy SPY or QQQ; negative or
unclear means buy the defensive ETF; sell the current position before buying the other; quantity =
floor(cash / last price); moderate sizing; never duplicate open orders, so check account/orders first);
and memory duty (`remember_decision` after every decision, `open_thesis`/`close_thesis` when the regime
call changes). `TASK` is the short per-iteration instruction plus the datetime context.

Prompt wording will be iterated against the real model during the manual verification step; the
structure above is fixed by this spec, the exact text is an implementation detail.

## 5. Wiring — `main.py`

`AGENT_STRATEGIES` changes from `{name: class}` to `{name: builder}` where a builder is
`(broker: AlpacaBroker, mode: TradingMode) -> Strategy | None` (returns `None` after printing its own
error, e.g. the missing universe file). `_run_strategy` keeps everything strategy-independent (env
loading, `AlpacaCredentials` check, broker construction, `strategy.run_strategy()`) and calls the builder.
`_build_cross_momentum` gets the existing universe logic verbatim; `_build_news_builtin` is
`NewsBuiltinStrategy(broker=broker, mode=mode)`. Registered under the name `news_builtin`, which matches
`env/.env.news_builtin.<mode>`. `strategy_name` for logs/memory/`client_order_id` is `news_builtin`.

## 6. Testing

All automated tests use hand-written fakes and never touch the network (repo rule).

- **`LLMCredentials`**: normalization table (full endpoint, double slash, trailing slash, already-clean
  URL, scheme's `//` preserved), placeholder key when `LLM_API_KEY` missing/blank, base URL still required.
- **News seam**: `AlpacaNewsProvider` against the existing fake news client (request built, response
  parsed, exception becomes `BrokerError`); `AlpacaBroker.news_provider()` present/absent;
  `news_tools` with a fake provider (clock clamp, limit clamps incl. `MAX_CONTENT_LIMIT`, missing provider
  returns the soft error); `parse_news` truncation with marker; `BacktestBroker` with an injected
  `news_source`, plus the lazy default not constructing anything until asked; `test_lazy_imports.py`
  still passing.
- **Strategy** (`tests/strategies/`): with a fake chat model passed through `agents.create(model=...)` (or
  a monkeypatched `AgentManager`), assert cadence (runs on iteration 1 and every 5th in backtest, every
  iteration otherwise), `sleeptime` per mode, that tools passed to the agent include `search_news` and the
  memory/trading tools, that an `AgentError` is logged not raised, and `run_backtesting` argument
  wiring (patch `run_backtest`).
- **`main.py`**: builder registry resolves `news_builtin` and `cross_momentum`; unknown name still exits.
- **Manual (not in `tests/`)**: `scripts/tests/smoke_strategy_news.py`, run by hand against the real vLLM
  and Alpaca credentials: one agent iteration against real news in paper mode (no order submitted unless
  the agent decides to), plus a short real backtest window
  (`uv run python -m trading_agent_framework.main news_builtin backtesting` with narrowed dates via
  parameters) to confirm end-to-end that news, tools, memory and report output work. Documented in
  `scripts/tests/HOWTO.md`.

## 7. Files touched

| File | Change |
|---|---|
| `agents/config.py`, `env/.env.example`, `env/.env.news_builtin.backtesting` | URL normalization, optional key, rename var |
| `brokers/news.py` (new), `brokers/base.py` | `NewsProvider` protocol, `Broker.news_provider()` |
| `brokers/alpaca/news.py` (new), `brokers/alpaca/broker.py`, `brokers/alpaca/market_data.py` | Alpaca provider, delegation, content truncation |
| `backtesting/broker.py`, `backtesting/runner.py`, `core/strategy.py` | `news_source=` plumbing |
| `agents/tools/news.py` | provider lookup, content-limit clamp |
| `strategies/news_builtin/` (new), `strategies/agent_alpaca_news_builtin.py` (deleted) | the strategy |
| `main.py` | builder registry |
| `tests/...`, `scripts/tests/smoke_strategy_news.py`, `scripts/tests/HOWTO.md` | tests and smoke script |
| `CLAUDE.md`, `README.md` | document the seam, the strategy, and how to run it |

## 8. Risks / open items

- **Local model quality.** A 30B "thinking" model may loop or mis-call tools. The agent's default
  LangGraph recursion limit bounds a runaway; `timeout_seconds` stays `None` per the framework's
  documented choice. Prompt iteration happens in the manual verification step.
- **Alpaca news history.** Historical news depth and rate limits (shared 200 req/min budget with data) are
  assumed adequate for the 2025-01 to 2026-04 window at roughly one agent run per week; to be confirmed
  by the manual backtest.
- **No LLM-call cache yet**, so every backtest rerun re-pays the LLM (tracked in `TODO.md`).

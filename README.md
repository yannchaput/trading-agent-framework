# trading-agent-framework
A trading agent harness managing runtime, the broker layer, agent layer, memory etc.

## ✏️ Environment variable management 

This directory holds the environment files that `trading_agent_framework.config.env`
loads credentials and configuration from. **Only two files in this directory are
committed to git: `.env.example` and this `README.md`.** Everything else matching
`env/.env.*` is gitignored (see `/env/*` with `!/env/README.md` and
`!/env/.env.example` in the repository `.gitignore`) because those files carry real
secrets or point at real accounts.

| Goal | Commands |
| --- | --- |
| Upgrade dependencies | `uv lock --upgrade` then `uv sync` to install them in .venv. |
| Upgrade uv | `uv self update` |
| To package and distribute | `uv build` |
| Run a strategy | `uv run agent <strategy_name> <trading_mode>` or `uv run poe <script_name>` where script name is defined in `pyproject.tom`|
| Run the dashboard | `uv run dashboard` or `uv run poe dashboard` |
| Run a batch | `uv run batch-universe` or `uv run poe batch-universe` |
| Run the ruff linter | `uv run ruff check` |
| Run unit tests | `uv run pytest` |
| Run code coverage and view the report | `coverage run -m pytest` and `coverage report` |
| Display dependency tree | `uv tree` |
| List outdated package | `uv tree --outdated` and `uv tree --outdated --depth=1` |
| Test upgrade of a package | `uv lock --upgrade-package <package>`, `git diff uv.lock` to see the changes, `git restore uv.lock` in case of failures |
| Identify conflict on a package | `uv tree --invert <culprit>` |

## Naming convention

Strategy- and mode-specific env files follow:

```
env/.env.{strategy_name}.{live|paper|backtesting}
```

For example, a strategy named `momentum` running in paper mode reads from
`env/.env.momentum.paper`. `{strategy_name}` should match the `strategy_name` you
pass to `load_strategy_env`, and the mode suffix must be one of the three values in
`trading_agent_framework.config.env.TRADING_MODES`: `live`, `paper`, `backtesting`.

## Resolution order

`trading_agent_framework.config.env.resolve_env_file` (called by
`load_strategy_env`) looks for the first file that exists, in this order:

1. `env/.env.{strategy_name}.{trading_mode}` -- the strategy- and mode-specific file.
2. `env/.env` -- a shared fallback for all strategies/modes.
3. `.env` at the repository root -- a last-resort fallback.

The first candidate found is loaded (via `python-dotenv`, with `override=True`) and
its path is returned. If none of the three exist, `load_strategy_env` raises
`ConfigurationError`.

## Getting started

Copy `.env.example` to a strategy- and mode-specific file (or to `env/.env` for a
shared default) and fill in real values:

```bash
cp env/.env.example env/.env.momentum.paper
```

Never commit the copy -- it is already covered by the `env/.env.*` gitignore
pattern, so a plain `git add` will not pick it up.

## Environment variables

| Variables | Used by | Required when |
|---|---|---|
| `BROKER`, `BROKER_API_IS_PAPER` | broker factory | optional (`alpaca`, `true`) |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Alpaca trading | `BROKER=alpaca`, paper/live |
| `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` | IBKR trading | `BROKER=ibkr` (all have defaults) |
| `ALPACA_DATA_API_KEY`, `ALPACA_DATA_API_SECRET`, `ALPACA_DATA_IS_PAPER` | market data and calendar | paper/live (both brokers); `AlpacaBacktestData` backtests |
| `ALPACA_NEWS_API_KEY`, `ALPACA_NEWS_API_SECRET` | news tool | a strategy uses the news tool |

Groups never fall back to each other; with one Alpaca key pair, repeat it in each group you need.

`FRED_API_KEY` is needed only if a strategy wires in `agents.tools.macro_tools` (FRED macro series). Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html.

`SEC_EDGAR_USER_AGENT` is needed only if a strategy wires in `agents.tools.fundamentals_tools` (SEC company facts/filings). SEC's fair-access policy requires a real identity string on every request: `"<app or project name> <contact email>"`.

## Components

### 📊 Strategy Dashboard   

Compare backtesting runs across all strategies in a Streamlit web app.

**Run:** `uv run dashboard`

**What it shows:**
- **Scorecard page** — aggregate table of all strategies' latest runs with key metrics (CAGR, Sharpe, Sortino, Max DD, Win Days%, etc.)
- **Run Detail page** — deep dive into a single run with equity curve (with cash/asset decomposition), drawdown chart, cumulative returns vs SPY benchmark, rolling Sharpe/Sortino/volatility charts, monthly returns heatmap, daily returns distribution, parameters & LLM telemetry table, and yearly returns vs benchmark table.
- **Side-by-Side page** — compare two runs across all metrics.

**Data sources:** Reads from `logs/` directory — auto-discovers backtesting runs by scanning for `*_tearsheet_metrics.json` files. No manual indexing needed.

**Why:** Rapidly compare strategy performance, debug LLM agent behavior (token usage, latency, call counts), and validate that config changes (model, tools, prompts) produce measurable improvements.


### 📈 Strategies

#### 📈 `triple_screen` — NASDAQ Triple Screen
| Field | Value |
|---|---|
| **File** | `agent_triple_screen.py` |
| **Model** | `openai/deepseek-v4-flash` |
| **Agent** | `triple_screen_portfolio_manager` |
| **Tools** | `run_triple_screen` (custom @agent_tool) |
| **Asset universe** | 50+ large-cap US stocks across tech, semiconductors, cloud, cybersecurity, biotech, consumer, financials, energy, industrials (see `portfolio.py`) |
| **Agent frequency** | every 5 trading days (backtest) |
| **Trading modes** | backtest |
| **Benchmark** | SPY |

Alexander Elder's Triple Screen system: Screen 1 (weekly MACD histogram slope), Screen 2 (daily Slow Stochastic oversold + turning up), Screen 3 (price break above prior-day high). A single agent tool batches all 3 screens. The AI selects up to 5 passing stocks for equal-weight allocation; holds cash when no setups pass.

Run: `uv run python -m lumibot_trading_agent.main triple_screen backtesting`

#### 📈 `news_sentiment` — News Sentiment
| Field | Value |
|---|---|
| **File** | `agent_news_sentiment.py` |
| **Model** | `openai/deepseek-v4-pro` |
| **Agent** | `news_scout` |
| **Tools** | `search_news` (custom @agent_tool wrapping Alpaca News REST API) |
| **Asset universe** | any well-known US-listed stocks (AAPL, MSFT, NVDA…), rotates to SHV when defensive |
| **Agent frequency** | every 5 trading days (backtest) |
| **Trading modes** | backtest |
| **Benchmark** | SPY |

Event-driven strategy: the agent searches Alpaca news headlines for catalysts (earnings beats, upgrades, product launches, deals), discovers stocks to buy, and rotates to SHV when conviction is weak. Demonstrates @agent_tool wrapping a REST API, agent-driven stock discovery, and replay caching of deterministic runs.

Docs: [news-sentiment-strategy](https://lumibot.lumiwealth.com/agents_canonical_demos.html#news-sentiment-strategy)

Run: `uv run python -m lumibot_trading_agent.main news_sentiment backtesting`

#### 📈 `news_builtin` — Alpaca News Built-in
| Field | Value |
|---|---|
| **File** | `strategies/news_builtin/agent_news_builtin.py` |
| **Model** | from `LLM_MODEL` in the env file |
| **Agent** | `news_trader` |
| **Tools** | `PrebuiltTools.all(strategy)` + `news_tools(strategy)` (`search_news`) |
| **Asset universe** | SPY, QQQ, DIA, IWM + defensive ETF |
| **Agent frequency** | every 5 trading days (backtest) |
| **Trading modes** | backtest, paper |
| **Benchmark** | SPY |

News-driven trading through the framework's `search_news` tool (broker-agnostic `NewsProvider`; works in backtests, gated on the simulated clock). The agent scans broad-market headlines with `search_news`, reads the most relevant article in full (article content is capped), then holds SPY/QQQ when the regime is bullish or the defensive ETF (SHV) when it is negative or unclear. It uses the framework's memory tools to record decisions and its regime thesis, and compares article timestamps against the simulated datetime.

Needs `LLM_BASE_URL` and `LLM_MODEL` (OpenAI-compatible server, e.g. vLLM) plus Alpaca credentials in `env/.env.news_builtin.<mode>`; defensive ETF is `SHV`.

Run: `uv run python -m trading_agent_framework.main news_builtin backtesting`

#### 📈 `macro_risk` — Macro Risk
| Field | Value |
|---|---|
| **File** | `agent_macro_risk.py` |
| **Model** | `openai/deepseek-v4-flash` |
| **Agent** | `macro_allocator` |
| **Tools** | `get_stock_bars`, `get_market_movers` (custom @agent_tool wrapping Alpaca market data API) |
| **Asset universe** | TQQQ (risk-on), SHV (risk-off) |
| **Agent frequency** | every 5 trading days (backtest) |
| **Trading modes** | backtest, paper, live |
| **Benchmark** | SPY |

AI-driven macro risk allocation: the agent checks TQQQ and SPY price trends via historical bars, plus market movers for additional context. If TQQQ is trending up it goes risk-on; if trending down it rotates to SHV. Always fully invested — never cash.

Run: `uv run python -m lumibot_trading_agent.main macro_risk backtesting`

#### 📈 `warren_buffett` — Warren Buffett AI team
| Field | Value |
|---|---|
| **File** | `agent_warrenbuffett_tradingteam.py` |
| **Model** | `deepseek/deepseek-v4-flash` |
| **Agents** | `annual_report_reader`, `valuation_skeptic`, `portfolio_manager` |
| **Tools** | `BuiltinTools.*` |
| **Asset universe** | Custom basket |
| **Agent frequency** | every 15 trading days ( 1 month for backtest) |
| **Trading modes** | backtest |
| **Benchmark** | SPY |

An AI team of agent applying Warren Buffett trading method: an annual report screener, a skeptical assessor and the portfolio manager decides to trade (or not).
This is a long term strategy with long trading windows.

Run: `uv run python -m lumibot_trading_agent.main warren_buffett backtesting`

#### 📈 `cross_momentum_v5` — Cross Sectional Momentum (Final Version)

Final Version of the **cross_momentum** strategy is `V5` drawn from `V2.3c` and ` V2` flavors. This strategy keeps a good tradeoff between risk appetence and performance (Sortino, CAGR) and loss (DrawDown).

__Note:__ The `cross_momentum` strategy can enable a "diagnostic mode" storing key KPIs during backtesting for further analysis.
This utility is hard coded in the method `_compute_and_persist_diagnostics` and is enabled with `enable_diagnostics` parameter. It applies only during backtesting.
Those diagnostics are useful for forensics. Use 
```python
uv run python -m lumibot_trading_agent.strategies.candidates.cross_momentum.batch_diagnostics_drawdown_episodes logs/agent_cross_momentum_v23/backtesting/2024-01-01_120000_backtesting/diagnostics.parquet
```
to analyze the parquet file.

#### 🚴‍♂️ Cross momentum universe batch

Run `uv run batch-universe` to retrieve an extended list of US shares and filter them based on market cap, vol, etc.
The strategy will rely on those symbols as input.

#### 📈 `opening_range_breakout` — Agent based opening range breakout strategy (ORB)
| Field | Value |
|---|---|
| **File** | `agent_opening_range_breakout.py` |
| **Model** | `deepseek/deepseek-v4-flash` |
| **Agent** | `opening_range_breakout` |
| **Tools** | Custom: `market_last_price_from_bars`, `self.compute_atr_tool` + (Lumibot built-in) |
| **Asset universe** | 100 top US stocks and ETFs |
| **Agent frequency** | every trading hour |
| **Trading modes** | backtest, paper, live |
| **Benchmark** | SPY |

An agent implementing an opening range breakout strategy. During the first 15min of the regular cash session, the agent set the Opening Range (OR) window. Then the nest 2 hours, it picks up the "breakouts": stock prices and volume raising up a parameterized threshold. Rest of day is to set a traling stop and sell if the price goes down the stop.

Run: `uv run agent opening_range_breakout backtesting`


### Feedbacks 💰 

| Strategy        | status            | Observations                                   |
|-----------------|-------------------|-------------------------------------------------|
| macro_risk      | live              | Good balance between aggressive position (TQQQ) and defensive one (SHV). Very slow growth.<br>This strategy has too much latency: if TQQQ drops due to some signal, it's already too late to switch in defensive mode. The same the overway around. Too much inertia. |
| m2_liquidity    | paper             | m2_liquidity decide to buy TQQQ (risk on) or SHV (risk off) based on macro economic data: M2 money supply, employment, prices, etc. Opposite to macro_risk, its decisions are based on long term indicators not so frequently updated. Hence it is less sensible to day 2 day volatility. However could be risky if the stock drops durably. Long term strategy. Quite analogous to macro_risk on a longer timeframe. |
| news_builtin    | live          |  Strategy based on news sentiment analysis. The strategy buy QQQ or SPY if market regime is bullish. Defensive ETF otherwise. This strategy is more stable than benchmark SPY. This is a good tradeoff. |
| cross sectional momentum (V5) | live  | Best candidate so far: robut over a 10 year window and good metrics. |
| opening range breakout | live | Every day, select candidates breakouts. The rest of the day (or longer) detect sudden drops to sell the goods. This is a short term strategy with immediate earnings. |


## Supporting documentation:
* [Lumibot Agent](https://lumibot.lumiwealth.com/agents.html)
* [Lumibot observability](https://lumibot.lumiwealth.com/agents_observability.html)
* [Lumibot environment variables](https://lumibot.lumiwealth.com/environment_variables.html)
* [Broker configuration](https://lumibot.lumiwealth.com/deployment.html#alpaca-configuration)
* [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server?tab=readme-ov-file#claude-code-configuration)



## Backtesting

`Strategy.run_backtesting(start=..., end=...)` runs a strategy against simulated time
and simulated fills -- no network calls to a broker, and (with the default data
source) no Alpaca account needed at all.

```python
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
end = datetime.now(ET)

result = my_strategy.run_backtesting(
    start=end - timedelta(days=365),
    end=end,
)
print(result.metrics["sharpe_strategy"], result.run_dir)
```

`start` and `end` must be **timezone-aware** (trading sessions carry a market
timezone, so a naive bound cannot be compared against them) -- `datetime.now(ET)`,
not `datetime.now()`. A naive bound raises a `BacktestError` saying so.

`start`/`end`/`budget`/`benchmark` fall back to the `backtesting_start`/
`backtesting_end`/`budget`/`benchmark_symbol` class attributes when the matching
keyword argument is omitted (`budget` defaults to `Decimal("10000")` and
`benchmark_symbol` to `"SPY"` if neither is set). `run_backtesting` also accepts
`data_source`, `timestep` (`"day"` by default), `commission`, `slippage` and
`risk_free_rate` keyword arguments.

- **Data source**: defaults to `YahooBacktestData(start, end)` -- free daily OHLCV, no
  API key, requires the `backtesting-yahoo` extra (`uv sync --extra backtesting-yahoo`).
  Pass `data_source=AlpacaBacktestData(...)` for feed parity with paper/live trading,
  or wrap either in `backtesting.CachedDataSource(source, cache_dir)` to cache fetched
  bars under `<cache_dir>/<source.name>/` for network-free reruns against the same
  window (useful when iterating on an agent prompt against a fixed period).
- **Fill model**: orders fill against the *next* bar's open (never the bar they were
  submitted on), so a strategy can't trade a price it has already observed. Limit and
  stop orders fill only when the bar's range actually touches the trigger price.
- **Output**: `logs/<strategy>/backtesting/<timestamp>_backtesting/` -- `settings.json`,
  `metrics.json` (Sharpe, Sortino, Calmar, max drawdown, and the rest of the standard
  tearsheet), and three parquet files (`equity.parquet`, `trades.parquet`,
  `indicators.parquet`). `run_backtesting` returns a `BacktestResult` with `run_dir`,
  `settings` and `metrics` attributes pointing at the same data.
- **Agent telemetry**: on by default (`run_backtesting(agent_telemetry=False)` to skip it). Per-agent
  totals -- model calls, tool calls, tokens, latency -- go to `settings.json["agents"]`, and each
  model call to `memory/<strategy>/backtesting/llm_stats.sqlite`, which is wiped when the next
  backtest of that strategy starts (like the agent memory), so the dashboard's per-call chart covers
  the latest run only. Paper/live strategies opt in with `agent_telemetry = True` on the class; their
  database accumulates across runs, so delete it by hand when it grows -- it is separate from
  `memory.sqlite`, so agent memory is untouched.
- **No look-ahead, structurally**: every price the strategy can see is gated by
  `clock.now()` -- a bar is visible only after it has *closed*. See
  `docs/superpowers/specs/2026-09-12-backtesting-framework-design.md` for the full
  design and its guarantees.

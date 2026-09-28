# trading-agent-framework
A trading agent harness managing runtime, the broker layer, agent layer, memory etc.

## 👷 Commands


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

## ✏️ Environment variables

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

## 📈 Interactive Brokers (IBKR)

Set `BROKER=ibkr` in the strategy's paper/live env file. IBKR handles orders, account and
positions; prices, bars, the market calendar and news still come from Alpaca (`ALPACA_DATA_*`,
`ALPACA_NEWS_*`). Backtests are unaffected (Yahoo or Alpaca data).

1. Install and start **IB Gateway**, and log in with your paper (or live) IBKR username.
2. In IB Gateway, *Configure > Settings > API > Settings*: enable "ActiveX and Socket Clients",
   untick "Read-Only API", keep socket port 4002 (paper) / 4001 (live), trust 127.0.0.1.
3. Use a **cash** account: the broker refuses to trade live on a margin account (IBKR cannot
   turn off margin or shorting through the API).
4. Give each strategy running at the same time its own `IBKR_CLIENT_ID`, and keep it stable
   across restarts: only the client id that placed an order can modify or cancel it.
5. IB Gateway logs out and restarts daily. For unattended runs use [IBC](https://github.com/IbcAlpha/IBC)
   to restart and log in automatically; the broker reconnects on its next call.
6. IBKR trades whole shares through the API: fractional quantities are rounded down, and
   notional (dollar-amount) orders are rejected.

Manual checks against a paper Gateway (env file `env/.env.ibkr.integration-tests`):

```bash
uv run python scripts/tests/smoke_ibkr_account.py   # read-only
uv run python scripts/tests/smoke_ibkr_orders.py    # places paper orders
```

## Components

### 🚧 Backtesting

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
`data_source`, `timestep` (`"day"` by default), `fees`, `slippage` and
`risk_free_rate` keyword arguments.

Fees default to the broker named by `BROKER` in the backtest env file (`alpaca` when unset,
`ibkr`): `brokers/fees.py`'s `TradingFeeFactory` charges each order the broker's commission
(Alpaca $0, IBKR Pro Fixed max($1, $0.005/share)) plus the regulatory fees every broker passes
on (SEC and FINRA TAF on sells, CAT on both sides). The run's totals are in `settings.json["fees"]`.

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

### 📊 Strategy Dashboard   

Compare backtesting runs across all strategies, and the local vLLM models benchmarked for the agents, in a dark-themed Streamlit web app with two tabs.

**Run:** `uv run dashboard` (Models tab reads `../benchmark-vllm-models/results` by default; override with `uv run dashboard --benchmark-dir PATH`)

**What it shows:**
- **Backtesting tab** — the three pages below, reached from the sidebar:
  - **Scorecard page** — aggregate table of all strategies' latest runs with key metrics (CAGR, Sharpe, Sortino, Max DD, Win Days%, etc.)
  - **Run Detail page** — deep dive into a single run with equity curve (with cash/asset decomposition), drawdown chart, cumulative returns vs SPY benchmark, rolling Sharpe/Sortino/volatility charts, monthly returns heatmap, daily returns distribution, parameters & LLM telemetry table, and yearly returns vs benchmark table.
  - **Side-by-Side page** — compare two runs across all metrics.
- **Models tab** — pick a benchmark run (latest by default): best/fastest model cards, a summary table (overall and per-category scores, runs passed, text tool calls, latency, tokens/s), category bars, a quality-vs-speed scatter, a per-scenario heatmap, and a drill-down listing each repeat's checks with their pass/fail reasons.

**Data sources:** Reads from `logs/` directory — auto-discovers backtesting runs by scanning for `*_tearsheet_metrics.json` files. No manual indexing needed.

**Why:** Rapidly compare strategy performance, debug LLM agent behavior (token usage, latency, call counts), and validate that config changes (model, tools, prompts) produce measurable improvements.

### Feedbacks 💰 

| Strategy        | status            | Observations                                   |
|-----------------|-------------------|-------------------------------------------------|
| macro_risk      | discarded              | Good balance between aggressive position (TQQQ) and defensive one (SHV). Very slow growth.<br>This strategy has too much latency: if TQQQ drops due to some signal, it's already too late to switch in defensive mode. The same the overway around. Too much inertia. |
| m2_liquidity    | discarded             | m2_liquidity decide to buy TQQQ (risk on) or SHV (risk off) based on macro economic data: M2 money supply, employment, prices, etc. Opposite to macro_risk, its decisions are based on long term indicators not so frequently updated. Hence it is less sensible to day 2 day volatility. However could be risky if the stock drops durably. Long term strategy. Quite analogous to macro_risk on a longer timeframe. |
| news_binary    | discarded          |  Strategy based on news sentiment analysis. The strategy buy QQQ or SPY if market regime is bullish. Defensive ETF otherwise. This strategy is more stable than benchmark SPY but also generate a lesser yield and globally underperform SPY and QQQ. |
| cross sectional momentum (V5) | retained  | Best candidate so far: robut over a 10 year window and good metrics. Very sensitive to momentum. So either very high whne momentum is there or very low. |
| opening range breakout | under study | Every day, select candidates breakouts. The rest of the day (or longer) detect sudden drops to sell the goods. This is a short term strategy with immediate earnings. |


## 📈 Strategies

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

#### 📈 `news_binary` — Alpaca News Built-in
| Field | Value |
|---|---|
| **File** | `strategies/news_binary/agent_news_binary.py` |
| **Model** | from `LLM_MODEL` in the env file |
| **Agent** | `news_binary` |
| **Tools** | `PrebuiltTools.all(strategy)` + `news_tools(strategy)` (`search_news`) |
| **Asset universe** | SPY, QQQ, DIA, IWM + defensive ETF |
| **Agent frequency** | every 1h |
| **Trading modes** | backtest, paper |
| **Benchmark** | SPY |

News-driven trading through the framework's `search_news` tool (broker-agnostic `NewsProvider`; works in backtests, gated on the simulated clock). The agent scans broad-market headlines with `search_news`, reads the most relevant article in full (article content is capped), then holds SPY/QQQ when the regime is bullish or the defensive ETF (SHV) when it is negative or unclear. It uses the framework's memory tools to record decisions and its regime thesis, and compares article timestamps against the simulated datetime.


#### 📈 `cross_momentum` — Cross Sectional Momentum (Final Version)

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

#### Model pick

##### Last benchmark result

┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric               ┃ GLM-4.7-Flash ┃ Gpt-OSS-20b ┃ Qwen3-30B-Thinking ┃ Qwen3.6-35B-A3B-AWQ ┃ Qwen3.6-27B-AWQ (winner) ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Overall score        │           64% │         57% │                65% │                 79% │                      94% │
│   Reasoning          │           52% │         60% │                44% │                 64% │                      88% │
│   Memory             │           65% │         55% │                55% │                 75% │                     100% │
│   Workflow           │           53% │         47% │                67% │                 80% │                      93% │
│   Tools              │           87% │         67% │                93% │                 97% │                      97% │
│ Runs passed          │         60/90 │       53/90 │              60/90 │               72/90 │                    85/90 │
│ Text tool calls      │             0 │           0 │                  8 │                   0 │                        0 │
│ Avg tool calls / run │          3.24 │        4.19 │               2.84 │                3.62 │                     3.51 │
│ Median run time      │         4.9 s │       4.9 s │             16.6 s │                30 s │                   29.7 s │
│ Tokens/s (median)    │         131.7 │         195 │              197.7 │                28.9 │                     44.5 │
│ Timeouts / errors    │           0/0 │         0/0 │                0/0 │                 0/0 │                      1/0 │
└──────────────────────┴───────────────┴─────────────┴────────────────────┴─────────────────────┴──────────────────────────┘
                                   Per-scenario results (passed/runs, mean partial score)                                    
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Scenario                       ┃ GLM-4.7-Flash ┃ Gpt-OSS-20b ┃ Qwen3-30B-Thinking ┃ Qwen3.6-35B-A3B-AWQ ┃ Qwen3.6-27B-AWQ ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ reasoning.conflicting_signals  │     1/5 (72%) │   1/5 (72%) │          0/5 (56%) │          5/5 (100%) │      5/5 (100%) │
│ reasoning.headline_trap        │    5/5 (100%) │   3/5 (90%) │          0/5 (60%) │          5/5 (100%) │      5/5 (100%) │
│ reasoning.insufficient_cash    │     3/5 (90%) │   3/5 (90%) │         5/5 (100%) │           0/5 (75%) │       3/5 (90%) │
│ reasoning.position_sizing      │     4/5 (87%) │  5/5 (100%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ reasoning.rsi_signal           │     0/5 (80%) │   3/5 (88%) │          1/5 (80%) │           1/5 (84%) │       4/5 (96%) │
│ memory.cross_session_rule      │     3/5 (90%) │  5/5 (100%) │          3/5 (90%) │          5/5 (100%) │      5/5 (100%) │
│ memory.profit_take_rule        │     1/5 (60%) │   0/5 (63%) │          0/5 (50%) │           0/5 (50%) │      5/5 (100%) │
│ memory.seeded_lesson           │    5/5 (100%) │   2/5 (76%) │          3/5 (76%) │          5/5 (100%) │      5/5 (100%) │
│ memory.thesis_lifecycle        │     4/5 (92%) │   4/5 (92%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ workflow.news_no_trade         │     4/5 (94%) │   4/5 (97%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ workflow.news_trade            │     4/5 (92%) │   2/5 (76%) │          1/5 (72%) │           4/5 (96%) │      5/5 (100%) │
│ workflow.stop_loss_recall      │     0/5 (83%) │   1/5 (71%) │          4/5 (94%) │           3/5 (89%) │       4/5 (94%) │
│ tools.cancel_stale_limit       │     4/5 (86%) │   4/5 (91%) │         5/5 (100%) │          5/5 (100%) │       4/5 (80%) │
│ tools.close_half_loser         │    5/5 (100%) │   1/5 (77%) │          3/5 (83%) │           4/5 (97%) │      5/5 (100%) │
│ tools.error_recovery           │     3/5 (80%) │   4/5 (90%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ tools.forced_remember_decision │    5/5 (100%) │  5/5 (100%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ tools.indicator_params         │     4/5 (90%) │   3/5 (85%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
│ tools.limit_order              │    5/5 (100%) │   3/5 (80%) │         5/5 (100%) │          5/5 (100%) │      5/5 (100%) │
└────────────────────────────────┴───────────────┴─────────────┴────────────────────┴─────────────────────┴─────────────────┘

__Winner:__
__Qwen3.6-27B-AWQ__ is the clear winner on all aspects.
The only oddity is the latency comparable to Qwen3.6-35B-A3B-AWQ when the latter has 4Gb offloaded on the CPU and the former not.
However Qwen3.6-35B-A3B-AWQ is 35B with 3B MoE. Qwen3.6-27B-AWQ is 27B full dense model loading all parameters at every pass.

##### Benchmark result applied to news_binary
Ranking for news_binary, worst → best fit

┌───────────┬─────────────────────┬──────────┬────────┬───────────┬───────┬─────────┬──────────────────┐
│   Rank    │        Model        │ workflow │ memory │ reasoning │ tools │ overall │ median s / tok·s │
├───────────┼─────────────────────┼──────────┼────────┼───────────┼───────┼─────────┼──────────────────┤
│ 1 (worst) │ Qwen3-30B-Thinking  │ 0.50     │ 0.73   │ 0.73      │ 0.88  │ 0.71    │ 15.0s / 197      │
├───────────┼─────────────────────┼──────────┼────────┼───────────┼───────┼─────────┼──────────────────┤
│ 2         │ GLM-4.7-Flash       │ 0.30     │ 0.73   │ 0.80      │ 0.92  │ 0.69    │ 5.4s / 131       │
├───────────┼─────────────────────┼──────────┼────────┼───────────┼───────┼─────────┼──────────────────┤
│ 3         │ Gpt-OSS-20b         │ 0.80     │ 0.67   │ 0.73      │ 0.72  │ 0.73    │ 3.8s / 192       │
├───────────┼─────────────────────┼──────────┼────────┼───────────┼───────┼─────────┼──────────────────┤
│ 4 (best)  │ Qwen3.6-35B-A3B-AWQ │ 0.90     │ 0.87   │ 0.67      │ 0.92  │ 0.84    │ 30.8s / 29       │
└───────────┴─────────────────────┴──────────┴────────┴───────────┴───────┴─────────┴──────────────────┘

Reasoning, since raw category scores alone are misleading:

1. Qwen3-30B-Thinking — least fit. Fails the core workflow.news_trade cycle most often (1/5), and in one repeat it actually bought into the headline trap it was supposed to catch (bullish headline, bearish body — it submitted the buy anyway). It also wrote 7 tool calls as raw text instead of structured calls across the run, which in a live deployment means broken function-calling, not just a wrong decision. Best-in-class on thesis_lifecycle (5/5) doesn't offset a real bad trade + unreliable tool emission.

2. GLM-4.7-Flash. Excellent tool mechanics (0.92, zero text-tool-calls, zero errors) and good trap-avoidance (4/5), but it never fully passes news_trade (0/5) — the exact "confirmed bullish news → buy" leg the bot exists for. Failure modes alternate between missing a genuine buy signal entirely and, in two repeats, sizing an order that violates the cash constraint (quantity * price > cash). For a bot whose entire job is executing on binary news signals, failing the buy leg every time is close to disqualifying regardless of how clean its tool syntax is.

3. Gpt-OSS-20b. Best headline-trap avoidance (5/5, and its insufficient_cash "failures" were only a missing word in the final answer — it never actually oversized an order, unlike the other three). Fastest and most token-efficient. But weakest memory hygiene: it fails to close an invalidated thesis 4/5 times (memory.thesis_lifecycle) — i.e., after news arrives that should kill a thesis, it leaves it open, which is a real problem for a bot that runs repeatedly and is meant to carry state across sessions. It also skips get_positions/get_position before acting in close_half_loser in 2 of 5 runs, and had 2 outright HTTP 500 crashes from the vLLM server choking on gpt-oss's harmony "commentary channel" tokens — a serving-compatibility issue on vllm==0.30.0, worth flagging separately from model quality.

4. Qwen3.6-35B-A3B-AWQ — best fit. Dominant on the three axes that matter most for this exact strategy: workflow (0.90 — actually completes the news→memory→trade→remember_decision cycle), memory (0.87 — best cross-session rule/lesson/thesis handling), and tied-best tools (0.92). Trap avoidance is strong (4/5). Its one real weakness: it violates the cash constraint in insufficient_cash 4/5 times — worse than any other model on that specific check — and it's 4-8x slower (30.8s median, 29 tok/s) than the others, which matters for a news-reactive bot.

Caveat that applies regardless of pick

Every model violates the cash/position-sizing constraint at least once (GLM 1/5, Qwen-thinking 1/5, Qwen3.6 4/5, Gpt-OSS 0/5-actual). Sizing/cash checks are the one axis no model here can be trusted on unsupervised — this should be enforced as a hard deterministic clamp in trading-agent-framework's order-submission path (reject or resize any order where qty*price > cash), not left to the LLM's arithmetic. That removes qwen3.6's biggest liability and makes its overall lead (workflow + memory + tools) the deciding factor.

Recommendation: Qwen3.6-35B-A3B-AWQ for news_binary, provided you add a code-level pre-trade cash/sizing guard — its latency (~30s/decision) is likely acceptable since news-driven decisions aren't sub-second, but confirm that against your actual polling/latency budget.

##### Benchmark on 'news binary'
1. The strategy trails SPY with both models.

┌──────────────────┬──────────────┬────────┬──────────────┬────────────────┐
│                  │ Total return │ Sharpe │ Max drawdown │ Avg % invested │
├──────────────────┼──────────────┼────────┼──────────────┼────────────────┤
│ SPY buy-and-hold │ 12.3%        │ 1.31   │ −8.9%        │ 100%           │
├──────────────────┼──────────────┼────────┼──────────────┼────────────────┤
│ QQQ buy-and-hold │ 15.7%        │ —      │ —            │ 100%           │
├──────────────────┼──────────────┼────────┼──────────────┼────────────────┤
│ GLM-4.7-flash    │ 10.8%        │ 1.80   │ −3.0%        │ 50%            │
├──────────────────┼──────────────┼────────┼──────────────┼────────────────┤
│ Qwen3.6-35B-A3B  │ 6.6%         │ 0.71   │ −12.0%       │ 77%            │
└──────────────────┴──────────────┴────────┴──────────────┴────────────────┘

GLM's better Sharpe is mostly a side effect of being in cash half the time, not good calls. For example, it sat in cash through April, when SPY rose 10.5%. Part of that cash was a sizing problem: it often bought 1 SPY or 10 SHV (about $1k) on a $10k account.

2. The two runs are not a clean model comparison.
- The prompt changed between the runs. Four news_binary commits landed between them (f605a9f, 7b31394, dd192d1, 7f85c30). They added a "hawkish = bearish" definition, the line "staying in the current holding needs no signal at all", and a new retry prompt. Qwen ran on a different strategy than GLM did.
- One run per model. LLM runs aren't deterministic, and the two runs held the same asset on only 20% of days. A 4-point gap from a single path each can't be told apart from noise.

3. Where Qwen lost the ground. It churned QQQ: it sold after dips and bought back days later at similar or higher prices.
- January: bought at 625, sold at 606, bought at 619, sold at 600, bought at 606.
- July: bought at 718, sold at 703, bought at 702, sold at 694, bought at 690, sold at 670.

Its long QQQ hold from February to May (606 → 712) made about +$1.5k, and the churn gave it all back: realized QQQ profit for the whole run is about −$14. So the gap comes mostly from the prompt's buy/sell switching rules not stopping the churn, not from the model simply being worse. The rule to leave QQQ/SPY needs two bearish signals, but with a run every 3 hours two signals can pile up within a day.

4. Qwen actually ran the workflow better. It had 0 runs that needed the "no decisi117 for GLM, and it stayed more fully invested. That fits your LLM benchmarkranking. Those benchmarks measure reasoning and tool use, not trading P&L. Qwen was also about 8× slower per call (27s vs 3.5s on average).

## Supporting documentation:
* [Lumibot Agent](https://lumibot.lumiwealth.com/agents.html)
* [Lumibot observability](https://lumibot.lumiwealth.com/agents_observability.html)
* [Lumibot environment variables](https://lumibot.lumiwealth.com/environment_variables.html)
* [Broker configuration](https://lumibot.lumiwealth.com/deployment.html#alpaca-configuration)
* [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server?tab=readme-ov-file#claude-code-configuration)
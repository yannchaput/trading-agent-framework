# ray_dalio -- idea-meritocracy leveraged ETF team (design)

Date: 2026-11-20. Status: approved 2026-11-20.

## Intent

Port lumibot's "Ray Dalio idea meritocracy" AI trading team, **leveraged ETF variant**, into this framework with the
**same logic**, as a candidate for real money (paper first). Source of truth, MIT-licensed:
`lumibot/example_strategies/ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py` at Lumiwealth/lumibot commit
`38e23c72248dd1555387b1487ca7b676ef6e95ca` (page: lumibot.lumiwealth.com/agents_example_ray_dalio_idea_meritocracy.html).

Inspired by Ray Dalio's public writing; not affiliated with Ray Dalio or Bridgewater (same disclaimer as lumibot).

## Goals

- Same five agents, same prompts (verbatim, tool names swapped), same task prompts, same context, same 54-symbol
  universe (inverse ETFs included, SHV as the cash escape), `min_positions` 3, once a day, SPY benchmark.
- LangChain agents as nodes of a LangGraph graph; native LangGraph `StateGraph` + a plain `TypedDict` state.
- Reuse the framework: `Strategy`/executor (three modes), `AgentManager.create`, the existing news, macro, SEC,
  account, market-data and trading tools, `require_news_provider`, the backtest runner.

## Non-goals

- No believability scores, vote weights or learning: lumibot has none; the "meritocracy" is in the prompts.
- No weight caps, stops or drawdown limits in code (lumibot has none).
- No reuse of sector_pods code (`AgentNode`, `NodeRecorder`, submit tools, length gate, `Rebalancer`).
- No new classes or types beyond the strategy class and the state `TypedDict`.

## Anti-goals

- The trader placing an order it has not checked: lumibot blocks that in code, so must we (below).
- A backtest reading the future: every new data tool takes its cutoff from `strategy.clock.now()`.

## Approaches considered

- **A. Faithful port (chosen).** Add the two FRED tools lumibot's prompts call (`get_fred_latest`, `get_fred_snapshot`)
  and lumibot's order-readiness check; copy the prompts verbatim with tool names swapped.
- B. Prompts pointed at the existing `get_fred_series` only, readiness rule as text. Less code, but each analyst sees
  one series instead of a snapshot and nothing stops a blind order. Rejected: not the same logic, and the readiness
  gap is the riskiest for real money.
- C. Trader returns weights, code trades them via `Rebalancer`. Safer, but not lumibot's logic. Rejected by the user.

## Design

### Graph (`strategies/ray_dalio/`)

```
START -> growth --------\
START -> inflation ------> disagreement -> trader -> END
START -> liquidity -----/
```

State: `class DalioState(TypedDict, total=False)` with `date: str`, `growth: str`, `inflation: str`,
`liquidity: str`, `disagreement: str`. One writer per slot, so no reducers. Graph compiled once in `initialize`,
invoked in `on_trading_iteration` with `{"date": session date}`.

Each node is a small function: `self.agents[name].run(task_prompt, context=...)`, returns `{slot: result.output}`
(`output` stands in for lumibot's `.summary`). The trader node returns `{}`.

- The three analysts run in parallel. Lumibot calls them one after another, but none sees another's output, so the
  logic is identical. They only hold read-only tools, which is what the root rule allows on worker threads.
- `disagreement` and `trader` are single-node steps, so with `invoke()` and no `step_timeout` they run on the
  strategy thread: the trader's orders follow the root threading rule. Do not add streaming, `subgraphs=True` or a
  `step_timeout` (they would move the trader to a worker thread).

### Agents (created once in `initialize` via `self.agents.create`, run by the nodes)

| Agent | Tools | Writes |
|---|---|---|
| `growth_agent`, `inflation_agent`, `debt_liquidity_agent`, `thoughtful_disagreement` | macro (`get_fred_series`, `get_fred_latest`, `get_fred_snapshot`), `search_news`, fundamentals (SEC), market data, account, indicators -- each agent its OWN tool instances | its slot |
| `trader` | the same read-only tools + trading tools, with the readiness guard | orders |

Lumibot's `allow_trading=False` = no trading tools. Model: the `LLM_*` env (lumibot used `gpt-6-luna`,
`reasoning_effort="high"` -> set `LLM_REASONING_EFFORT=high` in the env file). One `news_tools(self)` /
`macro_tools(self)` call per agent: `news_tools` keeps a per-instance memo of the latest run id only.

### Prompts (`prompts.py`)

Verbatim copies of lumibot's `ORDER_READINESS_RULE`, `DATA_USAGE_RULE`, `LEVERAGE_RULE`, the five system prompts
and the five task prompts, with an MIT attribution comment. Substitutions only:

| lumibot | here |
|---|---|
| `alpaca_news` | `search_news` |
| `get_fred_snapshot`, `get_fred_latest`, `get_fred_series` | same names |
| `list_fred_series` | dropped from `DATA_USAGE_RULE` (no such tool; see calls) |
| `account_portfolio`, `account_positions`, `market_last_price`, `orders_submit_order` | `get_account_balance`, `get_positions`, `get_last_price`, `submit_order` |
| "LumiBot rejects blind orders with ORDER_READINESS_REQUIRED" | "blind orders are rejected with ORDER_READINESS_REQUIRED" |

Context passed to every run: `date`, `universe`, `min_positions`, `data_expectation`, `leverage_expectation` (lumibot's
strings), plus `growth`/`inflation`/`liquidity` for `disagreement`, and those plus `disagreement` for `trader`.
Lumibot's `data_tool_validation_run_id` (an id of its own smoke test) is dropped.

### New framework tools (`agents/tools/macro.py`, returned by `macro_tools` after `get_fred_series`)

- `get_fred_latest(series_id)`: the last observation known on `strategy.clock.now().date()` (`_fetch_series` with
  the same vintage pinning), `{"series_id", "as_of", "date", "value"}` or `{"error"}`.
- `get_fred_snapshot(series_ids)`: list or comma-separated string (lumibot: "such as FEDFUNDS,DGS10,CPIAUCSL,UNRATE");
  one `get_fred_latest` row per id, an id that fails gets its own `{"series_id", "error"}` row. Cap the count
  (e.g. 10) so a payload stays small for the local model.

Existing `macro_tools` users (sector_pods' pods) gain two tools; their prompts don't mention them.

### Order-readiness guard (`strategies/ray_dalio/readiness.py`, one function)

Mirror of lumibot's `ORDER_READINESS_REQUIRED` (lumibot `components/agents/builtins.py`), in the shape of
`news_builtin/grounding.require_search_news_before`: `require_readiness_before_orders(tools) -> tools`. Per agent run
(`current_run_id()`), `submit_order(symbol, ...)` is refused with `{"error": "ORDER_READINESS_REQUIRED: ..."}`
(naming what is missing) unless, in this run, `get_account_balance` and `get_positions` were called after the last
accepted order (or at all, before the first) and `get_last_price` was called for that symbol. Other tools pass
through unchanged. A refusal is a tool result, so the model can make the calls and retry.

### Strategy (`strategies/ray_dalio/agent_ray_dalio.py`)

- `RayDalioStrategy(Strategy)`, registered as `Strategies.RAY_DALIO` (`"ray_dalio"`).
- `initialize`: `sleeptime = "1D"`; `require_news_provider`; create the five agents; build and compile the graph.
- `on_trading_iteration`: invoke the graph; log each agent's output (`log_info`). An agent error propagates: the
  executor logs it and runs again next session, as lumibot would.
- `run_backtesting` defaults: `YahooBacktestData` daily bars (the framework default, lumibot's backtest data),
  `timestep="day"`, 54 symbols + SPY preloaded, `agent_telemetry=True`, a short default window inside the news archive
  (it starts 2026-04-29). Costs: the framework's broker fees plus 5 bps slippage per side (user's call, in place of
  lumibot's 0.1% `percent_fee`), which is already `Strategy.backtesting_slippage`'s default: nothing to set.
- Iterations start at the market open (`iteration_start_time` default), as lumibot's `"1D"` does.
- Env: `env/.env.ray_dalio.<mode>`: `LLM_*`, `ALPACA_NEWS_*`, `FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`, broker keys.

### Tests (no network, hand-written fakes)

- Graph: with fake agent handles, each analyst gets lumibot's task/context; `disagreement` gets the three outputs;
  `trader` gets all four; runs once per iteration.
- Readiness guard: refused without each check, accepted with all, re-check required after an order, per-run reset,
  a different symbol's price does not count.
- `get_fred_latest` / `get_fred_snapshot` with a fake FRED client: cutoff = clock date, comma string and list, bad id
  row, cap.
- Factory: `ray_dalio` builds the strategy.

### Docs

`strategies/ray_dalio/CLAUDE.md`, the index in `strategies/CLAUDE.md` and the root `CLAUDE.md` list, the two FRED tools
in `agents/CLAUDE.md`.

## Calls made (not discussed; check these)

1. Agents are created once in `initialize` and the nodes run them: `AgentManager.create` raises on a duplicate name,
   so creating inside the node would fail from the second day.
2. Analysts in parallel (logic unchanged, faster).
3. `list_fred_series` dropped instead of building a FRED search tool. Two lumibot-only phrases go too: "Do not rely
   on a custom public CSV FRED helper." (`DATA_USAGE_RULE`) and "smoke tests inspect agent_detail"
   (`data_expectation`, now "the run log records the actual tool calls").
4. Readiness guard local to this strategy, not added to every strategy's `submit_order`.
5. The repo rule "size agent buys against the SMALLER of `buying_power` and `cash` + same-run sell proceeds"
   (`strategies/CLAUDE.md`) is added to the trader's task prompt: it's a project rule for every agent that buys, and on
   a margin account lumibot's prompt would let the trader lever 3x ETFs further.
6. Analysts also get the read-only market-data, account and indicator tools, as lumibot's built-ins do.
7. Lumibot's 0.1% fee per side is replaced by 5 bps slippage on top of the framework's real broker fees (5 bps is the
   user's choice and the framework default, so no code).

## Risks to know

- Run it on an account holding only this strategy: the trader sees and may sell every position.
- Some of the 54 tickers may be delisted or not tradable at Alpaca; the trader then sees a refused order and adapts.
  Not checked yet.
- A backtest before 2026-04-29 has no news (Alpaca archive).
- Lumibot published no backtest of this variant. Daily rebalancing of 3x and inverse ETFs adds volatility decay and fees.

## Left to the builder

File split inside `strategies/ray_dalio/`, the snapshot cap value, the default backtest window, log wording.

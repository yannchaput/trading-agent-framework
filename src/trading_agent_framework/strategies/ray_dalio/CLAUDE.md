# strategies/ray_dalio/ -- ray_dalio leveraged ETF team

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

An LLM team (growth, inflation and debt/liquidity analysts, a thoughtful-disagreement round whose basket plain code trades) that holds at least three leveraged ETFs (or SHV) once a session. Adapted from LumiBot's `ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py` (MIT, commit `38e23c72248dd1555387b1487ca7b676ef6e95ca`) with this framework's tool names; not affiliated with Ray Dalio or Bridgewater. Specs: `docs/superpowers/specs/2026-11-20-ray-dalio-meritocracy-design.md` and `2026-11-22-ray-dalio-code-executes-basket-design.md` (the trader agent was replaced by code); plans: the matching files in `docs/superpowers/plans/`. Registered as `"ray_dalio"`.

## Architecture

- `agent_ray_dalio.py` -- `RayDalioStrategy`, `DalioState`, the LangGraph graph (`_build_graph`). `prompts.py` -- `UNIVERSE` (54 symbols: 53 leveraged funds + SHV, each mapped to its signed daily leverage, what it tracks and its issuer; `CONTEXT["universe"]` hands that table to every agent so they stop guessing funds from memory: the first backtest called TMF 2x and TWM a 3x S&P fund, and held UGL with its inverse GLL), the system/task prompts and `CONTEXT`. `basket.py` -- `basket_problem` (the checks) and `basket_tool` (the `submit_basket` tool).
- Graph: `START -> growth_agent | inflation_agent | debt_liquidity_agent -> thoughtful_disagreement -> rebalance -> END`, one run per daily session (`sleeptime = "1D"`). Node names are the agent names (a node cannot share a name with a state slot); `rebalance` is plain code, not an agent.
- Agents are created once in `initialize` (`AgentManager.create` rejects duplicate names); the agent nodes only run them. Each agent gets its own tool instances (`news_tools` keeps a per-instance run memo).

| Agent | Tools |
|---|---|
| the 3 analysts | `macro_tools` (`get_fred_*`), `news_tools`, `fundamentals_tools`, `market_data_tools`, `account_tools`, `indicator_tools` (read-only) |
| disagreement | the same, plus `submit_basket(weights, reason)` |

No agent can order. **Code trades the disagreement's basket** (why: in the first backtest the trader agent kept TMF in 50 of the 103 sessions whose basket left it out, and filled orders on only 64 of about 129 sessions). `submit_basket` refuses an unknown symbol, a weight not above 0 (NaN too), a stock weight of 0.03 or less (`MIN_STOCK_WEIGHT`, the rebalance band: the `Rebalancer` never buys a fund into an empty position for a target that small, so the book would silently hold less than submitted), fewer than 3 symbols (SHV counts as one) and non-SHV weights above 0.98 (the framework keeps 2% cash; weights are fractions, 0.35 = 35%), and the agent fixes and resubmits (an all-stock basket of 1.00 gets the 0.98 message; the fractions hint only shows when a weight is above 1); an accepted basket replaces the earlier one and is logged with its reason (`basket: {...} (reason)`). The tool carries `mutates_trading`, so `AgentManager` runs the disagreement agent's tool calls one at a time and two `submit_basket` calls in one model turn cannot interleave. The disagreement node empties the tool's dict before it runs and writes the accepted basket to `state["basket"]`. The `rebalance` node hands it to `target_portfolio` + `common/rebalancer.Rebalancer` (band 3%, smallest trade 0.5% of the book, sector_pods' values): holdings the basket leaves out are sold first, then the buys, and **SHV is the remainder** (its basket weight is not traded as such; the stocks' weights are). **No accepted basket = a warning and no order that session** (the book is held; a basket never carries over).

- **Threading:** the analysts run in parallel on worker threads and only have read-only tools. Disagreement and `rebalance` are single nodes run by plain `graph.invoke(...)`, so they run on the strategy thread, which is where the root rule wants order code. Never add streaming, `subgraphs=True` or `step_timeout`: that would send `rebalance` to a worker thread.
- An agent error propagates out of `on_trading_iteration`: the executor logs it and the next session runs again.

- Prompt substitutions beyond the tool renames (see the spec table): "official LumiBot built-in tools" -> "the built-in tools", "Alpaca News call/limit" -> "search_news call/limit", "FRED/Alpaca evidence" -> "FRED/news evidence".

## Env

`env/.env.ray_dalio.<mode>`: `LLM_*` (with `LLM_REASONING_EFFORT=high`), `ALPACA_NEWS_*` (required: it refuses to start without a news source), `FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`, plus the broker keys in paper/live.

## Gotchas

- **Run it on an account holding only this strategy:** the rebalance sells every position the basket leaves out.
- Alpaca news before 2026-04-29 is empty in backtests, so earlier windows run blind on news.
- The fund descriptions were checked against Yahoo on 2026-11-21. ERX/ERY, GUSH/DRIP and NUGT/DUST were 3x before 2020, so the table is wrong for earlier windows.
- The 54 tickers were taken from lumibot and not verified tradable; the default backtest window is `PredefinedWindow.BI_YEAR_LATEST` (about two years, so mostly before the news archive) on Yahoo daily bars.

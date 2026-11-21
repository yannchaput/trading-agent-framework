# strategies/ray_dalio/ -- ray_dalio leveraged ETF team

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

An LLM team (growth, inflation and debt/liquidity analysts, a thoughtful-disagreement round, a trader) that holds at least three leveraged ETFs (or SHV) once a session. Adapted from LumiBot's `ai_trading_team_ray_dalio_idea_meritocracy_leveraged.py` (MIT, commit `38e23c72248dd1555387b1487ca7b676ef6e95ca`) with this framework's tool names; not affiliated with Ray Dalio or Bridgewater. Spec: `docs/superpowers/specs/2026-11-20-ray-dalio-meritocracy-design.md`; plan: `docs/superpowers/plans/2026-11-20-ray-dalio-meritocracy.md`. Registered as `"ray_dalio"`.

## Architecture

- `agent_ray_dalio.py` -- `RayDalioStrategy`, `DalioState`, the LangGraph graph (`_build_graph`). `prompts.py` -- `UNIVERSE` (54 symbols: 53 leveraged ETFs + SHV), the system/task prompts and `CONTEXT`. `readiness.py` -- `require_readiness_before_orders`.
- Graph: `START -> growth_agent | inflation_agent | debt_liquidity_agent -> thoughtful_disagreement -> trader -> END`, one run per daily session (`sleeptime = "1D"`). Node names are the agent names (a node cannot share a name with a state slot).
- Agents are created once in `initialize` (`AgentManager.create` rejects duplicate names); nodes only run them. Each agent gets its own tool instances (`news_tools` keeps a per-instance run memo).

| Agent | Tools |
|---|---|
| the 3 analysts, disagreement | `macro_tools` (`get_fred_*`), `news_tools`, `fundamentals_tools`, `market_data_tools`, `account_tools`, `indicator_tools` |
| trader | the same, plus `submit_order`, `cancel_order`, `cancel_open_orders`, `get_orders`, `get_order` (no `close_position` / `sell_all`: they order without the guard; it sells with `submit_order(side="sell")`), all wrapped by `require_readiness_before_orders` |

- **Readiness rule (from lumibot):** within one agent run, `submit_order` is refused with `ORDER_READINESS_REQUIRED` unless `get_account_balance` and `get_positions` were read since the last accepted order and `get_last_price` for that symbol this run. Outside an agent run nothing is gated.
- **Threading:** the analysts run in parallel on worker threads and only have read-only tools. Disagreement and trader are single nodes run by plain `graph.invoke(...)`, so the trader (and its orders) runs on the strategy thread. Never add streaming, `subgraphs=True` or `step_timeout`: that would send the trader to a worker thread.
- An agent error propagates out of `on_trading_iteration`: the executor logs it and the next session runs again.

- Prompt substitutions beyond the tool renames (see the spec table): "official LumiBot built-in tools" -> "the built-in tools", "Alpaca News call/limit" -> "search_news call/limit", "FRED/Alpaca evidence" -> "FRED/news evidence".

## Env

`env/.env.ray_dalio.<mode>`: `LLM_*` (with `LLM_REASONING_EFFORT=high`), `ALPACA_NEWS_*` (required: it refuses to start without a news source), `FRED_API_KEY`, `SEC_EDGAR_USER_AGENT`, plus the broker keys in paper/live.

## Gotchas

- **Run it on an account holding only this strategy:** the trader may sell any position.
- Alpaca news before 2026-04-29 is empty in backtests, so earlier windows run blind on news.
- The 54 tickers were taken from lumibot and not verified tradable; the default backtest window is `PredefinedWindow.BI_YEAR_LATEST` (about two years, so mostly before the news archive) on Yahoo daily bars.

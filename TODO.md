# TODOS

## General

* Migrate projects to the 2nd hard drive (with the models)
* ~~Enable IBKR broker~~
* Build a log "purge" batch for logs more than 1 month old (except backstesting). Same for 'cache' (inclusing backtesting)
* ~~Cache LLM calls during backtesting so reruns against the same period are deterministic and free (design spec section 1.3 -- currently every backtest rerun calls the live LLM again)~~ => Too complex
* ~~Add agent call telemetry (token counts, latency) to backtesting settings.json and `llm_stats.sqlite`, shown in the dashboard's Parameters tab (no cache hits: LLM call caching was dropped as too complex)~~
* ~~Create a batch to run all the smoke tests altogether~~
* ~~Add cash available during backtesting~~
* ~~Update pyproject.toml to make dashboard an "extra" dependency~~
* ~~Add a gloabl factory/builder for strategy in main.py~~
* Log P&L every day in a sqllite database for the dashboard to display it: for backtest, paper and live
* Add a skill to review, challenge and validate the design of a strategy

## BUGS

* ~~Review `uv check` errors and fix or silent them~~
* ~~Replace every instance variable `self.` with `self.vars` to avoid namespace collisions~~
* Update tools comments with parameters

## Strategies to test

* [AI investment comitee](https://lumibot.lumiwealth.com/agents_investment_committee.html)

## PEA Manager

* Trouver un skills ou MCP permettant de remplacer EODHD quand son credit est epuisé pour le scan des tickers

## Tests

* Test Typesafe.AI sur news binary.

## Evolutions

### Dashboard

* ~~Add a column for the broker~~
* ~~REplace VIX, ADX etc with a "regime" indicator: 1 =bullish, neutral =0, bearish = -1~~

### Cross sectional momentum

* ~~Add a market regime filter to cross sectional momentum strategy.if bearish -> defensive else cross momentum.~~
* ~~When a nex position is taken, put a stop order at 1 x ATR (confirm threshold)~~ => Poor results
* ~~Restrain the number of positions to a hard cap of 20: do not buy if more than 20~~ => Poor results
* ~~Compute the residual volatility compared to the market (cf chatgpt thread)~~

### News sentiment analysis

* ~~Find the root cause of having the same decision to remember called several times in the same iteration: help with log and the memory database.~~
* ~~Investigate "current_regime": None and defensive posture~~
* ~~Add a log in executor to log the execution hour : usefull for intraday.~~
* ~~Log the system prompt in 'initialize' method~~
* ~~Fix prompt to avoid chineese character with GLM models~~

### LLM benchmark

* ~~Log a frontmatter before each model benchmark~~
* Update with scenarios from bill ackman and vwap pullback continuation

### VWAP Pullback continuation

* ~~Add indicators on equity curve~~
* ~~Remove entry agent: it is effectless~~
* ~~Change agentic architecture for : Make the agent "agentic" in the right place (cf chatgpt)~~ => agents removed, the strategy is code only

### Bill Ackman

* ~~Update claude.md with exempt_tools and budget_tools~~
* ~~Update dashboard to display tempearature~~
* ~~Log output of each agent to understand what was their decision~~

---

## Archives

### MIGRATION

* ~~Alpaca broker~~
* ~~Orchestration framework (scheduling, lifecycle,...)~~
* ~~Data and accountig tools~~
* ~~Agentic framework - langchain foundation~~
* ~~Tools~~
* ~~Memory~~
* ~~Account configuration (no short sell, no margin trading)~~
* ~~Backtesting (vectorbt)~~
* ~~Cache management~~ => not needed
* ~~dashboard~~
* ~~Fix `Bar.empty`, `bars.pandas_df`.~~
* ~~Add warmup_trading_days to AlpacaBacktestDataSource~~

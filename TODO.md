# TODOS

## General
* Migrate projects to the 2nd hard dreive (with the models)
* Enable IBKR broker
* Build a log "purge" batch for logs more than 1 month old (except backstesting). Same for 'cache' (inclusing backtesting)
* Compose strategies, e.g news sentiment analysis + momentum
* Use "https://data.alpaca.markets/v1beta1/screener/stocks/movers" to screen values from NASDAQ and explore potentials.
* ~~Cache LLM calls during backtesting so reruns against the same period are deterministic and free (design spec section 1.3 -- currently every backtest rerun calls the live LLM again)~~ => Too complex
* ~~Add agent call telemetry (token counts, latency) to backtesting settings.json and `llm_stats.sqlite`, shown in the dashboard's Parameters tab (no cache hits: LLM call caching was dropped as too complex)~~
* Create a batch to run all the smoke tests altogether
* Add cash available during backtesting

## MIGRATION
* ~~Alpaca broker~~
* ~~Orchestration framework (scheduling, lifecycle,...)~~
* ~~Data and accountig tools~~
* ~~Agentic framework - langchain foundation~~
* ~~Tools~~
* ~~Memory~~
* ~~Account configuration (no short sell, no margin trading)~~
* ~~Backtesting (vectorbt)~~
* Cache management
* ~~dashboard~~
* ~~Fix `Bar.empty`, `bars.pandas_df`.~~
* ~~Add warmup_trading_days to AlpacaBacktestDataSource~~


## BUGS
* Review `uv check` errors and fix or silent them
* ~~Replace every instance variable `self.` with `self.vars` to avoid namespace collisions~~


## Strategies to test

* [AI investment comitee](https://lumibot.lumiwealth.com/agents_investment_committee.html)

## Evolutions

### Cross sectional momentum
* Add a market regime filter to cross sectional momentum strategy.if bearish -> defensive else cross momentum.
* ~~When a nex position is taken, put a stop order at 1 x ATR (confirm threshold)~~ => Poor results
* ~~Restrain the number of positions to a hard cap of 20: do not buy if more than 20~~ => Poor results
* Compute the residual volatility compared to the market (cf chatgpt thread)

### Opening range breakout
* ~~Override LUMIBOT_CACHE_FOLDER~~
* ~~Increase number of positions to 20~~
* Clean up the "think" tags leaving only the result in te logs tha gent produces
* Test various combination of "volume_confirm_threshold" : 0,75 -> 1.0 -> 0.5 -> 0.0
* Improve code regarding the universe management (too many entry points) => refactor in a factory
* Write unit test (including new helpers methods)
* Fix indicators not showing up
* Add more indicators than portfolio value (e.g SMA 200)
* Automatize deletion of memory sqllite db before running a backtest

### News sentiment analysis
* Find the root cause of having the same decision to remember called several times in the same iteration: help with log and the memory database.
* ~~Investigate "current_regime": None and defensive posture~~
* Add a log in executor to log the execution hour : usefull for intraday.
* ~~Log the system prompt in 'initialize' method~~

### VWAP + idiosyncratic mean reversion

# TODOS

## General
* Migrate projects to the 2nd hard dreive (with the models)
* Enable IBKR broker
* Make a factory out of the various cross_momentum strategies to select one
* Build a log "purge" batch for logs more than 1 month old (except backstesting). Same for 'cache' (inclusing backtesting)
* Compose strategies, e.g news sentiment analysis + momentum
* Use "https://data.alpaca.markets/v1beta1/screener/stocks/movers" to screen values from NASDAQ and explore potentials.

## MIGRATION
* ~~Alpaca broker~~
* ~~Orchestration framework (scheduling, lifecycle,...)~~
* ~~Data and accountig tools~~
* ~~Agentic framework - langchain foundation~~
* Tools
* ~~Memory~~
* Account configuration (no short sell, no margin trading)
* Backtesting (vectorbt)
* Cache management
* dashboard
* Check Executor:_initialize method. All the parameters are supposed to be assigned automatically to Strategy parameters instance. But they are not. Is it a bug?

## BUGS
* Review `uv check` errors and fix or silent them
* Investigate why indicators are not logged any more in "backtesting"
* Check if wrapping order account methods are still needed in "WrappingStrategy"
* Replace every instance variable `self.` with `self.vars` to avoid namespace collisions


## Strategies to test

* [AI investment comitee](https://lumibot.lumiwealth.com/agents_investment_committee.html)

## Evolutions

### Cross sectional momentum
* Add a market regime filter to cross sectional momentum strategy.if bearish -> defensive else cross momentum.
* When a nex position is taken, put a stop order at 1 x ATR (confirm threshold)
* Too much sector correlation, add a sector isolation strategy

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
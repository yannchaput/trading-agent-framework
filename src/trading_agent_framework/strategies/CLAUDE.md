# strategies/ -- concrete strategies

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- Concrete strategies, one directory each; `utils/strategy_factory.py` registers them.
- `utils/strategy_factory.py`'s `Strategies` enum lists the strategy names and `build_strategy(strategy, broker, mode)` builds one (`Strategy | None`: `None`, with a warning, when a universe strategy has no universe file).

## Rules that live elsewhere (apply here too)

- **Agent buys are sized from `spendable`** (`get_account_balance`, `Strategy.spendable_cash()`): the smaller of `buying_power` and `cash` after open orders, less 2% of the book. Prompts say "never buy for more than `spendable`"; `Strategy.submit_order` refuses a bigger buy with `InsufficientFundsError` naming the max quantity. Why `buying_power` is netted in backtests: `backtesting/CLAUDE.md` ("In backtesting, `get_account()` nets only `buying_power`").
- **`Rebalancer`, `TradeDesk`, cross_momentum, earnings_drift (`Desk.max_quantity`) and vwap_pullback (`risk.plan_entry(spendable=...)`) size buys from `Strategy.spendable_cash()`** (account read first, open orders second, so no fill counts twice). cross_momentum, earnings_drift and vwap_pullback keep no cash buffer of their own on top of the framework's 2%. The Rebalancer, cross_momentum and earnings_drift send every buy through `common/rebalancer.submit_buy`: one refused after a price uptick is retried once at its `max_quantity`. The Rebalancer's invariant and its one gap: `strategies/common/CLAUDE.md`. A Rebalancer strategy's `cash_buffer` (bull_bear, bill_ackman, sector_pods) must be at least `BUY_BUFFER` (2%): it shrinks the target weights, and below the framework's buffer the targets would always be short of money and the trims would sell stocks to chase buys that cannot fit.
- **A strategy's own rules are in its directory's `CLAUDE.md`**: `bear_put_spread/` (research only, not in `Strategies`), `bill_ackman/`, `bull_bear/`, `congress_trades/`, `cross_momentum/`, `earnings_drift/`, `insider_trades/`, `news_builtin/`, `ray_dalio/`, `sector_pods/`, `vwap_pullback/`.

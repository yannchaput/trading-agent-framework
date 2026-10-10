# strategies/ -- concrete strategies

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Paths are relative to `src/trading_agent_framework/` unless they start with `tests/`, `scripts/` or `docs/`.

## Architecture

- Concrete strategies, one directory each; `utils/strategy_factory.py` registers them.
- `utils/strategy_factory.py`'s `Strategies` enum lists the strategy names and `build_strategy(strategy, broker, mode)` builds one (`Strategy | None`: `None`, with a warning, when a universe strategy has no universe file).

## Rules that live elsewhere (apply here too)

- **Size agent buys against the SMALLER of `buying_power` and `cash` + the proceeds of the sells submitted in the same run** -- never `min(cash, buying_power)`, and never `buying_power` alone. Why and the backtest netting rule: `backtesting/CLAUDE.md` ("In backtesting, `get_account()` nets only `buying_power`").
- **`Rebalancer` reads the account ONCE**, before positions, open orders and any order; `cash` is never read again. The invariant and its one gap: `strategies/common/CLAUDE.md`.
- **A strategy's own rules are in its directory's `CLAUDE.md`**: `bill_ackman/`, `bull_bear/`, `congress_trades/`, `cross_momentum/`, `earnings_drift/`, `news_builtin/`, `vwap_pullback/`.

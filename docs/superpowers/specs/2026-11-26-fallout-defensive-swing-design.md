# fallout_defensive_swing -- design (short)

## Why

A tactical strategy for a short, volatile period (VIX ~28, tech-led sell-off, Nov 2026), on the user's $768 Tastytrade
cash account. Goal: small, frequent locked-in gains with a small drawdown, few orders (Tastytrade charges $0.10 per
fractional order), until the user switches to another strategy. The user checks the result daily and stops it by hand.

## Study (scratchpad scripts, not in the repo)

- 63 ETFs screened; mix **B = XLP + XLU, 50/50** chosen by the user (A = + IEF also studied).
- Rule tested on Yahoo hourly bars (Dec 2023 - Nov 2026), Tastytrade fees, 2 bps slippage, cash-account settlement:
  - Nov 3-25 2026: buy-back at any dip +$16.99; at a 1% dip **+$40.34, max DD -0.8%**, ends in cash; hold B +$12.72.
  - 3 years: 1% dip +4.7%, max DD -3.5%, **invested only 5% of the time** (after a sale in a rising market the price
    never comes back 1% lower). Hold B +33%, DD -14.7%.
- Accepted by the user: rare losses beyond $10 (no stop-loss: a 2% stop made every test worse); possibly sitting in cash
  for a long time after a sale (opportunity cost, not a loss).

## The rule

Every hour during the session (ticks at 10:30, 11:30, ..., 15:30 ET; `sleeptime = "60M"`,
`iteration_start_time = 10:30`):

1. **No position, never sold yet:** buy the mix: `spendable_cash()` split equally, one fractional market order per ETF.
2. **Holding:** value = sum(quantity x last price) - estimated sell fees (`brokers/fees.py`, Tastytrade). If
   value >= cost basis x (1 + `take_profit`) and the sale is allowed (settlement, below), sell every ETF of the mix
   (market orders, full quantities). Cost basis = cash spent on the last buy, fees included.
3. **In cash after a sale:** buy back when mean(last price / sale price) over the mix's ETFs <= 1 - `buyback_dip`.
4. Otherwise do nothing.

**Cash-account settlement:** a sale's proceeds settle the next trading day. A position bought with them must not be
sold before then (good-faith violation). Rule: a sell is allowed only on a later session than the last sale.

Parameters (`parameters` dict): `symbols = ["XLP", "XLU"]`, `take_profit = 0.01`, `buyback_dip = 0.01`, `budget`
(backtests only).

## Shape

- `strategies/fallout_defensive_swing/`: `agent_fallout_defensive_swing.py` (the `Strategy` subclass: hooks, orders, state I/O) and
  `rule.py` (**pure** `decide(state, prices, sell_fees, session) -> Action`: `BUY`, `SELL` or `HOLD`), plus a
  `CLAUDE.md`.
- State in a `JsonStateStore` (`strategies/common/json_state.py`), one file per strategy and mode: phase
  (`fresh`/`holding`/`cash`), cost basis, per-ETF sale prices, last sale session. Saved after each filled buy or sale
  (phase changes only once the orders are filled: `wait_for_order_execution`).
- Partial failures: if one buy fills and the other does not, the phase is `holding` with the basis actually spent (the
  next tick does not top up). If one sale fails, the phase stays `holding` and the next allowed tick sells what is left.
- On start: if the account holds the mix but the state says cash/fresh (or the reverse), log a warning and adopt what
  the broker shows (basis = current market value). The account is dedicated to this strategy.
- Registered in `Strategies` (`FALLOUT_DEFENSIVE_SWING`) and `build_strategy`; env files
  `env/.env.fallout_defensive_swing.{live,backtesting}` created by the user (the README documents the naming).
- No LLM, no agent, no news. The market regime is logged by the framework as for every strategy; nothing trades on it.

## Approaches

- **Chosen:** a small standalone strategy with a pure decision function -- testable without a broker, same
  paper/live/backtest paths as every other strategy, one place for the settlement rule.
- Rejected: the `Rebalancer` (trades toward target weights inside a band; it has no "all in / all out" phases or sale
  memory); an external script outside the framework (no paper mode, no backtest, no logs or dashboard).

## Calls I made (check these)

- Name `fallout_defensive_swing`.
- Ticks at 10:30-15:30 (6 per session), matching the study's hourly bar closes; no tick at the open (9:30 fills and
  fractional orders at the open are unverified) and none at 16:00 (market orders would not fill).
- The dip test uses each ETF's own sale price (mean of ratios), a simpler equivalent of the study's index.
- Buys use `spendable_cash()`, so ~2% of the book stays as cash (framework `BUY_BUFFER`); the study invested 100%.
- Stopping the process leaves positions as they are; the user sells by hand if wanted (no "flatten on stop").
- The $5 Tastytrade fractional minimum is not handled: each order is about half the book (~$380).

## Not included

Stop-loss, re-entry timeout, mixes other than a list of symbols with equal weights, automatic switch to another strategy.

## Tests

`tests/strategies/fallout_defensive_swing/`: `decide` (first buy; hold below target; sell at target net of fees; no sell on the
sale's own session; no buy-back above the dip; buy-back at the dip); the strategy with the fake broker (buy splits cash
equally, sell closes every ETF, state saved only after fills, restart adopts broker positions). A backtest on minute bars,
Nov 3-25 2026, against the study's +$40.34 (expect close, not equal: next-bar fills, 5 bps default slippage, 2% buffer).

## README

A `fallout_defensive_swing` entry in the README's strategy list, with its **conditions of applicability**: a short,
volatile, trendless market (study: VIX ~28, Nov 2026); a dedicated cash account; daily human review and a manual stop
when the turbulence ends. Not for a rising market (it sits in cash after a sale: 3-year study invested 5% of the time)
nor a long fall (no stop-loss: it holds until +1%, the study's longest hold was 232 days). Plus its run command and the
backtest window.

## Rollout (user's decisions)

- **No Alpaca paper run**: Alpaca paper does not reflect Tastytrade's fees or fills.
- **Backtest only Nov 3 - Nov 25 2026** (`backtesting_start`/`backtesting_end` defaults), with `BROKER=tastytrade` so the
  fee model is Tastytrade's.
- Then live on the Tastytrade cash account after one checked fractional buy and sale. Env files: `env/.env.fallout_defensive_swing.{live,backtesting}`.

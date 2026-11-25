# Tastytrade clearing fees in backtests -- design (short)

## Why

A backtest run with `BROKER=tastytrade` should pay what the live Tastytrade account pays. Today
`brokers/fees.py` charges Tastytrade no broker fee at all, only the regulatory fees (SEC, FINRA TAF, CAT).

## Source

Tastytrade "Commissions & Fees", last updated 2026-10-28 (tastytrade.com/commissions-and-fees):

- Stock & ETFs: $0 commission, "a clearing fee of $0.0008 per share + regulatory fee".
- Fractional Shares: $0 commission, "a clearing fee of $0.10 per transaction".
- Fee table: "Clearing Fees - Equities $0.0008 / share" and "Clearing Fees - Fractional Shares $0.10 / transaction"
  are separate lines.

## Change

In `brokers/fees.py`, the Tastytrade per-order charge (`_tastytrade_commission`), on buys and sells:

- whole-share quantity: `$0.0008 x shares`;
- any fractional quantity: `$0.10` flat, **instead of** the per-share fee (the whole order is billed as fractional).

Regulatory fees stay on top, as for every broker (SEC + FINRA TAF on sells, CAT on both sides); each side's
total is still rounded up to the cent.

## Approaches

- **Chosen:** both rates in the Tastytrade entry of `COMMISSIONS`, which already sees one order's share count.
- Rejected: a separate clearing-fee field in `TradeFees` (reports only sum buy and sell fees, nothing would read
  it); a fee class per broker (too much structure for two rates).

## Calls made

- $0.10 replaces the per-share fee for a fractional order (the user's reading of the schedule: two separate
  clearing-fee lines). Not confirmed by Tastytrade; a market-hours dry run (`fee-calculation`) or the first real
  fractional trade's confirmation can confirm it.
- A 4.5-share order counts as fractional as a whole (one Tastytrade order); the difference is under a cent.
- FINRA TAF kept although Tastytrade waives it 2026-10-01 to 2026-12-31: rates are dated constants and backtests
  mostly cover earlier periods.
- CAT kept: the schedule's "regulatory fee" covers it; tiny.
- Rates date (2026-10-28) noted in the module docstring and `backtesting/CLAUDE.md`.

## Not included

- Tastytrade's $5 minimum on fractional orders (an order rule, not a fee).
- The live Tastytrade broker (Tastytrade charges real fees itself).
- Env files: `env/.env.sector_pods.backtesting` keeps `BROKER=alpaca` until the user changes it; bull_bear,
  ray_dalio and insider_trades backtests already use `tastytrade` and start paying these fees.

## Tests

`tests/brokers/test_fees.py`: whole-share and fractional orders, buy and sell, exact cent amounts; the existing
Tastytrade test updated.

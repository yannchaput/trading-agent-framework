# cross_momentum: park de-risked capital in SHV instead of idle cash

Date: 2026-09-29 · Branch: `feature/cross-momentum-lowbeta-parking`

## Problem

cross_momentum's exposure legs (risk overlay, fast/slow vol targeting, combined via `min()`) scale the stock
target weights down to as little as 40%. Two things undercut them today:

1. `rebalance()` never trims a held position: a lower exposure only shrinks new buys, so the held book keeps
   its beta until names rotate out.
2. Whatever the legs remove sits in cash, which earns nothing in the backtest.

A previous experiment (beta-adjusted ranking + beta-targeted sizing + trim) regressed CAGR/Sharpe/Sortino for
little beta reduction and was discarded. That branch mixed the ranking change with the trim, so its result
does not say which one hurt. This change keeps the raw-momentum ranking untouched, which isolates the
trim-and-park effect.

## Goal

Keep the exposure legs' beta reduction real (trim held positions) and earn the T-bill rate on the de-risked
capital by parking it in SHV. Judged by the user's backtest against `logs/cross_momentum/backtesting/2026-09-28_120209_backtesting`
(CAGR 0.32, Sharpe 1.10, Sortino 1.54, beta 1.07, correlation 0.65, max drawdown −0.35).

Replaces behaviour outright (no enable flag). Unchanged: universe, filters, ranking, weights, exposure legs,
weekly Tuesday rebalance, rank-35 hysteresis, 5% `cash_buffer_pct`. No inverse ETFs, no other sleeve assets.

## 1. Sleeve target

Computed inside `rebalance()` on rebalance days only, after the exposure legs have scaled `target_weight`:

```
shv_target_value = max(0, pv * (1 - cash_buffer_pct) - Σ stock target values - hysteresis_value)
```

- `pv` = `self.portfolio_value` (as today).
- Σ stock target values = `pv * Σ target_weight` over the target list.
- `hysteresis_value` = market value of positions held only by hysteresis (not in the target, rank ≤
  `sell_rank_threshold`), valued at `quantity * get_last_price(symbol)` — the same price source the exit path
  uses today.

This parks both what the exposure legs remove and the weight left over when fewer than `top_n` names pass the
filters. The 5% reserve stays in cash (orders fill at a later open, plus fees).

## 2. Order sequencing in `rebalance()`

`_REBALANCE_BAND = 0.20` becomes a module constant (it replaces the inline `0.20` in the buy path).

**Phase 1 — sells**, all feeding `estimated_sell_proceeds`:

- The parking symbol is skipped by the exit/hysteresis loop (it is never ranked, so today's loop would sell it
  every week as "not ranked").
- Exits and hysteresis keeps: unchanged. Kept positions add to `hysteresis_value`.
- **Trim:** a target position whose `current_value > target_value * (1 + _REBALANCE_BAND)` is sold down to
  target: `fractional_qty((current_value - target_value) / price)`, where `price` is the target entry's price.
- **SHV sell:** if `shv_value > shv_target_value * (1 + _REBALANCE_BAND)`, sell
  `fractional_qty((shv_value - shv_target_value) / shv_price)`. With a zero target this sells the whole
  holding.

**Phase 2 — stock buys:** unchanged.

**Phase 3 — SHV buy:** if `shv_value < shv_target_value * (1 - _REBALANCE_BAND)`, buy
`fractional_qty(min(shv_target_value - shv_value, available_cash) / shv_price)`, where `available_cash` is
what Phase 2 left (the reserve is already excluded).

**Minimum trade:** a trim, SHV sell or SHV buy whose dollar size is below `min_trade_pct * pv` is skipped, so
per-order fees don't nibble the book weekly. Stock exits and stock buys keep their current rules.

**Missing SHV price** (`get_last_price` returns `None` or `0`): log a warning and place no SHV order this
week; trims and stock orders still go through.

Order failures follow the existing pattern: `log_error` / `log_warning`, carry on with the next order; a
failed sell adds nothing to `estimated_sell_proceeds`.

## 3. Config (`parameters.py`)

```python
"parking": {
    "symbol": "SHV",
    "min_trade_pct": 0.01,
},
```

## 4. Wiring

- `run_backtesting()` adds `Asset(symbol=parking symbol)` to `preload_assets` so the Yahoo source loads it with
  the universe.
- Class and module docstrings mention the SHV sleeve and the trim.
- Rebalance logs: one line per trim and per SHV order, plus one summary line
  `Parking: SHV target $X (current $Y)`.

Known and accepted:

- With `BROKER=ibkr` on an EU retail account, IBKR rejects US ETFs (PRIIPs, see `CLAUDE.md`): SHV orders fail,
  are logged, and the capital stays in cash. Alpaca is unaffected.
- The risk diagnostics report SHV as an "unknown sector" position.

## Testing (TDD, `tests/strategies/test_cross_momentum_rebalance.py`, hand-written `FakeStrategy`)

1. Idle cash is swept into SHV (all cash, no target stocks held beyond target → SHV buy ≈ pv·(1−buffer) − Σ targets).
2. An exposure drop trims an overweight target position to target and parks the proceeds in SHV.
3. An exposure rise sells SHV so the stock buys are funded (SHV sell precedes the buys; buys fit the cash).
4. SHV is never exited as unranked.
5. A trade below `min_trade_pct * pv` is skipped (trim and SHV).
6. A missing SHV price places no SHV order, logs a warning, and still places the stock orders.
7. Hysteresis holdings reduce the SHV target by their market value.
8. The existing cash-buffer tests still hold (the reserve is never spent on SHV).

# fallout_defensive_swing/ -- XLP + XLU take-profit / buy-back tactic (no LLM)

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Conditions of applicability: `README.md`. Design: `docs/superpowers/specs/2026-11-26-fallout-defensive-swing-design.md`.

- `rule.py` -- pure `decide(state, ...) -> Action`; `store.py` -- the `SwingState` JSON (`data/fallout_defensive_swing_state_<mode>.json`); `agent_fallout_defensive_swing.py` -- the hourly tick and the orders.
- **Settlement guard:** no sale unless `session > last_sale_session`. It is checked before `selling`, so a retried sale obeys it too. Never remove it: it prevents a good-faith violation on a cash account.
- **The saved state is corrected from the account on every tick.** Mix held but phase not holding: adopt it, basis = market value. Phase holding but nothing held: cash, sale prices = today's, `last_sale_session` = today (the sale may have been today, so nothing bought now is sold before tomorrow).
- **Orders:** waits are bounded (`ORDER_WAIT_SECONDS`), an unfinished order is canceled, nothing is saved while one is still active, and a tick with an active order does nothing. A buy that filled after a basis saved without it would look like a huge gain and be sold at once.
- Each buy is capped at a fresh `spendable_cash()`: the earlier buy fills above its sizing price.
- `selling=True` only when part of a sale filled; the rest then sells whatever the price. A sale that fills nothing changes no state.
- Fill prices come from `order.transactions`: the live brokers do not set `order.avg_fill_price`.
- Known limit: if one leg is sold by hand while holding, the basis still covers both, so the target is not reached; sell the other leg by hand too.
- **Do not tune the thresholds to a backtest.** +1% / 1% dip were picked on 3-25 November 2026; no stop-loss because a 2% stop did worse in every study run.

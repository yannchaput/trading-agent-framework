# fallout_defensive_swing/ -- XLP + XLU take-profit / buy-back tactic (no LLM)

> Nested `CLAUDE.md`; project-wide rules are in the root `CLAUDE.md`. Conditions of applicability: `README.md`. Design: `docs/superpowers/specs/2026-11-26-fallout-defensive-swing-design.md`.

- `rule.py` -- pure `decide(state, ...) -> Action`; `store.py` -- the `SwingState` JSON (`data/fallout_defensive_swing_state_<mode>.json`); `agent_fallout_defensive_swing.py` -- the hourly tick and the orders.
- **Settlement guard:** no sale until the last sale has settled: a later session, with at least one day since the sale on which banks are open (Columbus Day and Veterans Day trade but do not settle). It is checked before `selling`, so a retried sale obeys it too. Never remove it: it prevents a good-faith violation on a cash account.
- **Cash reset:** in cash, the mix is bought back at market once `cash_reset_days` (5; `None` turns it off) trading days have passed since the last sale, dip or not; a dip still buys earlier. Days are weekdays after the sale session (`rule.trading_days_since`), so a holiday counts and the reset can come a day early. Added because a backtest from 2024-09 sold on 2025-04-22 and never bought again.
- **The saved state is corrected from the account on every tick.** Mix held but phase not holding: adopt it, basis = market value. Phase holding but nothing held: cash, sale prices = today's, `last_sale_session` = today (the sale may have been today, so nothing bought now is sold before tomorrow).
- **Orders:** waits are bounded (`ORDER_WAIT_SECONDS`), an unfinished order is canceled, nothing is saved while one is still active, and a tick with an active order does nothing. A buy that filled after a basis saved without it would look like a huge gain and be sold at once.
- Each buy is capped at a fresh `spendable_cash()`: the earlier buy fills above its sizing price.
- `selling=True` only when part of a sale filled; the rest then sells whatever the price. A sale that fills nothing changes no state.
- An order whose submit or cancel raises is logged and the tick goes on: the other leg must still be recorded. A sale is saved even with an order still open.
- Known limit: if the account stream reports nothing, the order stays open in the tracker and every tick waits; a restart recovers (the tick adopts the account).
- Known limit: if one leg is sold by hand while holding, the basis still covers both, so the target is not reached; sell the other leg by hand too.
- **Do not tune the thresholds to a backtest.** +1% / 1% dip were picked on 3-25 November 2026; no stop-loss because a 2% stop did worse in every study run.

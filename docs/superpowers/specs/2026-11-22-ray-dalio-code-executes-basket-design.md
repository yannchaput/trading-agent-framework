# ray_dalio: code executes the disagreement basket

## Why

In the first backtest (`logs/ray_dalio/backtesting/2026-11-21_084028_backtesting`) the trader agent did not follow
the basket the thoughtful-disagreement agent chose: in 50 of the 103 sessions where that basket left TMF out, the
trader still held TMF, which lost about $880. The basket was free text, and the trader's prompt invited it to
"reconcile overrides". It mostly failed by not selling: it filled orders on only 64 of about 129 sessions.

## Goal

What the disagreement agent picks is what the account holds, every session, with no agent able to override it.

Non-goals: a cap on bonds or SHV (dropped on 2026-11-22: the disagreement agent sizes the bonds itself), a guard
against holding a fund and its inverse, changes to the three analysts.

Anti-goal: code the user cannot read. Plain functions and dicts only, no new classes or type machinery.

## Approaches (A picked)

- **A. Code executes the basket (picked).** The disagreement agent submits its basket through a checked tool, and
  the shared `Rebalancer` trades to it. The trader agent is removed. Following the basket is then guaranteed by
  construction, and each session runs one LLM agent fewer.
- B. Keep the trader and add guards: refuse off-basket buys, sell off-basket holdings after it. Two places decide the
  book, and the trader's only remaining job is sizing, which code already does.
- C. Prompt only: tell the trader to execute the basket. Nothing enforces it.

## Design

1. **`submit_basket(weights, reason)`**, a new tool given only to the disagreement agent. `weights` is a dict of
   symbol to fraction of the portfolio, for example `{"TQQQ": 0.35, "UPRO": 0.25, "TMF": 0.25, "SHV": 0.15}`.
   It refuses with an error message the agent can act on (and resubmit) when:
   - a symbol is not in `UNIVERSE`;
   - a weight is not above 0;
   - fewer than 3 symbols (SHV counts as one, as the leverage rule already says);
   - the weights other than SHV add up to more than 0.98 (the framework keeps 2% in cash).

   An accepted basket is stored for the session (a plain dict on the strategy) and logged on one line.
2. **SHV is the remainder.** The basket's SHV weight is not traded as such: whatever the other weights leave
   unallocated is parked in SHV by the `Rebalancer`, as in bull_bear and sector_pods.
3. **A `rebalance` node replaces the trader node:** `START -> 3 analysts -> thoughtful_disagreement -> rebalance -> END`,
   so the whole session stays one graph run (asked 2026-11-22). The disagreement node writes the accepted basket to
   the graph state (`basket`); the `rebalance` node is plain code, not an agent: it hands the basket to
   `target_portfolio` + `Rebalancer.rebalance`, which sells what the basket leaves out first, then buys, then parks
   the rest in SHV. As a single node run by plain `graph.invoke()` it runs on the strategy thread (root CLAUDE.md
   threading rule), so never add streaming, `subgraphs=True` or `step_timeout`.
4. **No valid basket, no trades.** If the agent never submits an accepted basket, the session logs a warning and
   holds the current book.
5. **The rebalance band is 3%, the smallest trade 0.5% of the book** (sector_pods' values), so the daily drift of
   3x funds does not cause a trade every day.
6. **Removed:** the trader agent, its prompts, `ORDER_READINESS_RULE`, `SIZING_RULE`, `readiness.py` and their
   tests. The `Rebalancer` already reads the account, positions and prices before every order, which is what the
   readiness guard was for.
7. The disagreement agent's prompt says it must end by calling `submit_basket`, and that its basket is executed
   exactly as submitted.

## Left to the builder

- How to pass the three `Rebalancer` settings (`parking_symbol`, `rebalance_band`, `min_trade_pct`) in the plainest
  way that pyright accepts.
- Test names and fixtures, following `tests/strategies/ray_dalio/`.
- Updating `strategies/ray_dalio/CLAUDE.md` (graph, agents table, readiness and threading notes).

## How we'll know it worked

The tests show the executed book equals the submitted basket, and that a holding left out of the basket is sold.
A rerun on the same window as the first backtest then shows the strategy's results; matching the basket no longer
needs a rerun to check.

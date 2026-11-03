# Orders with an unknown outcome, for bull_bear and cross_momentum on Tastytrade -- design

Date: 2026-11-03. Status: approved in conversation, awaiting a plan.

## Why

The user will host bull_bear on Tastytrade live around 2026-11-17, and cross_momentum on Tastytrade too.

`TastytradeBroker` raises `BrokerUncertainError` (a `BrokerError`, `utils/errors.py`) when an order's outcome is unknown:
a timeout, a transport failure, a 5xx on the write, or a 2xx naming no order. It keeps the order tracked as
`SUBMITTED`, never resends it, and resolves it later by an `external-identifier` lookup (`brokers/CLAUDE.md`). The
order may well exist at Tastytrade.

The strategies read that error as "order refused":

- **bull_bear** (and bill_ackman) place every order through `strategies/common/rebalancer.Rebalancer`. Its `submit`
  catches `TradingFrameworkError`, logs "was refused" and returns False, so:
  - an unknown **buy** is not deducted from `available`, and the buys after it can spend the same cash twice;
  - an unknown **sell** adds no proceeds (safe: the buys get less) and is not waited for;
  - the order is missing from `placed`, so from `reviews.jsonl`'s `orders` and from the abandoned-review cooldown
    logic (`pipeline.py`, sells "actually placed").
- **cross_momentum** (`agent_cross_momentum._rebalance_book`) catches `Exception` around each order:
  - an unknown **buy** already had its cost deducted before the call (budget safe), but is logged "Failed to submit
    buy order";
  - an unknown **sell** (exit, trim, sleeve) adds no proceeds (safe) and is logged as failed; an exit is not counted
    in the summary's `sold`.

Across reviews nothing is sent twice already: the order stays tracked, and both strategies count open orders before
sending (`Rebalancer._in_flight`, cross_momentum's `in_flight_quantities` since b82b8e2), or see the fill in positions.

Separately, `IbkrBroker._submit_order` re-raises the raw SDK exception after `set_error` (`brokers/ibkr/broker.py`),
which the `Rebalancer` (catching `TradingFrameworkError` only) does not catch: the same half-done rebalance AlpacaBroker
had before b82b8e2.

## Goals

1. An order whose outcome is unknown is **never resent** and **never treated as refused** by the `Rebalancer` or
   cross_momentum's rebalance.
2. The budget stays on the safe side either way: an unknown **buy** counts as if it happened (its cost is deducted),
   an unknown **sell** counts as if it did not (its proceeds are not added). A run can then neither overspend nor
   double up.
3. It is visible: a WARNING naming the order and saying the outcome is unknown, it was not resent, and the account
   should be checked; in bull_bear/bill_ackman the order is in `placed` (hence `reviews.jsonl`) marked unconfirmed.
4. `IbkrBroker._submit_order` raises a `BrokerError` instead of the raw exception, after `set_error`, as
   `AlpacaBroker._submit_order` does.

## Non-goals

- The other order paths: vwap_pullback's and earnings_drift's desks, the shared `TradeDesk` (congress_trades,
  insider_trades), `Strategy.cancel_orders`/`cancel_open_orders`. None runs on Tastytrade soon; the "KNOWN
  LIMITATION" note in `brokers/CLAUDE.md` stays, narrowed to them.
- Fractional quantities on Tastytrade (eligibility untested, non-eligible orders floored to whole shares). A separate
  piece of work.
- An Alpaca or IBKR equivalent of the Tastytrade lookup: a timed-out Alpaca or IBKR submit is still reported as a
  failed `BrokerError` (the accepted risk documented in `brokers/CLAUDE.md`).
- Any change to how `TastytradeBroker` resolves unknown writes.

## Anti-goals

- Resending an order whose outcome is unknown, in the same run or by any retry.
- Counting an unknown sell's proceeds, or giving an unknown buy's cost back to the budget.
- Changing behaviour for an ordinary refusal (`OrderValidationError`, a broker rejection): it stays "refused", with
  the buying-power resync as today.
- Changing anything in backtests (`BacktestBroker` never raises `BrokerUncertainError`).

## Decided: how it works

### Rebalancer (bull_bear, bill_ackman)

- `PlacedOrder` gains `confirmed: bool = True`.
- In `submit`, `BrokerUncertainError` is caught BEFORE the `TradingFrameworkError` handler. It is logged as a
  WARNING (`{side} {quantity} {symbol} ({why}): outcome unknown at the broker, not resent -- check the account:
  {exc}`), appended to `placed` with `confirmed=False`, and `submit` reports it so each caller can apply goal 2:
  - a stock **buy** or the **parking buy**: deduct its cost from `available` as for an accepted order;
  - a stock **sell**: do not add its proceeds, and do not add it to the sells waited on;
  - the **parking sell** funding the buys: do not add its value to `available`.
- `refusals` is not updated by an unknown outcome (it feeds the buying-power resync, which reads a refusal).
- The abandoned-review path (`bull_bear/pipeline.py`, sells "actually placed" start a cooldown) keeps reading
  `placed`: an unconfirmed sell starts its cooldown, the safe reading. Same for bill_ackman's pipeline if it reads
  `placed` that way.

### cross_momentum

- In `_rebalance_book`, each order site catches `BrokerUncertainError` before its generic `except Exception`:
  - logs a WARNING with the same wording;
  - an unknown buy (stock or sleeve) keeps its cost deducted (as today, the deduction happens before the call);
  - an unknown sell (exit, trim, sleeve) adds nothing to `sells` (so no proceeds), as today;
  - an unknown exit is counted in the summary line as unconfirmed (e.g. `..., 3 sold (1 unconfirmed)`).
- The buying-power resync is not attempted on an unknown outcome.

### IBKR

- `IbkrBroker._submit_order`'s generic `except Exception` keeps `set_error` and `untrack` first, then re-raises the
  exception unchanged if it is already a `TradingFrameworkError`, else raises `BrokerError(str(exc)) from exc`.

### Docs

- `brokers/CLAUDE.md`: the "KNOWN LIMITATION" bullet says the `Rebalancer` and cross_momentum handle
  `BrokerUncertainError` (what they do), and the remaining desks do not; the "never let a raw SDK exception escape"
  bullet drops "IbkrBroker still re-raises raw".
- `strategies/common/CLAUDE.md` (`rebalancer.py`) and `strategies/cross_momentum/CLAUDE.md`: one line each.

## Left to the builder

- How `submit` reports the third outcome (an enum, a tri-state, or a return value) as long as each caller applies
  goal 2 and the existing refused path is unchanged.
- How the cross_momentum handling avoids repeating the same three lines at its five order sites (a small helper is
  fine; do not restructure `_rebalance_book`).
- Whether bill_ackman's pipeline needs a change beyond the shared `Rebalancer` (check how it reads `placed`).

## How we know it worked

Unit tests with the existing hand-written fakes (no network), one per behaviour:

- Rebalancer: a fake strategy whose `submit_order` raises `BrokerUncertainError` for one buy: the later buys are sized
  as if that buy's cost was spent (literal quantities), it is in `placed` with `confirmed=False`, a WARNING is logged,
  it is submitted once only.
- Rebalancer: an unknown sell adds no proceeds and is not waited on; an unknown parking sell does not fund the buys.
- Rebalancer: an ordinary `OrderValidationError` refusal still behaves as today (regression).
- cross_momentum: an unknown buy is submitted once, keeps its cost deducted, logs the WARNING (not "Failed to
  submit"); an unknown exit is counted as unconfirmed in the summary and adds no proceeds.
- IBKR: a raw exception from the connection surfaces as `BrokerError` with the original as `__cause__`, `set_error`
  done, order untracked; a `BrokerError` from the connection is re-raised unchanged.

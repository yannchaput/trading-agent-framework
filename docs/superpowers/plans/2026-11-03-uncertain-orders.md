# Orders with an unknown outcome -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** bull_bear (via the shared `Rebalancer`) and cross_momentum treat a Tastytrade order whose outcome is unknown (`BrokerUncertainError`) as sent-but-unconfirmed: never resent, budgeted on the safe side, logged as such; and `IbkrBroker._submit_order` stops leaking raw exceptions.

**Architecture:** Each order site catches `BrokerUncertainError` before its generic handler. An unknown buy counts as spent, an unknown sell brings nothing; the `Rebalancer` records it in `placed` with `confirmed=False`. IBKR wraps a non-framework exception in `BrokerError`.

**Tech Stack:** Python 3.14, uv, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-03-uncertain-orders-design.md`

## Global Constraints

- Never resend an order whose outcome is unknown, in the same run or by any retry.
- Never count an unknown sell's proceeds; never give an unknown buy's cost back to the budget.
- An ordinary refusal (`OrderValidationError`, a broker rejection) behaves exactly as today, buying-power resync included.
- Nothing changes in backtests (`BacktestBroker` never raises `BrokerUncertainError`).
- WARNING wording, at every site: `{side} {quantity:g} {symbol} ({why}): outcome unknown at the broker, not resent -- check the account: {exc}`.
- Out of scope: vwap_pullback/earnings_drift desks, `TradeDesk`, `Strategy.cancel_orders`, fractional quantities, any change to `TastytradeBroker`.
- Tests use the existing hand-written fakes, never the network. Paths below are relative to the repo root.

## Review Focus

1. **The parking (SHV) sell that funds the buys returns an unknown outcome.** Its value must not reach `available`, or every buy after it spends money that may not exist. Test in Task 2.
2. **An unknown sell must not be waited on** (paper/live wait up to 30 s for this run's sells). Waiting on an order whose existence is unknown burns the timeout and then sizes the buys anyway. Test in Task 2.
3. **The buying-power resync must not read an unknown outcome** (`refusals` feeds `parse_insufficient_buying_power`). An earlier ordinary refusal must not be overwritten by it either. Test in Task 2.
4. **The order is submitted exactly once.** No retry in any handler. Asserted in every Task 2 and Task 3 test.
5. **IBKR: an exception that is already a framework error** (e.g. a `BrokerError` from the connection) passes through as the same object, not double-wrapped. Test in Task 1.

---

### Task 1: IBKR wraps raw submit errors

**Files:**
- Modify: `src/trading_agent_framework/brokers/ibkr/broker.py` (`_submit_order`'s `except Exception`)
- Modify: `src/trading_agent_framework/brokers/CLAUDE.md` (drop "`IbkrBroker._submit_order` still re-raises raw.")
- Test: `tests/brokers/ibkr/test_broker.py`

**Interfaces:**
- Produces: `IbkrBroker._submit_order` raises `BrokerError(str(exc)) from exc` for a non-`TradingFrameworkError`, after `set_error` and `untrack` (unchanged order); re-raises a `TradingFrameworkError` as is.

- [ ] **Step 1: Write the failing tests**

- `test_a_raw_exception_from_placing_surfaces_as_broker_error`: make the fake IB's `placeOrder` raise `RuntimeError("boom")` (monkeypatch `ib.placeOrder`). `broker.submit_order(_buy())` raises `BrokerError` matching `boom`, with `raised.value.__cause__` the RuntimeError, `order.status is OrderStatus.ERROR`, and `broker.tracker.get_tracked_order(order.identifier) is None`.
- `test_a_framework_error_from_placing_is_re_raised_unchanged`: `placeOrder` raises `err = BrokerError("gateway gone")`; the raised exception `is err`.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/brokers/ibkr/test_broker.py -q`
Expected: the first FAILS (RuntimeError escapes). The second may already pass if the connection passes `BrokerError` through; keep it as the regression pin.

- [ ] **Step 3: Implement**

`except TradingFrameworkError: set_error, log, untrack, raise`, then `except Exception as exc: same, raise BrokerError(str(exc)) from exc`. Or one handler with an `isinstance` check; the comment keeps "set_error BEFORE raising -- lumibot's contract". Update the `brokers/CLAUDE.md` bullet.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/brokers/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/broker.py src/trading_agent_framework/brokers/CLAUDE.md tests/brokers/ibkr/test_broker.py
git commit -m "fix(ibkr): Task 1 - wrap raw submit errors in BrokerError"
```

### Task 2: Rebalancer handles an unknown outcome

**Files:**
- Modify: `src/trading_agent_framework/strategies/common/rebalancer.py` (`PlacedOrder`, `rebalance`'s `submit` and its callers)
- Modify: `src/trading_agent_framework/strategies/common/CLAUDE.md` (one line under `rebalancer.py`)
- Test: `tests/strategies/common/test_common_rebalancer.py`

**Interfaces:**
- Produces: `PlacedOrder(symbol: str, side: str, quantity: float, confirmed: bool = True)`. bull_bear and bill_ackman serialize it with `asdict`, so `reviews.jsonl` orders gain `"confirmed"`. Their pipelines need no change: an unconfirmed sell in `placed` starts its cooldown on an abandoned review (the safe reading).
- `submit` gets a third outcome (accepted / refused / unknown). The shape is the builder's choice; every caller applies the Global Constraints.

- [ ] **Step 1: Write the failing tests**

Use `_book`, `_orders` and the monkeypatched `strategy.submit_order` pattern of `test_a_refused_buy_resyncs_...`. Count calls per symbol to assert a single submission. The values were derived by hand with a portfolio value of 10,000, a 200 reserve and AckmanParams' band/min trade:

- `test_an_unknown_buy_counts_as_spent_for_the_buys_after_it`: `positions={}`, prices A 50 / B 25 / SHV 100, `cash=5200`, target `{"A": 0.35, "B": 0.35}`; A's buy raises `BrokerUncertainError("timeout")`. Expected `placed == [PlacedOrder("A", "buy", 70.0, confirmed=False), PlacedOrder("B", "buy", 60.0)]`: available 5,000, less A's 3,500, leaves 1,500, which is 60 B. A refused A would give B 140. A was submitted once. A WARNING containing `outcome unknown at the broker, not resent` is logged (caplog).
- `test_an_unknown_sell_brings_no_proceeds_and_is_not_waited_on`: `positions={"B": 100}`, `cash=7500`, target `{"A": 0.35}`, B's sell raises `BrokerUncertainError`; spy `strategy.wait_for_orders_execution`. Expected `placed == [PlacedOrder("B", "sell", 100.0, confirmed=False), PlacedOrder("A", "buy", 70.0), PlacedOrder("SHV", "buy", 38.0)]` and the wait is never called.
- `test_an_unknown_parking_sell_does_not_fund_the_buys`: `positions={"SHV": 98}`, `cash=200`, target `{"A": 0.35, "B": 0.25}`, the SHV sell raises `BrokerUncertainError`. Expected `placed == [PlacedOrder("SHV", "sell", 60.0, confirmed=False)]`: no buy, since the available cash is still 0.
- `test_an_unknown_outcome_does_not_feed_the_buying_power_resync`: as the first test, but A raises `BrokerUncertainError('{"buying_power":"100000","message":"insufficient buying power"}')`. B is still 60, not resynced up.
- The existing `test_a_refused_buy_resyncs_...` and `test_a_refused_sell_is_not_counted_as_proceeds` stay unchanged (regression).

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/common/test_common_rebalancer.py -q`
Expected: the four new tests FAIL; the existing ones pass.

- [ ] **Step 3: Implement**

In `submit`, catch `BrokerUncertainError` before `TradingFrameworkError`: WARNING (Global Constraints wording), append `PlacedOrder(..., confirmed=False)`, leave `refusals` untouched, and return the unknown outcome. A stock buy and the parking buy deduct their cost from `available`. A stock sell adds no proceeds and is not appended to `sells`. The parking sell adds nothing to `available` and leaves `parking_value` unchanged. Add a one-line note in the class docstring and in `strategies/common/CLAUDE.md`.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/strategies/ -q`
Expected: PASS (bull_bear and bill_ackman tests included).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/common/rebalancer.py src/trading_agent_framework/strategies/common/CLAUDE.md tests/strategies/common/test_common_rebalancer.py
git commit -m "fix(rebalancer): Task 2 - an order with an unknown outcome counts as sent, never resent"
```

### Task 3: cross_momentum handles an unknown outcome

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`_rebalance_book`'s five order sites and the summary line)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md`, `src/trading_agent_framework/brokers/CLAUDE.md` (the KNOWN LIMITATION bullet: the Rebalancer and cross_momentum now handle `BrokerUncertainError`, the other desks still do not)
- Test: `tests/strategies/test_cross_momentum_rebalance.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2 (independent code path).
- The summary line keeps its format and adds ` ({n} unconfirmed)` after `{sold} sold` only when n > 0. An unknown exit counts in `sold` and in n.

- [ ] **Step 1: Write the failing tests**

Add an `uncertain=()` parameter to the file's `FakeStrategy`: `submit_order` raises `BrokerUncertainError(f"timeout {symbol}")` for those symbols and counts attempts per symbol.

- `test_an_unknown_buy_keeps_its_cost_and_is_logged_as_unknown`: `cash=1000`, prices AAA 100 / BBB 50, target AAA 0.5 / BBB 0.5, AAA uncertain. Expected: BBB buy `[9.0]` (950 less AAA's 500 leaves 450); AAA attempted once. A warning contains `outcome unknown at the broker, not resent`, and none contains `Failed to submit buy order for AAA`.
- `test_an_unknown_exit_brings_no_proceeds_and_is_counted_unconfirmed`: `cash=0`, `positions=[_held("OLD", 10.0)]`, prices OLD 100 / AAA 50, target AAA 0.5, OLD unranked and uncertain. Expected: no AAA buy, OLD attempted once, a summary line containing `1 sold (1 unconfirmed)`.
- `test_an_unknown_trim_is_not_resent`: `positions=[_held("AAA", 10.0)]`, `cash=0`, AAA price 100, target AAA 0.4, AAA uncertain. Expected: AAA attempted once; the warning is logged.
- The existing `reject=` tests stay unchanged (regression).

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -q`
Expected: the new tests FAIL (the generic handler logs "Failed to submit", the summary has no `unconfirmed`).

- [ ] **Step 3: Implement**

At each of the five sites, `except BrokerUncertainError as exc:` before `except Exception`, logging the WARNING. A buy keeps the cost already deducted. A sell adds nothing to `sells`. An exit increments `sold_count` and an unconfirmed counter. No buying-power resync. A small local helper for the warning is fine; do not restructure `_rebalance_book`. Update both CLAUDE.md files.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest -q` then `uv run ruff check`
Expected: full suite PASS, "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md src/trading_agent_framework/brokers/CLAUDE.md tests/strategies/test_cross_momentum_rebalance.py
git commit -m "fix(cross_momentum): Task 3 - an order with an unknown outcome is logged as such, never resent"
```

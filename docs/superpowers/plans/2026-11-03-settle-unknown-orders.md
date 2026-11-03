# Settle unknown orders before each tick -- implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a phantom Tastytrade order (outcome unknown, never created at Tastytrade) is settled before the next weekly rebalance, so bull_bear and cross_momentum stop skipping its symbol or counting its money.

**Architecture:** a new no-op `Broker.settle_unknown_orders()`, overridden by `TastytradeBroker` with its existing `_resolve_unknown_writes()`. The executor calls it before every strategy tick and dispatches the resulting order events before the tick runs.

**Tech Stack:** Python 3.14, uv, pytest, ruff.

**Spec:** small change approved in chat on 2026-11-03; the design is below.

## Design (approved)

- **Purpose:** a phantom Tastytrade order must be settled before the next weekly rebalance. Faster settling is not needed.
- `Broker.settle_unknown_orders() -> None`: a non-abstract no-op on the base class, like `start_stream`. Alpaca, IBKR and `BacktestBroker` inherit it, so nothing changes for them.
- `TastytradeBroker.settle_unknown_orders()` calls `self._resolve_unknown_writes()`, unchanged. It is cheap with no unknown orders (no HTTP call). A found order is adopted. An order missing in two lookups at least 30 s apart (`UNKNOWN_WRITE_GRACE`) fails with an error event, and is never resent.
- **Executor:** before every `_iterate()`, call `broker.settle_unknown_orders()`, then `_dispatch_events()`, so the tick sees the outcome. Any exception from settling is logged as a WARNING and the tick runs anyway: no `on_bot_crash`, the same rule as `_refresh_regime`.
- **Timeline:** both strategies tick daily (`sleeptime = "1D"`). A phantom from Tuesday's rebalance takes its first miss on Wednesday's tick and fails on Thursday's, before the next Tuesday.
- **Docs:** the "phantom order stays until restart" notes in `strategies/common/CLAUDE.md`, `strategies/cross_momentum/CLAUDE.md` and `brokers/CLAUDE.md` become "settled before each tick", with the restart kept only as a fallback. `core/CLAUDE.md` gets one line on the executor step.
- **Non-goals:** settling between ticks (every 60 s), any change to `_resolve_unknown_writes` or its grace rule, and other brokers' timeouts.

## Global Constraints

- `_resolve_unknown_writes`, `UNKNOWN_WRITE_GRACE` and the two-miss rule are unchanged; nothing is ever resent.
- A failure while settling never skips or crashes a tick (WARNING only).
- Backtests unchanged: `BacktestBroker` inherits the no-op.
- Broker code only feeds `OrderTracker`; strategy hooks run on the executor thread (`_dispatch_events`).
- Tests use the existing hand-written fakes, never the network. Paths are relative to the repo root.

## Review Focus

1. **Settle happens BEFORE the tick and its events are dispatched before `on_trading_iteration`.** Otherwise the rebalance still sees the phantom. Test in Task 2.
2. **A raising `settle_unknown_orders` must not stop the tick, nor call `on_bot_crash`.** Test in Task 2.
3. **Settling runs on every iteration, not once per session or only at start.** With an interval sleeptime, every tick. Test in Task 2.
4. **The Tastytrade override settles a phantom in two calls spaced past the grace, and adopts a real order.** Reuse the existing `reconcile` resolution tests' setup. Test in Task 1.

---

### Task 1: `Broker.settle_unknown_orders`

**Files:**
- Modify: `src/trading_agent_framework/brokers/base.py` (new no-op method)
- Modify: `src/trading_agent_framework/brokers/tastytrade/broker.py` (override)
- Test: `tests/brokers/tastytrade/test_broker.py`, and a base-class test next to the other `Broker` tests (`tests/brokers/test_base.py` if it exists, else the Alpaca broker tests)

**Interfaces:**
- Produces: `Broker.settle_unknown_orders(self) -> None` (no-op, `# noqa: B027` like `start_stream`); `TastytradeBroker.settle_unknown_orders(self) -> None` = `self._resolve_unknown_writes()`.

- [ ] **Step 1: Write the failing tests**

Model these on `test_reconcile_resolves_an_unknown_submission_that_exists` and `test_reconcile_leaves_an_unknown_submission_while_lookups_still_fail` (tests/brokers/tastytrade/test_broker.py ~650-720), with the same fakes:
- `test_settle_adopts_an_unknown_submission_that_exists`: same setup as the reconcile test, call `broker.settle_unknown_orders()` instead of `reconcile()`, same assertions.
- `test_settle_fails_a_phantom_submission_after_two_spaced_misses`: an unknown submission absent from every lookup. Settle once past `UNKNOWN_WRITE_GRACE`: still active. Advance the fake clock past the grace and settle again: the order is no longer active and an error event was emitted. It was never resent (the session saw one POST).
- `test_settle_makes_no_request_without_unknown_orders`: a fresh broker; settle sends no HTTP request.
- Base class: `test_the_base_broker_settle_is_a_no_op`: a non-Tastytrade broker (e.g. `FakeBroker` or `AlpacaBroker` with fakes) returns None from `settle_unknown_orders()` and makes no client call.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/brokers/ -q`
Expected: the new tests FAIL with AttributeError.

- [ ] **Step 3: Implement** the two methods, with a docstring on the base method saying when it is called (the executor, before every tick) and that the default is a no-op.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest tests/brokers/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/base.py src/trading_agent_framework/brokers/tastytrade/broker.py tests/brokers/
git commit -m "feat(brokers): Task 1 - settle_unknown_orders, Tastytrade resolves its unknown writes"
```

### Task 2: The executor settles before every tick

**Files:**
- Modify: `src/trading_agent_framework/core/executor.py` (`_trade` or `_iterate`, plus a `_settle_unknown_orders` helper beside `_refresh_regime`)
- Modify: `src/trading_agent_framework/core/CLAUDE.md`, `src/trading_agent_framework/strategies/common/CLAUDE.md`, `src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md`, `src/trading_agent_framework/brokers/CLAUDE.md`
- Test: `tests/core/test_executor.py` (or `tests/core/test_executor_events.py` for the event test)

**Interfaces:**
- Consumes: `Broker.settle_unknown_orders()` (Task 1).

- [ ] **Step 1: Write the failing tests** (a `FakeBroker` subclass that records calls; the strategy records hook order in a shared list)

- `test_unknown_orders_are_settled_before_each_iteration`: with `sleeptime` "1D" over 3 sessions, the recorded sequence has `settle` immediately before each `iteration` (3 pairs).
- `test_an_interval_sleeptime_settles_before_every_tick`: with an interval sleeptime giving N ticks in a session, there are N settles, each just before its tick.
- `test_order_events_from_settling_are_dispatched_before_the_iteration`: `settle_unknown_orders` makes the tracker emit an error event for a tracked order (as the Tastytrade resolution does). The strategy's error hook (`on_error_order` or the repo's name for it; see test_executor_events.py) runs before `on_trading_iteration`.
- `test_a_failing_settle_is_logged_and_the_iteration_still_runs`: settle raises `BrokerError("down")`. The iteration still runs, a WARNING names the failure, and `on_bot_crash` is not called.

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/core/ -q`
Expected: the new tests FAIL (settle never called).

- [ ] **Step 3: Implement**

`_settle_unknown_orders()`: `try: broker.settle_unknown_orders()` / `except Exception: logger.warning(..., exc_info=True)`, then `self._dispatch_events()`. Call it right before `self._iterate()` in `_trade`. Update the four CLAUDE.md files per the Design section.

- [ ] **Step 4: Run, expect green**

Run: `uv run pytest -q` then `uv run ruff check`
Expected: full suite PASS, "All checks passed!".

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/core/executor.py src/trading_agent_framework/core/CLAUDE.md src/trading_agent_framework/strategies/common/CLAUDE.md src/trading_agent_framework/strategies/cross_momentum/CLAUDE.md src/trading_agent_framework/brokers/CLAUDE.md tests/core/
git commit -m "feat(core): Task 2 - the executor settles unknown orders before every tick"
```

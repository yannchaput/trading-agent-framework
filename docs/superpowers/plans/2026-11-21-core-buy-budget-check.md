# Core buy budget check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every buy, from any strategy or LLM tool, passes one budget check in `Strategy.submit_order`, sized by one shared `Strategy.spendable_cash()` that keeps a fixed 2% buffer.

**Architecture:** `core/strategy.py` gains `BUY_BUFFER`, `spendable_cash()` and the check; `utils/errors.py` gains `InsufficientFundsError`. The seven copies of the sizing rule (Rebalancer, trade_desk, earnings_drift, cross_momentum, vwap_pullback, the ray_dalio and news_builtin prompts) switch to it; the larger per-strategy buffers are deleted.

**Tech Stack:** Python 3.14, Decimal, pytest with `tests/fakes.py` (`FakeBroker`, real objects, no `MagicMock`).

**Spec:** `docs/superpowers/specs/2026-11-21-core-buy-budget-check-design.md`

## Global Constraints

- `spendable = min(buying_power, cash + unfilled open sells - unfilled open buys) - BUY_BUFFER x portfolio_value`, `BUY_BUFFER = Decimal("0.02")`, a constant defined in `utils/helpers.py` (next to `fractional_qty`; `core/strategy.py` is heavy and the pure parameter modules need it too), never a parameter.
- Read the account FIRST, then `broker.tracker.get_active_orders()`. Count `quantity - filled_quantity` only.
- An open order is valued at its `limit_price`, else `get_last_price`; no price: a sell credits nothing, a buy costs nothing. STOP, STOP_LIMIT and TRAIL sells credit nothing.
- Money is `Decimal` in core; no `datetime.now` / `time.sleep`.
- Only BUY orders with a quantity are checked; a buy with no price estimate is submitted unchecked.
- Gate: `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, `uv run pytest --cov` (>= 80%), `uv run interrogate -v` (>= 80%); no `noqa` / ignores; never loosen a test to pass, update tests that sized to 100% of cash to expect the 2% buffer.
- Commit after `/ponytail:ponytail-review` on the staged changes, message ending with the attribution line.

## Review Focus

- Live double count: a sell that fills between the account read and the orders read is credited zero times, never twice (Task 1 test `test_a_sell_filling_between_the_reads_is_not_credited_twice`).
- A batch through `submit_orders`: the second buy sees the first as an open order and is refused when together they exceed the budget (Task 1 test `test_submit_orders_checks_each_buy_against_the_ones_before`).
- A fully invested book: `spendable_cash()` <= 0 gives `max_quantity == 0`, not a negative quantity (Task 1 test).
- A Rebalancer strategy configured with `cash_buffer` below 0.02 is rejected at construction, not left to chase unfundable buys (Task 3 tests).
- An LLM trader gets the refusal as `{"error": ...}` naming the max quantity, never an exception out of the tool (Task 2 test).

---

### Task 1: `spendable_cash()` and the check in `Strategy.submit_order`

**Files:**
- Modify: `src/trading_agent_framework/core/strategy.py` (near `submit_order`, :537-543), `src/trading_agent_framework/utils/helpers.py` (`BUY_BUFFER`)
- Modify: `src/trading_agent_framework/utils/errors.py`
- Modify: `src/trading_agent_framework/core/CLAUDE.md`, root `CLAUDE.md` (Key patterns: one bullet on the rule and where it lives)
- Test: `tests/core/test_strategy_buy_budget.py` (new)

**Interfaces:**
- Produces: `BUY_BUFFER: Decimal` (in `utils/helpers.py`, imported by `core/strategy.py`); `Strategy.spendable_cash() -> Decimal`; `InsufficientFundsError(OrderValidationError)` with constructor `InsufficientFundsError(message: str, *, spendable: Decimal, max_quantity: Decimal)` and those two attributes.
- `Strategy.submit_orders(orders)` returns `[self.submit_order(o) for o in orders]`.

- [ ] **Step 1: Write the failing tests** (FakeBroker defaults: cash 10,000, portfolio 25,000, buying power 20,000; set `broker.last_prices` / `broker.account` as the existing core tests do)

```python
def test_spendable_is_the_smaller_term_less_two_percent_of_the_book():  # min(20000, 10000) - 500
    assert strategy.spendable_cash() == Decimal("9500")

def test_buying_power_binds_when_it_is_smaller():  # cash 10000, bp 4000, pv 25000 -> 3500
def test_open_sells_credit_only_their_unfilled_part():  # sell 10 @ limit 100, 4 filled -> +600
def test_open_buys_are_deducted():  # open buy 5 at last price 200 -> -1000
def test_a_stop_sell_brings_no_credit():
def test_a_sell_filling_between_the_reads_is_not_credited_twice():  # account read sees pre-fill cash; the fill lands before the orders read -> credit 0
def test_a_buy_over_spendable_raises_with_the_max_quantity():
    with pytest.raises(InsufficientFundsError) as err:
        strategy.submit_order(strategy.create_order("AAA", 100, "buy"))  # last price 100 -> $10,000 > $9,500
    assert err.value.max_quantity == Decimal("95") and err.value.spendable == Decimal("9500")
def test_a_buy_at_spendable_passes():  # 95 @ 100
def test_a_sell_and_a_buy_without_a_price_pass_unchecked():
def test_a_fully_invested_book_gives_max_quantity_zero():
def test_submit_orders_checks_each_buy_against_the_ones_before():  # two buys of 60 @ 100: first passes, second raises
```

- [ ] **Step 2: Run** `uv run pytest tests/core/test_strategy_buy_budget.py -v` -> FAIL (`spendable_cash` / `InsufficientFundsError` missing).

- [ ] **Step 3: Implement** in `core/strategy.py` and `utils/errors.py` per the Global Constraints. Order value for a buy being checked: quantity x (its limit price, else `get_last_price(asset)`). `max_quantity = fractional_qty(spendable / price)` as `Decimal` (0 when spendable <= 0). Message: `buy {qty} {symbol} needs ~${cost:,.2f}, spendable is ${spendable:,.2f} (cash after open orders, 2% of the book kept back): at most {max_quantity} shares`.

- [ ] **Step 4: Run** `uv run pytest tests/core -q` -> PASS; then the full suite redirected to a file. Expected: failures only in tests that buy above 95% of the fake's cash. List every failing test by name in your report (Tasks 2-4 clear them; the branch is not green between Tasks 1 and 4 by design, and the full gate runs at the end of Task 4). Do not fix them here unless they are core tests.

- [ ] **Step 5: Commit** `feat(core): one buy budget check in Strategy.submit_order`

### Task 2: Agent tools and prompts read `spendable`

**Files:**
- Modify: `src/trading_agent_framework/agents/tools/account.py:33-43` (`get_account_balance`)
- Modify: `src/trading_agent_framework/strategies/ray_dalio/prompts.py` (`SIZING_RULE`), `src/trading_agent_framework/strategies/news_builtin/agent_news_binary.py:145-148`
- Modify: `strategies/CLAUDE.md` (sizing rule), `backtesting/CLAUDE.md:26` (point agents to `spendable`), `agents/CLAUDE.md` if it documents `get_account_balance`
- Test: `tests/agents/` account-tool test file, `tests/strategies/ray_dalio/test_prompts.py`, news_builtin prompt test if one pins the sentence

**Interfaces:** Consumes `Strategy.spendable_cash()`, `InsufficientFundsError`.

- [ ] **Step 1: Failing tests:** `get_account_balance()["spendable"] == 9500.0` with FakeBroker defaults; the trading tool's `submit_order` over budget returns a dict whose `"error"` contains `"at most 95"`; `"spendable" in SIZING_RULE`; news_builtin's prompt contains `spendable` and no longer `95%`.
- [ ] **Step 2: Run** the touched test files -> FAIL.
- [ ] **Step 3: Implement.** New `SIZING_RULE` text: `"Never buy for more than 'spendable' from get_account_balance (it already counts open orders and keeps 2% of the book back); read it again after each order."` Same sentence replaces news_builtin's 95% sizing sentence.
- [ ] **Step 4: Run** `uv run pytest tests/agents tests/strategies/ray_dalio tests/strategies/news_builtin -q` -> PASS.
- [ ] **Step 5: Commit** `feat(agents): get_account_balance reports spendable; prompts size from it`

### Task 3: Rebalancer, trade_desk and the `cash_buffer` floor

**Files:**
- Modify: `src/trading_agent_framework/strategies/common/rebalancer.py:222,293-297,345-352` (drop `reserve`; `available = float(strategy.spendable_cash())` where buying power is re-read; on `InsufficientFundsError` resync `available = float(exc.spendable)`, keep the Alpaca `parse_insufficient_buying_power` path)
- Modify: `src/trading_agent_framework/strategies/common/trade_desk.py:289` (cash clause uses `spendable_cash()`; its weight/minimum checks stay)
- Modify: `bull_bear/parameters.py`, `sector_pods/params.py`, `bill_ackman/parameters.py`: validation `cash_buffer must be in [0.02, 1)` (import `BUY_BUFFER` from `utils/helpers.py`, compare with `float(BUY_BUFFER)`; never import `core/strategy.py` from a parameter module)
- Modify: `strategies/common/CLAUDE.md` (rebalancer and trade_desk lines)
- Test: `tests/strategies/common/test_common_rebalancer.py`, trade_desk tests, the three params tests

**Interfaces:** Consumes `Strategy.spendable_cash()`, `InsufficientFundsError(message, *, spendable, max_quantity)`, `BUY_BUFFER` (`utils/helpers.py`).

- [ ] **Step 1: Failing tests:** each params class rejects `cash_buffer=0.01` and accepts `0.02`; a Rebalancer whose buy is refused with `InsufficientFundsError(spendable=...)` continues with the next symbol sized from `exc.spendable`; trade_desk refuses a buy above `spendable_cash()` with "at most N shares".
- [ ] **Step 2: Run** them -> FAIL.
- [ ] **Step 3: Implement** as in Files. The account is still read once at the start for `portfolio_value`/`cash` used in planning; the cash invariant in the Rebalancer docstring is updated to say `spendable_cash()` does its own account-then-orders read.
- [ ] **Step 4: Run** `uv run pytest tests/strategies/common tests/strategies/bull_bear tests/strategies/sector_pods tests/strategies/bill_ackman -q` -> PASS (update tests that expected sizing to the last dollar, never loosen a check).
- [ ] **Step 5: Commit** `refactor(strategies): Rebalancer and trade_desk size from spendable_cash`

### Task 4: cross_momentum, earnings_drift and vwap_pullback drop their own formula and buffer

**Files:**
- Modify: `strategies/cross_momentum/agent_cross_momentum.py:707,743-751,809`, `strategies/cross_momentum/parameters.py:33` (delete `cash_buffer_pct`; parking target uses `1 - BUY_BUFFER`); `cross_momentum/utils.py` `unfilled_sell_proceeds` deleted if unused after
- Modify: `strategies/earnings_drift/desk.py:170` (`available = strategy.spendable_cash()`; keep `_cycle_*` only if still used)
- Modify: `strategies/vwap_pullback/risk.py:94` and `plan_entry` signature (take `spendable: Decimal` instead of `buying_power`, `cash`, `pending_sell_proceeds`), `vwap_pullback/desk.py:160-169`, `vwap_pullback/parameters.py:49` (delete `cash_buffer`)
- Modify: each strategy's `CLAUDE.md` where it documents the removed parameter
- Test: `tests/strategies/test_cross_momentum_rebalance.py`, earnings_drift desk tests, vwap_pullback risk/desk tests

**Interfaces:** Consumes `Strategy.spendable_cash()`, `BUY_BUFFER` (`utils/helpers.py`). Produces `risk.plan_entry(..., spendable: Decimal, ...)`. Deleting `unfilled_sell_proceeds` also deletes its test in `tests/strategies/test_cross_momentum_rebalance.py` (`test_unfilled_sell_proceeds_counts_t...`).

- [ ] **Step 1: Failing tests:** cross_momentum sizes its buys to `spendable_cash()` and its parameters no longer contain `cash_buffer_pct`; `plan_entry` caps quantity by `spendable / limit`; earnings_drift's budget equals `min(cap, spendable_cash())`.
- [ ] **Step 2: Run** them -> FAIL.
- [ ] **Step 3: Implement** as in Files; delete the dead helpers and parameters (grep each removed name: `cash_buffer_pct`, `unfilled_sell_proceeds`, vwap's `cash_buffer`, across `src`, `tests`, `env`, `scripts`, docs).
- [ ] **Step 4: Run** the full gate (Global Constraints) -> all green; report the numbers.
- [ ] **Step 5: Commit** `refactor(strategies): cross_momentum, earnings_drift and vwap_pullback size from spendable_cash; drop their 5% buffers`

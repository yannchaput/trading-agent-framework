# Tastytrade Clearing Fees in Backtests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtests with `BROKER=tastytrade` charge Tastytrade's clearing fees: $0.0008/share for a whole-share order, $0.10 flat for a fractional one.

**Architecture:** One change to the pure fee model `src/trading_agent_framework/brokers/fees.py`: the Tastytrade entry of `COMMISSIONS` (`_tastytrade_commission(shares)`) returns the clearing fee instead of 0. `TradingFeeFactory.fees` already adds the regulatory fees and rounds each side up to the cent; nothing else changes.

**Tech Stack:** Python 3.14, `Decimal`, pytest (`uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-11-26-tastytrade-clearing-fees-design.md`

## Global Constraints

- Rates: `TASTYTRADE_CLEARING_PER_SHARE = Decimal("0.0008")`, `TASTYTRADE_FRACTIONAL_CLEARING_PER_ORDER = Decimal("0.10")` (Tastytrade "Commissions & Fees", last updated 2026-10-28).
- Fractional order: $0.10 flat **instead of** the per-share fee; the whole order counts as fractional.
- Regulatory fees (SEC, FINRA TAF, CAT) unchanged and still added on top; each side rounded up to the cent.
- Money stays `Decimal`. No change to the live Tastytrade broker or to any env file.

## Review Focus

- A whole quantity written with trailing zeros (`Decimal("10.000")`) must be billed per share, not as fractional -> test in Task 1.
- A tiny fractional quantity (`Decimal("0.00001")`) still pays the full $0.10 -> test in Task 1.
- A large whole order: per-share fee scales (10,000 shares buy = $8.03 with CAT) -> test in Task 1.
- Zero shares on a side: no fee on that side (existing behaviour, already tested by `test_a_side_without_shares_costs_nothing_even_at_ibkr`; unchanged).
- Alpaca and IBKR fees unchanged (existing tests stay green).

---

### Task 1: Tastytrade clearing fee in `brokers/fees.py`

**Files:**
- Modify: `src/trading_agent_framework/brokers/fees.py` (docstring, constants, `_tastytrade_commission`)
- Modify: `src/trading_agent_framework/backtesting/CLAUDE.md` (the "Backtest fees come from `BROKER`" bullet: add "Tastytrade $0.0008/share clearing, $0.10 flat for a fractional order, rates 2026-10-28")
- Test: `tests/brokers/test_fees.py`

**Interfaces:**
- Consumes: `_buy(model, shares, value)` / `_sell(model, shares, value)` helpers and `TASTYTRADE` in `tests/brokers/test_fees.py` (they take `int`; add `Decimal`-string variants or call `model.fees(...)` directly for fractional quantities).
- Produces: `_tastytrade_commission(shares: Decimal) -> Decimal`; constants `TASTYTRADE_CLEARING_PER_SHARE`, `TASTYTRADE_FRACTIONAL_CLEARING_PER_ORDER`.

- [ ] **Step 1: Write the failing tests** (replace `test_tastytrade_charges_no_commission_only_regulatory_fees`)

```python
def _fees(model, side, shares, value):
    zero = D(0)
    if side == "buy":
        return model.fees(buy_shares=D(shares), sell_shares=zero, buy_value=D(value), sell_value=zero)
    return model.fees(buy_shares=zero, sell_shares=D(shares), buy_value=zero, sell_value=D(value))

def test_tastytrade_whole_shares_pay_per_share_clearing() -> None:
    assert _fees(TASTYTRADE, "buy", "100", "65000") == TradeFees(buy=D("0.09"), sell=D(0))   # 0.08 + CAT 0.0003
    assert _fees(TASTYTRADE, "sell", "100", "65000") == TradeFees(buy=D(0), sell=D("1.44"))  # 0.08 + SEC 1.339 + TAF 0.0195 + CAT

def test_tastytrade_fractional_order_pays_flat_clearing_instead_of_per_share() -> None:
    assert _fees(TASTYTRADE, "buy", "4.5", "450") == TradeFees(buy=D("0.11"), sell=D(0))    # 0.10 + CAT, rounded up
    assert _fees(TASTYTRADE, "sell", "4.5", "450") == TradeFees(buy=D(0), sell=D("0.12"))   # 0.10 + SEC 0.00927 + TAF + CAT

def test_tastytrade_tiny_fraction_still_pays_the_flat_fee() -> None:
    assert _fees(TASTYTRADE, "buy", "0.00001", "0.01") == TradeFees(buy=D("0.11"), sell=D(0))

def test_tastytrade_whole_quantity_with_trailing_zeros_is_not_fractional() -> None:
    assert _fees(TASTYTRADE, "buy", "10.000", "1000") == TradeFees(buy=D("0.01"), sell=D(0))  # 0.008 + CAT

def test_tastytrade_per_share_clearing_scales_with_size() -> None:
    assert _fees(TASTYTRADE, "buy", "10000", "500000") == TradeFees(buy=D("8.03"), sell=D(0))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brokers/test_fees.py -k tastytrade -v`
Expected: the new tests FAIL (current Tastytrade fee is regulatory only, e.g. buy `0.01` instead of `0.09`).

- [ ] **Step 3: Implement `_tastytrade_commission(shares: Decimal) -> Decimal`**

`TASTYTRADE_FRACTIONAL_CLEARING_PER_ORDER` when `shares % 1 != 0`, else `TASTYTRADE_CLEARING_PER_SHARE * shares`. Update the module docstring: Tastytrade rates as of 2026-10-28 (Commissions & Fees), and that FINRA TAF is kept although Tastytrade waives it 2026-10-01..2026-12-31.

- [ ] **Step 4: Run the fee and backtest tests**

Run: `uv run pytest tests/brokers/test_fees.py tests/backtesting -q`
Expected: all pass.

- [ ] **Step 5: Update `backtesting/CLAUDE.md`** (the fee bullet, as in Files above).

- [ ] **Step 6: Quality gate**

Run: `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest --cov -q && uv run interrogate -v`
Expected: exit 0, 0 pyright errors, coverage >= 80%, interrogate >= 80% (interrogate crashed on clean main before: report, don't fix unasked).

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/brokers/fees.py src/trading_agent_framework/backtesting/CLAUDE.md tests/brokers/test_fees.py
git commit -m "feat(fees): Tastytrade clearing fees in backtests"
```

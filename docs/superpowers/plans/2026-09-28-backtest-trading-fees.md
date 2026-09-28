# Broker Trading Fees in Backtesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtests charge the fees the configured broker (`BROKER`) really charges on US stocks/ETFs, with different amounts for buy and sell orders, instead of a hardcoded symmetric `commission` rate.

**Architecture:** A new pure module `brokers/fees.py` holds `TradingFeeFactory`: shared regulatory fees (SEC, FINRA TAF, CAT) plus a per-broker, per-order commission. `BacktestBroker` asks it for the fee of every fill, so the existing cash checks, cash debits and `trade_cost` records all include it. The `commission` parameter is removed from the whole backtest chain (`fills` -> `BacktestBroker` -> `run_backtest` -> `Strategy.run_backtesting` -> strategies), and `settings.json` gains per-run fee totals.

**Tech Stack:** Python 3.14, `Decimal`, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-28-backtest-trading-fees-design.md`

## Global Constraints

- Money stays `Decimal` in `brokers/fees.py`, `backtesting/fills.py` and `backtesting/broker.py`; floats only in `settings.json` (the report boundary).
- `brokers/fees.py` is pure: no I/O, imports only the standard library, `trading_agent_framework.config.env` and `trading_agent_framework.utils.errors`.
- Rates, verbatim: SEC `0.0000206` x sell value; FINRA TAF `0.000195` x shares sold, max `9.79` per order; CAT `0.000003` x shares (both sides); Alpaca commission `0`; IBKR commission `max(1.00, 0.005 x shares)` per order.
- Each side's fee is rounded **up** to the cent. A side with 0 shares costs exactly `0`.
- `BROKER` unset means Alpaca; unknown value raises `ConfigurationError` (existing `BrokerSettings.from_env`).
- Test suite stays offline. Run tests with `uv run pytest`, lint with `uv run ruff check`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A strategy that sizes a buy as `floor(cash / price)` shares can now be turned away by the fee (IBKR's $1 minimum) -- expected: submission raises `OrderValidationError` saying `(fees included)`, cash unchanged (pinned by Task 2's `test_fees_count_toward_the_cash_a_buy_needs`).
2. A same-bar rebalance (sell submitted before a buy) where the buy is funded only by the sell proceeds -- expected: the projection credits the sell's proceeds **net of its sell fee**, so a buy needing more than that is rejected (Task 2's `test_a_buy_funded_by_a_sell_is_charged_both_fees`).
3. The sell fee depends on the value actually sold -- expected: computed from the fill-time execution price (after slippage), not the submission estimate. Pinned by construction (`_fee` only ever receives the notional `_execution_terms` computed from the fill's price) and by Task 2's `test_a_sell_credits_the_notional_minus_its_own_larger_fee`, which checks the recorded fill's `trade_cost`.
4. A run that places no trades -- expected: `settings.json` still has a complete `"fees"` block of zeros with the broker name (Task 3's `test_run_backtest_records_zero_fee_totals_without_trades`).
5. A backtest env file without `BROKER` -- expected: Alpaca fees, never an error (Task 1's `test_from_env_defaults_to_alpaca` and Task 2's `test_run_backtesting_resolves_fees_from_broker`).

---

### Task 1: `TradingFeeFactory` (pure fee model)

**Files:**
- Create: `src/trading_agent_framework/brokers/fees.py`
- Create: `tests/brokers/test_fees.py`
- Modify: `src/trading_agent_framework/config/env.py:117-118` (`BrokerKind` docstring)
- Modify: `src/trading_agent_framework/brokers/factory.py:1-5` (module docstring)

**Interfaces:**
- Consumes: `BrokerKind`, `BrokerSettings.from_env(env)` from `config/env.py`; `ConfigurationError` from `utils/errors.py`.
- Produces:
  - `TradeFees(buy: Decimal, sell: Decimal)` -- frozen dataclass.
  - `TradingFeeFactory(broker: BrokerKind)`; `TradingFeeFactory.from_env(env: Mapping[str, str] | None = None) -> TradingFeeFactory`; property `broker -> BrokerKind`; `fees(*, buy_shares: Decimal, sell_shares: Decimal, buy_value: Decimal, sell_value: Decimal) -> TradeFees`.
  - Module dict `COMMISSIONS: dict[BrokerKind, Callable[[Decimal], Decimal]]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/test_fees.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers import fees
from trading_agent_framework.brokers.fees import TradeFees, TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
from trading_agent_framework.utils.errors import ConfigurationError

D = Decimal
ALPACA = TradingFeeFactory(BrokerKind.ALPACA)
IBKR = TradingFeeFactory(BrokerKind.IBKR)


def _buy(model: TradingFeeFactory, shares: int, value: int) -> TradeFees:
    return model.fees(buy_shares=D(shares), sell_shares=D(0), buy_value=D(value), sell_value=D(0))


def _sell(model: TradingFeeFactory, shares: int, value: int) -> TradeFees:
    return model.fees(buy_shares=D(0), sell_shares=D(shares), buy_value=D(0), sell_value=D(value))


def test_alpaca_buy_pays_only_the_cat_fee_rounded_up_to_the_cent() -> None:
    assert _buy(ALPACA, 100, 65000) == TradeFees(buy=D("0.01"), sell=D(0))  # CAT 0.0003


def test_alpaca_sell_pays_sec_taf_and_cat() -> None:
    # SEC 65000 * 0.0000206 = 1.339, TAF 100 * 0.000195 = 0.0195, CAT 0.0003 -> 1.3588
    assert _sell(ALPACA, 100, 65000) == TradeFees(buy=D(0), sell=D("1.36"))


def test_ibkr_buy_pays_the_one_dollar_minimum_below_200_shares() -> None:
    assert _buy(IBKR, 100, 65000).buy == D("1.01")  # max(1, 0.5) + CAT 0.0003


def test_ibkr_buy_pays_half_a_cent_a_share_above_200_shares() -> None:
    assert _buy(IBKR, 1000, 650000).buy == D("5.01")  # 5 + CAT 0.003


def test_ibkr_sell_adds_the_regulatory_fees_to_its_commission() -> None:
    assert _sell(IBKR, 100, 65000).sell == D("2.36")  # 1 + 1.339 + 0.0195 + 0.0003


def test_the_taf_is_capped_per_order() -> None:
    # TAF would be 100,000 * 0.000195 = 19.5, capped at 9.79; CAT 0.3; SEC on value 0 is 0
    assert _sell(ALPACA, 100000, 0).sell == D("10.09")


def test_a_side_without_shares_costs_nothing_even_at_ibkr() -> None:
    zero = D(0)
    assert IBKR.fees(buy_shares=zero, sell_shares=zero, buy_value=zero, sell_value=zero) == TradeFees(buy=zero, sell=zero)


def test_both_sides_in_one_call_are_two_orders() -> None:
    result = IBKR.fees(buy_shares=D(10), sell_shares=D(10), buy_value=D(1000), sell_value=D(1000))
    # buy: 1 + 0.00003; sell: 1 + 0.0206 + 0.00195 + 0.00003 = 1.02258
    assert result == TradeFees(buy=D("1.01"), sell=D("1.03"))


@pytest.mark.parametrize("field", ["buy_shares", "sell_shares", "buy_value", "sell_value"])
def test_negative_inputs_are_rejected(field: str) -> None:
    kwargs = {"buy_shares": D(0), "sell_shares": D(0), "buy_value": D(0), "sell_value": D(0), field: D(-1)}
    with pytest.raises(ValueError, match=field):
        ALPACA.fees(**kwargs)


def test_from_env_defaults_to_alpaca() -> None:
    assert TradingFeeFactory.from_env({}).broker is BrokerKind.ALPACA


def test_from_env_reads_broker() -> None:
    assert TradingFeeFactory.from_env({"BROKER": "ibkr"}).broker is BrokerKind.IBKR


def test_from_env_rejects_an_unknown_broker() -> None:
    with pytest.raises(ConfigurationError, match="Unknown BROKER"):
        TradingFeeFactory.from_env({"BROKER": "foo"})


def test_a_broker_without_a_fee_schedule_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(fees.COMMISSIONS, BrokerKind.IBKR)
    with pytest.raises(ConfigurationError, match="BROKER=ibkr"):
        TradingFeeFactory(BrokerKind.IBKR)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/test_fees.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.fees'`.

- [ ] **Step 3: Write the implementation**

Create `src/trading_agent_framework/brokers/fees.py`:

```python
"""Simulated broker fees for backtesting US stocks/ETFs, chosen by `BROKER` (`alpaca` default, `ibkr`).

Pure: no I/O. Rates as of 2026-09-28 (Alpaca Brokerage Fee Schedule rev. 2026-09-17, SEC Fee
Rate Advisory 2026-2). They are constants, so a backtest of an earlier period pays today's
rates (the SEC fee was 0 before 2026-04-04). Every rounding and omission errs toward higher fees.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal

from trading_agent_framework.config.env import BrokerKind, BrokerSettings
from trading_agent_framework.utils.errors import ConfigurationError

SEC_FEE_RATE = Decimal("0.0000206")  # x sell value
FINRA_TAF_PER_SHARE = Decimal("0.000195")  # x shares sold
FINRA_TAF_MAX_PER_ORDER = Decimal("9.79")
CAT_FEE_PER_SHARE = Decimal("0.000003")  # x shares, buys and sells
IBKR_FIXED_PER_SHARE = Decimal("0.005")
IBKR_FIXED_MIN_PER_ORDER = Decimal("1.00")

_CENT = Decimal("0.01")


def _alpaca_commission(shares: Decimal) -> Decimal:
    return Decimal(0)


def _ibkr_commission(shares: Decimal) -> Decimal:
    # IBKR Pro Fixed; its 1%-of-trade-value cap is deliberately not applied (worst case).
    return max(IBKR_FIXED_MIN_PER_ORDER, IBKR_FIXED_PER_SHARE * shares)


COMMISSIONS: dict[BrokerKind, Callable[[Decimal], Decimal]] = {
    BrokerKind.ALPACA: _alpaca_commission,
    BrokerKind.IBKR: _ibkr_commission,
}


@dataclass(frozen=True, slots=True)
class TradeFees:
    """USD fees for the buy side and the sell side of one `TradingFeeFactory.fees` call."""

    buy: Decimal
    sell: Decimal


class TradingFeeFactory:
    """One broker's fee model: its per-order commission plus the regulatory fees every broker passes on."""

    def __init__(self, broker: BrokerKind) -> None:
        commission = COMMISSIONS.get(broker)
        if commission is None:
            raise ConfigurationError(f"No trading fee schedule for BROKER={broker.value}")
        self._broker = broker
        self._commission = commission

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TradingFeeFactory:
        """The fee model of the broker named by `BROKER`."""
        return cls(BrokerSettings.from_env(env).kind)

    @property
    def broker(self) -> BrokerKind:
        return self._broker

    def fees(self, *, buy_shares: Decimal, sell_shares: Decimal, buy_value: Decimal, sell_value: Decimal) -> TradeFees:
        """Each side with shares is one order; each side's total is rounded up to the cent."""
        for name, value in (
            ("buy_shares", buy_shares), ("sell_shares", sell_shares),
            ("buy_value", buy_value), ("sell_value", sell_value),
        ):
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")
        buy = sell = Decimal(0)
        if buy_shares > 0:
            buy = _ceil_cent(self._commission(buy_shares) + CAT_FEE_PER_SHARE * buy_shares)
        if sell_shares > 0:
            sell = _ceil_cent(
                self._commission(sell_shares)
                + SEC_FEE_RATE * sell_value
                + min(FINRA_TAF_PER_SHARE * sell_shares, FINRA_TAF_MAX_PER_ORDER)
                + CAT_FEE_PER_SHARE * sell_shares
            )
        return TradeFees(buy=buy, sell=sell)


def _ceil_cent(amount: Decimal) -> Decimal:
    return amount.quantize(_CENT, rounding=ROUND_CEILING)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/test_fees.py tests/brokers/test_lazy_imports.py -v`
Expected: all PASS.

- [ ] **Step 5: Update the two docstrings that say `BROKER` is meaningless in backtesting**

In `src/trading_agent_framework/config/env.py`, replace the `BrokerKind` docstring:

```python
class BrokerKind(StrEnum):
    """Which broker trades in paper/live mode (`BROKER`). In backtesting it only picks the
    simulated fee schedule (`brokers/fees.py`); the broker itself is always `BacktestBroker`."""
```

In `src/trading_agent_framework/brokers/factory.py`, replace the module docstring's second paragraph:

```python
"""Paper/live broker selection from the strategy env file (`BROKER=alpaca|ibkr`).

Backtests never come here (`main.py` gives them a `PlaceholderBroker`); they read `BROKER` only
for fees, via `brokers/fees.py`. Broker classes are imported inside the builders, so importing
this module stays as light as `brokers/__init__.py`.
"""
```

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check src/trading_agent_framework/brokers/fees.py tests/brokers/test_fees.py src/trading_agent_framework/config/env.py src/trading_agent_framework/brokers/factory.py`
Expected: `All checks passed!`

```bash
git add src/trading_agent_framework/brokers/fees.py tests/brokers/test_fees.py src/trading_agent_framework/config/env.py src/trading_agent_framework/brokers/factory.py
git commit -m "$(cat <<'EOF'
Task 1: TradingFeeFactory -- broker commission plus shared regulatory fees, per side

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Charge the fee model on every backtest fill (replace `commission`)

**Files:**
- Modify: `src/trading_agent_framework/backtesting/fills.py:27-30` (module docstring), `:121-128` (`apply_commission_and_slippage`)
- Modify: `src/trading_agent_framework/backtesting/broker.py:19-33` (imports), `:62-80` (constructor), `:331-390` (`_projection`, `_projected_rejection_reason`), `:477-515` (`_execution_terms`, `_rejection_reason`, `_fill`)
- Modify: `src/trading_agent_framework/backtesting/runner.py:68-90` (`run_backtest` signature), `:159`, `:190-205` (`_run` signature), `:250-260` (`BacktestBroker(...)` call), `:336` (settings `"commission"` key)
- Modify: `src/trading_agent_framework/core/strategy.py:18-40` (imports), `:470-560` (`run_backtesting`)
- Modify: `src/trading_agent_framework/strategies/news_builtin/agent_news_binary.py:179`, `:336`
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py:546` (and its now-unused `from decimal import Decimal` import)
- Test: `tests/backtesting/test_fills.py`, `tests/backtesting/test_broker_fills.py`, `tests/backtesting/test_runner.py`, `tests/core/test_runners.py`, `tests/strategies/test_news_builtin.py:463-474`

**Interfaces:**
- Consumes (Task 1): `TradingFeeFactory(broker)`, `TradingFeeFactory.from_env()`, `.broker`, `.fees(*, buy_shares, sell_shares, buy_value, sell_value) -> TradeFees` with `.buy`/`.sell`.
- Produces:
  - `fills.apply_slippage(price: Decimal, side: OrderSide, *, slippage: Decimal) -> Decimal` (replaces `apply_commission_and_slippage`).
  - `BacktestBroker(..., fees: TradingFeeFactory | None = None, slippage=..., ...)`, stored as `self._fees`; `None` means no fees.
  - `run_backtest(..., fees: TradingFeeFactory | None = None, ...)` and `_run(..., fees: TradingFeeFactory | None, ...)` -- the `commission` parameter no longer exists anywhere.
  - `Strategy.run_backtesting(..., fees: TradingFeeFactory | None = None, ...)`: `None` resolves `TradingFeeFactory.from_env()`.

- [ ] **Step 1: Rewrite the fills test for `apply_slippage`**

In `tests/backtesting/test_fills.py`, change the import line to

```python
from trading_agent_framework.backtesting.fills import Bar, apply_slippage, evaluate_fill
```

and replace the whole `test_commission_and_slippage` parametrized test (the `@pytest.mark.parametrize("side,commission,slippage,...` block through the end of that function) with:

```python
@pytest.mark.parametrize(
    "side,slippage,expected_price",
    [
        (OrderSide.BUY, D(0), D(100)),
        (OrderSide.BUY, D("0.01"), D(101)),  # buys pay more
        (OrderSide.SELL, D("0.01"), D(99)),  # sells receive less
    ],
)
def test_slippage(side: OrderSide, slippage: Decimal, expected_price: Decimal) -> None:
    assert apply_slippage(D(100), side, slippage=slippage) == expected_price
```

- [ ] **Step 2: Rewrite the broker fee tests and add the new ones**

In `tests/backtesting/test_broker_fills.py`, add imports after the existing `from trading_agent_framework.backtesting.clock import BacktestClock` line:

```python
from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
```

and a module constant after `DAY3 = ...`:

```python
IBKR_FEES = TradingFeeFactory(BrokerKind.IBKR)
ALPACA_FEES = TradingFeeFactory(BrokerKind.ALPACA)
```

Replace `test_commission_and_slippage_reduce_cash_beyond_the_raw_notional` with:

```python
def test_fees_and_slippage_reduce_cash_beyond_the_raw_notional() -> None:
    source = FakeBacktestDataSource()
    df = make_close_indexed_frame([100.0, 100.0], start=DAY1, freq="1D")
    source.set_bars(AAPL, df)
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker(
        "momentum", data_source=source, clock=clock, budget=Decimal(10000),
        fees=IBKR_FEES, slippage=Decimal("0.01"),
    )
    clock.on_advance = broker.on_advance
    broker.submit_order(Order(strategy_name="momentum", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    clock._now = DAY2
    broker.on_advance(DAY1, DAY2)

    # execution_price = 100 * 1.01 = 101; IBKR buy of 10 shares = max(1, 0.05) + CAT 0.00003 -> 1.01
    expected_cash = Decimal(10000) - (Decimal(10) * Decimal("101.00") + Decimal("1.01"))
    assert broker._cash == expected_cash
    assert broker.ledger.fills[0].trade_cost == Decimal("1.01")
```

Change the `_account` helper to take a fee model:

```python
def _account(closes: list[float], *, budget: Decimal, fees: TradingFeeFactory | None = None) -> tuple[BacktestBroker, BacktestClock]:
    source = FakeBacktestDataSource()
    source.set_bars(AAPL, make_close_indexed_frame(closes, start=DAY1, freq="1D"))
    clock = BacktestClock(start=DAY1, sessions=[])
    broker = BacktestBroker("momentum", data_source=source, clock=clock, budget=budget, fees=fees)
    clock.on_advance = broker.on_advance
    return broker, clock
```

Replace `test_commission_counts_toward_the_cash_a_buy_needs` with:

```python
def test_fees_count_toward_the_cash_a_buy_needs() -> None:
    # 10 shares at 100 = 1000 notional + IBKR's 1.01 buy fee = 1001.01 needed. At flat prices the
    # submission-time estimate equals the fill price, so a budget that covers only the notional
    # is turned away at submission.
    covered, covered_clock = _account([100.0, 100.0], budget=Decimal("1001.01"), fees=IBKR_FEES)
    covered_order = covered.submit_order(_order(OrderSide.BUY, 10))
    _advance(covered, covered_clock, DAY1, DAY2)

    notional_only, _ = _account([100.0, 100.0], budget=Decimal(1001), fees=IBKR_FEES)

    assert covered_order.status is OrderStatus.FILL
    assert covered._cash == Decimal(0)
    with pytest.raises(OrderValidationError, match=r"needs about 1001\.01 \(fees included\)"):
        notional_only.submit_order(_order(OrderSide.BUY, 10))
    assert notional_only._cash == Decimal(1001)


def test_a_sell_credits_the_notional_minus_its_own_larger_fee() -> None:
    # Alpaca: a buy pays only CAT (0.00003 -> 0.01); a sell adds SEC + TAF
    # (0.0206 + 0.00195 + 0.00003 = 0.02258 -> 0.03).
    broker, clock = _account([100.0, 100.0, 100.0], budget=Decimal(10000), fees=ALPACA_FEES)
    broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY1, DAY2)
    broker.submit_order(_order(OrderSide.SELL, 10))
    _advance(broker, clock, DAY2, DAY3)

    assert [fill.trade_cost for fill in broker.ledger.fills] == [Decimal("0.01"), Decimal("0.03")]
    assert broker._cash == Decimal(10000) - Decimal("0.01") - Decimal("0.03")


@pytest.mark.parametrize("leftover,accepted", [(Decimal("2.04"), True), (Decimal("2.03"), False)])
def test_a_buy_funded_by_a_sell_is_charged_both_fees(leftover: Decimal, accepted: bool) -> None:
    # Hold 10 shares (bought for 1000 + 1.01 IBKR fee) with `leftover` cash, then sell them and
    # buy 10 again in the same iteration. The pending sell credits 1000 - its 1.03 fee
    # (1 + SEC 0.0206 + TAF 0.00195 + CAT 0.00003 -> 1.03), and the buy needs 1001.01, so it fits
    # only when leftover >= 2.04. Without netting the sell fee, 2.03 would wrongly fit too.
    broker, clock = _account([100.0, 100.0, 100.0], budget=Decimal("1001.01") + leftover, fees=IBKR_FEES)
    broker.submit_order(_order(OrderSide.BUY, 10))
    _advance(broker, clock, DAY1, DAY2)
    assert broker._cash == leftover

    broker.submit_order(_order(OrderSide.SELL, 10))
    if accepted:
        broker.submit_order(_order(OrderSide.BUY, 10))
        _advance(broker, clock, DAY2, DAY3)
        assert broker._cash == Decimal(0)  # 2.04 + (1000 - 1.03) - (1000 + 1.01)
    else:
        with pytest.raises(OrderValidationError, match=r"needs about 1001\.01 \(fees included\)"):
            broker.submit_order(_order(OrderSide.BUY, 10))
```

- [ ] **Step 3: Update the runner, core and strategy tests to the new parameter**

Run: `sed -i 's/commission=Decimal(0), //' tests/backtesting/test_runner.py`
Then: `grep -n commission tests/backtesting/test_runner.py` -- expected: no output.

In `tests/strategies/test_news_builtin.py`, rename `test_run_backtesting_wires_alpaca_data_preload_and_fees` to `test_run_backtesting_wires_alpaca_data_and_preload` and delete its line
`assert captured["commission"] == Decimal(0)  # Alpaca charges no commission on US ETFs`, then add in its place:

```python
    assert "fees" not in captured  # left to Strategy.run_backtesting, which reads BROKER
```

Add to the end of `tests/core/test_runners.py`:

```python
def test_run_backtesting_resolves_fees_from_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.backtesting.fakes import FakeBacktestDataSource, make_close_indexed_frame

    from trading_agent_framework.config.env import BrokerKind

    sessions = weekday_sessions(date(2026, 1, 5), 3)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(Asset("SPY"), make_close_indexed_frame([150.0, 151.0, 152.0], start=sessions[0].close, freq="1D"))
    monkeypatch.setenv("BROKER", "ibkr")

    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.run_backtesting(start=sessions[0].open, end=sessions[-1].close, data_source=source, benchmark="SPY")

    assert isinstance(strategy.broker, BacktestBroker)
    assert strategy.broker._fees is not None
    assert strategy.broker._fees.broker is BrokerKind.IBKR
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/backtesting/test_fills.py tests/backtesting/test_broker_fills.py tests/backtesting/test_runner.py tests/core/test_runners.py -q`
Expected: FAIL -- `ImportError: cannot import name 'apply_slippage'`, and `TypeError: ... unexpected keyword argument 'fees'`.

- [ ] **Step 5: Implement `apply_slippage` in `fills.py`**

Replace the module-docstring paragraph (lines 27-30, "Commission is a fraction of trade notional ... receive price * (1 - slippage).") with:

```
Slippage is a fraction applied against the trader: buys pay price * (1 + slippage), sells
receive price * (1 - slippage). Fees are not applied here: `BacktestBroker` asks its
`TradingFeeFactory` (`brokers/fees.py`) for each order's fee.
```

Replace `apply_commission_and_slippage` with:

```python
def apply_slippage(price: Decimal, side: OrderSide, *, slippage: Decimal) -> Decimal:
    """The execution price after slippage, which always goes against the trader."""
    return price * (1 + slippage) if side is OrderSide.BUY else price * (1 - slippage)
```

- [ ] **Step 6: Charge the fee model in `BacktestBroker`**

In `src/trading_agent_framework/backtesting/broker.py`, add the import after `from trading_agent_framework.brokers.base import Broker`:

```python
from trading_agent_framework.brokers.fees import TradingFeeFactory
```

In `__init__`, replace the parameter `commission: Decimal = Decimal(0),` with `fees: TradingFeeFactory | None = None,` and the assignment `self._commission = commission` with `self._fees = fees`.

Replace `_execution_terms` with these two methods:

```python
    def _execution_terms(self, order: Order, raw_price: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        """`(execution_price, fee, notional)` for filling `order` at `raw_price`."""
        assert order.quantity is not None  # notional orders are rejected at submission
        execution_price = fills.apply_slippage(raw_price, order.side, slippage=self._slippage)
        notional = execution_price * order.quantity
        return execution_price, self._fee(order.side, order.quantity, notional), notional

    def _fee(self, side: OrderSide, quantity: Decimal, notional: Decimal) -> Decimal:
        """The broker's fee for one order -- a backtest fill is always the whole order."""
        if self._fees is None:
            return Decimal(0)
        zero = Decimal(0)
        if side is OrderSide.BUY:
            return self._fees.fees(buy_shares=quantity, sell_shares=zero, buy_value=notional, sell_value=zero).buy
        return self._fees.fees(buy_shares=zero, sell_shares=quantity, buy_value=zero, sell_value=notional).sell
```

In `_projection`, `_projected_rejection_reason`, `_rejection_reason` and `_fill`, rename the local `commission_cost` to `fee` (every occurrence, including `trade_cost=fee` in `_fill`), and change both message fragments `(commission included)` to `(fees included)`.

Then: `grep -n commission src/trading_agent_framework/backtesting/broker.py src/trading_agent_framework/backtesting/fills.py` -- expected: no output.

- [ ] **Step 7: Replace `commission` with `fees` in the runner**

In `src/trading_agent_framework/backtesting/runner.py`:
- add `from trading_agent_framework.brokers.fees import TradingFeeFactory` after the `from trading_agent_framework.backtesting.warmup import warmup_calendar_days` line;
- in `run_backtest`'s signature replace `commission: Decimal,` with `fees: TradingFeeFactory | None = None,`;
- in the `_run(...)` call inside `run_backtest` replace `commission=commission,` with `fees=fees,`;
- in `_run`'s signature replace `commission: Decimal,` with `fees: TradingFeeFactory | None = None,`;
- in the `BacktestBroker(...)` call replace `commission=commission,` with `fees=fees,`;
- delete the settings line `"commission": float(commission),` (Task 3 adds the `"fees"` block).

Then: `grep -n commission src/trading_agent_framework/backtesting/runner.py` -- expected: no output.

- [ ] **Step 8: Replace `commission` with `fees` in `Strategy.run_backtesting`**

In `src/trading_agent_framework/core/strategy.py`, add after `from trading_agent_framework.brokers.base import Broker`:

```python
from trading_agent_framework.brokers.fees import TradingFeeFactory
```

In `run_backtesting`'s signature replace `commission: Number = Decimal(0),` with `fees: TradingFeeFactory | None = None,`. Replace the docstring line starting `commission: per-trade commission (default 0)...` with:

```
            fees: the broker fee model (`brokers/fees.py`) charged on every fill -- different
                for buys and sells, per order. Defaults to `TradingFeeFactory.from_env()`: the
                broker named by `BROKER` in the backtest env file (Alpaca when unset).
```

In the `run_backtest(...)` call replace `commission=_to_decimal(commission),` with:

```python
            fees=fees if fees is not None else TradingFeeFactory.from_env(),
```

- [ ] **Step 9: Stop the strategies from passing a commission**

In `src/trading_agent_framework/strategies/news_builtin/agent_news_binary.py`, delete the parameters entry `"commission": 0.0,  # Commission is 0 on US ETF (Alpaca)` and the kwarg line `commission=Decimal(str(self.parameters["commission"])),`.

In `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py`, delete the kwarg line `commission=Decimal("0.001"),` and the now-unused `from decimal import Decimal` import.

Then: `grep -rn "commission=" src/` -- expected: no output.

- [ ] **Step 10: Run the full suite and lint**

Run: `uv run pytest -q`
Expected: all PASS.
Run: `uv run ruff check`
Expected: `All checks passed!`

- [ ] **Step 11: Commit**

```bash
git add src/trading_agent_framework/backtesting/fills.py src/trading_agent_framework/backtesting/broker.py src/trading_agent_framework/backtesting/runner.py src/trading_agent_framework/core/strategy.py src/trading_agent_framework/strategies/news_builtin/agent_news_binary.py src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/backtesting/test_fills.py tests/backtesting/test_broker_fills.py tests/backtesting/test_runner.py tests/core/test_runners.py tests/strategies/test_news_builtin.py
git commit -m "$(cat <<'EOF'
Task 2: charge the BROKER fee model on every backtest fill, replacing commission

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Per-run fee totals in `settings.json`, and docs

**Files:**
- Modify: `src/trading_agent_framework/backtesting/runner.py` (settings dict in `_run`; new helper `_fee_totals` after `_run`)
- Modify: `README.md:132`
- Modify: `CLAUDE.md` (architecture list after the `brokers/factory.py` line; `BacktestBroker` gotcha at line 91; new gotcha)
- Test: `tests/backtesting/test_runner.py`

**Interfaces:**
- Consumes (Task 2): `run_backtest(..., fees: TradingFeeFactory | None = None, ...)`; `broker.ledger.fills: list[FillRecord]` whose `trade_cost` is the order's fee; `TradingFeeFactory.broker`.
- Produces: `settings["fees"]` = `{"broker": str | None, "buy_orders": int, "sell_orders": int, "buy_shares": float, "sell_shares": float, "buy_value": float, "sell_value": float, "buy_fees": float, "sell_fees": float}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/backtesting/test_runner.py` imports:

```python
from trading_agent_framework.brokers.fees import TradingFeeFactory
from trading_agent_framework.config.env import BrokerKind
```

and append:

```python
def _four_session_source() -> tuple[FakeBacktestDataSource, list[MarketSession]]:
    sessions = _sessions(date(2026, 1, 5), 4)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(AAPL, _bars(sessions, [150.0, 151.0, 152.0, 153.0]))
    source.set_bars(SPY, _bars(sessions, [400.0, 402.0, 401.0, 405.0]))
    return source, sessions


def test_run_backtest_records_fee_totals_in_settings(tmp_path: Path) -> None:
    source, sessions = _four_session_source()
    start = sessions[0].open - timedelta(hours=1)
    result = run_backtest(
        _placeholder_strategy(BuyOnceStrategy, tmp_path, start), start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        fees=TradingFeeFactory(BrokerKind.IBKR), slippage=Decimal(0), risk_free_rate=0.0,
    )

    [fill] = pd.read_parquet(result.run_dir / "trades.parquet").to_dict("records")
    expected = {
        "broker": "ibkr", "buy_orders": 1, "sell_orders": 0,
        "buy_shares": 5.0, "sell_shares": 0.0,
        "buy_value": pytest.approx(fill["price"] * 5), "sell_value": 0.0,
        "buy_fees": 1.01, "sell_fees": 0.0,  # IBKR: max(1, 5 * 0.005) + CAT 0.000015
    }
    assert result.settings["fees"] == expected
    assert json.loads((result.run_dir / "settings.json").read_text())["fees"] == expected
    assert "commission" not in result.settings


def test_run_backtest_records_zero_fee_totals_without_trades(tmp_path: Path) -> None:
    source, sessions = _four_session_source()
    start = sessions[0].open - timedelta(hours=1)
    result = run_backtest(
        _placeholder_strategy(Strategy, tmp_path, start), start=start, end=sessions[-1].close,
        budget=Decimal(10000), data_source=source, benchmark="SPY", timestep="day",
        slippage=Decimal(0), risk_free_rate=0.0,
    )

    assert result.settings["fees"] == {
        "broker": None, "buy_orders": 0, "sell_orders": 0,
        "buy_shares": 0.0, "sell_shares": 0.0, "buy_value": 0.0, "sell_value": 0.0,
        "buy_fees": 0.0, "sell_fees": 0.0,
    }
```

(`Strategy` is concrete; its default `on_trading_iteration` places no order.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backtesting/test_runner.py -k fee_totals -v`
Expected: FAIL with `KeyError: 'fees'`.

- [ ] **Step 3: Implement the totals**

In `src/trading_agent_framework/backtesting/runner.py`, add imports:

```python
from trading_agent_framework.backtesting.ledger import FillRecord
from trading_agent_framework.entities.enums import OrderSide
```

In `_run`'s `settings` dict, where the `"commission"` line used to be (after `"sleeptime": strategy.sleeptime,`), add:

```python
        "fees": _fee_totals(broker.ledger.fills, fees),
```

Add after `_run`:

```python
def _fee_totals(fills: Sequence[FillRecord], fees: TradingFeeFactory | None) -> dict[str, Any]:
    """Per-side order count, shares, traded value and fees over the run (floats: report boundary)."""
    totals: dict[str, Any] = {"broker": None if fees is None else fees.broker.value}
    for side in (OrderSide.BUY, OrderSide.SELL):
        side_fills = [fill for fill in fills if fill.side is side]
        totals[f"{side.value}_orders"] = len(side_fills)
        totals[f"{side.value}_shares"] = float(sum((fill.filled_quantity for fill in side_fills), Decimal(0)))
        totals[f"{side.value}_value"] = float(sum((fill.price * fill.filled_quantity for fill in side_fills), Decimal(0)))
        totals[f"{side.value}_fees"] = float(sum((fill.trade_cost for fill in side_fills), Decimal(0)))
    return totals
```

The dict's key order is `broker, buy_*, sell_*`; the tests compare dicts, so order does not matter.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backtesting/test_runner.py tests/dashboard -q`
Expected: all PASS (the dashboard never reads `"commission"`: `grep -rn commission src/trading_agent_framework/dashboard` prints nothing).

- [ ] **Step 5: Update README and CLAUDE.md**

In `README.md` line 132, replace `` `data_source`, `timestep` (`"day"` by default), `commission`, `slippage` and `` with `` `data_source`, `timestep` (`"day"` by default), `fees`, `slippage` and ``, and append this paragraph right after that paragraph (after `` `risk_free_rate` keyword arguments.``):

```markdown
Fees default to the broker named by `BROKER` in the backtest env file (`alpaca` when unset,
`ibkr`): `brokers/fees.py`'s `TradingFeeFactory` charges each order the broker's commission
(Alpaca $0, IBKR Pro Fixed max($1, $0.005/share)) plus the regulatory fees every broker passes
on (SEC and FINRA TAF on sells, CAT on both sides). The run's totals are in `settings.json["fees"]`.
```

In `CLAUDE.md`, add after the `brokers/factory.py` architecture line:

```markdown
- `brokers/fees.py` -- **pure** `TradingFeeFactory`: the backtest fee model named by `BROKER` (per-order broker commission + shared SEC/FINRA TAF/CAT regulatory fees), different for buys and sells.
```

In the `BacktestBroker is a long-only cash account` gotcha, change `(commission included)` to `(fees included)`, and add a new gotcha after it:

```markdown
- **Backtest fees come from `BROKER`, per order, and differ by side.** `Strategy.run_backtesting(fees=None)` resolves `TradingFeeFactory.from_env()`, so the backtest env file's `BROKER` (Alpaca when unset) picks the schedule; `BacktestBroker(fees=None)` (tests) charges nothing. A sell pays SEC (on value) + FINRA TAF (capped per order) + CAT; a buy pays CAT only -- plus the broker's commission on both (IBKR `max($1, $0.005/share)`, Alpaca $0). Each side rounds up to the cent. Rates are dated constants in `brokers/fees.py` (2026-09-28): an older backtest period pays today's rates. There is no `commission` parameter any more.
```

- [ ] **Step 6: Full suite, lint, commit**

Run: `uv run pytest -q` -- expected: all PASS.
Run: `uv run ruff check` -- expected: `All checks passed!`

```bash
git add src/trading_agent_framework/backtesting/runner.py tests/backtesting/test_runner.py README.md CLAUDE.md
git commit -m "$(cat <<'EOF'
Task 3: per-run fee totals in settings.json; document BROKER-driven backtest fees

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

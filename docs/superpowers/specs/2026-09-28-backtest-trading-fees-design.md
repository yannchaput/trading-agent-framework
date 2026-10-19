# Broker trading fees in backtesting -- design

Date: 2026-09-28
Status: approved (brainstorming), pending implementation plan

## 1. Goal and scope

Backtests charge the fees the configured broker would really charge on US stocks/ETFs,
with a different amount for buy and sell orders, instead of the single symmetric
`commission` rate each strategy hardcodes today.

In scope:

- `TradingFeeFactory` in a new pure module `brokers/fees.py`, selected by the existing
  `BROKER` env var (`alpaca` default, `ibkr`).
- Replacing the `commission: Decimal` parameter (fraction of notional) with a fee model in
  `BacktestBroker`, `runner.run_backtest` and `Strategy.run_backtesting`.
- Per-run fee totals in `settings.json`.
- Wiring `news_binary` and `cross_momentum` through the factory (their literals are removed).

Out of scope: live/paper fees (real brokers charge them for real), IBKR Tiered pricing,
IBKR Lite, Alpaca Elite Smart Router, historical fee rates, dashboard display of the totals.

## 2. Fee schedule (researched 2026-09-28)

Sources: Alpaca Brokerage Fee Schedule (PDF revised 2026-09-17, read in full), SEC Fee
Rate Advisory 2026-2, FINRA TAF guidance; IBKR figures from search excerpts of IBKR's own
pricing pages plus bankeronwheels.com (interactivebrokers.com returns 403 to automated
fetches).

**Regulatory fees -- identical for both brokers (shared code):**

| Fee | Side | Rate |
|---|---|---|
| SEC Section 31 transaction fee | sell | `0.0000206` x sell value ($20.60 per $1M, since 2026-04-04) |
| FINRA Trading Activity Fee (TAF) | sell | `0.000195` x shares sold, max `9.79` per order |
| FINRA Consolidated Audit Trail (CAT) | buy and sell | `0.000003` x shares |

**Broker commission, per order, same on both sides:**

| Broker | Commission |
|---|---|
| Alpaca (self-directed API account) | `0` |
| IBKR Pro Fixed | `max(1.00, 0.005 x shares)` -- worst case: the 1%-of-trade-value cap is not applied |

Known simplifications, all erring toward higher fees or negligible:

- Rates are today's constants. The SEC rate was `0` before 2026-04-04 and differed in
  earlier fiscal years; FINRA proposes pausing TAF 2026-10-01..2026-12-31. Neither is modelled.
- Each side's fee is rounded **up** to the cent per order (Alpaca rounds once per day per
  fee type, so this slightly overstates it).
- CAT for OTC equities (0.01 equivalent share) is ignored: backtests trade NMS stocks/ETFs.

## 3. Components

### 3.1 `brokers/fees.py` (new, pure: no I/O, no state beyond the broker kind)

```python
SEC_FEE_RATE = Decimal("0.0000206")
FINRA_TAF_PER_SHARE = Decimal("0.000195")
FINRA_TAF_MAX_PER_ORDER = Decimal("9.79")
CAT_FEE_PER_SHARE = Decimal("0.000003")
IBKR_FIXED_PER_SHARE = Decimal("0.005")
IBKR_FIXED_MIN_PER_ORDER = Decimal("1.00")


@dataclass(frozen=True, slots=True)
class TradeFees:
    buy: Decimal
    sell: Decimal


class TradingFeeFactory:
    def __init__(self, broker: BrokerKind) -> None: ...
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TradingFeeFactory:
        """The fee model of the broker named by `BROKER` (via `BrokerSettings.from_env`)."""

    @property
    def broker(self) -> BrokerKind: ...
    def fees(self, *, buy_shares: Decimal, sell_shares: Decimal, buy_value: Decimal, sell_value: Decimal) -> TradeFees: ...
```

`fees()` treats each side with a non-zero share count as **one order**:

- `buy = ceil_cent(commission(buy_shares) + CAT x buy_shares)`
- `sell = ceil_cent(commission(sell_shares) + SEC x sell_value + min(TAF x sell_shares, 9.79) + CAT x sell_shares)`
- A side with 0 shares costs exactly `0` (no IBKR minimum).
- `commission(shares)` is dispatched from a `dict[BrokerKind, Callable[[Decimal], Decimal]]`
  (Alpaca `0`, IBKR `max(1.00, 0.005 x shares)`), mirroring `brokers/factory.py`'s `BUILDERS`.
  A `BrokerKind` missing from it raises `ConfigurationError` in `__init__`.
- `buy_value` is part of the interface as agreed; no current formula uses it.
- Negative inputs raise `ValueError`.

Imports only `config.env` and the standard library, so importing it never pulls in
`alpaca`/`ib_async` (same rule as `brokers/news.py`).

### 3.2 `backtesting/fills.py`

`apply_commission_and_slippage` becomes `apply_slippage(price, side, *, slippage) -> Decimal`.
Fees no longer live here. The module docstring drops the commission paragraph.

### 3.3 `backtesting/broker.py` (`BacktestBroker`)

- Constructor: `commission: Decimal = Decimal(0)` is replaced by
  `fees: TradingFeeFactory | None = None`. `None` means no fees (the default the existing
  tests rely on).
- `_execution_terms(order, raw_price)` keeps returning `(execution_price, fee, notional)`.
  `fee` now comes from `fees.fees(...)` with that order's side filled in and the other side
  zero (`.buy` or `.sell`). One backtest fill is always the whole order, so "one call = one
  order" holds.
- `_projection`, `_projected_rejection_reason`, `_rejection_reason` and `_fill` are unchanged
  apart from variable names: they already add the fee to what a buy needs and subtract it
  from what a sell brings in. Messages say `(fees included)` instead of `(commission included)`.
- `FillRecord.trade_cost` keeps recording the per-order fee, so `trades.parquet` is unchanged.

### 3.4 `backtesting/runner.py`

- `run_backtest(..., fees: TradingFeeFactory | None = None, ...)` replaces `commission`,
  forwarded to `BacktestBroker`.
- `settings.json`: the `"commission"` key is replaced by

```json
"fees": {"broker": "ibkr", "buy_orders": 12, "sell_orders": 9,
         "buy_shares": 1530.0, "sell_shares": 1210.0,
         "buy_value": 81234.5, "sell_value": 80321.1,
         "buy_fees": 12.4, "sell_fees": 11.2}
```

  computed from `broker.ledger.fills` by a small pure helper (sums per side of
  `filled_quantity`, `price x filled_quantity`, `trade_cost`). `"broker"` is `null` when
  `fees` is `None`. Values are floats (the report float boundary).

### 3.5 `core/strategy.py` (`Strategy.run_backtesting`)

`commission: Number = Decimal(0)` is replaced by `fees: TradingFeeFactory | None = None`.
When omitted, it resolves `TradingFeeFactory.from_env()`: the env file is already loaded
by `main.py` before a backtest starts, so `BROKER=ibkr` in
`env/.env.<strategy>.backtesting` simulates IBKR fees and no `BROKER` means Alpaca.
Docstring updated.

### 3.6 Strategies

- `news_binary`: drop the `"commission": 0.0` parameter and the `commission=` kwarg.
- `cross_momentum`: drop `commission=Decimal("0.001")`.

Both now get the fees of the `BROKER` their backtest env file names.

### 3.7 Config and docs

- `BrokerKind` docstring (`config/env.py`): in backtesting, `BROKER` selects the simulated
  fee schedule (the broker itself is always `BacktestBroker`).
- `brokers/factory.py` module docstring: note that backtests read `BROKER` only for fees.
- README (`run_backtesting` keyword list, around line 132): `commission` becomes `fees`,
  with one sentence on `BROKER`.
- CLAUDE.md: `brokers/fees.py` in the architecture list; one gotcha line saying that fees are
  per order, buy/sell asymmetric, `BROKER`-driven, and that the rates are dated constants.

## 4. Error handling

- Unknown `BROKER` value: `ConfigurationError` from `BrokerSettings.from_env`, before the
  run starts (in `Strategy.run_backtesting`).
- Negative shares/values passed to `fees()`: `ValueError` (a programming error, never user input).
- No new failure modes at fill time: fees are pure arithmetic.

## 5. Testing

- `tests/brokers/test_fees.py` (new):
  - Alpaca buy 100 shares -> CAT `0.0003` -> `0.01`.
  - Alpaca sell 100 shares worth 65,000 -> `1.339 + 0.0195 + 0.0003` -> `1.36`.
  - IBKR buy 100 shares -> `max(1, 0.5) + 0.0003` -> `1.01`; buy 1,000 shares -> `5.003` -> `5.01`.
  - TAF cap: sell 100,000 shares -> TAF contributes `9.79`, not `19.5`.
  - A zero-share side is `0` for both brokers (no IBKR minimum).
  - `from_env`: no `BROKER` -> Alpaca; `BROKER=ibkr` -> IBKR; `BROKER=foo` -> `ConfigurationError`.
  - Negative input -> `ValueError`.
- `tests/backtesting/test_fills.py`: `test_commission_and_slippage` becomes an `apply_slippage` test.
- `tests/backtesting/test_broker_fills.py`: the two commission tests are rewritten on
  `fees=TradingFeeFactory(BrokerKind.IBKR)` (e.g. 10 shares at 100 need `1000 + 1.01`),
  plus one test that a sell credits `notional - sell fee` (the asymmetric side).
- `tests/backtesting/test_runner.py`: `commission=Decimal(0)` args removed; one test asserts
  the `settings.json` `"fees"` block totals.
- `tests/strategies/test_news_builtin.py:472`: the captured `commission` assertion is removed
  (no strategy passes fees now).
- Dashboard fixtures keep their `"commission"` key: the reader ignores it, and old run
  directories still have it.

The suite stays offline; the factory reads only the given `env` mapping.

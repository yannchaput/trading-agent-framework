# cross_momentum SHV Parking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cross_momentum's exposure cuts real (trim held positions to their scaled target) and park the de-risked capital in SHV instead of idle cash.

**Architecture:** All trading logic stays in `CrossMomentumStrategy.rebalance()`: Phase 1 gains a trim of overweight target positions and an SHV sell, a new Phase 3 buys SHV with what the stock buys left. Config gets a `parking` block; `run_backtesting()` preloads SHV through a small testable helper.

**Tech Stack:** Python 3.14, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-29-cross-momentum-lowbeta-parking-design.md`

## Global Constraints

- Branch: `feature/cross-momentum-lowbeta-parking`. Stage explicit paths only.
- Replace behaviour outright — no enable flag. Ranking, filters, weights and exposure legs are untouched.
- Parking symbol `"SHV"`, `min_trade_pct` `0.01`, band `_REBALANCE_BAND = 0.20`, reserve = existing `cash_buffer_pct` (0.05).
- `shv_target_value = max(0, pv * (1 - cash_buffer_pct) - Σ stock target values - hysteresis_value)`.
- The minimum-trade rule applies to trims and SHV orders only; stock exits and stock buys keep their current rules.
- Tests never touch the network; hand-written fakes, no `MagicMock`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; subjects use `Task N: ...`.

## Review Focus

1. **SHV already held when the parking symbol is also somehow in the ranked target list** — expect the target path to own it and no double order. Unlikely (the universe is ≥$2B stocks) and not tested; reviewer should confirm the skip-by-symbol runs before the target check.
2. **Parking target of zero with an SHV holding** — expect the whole holding sold (exact `pos.quantity`, no dust left by float flooring) → pinned in Task 1 (`test_zero_parking_target_sells_the_whole_shv_holding`).
3. **SHV order rejected by the broker** (IBKR PRIIPs) — expect a logged error/warning and the rest of the rebalance to proceed → pinned in Task 1 (`test_a_rejected_shv_buy_is_logged_and_does_not_raise`).
4. **Missing SHV price** — expect no SHV order, a warning, and stock orders still placed → pinned in Task 1.
5. **Exposure rising back** — expect SHV sold before stock buys so the buys are funded → pinned in Task 1.

---

### Task 1: Trim and SHV sleeve in `rebalance()`

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/parameters.py` (add `parking`)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (`_REBALANCE_BAND` constant, `rebalance()`)
- Test: `tests/strategies/test_cross_momentum_rebalance.py` (update fake, append tests)

**Interfaces:**
- Produces: `CONFIG["parking"] == {"symbol": "SHV", "min_trade_pct": 0.01}`; module constant `_REBALANCE_BAND = 0.20`; `rebalance(target, all_ranks)` signature unchanged.

- [ ] **Step 1: Update the fake and write the failing tests**

In `tests/strategies/test_cross_momentum_rebalance.py`, replace the `FakeStrategy` class with:

```python
class FakeStrategy:
    """Just enough of `Strategy` for `rebalance`; records the orders it places."""

    def __init__(self, *, cash, positions=(), last_prices=None, cash_buffer_pct=0.05, min_trade_pct=0.01, reject=()):
        self.parameters = {
            "sell_rank_threshold": 35,
            "cash_buffer_pct": cash_buffer_pct,
            "parking": {"symbol": "SHV", "min_trade_pct": min_trade_pct},
        }
        self._cash = cash
        self._positions = list(positions)
        self._last_prices = last_prices or {}
        self._reject = set(reject)
        self.portfolio_value = cash + sum(p.quantity * self._last_prices[p.asset.symbol] for p in self._positions)
        self.orders = []
        self.warnings: list[str] = []

    def get_positions(self):
        return self._positions

    def get_cash(self):
        return self._cash

    def get_last_price(self, symbol):
        return self._last_prices.get(symbol)

    def create_order(self, symbol, quantity, side, **kwargs):
        return SimpleNamespace(symbol=symbol, quantity=quantity, side=side)

    def submit_order(self, order):
        if order.symbol in self._reject:
            raise RuntimeError(f"rejected {order.symbol}")
        self.orders.append(order)

    def log_info(self, *args, **kwargs): ...

    def log_warning(self, message, *args, **kwargs):
        self.warnings.append(message)

    def log_error(self, message, *args, **kwargs):
        self.warnings.append(message)
```

Append at the end of the file:

```python
def _held(symbol, quantity):
    return SimpleNamespace(asset=SimpleNamespace(symbol=symbol), quantity=quantity)


def _orders(fake, symbol, side):
    return [o.quantity for o in fake.orders if o.symbol == symbol and o.side == side]


def test_idle_cash_is_swept_into_shv():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert _orders(fake, "SHV", "buy") == [13.0]  # 950 - 300 = 650 parked, the 5% reserve stays cash


def test_an_exposure_drop_trims_the_position_and_parks_the_proceeds():
    fake = FakeStrategy(cash=0.0, positions=[_held("AAA", 10.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.4, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "sell") == [6.0]
    assert _orders(fake, "SHV", "buy") == [11.0]  # target 950 - 400 = 550


def test_an_exposure_rise_sells_shv_before_funding_the_stock_buys():
    fake = FakeStrategy(cash=0.0, positions=[_held("SHV", 20.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.9, 100.0, 1)], {"AAA": 1})

    assert [(o.symbol, o.side, o.quantity) for o in fake.orders] == [("SHV", "sell", 19.0), ("AAA", "buy", 9.0)]


def test_shv_within_its_band_is_left_alone_and_never_exited_as_unranked():
    fake = FakeStrategy(cash=500.0, positions=[_held("SHV", 10.0)], last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.45, 100.0, 1)], {"AAA": 1})  # SHV target 500 = held

    assert _orders(fake, "SHV", "sell") == []
    assert _orders(fake, "SHV", "buy") == []


def test_zero_parking_target_sells_the_whole_shv_holding():
    fake = FakeStrategy(cash=0.0, positions=[_held("SHV", 3.333333)], last_prices={"AAA": 100.0, "SHV": 30.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 1.0, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "SHV", "sell") == [3.333333]


def test_a_trim_below_the_minimum_trade_is_skipped():
    fake = FakeStrategy(cash=9950.0, positions=[_held("AAA", 0.5)], last_prices={"AAA": 100.0, "SHV": 50.0})

    # 50 held vs target 20: above the band, but the 30 trim is under 1% of 10 000
    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.002, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "sell") == []


def test_an_shv_buy_below_the_minimum_trade_is_skipped():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.945, 100.0, 1)], {"AAA": 1})  # SHV target 5 < 10

    assert _orders(fake, "SHV", "buy") == []


def test_a_missing_shv_price_skips_parking_but_not_the_stock_orders():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert [o for o in fake.orders if o.symbol == "SHV"] == []
    assert any("SHV" in message for message in fake.warnings)


def test_hysteresis_holdings_reduce_the_parking_target():
    prices = {"AAA": 100.0, "HYS": 50.0, "SHV": 50.0}
    fake = FakeStrategy(cash=800.0, positions=[_held("HYS", 4.0)], last_prices=prices)  # HYS worth 200

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1, "HYS": 30})

    assert _orders(fake, "HYS", "sell") == []
    assert _orders(fake, "SHV", "buy") == [9.0]  # 950 - 300 - 200 = 450


def test_a_rejected_shv_buy_is_logged_and_does_not_raise():
    fake = FakeStrategy(cash=1000.0, last_prices={"AAA": 100.0, "SHV": 50.0}, reject={"SHV"})

    CrossMomentumStrategy.rebalance(fake, [_target("AAA", 0.3, 100.0, 1)], {"AAA": 1})

    assert _orders(fake, "AAA", "buy") == [3.0]
    assert any("SHV" in message for message in fake.warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -q`
Expected: the three existing tests pass; the new ones fail (no SHV orders, no trim, `KeyError`-free because `get_last_price` now uses `.get`). `test_a_trim_below_the_minimum_trade_is_skipped` and `test_an_shv_buy_below_the_minimum_trade_is_skipped` already pass (today's code never trims or buys SHV) — expected; they pin the minimum once those orders exist. Every other new test must fail.

- [ ] **Step 3: Implement**

In `parameters.py`, add after the `"volatility_targeting"` block:

```python
    # ── Parking sleeve ─────────────────────────
    # Capital the exposure legs take out of stocks is parked in this T-bill ETF instead of idle cash.
    # Trims and parking orders smaller than min_trade_pct of the portfolio are skipped (per-order fees).
    "parking": {
        "symbol": "SHV",
        "min_trade_pct": 0.01,
    },
```

In `agent_cross_momentum.py`, below `logger = logging.getLogger(__name__)` add:

```python

# A position within ±20% of its target value is left alone (no top-up, no trim) to avoid overtrading.
_REBALANCE_BAND = 0.20
```

In `rebalance()`:

(a) Replace the docstring's first line with `"""Compare holdings to target, apply hysteresis, trim, submit orders, and park the rest in SHV.`.

(b) Replace everything from `        target_symbols = {entry["symbol"] for entry in target}` down to (not including) `        # Phase 2: Buy` with:

```python
        target_symbols = {entry["symbol"] for entry in target}
        target_by_symbol = {entry["symbol"]: entry for entry in target}
        parking_symbol = self.parameters["parking"]["symbol"]

        sell_threshold = self.parameters["sell_rank_threshold"]
        portfolio_value = float(self.portfolio_value or 1.0)
        min_trade_value = portfolio_value * self.parameters["parking"]["min_trade_pct"]

        current_positions = self.get_positions()

        # Phase 1: Sell (exits, trims, excess parking)
        estimated_sell_proceeds = 0.0
        hysteresis_value = 0.0
        parking_position = None
        for pos in current_positions:
            symbol = pos.asset.symbol
            if symbol == parking_symbol:
                # The parking sleeve is never ranked: it must not be exited as "not ranked" below
                parking_position = pos
                continue
            if symbol in target_symbols:
                # Trim a target position that sits above its band, so an exposure cut lowers the held book
                # too, not just new buys; the proceeds are parked or fund this week's buys.
                entry = target_by_symbol[symbol]
                target_value = portfolio_value * entry["target_weight"]
                current_value = float(pos.quantity) * entry["price"]
                trim_value = current_value - target_value
                if current_value > target_value * (1 + _REBALANCE_BAND) and trim_value >= min_trade_value:
                    trim_qty = fractional_qty(trim_value / entry["price"])
                    if trim_qty > 0:
                        self.log_info(f"Trimming {trim_qty} {symbol} @ ${entry['price']:.2f} (value ${current_value:,.0f} > target ${target_value:,.0f})")
                        try:
                            self.submit_order(self.create_order(symbol, trim_qty, "sell", time_in_force="day"))
                            estimated_sell_proceeds += trim_qty * entry["price"]
                        except Exception as e:
                            self.log_error(f"Failed to submit trim order for {symbol}: {e}")
                continue

            rank = all_ranks.get(symbol)
            if rank is None or rank > sell_threshold:
                if rank is None:
                    # The position did not pass the scoring pipeline and was kicked out by the filters (volatility, price or not able to compute indicators)
                    self.log_warning(f"Selling {symbol} (not ranked — failed filters or price data unavailable)")
                else:
                    # The position is ranked but outside the sell threshold (hysteresis band)
                    self.log_warning(f"Selling {symbol} (rank {rank} > {sell_threshold}) - outside hysteresis band")
                try:
                    sell_order = self.create_order(symbol, pos.quantity, "sell", time_in_force="day")
                    self.submit_order(sell_order)
                    last_price = self.get_last_price(symbol) or 0.0
                    estimated_sell_proceeds += float(pos.quantity) * float(last_price)
                except Exception as e:
                    self.log_error(f"Failed to submit sell order for {symbol}: {e}")
            # Rank <= sell_threshold means keep the position inside the histeresis band (do not sell)
            else:
                self.log_info(f"Keeping {symbol} (rank {rank} ≤ {sell_threshold}, within hysteresis band)")
                hysteresis_value += float(pos.quantity) * float(self.get_last_price(symbol) or 0.0)

        # Parking target: everything not meant for stocks, except the cash reserve, goes to the sleeve
        stock_target_value = portfolio_value * sum(entry["target_weight"] for entry in target)
        parking_target = max(0.0, portfolio_value * (1 - self.parameters["cash_buffer_pct"]) - stock_target_value - hysteresis_value)
        parking_price = float(self.get_last_price(parking_symbol) or 0.0)
        parking_value = float(parking_position.quantity) * parking_price if parking_position else 0.0
        if parking_price <= 0:
            self.log_warning(f"Parking: no price for {parking_symbol} — no parking orders this week")
        else:
            self.log_info(f"Parking: {parking_symbol} target ${parking_target:,.0f} (current ${parking_value:,.0f})")
            excess = parking_value - parking_target
            if parking_value > parking_target * (1 + _REBALANCE_BAND) and excess >= min_trade_value:
                # A zero target sells the exact holding, so float flooring leaves no dust behind
                sell_qty = float(parking_position.quantity) if parking_target == 0 else fractional_qty(excess / parking_price)
                if sell_qty > 0:
                    self.log_info(f"Selling {sell_qty} {parking_symbol} @ ${parking_price:.2f} (parking above target)")
                    try:
                        self.submit_order(self.create_order(parking_symbol, sell_qty, "sell", time_in_force="day"))
                        estimated_sell_proceeds += sell_qty * parking_price
                    except Exception as e:
                        self.log_error(f"Failed to submit parking sell order for {parking_symbol}: {e}")

```

(c) In Phase 2, replace `if current_value > 0 and abs(diff_value) / target_value < 0.20:` with `if current_value > 0 and abs(diff_value) / target_value < _REBALANCE_BAND:`.

(d) After the Phase 2 `for entry in target:` loop (end of the method), append:

```python

        # Phase 3: Park what the stock buys left, up to the parking target
        if parking_price > 0 and parking_value < parking_target * (1 - _REBALANCE_BAND):
            buy_value = min(parking_target - parking_value, available_cash)
            if buy_value >= min_trade_value:
                quantity = fractional_qty(buy_value / parking_price)
                if quantity > 0:
                    self.log_info(f"Buying {quantity} {parking_symbol} @ ${parking_price:.2f} (parking)")
                    try:
                        self.submit_order(self.create_order(parking_symbol, quantity, "buy", time_in_force="day"))
                    except Exception as e:
                        self.log_warning(f"Failed to submit parking buy order for {parking_symbol}: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/strategies -q && uv run ruff check src/trading_agent_framework/strategies/cross_momentum tests/strategies`
Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/parameters.py src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/strategies/test_cross_momentum_rebalance.py
git commit -m "Task 1: rebalance trims to the scaled target and parks the rest in SHV

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Preload SHV in backtests; docstrings

**Files:**
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py` (module docstring, class docstring, `_backtest_preload_assets` new, `run_backtesting`)
- Test: `tests/strategies/test_cross_momentum_rebalance.py` (append)

**Interfaces:**
- Consumes: `CONFIG["parking"]["symbol"]` (Task 1).
- Produces: `CrossMomentumStrategy._backtest_preload_assets(self) -> list[Asset]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/strategies/test_cross_momentum_rebalance.py` (add `from trading_agent_framework.entities.asset import Asset` to the imports at the top):

```python
def test_backtests_preload_the_parking_symbol_once():
    fake = SimpleNamespace(parameters={"parking": {"symbol": "SHV"}}, vars=SimpleNamespace(universe=["AAA", "SHV", "BBB"]))

    assets = CrossMomentumStrategy._backtest_preload_assets(fake)

    assert [a.symbol for a in assets] == ["AAA", "SHV", "BBB"]
    fake.vars.universe = ["AAA"]
    assert CrossMomentumStrategy._backtest_preload_assets(fake) == [Asset(symbol="AAA"), Asset(symbol="SHV")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/strategies/test_cross_momentum_rebalance.py -q`
Expected: FAIL — `AttributeError: type object 'CrossMomentumStrategy' has no attribute '_backtest_preload_assets'`.

- [ ] **Step 3: Implement**

Add above `run_backtesting`:

```python
    def _backtest_preload_assets(self) -> list[Asset]:
        """The universe plus the parking symbol, each once: what a backtest loads up front."""
        symbols = list(dict.fromkeys([*self.vars.universe, self.parameters["parking"]["symbol"]]))
        return [Asset(symbol=symbol) for symbol in symbols]
```

In `run_backtesting`, replace
`preload_assets=[Asset(symbol=ticker) for ticker in self.vars.universe],  # preload the ticker universe in memory`
with
`preload_assets=self._backtest_preload_assets(),  # preload the ticker universe and the parking ETF in memory`.

Module docstring: replace the line `  8. Hysteresis: sell below rank 35, buy top 20` with:

```
  8. Hysteresis: sell below rank 35, buy top 20; trim target positions above the ±20% band
  9. Park the de-risked capital (everything but the cash reserve) in SHV instead of idle cash
```

Class docstring: replace `    determines the final exposure multiplier.` with:

```
    determines the final exposure multiplier. Held positions are trimmed to
    the scaled target, and the capital taken out of stocks is parked in SHV.
```

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py tests/strategies/test_cross_momentum_rebalance.py
git commit -m "Task 2: preload SHV in backtests; document the parking sleeve

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the plan

The user runs `uv run agent cross_momentum backtesting` and compares `metrics.json` against
`logs/cross_momentum/backtesting/2026-09-28_120209_backtesting` (CAGR 0.32, Sharpe 1.10, Sortino 1.54, beta 1.07,
correlation 0.65, max drawdown −0.35), plus the trade count and fees in `settings.json`.

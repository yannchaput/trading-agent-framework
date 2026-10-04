# vwap_pullback_continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A strictly intraday VWAP pullback continuation strategy, `"vwap_pullback_continuation"`, in which a per-tick LangGraph graph runs a Python classifier and then, only when due, an exit agent and an entry agent; code enforces every risk rule.

**Architecture:** Pure modules (`parameters`, `features`, `screening`, `setups`, `risk`, `trades`, `news`, `prompts`, `session`) hold the logic; three I/O classes wire it: `Scanner` (bars and news in, `SessionState` out), `Desk` (every order the strategy places) and `VwapPullbackStrategy` (hooks, agents, graph). `graph.py` builds the per-tick `StateGraph` from injected node callables. The backtest broker learns to simulate trailing stops.

**Tech Stack:** Python 3.14, uv, pandas, LangChain `create_agent` (through the existing `AgentManager`), LangGraph `StateGraph`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-vwap-pullback-continuation-design.md`

## Global Constraints

- Python 3.14, `uv` only. Run tests with `uv run pytest`, lint with `uv run ruff check` (line length 200).
- Money is `Decimal` in `risk.py`, `trades.py`, `desk.py`. `float` only for bar maths in `features.py`, `screening.py`, `setups.py` (the `Bars.df` float boundary). Prices crossing into orders go through `risk.to_price` (cents).
- No `time.sleep` / `datetime.now` in strategy code: `strategy.clock.now()`, `strategy.get_datetime()`, `strategy.sleep()`.
- The test suite never touches the network. Hand-written fakes live in `tests/fakes.py`; no `MagicMock`. Tests import them as `from tests.fakes import ...`.
- `strategies/vwap_pullback/tools.py` has no `from __future__ import annotations` (LangChain reads the real annotations) and every tool docstring is one line.
- `langgraph` is imported only inside `graph.build_tick_graph` (a strategy that never builds the graph must not load it).
- Never raise a raw SDK exception out of a tool: tools return `{"error": ...}`; hooks log.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Repo cadence: commit each task as `Task N: ...`; a review-fix commit may follow before the next task.

## Deviations from the spec (decided while planning, all small)

1. **Three more modules**: `session.py` (the `SessionState`/`CandidateInfo` data), `scanner.py` (stage 1/2 I/O) and `desk.py` (all order flow). The spec put this logic in the strategy class and `tools.py`; splitting it keeps each file testable on its own.
2. **Large bearish candle = red body > `bearish_body_atr` (0.25) × daily ATR.** The spec's "0.5 × daily ATR ÷ 6" is 0.083 ATR, which an ordinary 5-minute pullback bar exceeds; it would break almost every setup.
3. **Broken checks start at `IMPULSE`** (a `WATCH` symbol has no impulse to break).
4. **End-of-day flatten waits up to `flatten_wait_seconds` (60 s) for its sells (live) and logs an error for any still open.** The spec said "the next tick retries", but the executor runs no tick after `before_market_closes`.
5. **Only this strategy's trades are flattened, and the restart rule closes only positions that this strategy's own open orders point at.** An account may hold positions this strategy never opened.
6. **Start-of-tick `reconcile`**: an entry rejected or errored at the broker (no hook reaches the strategy for `ERROR`) is dropped, and a protective stop that errored or was cancelled from outside is placed again.
7. **`live_bar_delay_seconds` (20 s)**: live, each tick waits for the last minute bar to be published before scanning; a 5-minute bucket missing its last minute is not treated as complete until a minute after it closes.
8. **`BacktestBroker.preload_bars(assets, start, end, timestep)`**: stage 1 batch-loads the universe's daily bars and the survivors' minute bars instead of one lazy fetch per symbol.
9. `fills.evaluate_fill` still raises for `TRAIL`; `BacktestBroker` routes `TRAIL` to the new `fills.evaluate_trailing_stop`.
10. `langgraph` becomes an explicit dependency (it is already installed through `langchain`).

## Review Focus

1. **Live minute-bar lag**: at a 10:00:00 tick the 09:59 minute bar is often not published yet; the 09:55–10:00 bucket must not be read as complete without it (Task 2 test `test_intraday_contexts_waits_for_a_bucket_whose_last_minute_is_not_published_yet`).
2. **Entry rejected at the broker** (backtest: not enough cash at fill; live: broker rejection): only an `ERROR` event, no strategy hook; the trade must not hold a slot forever (Task 8 test `test_reconcile_drops_an_entry_the_broker_rejected`).
3. **Protective stop cancelled or rejected outside the strategy** while the position is open: it must be placed again (Task 8 tests `test_reconcile_replaces_a_stop_that_errored`, `test_unexpected_stop_cancel_places_it_again`).
4. **Shared account**: a position this strategy did not open must never be flattened or closed by it (Task 8 test `test_close_unknown_positions_leaves_foreign_positions_alone`).
5. **Symbol with no bars today / no RVOL baseline** (halted, IPO, data gap): no crash, and it cannot pass the stage-2 floor (Task 2 `test_intraday_contexts_empty_frame`, Task 3 `test_rank_stage2_excludes_missing_rvol`).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/trading_agent_framework/backtesting/fills.py` | modify | pure `evaluate_trailing_stop` |
| `src/trading_agent_framework/backtesting/broker.py` | modify | `TRAIL` pending orders, `preload_bars` |
| `src/trading_agent_framework/core/strategy.py` | modify | `create_order(..., trail_price, trail_percent)` |
| `src/trading_agent_framework/strategies/vwap_pullback/__init__.py` | create | exports `VwapPullbackStrategy` |
| `.../vwap_pullback/parameters.py` | create | `VwapPullbackParameters` |
| `.../vwap_pullback/features.py` | create | session slicing, 5-min contexts, VWAP, RVOL, ATR, EMA, beta, z-scores |
| `.../vwap_pullback/screening.py` | create | stage 1 / stage 2 selection |
| `.../vwap_pullback/setups.py` | create | setup state machine |
| `.../vwap_pullback/risk.py` | create | stop, sizing, windows, breaker |
| `.../vwap_pullback/trades.py` | create | `Trade`, `TradeBook`, exit-review flags |
| `.../vwap_pullback/news.py` | create | lean headlines |
| `.../vwap_pullback/prompts.py` | create | the two system prompts |
| `.../vwap_pullback/session.py` | create | `SessionState`, `CandidateInfo` |
| `.../vwap_pullback/scanner.py` | create | `Scanner` (I/O) |
| `.../vwap_pullback/desk.py` | create | `Desk` (order flow, I/O) |
| `.../vwap_pullback/tools.py` | create | agent tools and row views |
| `.../vwap_pullback/graph.py` | create | `TickState`, routing, `build_tick_graph` |
| `.../vwap_pullback/agent_vwap_pullback.py` | create | `VwapPullbackStrategy` |
| `src/trading_agent_framework/main.py` | modify | register the strategy |
| `tests/fakes.py` | modify | `FrameDataSource`, `minute_ohlc`, `FakeNewsProvider`, `FakeBroker` timestep frames and news |
| `tests/backtesting/test_fills_trailing.py`, `tests/backtesting/test_broker_trailing.py`, `tests/core/test_create_order_trailing.py` | create | Task 1 |
| `tests/strategies/vwap_pullback/test_vwap_*.py` | create | Tasks 2–12 |
| `CLAUDE.md`, `pyproject.toml`, `uv.lock` | modify | docs, dependency |

---

### Task 1: Trailing stops in backtests, `create_order` trail args, `preload_bars`

**Files:**
- Modify: `src/trading_agent_framework/backtesting/fills.py`
- Modify: `src/trading_agent_framework/backtesting/broker.py`
- Modify: `src/trading_agent_framework/core/strategy.py:385-417`
- Modify: `tests/fakes.py` (append helpers)
- Test: `tests/backtesting/test_fills_trailing.py`, `tests/backtesting/test_broker_trailing.py`, `tests/core/test_create_order_trailing.py`

**Interfaces:**
- Produces: `fills.evaluate_trailing_stop(*, side: OrderSide, bar: Bar, reference: Decimal, trail_price: Decimal | None = None, trail_percent: Decimal | None = None) -> tuple[FillResult | None, Decimal]`
- Produces: `BacktestBroker.preload_bars(assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None`
- Produces: `Strategy.create_order(asset, quantity, side, *, limit_price=None, stop_price=None, trail_price=None, trail_percent=None, time_in_force=TimeInForce.DAY) -> Order` (`OrderType.TRAIL` when a trail is given)
- Produces (tests/fakes.py): `FrameDataSource(frames: dict[tuple[str, str], pd.DataFrame], sessions=())` with `.load_calls`; `minute_ohlc(first: datetime, rows: Sequence[tuple[float, float, float, float, float]]) -> pd.DataFrame`

- [ ] **Step 1: Add the test helpers to `tests/fakes.py`**

Add to the imports of `tests/fakes.py`:

```python
from trading_agent_framework.backtesting.data.base import BacktestDataSource
```

Append at the end of `tests/fakes.py`:

```python
# --- backtest data ------------------------------------------------------------------


def minute_ohlc(first: datetime, rows: Sequence[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    """Consecutive one-minute `(open, high, low, close, volume)` rows from `first` (tz-aware), shaped like `Bars.df`."""
    index = pd.date_range(first, periods=len(rows), freq="1min", name="timestamp")
    return pd.DataFrame(list(rows), columns=["open", "high", "low", "close", "volume"], index=index, dtype=float)


class FrameDataSource(BacktestDataSource):
    """In-memory `BacktestDataSource`: frames indexed by bar CLOSE time, keyed by `(symbol, timestep)`."""

    name: ClassVar[str] = "frames"

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame], sessions: Sequence[MarketSession] = ()) -> None:
        self.frames = frames
        self._sessions = list(sessions)
        self.load_calls: list[tuple[tuple[str, ...], datetime, datetime, str]] = []

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        self.load_calls.append((tuple(asset.symbol for asset in assets), start, end, timestep))

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        frame = self.frames.get((asset.symbol, timestep))
        if frame is None:
            return None
        visible = frame[frame.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return [s for s in self._sessions if start.date() <= s.open.date() <= end.date()]
```

- [ ] **Step 2: Write the failing fill tests**

Create `tests/backtesting/test_fills_trailing.py`:

```python
from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading_agent_framework.backtesting import fills
from trading_agent_framework.entities.enums import OrderSide


def _bar(o: str, h: str, low: str, c: str) -> fills.Bar:
    return fills.Bar(open=D(o), high=D(h), low=D(low), close=D(c))


def test_sell_trail_fills_at_the_level_when_touched() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("101", "101.5", "98.9", "99.5"), reference=D("101"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("99"))
    assert reference == D("101")


def test_sell_trail_gapping_through_fills_at_the_open() -> None:
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("97", "97.5", "96", "96.5"), reference=D("101"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("97"))


def test_sell_trail_ratchets_the_reference_on_a_bar_that_does_not_trigger() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("101", "104", "100", "103.5"), reference=D("101"), trail_price=D("2"))
    assert result is None
    assert reference == D("104")


def test_sell_trail_tests_the_previous_level_before_this_bars_high_can_raise_it() -> None:
    # Level 98 from reference 100. The bar's 105 high would lift the level to 103 if applied first;
    # pessimistically it is not, so a 97.9 low fills at 98.
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("100", "105", "97.9", "104"), reference=D("100"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("98"))


def test_sell_trail_percent() -> None:
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("100", "100", "94.9", "95"), reference=D("100"), trail_percent=D("5"))
    assert result == fills.FillResult(price=D("95"))


def test_buy_trail_mirrors_with_a_low_water_mark() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.BUY, bar=_bar("99", "99.5", "96", "96.5"), reference=D("99"), trail_price=D("1"))
    assert result is None
    assert reference == D("96")
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.BUY, bar=_bar("100", "100.5", "97", "98"), reference=D("99"), trail_price=D("1"))
    assert result == fills.FillResult(price=D("100"))


def test_trailing_stop_needs_exactly_one_trail_field() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("1", "1", "1", "1"), reference=D("1"))
```

- [ ] **Step 3: Run it to see it fail**

Run: `uv run pytest tests/backtesting/test_fills_trailing.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'evaluate_trailing_stop'`

- [ ] **Step 4: Implement `evaluate_trailing_stop`**

In `src/trading_agent_framework/backtesting/fills.py`, replace the docstring line

```
- TRAIL: not supported (needs a trailing reference price tracked across bars,
  which is out of scope -- design spec, section 1.3); raises ValueError.
```

with

```
- TRAIL: not handled by `evaluate_fill` (it raises ValueError): a trailing stop needs a
  reference price carried from bar to bar, so `BacktestBroker` calls
  `evaluate_trailing_stop` with the reference it keeps per pending order.
```

Replace the last line of `evaluate_fill`:

```python
    raise ValueError(f"evaluate_fill does not handle order_type={order_type}; use evaluate_trailing_stop for TRAIL")
```

Add after `_stop_limit_fill`:

```python
def evaluate_trailing_stop(
    *,
    side: OrderSide,
    bar: Bar,
    reference: Decimal,
    trail_price: Decimal | None = None,
    trail_percent: Decimal | None = None,
) -> tuple[FillResult | None, Decimal]:
    """One bar of a trailing stop: `(fill or None, the reference to carry to the next bar)`.

    `reference` is the high-water mark (sell) or low-water mark (buy) of the bars before this one;
    `trail_percent` is in percent (5 = 5%), as Alpaca and IBKR take it. Ties go against the trader,
    like every other rule here: the level built from the previous reference is tested first, and only
    a bar that does not trigger may move the reference with its own high (sell) or low (buy) -- the
    bar's high might have come after its low.
    """
    if (trail_price is None) == (trail_percent is None):
        raise ValueError("a trailing stop needs exactly one of trail_price or trail_percent")
    if side is OrderSide.SELL:
        level = reference - trail_price if trail_price is not None else reference * (1 - trail_percent / 100)  # ty: ignore[unsupported-operator]
        if bar.low <= level:
            return FillResult(price=min(bar.open, level)), reference
        return None, max(reference, bar.high)
    level = reference + trail_price if trail_price is not None else reference * (1 + trail_percent / 100)  # ty: ignore[unsupported-operator]
    if bar.high >= level:
        return FillResult(price=max(bar.open, level)), reference
    return None, min(reference, bar.low)
```

- [ ] **Step 5: Run the fill tests**

Run: `uv run pytest tests/backtesting/test_fills_trailing.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Write the failing broker and `create_order` tests**

Create `tests/backtesting/test_broker_trailing.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from tests.fakes import FakeClock, FrameDataSource, et, minute_ohlc
from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

ROWS = [
    (100, 100, 100, 100, 1000),  # closes 09:31
    (100, 101, 99.5, 100.5, 1000),  # 09:32: the buy fills at this open (100)
    (100.5, 103, 100.4, 102.8, 1000),  # 09:33: level 98.5, reference -> 103
    (102.8, 104, 103, 103.5, 1000),  # 09:34: level 101, reference -> 104
    (103.5, 103.6, 101.9, 102, 1000),  # 09:35: level 102 touched -> fills at 102
    (102, 102.5, 101.5, 102, 1000),  # 09:36
]
AAPL = Asset("AAPL")


def _broker() -> tuple[BacktestBroker, FakeClock, FrameDataSource]:
    source = FrameDataSource({("AAPL", "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)})
    clock = FakeClock(et(2026, 9, 1, 9, 31))
    broker = BacktestBroker("s", data_source=source, clock=clock, budget=Decimal("100000"), timestep="minute")
    return broker, clock, source


def _advance(broker: BacktestBroker, clock: FakeClock, seconds: float) -> None:
    before = clock.now()
    clock.advance(seconds)
    broker.on_advance(before, clock.now())


def _buy_ten(broker: BacktestBroker, clock: FakeClock) -> None:
    broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.BUY, quantity=Decimal(10)))
    _advance(broker, clock, 60)  # 09:32


def _trail(quantity: int = 10, **trail: Decimal) -> Order:
    return Order(strategy_name="s", asset=AAPL, side=OrderSide.SELL, order_type=OrderType.TRAIL, quantity=Decimal(quantity), **trail)


def test_trailing_stop_ratchets_across_a_multi_bar_jump_and_fills_at_its_level() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    trail = broker.submit_order(_trail(trail_price=Decimal(2)))  # reference seeded at the 09:32 close, 100.5
    _advance(broker, clock, 180)  # 09:35 in one jump: three bars walked
    assert trail.is_filled()
    assert broker.pull_positions() == []
    assert broker.get_account().cash == Decimal("100020")  # bought 10 at 100, sold 10 at 102


def test_a_pending_trailing_sell_counts_against_the_position() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    broker.submit_order(_trail(trail_price=Decimal(2)))
    with pytest.raises(OrderValidationError, match="insufficient position"):
        broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.SELL, quantity=Decimal(1)))


def test_a_trailing_stop_without_a_trail_is_refused_at_submission() -> None:
    broker, clock, _ = _broker()
    _buy_ten(broker, clock)
    with pytest.raises(OrderValidationError, match="exactly one"):
        broker.submit_order(_trail())


def test_preload_bars_forwards_to_the_data_source() -> None:
    broker, _, source = _broker()
    broker.preload_bars([AAPL], et(2026, 5, 1), et(2026, 9, 30), "day")
    assert source.load_calls == [(("AAPL",), et(2026, 5, 1), et(2026, 9, 30), "day")]
```

Create `tests/core/test_create_order_trailing.py`:

```python
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from tests.fakes import FakeBroker, FakeClock, et
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.enums import OrderType
from trading_agent_framework.utils.errors import OrderValidationError


def _strategy(tmp_path: Path) -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 1, 10))), project_root=tmp_path)


def test_create_order_with_a_trail_price_builds_a_trailing_stop(tmp_path: Path) -> None:
    order = _strategy(tmp_path).create_order("AAPL", 10, "sell", trail_price=1.5)
    assert order.order_type is OrderType.TRAIL
    assert order.trail_price == Decimal("1.5")
    assert order.stop_price is None and order.limit_price is None


def test_create_order_with_a_trail_percent(tmp_path: Path) -> None:
    order = _strategy(tmp_path).create_order("AAPL", 10, "sell", trail_percent=2)
    assert order.order_type is OrderType.TRAIL
    assert order.trail_percent == Decimal("2")


def test_create_order_refuses_a_trail_with_a_stop_price(tmp_path: Path) -> None:
    with pytest.raises(OrderValidationError, match="trailing stop"):
        _strategy(tmp_path).create_order("AAPL", 10, "sell", stop_price=99, trail_price=1)
```

- [ ] **Step 7: Run them to see them fail**

Run: `uv run pytest tests/backtesting/test_broker_trailing.py tests/core/test_create_order_trailing.py -v`
Expected: FAIL (`TRAIL` rejected with "evaluate_fill does not handle", no `preload_bars`, `create_order()` got an unexpected keyword argument `trail_price`)

- [ ] **Step 8: Implement trailing stops in `BacktestBroker`**

In `src/trading_agent_framework/backtesting/broker.py`:

Change the enums import to:

```python
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
```

Add a field to `_PendingOrder` (after `needs_skip`):

```python
    # A TRAIL order's high-water mark (sell) or low-water mark (buy), carried from bar to bar by
    # `fills.evaluate_trailing_stop`. Seeded with the latest close at submission, like a real broker
    # seeds it with the price at acceptance; the skipped forming bar does not move it (pessimistic).
    trail_reference: Decimal | None = None
```

In `_submit_order`, right after the `notional` check and before `projected_rejection = ...`, add:

```python
        if order.order_type is OrderType.TRAIL and (order.trail_price is None) == (order.trail_percent is None):
            raise OrderValidationError("a trailing stop order needs exactly one of trail_price or trail_percent")
```

and replace the `self._pending[order.identifier] = _PendingOrder(...)` statement with:

```python
trail_reference = found[0].close if order.order_type is OrderType.TRAIL and found is not None else None
self._pending[order.identifier] = _PendingOrder(order=order, asset=order.asset, last_evaluated=now, needs_skip=needs_skip, trail_reference=trail_reference)
```

Add after `modify_order`:

```python
    def preload_bars(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        """Batch-fetch `assets` at `timestep` for `[start, end]` before they are read.

        For a strategy that picks its symbols per session (the runner only preloads what it knows up
        front): one batched `load()` instead of one lazy fetch per symbol. `start`/`end` must be the
        data source's own window, or a source that caches per asset would keep a shorter frame.
        """
        self._data_source.load(assets, start, end, timestep)
```

In `_evaluate`, replace the `try: result = fills.evaluate_fill(...)` block with:

```python
        try:
            if order.order_type is OrderType.TRAIL:
                reference = pending.trail_reference if pending.trail_reference is not None else bar.open
                result, pending.trail_reference = fills.evaluate_trailing_stop(
                    side=order.side, bar=bar, reference=reference,
                    trail_price=order.trail_price, trail_percent=order.trail_percent,
                )
            else:
                result = fills.evaluate_fill(
                    order_type=order.order_type, side=order.side, bar=bar,
                    limit_price=order.limit_price, stop_price=order.stop_price,
                    stop_limit_price=order.stop_limit_price,
                )
        except ValueError as exc:
```

(the `except ValueError` body stays as it is).

- [ ] **Step 9: Implement the `create_order` trail arguments**

In `src/trading_agent_framework/core/strategy.py`, change the errors import to:

```python
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, LLMStatsError, OrderValidationError
```

Replace `create_order` with:

```python
    def create_order(
        self,
        asset: Asset | str,
        quantity: Number,
        side: OrderSide | str,
        *,
        limit_price: Number | None = None,
        stop_price: Number | None = None,
        trail_price: Number | None = None,
        trail_percent: Number | None = None,
        time_in_force: TimeInForce | str = TimeInForce.DAY,
    ) -> Order:
        """Build (not submit) an order; the type follows from the prices given (a trail makes a trailing stop)."""
        limit = _to_optional_decimal(limit_price)
        stop = _to_optional_decimal(stop_price)
        trail = _to_optional_decimal(trail_price)
        trail_pct = _to_optional_decimal(trail_percent)
        if trail is not None or trail_pct is not None:
            if limit is not None or stop is not None:
                raise OrderValidationError("a trailing stop takes a trail_price or trail_percent, not a limit_price or stop_price")
            order_type = OrderType.TRAIL
        elif limit is not None and stop is not None:
            order_type = OrderType.STOP_LIMIT
        elif limit is not None:
            order_type = OrderType.LIMIT
        elif stop is not None:
            order_type = OrderType.STOP
        else:
            order_type = OrderType.MARKET
        is_stop_limit = order_type is OrderType.STOP_LIMIT
        return Order(
            strategy_name=self.name,
            asset=_to_asset(asset),
            side=OrderSide(side),
            order_type=order_type,
            quantity=_to_decimal(quantity),
            time_in_force=TimeInForce(time_in_force),
            limit_price=None if is_stop_limit else limit,
            stop_price=stop,
            stop_limit_price=limit if is_stop_limit else None,
            trail_price=trail,
            trail_percent=trail_pct,
        )
```

- [ ] **Step 10: Run the new tests and the whole suite**

Run: `uv run pytest tests/backtesting/test_fills_trailing.py tests/backtesting/test_broker_trailing.py tests/core/test_create_order_trailing.py -v`
Expected: PASS (14 tests)

Run: `uv run pytest -q && uv run ruff check`
Expected: all pass, no lint errors. (If an existing test asserted the old `ValueError` text for `TRAIL`, update its `match=` to `"evaluate_fill does not handle"`.)

- [ ] **Step 11: Commit**

```bash
git add src/trading_agent_framework/backtesting/fills.py src/trading_agent_framework/backtesting/broker.py src/trading_agent_framework/core/strategy.py tests/fakes.py tests/backtesting/test_fills_trailing.py tests/backtesting/test_broker_trailing.py tests/core/test_create_order_trailing.py
git commit -m "Task 1: simulate trailing stops in backtests, trail args on create_order, BacktestBroker.preload_bars

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Parameters and pure intraday features

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/__init__.py` (empty for now)
- Create: `src/trading_agent_framework/strategies/vwap_pullback/parameters.py`
- Create: `src/trading_agent_framework/strategies/vwap_pullback/features.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_features.py`

**Interfaces:**
- Consumes: `tests.fakes.minute_ohlc`, `tests.fakes.make_bars_frame`, `tests.fakes.et`
- Produces: `VwapPullbackParameters` (frozen dataclass, fields below)
- Produces (features): `BarStamp = Literal["open", "close"]`; `BarContext(time, open, high, low, close, volume, vwap, rs, rvol, session_open, session_high)`; `Levels(close, vwap, ema, atr)`; `minute_starts`, `session_slice(df, session_open, session_close, bar_stamp)`, `minute_of_session`, `vwap_series(df)`, `cumulative_volume_by_minute(df, session_open, bar_stamp)`, `rvol_baseline(prior_sessions)`, `rvol_at(cumulative_volume, minute, baseline)`, `intraday_contexts(df, benchmark_df, *, session_open, now, bar_stamp, beta, baseline, minutes=5) -> list[BarContext]`, `daily_atr(df, length) -> float | None`, `bar_atr(contexts, length) -> float | None`, `ema_last(values, length) -> float | None`, `beta(stock_closes, bench_closes, lookback) -> float`, `zscores(values) -> dict[str, float]`, `latest_levels(contexts, *, ema_length, atr_length) -> Levels | None`

- [ ] **Step 1: Write the parameters module**

Create `src/trading_agent_framework/strategies/vwap_pullback/__init__.py` with a single docstring line:

```python
"""Intraday VWAP pullback continuation: two LangGraph-orchestrated agents over a Python setup scanner."""
```

Create `src/trading_agent_framework/strategies/vwap_pullback/parameters.py`:

```python
"""Every threshold, window and risk knob of the vwap_pullback strategy (spec §8), in one frozen dataclass.

Ratios are floats (they multiply bar maths); `risk.py` turns them into `Decimal` before touching money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time


@dataclass(frozen=True, slots=True)
class VwapPullbackParameters:
    # stage 1 (daily)
    stage1_lookback_sessions: int = 70
    min_price: float = 5.0
    atr_pct_band: tuple[float, float] = (0.015, 0.08)
    dollar_volume_percentile: float = 0.60
    stage1_size: int = 150
    beta_lookback_sessions: int = 60
    atr_length: int = 14
    # stage 2 (intraday)
    rvol_baseline_sessions: int = 10
    rvol_min: float = 1.5
    tracked_size: int = 30
    bar_minutes: int = 5
    ema_length: int = 9
    # setups
    impulse_move_atr: float = 0.8
    pullback_min_retrace: float = 0.25
    pullback_max_retrace: float = 0.618
    bearish_body_atr: float = 0.25
    selling_volume_ratio: float = 1.5
    # risk
    stop_buffer_atr: float = 0.1
    r_band_atr: tuple[float, float] = (0.15, 1.0)
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.25
    cash_buffer: float = 0.95
    max_positions: int = 4
    max_daily_loss_pct: float = 0.015
    chase_guard_r: float = 0.3
    entry_limit_atr: float = 0.05
    no_entry_before: time = time(9, 45)
    no_entry_after: time = time(15, 0)
    # agents
    exit_review_minutes: int = 15
    headlines_per_symbol: int = 3
    news_calls_per_run: int = 4
    none_catalyst_min_z: float = 2.0
    tp1_fraction_band: tuple[float, float] = (0.25, 0.5)
    trail_atr_band: tuple[float, float] = (0.5, 2.0)
    # order handling
    cancel_wait_seconds: float = 10.0
    flatten_wait_seconds: float = 60.0
    live_bar_delay_seconds: float = 20.0
```

- [ ] **Step 2: Write the failing feature tests**

Create `tests/strategies/vwap_pullback/test_vwap_features.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from tests.fakes import et, make_bars_frame, minute_ohlc
from trading_agent_framework.strategies.vwap_pullback.features import (
    bar_atr,
    beta,
    cumulative_volume_by_minute,
    daily_atr,
    ema_last,
    intraday_contexts,
    latest_levels,
    rvol_at,
    rvol_baseline,
    session_slice,
    vwap_series,
    zscores,
)

OPEN = et(2026, 9, 1, 9, 30)
CLOSE = et(2026, 9, 1, 16, 0)
FLAT = (1.0, 1.0, 1.0, 1.0, 1.0)


def _rising(count: int) -> list[tuple[float, float, float, float, float]]:
    return [(100 + 0.1 * i, 100 + 0.1 * i + 0.05, 100 + 0.1 * i - 0.05, 100 + 0.1 * (i + 1), 200.0) for i in range(count)]


def test_session_slice_drops_premarket_for_open_and_close_stamped_bars() -> None:
    live = minute_ohlc(et(2026, 9, 1, 9, 29), [FLAT] * 3)  # bars starting 09:29, 09:30, 09:31
    assert list(session_slice(live, OPEN, CLOSE, "open").index) == [et(2026, 9, 1, 9, 30), et(2026, 9, 1, 9, 31)]
    backtest = minute_ohlc(et(2026, 9, 1, 9, 30), [FLAT] * 3)  # bars closing 09:30 (premarket), 09:31, 09:32
    assert list(session_slice(backtest, OPEN, CLOSE, "close").index) == [et(2026, 9, 1, 9, 31), et(2026, 9, 1, 9, 32)]


def test_vwap_weights_the_typical_price_by_volume() -> None:
    df = minute_ohlc(OPEN, [(10, 12, 8, 10, 100), (20, 22, 18, 20, 300)])
    assert vwap_series(df).tolist() == pytest.approx([10.0, 17.5])


def test_cumulative_volume_carries_over_minutes_without_a_bar() -> None:
    df = pd.concat([minute_ohlc(OPEN, [(1, 1, 1, 1, 100)]), minute_ohlc(et(2026, 9, 1, 9, 33), [(1, 1, 1, 1, 50)])])
    assert cumulative_volume_by_minute(df, OPEN, "open").tolist() == [100, 100, 100, 150]


def test_rvol_baseline_averages_only_the_sessions_that_reached_the_minute() -> None:
    baseline = rvol_baseline([pd.Series([10.0, 20.0, 30.0]), pd.Series([30.0, 40.0])])
    assert baseline.tolist() == [20.0, 30.0, 30.0]
    assert rvol_at(60.0, 1, baseline) == 2.0
    assert rvol_at(60.0, 5, baseline) is None


def test_intraday_contexts_builds_completed_bars_with_rs_and_rvol() -> None:
    stock = minute_ohlc(OPEN, _rising(10))
    spy = minute_ohlc(OPEN, [(400, 400, 400, 400, 1000)] * 9 + [(400, 404, 400, 404, 1000)])
    baseline = pd.Series([100.0 * (m + 1) for m in range(390)])
    contexts = intraday_contexts(stock, spy, session_open=OPEN, now=et(2026, 9, 1, 9, 40), bar_stamp="open", beta=1.0, baseline=baseline)
    assert [c.time for c in contexts] == [et(2026, 9, 1, 9, 35), et(2026, 9, 1, 9, 40)]
    first, second = contexts
    assert first.close == pytest.approx(100.5)
    assert first.volume == 1000
    assert first.rvol == pytest.approx(2.0)  # 1000 traded vs 500 expected by minute 4
    assert first.rs == pytest.approx(0.005)
    assert second.rs == pytest.approx(0.0)  # +1% while SPY did +1%
    assert second.session_high == pytest.approx(100.95)
    assert second.session_open == 100.0


def test_intraday_contexts_waits_for_a_bucket_whose_last_minute_is_not_published_yet() -> None:
    stock = minute_ohlc(OPEN, _rising(9))  # the 09:39 bar has not landed yet
    spy = minute_ohlc(OPEN, [(400, 400, 400, 400, 1000)] * 9)
    baseline = pd.Series([100.0 * (m + 1) for m in range(390)])
    kwargs = {"session_open": OPEN, "bar_stamp": "open", "beta": 1.0, "baseline": baseline}
    assert len(intraday_contexts(stock, spy, now=et(2026, 9, 1, 9, 40, 10), **kwargs)) == 1
    assert len(intraday_contexts(stock, spy, now=et(2026, 9, 1, 9, 41), **kwargs)) == 2


def test_intraday_contexts_of_an_empty_frame_is_empty() -> None:
    empty = minute_ohlc(OPEN, [])
    assert intraday_contexts(empty, empty, session_open=OPEN, now=CLOSE, bar_stamp="open", beta=1.0, baseline=pd.Series(dtype=float)) == []


def test_daily_atr_uses_the_true_range() -> None:
    assert daily_atr(make_bars_frame([100.0] * 20), 14) == pytest.approx(2.0)
    assert daily_atr(make_bars_frame([100.0] * 10), 14) is None


def test_beta_of_a_doubled_return_series_is_two() -> None:
    moves = [0.01 if i % 2 else -0.01 for i in range(40)]
    spy, stock = [100.0], [100.0]
    for move in moves:
        spy.append(spy[-1] * (1 + move))
        stock.append(stock[-1] * (1 + 2 * move))
    index = pd.date_range(et(2026, 6, 1), periods=41, freq="1D")
    assert beta(pd.Series(stock, index=index), pd.Series(spy, index=index), 60) == pytest.approx(2.0)
    assert beta(pd.Series(stock[:5], index=index[:5]), pd.Series(spy[:5], index=index[:5]), 60) == 1.0  # too few returns


def test_zscores() -> None:
    assert zscores({"A": 1.0, "B": 1.0}) == {"A": 0.0, "B": 0.0}
    assert zscores({"A": 1.0, "B": 3.0}) == {"A": -1.0, "B": 1.0}
    assert zscores({"A": 5.0}) == {"A": 0.0}


def test_ema_bar_atr_and_latest_levels() -> None:
    assert ema_last([1.0, 1.0, 1.0], 9) == 1.0
    assert ema_last([], 9) is None
    contexts = intraday_contexts(minute_ohlc(OPEN, _rising(10)), minute_ohlc(OPEN, []), session_open=OPEN, now=CLOSE, bar_stamp="open", beta=1.0, baseline=pd.Series(dtype=float))
    assert bar_atr(contexts, 14) == pytest.approx(0.5)  # both 5-minute bars span 0.5
    levels = latest_levels(contexts, ema_length=9, atr_length=14)
    assert levels is not None and levels.close == pytest.approx(101.0)
    assert latest_levels([], ema_length=9, atr_length=14) is None
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_features.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.features`

- [ ] **Step 4: Implement `features.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/features.py`:

```python
"""Pure intraday features over minute bars (spec §2): session slicing, 5-minute contexts, VWAP, RVOL, RS, ATR, EMA.

float64 throughout: this is indicator maths on `Bars.df`, the codebase's float boundary. `bar_stamp` says how
the source stamps a minute bar: "open" (live Alpaca `Bars`) or "close" (every `BacktestDataSource`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

BarStamp = Literal["open", "close"]
_MINUTE = pd.Timedelta(minutes=1)


@dataclass(frozen=True, slots=True)
class BarContext:
    """One completed intraday bar (5-minute by default) and the session state as of its close."""

    time: datetime  # the bar's close
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float
    rs: float  # session return minus beta x benchmark session return, at this close
    rvol: float | None  # cumulative volume over the baseline at this minute; None without a baseline
    session_open: float  # the session's first regular-hours open
    session_high: float  # the session's high up to this bar


@dataclass(frozen=True, slots=True)
class Levels:
    """The latest completed bar's close and the levels the exit review watches."""

    close: float
    vwap: float
    ema: float | None
    atr: float | None


def minute_starts(index: pd.DatetimeIndex, bar_stamp: BarStamp) -> pd.DatetimeIndex:
    """When each minute bar started: live Alpaca bars are stamped at their open, backtest bars at their close."""
    return index if bar_stamp == "open" else index - _MINUTE


def session_slice(df: pd.DataFrame, session_open: datetime, session_close: datetime, bar_stamp: BarStamp) -> pd.DataFrame:
    """The regular-session rows of a minute frame: the bars that start in `[open, close)`."""
    starts = minute_starts(df.index, bar_stamp)
    return df[(starts >= session_open) & (starts < session_close)]


def minute_of_session(df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp) -> pd.Series:
    """Minutes since the open at which each row's bar started (0 = the opening minute)."""
    starts = minute_starts(df.index, bar_stamp)
    return pd.Series(((starts - session_open) // _MINUTE).astype(int), index=df.index)


def vwap_series(df: pd.DataFrame) -> pd.Series:
    """Session VWAP after each row: cumulative typical price x volume over cumulative volume (NaN before any volume)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].astype(float)
    cumulative_volume = volume.cumsum()
    return (typical * volume).cumsum() / cumulative_volume.where(cumulative_volume > 0)


def cumulative_volume_by_minute(df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp) -> pd.Series:
    """Cumulative volume at every minute-of-session up to the last bar; a minute with no bar carries the previous total."""
    if df.empty:
        return pd.Series(dtype=float)
    minutes = minute_of_session(df, session_open, bar_stamp)
    cumulative = pd.Series(df["volume"].astype(float).cumsum().to_numpy(), index=minutes.to_numpy())
    cumulative = cumulative[~cumulative.index.duplicated(keep="last")]
    return cumulative.reindex(range(int(cumulative.index.max()) + 1)).ffill().fillna(0.0)


def rvol_baseline(prior_sessions: Sequence[pd.Series]) -> pd.Series:
    """Mean cumulative volume at each minute-of-session over prior sessions; a minute only averages the sessions that reached it (early closes)."""
    if not prior_sessions:
        return pd.Series(dtype=float)
    return pd.concat(list(prior_sessions), axis=1).mean(axis=1, skipna=True).sort_index()


def rvol_at(cumulative_volume: float, minute: int, baseline: pd.Series) -> float | None:
    """Today's cumulative volume over the baseline's at the same minute; None when the baseline has nothing there."""
    expected = baseline.get(minute)
    if expected is None or not expected > 0:  # NaN > 0 is False
        return None
    return float(cumulative_volume) / float(expected)


def intraday_contexts(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    session_open: datetime,
    now: datetime,
    bar_stamp: BarStamp,
    beta: float,
    baseline: pd.Series,
    minutes: int = 5,
) -> list[BarContext]:
    """Completed `minutes`-minute bars of one session, oldest first, each with VWAP, RS and RVOL at its close.

    `df`/`benchmark_df` are already `session_slice`d. A bucket counts as complete once its close time has
    passed AND either its last minute is in the data or a full minute has gone by since it closed: live
    minute bars land a few seconds late, and a bucket read without its last minute would pass for a
    finished bar. A bucket with no trade at all is simply absent.
    """
    if df.empty:
        return []
    mos = minute_of_session(df, session_open, bar_stamp)
    buckets = mos // minutes
    vwaps = vwap_series(df)
    cumulative = df["volume"].astype(float).cumsum()
    open_price = float(df["open"].iloc[0])
    bench_closes, bench_open = _benchmark_closes(benchmark_df, session_open, bar_stamp, minutes)
    contexts: list[BarContext] = []
    session_high = -math.inf
    for bucket_value, rows in df.groupby(buckets.to_numpy(), sort=True):
        bucket = int(bucket_value)
        close_time = session_open + timedelta(minutes=(bucket + 1) * minutes)
        last_minute = (bucket + 1) * minutes - 1
        label = rows.index[-1]
        has_last_minute = int(mos.loc[label]) == last_minute
        if close_time > now or (not has_last_minute and now - close_time < timedelta(minutes=1)):
            break
        high = float(rows["high"].max())
        session_high = max(session_high, high)
        close = float(rows["close"].iloc[-1])
        bench_return = _benchmark_return(bench_closes, bench_open, bucket)
        contexts.append(
            BarContext(
                time=close_time,
                open=float(rows["open"].iloc[0]),
                high=high,
                low=float(rows["low"].min()),
                close=close,
                volume=float(rows["volume"].sum()),
                vwap=float(vwaps.loc[label]),
                rs=(close / open_price - 1) - beta * bench_return,
                rvol=rvol_at(float(cumulative.loc[label]), last_minute, baseline),
                session_open=open_price,
                session_high=session_high,
            )
        )
    return contexts


def _benchmark_closes(benchmark_df: pd.DataFrame, session_open: datetime, bar_stamp: BarStamp, minutes: int) -> tuple[pd.Series, float | None]:
    if benchmark_df.empty:
        return pd.Series(dtype=float), None
    buckets = minute_of_session(benchmark_df, session_open, bar_stamp) // minutes
    closes = benchmark_df["close"].astype(float).groupby(buckets.to_numpy()).last()
    return closes, float(benchmark_df["open"].iloc[0])


def _benchmark_return(closes: pd.Series, bench_open: float | None, bucket: int) -> float:
    if not bench_open:
        return 0.0
    upto = closes[closes.index <= bucket]
    return 0.0 if upto.empty else float(upto.iloc[-1]) / bench_open - 1


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous = close.shift(1)
    return pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)


def daily_atr(df: pd.DataFrame, length: int) -> float | None:
    """Mean true range of the last `length` daily bars; None with fewer than `length + 1` bars."""
    if len(df) < length + 1:
        return None
    tr = _true_range(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float))
    return float(tr.iloc[-length:].mean())


def bar_atr(contexts: Sequence[BarContext], length: int) -> float | None:
    """Mean true range of the last `length` intraday bars (fewer early in the session); None without bars."""
    if not contexts:
        return None
    frame = pd.DataFrame({"high": [c.high for c in contexts], "low": [c.low for c in contexts], "close": [c.close for c in contexts]})
    return float(_true_range(frame["high"], frame["low"], frame["close"]).iloc[-length:].mean())


def ema_last(values: Sequence[float], length: int) -> float | None:
    """The last value of an exponential moving average with span `length`; None without values."""
    if not values:
        return None
    return float(pd.Series(list(values), dtype=float).ewm(span=length, adjust=False).mean().iloc[-1])


def beta(stock_closes: pd.Series, bench_closes: pd.Series, lookback: int) -> float:
    """Beta of daily returns over the last `lookback` returns; 1.0 when there is too little data to say."""
    joined = pd.concat([stock_closes.astype(float), bench_closes.astype(float)], axis=1, join="inner").dropna()
    returns = joined.pct_change().dropna().iloc[-lookback:]
    if len(returns) < 10:
        return 1.0
    variance = returns.iloc[:, 1].var()
    if not variance > 0:
        return 1.0
    return float(returns.iloc[:, 0].cov(returns.iloc[:, 1]) / variance)


def zscores(values: Mapping[str, float]) -> dict[str, float]:
    """Cross-sectional z-scores (population std); all 0.0 when there is no spread to measure."""
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    series = pd.Series(dict(values), dtype=float)
    std = series.std(ddof=0)
    if not std > 0:
        return dict.fromkeys(values, 0.0)
    return {str(key): float(value) for key, value in ((series - series.mean()) / std).items()}


def latest_levels(contexts: Sequence[BarContext], *, ema_length: int, atr_length: int) -> Levels | None:
    """The latest bar's close and VWAP, the EMA of the closes and the bar ATR; None without bars."""
    if not contexts:
        return None
    last = contexts[-1]
    return Levels(close=last.close, vwap=last.vwap, ema=ema_last([c.close for c in contexts], ema_length), atr=bar_atr(contexts, atr_length))
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_features.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (11 tests), no lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/__init__.py src/trading_agent_framework/strategies/vwap_pullback/parameters.py src/trading_agent_framework/strategies/vwap_pullback/features.py tests/strategies/vwap_pullback/test_vwap_features.py
git commit -m "Task 2: vwap_pullback parameters and pure intraday features

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Stage 1 and stage 2 screening (pure)

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/screening.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_screening.py`

**Interfaces:**
- Consumes: `features.daily_atr`, `features.beta`, `features.zscores`, `features.BarContext`, `VwapPullbackParameters`
- Produces: `DailyProfile(symbol, last_close, daily_atr, atr_pct, dollar_volume, momentum, beta)`; `daily_profile(symbol, daily, bench_daily, params) -> DailyProfile | None`; `select_stage1(profiles, params) -> list[DailyProfile]`; `IntradaySnapshot(symbol, ret, rs, rvol, last_close, vwap)`; `snapshot_from(symbol, contexts) -> IntradaySnapshot | None`; `RankedCandidate(symbol, composite, z_rs, z_rvol)`; `rank_stage2(snapshots, params, sticky) -> list[RankedCandidate]`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_screening.py`:

```python
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from tests.fakes import et, make_bars_frame
from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.screening import (
    DailyProfile,
    IntradaySnapshot,
    daily_profile,
    rank_stage2,
    select_stage1,
    snapshot_from,
)

PARAMS = VwapPullbackParameters()


def _daily(closes: list[float]) -> pd.DataFrame:
    return make_bars_frame(closes, start=et(2026, 6, 1))


def test_daily_profile_measures_atr_momentum_and_dollar_volume() -> None:
    closes = [100.0] * 50 + [100.0 + i for i in range(21)]  # +20% over the last 20 sessions
    profile = daily_profile("AAA", _daily(closes), _daily([400.0] * 71), PARAMS)
    assert profile is not None
    assert profile.last_close == 120.0
    assert profile.momentum == pytest.approx(120 / 100 - 1)
    assert profile.daily_atr == pytest.approx(2.0, abs=0.1)
    assert profile.dollar_volume == pytest.approx(sum(closes[-20:]) / 20 * 1000)


def test_daily_profile_needs_enough_history() -> None:
    assert daily_profile("AAA", _daily([100.0] * 10), None, PARAMS) is None


def _profile(symbol: str, *, close: float = 100.0, atr_pct: float = 0.02, dollar_volume: float = 1e6, momentum: float = 0.0) -> DailyProfile:
    return DailyProfile(symbol=symbol, last_close=close, daily_atr=close * atr_pct, atr_pct=atr_pct, dollar_volume=dollar_volume, momentum=momentum, beta=1.0)


def test_select_stage1_filters_price_atr_band_and_volume_percentile_then_ranks() -> None:
    profiles = [
        _profile("CHEAP", close=3.0),
        _profile("WILD", atr_pct=0.10),
        _profile("CALM", atr_pct=0.01),
        _profile("THIN", dollar_volume=1.0),
        _profile("FAST", atr_pct=0.05, momentum=0.2),
        _profile("SLOW", atr_pct=0.02, momentum=0.0),
    ]
    chosen = select_stage1(profiles, PARAMS)
    assert [p.symbol for p in chosen] == ["FAST", "SLOW"]
    assert [p.symbol for p in select_stage1(profiles, dataclasses.replace(PARAMS, stage1_size=1))] == ["FAST"]


def _snapshot(symbol: str, *, ret: float = 0.02, rs: float = 0.01, rvol: float | None = 2.0, above_vwap: bool = True) -> IntradaySnapshot:
    return IntradaySnapshot(symbol=symbol, ret=ret, rs=rs, rvol=rvol, last_close=101.0, vwap=100.0 if above_vwap else 102.0)


def test_rank_stage2_applies_the_floor_and_keeps_the_top_n() -> None:
    snapshots = [
        _snapshot("LOWVOL", rvol=1.2),
        _snapshot("WEAK", rs=-0.01),
        _snapshot("UNDER", above_vwap=False),
        _snapshot("A", ret=0.05, rs=0.04, rvol=4.0),
        _snapshot("B", ret=0.02, rs=0.01, rvol=2.0),
        _snapshot("C", ret=0.01, rs=0.005, rvol=1.6),
    ]
    ranked = rank_stage2(snapshots, dataclasses.replace(PARAMS, tracked_size=2), sticky=set())
    assert [c.symbol for c in ranked] == ["A", "B"]
    assert ranked[0].z_rs > 0 and ranked[0].z_rvol > 0


def test_rank_stage2_keeps_sticky_symbols_even_when_they_fail_the_floor() -> None:
    ranked = rank_stage2([_snapshot("A"), _snapshot("HELD", rs=-0.01)], PARAMS, sticky={"HELD", "GONE"})
    assert [c.symbol for c in ranked] == ["A", "GONE", "HELD"]
    assert ranked[2].composite == 0.0


def test_rank_stage2_excludes_missing_rvol() -> None:
    assert rank_stage2([_snapshot("NEW", rvol=None)], PARAMS, sticky=set()) == []


def test_snapshot_from_uses_the_latest_context() -> None:
    context = BarContext(time=et(2026, 9, 1, 9, 35), open=100, high=101, low=99.5, close=101, volume=1000, vwap=100.5, rs=0.008, rvol=2.5, session_open=100, session_high=101)
    snapshot = snapshot_from("AAA", [context])
    assert snapshot == IntradaySnapshot(symbol="AAA", ret=pytest.approx(0.01), rs=0.008, rvol=2.5, last_close=101, vwap=100.5)
    assert snapshot_from("AAA", []) is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_screening.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.screening`

- [ ] **Step 3: Implement `screening.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/screening.py`:

```python
"""Pure candidate selection (spec §2): stage 1 on daily bars, stage 2 on intraday contexts."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import pandas as pd

from trading_agent_framework.strategies.vwap_pullback.features import BarContext, beta, daily_atr, zscores
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

_MOMENTUM_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class DailyProfile:
    symbol: str
    last_close: float
    daily_atr: float
    atr_pct: float
    dollar_volume: float  # 20-session average close x volume
    momentum: float  # 20-session return
    beta: float


@dataclass(frozen=True, slots=True)
class IntradaySnapshot:
    symbol: str
    ret: float
    rs: float
    rvol: float | None
    last_close: float
    vwap: float


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    symbol: str
    composite: float
    z_rs: float
    z_rvol: float


def daily_profile(symbol: str, daily: pd.DataFrame, bench_daily: pd.DataFrame | None, params: VwapPullbackParameters) -> DailyProfile | None:
    """One symbol's stage-1 measures from its daily bars; None with too little history or no usable price."""
    if len(daily) < max(_MOMENTUM_SESSIONS + 1, params.atr_length + 1):
        return None
    close = daily["close"].astype(float)
    last_close = float(close.iloc[-1])
    atr = daily_atr(daily, params.atr_length)
    if atr is None or not last_close > 0:
        return None
    dollar_volume = float((close * daily["volume"].astype(float)).tail(_MOMENTUM_SESSIONS).mean())
    momentum = last_close / float(close.iloc[-(_MOMENTUM_SESSIONS + 1)]) - 1
    stock_beta = beta(close, bench_daily["close"], params.beta_lookback_sessions) if bench_daily is not None and not bench_daily.empty else 1.0
    return DailyProfile(symbol=symbol, last_close=last_close, daily_atr=atr, atr_pct=atr / last_close, dollar_volume=dollar_volume, momentum=momentum, beta=stock_beta)


def select_stage1(profiles: Sequence[DailyProfile], params: VwapPullbackParameters) -> list[DailyProfile]:
    """Filter on price, ATR% band and dollar-volume percentile, then keep the top `stage1_size` by mean z(ATR%), z(momentum).

    The volume cut is a percentile of the whole profiled universe, not an absolute number: Alpaca's IEX
    volume is a small slice of the consolidated tape.
    """
    if not profiles:
        return []
    threshold = float(pd.Series([p.dollar_volume for p in profiles]).quantile(1 - params.dollar_volume_percentile))
    low, high = params.atr_pct_band
    eligible = [p for p in profiles if p.last_close >= params.min_price and low <= p.atr_pct <= high and p.dollar_volume >= threshold]
    z_atr = zscores({p.symbol: p.atr_pct for p in eligible})
    z_momentum = zscores({p.symbol: p.momentum for p in eligible})
    ranked = sorted(eligible, key=lambda p: (-(z_atr[p.symbol] + z_momentum[p.symbol]) / 2, p.symbol))
    return ranked[: params.stage1_size]


def snapshot_from(symbol: str, contexts: Sequence[BarContext]) -> IntradaySnapshot | None:
    """Stage-2 measures from the latest completed bar; None before the first one."""
    if not contexts:
        return None
    last = contexts[-1]
    return IntradaySnapshot(symbol=symbol, ret=last.close / last.session_open - 1, rs=last.rs, rvol=last.rvol, last_close=last.close, vwap=last.vwap)


def rank_stage2(snapshots: Sequence[IntradaySnapshot], params: VwapPullbackParameters, sticky: Collection[str]) -> list[RankedCandidate]:
    """The top `tracked_size` symbols passing the floor by composite z-score, then every `sticky` symbol not already in (by name).

    Floor: RVOL at least `rvol_min`, RS above 0, last close above VWAP. A symbol with no RVOL baseline
    cannot pass it. A sticky symbol (a setup already past WATCH) stays tracked whatever its rank; it gets
    zeros when it is not among the symbols passing the floor.
    """
    passing = [s for s in snapshots if s.rvol is not None and s.rvol >= params.rvol_min and s.rs > 0 and s.last_close > s.vwap]
    z_ret = zscores({s.symbol: s.ret for s in passing})
    z_rs = zscores({s.symbol: s.rs for s in passing})
    z_rvol = zscores({s.symbol: s.rvol for s in passing if s.rvol is not None})
    candidates = [RankedCandidate(symbol=s.symbol, composite=(z_ret[s.symbol] + z_rs[s.symbol] + z_rvol[s.symbol]) / 3, z_rs=z_rs[s.symbol], z_rvol=z_rvol[s.symbol]) for s in passing]
    ranked = sorted(candidates, key=lambda c: (-c.composite, c.symbol))[: params.tracked_size]
    kept = {c.symbol for c in ranked}
    by_symbol = {c.symbol: c for c in candidates}
    extras = [by_symbol.get(symbol, RankedCandidate(symbol=symbol, composite=0.0, z_rs=0.0, z_rvol=0.0)) for symbol in sorted(set(sticky) - kept)]
    return ranked + extras
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_screening.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/screening.py tests/strategies/vwap_pullback/test_vwap_screening.py
git commit -m "Task 3: vwap_pullback stage 1 and stage 2 screening

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Setup state machine (pure)

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/setups.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_setups.py`

**Interfaces:**
- Consumes: `features.BarContext`, `VwapPullbackParameters`
- Produces: `SetupState` (StrEnum: `WATCH`, `IMPULSE`, `PULLBACK`, `TRIGGERED`, `IN_TRADE`, `DONE`, `BROKEN`); frozen `Setup(symbol, state=WATCH, last_bar_time=None, bars_seen=0, volume_total=0.0, impulse_high=None, impulse_bars=0, impulse_volume_total=0.0, pullback_low=None, pullback_bars=0, pullback_volume_total=0.0, prev_high=None, trigger_close=None, triggered_at=None, retracement=0.0, largest_red_body_atr=0.0, last_close=None, last_vwap=None, last_rs=None, last_rvol=None, broken_reason=None)` with properties `impulse_avg_volume`, `pullback_avg_volume`; `step(setup, bar, daily_atr, params) -> Setup`; `advance(setup, bars, daily_atr, params) -> Setup`; `back_to_pullback(setup)`, `mark_in_trade(setup)`, `mark_done(setup)`; `health(setup) -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_setups.py`:

```python
from __future__ import annotations

import functools
from datetime import timedelta

import pytest

from tests.fakes import et
from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState, advance, back_to_pullback, health, mark_in_trade, step

PARAMS = VwapPullbackParameters()
ATR = 2.0  # daily ATR: impulse = 1.6 move, large red body > 0.5


def bar(minute: int, o: float, h: float, low: float, c: float, v: float, *, vwap: float, rs: float = 0.01, rvol: float = 2.0, high_so_far: float | None = None) -> BarContext:
    return BarContext(
        time=et(2026, 9, 1, 9, 30) + timedelta(minutes=minute),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=v,
        vwap=vwap,
        rs=rs,
        rvol=rvol,
        session_open=100.0,
        session_high=high_so_far if high_so_far is not None else h,
    )


B1 = bar(5, 100, 101, 99.9, 100.9, 1000, vwap=100.3)  # move 0.5 ATR: still WATCH
B2 = bar(10, 100.9, 101.8, 100.8, 101.7, 1000, vwap=100.8)  # move 0.9 ATR: IMPULSE
B3 = bar(15, 101.7, 101.75, 101.2, 101.3, 500, vwap=101.0, high_so_far=101.8)  # 28% retrace: PULLBACK
B4 = bar(20, 101.3, 101.4, 101.1, 101.25, 400, vwap=101.05, high_so_far=101.8)  # still pulling back
B5 = bar(25, 101.25, 101.9, 101.2, 101.85, 800, vwap=101.1, high_so_far=101.9)  # resumption: TRIGGERED


def _run(*bars: BarContext) -> Setup:
    return advance(Setup(symbol="AAA"), list(bars), ATR, PARAMS)


def test_a_healthy_path_goes_watch_impulse_pullback_triggered() -> None:
    assert _run(B1).state is SetupState.WATCH
    impulse = _run(B1, B2)
    assert impulse.state is SetupState.IMPULSE
    assert impulse.impulse_high == 101.8
    assert impulse.impulse_avg_volume == 1000
    pullback = _run(B1, B2, B3, B4)
    assert pullback.state is SetupState.PULLBACK
    assert pullback.pullback_low == 101.1
    assert pullback.pullback_avg_volume == 450
    triggered = _run(B1, B2, B3, B4, B5)
    assert triggered.state is SetupState.TRIGGERED
    assert triggered.trigger_close == 101.85
    assert triggered.triggered_at == B5.time


def test_health_flags() -> None:
    flags = health(_run(B1, B2, B3, B4))
    assert flags["state"] == "pullback"
    assert flags["vol_ratio"] == 0.45
    assert flags["duration_ratio"] == 1.0
    assert flags["retracement_pct"] == pytest.approx(30.6, abs=0.1)
    assert flags["above_vwap"] is True


@pytest.mark.parametrize(
    ("last_bar", "reason"),
    [
        (bar(20, 101.3, 101.35, 100.9, 100.95, 400, vwap=101.0), "close below VWAP"),
        (bar(20, 101.3, 101.4, 101.1, 101.25, 400, vwap=101.05, rs=-0.001), "relative strength lost"),
        (bar(20, 100.8, 100.85, 100.6, 100.65, 400, vwap=100.5), "retraced more than 61.8% of the impulse"),
        (bar(20, 101.4, 101.45, 100.8, 100.85, 400, vwap=100.5), "large bearish candle"),
        (bar(20, 101.3, 101.35, 101.1, 101.2, 1600, vwap=101.0), "heavy selling volume"),
    ],
)
def test_each_broken_reason(last_bar: BarContext, reason: str) -> None:
    broken = _run(B1, B2, B3, last_bar)
    assert broken.state is SetupState.BROKEN
    assert broken.broken_reason == reason


def test_a_pullback_longer_than_the_impulse_breaks() -> None:
    broken = _run(B1, B2, B3, B4, bar(25, 101.25, 101.3, 101.15, 101.2, 300, vwap=101.05))
    assert broken.state is SetupState.BROKEN
    assert broken.broken_reason == "pullback lasted longer than the impulse"


def test_a_new_high_without_trigger_volume_extends_the_impulse() -> None:
    extended = _run(B1, B2, B3, B4, bar(25, 101.25, 102.0, 101.2, 101.95, 300, vwap=101.1, high_so_far=102.0))
    assert extended.state is SetupState.IMPULSE
    assert extended.impulse_high == 102.0
    assert extended.pullback_low is None


def test_a_trigger_is_stale_after_one_more_bar() -> None:
    later = _run(B1, B2, B3, B4, B5, bar(30, 101.85, 101.9, 101.5, 101.6, 300, vwap=101.15, high_so_far=101.9))
    assert later.state is not SetupState.TRIGGERED  # this bar makes a new high without trigger volume: back to IMPULSE
    assert later.trigger_close is None and later.triggered_at is None


def test_replaying_all_bars_at_once_equals_one_bar_per_tick() -> None:
    bars = [B1, B2, B3, B4, B5]
    one_by_one = functools.reduce(lambda setup, b: advance(setup, [b], ATR, PARAMS), bars, Setup(symbol="AAA"))
    at_once = _run(*bars)
    assert at_once == one_by_one
    assert advance(at_once, bars, ATR, PARAMS) == at_once  # bars already seen are skipped


def test_in_trade_ignores_price_action_and_back_to_pullback_clears_the_trigger() -> None:
    in_trade = mark_in_trade(_run(B1, B2, B3, B4, B5))
    after = step(in_trade, bar(30, 101.8, 101.9, 99.0, 99.5, 5000, vwap=101.0), ATR, PARAMS)
    assert after.state is SetupState.IN_TRADE
    reverted = back_to_pullback(in_trade)
    assert reverted.state is SetupState.PULLBACK and reverted.trigger_close is None
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_setups.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.setups`

- [ ] **Step 3: Implement `setups.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/setups.py`:

```python
"""Pure per-candidate state machine (spec §3), advanced on completed 5-minute bars.

WATCH -> IMPULSE -> PULLBACK -> TRIGGERED -> IN_TRADE -> DONE, or BROKEN (terminal for the session).
`advance` replays every bar newer than the last one seen, so a tick that finds several new bars gives
exactly the result of one bar per tick. A trigger lasts one bar: the next bar re-evaluates the setup as
a pullback (it may trigger again).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters


class SetupState(StrEnum):
    WATCH = "watch"
    IMPULSE = "impulse"
    PULLBACK = "pullback"
    TRIGGERED = "triggered"
    IN_TRADE = "in_trade"
    DONE = "done"
    BROKEN = "broken"


_FROZEN = frozenset({SetupState.IN_TRADE, SetupState.DONE, SetupState.BROKEN})


@dataclass(frozen=True, slots=True)
class Setup:
    symbol: str
    state: SetupState = SetupState.WATCH
    last_bar_time: datetime | None = None
    bars_seen: int = 0
    volume_total: float = 0.0
    impulse_high: float | None = None
    impulse_bars: int = 0  # bars from the open up to the impulse high
    impulse_volume_total: float = 0.0  # their volume
    pullback_low: float | None = None
    pullback_bars: int = 0
    pullback_volume_total: float = 0.0
    prev_high: float | None = None
    trigger_close: float | None = None
    triggered_at: datetime | None = None
    retracement: float = 0.0  # of the impulse leg (session open -> impulse high), at the last close
    largest_red_body_atr: float = 0.0
    last_close: float | None = None
    last_vwap: float | None = None
    last_rs: float | None = None
    last_rvol: float | None = None
    broken_reason: str | None = None

    @property
    def impulse_avg_volume(self) -> float:
        return self.impulse_volume_total / self.impulse_bars if self.impulse_bars else 0.0

    @property
    def pullback_avg_volume(self) -> float:
        return self.pullback_volume_total / self.pullback_bars if self.pullback_bars else 0.0


def advance(setup: Setup, bars: Sequence[BarContext], daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """Replay every bar newer than `setup.last_bar_time`, oldest first."""
    for bar in bars:
        if setup.last_bar_time is None or bar.time > setup.last_bar_time:
            setup = step(setup, bar, daily_atr, params)
    return setup


def step(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    """The setup after one more completed bar."""
    seen = replace(
        setup,
        last_bar_time=bar.time,
        bars_seen=setup.bars_seen + 1,
        volume_total=setup.volume_total + bar.volume,
        last_close=bar.close,
        last_vwap=bar.vwap,
        last_rs=bar.rs,
        last_rvol=bar.rvol,
    )
    if seen.state in _FROZEN:
        return seen
    if seen.state is SetupState.TRIGGERED:  # the trigger was the previous bar: re-evaluate as a pullback
        seen = back_to_pullback(seen)
    if seen.state is SetupState.WATCH:
        return replace(_watch(seen, bar, daily_atr, params), prev_high=bar.high)
    return replace(_impulse_or_pullback(seen, bar, daily_atr, params), prev_high=bar.high)


def _watch(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    move_atr = (bar.session_high - bar.session_open) / daily_atr if daily_atr > 0 else 0.0
    if move_atr >= params.impulse_move_atr and bar.rvol is not None and bar.rvol >= params.rvol_min and bar.rs > 0 and bar.close > bar.vwap:
        return replace(setup, state=SetupState.IMPULSE, impulse_high=bar.session_high, impulse_bars=setup.bars_seen, impulse_volume_total=setup.volume_total)
    return setup


def _impulse_or_pullback(setup: Setup, bar: BarContext, daily_atr: float, params: VwapPullbackParameters) -> Setup:
    assert setup.impulse_high is not None  # set on entering IMPULSE
    leg = setup.impulse_high - bar.session_open
    retracement = (setup.impulse_high - bar.close) / leg if leg > 0 else 0.0
    setup = replace(setup, retracement=max(retracement, 0.0))
    red_body = bar.open - bar.close
    reason = _broken_reason(setup, bar, red_body, retracement, daily_atr, params)
    if reason is not None:
        return replace(setup, state=SetupState.BROKEN, broken_reason=reason)
    if setup.state is SetupState.PULLBACK:
        if setup.prev_high is not None and bar.close > setup.prev_high and bar.volume > setup.pullback_avg_volume:
            return replace(setup, state=SetupState.TRIGGERED, trigger_close=bar.close, triggered_at=bar.time)
        if bar.high > setup.impulse_high:
            return _new_high(setup, bar)
        pulled = replace(
            setup,
            pullback_low=min(setup.pullback_low if setup.pullback_low is not None else bar.low, bar.low),
            pullback_bars=setup.pullback_bars + 1,
            pullback_volume_total=setup.pullback_volume_total + bar.volume,
            largest_red_body_atr=max(setup.largest_red_body_atr, red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0),
        )
        if pulled.pullback_bars > pulled.impulse_bars:
            return replace(pulled, state=SetupState.BROKEN, broken_reason="pullback lasted longer than the impulse")
        return pulled
    if bar.high > setup.impulse_high:
        setup = _new_high(setup, bar)
    if retracement >= params.pullback_min_retrace:
        return replace(
            setup,
            state=SetupState.PULLBACK,
            pullback_low=bar.low,
            pullback_bars=1,
            pullback_volume_total=bar.volume,
            largest_red_body_atr=red_body / daily_atr if red_body > 0 and daily_atr > 0 else 0.0,
        )
    return setup


def _broken_reason(setup: Setup, bar: BarContext, red_body: float, retracement: float, daily_atr: float, params: VwapPullbackParameters) -> str | None:
    if bar.close < bar.vwap:
        return "close below VWAP"
    if bar.rs <= 0:
        return "relative strength lost"
    if retracement > params.pullback_max_retrace:
        return f"retraced more than {params.pullback_max_retrace:.1%} of the impulse"
    if red_body > params.bearish_body_atr * daily_atr:
        return "large bearish candle"
    if red_body > 0 and setup.impulse_avg_volume > 0 and bar.volume > params.selling_volume_ratio * setup.impulse_avg_volume:
        return "heavy selling volume"
    return None


def _new_high(setup: Setup, bar: BarContext) -> Setup:
    return replace(
        setup,
        state=SetupState.IMPULSE,
        impulse_high=bar.high,
        impulse_bars=setup.bars_seen,
        impulse_volume_total=setup.volume_total,
        pullback_low=None,
        pullback_bars=0,
        pullback_volume_total=0.0,
        largest_red_body_atr=0.0,
        retracement=0.0,
    )


def back_to_pullback(setup: Setup) -> Setup:
    """A trigger that was not (or could not be) acted on: keep the pullback, drop the trigger."""
    return replace(setup, state=SetupState.PULLBACK, trigger_close=None, triggered_at=None)


def mark_in_trade(setup: Setup) -> Setup:
    return replace(setup, state=SetupState.IN_TRADE)


def mark_done(setup: Setup) -> Setup:
    return replace(setup, state=SetupState.DONE)


def health(setup: Setup) -> dict[str, object]:
    """The health flags the entry agent reads (spec §3), rounded for the prompt."""
    return {
        "state": setup.state.value,
        "above_vwap": setup.last_close is not None and setup.last_vwap is not None and setup.last_close > setup.last_vwap,
        "vol_ratio": round(setup.pullback_avg_volume / setup.impulse_avg_volume, 2) if setup.impulse_avg_volume else None,
        "duration_ratio": round(setup.pullback_bars / setup.impulse_bars, 2) if setup.impulse_bars else None,
        "retracement_pct": round(100 * setup.retracement, 1),
        "rs_now_pct": round(100 * setup.last_rs, 2) if setup.last_rs is not None else None,
        "rvol_now": round(setup.last_rvol, 2) if setup.last_rvol is not None else None,
        "largest_red_body_atr": round(setup.largest_red_body_atr, 2),
    }
```

Note: in the `(bar(20, 100.8, ...), "retraced ...")` test case the reason string is built with `:.1%`, which renders `61.8%`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_setups.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/setups.py tests/strategies/vwap_pullback/test_vwap_setups.py
git commit -m "Task 4: vwap_pullback setup state machine

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Risk rules and the trade book (pure, `Decimal`)

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/risk.py`
- Create: `src/trading_agent_framework/strategies/vwap_pullback/trades.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_risk.py`, `tests/strategies/vwap_pullback/test_vwap_trades.py`

**Interfaces:**
- Consumes: `VwapPullbackParameters`, `utils.clock.MARKET_TZ`
- Produces (risk): `EntryRefused(Exception)`; `EntryPlan(quantity: Decimal, limit_price: Decimal, stop_price: Decimal, r_per_share: Decimal)`; `to_price(value: float | Decimal, rounding: str) -> Decimal`; `planned_stop(pullback_low: float, daily_atr: float, params) -> Decimal`; `plan_entry(*, trigger_close: float, pullback_low: float, last_price: Decimal, daily_atr: float, equity: Decimal, buying_power: Decimal, cash: Decimal, pending_sell_proceeds: Decimal, params) -> EntryPlan`; `in_entry_window(now: datetime, params) -> bool`; `free_slots(open_trades: int, pending_entries: int, params) -> int`; `circuit_breaker_tripped(session_pnl: Decimal, session_open_equity: Decimal, params) -> bool`
- Produces (trades): `TradeStatus` (`PENDING`, `OPEN`, `CLOSED`); mutable `Trade(symbol, entry_order_id, planned_quantity, stop_price, r_per_share, catalyst, reason, entered_at, ...)` with `stop_level`, `stop_kind: str | None`, `trail_price`, `stop_order_id`, `exit_order_ids: list[str]`, `tp1_done`, `last_review_at`, `review_flags: frozenset[str]`, `exit_reason`, `realised_pnl`, `quantity`, `filled_quantity`, `entry_price`, `status`, `closed_at`; methods `order_ids()`, `record_entry_fill(total_filled, avg_price)`, `record_exit_fill(quantity, price, at)`, `unrealised_pnl(last_price)`, `unrealised_r(last_price)`, `to_json()`; `TradeBook` with `add`, `get`, `active`, `open_trades`, `pending`, `by_order_id`, `discard`, `archive`, `closed`, `session_pnl(last_prices)`; `trade_flags(trade, *, last_close: Decimal, vwap: float | None, ema: float | None) -> frozenset[str]`; `exit_review_due(trade, flags, *, now, has_new_headline, params) -> bool`

- [ ] **Step 1: Write the failing risk tests**

Create `tests/strategies/vwap_pullback/test_vwap_risk.py`:

```python
from __future__ import annotations

from decimal import Decimal as D

import pytest

from tests.fakes import et
from trading_agent_framework.strategies.vwap_pullback import risk
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

PARAMS = VwapPullbackParameters()


def _plan(**overrides: object) -> risk.EntryPlan:
    kwargs: dict[str, object] = {
        "trigger_close": 101.85,
        "pullback_low": 101.1,
        "last_price": D("101.90"),
        "daily_atr": 2.0,
        "equity": D("100000"),
        "buying_power": D("100000"),
        "cash": D("100000"),
        "pending_sell_proceeds": D(0),
        "params": PARAMS,
    }
    return risk.plan_entry(**(kwargs | overrides))  # ty: ignore[invalid-argument-type]


def test_planned_stop_sits_a_tenth_of_an_atr_under_the_pullback_low() -> None:
    assert risk.planned_stop(101.1, 2.0, PARAMS) == D("100.90")


def test_plan_entry_takes_the_smallest_of_risk_size_and_cash_caps() -> None:
    plan = _plan()
    assert plan.stop_price == D("100.90")
    assert plan.r_per_share == D("0.95")
    assert plan.limit_price == D("102.00")
    assert plan.quantity == D(245)  # risk 526, 25% cap 245, cash 931


def test_plan_entry_refuses_a_stop_too_tight_for_the_band() -> None:
    with pytest.raises(risk.EntryRefused, match="outside"):
        _plan(pullback_low=101.8)


def test_plan_entry_refuses_to_chase() -> None:
    with pytest.raises(risk.EntryRefused, match="chasing"):
        _plan(last_price=D("102.20"))


def test_plan_entry_refuses_a_size_that_rounds_to_zero() -> None:
    with pytest.raises(risk.EntryRefused, match="0 shares"):
        _plan(cash=D("50"), buying_power=D("50"))


def test_entry_window() -> None:
    assert not risk.in_entry_window(et(2026, 9, 1, 9, 44), PARAMS)
    assert risk.in_entry_window(et(2026, 9, 1, 9, 45), PARAMS)
    assert risk.in_entry_window(et(2026, 9, 1, 15, 0), PARAMS)
    assert not risk.in_entry_window(et(2026, 9, 1, 15, 1), PARAMS)


def test_free_slots_and_circuit_breaker() -> None:
    assert risk.free_slots(3, 1, PARAMS) == 0
    assert risk.free_slots(1, 1, PARAMS) == 2
    assert risk.circuit_breaker_tripped(D("-1500"), D("100000"), PARAMS)
    assert not risk.circuit_breaker_tripped(D("-1499"), D("100000"), PARAMS)
```

- [ ] **Step 2: Write the failing trade tests**

Create `tests/strategies/vwap_pullback/test_vwap_trades.py`:

```python
from __future__ import annotations

import dataclasses
from decimal import Decimal as D

from tests.fakes import et
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeBook, TradeStatus, exit_review_due, trade_flags

PARAMS = VwapPullbackParameters()
T0 = et(2026, 9, 1, 10, 0)


def _trade(symbol: str = "AAA") -> Trade:
    return Trade(symbol=symbol, entry_order_id=f"{symbol}-entry", planned_quantity=D(100), stop_price=D("99.00"), r_per_share=D("1.00"), catalyst="earnings", reason="clean pullback", entered_at=T0)


def test_a_trade_lifecycle_records_pnl_and_closes() -> None:
    trade = _trade()
    assert trade.status is TradeStatus.PENDING and trade.stop_level == D("99.00")
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.status is TradeStatus.OPEN and trade.quantity == D(100)
    trade.record_exit_fill(D(50), D("101.00"), et(2026, 9, 1, 10, 30))
    assert trade.realised_pnl == D("50.00") and trade.quantity == D(50)
    trade.record_exit_fill(D(50), D("99.00"), et(2026, 9, 1, 11, 0))
    assert trade.status is TradeStatus.CLOSED
    assert trade.realised_pnl == D("0.00")
    assert trade.closed_at == et(2026, 9, 1, 11, 0)
    row = trade.to_json()
    assert row["symbol"] == "AAA" and row["realised_pnl"] == "0.00" and row["catalyst"] == "earnings"


def test_unrealised_pnl_and_r() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    assert trade.unrealised_pnl(D("101.50")) == D("150.00")
    assert trade.unrealised_r(D("101.50")) == 1.5


def test_book_indexes_trades_by_every_order_id_and_sums_session_pnl() -> None:
    book = TradeBook()
    a, b = _trade("AAA"), _trade("BBB")
    book.add(a)
    book.add(b)
    a.record_entry_fill(D(100), D("100.00"))
    a.stop_order_id = "AAA-stop"
    a.exit_order_ids.append("AAA-exit")
    assert book.by_order_id("AAA-stop") is a and book.by_order_id("AAA-exit") is a and book.by_order_id("BBB-entry") is b
    assert book.open_trades() == [a] and book.pending() == [b]
    a.record_exit_fill(D(100), D("101.00"), T0)
    book.archive(a)
    book.discard("BBB")
    assert book.active() == [] and book.closed == [a]
    assert book.session_pnl({}) == D("100.00")


def test_exit_review_is_due_on_a_new_flag_a_new_headline_or_elapsed_time() -> None:
    trade = _trade()
    trade.record_entry_fill(D(100), D("100.00"))
    flags = trade_flags(trade, last_close=D("101.10"), vwap=100.5, ema=100.8)
    assert flags == frozenset({"reached_1r"})
    assert exit_review_due(trade, flags, now=T0, has_new_headline=False, params=PARAMS)
    trade.review_flags, trade.last_review_at = flags, T0
    assert not exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 5), has_new_headline=False, params=PARAMS)
    assert exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 5), has_new_headline=True, params=PARAMS)
    assert exit_review_due(trade, flags, now=et(2026, 9, 1, 10, 15), has_new_headline=False, params=PARAMS)
    below = trade_flags(trade, last_close=D("100.40"), vwap=100.5, ema=100.8)
    assert below == frozenset({"below_vwap", "below_ema"})
    tp1 = dataclasses.replace(trade, tp1_done=True)
    assert trade_flags(tp1, last_close=D("101.10"), vwap=100.5, ema=100.8) == frozenset()
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_risk.py tests/strategies/vwap_pullback/test_vwap_trades.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `risk.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/risk.py`:

```python
"""Pure risk rules (spec §5): stop placement, sizing, the entry window, slots and the daily circuit breaker.

Everything that becomes an order price or a quantity is `Decimal`; float inputs come from bar maths and are
converted through `to_price`, rounded to the cent in the direction that is safe for that price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, ROUND_UP, Decimal

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.utils.clock import MARKET_TZ

_CENT = Decimal("0.01")


class EntryRefused(Exception):
    """Why `plan_entry` refused; the message reaches the entry agent as `{"error": ...}`."""


@dataclass(frozen=True, slots=True)
class EntryPlan:
    quantity: Decimal
    limit_price: Decimal
    stop_price: Decimal
    r_per_share: Decimal


def _ratio(value: float) -> Decimal:
    return Decimal(str(value))


def to_price(value: float | Decimal, rounding: str) -> Decimal:
    """A price in cents, rounded as asked (down for a sell stop, up for a buy limit)."""
    return Decimal(str(value)).quantize(_CENT, rounding=rounding)


def planned_stop(pullback_low: float, daily_atr: float, params: VwapPullbackParameters) -> Decimal:
    return to_price(pullback_low - params.stop_buffer_atr * daily_atr, ROUND_DOWN)


def plan_entry(
    *,
    trigger_close: float,
    pullback_low: float,
    last_price: Decimal,
    daily_atr: float,
    equity: Decimal,
    buying_power: Decimal,
    cash: Decimal,
    pending_sell_proceeds: Decimal,
    params: VwapPullbackParameters,
) -> EntryPlan:
    """The size, limit and stop of an entry, or `EntryRefused` with the reason."""
    stop = planned_stop(pullback_low, daily_atr, params)
    trigger = to_price(trigger_close, ROUND_HALF_UP)
    r = trigger - stop
    atr = _ratio(daily_atr)
    low, high = params.r_band_atr
    if r <= 0:
        raise EntryRefused(f"the stop {stop} is not below the trigger close {trigger}")
    if r < _ratio(low) * atr or r > _ratio(high) * atr:
        raise EntryRefused(f"risk per share {r} is outside {low}-{high} x daily ATR ({atr.quantize(_CENT)})")
    if last_price > trigger + _ratio(params.chase_guard_r) * r:
        raise EntryRefused(f"price {last_price} is more than {params.chase_guard_r}R above the trigger close {trigger}; not chasing")
    limit = (last_price + _ratio(params.entry_limit_atr) * atr).quantize(_CENT, rounding=ROUND_UP)
    by_risk = equity * _ratio(params.risk_per_trade) / r
    by_size = equity * _ratio(params.max_position_pct) / limit
    by_cash = min(buying_power, cash + pending_sell_proceeds) * _ratio(params.cash_buffer) / limit
    quantity = min(by_risk, by_size, by_cash).to_integral_value(rounding=ROUND_FLOOR)
    if quantity <= 0:
        raise EntryRefused("the position size rounds to 0 shares (not enough equity or cash for this stop distance)")
    return EntryPlan(quantity=quantity, limit_price=limit, stop_price=stop, r_per_share=r)


def in_entry_window(now: datetime, params: VwapPullbackParameters) -> bool:
    return params.no_entry_before <= now.astimezone(MARKET_TZ).time() <= params.no_entry_after


def free_slots(open_trades: int, pending_entries: int, params: VwapPullbackParameters) -> int:
    return max(0, params.max_positions - open_trades - pending_entries)


def circuit_breaker_tripped(session_pnl: Decimal, session_open_equity: Decimal, params: VwapPullbackParameters) -> bool:
    return session_open_equity > 0 and session_pnl <= -_ratio(params.max_daily_loss_pct) * session_open_equity
```

- [ ] **Step 5: Implement `trades.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/trades.py`:

```python
"""Pure trade records (spec §5): one `Trade` per entry, the session's `TradeBook`, and the exit-review triggers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

_CENT = Decimal("0.01")


class TradeStatus(StrEnum):
    PENDING = "pending"  # entry order working, nothing filled
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Trade:
    symbol: str
    entry_order_id: str
    planned_quantity: Decimal
    stop_price: Decimal  # the initial stop, which defines R
    r_per_share: Decimal
    catalyst: str
    reason: str
    entered_at: datetime
    status: TradeStatus = TradeStatus.PENDING
    quantity: Decimal = Decimal(0)  # shares held now
    filled_quantity: Decimal = Decimal(0)  # shares bought in total
    entry_price: Decimal | None = None
    stop_level: Decimal = Decimal(0)  # the working stop's level (initial stop, raised, or the trail's start)
    stop_kind: str | None = None  # "stop" | "trail"
    trail_price: Decimal | None = None
    stop_order_id: str | None = None
    exit_order_ids: list[str] = field(default_factory=list)
    tp1_done: bool = False
    last_review_at: datetime | None = None
    review_flags: frozenset[str] = frozenset()
    exit_reason: str | None = None
    realised_pnl: Decimal = Decimal(0)
    closed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.stop_level:
            self.stop_level = self.stop_price

    def order_ids(self) -> set[str]:
        return {i for i in (self.entry_order_id, self.stop_order_id, *self.exit_order_ids) if i is not None}

    def record_entry_fill(self, total_filled: Decimal, avg_price: Decimal) -> None:
        """The entry's cumulative fill (not an increment): holdings and entry price follow it."""
        self.filled_quantity = total_filled
        self.quantity = total_filled
        self.entry_price = avg_price
        self.status = TradeStatus.OPEN

    def record_exit_fill(self, quantity: Decimal, price: Decimal, at: datetime) -> None:
        if self.entry_price is not None:
            self.realised_pnl = (self.realised_pnl + (price - self.entry_price) * quantity).quantize(_CENT)
        self.quantity -= quantity
        if self.quantity <= 0:
            self.quantity = Decimal(0)
            self.status = TradeStatus.CLOSED
            self.closed_at = at

    def unrealised_pnl(self, last_price: Decimal) -> Decimal:
        if self.entry_price is None:
            return Decimal(0)
        return ((last_price - self.entry_price) * self.quantity).quantize(_CENT)

    def unrealised_r(self, last_price: Decimal) -> float:
        if self.entry_price is None or self.r_per_share <= 0:
            return 0.0
        return round(float((last_price - self.entry_price) / self.r_per_share), 2)

    def to_json(self) -> dict[str, object]:
        risked = self.r_per_share * self.filled_quantity
        return {
            "symbol": self.symbol,
            "catalyst": self.catalyst,
            "reason": self.reason,
            "entered_at": self.entered_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "filled_quantity": str(self.filled_quantity),
            "stop_price": str(self.stop_price),
            "r_per_share": str(self.r_per_share),
            "realised_pnl": str(self.realised_pnl),
            "realised_r": round(float(self.realised_pnl / risked), 2) if risked > 0 else None,
            "tp1_done": self.tp1_done,
            "stop_kind": self.stop_kind,
            "exit_reason": self.exit_reason,
        }


class TradeBook:
    """The session's trades: active ones by symbol (one per symbol), closed ones in order."""

    def __init__(self) -> None:
        self._active: dict[str, Trade] = {}
        self.closed: list[Trade] = []

    def add(self, trade: Trade) -> None:
        self._active[trade.symbol] = trade

    def get(self, symbol: str) -> Trade | None:
        return self._active.get(symbol)

    def active(self) -> list[Trade]:
        return list(self._active.values())

    def open_trades(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.OPEN]

    def pending(self) -> list[Trade]:
        return [t for t in self._active.values() if t.status is TradeStatus.PENDING]

    def by_order_id(self, order_id: str) -> Trade | None:
        return next((t for t in self._active.values() if order_id in t.order_ids()), None)

    def discard(self, symbol: str) -> None:
        self._active.pop(symbol, None)

    def archive(self, trade: Trade) -> None:
        self._active.pop(trade.symbol, None)
        self.closed.append(trade)

    def session_pnl(self, last_prices: Mapping[str, Decimal]) -> Decimal:
        """Realised P&L of every trade plus the open trades' unrealised P&L at `last_prices` (a missing price counts 0)."""
        total = sum((t.realised_pnl for t in [*self.closed, *self._active.values()]), Decimal(0))
        for trade in self.open_trades():
            price = last_prices.get(trade.symbol)
            if price is not None:
                total += trade.unrealised_pnl(price)
        return total


def trade_flags(trade: Trade, *, last_close: Decimal, vwap: float | None, ema: float | None) -> frozenset[str]:
    """The exit-review events that currently hold for `trade`: +1R reached (before TP1), close below VWAP, close below the EMA."""
    flags: set[str] = set()
    if trade.entry_price is not None and not trade.tp1_done and last_close >= trade.entry_price + trade.r_per_share:
        flags.add("reached_1r")
    if vwap is not None and last_close < Decimal(str(vwap)):
        flags.add("below_vwap")
    if ema is not None and last_close < Decimal(str(ema)):
        flags.add("below_ema")
    return frozenset(flags)


def exit_review_due(trade: Trade, flags: frozenset[str], *, now: datetime, has_new_headline: bool, params: VwapPullbackParameters) -> bool:
    """Whether the exit agent should look at `trade` now: a flag that was not there at the last review, a new headline, or enough time."""
    if trade.status is not TradeStatus.OPEN:
        return False
    if flags - trade.review_flags or has_new_headline:
        return True
    since = trade.last_review_at or trade.entered_at
    return now - since >= timedelta(minutes=params.exit_review_minutes)
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_risk.py tests/strategies/vwap_pullback/test_vwap_trades.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (11 tests)

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/risk.py src/trading_agent_framework/strategies/vwap_pullback/trades.py tests/strategies/vwap_pullback/test_vwap_risk.py tests/strategies/vwap_pullback/test_vwap_trades.py
git commit -m "Task 5: vwap_pullback risk rules and trade book

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Session state, lean headlines and the two prompts

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/session.py`
- Create: `src/trading_agent_framework/strategies/vwap_pullback/news.py`
- Create: `src/trading_agent_framework/strategies/vwap_pullback/prompts.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_news_prompts.py`

**Interfaces:**
- Consumes: `features.BarContext`, `features.BarStamp`, `setups.Setup`, `trades.TradeBook`, `utils.clock.MarketSession`
- Produces: `CandidateInfo(symbol, daily_atr, beta, composite=0.0, z_rs=0.0, z_rvol=0.0)` (mutable); `SessionState(day, session, bar_stamp, session_open_equity, candidates={}, baselines={}, setups={}, contexts={}, book=TradeBook(), headlines={}, headlines_fetched_at={}, new_headline=set(), decided=set(), unknown_positions_checked=False, flattened=False)`; `lean_headlines(articles, limit) -> list[dict[str, str]]`; `CATALYSTS: tuple[str, ...]`; `build_entry_prompt(params) -> str`; `build_exit_prompt(params, *, flatten_time: str) -> str`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_news_prompts.py`:

```python
from __future__ import annotations

from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.prompts import CATALYSTS, build_entry_prompt, build_exit_prompt


def test_lean_headlines_keeps_the_newest_distinct_headlines() -> None:
    articles = [
        {"headline": "AAA beats estimates", "created_at": "2026-09-01T12:00:00Z", "source": "benzinga", "summary": "long text"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "", "created_at": "2026-09-01T14:00:00Z", "source": "x"},
        {"headline": "Old news", "created_at": "2026-08-30T12:00:00Z", "source": "x"},
    ]
    rows = lean_headlines(articles, 2)
    assert rows == [
        {"headline": "AAA raises guidance", "created_at": "2026-09-01T13:00:00Z", "source": "benzinga"},
        {"headline": "aaa BEATS estimates", "created_at": "2026-09-01T12:05:00Z", "source": "other"},
    ]


def test_entry_prompt_names_the_tools_the_catalysts_and_the_thresholds() -> None:
    prompt = build_entry_prompt(VwapPullbackParameters())
    for word in ("enter_long", "pass_on_setup", "search_news", "M&A", "offering", *CATALYSTS):
        assert word in prompt
    assert "2.0" in prompt  # none_catalyst_min_z
    assert "at most 4" in prompt  # news_calls_per_run


def test_exit_prompt_names_every_action_and_the_flatten_time() -> None:
    prompt = build_exit_prompt(VwapPullbackParameters(), flatten_time="15:50")
    for word in ("take_partial_profit", "tighten_stop", "replace_stop_with_trailing", "exit_position", "hold", "already_stopped_out", "15:50", "0.25", "2.0"):
        assert word in prompt
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_news_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the three modules**

Create `src/trading_agent_framework/strategies/vwap_pullback/session.py`:

```python
"""Per-session state of the vwap_pullback strategy: rebuilt at every session, kept in `strategy.vars.session`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

from trading_agent_framework.strategies.vwap_pullback.features import BarContext, BarStamp
from trading_agent_framework.strategies.vwap_pullback.setups import Setup
from trading_agent_framework.strategies.vwap_pullback.trades import TradeBook
from trading_agent_framework.utils.clock import MarketSession


@dataclass
class CandidateInfo:
    """A stage-1 survivor: its daily measures, plus its latest stage-2 scores."""

    symbol: str
    daily_atr: float
    beta: float
    composite: float = 0.0
    z_rs: float = 0.0
    z_rvol: float = 0.0


@dataclass
class SessionState:
    day: date
    session: MarketSession
    bar_stamp: BarStamp
    session_open_equity: Decimal
    candidates: dict[str, CandidateInfo] = field(default_factory=dict)
    baselines: dict[str, pd.Series] = field(default_factory=dict)
    setups: dict[str, Setup] = field(default_factory=dict)
    contexts: dict[str, list[BarContext]] = field(default_factory=dict)
    book: TradeBook = field(default_factory=TradeBook)
    headlines: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    headlines_fetched_at: dict[str, datetime] = field(default_factory=dict)
    new_headline: set[str] = field(default_factory=set)  # symbols with a headline the exit agent has not reviewed
    decided: set[str] = field(default_factory=set)  # triggered symbols the entry agent entered or passed this tick
    unknown_positions_checked: bool = False
    flattened: bool = False
```

Create `src/trading_agent_framework/strategies/vwap_pullback/news.py`:

```python
"""Pure headline trimming: what the agents see of a symbol's news in a setup or trade row."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_MAX_HEADLINE_CHARS = 200


def lean_headlines(articles: Sequence[Mapping[str, object]], limit: int) -> list[dict[str, str]]:
    """The `limit` newest distinct headlines (case-insensitive), as `{headline, created_at, source}`."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for article in sorted(articles, key=lambda a: str(a.get("created_at", "")), reverse=True):
        headline = str(article.get("headline") or "").strip()
        key = headline.casefold()
        if not headline or key in seen:
            continue
        seen.add(key)
        rows.append({"headline": headline[:_MAX_HEADLINE_CHARS], "created_at": str(article.get("created_at", "")), "source": str(article.get("source", ""))})
        if len(rows) >= limit:
            break
    return rows
```

Create `src/trading_agent_framework/strategies/vwap_pullback/prompts.py`:

```python
"""The entry and exit agents' system prompts, built from the parameters so no threshold is written twice."""

from __future__ import annotations

from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters

CATALYSTS: tuple[str, ...] = ("earnings", "guidance", "analyst", "contract_or_product", "sector_or_macro", "none")


def build_entry_prompt(params: VwapPullbackParameters) -> str:
    catalysts = ", ".join(CATALYSTS)
    return (
        "You are the entry trader of an intraday VWAP pullback continuation strategy. Long only, no margin, never short. "
        "Always write in English, in your reply and in every tool argument.\n\n"
        "The code has already found stocks with abnormal intraday strength (high relative volume, strength against SPY) that "
        "pulled back toward VWAP, and it marks a setup 'triggered' when a 5-minute bar resumed upward: it closed above the "
        "previous bar's high, above VWAP, on more volume than the pullback bars. You judge each triggered setup and enter it or "
        "pass. You never choose a size or a price: enter_long sizes the position and places the protective stop itself, and "
        "refuses anything that breaks a rule.\n\n"
        "For every setup whose state is 'triggered' in the context:\n"
        "1. Read its health. Healthy: vol_ratio below 1 (the pullback traded less than the impulse), duration_ratio at most 1, "
        f"retracement_pct between {100 * params.pullback_min_retrace:.0f} and {100 * params.pullback_max_retrace:.0f}, rs_now_pct above 0, "
        "above_vwap true, a small largest_red_body_atr. Several weak flags together mean pass.\n"
        f"2. Read its headlines (in the setup row). Call search_news only if they are ambiguous -- at most {params.news_calls_per_run} "
        "searches per run.\n"
        f"3. Label the catalyst, one of: {catalysts}.\n"
        "4. Never enter an M&A target (its price is pinned to the deal) or a stock with an offering, dilution or share-sale "
        "headline today.\n"
        f"5. With catalyst 'none', enter only if both z_rs and z_rvol are above {params.none_catalyst_min_z}: a big move without "
        "news is more likely to reverse.\n"
        "6. Call enter_long(symbol, catalyst, reason) to enter or pass_on_setup(symbol, reason) to pass -- exactly one of them "
        "for every triggered setup. If enter_long returns an 'error', nothing was placed: read the reason and do not retry that "
        "symbol in this run.\n"
        "Never enter because price merely touched VWAP: the trigger is the resumption. Setups in state 'pullback' are context "
        "only; they cannot be entered yet. Finish with a one-line summary and make no further tool call."
    )


def build_exit_prompt(params: VwapPullbackParameters, *, flatten_time: str) -> str:
    tp_low, tp_high = params.tp1_fraction_band
    trail_low, trail_high = params.trail_atr_band
    return (
        "You are the exit trader of an intraday VWAP pullback continuation strategy. Long only. Always write in English, "
        "in your reply and in every tool argument.\n\n"
        f"Every open trade already has a protective stop at the broker, placed by the code, and the code sells everything at "
        f"{flatten_time} whatever you decide. Your job is to keep the rare large winners running and to cut trades whose reason "
        "to exist is gone. R is the trade's initial risk per share; unrealised_r is its open profit in R.\n\n"
        "For every open trade in the context, choose exactly one action:\n"
        f"- take_partial_profit(symbol, fraction): once unrealised_r reaches about 1 and tp1_done is false, sell a fraction of "
        f"{tp_low} to {tp_high} of the position. Once per trade.\n"
        "- tighten_stop(symbol, stop_price): raise the stop under structure -- just below VWAP or the 9-EMA (both in the trade "
        "row) -- once the trade is above 1R. A stop only ever moves up.\n"
        f"- replace_stop_with_trailing(symbol, trail_atr): switch to a trailing stop of trail_atr ({trail_low} to {trail_high}) "
        "5-minute ATRs when the trade trends cleanly and should run without more reviews.\n"
        "- exit_position(symbol, reason): sell now -- on a bearish headline (downgrade, offering, halt), or when price loses "
        "VWAP on heavy volume.\n"
        "- hold(symbol, reason): change nothing this time. Moves inside 1R are noise: holding is the default.\n"
        f"Call search_news only if a new headline needs context -- at most {params.news_calls_per_run} searches per run. If a tool "
        "returns 'already_stopped_out', the stop already closed that trade: nothing more to do for it. If it returns an 'error', "
        "nothing was changed. Finish with a one-line summary and make no further tool call."
    )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_news_prompts.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/session.py src/trading_agent_framework/strategies/vwap_pullback/news.py src/trading_agent_framework/strategies/vwap_pullback/prompts.py tests/strategies/vwap_pullback/test_vwap_news_prompts.py
git commit -m "Task 6: vwap_pullback session state, lean headlines and agent prompts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `Scanner` (stage-1 session preparation and the per-tick scan)

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/scanner.py`
- Modify: `tests/fakes.py` (`FakeBroker` timestep frames and news; `FakeNewsProvider`)
- Test: `tests/strategies/vwap_pullback/test_vwap_scanner.py`

**Interfaces:**
- Consumes: everything from Tasks 2–6; `Strategy.get_historical_prices_for_assets`, `Strategy.get_portfolio_value`, `Strategy.clock.next_session()`, `Broker.news_provider()`
- Produces: `Scanner(strategy, params, universe, *, benchmark="SPY", preload: Callable[[Sequence[Asset], str], None] | None = None)` with `bar_stamp` property, `prepare_session() -> SessionState`, `scan(state) -> None`, `refresh_headlines(state, now) -> None`
- Produces (tests/fakes.py): `FakeBroker.timestep_frames: dict[tuple[str, str], pd.DataFrame]` (checked before `bar_frames`), `FakeBroker.news: NewsProvider | None` (returned by `news_provider()`), `FakeNewsProvider(articles: dict[str, list[dict]])` with `.calls`

- [ ] **Step 1: Extend the fakes**

In `tests/fakes.py`, in `FakeBroker.__init__` add:

```python
        self.timestep_frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.news: Any = None
```

Replace the `return {...}` of `FakeBroker.get_bars` with:

```python
        frames = {asset: self.timestep_frames.get((asset.symbol, timestep), self.bar_frames.get(asset.symbol)) for asset in requested}
        return {asset: Bars(asset=asset, timestep=timestep, df=frame.iloc[-length:]) for asset, frame in frames.items() if frame is not None}
```

Add to `FakeBroker`:

```python
    def news_provider(self) -> Any:
        return self.news
```

Append to `tests/fakes.py`:

```python
class FakeNewsProvider:
    """A `NewsProvider` over canned lean articles per symbol, filtered to `[start, end]` like the real one."""

    def __init__(self, articles: dict[str, list[dict[str, object]]] | None = None) -> None:
        self.articles = articles or {}
        self.calls: list[tuple[tuple[str, ...], datetime | None, datetime, int, bool]] = []

    def get_news(self, symbols: Sequence[str] = (), *, start: datetime | None = None, end: datetime, limit: int = 10, include_content: bool = False) -> list[dict[str, object]]:
        self.calls.append((tuple(symbols), start, end, limit, include_content))
        rows = [a for symbol in symbols for a in self.articles.get(symbol, [])]
        in_window = [a for a in rows if (start is None or datetime.fromisoformat(str(a["created_at"])) >= start) and datetime.fromisoformat(str(a["created_at"])) <= end]
        return in_window[:limit]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_scanner.py`:

```python
from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et, make_bars_frame, make_session, minute_ohlc
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState

DAY = date(2026, 9, 2)
PARAMS = dataclasses.replace(VwapPullbackParameters(), rvol_baseline_sessions=2)


def _strategy(tmp_path: Path, now: pd.Timestamp) -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap")
    return Strategy(broker, mode=TradingMode.PAPER, project_root=tmp_path), broker


def _minutes(day: date, count: int, volume: float) -> pd.DataFrame:
    return minute_ohlc(et(day.year, day.month, day.day, 9, 30), [(100, 100, 100, 100, volume)] * count)


def test_prepare_session_runs_stage_one_and_builds_rvol_baselines(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 8, 30))
    start = et(2026, 6, 1)
    broker.timestep_frames = {
        ("AAA", "day"): make_bars_frame([90.0 + 0.5 * i for i in range(71)], start=start),
        ("BBB", "day"): make_bars_frame([100.0] * 71, start=start),
        ("CHEAP", "day"): make_bars_frame([3.0] * 71, start=start),
        ("WILD", "day"): make_bars_frame([20.0] * 71, start=start),
        ("SPY", "day"): make_bars_frame([400.0] * 71, start=start),
        ("AAA", "minute"): pd.concat(
            [
                _minutes(date(2026, 8, 31), 5, 100),
                minute_ohlc(et(2026, 9, 1, 9, 29), [(100, 100, 100, 100, 999)]),  # premarket: excluded
                _minutes(date(2026, 9, 1), 5, 100),
            ]
        ),
    }
    state = Scanner(strategy, PARAMS, ["AAA", "BBB", "CHEAP", "WILD"]).prepare_session()
    assert state.day == DAY
    assert state.bar_stamp == "open"
    assert set(state.candidates) == {"AAA", "BBB"}
    assert state.candidates["BBB"].daily_atr == pytest.approx(2.0)
    assert state.baselines["AAA"].tolist() == [100.0, 200.0, 300.0, 400.0, 500.0]
    assert "BBB" not in state.baselines  # no minute data: no baseline, so it cannot pass the stage-2 floor
    assert state.session_open_equity == Decimal("25000")


def _state(**candidates: CandidateInfo) -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"), candidates=dict(candidates))


def test_scan_builds_contexts_and_advances_setups(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 9, 50, 30))
    rising = [(100 + 0.06 * i, 100 + 0.06 * (i + 1) + 0.01, 100 + 0.06 * i - 0.01, 100 + 0.06 * (i + 1), 300.0) for i in range(20)]
    broker.timestep_frames = {
        ("AAA", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), rising),
        ("SPY", "minute"): minute_ohlc(et(2026, 9, 2, 9, 30), [(400, 400, 400, 400, 1000)] * 20),
    }
    broker.news = FakeNewsProvider()
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.baselines["AAA"] = pd.Series([100.0 * (m + 1) for m in range(390)])
    Scanner(strategy, PARAMS, ["AAA"]).scan(state)
    assert [c.time for c in state.contexts["AAA"]][-1] == et(2026, 9, 2, 9, 50)
    assert state.setups["AAA"].state is SetupState.IMPULSE
    assert state.setups["AAA"].impulse_high == pytest.approx(101.21)
    assert broker.news.calls == []  # headlines are only fetched for pullback/triggered setups and open trades


def test_refresh_headlines_fetches_active_setups_and_flags_new_ones(tmp_path: Path) -> None:
    strategy, broker = _strategy(tmp_path, et(2026, 9, 2, 10, 0))
    broker.news = FakeNewsProvider({"AAA": [{"headline": "AAA beats", "created_at": "2026-09-02T07:00:00-04:00", "source": "b"}]})
    state = _state(AAA=CandidateInfo(symbol="AAA", daily_atr=1.0, beta=1.0))
    state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.PULLBACK)
    scanner = Scanner(strategy, PARAMS, ["AAA"])
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 0))
    assert state.headlines["AAA"][0]["headline"] == "AAA beats"
    assert state.new_headline == set()  # the first fetch is the baseline, not news
    broker.news.articles["AAA"].append({"headline": "AAA upgraded", "created_at": "2026-09-02T10:10:00-04:00", "source": "b"})
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 10))  # within the refresh interval: no call
    assert len(broker.news.calls) == 1
    scanner.refresh_headlines(state, et(2026, 9, 2, 10, 15))
    assert state.new_headline == {"AAA"}
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.scanner`

- [ ] **Step 4: Implement `scanner.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/scanner.py`:

```python
"""Session preparation (stage 1) and the per-tick scan (stage 2, setups, headlines), spec §2-§3.

The I/O side of candidate selection: bars and news come in through the strategy and broker, go through the
pure `features`/`screening`/`setups` modules, and land in `SessionState`. Price reads go through the strategy
(and so, in a backtest, through `BacktestBroker._source_bars`, the no-look-ahead gate); news is cut at the
strategy clock.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.vwap_pullback.features import (
    BarStamp,
    cumulative_volume_by_minute,
    intraday_contexts,
    minute_starts,
    rvol_baseline,
    session_slice,
)
from trading_agent_framework.strategies.vwap_pullback.news import lean_headlines
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.screening import daily_profile, rank_stage2, select_stage1, snapshot_from
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState, advance
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

Preload = Callable[[Sequence[Asset], str], None]

_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_SESSION_MINUTES = 390
_NEWS_LOOKBACK = timedelta(hours=18)  # from the open back to roughly the previous close
_EMPTY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz=MARKET_TZ))


class Scanner:
    def __init__(self, strategy: Strategy, params: VwapPullbackParameters, universe: Sequence[str], *, benchmark: str = "SPY", preload: Preload | None = None) -> None:
        self._strategy = strategy
        self._params = params
        self._universe = list(universe)
        self._benchmark = benchmark
        self._preload = preload

    @property
    def bar_stamp(self) -> BarStamp:
        return "close" if self._strategy.is_backtesting else "open"

    # --- stage 1 -------------------------------------------------------------------

    def prepare_session(self) -> SessionState:
        """Stage 1 for the next (or current) session: candidates, their beta and ATR, and their RVOL baselines."""
        strategy = self._strategy
        session = strategy.clock.next_session()
        if session is None:
            raise BrokerError("no upcoming market session to prepare")
        day = session.open.astimezone(MARKET_TZ).date()
        bench = Asset(self._benchmark)
        universe = [Asset(symbol) for symbol in self._universe]
        if self._preload is not None:
            self._preload([*universe, bench], "day")
        daily = strategy.get_historical_prices_for_assets([*universe, bench], self._params.stage1_lookback_sessions + 1, "day")
        bench_daily = _before(daily.get(bench), day)
        profiles = []
        for asset in universe:
            frame = _before(daily.get(asset), day)
            profile = daily_profile(asset.symbol, frame, bench_daily, self._params) if frame is not None else None
            if profile is not None:
                profiles.append(profile)
        chosen = select_stage1(profiles, self._params)
        state = SessionState(day=day, session=session, bar_stamp=self.bar_stamp, session_open_equity=strategy.get_portfolio_value())
        state.candidates = {p.symbol: CandidateInfo(symbol=p.symbol, daily_atr=p.daily_atr, beta=p.beta) for p in chosen}
        state.baselines = self._baselines([Asset(p.symbol) for p in chosen], day)
        strategy.log_info(f"stage 1 for {day}: {len(chosen)} candidates out of {len(profiles)} profiled symbols")
        return state

    def _baselines(self, assets: Sequence[Asset], day: date) -> dict[str, pd.Series]:
        if not assets:
            return {}
        if self._preload is not None:
            self._preload([*assets, Asset(self._benchmark)], "minute")
        length = (self._params.rvol_baseline_sessions + 1) * _SESSION_MINUTES
        bars = self._strategy.get_historical_prices_for_assets(assets, length, "minute")
        baselines: dict[str, pd.Series] = {}
        for asset in assets:
            found = bars.get(asset)
            if found is not None and not found.df.empty:
                baseline = self._baseline(found.df, day)
                if not baseline.empty:
                    baselines[asset.symbol] = baseline
        return baselines

    def _baseline(self, df: pd.DataFrame, day: date) -> pd.Series:
        starts = minute_starts(df.index, self.bar_stamp).tz_convert(MARKET_TZ)
        dates = sorted({d for d in starts.date if d < day})[-self._params.rvol_baseline_sessions :]
        per_session = []
        for session_day in dates:
            open_at = datetime.combine(session_day, _REGULAR_OPEN, tzinfo=MARKET_TZ)
            rows = session_slice(df, open_at, datetime.combine(session_day, _REGULAR_CLOSE, tzinfo=MARKET_TZ), self.bar_stamp)
            if not rows.empty:
                per_session.append(cumulative_volume_by_minute(rows, open_at, self.bar_stamp))
        return rvol_baseline(per_session)

    # --- stage 2 and setups -----------------------------------------------------------

    def scan(self, state: SessionState) -> None:
        """One tick: contexts for every candidate, stage-2 ranking, setups advanced, headlines refreshed."""
        now = self._strategy.get_datetime()
        bench = Asset(self._benchmark)
        minutes_open = int((now - state.session.open).total_seconds() // 60)
        length = max(10, min(minutes_open + 5, _SESSION_MINUTES + 10))
        bars = self._strategy.get_historical_prices_for_assets([*(Asset(s) for s in state.candidates), bench], length, "minute")
        bench_df = self._session_frame(bars.get(bench), state)
        snapshots = []
        for symbol, info in state.candidates.items():
            contexts = intraday_contexts(
                self._session_frame(bars.get(Asset(symbol)), state),
                bench_df,
                session_open=state.session.open,
                now=now,
                bar_stamp=state.bar_stamp,
                beta=info.beta,
                baseline=state.baselines.get(symbol, pd.Series(dtype=float)),
                minutes=self._params.bar_minutes,
            )
            state.contexts[symbol] = contexts
            snapshot = snapshot_from(symbol, contexts)
            if snapshot is not None:
                snapshots.append(snapshot)
        sticky = {symbol for symbol, setup in state.setups.items() if setup.state is not SetupState.WATCH}
        ranked = rank_stage2(snapshots, self._params, sticky)
        for candidate in ranked:
            info = state.candidates[candidate.symbol]
            info.composite, info.z_rs, info.z_rvol = candidate.composite, candidate.z_rs, candidate.z_rvol
        tracked = {candidate.symbol for candidate in ranked}
        for symbol in [s for s, setup in state.setups.items() if s not in tracked and setup.state is SetupState.WATCH]:
            del state.setups[symbol]
        for symbol in sorted(tracked):
            setup = state.setups.get(symbol, Setup(symbol=symbol))
            state.setups[symbol] = advance(setup, state.contexts.get(symbol, []), state.candidates[symbol].daily_atr, self._params)
        self.refresh_headlines(state, now)

    @staticmethod
    def _session_frame(bars: Bars | None, state: SessionState) -> pd.DataFrame:
        if bars is None or bars.df.empty:
            return _EMPTY
        return session_slice(bars.df, state.session.open, state.session.close, state.bar_stamp)

    def refresh_headlines(self, state: SessionState, now: datetime) -> None:
        """Headlines for pullback/triggered setups and open trades, at most once per `exit_review_minutes` per symbol.

        A headline that was not in the previous fetch marks the symbol in `state.new_headline` (an exit-review
        event); a symbol's first fetch is its baseline, not news.
        """
        wanted = {s for s, setup in state.setups.items() if setup.state in (SetupState.PULLBACK, SetupState.TRIGGERED)}
        wanted |= {trade.symbol for trade in state.book.open_trades()}
        if not wanted:
            return
        try:
            provider = self._strategy.broker.news_provider()
        except BrokerError as exc:
            self._strategy.log_warning(f"no news provider, setups go without headlines: {exc}")
            return
        if provider is None:
            return
        refresh = timedelta(minutes=self._params.exit_review_minutes)
        since = state.session.open - _NEWS_LOOKBACK
        for symbol in sorted(wanted):
            fetched = state.headlines_fetched_at.get(symbol)
            if fetched is not None and now - fetched < refresh:
                continue
            try:
                articles = provider.get_news([symbol], start=since, end=now, limit=self._params.headlines_per_symbol * 3)
            except BrokerError as exc:
                self._strategy.log_warning(f"news for {symbol} unavailable: {exc}")
                continue
            rows = lean_headlines(articles, self._params.headlines_per_symbol)
            previous = state.headlines.get(symbol)
            if previous is not None and {r["headline"] for r in rows} - {r["headline"] for r in previous}:
                state.new_headline.add(symbol)
            state.headlines[symbol] = rows
            state.headlines_fetched_at[symbol] = now


def _before(bars: Bars | None, day: date) -> pd.DataFrame | None:
    """Only the daily rows dated before `day`: a restart mid-session must not read today's forming daily bar."""
    if bars is None or bars.df.empty:
        return None
    df = bars.df
    dates = df.index.tz_convert(MARKET_TZ).date
    return df[dates < day]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_scanner.py -v && uv run pytest -q && uv run ruff check`
Expected: PASS (3 new tests), full suite green (the `FakeBroker.get_bars` change keeps `bar_frames` behaviour).

If `test_scan_builds_contexts_and_advances_setups`'s `impulse_high` differs slightly, recompute from the path: each row's high is `close + 0.01`; bucket 3's last close is `101.20`, so the impulse high is `101.21`. Fix the path, not the assertion's meaning.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/scanner.py tests/fakes.py tests/strategies/vwap_pullback/test_vwap_scanner.py
git commit -m "Task 7: vwap_pullback scanner (stage 1 preparation, per-tick scan, headlines)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `Desk` entry side: entries, protective stops, fills and cancels, reconcile, flatten, restart

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/desk.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_desk_entries.py`

**Interfaces:**
- Consumes: `risk`, `trades`, `setups.back_to_pullback/mark_in_trade/mark_done`, `features.latest_levels`, `SessionState`; `Strategy.create_order/submit_order/cancel_order/get_order/get_position/get_positions/close_position/wait_for_order_execution/wait_for_orders_execution/get_last_price/broker.get_account()/broker.tracker.get_active_orders()`
- Produces: `STOPPED_OUT = "already_stopped_out"`; `Desk(strategy, params, *, trade_log: Callable[[], Path | None])` with: `state` property; views `levels(symbol)`, `last_close(symbol) -> Decimal | None`, `planned_risk(symbol) -> tuple[Decimal, Decimal] | None`, `minutes_to_flatten(now) -> int`, `session_pnl()`, `breaker_tripped()`, `free_slots()`, `entry_due()`, `exit_review_due(now) -> list[str]`, `mark_reviewed(now)`; actions `enter_long(symbol, catalyst, reason) -> dict`, `pass_on_setup(symbol, reason) -> dict`, `reconcile(now)`, `on_order_filled(order, price, quantity)`, `on_order_canceled(order)`, `flatten_all(reason)`, `close_unknown_positions()`; internals used by Task 9: `_open_trade(symbol) -> Trade | dict`, `_release_stop(trade) -> str | None`, `_submit_stop(trade, quantity)`, `_market_sell(trade, quantity, reason) -> Order | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_desk_entries.py`:

```python
from __future__ import annotations

import dataclasses
import json
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

from tests.fakes import FakeClock, FrameDataSource, et, make_session, minute_ohlc
from trading_agent_framework.backtesting.broker import BacktestBroker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.features import BarContext
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo, SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.strategies.vwap_pullback.trades import TradeStatus

DAY = date(2026, 9, 1)
FLAT = (100.0, 100.2, 99.8, 100.0, 1000.0)
# Close-stamped minute bars 09:31..11:00: flat at 100, then a drop through 99.30 at 10:03.
ROWS = [FLAT] * 32 + [(100.0, 100.0, 99.0, 99.2, 1000.0)] + [(99.2, 99.4, 99.0, 99.2, 1000.0)] * 57


class Rig:
    def __init__(self, tmp_path: Path, *, now=et(2026, 9, 1, 10, 0), params: VwapPullbackParameters | None = None) -> None:
        self.clock = FakeClock(now, [make_session(DAY)])
        frames = {(symbol, "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS) for symbol in ("AAA", "MSFT")}
        self.broker = BacktestBroker("vwap", data_source=FrameDataSource(frames), clock=self.clock, budget=D("100000"), timestep="minute")
        self.strategy = Strategy(self.broker, mode=TradingMode.BACKTESTING, project_root=tmp_path)
        self.strategy.minutes_before_closing = 10  # as VwapPullbackStrategy: the flatten runs at 15:50
        self.strategy.vars.session = SessionState(day=DAY, session=make_session(DAY), bar_stamp="close", session_open_equity=D("100000"))
        self.state.candidates["AAA"] = CandidateInfo(symbol="AAA", daily_atr=2.0, beta=1.0, z_rs=2.5, z_rvol=2.5)
        self.state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
        self.state.contexts["AAA"] = [
            BarContext(time=et(2026, 9, 1, 10, 0), open=100, high=100.2, low=99.8, close=100, volume=5000, vwap=99.9, rs=0.01, rvol=2.0, session_open=99.0, session_high=100.2)
        ]
        self.log = tmp_path / "trades.jsonl"
        self.desk = Desk(self.strategy, params or VwapPullbackParameters(), trade_log=lambda: self.log)

    @property
    def state(self) -> SessionState:
        return self.strategy.vars.session

    def advance(self, seconds: float) -> None:
        before = self.clock.now()
        self.clock.advance(seconds)
        self.broker.on_advance(before, self.clock.now())

    def fill_hook(self, order: Order) -> None:
        assert order.is_filled()
        self.desk.on_order_filled(order, order.avg_fill_price, order.filled_quantity)

    def open_trade(self) -> None:
        self.desk.enter_long("AAA", "earnings", "clean pullback")
        entry = self.strategy.get_order(self.state.book.get("AAA").entry_order_id)
        self.advance(60)
        self.fill_hook(entry)


def test_enter_long_sizes_the_trade_and_submits_a_limit_buy(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    result = rig.desk.enter_long("aaa", "earnings", "clean pullback")
    assert result == {"symbol": "AAA", "quantity": 249, "limit_price": 100.1, "stop_price": 99.3, "r_per_share": 0.7, "status": "entry submitted"}
    trade = rig.state.book.get("AAA")
    assert trade.status is TradeStatus.PENDING and trade.catalyst == "earnings"
    assert rig.state.setups["AAA"].state is SetupState.IN_TRADE
    order = rig.strategy.get_order(trade.entry_order_id)
    assert order.order_type is OrderType.LIMIT and order.limit_price == D("100.10")
    assert "AAA" in rig.state.decided


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rig: rig.state.setups.__setitem__("AAA", Setup(symbol="AAA", state=SetupState.PULLBACK)), "no triggered setup"),
        (lambda rig: setattr(rig.state, "flattened", True), "flattened"),
        (lambda rig: rig.clock.advance(-20 * 60), "only allowed between"),
    ],
)
def test_enter_long_refusals(tmp_path: Path, change, message: str) -> None:
    rig = Rig(tmp_path)
    change(rig)
    assert message in rig.desk.enter_long("AAA", "earnings", "x")["error"]


def test_enter_long_refuses_an_unknown_catalyst_and_a_full_book(tmp_path: Path) -> None:
    assert "catalyst" in Rig(tmp_path).desk.enter_long("AAA", "rumour", "x")["error"]
    full = Rig(tmp_path, params=dataclasses.replace(VwapPullbackParameters(), max_positions=0))
    assert "slot" in full.desk.enter_long("AAA", "earnings", "x")["error"]


def test_an_entry_fill_places_the_protective_stop(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    assert trade.status is TradeStatus.OPEN and trade.quantity == D(249) and trade.entry_price == D("100")
    stop = rig.strategy.get_order(trade.stop_order_id)
    assert stop.order_type is OrderType.STOP and stop.stop_price == D("99.30") and stop.quantity == D(249) and stop.side is OrderSide.SELL


def test_a_stop_fill_closes_the_trade_and_writes_the_trade_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    stop = rig.strategy.get_order(rig.state.book.get("AAA").stop_order_id)
    rig.advance(120)  # the 10:03 bar trades through 99.30
    rig.fill_hook(stop)
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.DONE
    row = json.loads(rig.log.read_text().splitlines()[0])
    assert row["symbol"] == "AAA" and row["exit_reason"] == "stop" and row["realised_pnl"] == str(D("-0.70") * 249)


def test_reconcile_expires_an_unfilled_entry_from_an_earlier_tick(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.desk.enter_long("AAA", "earnings", "x")
    entry = rig.strategy.get_order(rig.state.book.get("AAA").entry_order_id)
    rig.clock.advance(5)  # still the same bar: nothing filled
    rig.desk.reconcile(rig.clock.now())
    assert entry.is_canceled()
    rig.desk.on_order_canceled(entry)
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.PULLBACK


def test_reconcile_drops_an_entry_the_broker_rejected(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.desk.enter_long("AAA", "earnings", "x")
    entry = rig.strategy.get_order(rig.state.book.get("AAA").entry_order_id)
    entry.set_error("insufficient cash")
    rig.broker.tracker.process_trade_event(entry, OrderEvent.ERROR)
    rig.desk.reconcile(rig.clock.now())
    assert rig.state.book.get("AAA") is None
    assert rig.state.setups["AAA"].state is SetupState.PULLBACK
    assert rig.desk.free_slots() == 4


def test_reconcile_replaces_a_stop_that_errored(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    rig.broker.cancel_order(old)  # drop it from the broker's queue ...
    old.set_error("rejected")  # ... and make it look rejected, with no hook reaching the strategy
    rig.desk.reconcile(rig.clock.now())
    assert trade.stop_order_id != old.identifier
    assert rig.strategy.get_order(trade.stop_order_id).is_active()


def test_unexpected_stop_cancel_places_it_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    old = rig.strategy.get_order(trade.stop_order_id)
    rig.broker.cancel_order(old)
    rig.desk.on_order_canceled(old)
    assert trade.stop_order_id != old.identifier and rig.strategy.get_order(trade.stop_order_id).is_active()


def test_flatten_all_cancels_the_stop_and_sells_and_later_entry_fills_are_sold(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    trade = rig.state.book.get("AAA")
    stop = rig.strategy.get_order(trade.stop_order_id)
    rig.desk.flatten_all("end of day")
    assert stop.is_canceled()
    sell = rig.strategy.get_order(trade.exit_order_ids[-1])
    assert sell.order_type is OrderType.MARKET and sell.quantity == D(249)
    assert trade.exit_reason == "end of day" and rig.state.flattened


def test_close_unknown_positions_leaves_foreign_positions_alone(tmp_path: Path) -> None:
    # AAA: held, with a stop of ours from before a restart, but no trade in the session -> closed.
    # MSFT: held with no order of ours (a shared account) -> untouched.
    rig = Rig(tmp_path)
    for symbol in ("AAA", "MSFT"):
        rig.broker.submit_order(Order(strategy_name="vwap", asset=Asset(symbol), side=OrderSide.BUY, quantity=D(10)))
    rig.advance(60)
    ours = rig.strategy.submit_order(rig.strategy.create_order("AAA", 10, "sell", stop_price=90))  # a stop from before the restart
    rig.desk.close_unknown_positions()
    assert ours.is_canceled()
    pending = [o for o in rig.broker.tracker.get_active_orders() if o.side is OrderSide.SELL]
    assert [(o.asset.symbol, o.quantity) for o in pending] == [("AAA", D(10))]
    assert rig.state.unknown_positions_checked


def test_entry_due_and_exit_review_due(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.desk.entry_due()
    rig.open_trade()
    assert not rig.desk.entry_due()  # the only triggered setup is now in a trade
    assert rig.desk.exit_review_due(rig.clock.now()) == []  # 100 is above VWAP 99.9 and the EMA, below +1R
    rig.clock.advance(timedelta(minutes=15).total_seconds())
    assert rig.desk.exit_review_due(rig.clock.now()) == ["AAA"]
    rig.desk.mark_reviewed(rig.clock.now())
    assert rig.desk.exit_review_due(rig.clock.now()) == []
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_desk_entries.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.desk`

- [ ] **Step 3: Implement `desk.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/desk.py`:

```python
"""Every order the vwap_pullback strategy places (spec §5): entries, protective stops, exit hand-offs, the flatten.

The only module of the package that submits, cancels or modifies orders. The agents reach it through
`tools.py`, the strategy's order hooks through `on_order_filled`/`on_order_canceled`. Agent-facing methods
return `{"error": ...}` instead of raising; hook paths log and never raise. A position is never left
without a stop: a stop that cannot be placed is replaced by an immediate market sell.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.vwap_pullback import risk
from trading_agent_framework.strategies.vwap_pullback.features import Levels, latest_levels
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.prompts import CATALYSTS
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState, back_to_pullback, mark_done, mark_in_trade
from trading_agent_framework.strategies.vwap_pullback.trades import Trade, TradeStatus, exit_review_due, trade_flags
from trading_agent_framework.utils.errors import BacktestError, BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.vwap_pullback.session import SessionState

STOPPED_OUT = "already_stopped_out"
_DATA_ERRORS = (BrokerError, BacktestError)


class Desk:
    def __init__(self, strategy: Strategy, params: VwapPullbackParameters, *, trade_log: Callable[[], Path | None] = lambda: None) -> None:
        self._strategy = strategy
        self._params = params
        self._trade_log = trade_log
        self._expected_cancels: set[str] = set()  # stops this desk cancelled itself (hand-offs, flatten)

    @property
    def state(self) -> SessionState:
        return self._strategy.vars.session

    # --- views ---------------------------------------------------------------------

    def levels(self, symbol: str) -> Levels | None:
        return latest_levels(self.state.contexts.get(symbol, []), ema_length=self._params.ema_length, atr_length=self._params.atr_length)

    def last_close(self, symbol: str) -> Decimal | None:
        levels = self.levels(symbol)
        return None if levels is None else Decimal(str(levels.close))

    def planned_risk(self, symbol: str) -> tuple[Decimal, Decimal] | None:
        """`(stop, R)` an entry in `symbol` would get now; None before a pullback exists."""
        setup, info = self.state.setups.get(symbol), self.state.candidates.get(symbol)
        if setup is None or info is None or setup.pullback_low is None:
            return None
        reference = setup.trigger_close if setup.trigger_close is not None else setup.last_close
        if reference is None:
            return None
        stop = risk.planned_stop(setup.pullback_low, info.daily_atr, self._params)
        return stop, risk.to_price(reference, ROUND_HALF_UP) - stop

    def minutes_to_flatten(self, now: datetime) -> int:
        flatten_at = self.state.session.close - timedelta(minutes=self._strategy.minutes_before_closing)
        return max(0, int((flatten_at - now).total_seconds() // 60))

    def session_pnl(self) -> Decimal:
        prices = {t.symbol: p for t in self.state.book.open_trades() if (p := self.last_close(t.symbol)) is not None}
        return self.state.book.session_pnl(prices)

    def breaker_tripped(self) -> bool:
        return risk.circuit_breaker_tripped(self.session_pnl(), self.state.session_open_equity, self._params)

    def free_slots(self) -> int:
        book = self.state.book
        return risk.free_slots(len(book.open_trades()), len(book.pending()), self._params)

    def entry_due(self) -> bool:
        triggered = any(setup.state is SetupState.TRIGGERED for setup in self.state.setups.values())
        now = self._strategy.get_datetime()
        return triggered and not self.state.flattened and risk.in_entry_window(now, self._params) and self.free_slots() > 0 and not self.breaker_tripped()

    def _flags(self, trade: Trade) -> frozenset[str] | None:
        levels = self.levels(trade.symbol)
        if levels is None:
            return None
        return trade_flags(trade, last_close=Decimal(str(levels.close)), vwap=levels.vwap, ema=levels.ema)

    def exit_review_due(self, now: datetime) -> list[str]:
        due = []
        for trade in self.state.book.open_trades():
            flags = self._flags(trade)
            if flags is not None and exit_review_due(trade, flags, now=now, has_new_headline=trade.symbol in self.state.new_headline, params=self._params):
                due.append(trade.symbol)
        return due

    def mark_reviewed(self, now: datetime) -> None:
        """After an exit-agent run: every open trade was reviewed now, with the flags that hold now."""
        for trade in self.state.book.open_trades():
            trade.last_review_at = now
            trade.review_flags = self._flags(trade) or frozenset()
            self.state.new_headline.discard(trade.symbol)

    # --- entries -------------------------------------------------------------------

    def enter_long(self, symbol: str, catalyst: str, reason: str) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        state = self.state
        setup = state.setups.get(symbol)
        if setup is None or setup.state is not SetupState.TRIGGERED:
            current = setup.state.value if setup is not None else "untracked"
            return {"error": f"{symbol} has no triggered setup right now (state: {current}); only triggered setups can be entered"}
        if catalyst not in CATALYSTS:
            return {"error": f"catalyst must be one of {', '.join(CATALYSTS)}"}
        if state.flattened:
            return {"error": "the session is already flattened; no more entries today"}
        now = self._strategy.get_datetime()
        if not risk.in_entry_window(now, self._params):
            return {"error": f"entries are only allowed between {self._params.no_entry_before:%H:%M} and {self._params.no_entry_after:%H:%M}"}
        if self.breaker_tripped():
            return {"error": "the daily loss limit is reached; no more entries today"}
        if self.free_slots() <= 0:
            return {"error": "no free position slot"}
        if self._has_exposure(symbol):
            return {"error": f"{symbol} already has a position or an open order"}
        info = state.candidates[symbol]
        try:
            last = self._strategy.get_last_price(symbol)
            account = self._strategy.broker.get_account()
        except _DATA_ERRORS as exc:
            return {"error": f"price or account unavailable: {exc}"}
        if last is None or setup.trigger_close is None or setup.pullback_low is None:
            return {"error": f"no price for {symbol}"}
        try:
            plan = risk.plan_entry(
                trigger_close=setup.trigger_close,
                pullback_low=setup.pullback_low,
                last_price=last,
                daily_atr=info.daily_atr,
                equity=account.portfolio_value,
                buying_power=account.buying_power,
                cash=account.cash,
                pending_sell_proceeds=self._pending_sell_proceeds(),
                params=self._params,
            )
        except risk.EntryRefused as exc:
            return {"error": str(exc)}
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(symbol, plan.quantity, "buy", limit_price=plan.limit_price))
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            return {"error": str(exc)}
        state.book.add(
            Trade(
                symbol=symbol,
                entry_order_id=submitted.identifier,
                planned_quantity=plan.quantity,
                stop_price=plan.stop_price,
                r_per_share=plan.r_per_share,
                catalyst=catalyst,
                reason=reason,
                entered_at=now,
            )
        )
        state.setups[symbol] = mark_in_trade(setup)
        state.decided.add(symbol)
        self._strategy.log_info(f"entry {symbol}: {plan.quantity} at limit {plan.limit_price}, stop {plan.stop_price}, R {plan.r_per_share} ({catalyst}: {reason})")
        return {
            "symbol": symbol,
            "quantity": int(plan.quantity),
            "limit_price": float(plan.limit_price),
            "stop_price": float(plan.stop_price),
            "r_per_share": float(plan.r_per_share),
            "status": "entry submitted",
        }

    def pass_on_setup(self, symbol: str, reason: str) -> dict[str, Any]:
        symbol = symbol.strip().upper()
        self.state.decided.add(symbol)
        self._strategy.log_info(f"pass {symbol}: {reason}")
        return {"symbol": symbol, "status": "passed"}

    def _has_exposure(self, symbol: str) -> bool:
        if self.state.book.get(symbol) is not None:
            return True
        if any(order.asset.symbol == symbol for order in self._strategy.broker.tracker.get_active_orders()):
            return True
        try:
            return self._strategy.get_position(symbol) is not None
        except _DATA_ERRORS:
            return True  # unknown: refuse rather than double up

    def _pending_sell_proceeds(self) -> Decimal:
        """What working market/limit sells should bring in (stops and trails only sell if triggered)."""
        total = Decimal(0)
        for order in self._strategy.broker.tracker.get_active_orders():
            if order.side is OrderSide.SELL and order.order_type in (OrderType.MARKET, OrderType.LIMIT) and order.quantity is not None:
                price = order.limit_price or self.last_close(order.asset.symbol)
                if price is not None:
                    total += order.quantity * price
        return total

    # --- order events ----------------------------------------------------------------

    def on_order_filled(self, order: Order, price: Decimal, quantity: Decimal) -> None:
        if getattr(self._strategy.vars, "session", None) is None:
            return
        trade = self.state.book.by_order_id(order.identifier)
        if trade is None:
            return
        filled = order.filled_quantity if order.filled_quantity > 0 else quantity
        fill_price = order.avg_fill_price if order.avg_fill_price is not None else price
        if order.identifier == trade.entry_order_id:
            trade.record_entry_fill(filled, fill_price)
            self._protect(trade)
            return
        trade.record_exit_fill(filled, fill_price, self._strategy.get_datetime())
        if order.identifier == trade.stop_order_id:
            trade.stop_order_id = None
            trade.exit_reason = trade.exit_reason or ("trailing stop" if trade.stop_kind == "trail" else "stop")
        if trade.status is TradeStatus.CLOSED:
            self._archive(trade)

    def on_order_canceled(self, order: Order) -> None:
        if getattr(self._strategy.vars, "session", None) is None:
            return
        if order.identifier in self._expected_cancels:
            self._expected_cancels.discard(order.identifier)
            return
        trade = self.state.book.by_order_id(order.identifier)
        if trade is None:
            return
        if order.identifier == trade.entry_order_id:
            self._settle_entry(trade, order)
        elif order.identifier == trade.stop_order_id:
            self._strategy.log_warning(f"the stop for {trade.symbol} was cancelled outside the strategy; placing it again")
            trade.stop_order_id = None
            self._submit_stop(trade, trade.quantity)

    def _settle_entry(self, trade: Trade, order: Order) -> None:
        """An entry that ended without a full fill: keep and protect what filled, or drop the trade."""
        if order.filled_quantity > 0:
            trade.record_entry_fill(order.filled_quantity, order.avg_fill_price or trade.stop_price)
            self._protect(trade)
            return
        self.state.book.discard(trade.symbol)
        setup = self.state.setups.get(trade.symbol)
        if setup is not None and setup.state is SetupState.IN_TRADE:
            self.state.setups[trade.symbol] = back_to_pullback(setup)
        self._strategy.log_info(f"entry {trade.symbol} ended unfilled ({order.status.value}); setup back to pullback")

    def _protect(self, trade: Trade) -> None:
        if self.state.flattened:  # a late entry fill after the flatten: never carry it
            self._market_sell(trade, trade.quantity, "filled after the end-of-day flatten")
            return
        if trade.stop_order_id is None:
            self._submit_stop(trade, trade.quantity)

    def reconcile(self, now: datetime) -> None:
        """Start-of-tick housekeeping: expire entries from earlier ticks, drop entries the broker rejected, re-place a stop that is gone."""
        for trade in list(self.state.book.active()):
            if trade.status is TradeStatus.PENDING:
                order = self._strategy.get_order(trade.entry_order_id)
                if order is None or (not order.is_active() and not order.is_filled()):
                    if order is None:
                        self.state.book.discard(trade.symbol)
                    else:
                        self._settle_entry(trade, order)
                elif order.is_active() and trade.entered_at < now:
                    self._cancel(order)
            elif trade.status is TradeStatus.OPEN and not self._has_working_stop(trade) and not self._exit_pending(trade):
                self._strategy.log_warning(f"{trade.symbol} has no working stop; placing it again")
                trade.stop_order_id = None
                self._submit_stop(trade, trade.quantity)

    def _has_working_stop(self, trade: Trade) -> bool:
        if trade.stop_order_id is None:
            return False
        order = self._strategy.get_order(trade.stop_order_id)
        return order is not None and (order.is_active() or order.is_filled())

    def _exit_pending(self, trade: Trade) -> bool:
        orders = [self._strategy.get_order(i) for i in trade.exit_order_ids]
        return any(o is not None and o.is_active() for o in orders)

    # --- order helpers ---------------------------------------------------------------

    def _open_trade(self, symbol: str) -> Trade | dict[str, Any]:
        trade = self.state.book.get(symbol.strip().upper())
        if trade is None or trade.status is not TradeStatus.OPEN:
            return {"error": f"no open trade in {symbol.strip().upper()}"}
        return trade

    def _submit_stop(self, trade: Trade, quantity: Decimal) -> None:
        """Place the trade's stop (plain or trailing) for `quantity`; if that fails, sell `quantity` now."""
        if quantity <= 0:
            return
        if trade.stop_kind == "trail" and trade.trail_price is not None:
            order = self._strategy.create_order(trade.symbol, quantity, "sell", trail_price=trade.trail_price)
        else:
            order = self._strategy.create_order(trade.symbol, quantity, "sell", stop_price=trade.stop_level)
        try:
            submitted = self._strategy.submit_order(order)
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"stop for {trade.symbol} could not be placed ({exc}); selling {quantity} now")
            self._market_sell(trade, quantity, "protective stop failed")
            return
        trade.stop_order_id = submitted.identifier
        trade.stop_kind = trade.stop_kind or "stop"

    def _market_sell(self, trade: Trade, quantity: Decimal, reason: str) -> Order | None:
        try:
            submitted = self._strategy.submit_order(self._strategy.create_order(trade.symbol, quantity, "sell"))
        except Exception as exc:  # a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)
            self._strategy.log_error(f"market sell of {quantity} {trade.symbol} failed ({reason}): {exc}")
            return None
        trade.exit_order_ids.append(submitted.identifier)
        return submitted

    def _cancel(self, order: Order) -> None:
        try:
            self._strategy.cancel_order(order)
        except BrokerError as exc:
            self._strategy.log_warning(f"cancel of {order.identifier} ({order.asset.symbol}) failed: {exc}")

    def _release_stop(self, trade: Trade) -> str | None:
        """Cancel the trade's working stop and wait for it: None once released, `STOPPED_OUT`, or an error message.

        Alpaca and `BacktestBroker` refuse a sell above held minus pending sells, so the stop must go before
        any other exit sell. If the stop filled while the cancel was on its way, the trade is already out.
        """
        if trade.stop_order_id is None:
            return None
        order = self._strategy.get_order(trade.stop_order_id)
        if order is None:
            trade.stop_order_id = None
            return None
        if order.is_filled():
            return STOPPED_OUT
        if order.is_active():
            self._expected_cancels.add(order.identifier)
            try:
                self._strategy.cancel_order(order)
            except BrokerError as exc:
                self._expected_cancels.discard(order.identifier)
                return f"could not cancel the stop: {exc}"
            self._strategy.wait_for_order_execution(order, timeout=self._params.cancel_wait_seconds)
            if order.is_filled():
                return STOPPED_OUT
            if order.is_active():
                return "the stop cancel was not confirmed in time; nothing else was changed"
        trade.stop_order_id = None
        return None

    def _archive(self, trade: Trade) -> None:
        self.state.book.archive(trade)
        setup = self.state.setups.get(trade.symbol)
        if setup is not None:
            self.state.setups[trade.symbol] = mark_done(setup)
        self._strategy.log_info(f"trade {trade.symbol} closed ({trade.exit_reason}): P&L {trade.realised_pnl}")
        path = self._trade_log()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trade.to_json()) + "\n")

    # --- session boundaries --------------------------------------------------------------

    def flatten_all(self, reason: str) -> None:
        """Cancel every entry and stop of this session's trades and market-sell what they hold (only this strategy's trades)."""
        state = getattr(self._strategy.vars, "session", None)
        if state is None:
            return
        state.flattened = True
        sells: list[Order] = []
        for trade in list(state.book.active()):
            if trade.status is TradeStatus.PENDING:
                order = self._strategy.get_order(trade.entry_order_id)
                if order is not None and order.is_active():
                    self._cancel(order)
                continue
            released = self._release_stop(trade)
            if released == STOPPED_OUT:
                continue
            if released is not None:
                self._strategy.log_error(f"flatten {trade.symbol}: {released}")
                continue
            trade.exit_reason = reason
            sell = self._market_sell(trade, trade.quantity, reason)
            if sell is not None:
                sells.append(sell)
        if sells and not self._strategy.is_backtesting:
            self._strategy.wait_for_orders_execution(sells, timeout=self._params.flatten_wait_seconds)
            still_open = [order.asset.symbol for order in sells if order.is_active()]
            if still_open:
                self._strategy.log_error(f"flatten: sells still open after {self._params.flatten_wait_seconds:.0f}s for {', '.join(still_open)}")

    def close_unknown_positions(self) -> None:
        """After a restart: close a position that this strategy's own open orders point at but no trade of the session knows.

        A position without an order of ours is not ours to touch: the account may be shared.
        """
        state = self.state
        state.unknown_positions_checked = True
        known = {trade.symbol for trade in state.book.active()}
        ours: dict[str, list[Order]] = {}
        for order in self._strategy.broker.tracker.get_active_orders():
            ours.setdefault(order.asset.symbol, []).append(order)
        try:
            positions = self._strategy.get_positions()
        except _DATA_ERRORS as exc:
            self._strategy.log_warning(f"could not check for positions left from before a restart: {exc}")
            return
        for position in positions:
            symbol = position.asset.symbol
            if symbol in known or symbol not in ours:
                continue
            for order in ours[symbol]:
                self._cancel(order)
            if not self._strategy.is_backtesting:
                self._strategy.wait_for_orders_execution(ours[symbol], timeout=self._params.cancel_wait_seconds)
            self._strategy.log_warning(f"closing {position.quantity} {symbol}: held with this strategy's open orders but no trade in this session (restart)")
            try:
                self._strategy.close_position(symbol)
            except BrokerError as exc:
                self._strategy.log_error(f"could not close {symbol}: {exc}")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_desk_entries.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (15 tests). Notes if one fails:
- `test_enter_long_sizes...`: stop 99.5 − 0.2 = 99.30, R 0.70, limit 100 + 0.1 = 100.10, sizes: risk 714, 25% cap 249, cash 949.
- `test_a_stop_fill_closes...`: the STOP fills at `min(open 100, 99.30)` on the 10:03 bar = 99.30, so P&L is −0.70 × 249. `str(D("-0.70") * 249)` is `"-174.30"`.
- `test_reconcile_expires...`: `entered_at` is 10:00:00 and the reconcile runs at 10:00:05 (`entered_at < now`), so the entry is cancelled.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/desk.py tests/strategies/vwap_pullback/test_vwap_desk_entries.py
git commit -m "Task 8: vwap_pullback desk: entries, protective stops, reconcile, flatten, restart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `Desk` exit hand-offs

**Files:**
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/desk.py` (add methods)
- Test: `tests/strategies/vwap_pullback/test_vwap_desk_exits.py`

**Interfaces:**
- Consumes: Task 8's `Desk` internals
- Produces: `Desk.take_partial_profit(symbol, fraction) -> dict`, `Desk.tighten_stop(symbol, stop_price) -> dict`, `Desk.replace_stop_with_trailing(symbol, trail_atr) -> dict`, `Desk.exit_position(symbol, reason) -> dict`, `Desk.hold(symbol, reason) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_desk_exits.py`:

```python
from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig
from trading_agent_framework.entities.enums import OrderType


def _open(tmp_path: Path) -> Rig:
    rig = Rig(tmp_path)
    rig.open_trade()
    return rig


def test_take_partial_profit_sells_the_fraction_and_resizes_the_stop(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    old_stop = rig.strategy.get_order(trade.stop_order_id)
    result = rig.desk.take_partial_profit("AAA", 0.5)
    assert result == {"status": "partial profit taken", "sold": 124, "remaining": 125, "stop_price": 99.3}
    assert old_stop.is_canceled()
    sell = rig.strategy.get_order(trade.exit_order_ids[-1])
    assert sell.order_type is OrderType.MARKET and sell.quantity == D(124)
    new_stop = rig.strategy.get_order(trade.stop_order_id)
    assert new_stop.quantity == D(125) and new_stop.stop_price == D("99.30")
    assert trade.tp1_done
    assert "already" in rig.desk.take_partial_profit("AAA", 0.5)["error"]
    assert "between" in rig.desk.take_partial_profit("AAA", 0.9)["error"]


def test_tighten_stop_only_moves_up(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk.tighten_stop("AAA", 99.8) == {"status": "stop raised", "stop_price": 99.8}
    assert trade.stop_level == D("99.80")
    assert rig.strategy.get_order(trade.stop_order_id).stop_price == D("99.80")
    assert "only move up" in rig.desk.tighten_stop("AAA", 99.5)["error"]
    assert "below the last price" in rig.desk.tighten_stop("AAA", 100.5)["error"]


def test_replace_stop_with_trailing(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    old_stop = rig.strategy.get_order(trade.stop_order_id)
    result = rig.desk.replace_stop_with_trailing("AAA", 1.0)  # 5-minute ATR 0.4 -> trail 0.40, starting at 99.60
    assert result == {"status": "trailing stop placed", "trail_price": 0.4, "starts_at": 99.6}
    assert old_stop.is_canceled()
    trail = rig.strategy.get_order(trade.stop_order_id)
    assert trail.order_type is OrderType.TRAIL and trail.trail_price == D("0.40") and trail.quantity == D(249)
    assert trade.stop_kind == "trail"
    assert "already a trailing stop" in rig.desk.tighten_stop("AAA", 99.9)["error"]


def test_a_trail_starting_below_the_current_stop_is_refused(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    rig.desk.tighten_stop("AAA", 99.8)
    assert "below the current stop" in rig.desk.replace_stop_with_trailing("AAA", 1.0)["error"]  # would start at 99.60


def test_exit_position_releases_the_stop_and_sells_everything(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    assert rig.desk.exit_position("AAA", "downgrade headline") == {"status": "exit submitted", "quantity": 249}
    assert trade.stop_order_id is None and trade.exit_reason == "downgrade headline"
    assert rig.strategy.get_order(trade.exit_order_ids[-1]).quantity == D(249)


def test_a_stop_that_already_filled_is_reported_not_sold_twice(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    trade = rig.state.book.get("AAA")
    rig.advance(120)  # the stop fills in the broker; the hook has not reached the desk yet
    assert rig.desk.exit_position("AAA", "x") == {"status": "already_stopped_out"}
    assert trade.exit_order_ids == []


def test_hold_and_unknown_symbols(tmp_path: Path) -> None:
    rig = _open(tmp_path)
    assert rig.desk.hold("AAA", "inside 1R") == {"symbol": "AAA", "status": "holding"}
    assert "no open trade" in rig.desk.exit_position("ZZZ", "x")["error"]
```

The `Rig` from Task 8 builds contexts with a single bar of range 0.4 (high 100.2, low 99.8), so the 5-minute ATR is 0.4.

`pythonpath = ["."]` makes `tests.strategies.vwap_pullback.test_vwap_desk_entries` importable (namespace package), so `Rig` is shared by importing it. If that import fails in this repo's pytest setup, move `Rig` into `tests/fakes.py` as `VwapRig` and import it from there in every vwap test file instead.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_desk_exits.py -v`
Expected: FAIL with `AttributeError: 'Desk' object has no attribute 'take_partial_profit'`

- [ ] **Step 3: Implement the exit hand-offs**

In `src/trading_agent_framework/strategies/vwap_pullback/desk.py`, change the decimal import to:

```python
from decimal import ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP, Decimal
```

Add a section before `# --- session boundaries`:

```python
# --- exits (the exit agent's actions) --------------------------------------------------


def take_partial_profit(self, symbol: str, fraction: float) -> dict[str, Any]:
    trade = self._open_trade(symbol)
    if isinstance(trade, dict):
        return trade
    low, high = self._params.tp1_fraction_band
    if not low <= fraction <= high:
        return {"error": f"fraction must be between {low} and {high}"}
    if trade.tp1_done:
        return {"error": "partial profit was already taken on this trade"}
    sold = (trade.quantity * Decimal(str(fraction))).to_integral_value(rounding=ROUND_FLOOR)
    if sold <= 0 or sold >= trade.quantity:
        return {"error": f"a position of {trade.quantity} shares is too small to split"}
    released = self._release_stop(trade)
    if released is not None:
        return {"status": STOPPED_OUT} if released == STOPPED_OUT else {"error": released}
    remaining = trade.quantity - sold
    if self._market_sell(trade, sold, "partial profit") is None:
        self._submit_stop(trade, trade.quantity)
        return {"error": "the sell failed; the stop was placed again for the whole position"}
    trade.tp1_done = True
    self._submit_stop(trade, remaining)
    self._strategy.log_info(f"partial profit {trade.symbol}: sold {sold}, {remaining} left under the stop")
    return {"status": "partial profit taken", "sold": int(sold), "remaining": int(remaining), "stop_price": float(trade.stop_level)}


def tighten_stop(self, symbol: str, stop_price: float) -> dict[str, Any]:
    trade = self._open_trade(symbol)
    if isinstance(trade, dict):
        return trade
    if trade.stop_kind == "trail":
        return {"error": "the stop is already a trailing stop; it ratchets up on its own"}
    new_level = risk.to_price(stop_price, ROUND_DOWN)
    if new_level <= trade.stop_level:
        return {"error": f"a stop can only move up (current stop {trade.stop_level})"}
    try:
        last = self._strategy.get_last_price(trade.symbol)
    except _DATA_ERRORS as exc:
        return {"error": f"price unavailable: {exc}"}
    if last is not None and new_level >= last:
        return {"error": f"the stop must stay below the last price {last}"}
    order = self._strategy.get_order(trade.stop_order_id) if trade.stop_order_id else None
    if order is None or not order.is_active():
        return {"error": "no working stop to raise"}
    try:
        replacement = self._strategy.modify_order(order, stop_price=new_level)
    except Exception as exc:  # the broker's modify may raise its own error type (BacktestError, BrokerError)
        return {"error": f"the stop could not be modified: {exc}"}
    trade.stop_order_id = replacement.identifier
    trade.stop_level = new_level
    self._strategy.log_info(f"stop {trade.symbol} raised to {new_level}")
    return {"status": "stop raised", "stop_price": float(new_level)}


def replace_stop_with_trailing(self, symbol: str, trail_atr: float) -> dict[str, Any]:
    trade = self._open_trade(symbol)
    if isinstance(trade, dict):
        return trade
    low, high = self._params.trail_atr_band
    if not low <= trail_atr <= high:
        return {"error": f"trail_atr must be between {low} and {high}"}
    levels = self.levels(trade.symbol)
    if levels is None or levels.atr is None or not levels.atr > 0:
        return {"error": "no 5-minute ATR yet for this symbol"}
    trail = risk.to_price(trail_atr * levels.atr, ROUND_HALF_UP)
    try:
        last = self._strategy.get_last_price(trade.symbol)
    except _DATA_ERRORS as exc:
        return {"error": f"price unavailable: {exc}"}
    if last is None or trail <= 0:
        return {"error": "no price to start the trail from"}
    starts_at = last - trail
    if starts_at < trade.stop_level:
        return {"error": f"a {trail_atr} ATR trail would start at {starts_at}, below the current stop {trade.stop_level}; use a tighter trail"}
    released = self._release_stop(trade)
    if released is not None:
        return {"status": STOPPED_OUT} if released == STOPPED_OUT else {"error": released}
    trade.stop_kind, trade.trail_price, trade.stop_level = "trail", trail, starts_at
    self._submit_stop(trade, trade.quantity)
    self._strategy.log_info(f"stop {trade.symbol} replaced by a {trail} trailing stop")
    return {"status": "trailing stop placed", "trail_price": float(trail), "starts_at": float(starts_at)}


def exit_position(self, symbol: str, reason: str) -> dict[str, Any]:
    trade = self._open_trade(symbol)
    if isinstance(trade, dict):
        return trade
    released = self._release_stop(trade)
    if released is not None:
        return {"status": STOPPED_OUT} if released == STOPPED_OUT else {"error": released}
    if self._market_sell(trade, trade.quantity, reason) is None:
        self._submit_stop(trade, trade.quantity)
        return {"error": "the sell failed; the stop was placed again"}
    trade.exit_reason = reason
    self._strategy.log_info(f"exit {trade.symbol}: {reason}")
    return {"status": "exit submitted", "quantity": int(trade.quantity)}


def hold(self, symbol: str, reason: str) -> dict[str, Any]:
    trade = self._open_trade(symbol)
    if isinstance(trade, dict):
        return trade
    self._strategy.log_info(f"hold {trade.symbol}: {reason}")
    return {"symbol": trade.symbol, "status": "holding"}
```

Note on `replace_stop_with_trailing`: the trailing sell is placed by `_submit_stop` (which reads `stop_kind == "trail"` and `trail_price`), so a failed submission falls back to an immediate market sell like every other stop.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/ -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (Task 8 + 7 new tests). If `test_a_stop_that_already_filled...` fails because the backtest stop did not fill: the rig's clock is at 10:01 after `open_trade`; `advance(120)` walks the 10:02 and 10:03 bars, and the 10:03 bar's low (99.0) is under 99.30.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/desk.py tests/strategies/vwap_pullback/test_vwap_desk_exits.py
git commit -m "Task 9: vwap_pullback desk exit hand-offs (partial profit, tighten, trail, exit, hold)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Agent tools and row views

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/tools.py`
- Test: `tests/strategies/vwap_pullback/test_vwap_tools.py`

**Interfaces:**
- Consumes: `Desk` (Tasks 8–9), `agents.tools.news.news_tools`, `memory.tools.current_run_id`, `setups.health`
- Produces: `setup_rows(desk) -> list[dict]`, `trade_rows(desk, now) -> list[dict]`, `budgeted_search_news(strategy, calls_per_run) -> Callable`, `entry_tools(strategy, desk) -> list[Callable]` (`get_setups`, `get_intraday_bars`, `search_news`, `enter_long`, `pass_on_setup`), `exit_tools(strategy, desk) -> list[Callable]` (`get_open_trades`, `get_intraday_bars`, `search_news`, `take_partial_profit`, `tighten_stop`, `replace_stop_with_trailing`, `exit_position`, `hold`)

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_tools.py`:

```python
from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeNewsProvider
from tests.strategies.vwap_pullback.test_vwap_desk_entries import Rig
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.strategies.vwap_pullback.tools import budgeted_search_news, entry_tools, exit_tools, setup_rows, trade_rows


def _tools(tools: list) -> dict:
    return {tool.__name__: tool for tool in tools}


def test_setup_rows_show_pullback_and_triggered_setups_with_plan_and_headlines(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.state.setups["WATCHED"] = Setup(symbol="WATCHED")
    rig.state.headlines["AAA"] = [{"headline": "AAA beats", "created_at": "2026-09-01T07:00:00-04:00", "source": "b"}]
    rows = setup_rows(rig.desk)
    assert [row["symbol"] for row in rows] == ["AAA"]
    row = rows[0]
    assert row["state"] == "triggered" and row["planned_stop"] == 99.3 and row["r_per_share"] == 0.7
    assert row["z_rs"] == 2.5 and row["headlines"][0]["headline"] == "AAA beats"


def test_trade_rows_show_the_stop_open_r_levels_and_new_headlines(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    rig.state.headlines["AAA"] = [
        {"headline": "before entry", "created_at": "2026-09-01T09:00:00-04:00", "source": "b"},
        {"headline": "after entry", "created_at": "2026-09-01T10:00:30-04:00", "source": "b"},
    ]
    row = trade_rows(rig.desk, rig.clock.now())[0]
    assert row["symbol"] == "AAA" and row["quantity"] == 249 and row["stop_kind"] == "stop" and row["stop_level"] == 99.3
    assert row["unrealised_r"] == 0.0 and row["vwap"] == 99.9 and row["minutes_to_flatten"] == 349
    assert [h["headline"] for h in row["headlines_since_entry"]] == ["after entry"]


def test_search_news_has_a_per_run_budget(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.broker._news_source = FakeNewsProvider()
    search = budgeted_search_news(rig.strategy, 2)
    with agent_call_context(run_id="run-1"):
        assert "error" not in search(symbols="AAA")
        assert "error" not in search(symbols="BBB")
        assert "budget" in search(symbols="CCC")["error"]
    with agent_call_context(run_id="run-2"):
        assert "error" not in search(symbols="AAA")


def test_entry_tools_delegate_to_the_desk(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    tools = _tools(entry_tools(rig.strategy, rig.desk))
    assert set(tools) == {"get_setups", "get_intraday_bars", "search_news", "enter_long", "pass_on_setup"}
    assert tools["get_setups"]()["setups"][0]["symbol"] == "AAA"
    bars = tools["get_intraday_bars"]("AAA", 100)
    assert len(bars["bars"]) == 1 and bars["bars"][0]["close"] == 100.0
    assert "error" in tools["get_intraday_bars"]("ZZZ")
    assert tools["enter_long"]("AAA", "earnings", "clean")["status"] == "entry submitted"
    assert rig.state.setups["AAA"].state is SetupState.IN_TRADE


def test_exit_tools_delegate_to_the_desk(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.open_trade()
    tools = _tools(exit_tools(rig.strategy, rig.desk))
    assert set(tools) == {"get_open_trades", "get_intraday_bars", "search_news", "take_partial_profit", "tighten_stop", "replace_stop_with_trailing", "exit_position", "hold"}
    assert tools["get_open_trades"]()["trades"][0]["symbol"] == "AAA"
    assert tools["hold"]("AAA", "noise")["status"] == "holding"
    for tool in tools.values():
        assert tool.__doc__ and "\n" not in tool.__doc__.strip()
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.tools`

- [ ] **Step 3: Implement `tools.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/tools.py` (no `from __future__ import annotations`):

```python
"""The two agents' tools (spec §4): lean views over the session, and actions that all go through `Desk`.

No `from __future__ import annotations` on purpose, like `memory/tools.py`: LangChain builds each tool's
schema from the real annotations. Docstrings are one line: each is sent to the model on every call.
Neither agent gets a raw order tool: sizes, prices and stops are the desk's.
"""

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.memory.tools import current_run_id
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState, health

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy
    from trading_agent_framework.strategies.vwap_pullback.desk import Desk

Catalyst = Literal["earnings", "guidance", "analyst", "contract_or_product", "sector_or_macro", "none"]
MAX_BARS = 24


def _after(created_at: str, moment: datetime) -> bool:
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")) > moment
    except ValueError:
        return False


def setup_rows(desk: "Desk") -> list[dict[str, Any]]:  # noqa: UP037
    """Pullback and triggered setups: health, stage-2 z-scores, the stop and R an entry would get, headlines."""
    state = desk.state
    rows: list[dict[str, Any]] = []
    for symbol, setup in sorted(state.setups.items()):
        if setup.state not in (SetupState.PULLBACK, SetupState.TRIGGERED):
            continue
        row: dict[str, Any] = {"symbol": symbol, **health(setup)}
        info = state.candidates.get(symbol)
        if info is not None:
            row |= {"z_rs": round(info.z_rs, 2), "z_rvol": round(info.z_rvol, 2)}
        planned = desk.planned_risk(symbol)
        if planned is not None:
            row |= {"planned_stop": float(planned[0]), "r_per_share": float(planned[1])}
        row["headlines"] = state.headlines.get(symbol, [])
        rows.append(row)
    return rows


def trade_rows(desk: "Desk", now: datetime) -> list[dict[str, Any]]:  # noqa: UP037
    """Open trades: size, stop, open profit in R, VWAP, 9-EMA, time left, and headlines since entry."""
    rows: list[dict[str, Any]] = []
    for trade in desk.state.book.open_trades():
        levels = desk.levels(trade.symbol)
        last = Decimal(str(levels.close)) if levels is not None else None
        rows.append(
            {
                "symbol": trade.symbol,
                "quantity": int(trade.quantity),
                "entry_price": float(trade.entry_price) if trade.entry_price is not None else None,
                "last_close": float(last) if last is not None else None,
                "stop_kind": trade.stop_kind,
                "stop_level": float(trade.stop_level),
                "r_per_share": float(trade.r_per_share),
                "unrealised_r": trade.unrealised_r(last) if last is not None else None,
                "tp1_done": trade.tp1_done,
                "vwap": round(levels.vwap, 2) if levels is not None else None,
                "ema9": round(levels.ema, 2) if levels is not None and levels.ema is not None else None,
                "minutes_to_flatten": desk.minutes_to_flatten(now),
                "headlines_since_entry": [h for h in desk.state.headlines.get(trade.symbol, []) if _after(h["created_at"], trade.entered_at)],
            }
        )
    return rows


def budgeted_search_news(strategy: "Strategy", calls_per_run: int) -> Callable[..., dict[str, Any]]:  # noqa: UP037
    """The shared `search_news` tool, refused after `calls_per_run` calls in one agent run."""
    inner = news_tools(strategy)[0]
    usage: dict[str, Any] = {"run_id": None, "count": 0}

    def search_news(symbols: str = "", start: str | None = None, end: str | None = None, limit: int = 10, include_content: bool = False) -> dict[str, Any]:
        """Search recent news headlines and summaries, optionally filtered to symbols."""
        run_id = current_run_id()
        if run_id != usage["run_id"]:
            usage["run_id"], usage["count"] = run_id, 0
        if usage["count"] >= calls_per_run:
            return {"error": "news budget for this run is spent; decide with what you have"}
        usage["count"] += 1
        return inner(symbols=symbols, start=start, end=end, limit=limit, include_content=include_content)

    return search_news


def _bars_tool(desk: "Desk") -> Callable[..., dict[str, Any]]:  # noqa: UP037
    def get_intraday_bars(symbol: str, length: int = 12) -> dict[str, Any]:
        """Get a tracked symbol's recent 5-minute bars with VWAP, oldest first."""
        contexts = desk.state.contexts.get(symbol.strip().upper())
        if not contexts:
            return {"error": f"no intraday bars for {symbol.strip().upper()}"}
        count = min(max(int(length), 1), MAX_BARS)
        return {
            "symbol": symbol.strip().upper(),
            "bars": [
                {"time": c.time.isoformat(), "open": round(c.open, 2), "high": round(c.high, 2), "low": round(c.low, 2), "close": round(c.close, 2), "volume": int(c.volume), "vwap": round(c.vwap, 2)}
                for c in contexts[-count:]
            ],
        }

    return get_intraday_bars


def entry_tools(strategy: "Strategy", desk: "Desk") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """The entry agent's tools."""

    def get_setups() -> dict[str, Any]:
        """List the pullback and triggered setups with their health, planned stop and headlines."""
        return {"setups": setup_rows(desk)}

    def enter_long(symbol: str, catalyst: Catalyst, reason: str) -> dict[str, Any]:
        """Enter a triggered setup; the code sizes the position and places the stop."""
        return desk.enter_long(symbol, catalyst, reason)

    def pass_on_setup(symbol: str, reason: str) -> dict[str, Any]:
        """Record that you pass on a triggered setup."""
        return desk.pass_on_setup(symbol, reason)

    return [get_setups, _bars_tool(desk), budgeted_search_news(strategy, desk.params.news_calls_per_run), enter_long, pass_on_setup]


def exit_tools(strategy: "Strategy", desk: "Desk") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """The exit agent's tools."""

    def get_open_trades() -> dict[str, Any]:
        """List the open trades with their stop, open profit in R, VWAP, 9-EMA and new headlines."""
        return {"trades": trade_rows(desk, strategy.get_datetime())}

    def take_partial_profit(symbol: str, fraction: float) -> dict[str, Any]:
        """Sell part (0.25 to 0.5) of an open trade once; the stop is resized to the rest."""
        return desk.take_partial_profit(symbol, fraction)

    def tighten_stop(symbol: str, stop_price: float) -> dict[str, Any]:
        """Raise an open trade's stop price (a stop only moves up)."""
        return desk.tighten_stop(symbol, stop_price)

    def replace_stop_with_trailing(symbol: str, trail_atr: float) -> dict[str, Any]:
        """Replace an open trade's stop with a trailing stop of trail_atr 5-minute ATRs."""
        return desk.replace_stop_with_trailing(symbol, trail_atr)

    def exit_position(symbol: str, reason: str) -> dict[str, Any]:
        """Sell the whole remaining position now."""
        return desk.exit_position(symbol, reason)

    def hold(symbol: str, reason: str) -> dict[str, Any]:
        """Keep an open trade unchanged this review."""
        return desk.hold(symbol, reason)

    return [
        get_open_trades,
        _bars_tool(desk),
        budgeted_search_news(strategy, desk.params.news_calls_per_run),
        take_partial_profit,
        tighten_stop,
        replace_stop_with_trailing,
        exit_position,
        hold,
    ]
```

In `desk.py`, add a read-only property so tools can read the parameters:

```python
    @property
    def params(self) -> VwapPullbackParameters:
        return self._params
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/ -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS. (`minutes_to_flatten` at 10:01 is 349: `Rig` sets `minutes_before_closing = 10`, so the flatten is at 15:50.)

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/tools.py src/trading_agent_framework/strategies/vwap_pullback/desk.py tests/strategies/vwap_pullback/test_vwap_tools.py
git commit -m "Task 10: vwap_pullback agent tools and row views

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The per-tick graph

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/graph.py`
- Modify: `pyproject.toml`, `uv.lock` (explicit `langgraph` dependency)
- Test: `tests/strategies/vwap_pullback/test_vwap_graph.py`

**Interfaces:**
- Produces: `TickState` (`TypedDict`, total=False: `now: datetime`, `exit_due: list[str]`, `entry_due: bool`, `runs: Annotated[list[dict[str, Any]], operator.add]`); `CLASSIFY_NODE = "classify"`, `EXIT_NODE = "exit_agent"`, `ENTRY_NODE = "entry_agent"`; `route_after_classify(state) -> str`, `route_after_exit(state) -> str`; `build_tick_graph(*, classify, run_exit, run_entry) -> compiled graph` (`.invoke({"now": ...})`)

- [ ] **Step 1: Add the dependency**

Run: `uv add "langgraph>=1.2,<2"`
Expected: `pyproject.toml` lists `langgraph` and `uv.lock` updates; nothing new is downloaded (it is already installed).

- [ ] **Step 2: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_graph.py`:

```python
from __future__ import annotations

import pytest

from tests.fakes import et
from trading_agent_framework.strategies.vwap_pullback.graph import build_tick_graph, route_after_classify, route_after_exit
from trading_agent_framework.utils.errors import FatalStrategyError

NOW = et(2026, 9, 1, 10, 0)


def test_routing() -> None:
    assert route_after_classify({"exit_due": ["AAA"], "entry_due": True}) == "exit_agent"
    assert route_after_classify({"exit_due": [], "entry_due": True}) == "entry_agent"
    assert route_after_classify({}) == "__end__"
    assert route_after_exit({"entry_due": True}) == "entry_agent"
    assert route_after_exit({"entry_due": False}) == "__end__"


def _graph(calls: list[str], *, classify_result: dict, exit_result: dict | None = None):
    def classify(state):
        calls.append("classify")
        assert state["now"] == NOW
        return classify_result

    def run_exit(state):
        calls.append("exit")
        return exit_result or {"runs": [{"agent": "exit"}]}

    def run_entry(state):
        calls.append("entry")
        return {"runs": [{"agent": "entry"}]}

    return build_tick_graph(classify=classify, run_exit=run_exit, run_entry=run_entry)


def test_a_quiet_tick_runs_only_the_classifier() -> None:
    calls: list[str] = []
    _graph(calls, classify_result={"exit_due": [], "entry_due": False}).invoke({"now": NOW})
    assert calls == ["classify"]


def test_exit_runs_before_entry_and_entry_follows_the_exit_nodes_update() -> None:
    calls: list[str] = []
    graph = _graph(calls, classify_result={"exit_due": ["AAA"], "entry_due": False}, exit_result={"runs": [{"agent": "exit"}], "entry_due": True})
    result = graph.invoke({"now": NOW})
    assert calls == ["classify", "exit", "entry"]
    assert result["runs"] == [{"agent": "exit"}, {"agent": "entry"}]


def test_entry_alone() -> None:
    calls: list[str] = []
    _graph(calls, classify_result={"exit_due": [], "entry_due": True}).invoke({"now": NOW})
    assert calls == ["classify", "entry"]


def test_a_fatal_error_in_a_node_propagates() -> None:
    def classify(state):
        raise FatalStrategyError("abort")

    graph = build_tick_graph(classify=classify, run_exit=lambda s: {}, run_entry=lambda s: {})
    with pytest.raises(FatalStrategyError, match="abort"):
        graph.invoke({"now": NOW})
```

- [ ] **Step 3: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: ... vwap_pullback.graph`

- [ ] **Step 4: Implement `graph.py`**

Create `src/trading_agent_framework/strategies/vwap_pullback/graph.py` (no `from __future__ import annotations`: LangGraph reads `TickState`'s annotations at runtime):

```python
"""The per-tick LangGraph (spec §4): classify, then the exit agent, then the entry agent, each only when due.

`build_tick_graph` takes the three node callables, so the routing is tested with fakes and the strategy
injects its real nodes. No checkpointer (what lasts between ticks lives in `strategy.vars`) and no retry
policy (re-running an agent node could place an order twice). `langgraph` is imported inside
`build_tick_graph`: `main.py` imports every strategy, and one that never builds this graph must not load it.
"""

import operator
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, TypedDict

CLASSIFY_NODE = "classify"
EXIT_NODE = "exit_agent"
ENTRY_NODE = "entry_agent"
_END = "__end__"  # langgraph.graph.END


class TickState(TypedDict, total=False):
    now: datetime
    exit_due: list[str]
    entry_due: bool
    runs: Annotated[list[dict[str, Any]], operator.add]


Node = Callable[[TickState], dict[str, Any]]


def route_after_classify(state: TickState) -> str:
    if state.get("exit_due"):
        return EXIT_NODE
    if state.get("entry_due"):
        return ENTRY_NODE
    return _END


def route_after_exit(state: TickState) -> str:
    return ENTRY_NODE if state.get("entry_due") else _END


def build_tick_graph(*, classify: Node, run_exit: Node, run_entry: Node) -> Any:
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(TickState)
    builder.add_node(CLASSIFY_NODE, classify)
    builder.add_node(EXIT_NODE, run_exit)
    builder.add_node(ENTRY_NODE, run_entry)
    builder.add_edge(START, CLASSIFY_NODE)
    builder.add_conditional_edges(CLASSIFY_NODE, route_after_classify, {EXIT_NODE: EXIT_NODE, ENTRY_NODE: ENTRY_NODE, _END: END})
    builder.add_conditional_edges(EXIT_NODE, route_after_exit, {ENTRY_NODE: ENTRY_NODE, _END: END})
    builder.add_edge(ENTRY_NODE, END)
    return builder.compile()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_graph.py -v && uv run ruff check src/trading_agent_framework/strategies/vwap_pullback`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/graph.py tests/strategies/vwap_pullback/test_vwap_graph.py pyproject.toml uv.lock
git commit -m "Task 11: vwap_pullback per-tick LangGraph (classify -> exit agent -> entry agent)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: `VwapPullbackStrategy`, registration and docs

**Files:**
- Create: `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py`
- Modify: `src/trading_agent_framework/strategies/vwap_pullback/__init__.py`
- Modify: `src/trading_agent_framework/main.py`
- Modify: `CLAUDE.md`
- Test: `tests/strategies/vwap_pullback/test_vwap_strategy.py`

**Interfaces:**
- Consumes: everything above; `Strategy`, `AgentRunResult`, `AgentError`, `FatalStrategyError`, `AlpacaBacktestData`, `warmup_calendar_days`
- Produces: `VwapPullbackStrategy(broker, *, mode, universe, settings=None, chat_model=None, **kwargs)` with `ENTRY_AGENT = "vwap_entry"`, `EXIT_AGENT = "vwap_exit"`, `sleeptime = "5M"`, `minutes_before_closing = 10`, `initialize()`, `on_trading_iteration()`, `before_market_opens()`, `before_market_closes()`, `on_filled_order(...)`, `on_canceled_order(order)`, `run_backtesting()`; `main.AGENT_STRATEGIES["vwap_pullback_continuation"]`

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/vwap_pullback/test_vwap_strategy.py`:

```python
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et, make_session
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.utils.errors import AgentError, FatalStrategyError

DAY = date(2026, 9, 1)


def _strategy(tmp_path: Path, *, mode: TradingMode = TradingMode.PAPER, now=et(2026, 9, 1, 10, 0)) -> VwapPullbackStrategy:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path)
    strategy.vars.session = None
    strategy.vars.consecutive_agent_errors = 0
    strategy._build_components()
    return strategy


def _session() -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"))


class _Handle:
    def __init__(self, outcome: AgentRunResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, dict]] = []

    def run(self, task_prompt: str, *, context=None, run_id=None, force_tool=None) -> AgentRunResult:
        self.calls.append((task_prompt, context))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Agents(dict):
    pass


def test_initialize_creates_both_agents_and_the_graph(tmp_path: Path) -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 8, 0), [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, universe=["AAA"], project_root=tmp_path, chat_model=FakeToolCallingChatModel(messages=iter([])))
    strategy.initialize()
    assert "vwap_entry" in strategy.agents and "vwap_exit" in strategy.agents
    assert strategy._graph is not None


def test_a_session_is_prepared_once_per_day_and_the_graph_runs_each_tick(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    prepared: list[date] = []
    strategy.scanner.prepare_session = lambda: prepared.append(DAY) or _session()
    invoked: list[object] = []
    strategy._graph = type("G", (), {"invoke": lambda self, state: invoked.append(state["now"])})()
    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    assert prepared == [DAY]
    assert len(invoked) == 2
    assert strategy.vars.session.unknown_positions_checked


def test_order_hooks_and_the_close_reach_the_desk(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    seen: list[str] = []
    strategy.desk.on_order_filled = lambda order, price, quantity: seen.append("filled")
    strategy.desk.on_order_canceled = lambda order: seen.append("canceled")
    strategy.desk.flatten_all = lambda reason: seen.append(reason)
    strategy.on_filled_order(None, object(), Decimal(1), Decimal(1), 1)
    strategy.on_canceled_order(object())
    strategy.before_market_closes()
    assert seen == ["filled", "canceled", "end-of-day flatten"]


def test_the_entry_node_runs_the_entry_agent_with_the_setups(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    strategy.vars.session.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED)
    handle = _Handle(AgentRunResult(output="passed", tool_calls=[ToolCallRecord(name="pass_on_setup", args={"symbol": "AAA"}, result="{}")]))
    strategy._agents = _Agents({"vwap_entry": handle})
    update = strategy._entry_node({"now": et(2026, 9, 1, 10, 0)})
    assert update["runs"] == [{"agent": "vwap_entry", "ok": True, "tool_calls": 1}]
    task, context = handle.calls[0]
    assert context["setups"][0]["symbol"] == "AAA" and "free_slots" in context


def test_three_agent_failures_in_a_row_abort_a_backtest(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.vars.session = _session()
    strategy._agents = _Agents({"vwap_entry": _Handle(AgentError("model down"))})
    strategy._entry_node({"now": et(2026, 9, 1, 10, 0)})
    strategy._entry_node({"now": et(2026, 9, 1, 10, 5)})
    with pytest.raises(FatalStrategyError, match="3 runs in a row"):
        strategy._entry_node({"now": et(2026, 9, 1, 10, 10)})


def test_main_registers_the_strategy() -> None:
    from trading_agent_framework.main import AGENT_STRATEGIES

    assert "vwap_pullback_continuation" in AGENT_STRATEGIES
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/strategies/vwap_pullback/test_vwap_strategy.py -v`
Expected: FAIL with `ImportError: cannot import name 'VwapPullbackStrategy'`

- [ ] **Step 3: Implement the strategy**

Create `src/trading_agent_framework/strategies/vwap_pullback/agent_vwap_pullback.py`:

```python
"""VwapPullbackStrategy: an intraday VWAP pullback continuation strategy with an entry agent and an exit agent.

Every tick (5 minutes) runs one LangGraph pass (`graph.py`): the Python classifier (`Scanner` + `Desk.reconcile`)
always, then the exit agent only when an open trade had an event, then the entry agent only when a setup
triggered and a slot is free. Code owns the safety net (`Desk`): sizing, the protective stop on every fill,
the loss limit, the entry window and the 15:50 flatten. See docs/superpowers/specs/2026-09-29-vwap-pullback-continuation-design.md.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.vwap_pullback.desk import Desk
from trading_agent_framework.strategies.vwap_pullback.graph import TickState, build_tick_graph
from trading_agent_framework.strategies.vwap_pullback.parameters import VwapPullbackParameters
from trading_agent_framework.strategies.vwap_pullback.prompts import build_entry_prompt, build_exit_prompt
from trading_agent_framework.strategies.vwap_pullback.scanner import Scanner
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import SetupState
from trading_agent_framework.strategies.vwap_pullback.tools import entry_tools, exit_tools, setup_rows, trade_rows
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, BrokerError, FatalStrategyError

# A persistent LLM misconfiguration would otherwise give a flat "successful" backtest (same rule as news_binary).
MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS = 3


class VwapPullbackStrategy(Strategy):
    ENTRY_AGENT = "vwap_entry"
    EXIT_AGENT = "vwap_exit"
    ENTRY_TASK = "Review the triggered pullback setups and enter or pass on each one. The current datetime and the setups are in the context below."
    EXIT_TASK = "Review the open trades and manage each one. The current datetime and the trades are in the context below."

    sleeptime = "5M"
    minutes_before_closing = 10  # before_market_closes, and so the flatten, runs at 15:50

    parameters = {
        "backtesting_start": datetime(2026, 8, 3, tzinfo=MARKET_TZ),
        "backtesting_end": datetime(2026, 8, 31, tzinfo=MARKET_TZ),
        "benchmark_symbol": "SPY",
        "warmup_trading_days": 75,  # 70 daily bars for stage 1, plus the RVOL baseline sessions
        "budget": 10000,
    }

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        universe: Sequence[str],
        settings: VwapPullbackParameters | None = None,
        chat_model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(broker, mode=mode, **kwargs)
        self.universe = list(universe)
        self.settings = settings or VwapPullbackParameters()
        self._chat_model = chat_model  # None: the model named by LLM_MODEL (tests inject a fake)
        self.desk: Desk | None = None
        self.scanner: Scanner | None = None
        self._graph: Any = None

    # --- lifecycle -------------------------------------------------------------------

    def initialize(self) -> None:
        self.vars.session = None
        self.vars.consecutive_agent_errors = 0
        self._build_components()
        assert self.desk is not None
        flatten_time = (datetime(2000, 1, 1, 16, 0) - timedelta(minutes=self.minutes_before_closing)).strftime("%H:%M")
        self.agents.create(name=self.ENTRY_AGENT, system_prompt=build_entry_prompt(self.settings), model=self._chat_model, tools=entry_tools(self, self.desk))
        self.agents.create(name=self.EXIT_AGENT, system_prompt=build_exit_prompt(self.settings, flatten_time=flatten_time), model=self._chat_model, tools=exit_tools(self, self.desk))
        self._graph = build_tick_graph(classify=self._classify_node, run_exit=self._exit_node, run_entry=self._entry_node)
        self.log_info(f"VwapPullbackStrategy initialized: {len(self.universe)} symbols, sleeptime {self.sleeptime}")

    def _build_components(self) -> None:
        self.desk = Desk(self, self.settings, trade_log=self._trade_log_path)
        self.scanner = Scanner(self, self.settings, self.universe, benchmark=self.parameters["benchmark_symbol"], preload=self._preload if self.is_backtesting else None)

    def before_market_opens(self) -> None:
        self._ensure_session()

    def on_trading_iteration(self) -> None:
        if not self.is_backtesting and self.settings.live_bar_delay_seconds > 0:
            self.sleep(self.settings.live_bar_delay_seconds)  # let the last minute bar be published
        state = self._ensure_session()
        if state is None or state.flattened:
            return
        assert self.desk is not None
        if not state.unknown_positions_checked:
            self.desk.close_unknown_positions()
        self._graph.invoke({"now": self.get_datetime()})

    def before_market_closes(self) -> None:
        if self.desk is not None:
            self.desk.flatten_all("end-of-day flatten")

    def on_filled_order(self, position: Position | None, order: Order, price: Decimal, quantity: Decimal, multiplier: int) -> None:
        if self.desk is not None:
            self.desk.on_order_filled(order, price, quantity)

    def on_canceled_order(self, order: Order) -> None:
        if self.desk is not None:
            self.desk.on_order_canceled(order)

    def _ensure_session(self) -> SessionState | None:
        """This session's state, prepared on first need: `before_market_opens` does not run when a live run starts mid-session."""
        today = self.get_datetime().astimezone(MARKET_TZ).date()
        state = self.vars.session
        if state is None or state.day != today:
            assert self.scanner is not None
            try:
                self.vars.session = self.scanner.prepare_session()
            except (BrokerError, BacktestError) as exc:
                self.log_error(f"session preparation failed, skipping this tick: {exc}")
                self.vars.session = None
        return self.vars.session

    # --- graph nodes -------------------------------------------------------------------

    def _classify_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None and self.scanner is not None
        session = self.vars.session
        now = state["now"]
        self.desk.reconcile(now)
        self.scanner.scan(session)
        session.decided.clear()
        return {"exit_due": self.desk.exit_review_due(now), "entry_due": self.desk.entry_due()}

    def _exit_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None
        now = state["now"]
        context = {"current_datetime": now.isoformat(), "open_trades": trade_rows(self.desk, now)}
        summary = self._run_agent(self.EXIT_AGENT, self.EXIT_TASK, context)
        self.desk.mark_reviewed(now)
        return {"runs": [summary], "entry_due": self.desk.entry_due()}

    def _entry_node(self, state: TickState) -> dict[str, Any]:
        assert self.desk is not None
        now = state["now"]
        context = {"current_datetime": now.isoformat(), "setups": setup_rows(self.desk), "free_slots": self.desk.free_slots()}
        summary = self._run_agent(self.ENTRY_AGENT, self.ENTRY_TASK, context)
        session = self.vars.session
        undecided = [s for s, setup in session.setups.items() if setup.state is SetupState.TRIGGERED and s not in session.decided]
        if undecided:
            self.log_warning(f"[{self.ENTRY_AGENT}] no enter_long or pass_on_setup for {', '.join(undecided)}; treated as a pass")
        return {"runs": [summary]}

    def _run_agent(self, name: str, task: str, context: dict[str, Any]) -> dict[str, Any]:
        """Run one agent; an `AgentError` is logged and counted, and aborts a backtest after 3 in a row."""
        try:
            result: AgentRunResult = self.agents[name].run(task, context=context, run_id=uuid.uuid4().hex)
        except AgentError as exc:
            self.vars.consecutive_agent_errors += 1
            self.log_error(f"[{name}] run failed: {exc}")
            if self.is_backtesting and self.vars.consecutive_agent_errors >= MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS:
                raise FatalStrategyError(f"[{name}] failed {self.vars.consecutive_agent_errors} runs in a row, aborting the backtest; last error: {exc}") from exc
            return {"agent": name, "ok": False, "tool_calls": 0}
        self.vars.consecutive_agent_errors = 0
        self.log_info(f"[{name}] output:\n{result.output}")
        for i, call in enumerate(result.tool_calls):
            self.log_info(f"[{name}] tool_call_{i}: {call}")
        return {"agent": name, "ok": True, "tool_calls": len(result.tool_calls)}

    # --- backtesting ---------------------------------------------------------------------

    def _trade_log_path(self) -> Path | None:
        if self.run_id is None:
            return None
        return self.project_root / "logs" / self.name / self.trading_mode.value / self.run_id / "trades.jsonl"

    def _preload(self, assets: Sequence[Asset], timestep: str) -> None:
        from trading_agent_framework.backtesting.broker import BacktestBroker
        from trading_agent_framework.backtesting.warmup import warmup_calendar_days

        if not isinstance(self.broker, BacktestBroker):
            return
        start = self.parameters["backtesting_start"] - timedelta(days=warmup_calendar_days(self.parameters["warmup_trading_days"]))
        self.broker.preload_bars(assets, start, self.parameters["backtesting_end"], timestep)

    def run_backtesting(self):
        # class parameters: the same window the data source is built with, so preload_bars matches it
        return super().run_backtesting(
            data_source=AlpacaBacktestData,  # minute bars with enough history (Yahoo keeps ~30 days of minutes)
            timestep="minute",
            start=self.parameters["backtesting_start"],
            end=self.parameters["backtesting_end"],
            benchmark=self.parameters["benchmark_symbol"],
            budget=Decimal(str(self.parameters["budget"])),
            warmup_trading_days=self.parameters["warmup_trading_days"],
            agent_telemetry=True,
        )
```

Replace `src/trading_agent_framework/strategies/vwap_pullback/__init__.py` with:

```python
"""Intraday VWAP pullback continuation: two LangGraph-orchestrated agents over a Python setup scanner."""

from trading_agent_framework.strategies.vwap_pullback.agent_vwap_pullback import VwapPullbackStrategy

__all__ = ["VwapPullbackStrategy"]
```

In `src/trading_agent_framework/main.py`, add the import:

```python
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
```

add the builder after `_build_news_binary`:

```python
def _build_vwap_pullback(broker: Broker, mode: TradingMode) -> Strategy | None:
    universe = load_cross_momentum_universe()
    if not universe:
        Console().print("Universe file not found — run `uv run batch-universe` before executing this strategy.", style="bold red")
        return None
    return VwapPullbackStrategy(broker=broker, mode=mode, universe=universe)
```

and register it:

```python
AGENT_STRATEGIES: dict[str, StrategyBuilder] = {
    "cross_momentum": _build_cross_momentum,
    "news_binary": _build_news_binary,
    "vwap_pullback_continuation": _build_vwap_pullback,
}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/strategies/vwap_pullback/ -v && uv run pytest -q && uv run ruff check`
Expected: all pass. If `test_a_session_is_prepared_once_per_day...` fails on `close_unknown_positions` with `FakeBroker` (no tracker orders, no positions), it must simply mark the flag; check `get_positions()` on `FakeBroker` returns `[]`.

- [ ] **Step 5: Document the strategy in `CLAUDE.md`**

In `CLAUDE.md`, in the `strategies/` bullet of **Architecture**, after the `news_builtin/` sentence, add:

```markdown
`vwap_pullback/` (`VwapPullbackStrategy`, registered as `"vwap_pullback_continuation"`): strictly intraday, 5-minute ticks. Pure `features`/`screening`/`setups`/`risk`/`trades`; `Scanner` (stage 1 before the open from the cross_momentum universe file, stage 2 and the setup state machine each tick), `Desk` (the only order code: code-sized entries, a protective stop on every fill, exit hand-offs, 15:50 flatten), `tools.py` (no raw order tools for the agents), `graph.py` (per-tick LangGraph: classify → exit agent → entry agent, agents only when due). Closed trades go to `trades.jsonl` in the run directory.
```

In **Key patterns / gotchas**, add:

```markdown
- **Backtests simulate trailing stops** (`fills.evaluate_trailing_stop`): `BacktestBroker` keeps each `TRAIL` order's high/low-water mark in `_PendingOrder.trail_reference`, seeded with the latest close at submission, and tests the level from the previous bars *before* a bar's own high may raise it (ties against the trader). `evaluate_fill` still refuses `TRAIL`.
- **vwap_pullback never leaves a position without a stop**: `Desk` cancels a stop and waits for the cancel before any other exit sell (both brokers refuse a sell above held minus pending sells), replaces a stop it cannot place with an immediate market sell, and `reconcile` re-places a stop that errored or was cancelled from outside. It flattens and restart-closes only positions this strategy owns (a shared account's other positions are left alone).
```

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies/vwap_pullback/ src/trading_agent_framework/main.py CLAUDE.md tests/strategies/vwap_pullback/test_vwap_strategy.py
git commit -m "Task 12: VwapPullbackStrategy wiring, registration and docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Manual check (not automated, needs credentials)**

Create `env/.env.vwap_pullback_continuation.backtesting` (see README for the naming; do not commit it if `git check-ignore env/.env.vwap_pullback_continuation.backtesting` says it is ignored, and never commit credentials) with `ALPACA_DATA_*`, `ALPACA_NEWS_*`, `LLM_BASE_URL`, `LLM_MODEL`. Then run a one-week backtest by temporarily narrowing `backtesting_start`/`backtesting_end`:

Run: `uv run agent vwap_pullback_continuation backtesting`
Expected: stage 1 logs ~150 candidates per session; the entry agent runs only on ticks with a triggered setup; `logs/vwap_pullback_continuation/backtesting/<ts>_backtesting/trades.jsonl` gets one line per closed trade; no position is held after 15:50 on any day (check `equity.parquet` or the log's flatten lines).

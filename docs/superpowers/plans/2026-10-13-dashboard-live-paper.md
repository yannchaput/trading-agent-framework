# Dashboard Live and Paper tabs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add **Live** and **Paper** tabs before **Backtesting** that show one strategy's realized and unrealized P&L per trading day, rebuilt on demand from its Alpaca account.

**Architecture:** A pure module `dashboard/pnl.py` holds the average-cost replay (shared with the backtest's `load_session_pnl`) and the daily P&L computation. A Streamlit-free `dashboard/account_reader.py` finds the strategies' env files, reads their keys with `dotenv_values` (never into `os.environ`), and calls three read-only Alpaca endpoints (account activities, daily bars, positions) through narrow Protocols. One page module `_pages/account.py` renders both tabs: sidebar selectbox, metric cards, a grouped bar chart and the positions table.

**Tech Stack:** Python 3.14, Streamlit (`st.navigation`, `st.cache_data`, `AppTest`), Plotly, alpaca-py 0.44 (`TradingClient`, `StockHistoricalDataClient`), python-dotenv, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-13-dashboard-live-paper-design.md`

## Global Constraints

- Work on `main` in the shared checkout `/home/yann/projets/trading-agent-framework`; no branch, no worktree. Commit after each task; do not push.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Stage only the files the task names (`TODO.md` has an unrelated user edit: never stage it).
- The dashboard never writes: only Alpaca GET endpoints, nothing written to `logs/`, `env/` or anywhere else.
- Credentials: read with `dotenv.dotenv_values`, never `load_dotenv`; never shown, logged or put in a cache key.
- Only `BROKER=alpaca` env files are listed (missing `BROKER` means alpaca, like `BrokerSettings.from_env`).
- Floats throughout the dashboard (display-only, like `reader.py`); no `Decimal`.
- Days are `MARKET_TZ` (`trading_agent_framework.utils.clock.MARKET_TZ`, America/New_York) dates.
- Daily closes: `StockBarsRequest`, `TimeFrame.Day`, `feed=DataFeed.IEX`, `adjustment=Adjustment.RAW` (fills are raw prices; never `market_data.build_bars_request`, which uses `Adjustment.ALL`).
- Backtesting stays the default tab; tab order `Live, Paper, Backtesting, Models`.
- Dark theme: `components/charts.CHART_TEMPLATE`, never `plotly_white`. No `use_container_width` in new code (`width="stretch"`).
- Tests never touch the network; fakes are hand-written (no `MagicMock`).
- `uv run pytest` and `uv run ruff check` green at the end of every task.

## Deviations from the spec (found while planning)

- **Activities are parsed into our own dataclasses, not alpaca-py's `TradeActivity`/`NonTradeActivity`.** The Trading API's activity items carry no `account_id`, which those models require, and a `FEE`'s `date` is a date-time. `pnl.Fill` and `pnl.Fee` hold only the fields the computation needs.
- **The cache has a 5-minute `ttl`.** Reopening the tab later fetches again, which is what "fetch when the tab opens and on Refresh" means, while widget reruns within those 5 minutes reuse the fetch. The spec only said "Refresh clears it".
- **`AccountPnl.unmatched_sales`** (symbol -> shares sold with no held shares) is added next to `mismatches`: a sale of shares bought before the first fetched fill leaves the replayed and broker quantities equal (both 0), so the quantity comparison alone would miss it.

## Review Focus

1. **A fill in the evening UTC belongs to the US/Eastern day** (`2026-10-06T01:30Z` is Oct 5 in New York): Task 2 pins it.
2. **Today never uses a bar close**, even when a partial bar for today exists: Task 2 pins it (and Task 4's bar request ends at today's midnight).
3. **Exactly 100 activities** (one full page) must fetch a second, empty page and stop, not loop or drop items: Task 4 pins it.
4. **A cancelled FEE** (`status: "canceled"`) must not reduce realized P&L: Task 4 pins it.
5. **Each tab keeps its strategy when you switch to another tab and back.** Streamlit drops a widget's state on a run that doesn't draw it. Blank keys must show an error that names the variable without any key value. Task 6 pins both.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/trading_agent_framework/dashboard/pnl.py` | Pure: `Fill`, `Sale`, `AverageCostBook`, `Fee`, `BrokerPosition`, `AccountPnl`, `market_day`, `daily_account_pnl` |
| Modify `src/trading_agent_framework/dashboard/reader.py` | `load_session_pnl` uses `AverageCostBook` (same output) |
| Create `src/trading_agent_framework/dashboard/account_reader.py` | Env-file discovery, credentials, Alpaca fetches, `load_account_pnl`, `load_strategy_pnl` |
| Modify `src/trading_agent_framework/dashboard/components/charts.py` | `_signed_usd` -> `signed_usd`; new `account_pnl_chart` |
| Create `src/trading_agent_framework/dashboard/_pages/account.py` | `page_account(mode)`, `page_live`, `page_paper` |
| Modify `src/trading_agent_framework/dashboard/app.py` | `NAV_PAGES` order, default by title |
| Modify `src/trading_agent_framework/dashboard/CLAUDE.md`, `CLAUDE.md` | Docs |
| Create `tests/dashboard/test_pnl.py`, `tests/dashboard/test_account_reader.py`, `tests/dashboard/test_account_pnl_chart.py`, `tests/dashboard/test_account_page.py` | Tests |
| Modify `tests/dashboard/test_app_smoke.py` | Tab-order test |

---

### Task 1: Shared average-cost book, `load_session_pnl` on top of it

**Files:**
- Create: `src/trading_agent_framework/dashboard/pnl.py`
- Modify: `src/trading_agent_framework/dashboard/reader.py` (`load_session_pnl`, lines ~584-647)
- Test: `tests/dashboard/test_pnl.py` (new); `tests/dashboard/test_reader.py` (unchanged, must stay green)

**Interfaces:**
- Produces:
  - `pnl.Fill(time: datetime, symbol: str, side: str, qty: float, price: float, fee: float = 0.0)` (frozen dataclass; `side` "buy" means buy, anything else is a sell)
  - `pnl.Sale(symbol: str, qty: float, gross: float, fees: float, basis: float)` with property `pnl -> float` (`gross - fees - basis`)
  - `pnl.AverageCostBook()`: `apply(fill: Fill) -> Sale | None`, `holdings() -> dict[str, tuple[float, float]]` (symbol -> (shares, remaining cost)), attribute `unmatched: dict[str, float]`

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_pnl.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_agent_framework.dashboard.pnl import AverageCostBook, Fill

T0 = datetime(2026, 10, 5, 14, tzinfo=UTC)


def _fill(symbol: str, side: str, qty: float, price: float, fee: float = 0.0) -> Fill:
    return Fill(time=T0, symbol=symbol, side=side, qty=qty, price=price, fee=fee)


# --- AverageCostBook ------------------------------------------------------------------------------


def test_a_buy_is_held_at_its_cost_with_the_fee() -> None:
    book = AverageCostBook()

    assert book.apply(_fill("AAA", "buy", 10, 100, fee=1)) is None
    assert book.holdings() == {"AAA": (10, 1001)}


def test_a_sale_leaves_at_the_average_cost_of_several_buys_net_of_its_fee() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 10, 100))
    book.apply(_fill("AAA", "buy", 10, 120))

    sale = book.apply(_fill("AAA", "sell", 5, 130, fee=2))

    assert sale is not None
    assert (sale.symbol, sale.qty, sale.gross, sale.fees, sale.basis) == ("AAA", 5, 650, 2, 550)
    assert sale.pnl == pytest.approx(98)
    assert book.holdings() == {"AAA": (15, pytest.approx(1650))}


def test_selling_everything_empties_the_holding() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 10, 100))
    book.apply(_fill("AAA", "sell", 10, 90))

    assert book.holdings() == {}
    assert book.unmatched == {}


def test_a_sale_with_no_held_shares_is_unmatched_and_books_nothing() -> None:
    book = AverageCostBook()

    assert book.apply(_fill("BBB", "sell", 5, 50)) is None
    assert book.unmatched == {"BBB": 5}
    assert book.holdings() == {}


def test_selling_more_than_held_closes_the_held_shares_and_counts_the_rest_unmatched() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 4, 100))

    sale = book.apply(_fill("AAA", "sell", 6, 110, fee=3))

    assert sale is not None
    assert (sale.qty, sale.fees, sale.basis) == (4, pytest.approx(2), 400)
    assert book.unmatched == {"AAA": pytest.approx(2)}
    assert book.holdings() == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_pnl.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'trading_agent_framework.dashboard.pnl'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/trading_agent_framework/dashboard/pnl.py`:

```python
"""Average-cost P&L, pure: the backtest's per-session realized P&L and the Live/Paper tabs' daily P&L."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. `side` "buy" is a buy, anything else a sell; `fee` is paid on top, in dollars."""

    time: datetime
    symbol: str
    side: str
    qty: float
    price: float
    fee: float = 0.0


@dataclass(frozen=True, slots=True)
class Sale:
    """The part of a sell fill that closed held shares, against their average cost."""

    symbol: str
    qty: float
    gross: float  # qty * sell price
    fees: float  # the sell fee's share for these shares
    basis: float  # the sold shares' average cost, buy fees included

    @property
    def pnl(self) -> float:
        return self.gross - self.fees - self.basis


class AverageCostBook:
    """Replays time-ordered fills: each symbol's shares and remaining cost (every buy fee in it). A sale leaves at
    the average cost and pays its own fee; the shares sold beyond what is held are counted in `unmatched`."""

    def __init__(self) -> None:
        self._holdings: dict[str, tuple[float, float]] = {}
        self.unmatched: dict[str, float] = {}

    def apply(self, fill: Fill) -> Sale | None:
        shares, cost = self._holdings.get(fill.symbol, (0.0, 0.0))
        sale = None
        if fill.side.lower() == "buy":
            shares, cost = shares + fill.qty, cost + fill.qty * fill.price + fill.fee
        else:
            closed = min(fill.qty, shares) if shares > _EPS else 0.0
            if fill.qty - closed > _EPS:
                self.unmatched[fill.symbol] = self.unmatched.get(fill.symbol, 0.0) + fill.qty - closed
            if closed > 0:
                basis = cost * closed / shares
                sale = Sale(fill.symbol, closed, closed * fill.price, fill.fee * closed / fill.qty, basis)
                shares, cost = shares - closed, cost - basis
        if shares > _EPS:
            self._holdings[fill.symbol] = (shares, cost)
        else:
            self._holdings.pop(fill.symbol, None)
        return sale

    def holdings(self) -> dict[str, tuple[float, float]]:
        return dict(self._holdings)
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_pnl.py -v`
Expected: 5 passed.

- [ ] **Step 5: Refactor `load_session_pnl` onto the book**

In `src/trading_agent_framework/dashboard/reader.py` add the import next to the other dashboard imports:

```python
from trading_agent_framework.dashboard.pnl import AverageCostBook, Fill
```

Replace the body of `load_session_pnl` from `holdings: dict[str, list[float]] = {}` down to (not including) `if not sold:` with:

```python
    book = AverageCostBook()
    rows: list[dict[str, Any]] = []
    for day, day_fills in cast(Any, fills.groupby(fills["time"].dt.date, sort=True)):
        sold: dict[str, dict[str, float]] = {}  # symbol -> shares sold, gross proceeds, fees, cost basis
        for _, fill in day_fills.iterrows():
            sale = book.apply(
                Fill(
                    time=fill["time"],
                    symbol=str(fill["symbol"]),
                    side=str(fill["side"]),
                    qty=float(fill["filled_quantity"]),
                    price=float(fill["price"]),
                    fee=float(fill["trade_cost"]),
                )
            )
            if sale is None:
                continue
            entry = sold.setdefault(sale.symbol, {"qty": 0.0, "gross": 0.0, "fees": 0.0, "basis": 0.0})
            entry["qty"] += sale.qty
            entry["gross"] += sale.gross
            entry["fees"] += sale.fees
            entry["basis"] += sale.basis
```

Everything from `if not sold:` to `return rows` stays as it is. Update the docstring's first paragraph end to say the replay is `pnl.AverageCostBook`'s (one short clause, e.g. "... priced against the position's average cost (`pnl.AverageCostBook`) ...").

- [ ] **Step 6: Run the reader and pnl tests**

Run: `uv run pytest tests/dashboard/test_reader.py tests/dashboard/test_pnl.py tests/dashboard/test_app_smoke.py -q`
Expected: all pass, including the 9 `test_load_session_pnl_*` tests unchanged.

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all passed, `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/dashboard/pnl.py src/trading_agent_framework/dashboard/reader.py tests/dashboard/test_pnl.py
git commit -m "refactor(dashboard): share the average-cost replay of load_session_pnl

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Daily account P&L (pure)

**Files:**
- Modify: `src/trading_agent_framework/dashboard/pnl.py`
- Test: `tests/dashboard/test_pnl.py`

**Interfaces:**
- Consumes: `Fill`, `AverageCostBook` (Task 1).
- Produces:
  - `pnl.Fee(day: date, amount: float)` (signed like Alpaca's `net_amount`: a fee is negative)
  - `pnl.BrokerPosition(symbol: str, qty: float, avg_entry_price: float, current_price: float | None, market_value: float | None, unrealized_pl: float | None, unrealized_plpc: float | None)`
  - `pnl.AccountPnl(days: list[dict[str, Any]], realized_total: float, unrealized_now: float, positions: list[BrokerPosition], mismatches: list[tuple[str, float, float]], unmatched_sales: dict[str, float], unpriced: list[str])`; each day is `{"date": "YYYY-MM-DD", "realized": float, "unrealized": float}`, oldest first
  - `pnl.market_day(moment: datetime) -> date`
  - `pnl.daily_account_pnl(fills: Sequence[Fill], fees: Sequence[Fee], closes: Mapping[str, Mapping[date, float]], positions: Sequence[BrokerPosition], today: date) -> AccountPnl`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_pnl.py` (add `date` to the `datetime` import and `BrokerPosition, Fee, daily_account_pnl` to the `pnl` import):

```python
# --- daily_account_pnl --------------------------------------------------------------------------

MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)


def _at(day: date, hour_utc: int = 15) -> datetime:
    return datetime(day.year, day.month, day.day, hour_utc, tzinfo=UTC)


def _position(symbol: str, qty: float, unrealized_pl: float) -> BrokerPosition:
    return BrokerPosition(symbol=symbol, qty=qty, avg_entry_price=100.0, current_price=None, market_value=None, unrealized_pl=unrealized_pl, unrealized_plpc=None)


def test_realized_is_booked_on_the_sale_day_net_of_fills_fees_and_fee_activities() -> None:
    fills = [
        Fill(_at(MON), "AAA", "buy", 10, 100, fee=1),
        Fill(_at(TUE), "AAA", "sell", 10, 110, fee=2),
    ]
    fees = [Fee(TUE, -0.05)]
    closes = {"AAA": {MON: 105.0, TUE: 111.0}}

    result = daily_account_pnl(fills, fees, closes, [], WED)

    assert result.days == [
        {"date": "2026-10-05", "realized": 0.0, "unrealized": 49.0},  # 10 * 105 - 1,001
        {"date": "2026-10-06", "realized": 96.95, "unrealized": 0.0},  # 1,100 - 2 - 1,001 - 0.05
        {"date": "2026-10-07", "realized": 0.0, "unrealized": 0.0},
    ]
    assert result.realized_total == pytest.approx(96.95)


def test_today_unrealized_comes_from_the_positions_not_a_close() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100)]
    closes = {"AAA": {MON: 101.0, TUE: 999.0}}  # a partial bar for today must be ignored

    result = daily_account_pnl(fills, [], closes, [_position("AAA", 10, 42.0)], TUE)

    assert result.days[-1] == {"date": "2026-10-06", "realized": 0.0, "unrealized": 42.0}
    assert result.unrealized_now == 42.0


def test_a_fill_in_the_evening_utc_belongs_to_the_new_york_day() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(datetime(2026, 10, 6, 1, 30, tzinfo=UTC), "AAA", "sell", 10, 110)]

    result = daily_account_pnl(fills, [], {"AAA": {MON: 100.0}}, [], WED)

    assert result.days[0] == {"date": "2026-10-05", "realized": 100.0, "unrealized": 0.0}


def test_a_missing_close_is_carried_forward_and_a_symbol_never_priced_is_named() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(_at(MON), "ZZZ", "buy", 1, 50)]
    closes = {"AAA": {MON: 102.0}, "BBB": {TUE: 7.0}}  # AAA has no Tuesday bar; ZZZ has none at all

    result = daily_account_pnl(fills, [], closes, [_position("AAA", 10, 0.0), _position("ZZZ", 1, 0.0)], WED)

    assert result.days[1] == {"date": "2026-10-06", "realized": 0.0, "unrealized": 20.0}  # 10 * 102 - 1,000
    assert result.unpriced == ["ZZZ"]


def test_replayed_holdings_differing_from_the_broker_are_reported() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(_at(MON), "BBB", "sell", 5, 50)]

    result = daily_account_pnl(fills, [], {}, [_position("AAA", 15, 0.0), _position("CCC", 3, 0.0)], TUE)

    assert result.mismatches == [("AAA", 10, 15), ("CCC", 0.0, 3)]
    assert result.unmatched_sales == {"BBB": 5}


def test_positions_without_fills_give_no_days_but_today_unrealized() -> None:
    result = daily_account_pnl([], [], {}, [_position("AAA", 10, -12.5)], TUE)

    assert result.days == []
    assert result.realized_total == 0.0
    assert result.unrealized_now == -12.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_pnl.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'BrokerPosition'`.

- [ ] **Step 3: Write the minimal implementation**

In `pnl.py` change the imports to:

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from trading_agent_framework.utils.clock import MARKET_TZ
```

Append:

```python
@dataclass(frozen=True, slots=True)
class Fee:
    """A fee activity, signed like Alpaca's `net_amount` (a fee is negative)."""

    day: date
    amount: float


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """An open position as the broker reports it now."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float | None
    market_value: float | None
    unrealized_pl: float | None
    unrealized_plpc: float | None


@dataclass(frozen=True, slots=True)
class AccountPnl:
    """`days`: `{date, realized, unrealized}` per trading day from the first fill to today, oldest first.
    `mismatches`: `(symbol, replayed shares, broker shares)` where the fills do not explain the positions;
    `unmatched_sales`: shares sold with no held shares to cost them against; `unpriced`: held symbols with no close."""

    days: list[dict[str, Any]]
    realized_total: float
    unrealized_now: float
    positions: list[BrokerPosition]
    mismatches: list[tuple[str, float, float]]
    unmatched_sales: dict[str, float]
    unpriced: list[str]


def market_day(moment: datetime) -> date:
    return moment.astimezone(MARKET_TZ).date()


def _close_on_or_before(closes: Mapping[date, float], day: date) -> float | None:
    earlier = [d for d in closes if d <= day]
    return closes[max(earlier)] if earlier else None


def daily_account_pnl(
    fills: Sequence[Fill],
    fees: Sequence[Fee],
    closes: Mapping[str, Mapping[date, float]],
    positions: Sequence[BrokerPosition],
    today: date,
) -> AccountPnl:
    """Realized P&L of each day's sales (net of fees, at average cost, plus that day's fee activities) and the
    unrealized P&L of the shares held at each day's close; today's unrealized is the broker's, not a close's.
    A day is any day with a close, a fill or a fee between the first fill and today, and today."""
    unrealized_now = sum(p.unrealized_pl or 0.0 for p in positions)
    ordered = sorted(fills, key=lambda f: f.time)
    book = AverageCostBook()
    days: list[dict[str, Any]] = []
    unpriced: set[str] = set()
    if ordered:
        fills_by_day: dict[date, list[Fill]] = {}
        for fill in ordered:
            fills_by_day.setdefault(market_day(fill.time), []).append(fill)
        fees_by_day: dict[date, float] = {}
        for fee in fees:
            fees_by_day[fee.day] = fees_by_day.get(fee.day, 0.0) + fee.amount
        first = market_day(ordered[0].time)
        calendar = {d for series in closes.values() for d in series} | set(fills_by_day) | set(fees_by_day) | {today}
        for day in sorted(d for d in calendar if first <= d <= today):
            realized = fees_by_day.get(day, 0.0)
            for fill in fills_by_day.get(day, []):
                sale = book.apply(fill)
                if sale is not None:
                    realized += sale.pnl
            if day == today:
                unrealized = unrealized_now
            else:
                unrealized = 0.0
                for symbol, (shares, cost) in book.holdings().items():
                    close = _close_on_or_before(closes.get(symbol, {}), day)
                    if close is None:
                        unpriced.add(symbol)
                    else:
                        unrealized += shares * close - cost
            days.append({"date": day.isoformat(), "realized": round(realized, 2), "unrealized": round(unrealized, 2)})

    held = {symbol: shares for symbol, (shares, _) in book.holdings().items()}
    broker = {p.symbol: p.qty for p in positions}
    mismatches = [
        (symbol, held.get(symbol, 0.0), broker.get(symbol, 0.0))
        for symbol in sorted(held.keys() | broker.keys())
        if abs(held.get(symbol, 0.0) - broker.get(symbol, 0.0)) > 1e-6
    ]
    return AccountPnl(
        days=days,
        realized_total=round(sum(d["realized"] for d in days), 2),
        unrealized_now=round(unrealized_now, 2),
        positions=list(positions),
        mismatches=mismatches,
        unmatched_sales=dict(book.unmatched),
        unpriced=sorted(unpriced),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_pnl.py -v`
Expected: 11 passed.

- [ ] **Step 5: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all passed, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/dashboard/pnl.py tests/dashboard/test_pnl.py
git commit -m "feat(dashboard): daily realized and unrealized P&L of an account from its fills

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Env-file discovery and credentials

**Files:**
- Create: `src/trading_agent_framework/dashboard/account_reader.py`
- Test: `tests/dashboard/test_account_reader.py` (new)

**Interfaces:**
- Consumes: `config.env.AlpacaCredentials.for_trading(env=...)`, `.for_data(env=...)`, `config.env.find_project_root()`, `utils.errors.ConfigurationError`, `brokers.alpaca.client.build_trading_client` / `build_stock_data_client`.
- Produces:
  - `account_reader.default_env_dir() -> Path` (`find_project_root() / "env"`)
  - `account_reader.strategies_for_mode(mode: str, env_dir: Path) -> list[str]`
  - `account_reader.load_account_credentials(strategy: str, mode: str, env_dir: Path) -> tuple[AlpacaCredentials, AlpacaCredentials]` (trading, data)
  - `account_reader.load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl` (calls `load_account_pnl`, written in Task 4; this task adds a stub that Task 4 replaces)

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_account_reader.py`:

```python
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.utils.errors import ConfigurationError

KEYS = "ALPACA_API_KEY=trade-key\nALPACA_API_SECRET=trade-secret\nALPACA_DATA_API_KEY=data-key\nALPACA_DATA_API_SECRET=data-secret\n"


def _env(env_dir: Path, name: str, body: str) -> None:
    env_dir.mkdir(exist_ok=True)
    (env_dir / name).write_text(body, encoding="utf-8")


def test_strategies_are_the_alpaca_env_files_of_the_mode(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.beta.paper", KEYS)  # no BROKER: alpaca
    _env(tmp_path, ".env.gamma.paper", "BROKER=ibkr\n")
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.alpha.backtesting", "BROKER=alpaca\n" + KEYS)
    _env(tmp_path, ".env.example", KEYS)

    assert account_reader.strategies_for_mode("paper", tmp_path) == ["alpha", "beta"]
    assert account_reader.strategies_for_mode("live", tmp_path) == ["alpha"]


def test_no_env_dir_lists_no_strategy(tmp_path: Path) -> None:
    assert account_reader.strategies_for_mode("paper", tmp_path / "missing") == []


def test_credentials_come_from_the_file_without_touching_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_DATA_API_KEY", "ALPACA_DATA_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    _env(tmp_path, ".env.alpha.live", "BROKER=alpaca\nBROKER_API_IS_PAPER=false\n" + KEYS)

    trading, data = account_reader.load_account_credentials("alpha", "live", tmp_path)

    assert (trading.api_key, trading.api_secret, trading.is_paper) == ("trade-key", "trade-secret", False)
    assert (data.api_key, data.api_secret) == ("data-key", "data-secret")
    assert "ALPACA_API_KEY" not in os.environ


def test_blank_keys_raise_a_configuration_error_without_the_values(tmp_path: Path) -> None:
    _env(tmp_path, ".env.alpha.paper", "ALPACA_API_KEY=\nALPACA_API_SECRET=s3cret\n")

    with pytest.raises(ConfigurationError, match="ALPACA_API_KEY") as excinfo:
        account_reader.load_account_credentials("alpha", "paper", tmp_path)

    assert "s3cret" not in str(excinfo.value)


def test_a_missing_env_file_raises_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"\.env\.alpha\.paper"):
        account_reader.load_account_credentials("alpha", "paper", tmp_path)


def test_load_strategy_pnl_builds_the_clients_from_the_strategys_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(tmp_path, ".env.alpha.paper", "BROKER_API_IS_PAPER=true\n" + KEYS)
    built: dict[str, object] = {}
    monkeypatch.setattr(account_reader, "build_trading_client", lambda creds: built.setdefault("trading", creds))
    monkeypatch.setattr(account_reader, "build_stock_data_client", lambda creds: built.setdefault("data", creds))
    monkeypatch.setattr(account_reader, "load_account_pnl", lambda trading, data, today: (trading, data, today))

    result = account_reader.load_strategy_pnl("alpha", "paper", tmp_path, date(2026, 10, 13))

    trading, data, today = result  # ty: ignore[not-iterable]
    assert trading is built["trading"] and data is built["data"]
    assert trading.api_key == "trade-key" and trading.is_paper is True
    assert data.api_key == "data-key"
    assert today == date(2026, 10, 13)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_account_reader.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'account_reader'`.

- [ ] **Step 3: Write the minimal implementation**

Create `src/trading_agent_framework/dashboard/account_reader.py`:

```python
"""A strategy's Alpaca account for the Live and Paper tabs: env-file discovery, credentials and read-only fetches.

Streamlit-free. Keys are read with `dotenv_values` -- never `load_dotenv` -- so one strategy's keys never reach
`os.environ` and switching strategies cannot mix accounts.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.config.env import AlpacaCredentials, find_project_root
from trading_agent_framework.dashboard.pnl import AccountPnl
from trading_agent_framework.utils.errors import ConfigurationError


def default_env_dir() -> Path:
    return find_project_root() / "env"


def _env_values(path: Path) -> dict[str, str]:
    return {key: value or "" for key, value in dotenv_values(path).items()}


def strategies_for_mode(mode: str, env_dir: Path) -> list[str]:
    """The strategies with an `env_dir/.env.<strategy>.<mode>` file trading on Alpaca (no `BROKER` means alpaca)."""
    if not env_dir.is_dir():
        return []
    prefix, suffix = ".env.", f".{mode}"
    names = []
    for path in env_dir.glob(f"{prefix}*{suffix}"):
        broker = (_env_values(path).get("BROKER") or "alpaca").strip().lower()
        if broker == "alpaca":
            names.append(path.name[len(prefix) : -len(suffix)])
    return sorted(names)


def load_account_credentials(strategy: str, mode: str, env_dir: Path) -> tuple[AlpacaCredentials, AlpacaCredentials]:
    """The trading and the market-data keys of `env_dir/.env.<strategy>.<mode>`."""
    path = env_dir / f".env.{strategy}.{mode}"
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    values = _env_values(path)
    return AlpacaCredentials.for_trading(env=values), AlpacaCredentials.for_data(env=values)


def load_account_pnl(trading: Any, data: Any, today: date) -> AccountPnl:
    raise NotImplementedError  # Task 4


def load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
    """The daily P&L of the strategy's Alpaca account, fetched now."""
    trading_creds, data_creds = load_account_credentials(strategy, mode, env_dir)
    return load_account_pnl(build_trading_client(trading_creds), build_stock_data_client(data_creds), today)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_account_reader.py -v`
Expected: 6 passed.

- [ ] **Step 5: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all passed, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/dashboard/account_reader.py tests/dashboard/test_account_reader.py
git commit -m "feat(dashboard): find the strategies' Alpaca env files and read their keys

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Alpaca fetches and `load_account_pnl`

**Files:**
- Modify: `src/trading_agent_framework/dashboard/account_reader.py`
- Test: `tests/dashboard/test_account_reader.py`

**Interfaces:**
- Consumes: `pnl.Fill`, `pnl.Fee`, `pnl.BrokerPosition`, `pnl.market_day`, `pnl.daily_account_pnl` (Tasks 1-2); `tests.fakes.FakeStockHistoricalDataClient`, `tests.fakes.bar_payload`, `tests.fakes.make_alpaca_position`.
- Produces:
  - `account_reader.AccountTradingClient` (Protocol: `get(path: str, data: dict[str, Any] | None = None) -> Any`, `get_all_positions() -> list[alpaca Position]`)
  - `account_reader.AccountDataClient` (Protocol: `get_stock_bars(request_params: StockBarsRequest) -> BarSet`)
  - `account_reader.PAGE_SIZE = 100`
  - `account_reader.fetch_activities(client: AccountTradingClient) -> tuple[list[Fill], list[Fee]]`
  - `account_reader.fetch_daily_closes(client: AccountDataClient, symbols: Sequence[str], first_day: date, today: date) -> dict[str, dict[date, float]]`
  - `account_reader.to_broker_position(position: alpaca Position) -> BrokerPosition`
  - `account_reader.load_account_pnl(trading: AccountTradingClient, data: AccountDataClient, today: date) -> AccountPnl` (replaces the Task 3 stub)

- [ ] **Step 1: Write the failing tests**

Append to `tests/dashboard/test_account_reader.py` (merge the imports at the top of the file):

```python
from datetime import datetime

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame
from tests.fakes import FakeStockHistoricalDataClient, bar_payload, make_alpaca_position

from trading_agent_framework.dashboard.pnl import Fee


def _fill_activity(n: int, symbol: str = "AAA", side: str = "buy", qty: str = "10", price: str = "100", time: str = "2026-10-05T15:00:00.123456789Z") -> dict[str, str]:
    return {
        "id": f"20261005150000000::{n:08d}",
        "activity_type": "FILL",
        "transaction_time": time,
        "type": "fill",
        "price": price,
        "qty": qty,
        "side": side,
        "symbol": symbol,
        "leaves_qty": "0",
        "order_id": "b5abe576-6a8a-49f3-a353-46b72c1ccae9",
        "cum_qty": qty,
        "order_status": "filled",
    }


def _fee_activity(n: int, day: str, amount: str, status: str = "executed") -> dict[str, str]:
    return {"id": f"20261006000000000::{n:08d}", "activity_type": "FEE", "activity_sub_type": "TAF", "date": f"{day}T00:00:00Z", "net_amount": amount, "status": status, "symbol": "AAA"}


class FakeAccountClient:
    """Answers `GET /account/activities` from `activities` like Alpaca: `page_size` items after the `page_token` id."""

    def __init__(self, activities: list[dict[str, str]], positions: list[object] | None = None) -> None:
        self.activities = activities
        self.positions = positions or []
        self.requests: list[tuple[str, dict[str, object]]] = []

    def get(self, path: str, data: dict[str, object] | None = None) -> list[dict[str, str]]:
        params = dict(data or {})
        self.requests.append((path, params))
        start = 0
        if "page_token" in params:
            start = next(i for i, item in enumerate(self.activities) if item["id"] == params["page_token"]) + 1
        return self.activities[start : start + int(params["page_size"])]  # ty: ignore[invalid-argument-type]

    def get_all_positions(self) -> list[object]:
        return self.positions


def test_activities_are_fills_and_fees_oldest_first() -> None:
    client = FakeAccountClient([_fill_activity(1), _fill_activity(2, side="sell", price="110"), _fee_activity(3, "2026-10-06", "-0.02")])

    fills, fees = account_reader.fetch_activities(client)

    assert client.requests == [("/account/activities", {"activity_types": "FILL,FEE", "direction": "asc", "page_size": 100})]
    assert [(f.symbol, f.side, f.qty, f.price) for f in fills] == [("AAA", "buy", 10.0, 100.0), ("AAA", "sell", 10.0, 110.0)]
    assert fills[0].time.isoformat() == "2026-10-05T15:00:00.123456+00:00"
    assert fees == [Fee(date(2026, 10, 6), -0.02)]


def test_activities_follow_the_page_token_through_a_full_page_to_an_empty_one() -> None:
    client = FakeAccountClient([_fill_activity(n) for n in range(100)])

    fills, _ = account_reader.fetch_activities(client)

    assert len(fills) == 100
    assert len(client.requests) == 2
    assert client.requests[1][1]["page_token"] == "20261005150000000::00000099"


def test_activities_over_several_pages_are_all_read() -> None:
    client = FakeAccountClient([_fill_activity(n) for n in range(150)])

    fills, _ = account_reader.fetch_activities(client)

    assert len(fills) == 150
    assert len(client.requests) == 2


def test_a_cancelled_fee_is_left_out() -> None:
    client = FakeAccountClient([_fee_activity(1, "2026-10-06", "-0.02", status="canceled")])

    assert account_reader.fetch_activities(client) == ([], [])


def test_daily_closes_are_raw_iex_day_bars_keyed_by_new_york_date() -> None:
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAA": [bar_payload("2026-10-05T04:00:00Z", 101.0), bar_payload("2026-10-06T04:00:00Z", 102.0)]}

    closes = account_reader.fetch_daily_closes(data, ["AAA", "BBB"], date(2026, 10, 5), date(2026, 10, 7))

    assert closes == {"AAA": {date(2026, 10, 5): 101.0, date(2026, 10, 6): 102.0}}
    (request,) = data.bars_requests
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.RAW
    assert request.timeframe.value == TimeFrame.Day.value
    # alpaca-py stores the bounds as naive UTC: midnight in New York is 04:00 UTC in October. Ending there leaves
    # today's partial bar out.
    assert request.start == datetime(2026, 10, 5, 4, 0)
    assert request.end == datetime(2026, 10, 7, 4, 0)


def test_no_symbol_or_a_first_day_of_today_fetches_no_bars() -> None:
    data = FakeStockHistoricalDataClient()

    assert account_reader.fetch_daily_closes(data, [], date(2026, 10, 5), date(2026, 10, 7)) == {}
    assert account_reader.fetch_daily_closes(data, ["AAA"], date(2026, 10, 7), date(2026, 10, 7)) == {}
    assert data.bars_requests == []


def test_load_account_pnl_rebuilds_the_days_from_the_three_endpoints() -> None:
    trading = FakeAccountClient(
        [_fill_activity(1, time="2026-10-05T15:00:00Z")],
        positions=[make_alpaca_position(symbol="AAA", qty="10", unrealized_pl="30.00", unrealized_plpc="0.03", current_price="103.00", market_value="1030.00")],
    )
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAA": [bar_payload("2026-10-05T04:00:00Z", 101.0), bar_payload("2026-10-06T04:00:00Z", 102.0)]}

    result = account_reader.load_account_pnl(trading, data, date(2026, 10, 7))

    assert result.days == [
        {"date": "2026-10-05", "realized": 0.0, "unrealized": 10.0},
        {"date": "2026-10-06", "realized": 0.0, "unrealized": 20.0},
        {"date": "2026-10-07", "realized": 0.0, "unrealized": 30.0},
    ]
    (position,) = result.positions
    assert (position.symbol, position.qty, position.current_price, position.unrealized_plpc) == ("AAA", 10.0, 103.0, 0.03)
    assert result.mismatches == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_account_reader.py -v`
Expected: the 7 new tests FAIL with `AttributeError: module ... has no attribute 'fetch_activities'` / `'fetch_daily_closes'`, and `NotImplementedError` for `load_account_pnl`; the 6 Task 3 tests still pass.

- [ ] **Step 3: Write the minimal implementation**

In `account_reader.py`, replace the imports with:

```python
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import dotenv_values

from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.config.env import AlpacaCredentials, find_project_root
from trading_agent_framework.dashboard.pnl import AccountPnl, BrokerPosition, Fee, Fill, daily_account_pnl, market_day
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.trading.models import Position
```

Add after `load_account_credentials` (and delete the Task 3 `load_account_pnl` stub):

```python
PAGE_SIZE = 100  # Alpaca's maximum for /account/activities


class AccountTradingClient(Protocol):
    """The only `TradingClient` calls the dashboard makes: GETs. No order method, so the reader cannot trade."""

    def get(self, path: str, data: dict[str, Any] | str | None = None, **kwargs: Any) -> Any: ...
    def get_all_positions(self) -> list[Position]: ...


class AccountDataClient(Protocol):
    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...


def _parse_activity(item: dict[str, Any]) -> Fill | Fee | None:
    # Parsed by hand: the Trading API's activities have no `account_id`, which alpaca-py's activity models require.
    kind = item.get("activity_type")
    if kind == "FILL":
        return Fill(
            time=datetime.fromisoformat(item["transaction_time"]),
            symbol=item["symbol"],
            side=item["side"],
            qty=float(item["qty"]),
            price=float(item["price"]),
        )
    if kind == "FEE" and item.get("status") != "canceled":
        return Fee(day=date.fromisoformat(str(item["date"])[:10]), amount=float(item["net_amount"]))
    return None


def fetch_activities(client: AccountTradingClient) -> tuple[list[Fill], list[Fee]]:
    """Every fill and fee of the account, oldest first, following `page_token` until a short page."""
    fills: list[Fill] = []
    fees: list[Fee] = []
    params: dict[str, Any] = {"activity_types": "FILL,FEE", "direction": "asc", "page_size": PAGE_SIZE}
    while True:
        page = cast(list[dict[str, Any]], client.get("/account/activities", params))
        for item in page:
            parsed = _parse_activity(item)
            if isinstance(parsed, Fill):
                fills.append(parsed)
            elif isinstance(parsed, Fee):
                fees.append(parsed)
        if len(page) < PAGE_SIZE:
            return fills, fees
        params = {**params, "page_token": page[-1]["id"]}


def fetch_daily_closes(client: AccountDataClient, symbols: Sequence[str], first_day: date, today: date) -> dict[str, dict[date, float]]:
    """Raw (unadjusted, like the fills) IEX daily closes from `first_day` up to, not including, today."""
    if not symbols or first_day >= today:
        return {}
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=datetime.combine(first_day, time.min, MARKET_TZ),
        end=datetime.combine(today, time.min, MARKET_TZ),
        feed=DataFeed.IEX,
        adjustment=Adjustment.RAW,
    )
    barset = client.get_stock_bars(request)
    return {symbol: {market_day(bar.timestamp): bar.close for bar in bars} for symbol, bars in barset.data.items() if bars}


def _float(value: str | None) -> float | None:
    return None if value is None else float(value)


def to_broker_position(position: Position) -> BrokerPosition:
    return BrokerPosition(
        symbol=position.symbol,
        qty=float(position.qty),
        avg_entry_price=float(position.avg_entry_price),
        current_price=_float(position.current_price),
        market_value=_float(position.market_value),
        unrealized_pl=_float(position.unrealized_pl),
        unrealized_plpc=_float(position.unrealized_plpc),
    )


def load_account_pnl(trading: AccountTradingClient, data: AccountDataClient, today: date) -> AccountPnl:
    """The account's daily P&L: its fills and fees, the daily closes of every symbol it traded, its positions now."""
    fills, fees = fetch_activities(trading)
    positions = [to_broker_position(p) for p in trading.get_all_positions()]
    closes: dict[str, dict[date, float]] = {}
    if fills:
        first_day = min(market_day(f.time) for f in fills)
        closes = fetch_daily_closes(data, sorted({f.symbol for f in fills}), first_day, today)
    return daily_account_pnl(fills, fees, closes, positions, today)
```

Cast the built clients in `load_strategy_pnl`, as `brokers/alpaca/broker.py` does (alpaca-py's methods return `X | RawData`, wider than the Protocols):

```python
def load_strategy_pnl(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
    """The daily P&L of the strategy's Alpaca account, fetched now."""
    trading_creds, data_creds = load_account_credentials(strategy, mode, env_dir)
    trading = cast(AccountTradingClient, build_trading_client(trading_creds))
    data = cast(AccountDataClient, build_stock_data_client(data_creds))
    return load_account_pnl(trading, data, today)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_account_reader.py -v`
Expected: 13 passed.

- [ ] **Step 5: Full suite, lint, type check**

Run: `uv run pytest -q && uv run ruff check && uv run pyright src/trading_agent_framework/dashboard/account_reader.py src/trading_agent_framework/dashboard/pnl.py`
Expected: all passed, `All checks passed!`, `0 errors`.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/dashboard/account_reader.py tests/dashboard/test_account_reader.py
git commit -m "feat(dashboard): fetch an Alpaca account's fills, closes and positions read-only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `account_pnl_chart`

**Files:**
- Modify: `src/trading_agent_framework/dashboard/components/charts.py` (rename `_signed_usd` at ~line 839 and its two uses in `session_pnl_chart`; add the new chart after `session_pnl_chart`)
- Test: `tests/dashboard/test_account_pnl_chart.py` (new)

**Interfaces:**
- Consumes: `AccountPnl.days` row shape `{"date", "realized", "unrealized"}` (Task 2).
- Produces:
  - `charts.signed_usd(value: float) -> str` (was `_signed_usd`)
  - `charts.account_pnl_chart(days: list[dict[str, Any]], title: str = "P&L per day") -> go.Figure`

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_account_pnl_chart.py`:

```python
from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import CHART_TEMPLATE, account_pnl_chart, signed_usd

DAYS = [
    {"date": "2026-10-05", "realized": 0.0, "unrealized": 49.0},
    {"date": "2026-10-06", "realized": -12.5, "unrealized": -3.0},
]


def test_two_grouped_bar_series_realized_then_unrealized() -> None:
    fig = account_pnl_chart(DAYS)

    realized, unrealized = fig.data
    assert (realized.type, realized.name, list(realized.x), list(realized.y)) == ("bar", "Realized", ["2026-10-05", "2026-10-06"], [0.0, -12.5])
    assert (unrealized.type, unrealized.name, list(unrealized.y)) == ("bar", "Unrealized", [49.0, -3.0])
    assert fig.layout.barmode == "group"


def test_bars_are_green_for_a_gain_and_red_for_a_loss_unrealized_hatched() -> None:
    realized, unrealized = account_pnl_chart(DAYS).data

    assert list(realized.marker.color) == ["#16a34a", "#dc2626"]
    assert list(unrealized.marker.color) == ["#16a34a", "#dc2626"]
    assert unrealized.marker.pattern.shape == "/"


def test_the_chart_uses_the_dark_template() -> None:
    assert account_pnl_chart(DAYS).layout.template == CHART_TEMPLATE


def test_no_day_shows_the_no_data_annotation() -> None:
    fig = account_pnl_chart([])

    assert fig.data == ()
    assert fig.layout.annotations[0].text == "No trade data available"


def test_signed_usd_formats_both_signs() -> None:
    assert (signed_usd(1234.5), signed_usd(-0.05)) == ("+$1,234.50", "-$0.05")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_account_pnl_chart.py -v`
Expected: collection ERROR, `ImportError: cannot import name 'account_pnl_chart'`.

- [ ] **Step 3: Write the minimal implementation**

In `charts.py`, rename `def _signed_usd` to `def signed_usd` and both calls inside `session_pnl_chart` to `signed_usd(...)`. After `session_pnl_chart`, add:

```python
def _sign_colors(values: Sequence[float]) -> list[str]:
    return ["#16a34a" if value >= 0 else "#dc2626" for value in values]


def account_pnl_chart(days: list[dict[str, Any]], title: str = "P&L per day") -> go.Figure:
    """Two bars per trading day (`pnl.daily_account_pnl`): the P&L realized by that day's sales, and the unrealized
    P&L of the positions open at its close (hatched). Green for a gain, red for a loss."""
    if not days:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available", showarrow=False)
        fig.update_layout(title=title, template=CHART_TEMPLATE)
        return fig

    dates = [day["date"] for day in days]
    realized = [day["realized"] for day in days]
    unrealized = [day["unrealized"] for day in days]
    fig = go.Figure(
        [
            go.Bar(
                x=dates,
                y=realized,
                name="Realized",
                marker=dict(color=_sign_colors(realized)),
                hovertemplate="%{x}<br>Realized: %{y:$,.2f}<extra></extra>",
            ),
            go.Bar(
                x=dates,
                y=unrealized,
                name="Unrealized",
                marker=dict(color=_sign_colors(unrealized), opacity=0.55, pattern=dict(shape="/")),
                hovertemplate="%{x}<br>Unrealized: %{y:$,.2f}<extra></extra>",
            ),
        ]
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#6b7280")
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        template=CHART_TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(type="category")  # trading days only: no empty weekend gaps
    return fig
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/dashboard/test_account_pnl_chart.py tests/dashboard/test_session_pnl_chart.py -v`
Expected: all passed (5 new, the 4 session chart tests unchanged).

- [ ] **Step 5: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all passed, `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/dashboard/components/charts.py tests/dashboard/test_account_pnl_chart.py
git commit -m "feat(dashboard): grouped realized/unrealized P&L bar chart

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Live and Paper pages, navigation, docs

**Files:**
- Create: `src/trading_agent_framework/dashboard/_pages/account.py`
- Modify: `src/trading_agent_framework/dashboard/app.py`
- Modify: `tests/dashboard/test_app_smoke.py` (`test_the_tabs_are_backtesting_then_models`, ~line 204)
- Modify: `src/trading_agent_framework/dashboard/CLAUDE.md`, `CLAUDE.md` (the `uv run dashboard` command line)
- Test: `tests/dashboard/test_account_page.py` (new)

**Interfaces:**
- Consumes: `account_reader.default_env_dir`, `account_reader.strategies_for_mode`, `account_reader.load_strategy_pnl` (Tasks 3-4); `pnl.AccountPnl`, `pnl.BrokerPosition` (Task 2); `charts.account_pnl_chart`, `charts.signed_usd` (Task 5); `components.metric_cards.render_metric_card(label, value, fmt=".2f", is_pct=True, tone=None, help=None)`.
- Produces: `_pages.account.page_account(mode: str) -> None`, `page_live() -> None`, `page_paper() -> None`; `app.NAV_PAGES` with Live and Paper; `app.DEFAULT_PAGE = "Backtesting"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dashboard/test_account_page.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from tests.fakes import make_api_error

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.dashboard.pnl import AccountPnl, BrokerPosition

KEYS = "ALPACA_API_KEY=k\nALPACA_API_SECRET=s\nALPACA_DATA_API_KEY=dk\nALPACA_DATA_API_SECRET=ds\n"

PNL = AccountPnl(
    days=[{"date": "2026-10-12", "realized": 0.0, "unrealized": 20.0}, {"date": "2026-10-13", "realized": 5.0, "unrealized": 30.0}],
    realized_total=5.0,
    unrealized_now=30.0,
    positions=[BrokerPosition("AAA", 10.0, 100.0, 103.0, 1030.0, 30.0, 0.03)],
    mismatches=[],
    unmatched_sales={},
    unpriced=[],
)


def _paper_script() -> None:
    # Serialized by AppTest.from_function: must be self-contained.
    from trading_agent_framework.dashboard._pages.account import page_paper

    page_paper()


def _live_script() -> None:
    from trading_agent_framework.dashboard._pages.account import page_live

    page_live()


def _switching_script() -> None:
    # Stands in for the top navigation: session_state["tab"] picks the tab drawn on this run.
    import streamlit as st

    from trading_agent_framework.dashboard._pages.account import page_account

    page_account(st.session_state.get("tab", "paper"))


@pytest.fixture
def env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    env = tmp_path / "env"
    env.mkdir()
    (env / ".env.alpha.paper").write_text(KEYS, encoding="utf-8")
    (env / ".env.beta.paper").write_text(KEYS, encoding="utf-8")
    (env / ".env.alpha.live").write_text("BROKER_API_IS_PAPER=false\n" + KEYS, encoding="utf-8")
    monkeypatch.setattr(account_reader, "default_env_dir", lambda: env)
    st.cache_data.clear()
    return env


def _run(script: Callable[[], None], monkeypatch: pytest.MonkeyPatch, result: object) -> tuple[AppTest, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def fake_load(strategy: str, mode: str, env_dir: Path, today: date) -> AccountPnl:
        calls.append((strategy, mode))
        if isinstance(result, BaseException):
            raise result
        return result  # ty: ignore[invalid-return-type]

    monkeypatch.setattr(account_reader, "load_strategy_pnl", fake_load)
    at = AppTest.from_function(script, default_timeout=30)
    at.run()
    return at, calls


def test_paper_tab_lists_the_paper_strategies_and_shows_the_first_ones_pnl(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_paper_script, monkeypatch, PNL)

    assert not at.exception
    assert at.sidebar.selectbox[0].options == ["alpha", "beta"]
    assert calls == [("alpha", "paper")]
    assert [title.value for title in at.title] == ["Paper"]
    assert len(at.get("plotly_chart")) == 1
    assert at.dataframe[0].value["symbol"].tolist() == ["AAA"]
    page_text = " ".join(md.value for md in at.markdown)
    assert "+$5.00" in page_text and "+$30.00" in page_text and "+$35.00" in page_text


def test_choosing_another_strategy_fetches_its_account(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_paper_script, monkeypatch, PNL)

    at.sidebar.selectbox[0].select("beta").run()

    assert calls[-1] == ("beta", "paper")


def test_live_lists_the_live_strategies(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, calls = _run(_live_script, monkeypatch, PNL)

    assert at.sidebar.selectbox[0].options == ["alpha"]
    assert calls == [("alpha", "live")]


def test_each_tab_remembers_its_strategy_across_tab_switches(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Streamlit drops a widget's state on a run that does not draw it (another tab): the page must keep the choice.
    at, _ = _run(_switching_script, monkeypatch, PNL)
    at.sidebar.selectbox[0].select("beta").run()

    at.session_state["tab"] = "live"
    at.run()
    assert at.sidebar.selectbox[0].value == "alpha"

    at.session_state["tab"] = "paper"
    at.run()
    assert at.sidebar.selectbox[0].value == "beta"


def test_mismatches_unmatched_sales_and_unpriced_symbols_are_warned(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    flawed = dataclasses.replace(PNL, mismatches=[("AAA", 10.0, 15.0)], unmatched_sales={"BBB": 5.0}, unpriced=["ZZZ"])
    at, _ = _run(_paper_script, monkeypatch, flawed)

    (warning,) = at.warning
    assert "AAA" in warning.value and "BBB" in warning.value and "ZZZ" in warning.value


def test_an_alpaca_error_is_shown_not_raised(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at, _ = _run(_paper_script, monkeypatch, make_api_error(500))

    assert not at.exception
    assert "Could not reach Alpaca" in at.error[0].value


def test_blank_keys_name_the_variable_and_never_a_value(env_dir: Path) -> None:
    (env_dir / ".env.alpha.paper").write_text("ALPACA_API_KEY=\nALPACA_API_SECRET=t0p-s3cret\n", encoding="utf-8")
    at = AppTest.from_function(_paper_script, default_timeout=30)  # the real load_strategy_pnl: fails before any network
    at.run()

    assert not at.exception
    assert "ALPACA_API_KEY" in at.error[0].value
    assert "t0p-s3cret" not in at.error[0].value


def test_no_fill_history_shows_positions_and_says_so(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    at, _ = _run(_paper_script, monkeypatch, dataclasses.replace(PNL, days=[], realized_total=0.0))

    assert len(at.get("plotly_chart")) == 0
    assert any("no fill history" in info.value.lower() for info in at.info)
    assert at.dataframe[0].value["symbol"].tolist() == ["AAA"]


def test_an_empty_account_says_no_trades_yet(env_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    at, _ = _run(_paper_script, monkeypatch, dataclasses.replace(PNL, days=[], realized_total=0.0, unrealized_now=0.0, positions=[]))

    assert any("No trades yet" in info.value for info in at.info)


def test_a_mode_without_env_files_says_how_to_add_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(account_reader, "default_env_dir", lambda: tmp_path / "nothing")
    at = AppTest.from_function(_paper_script, default_timeout=30)
    at.run()

    assert not at.exception
    assert ".env.<strategy>.paper" in at.info[0].value
```

In `tests/dashboard/test_app_smoke.py`, replace `test_the_tabs_are_backtesting_then_models` with:

```python
def test_the_tabs_are_live_paper_backtesting_then_models_with_backtesting_the_default() -> None:
    from trading_agent_framework.dashboard import app

    assert [(title, url_path) for title, url_path, _ in app.NAV_PAGES] == [
        ("Live", "live"),
        ("Paper", "paper"),
        ("Backtesting", "backtesting"),
        ("Models", "models"),
    ]
    assert app.DEFAULT_PAGE == "Backtesting"
```

(`test_backtesting_is_the_default_tab_with_its_sidebar` stays unchanged: it proves Backtesting still opens first.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/dashboard/test_account_page.py tests/dashboard/test_app_smoke.py -v`
Expected: `test_account_page.py` tests fail (`ModuleNotFoundError: ... _pages.account` raised inside the script -> `at.exception`, or the fixture's assertions), and the new tab-order test fails on the `NAV_PAGES` list.

- [ ] **Step 3: Write the page**

Create `src/trading_agent_framework/dashboard/_pages/account.py`:

```python
"""Live and Paper tabs: one strategy's realized and unrealized P&L per day, rebuilt from its Alpaca account."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime

import pandas as pd
import streamlit as st
from alpaca.common.exceptions import APIError
from requests import RequestException

from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.dashboard.components.charts import account_pnl_chart, signed_usd
from trading_agent_framework.dashboard.components.metric_cards import render_metric_card
from trading_agent_framework.dashboard.pnl import AccountPnl
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError


@st.cache_data(show_spinner="Fetching the account from Alpaca...", ttl="5m", max_entries=32)
def _account_pnl(strategy: str, mode: str, today: date) -> AccountPnl:
    # Keyed by name, mode and day only: the keys are re-read from the env file inside, never cached or shown.
    return account_reader.load_strategy_pnl(strategy, mode, account_reader.default_env_dir(), today)


def _problems(pnl: AccountPnl) -> list[str]:
    problems = [f"{symbol}: the fills give {replayed:g} shares, the broker holds {held:g}" for symbol, replayed, held in pnl.mismatches]
    problems += [f"{symbol}: {qty:g} shares sold with no recorded buy (their P&L is missing)" for symbol, qty in pnl.unmatched_sales.items()]
    problems += [f"{symbol}: no daily close, left out of the unrealized P&L before today" for symbol in pnl.unpriced]
    return problems


def _render_metrics(pnl: AccountPnl) -> None:
    total = pnl.realized_total + pnl.unrealized_now
    cards = [("Realized", pnl.realized_total), ("Unrealized now", pnl.unrealized_now), ("Total", total)]
    for column, (label, value) in zip(st.columns(len(cards)), cards, strict=True):
        with column:
            render_metric_card(label, signed_usd(value), tone="positive" if value >= 0 else "negative")


def _render_positions(pnl: AccountPnl) -> None:
    st.subheader("Open positions")
    if not pnl.positions:
        st.caption("No open position.")
        return
    frame = pd.DataFrame([asdict(p) for p in pnl.positions])
    frame["unrealized_plpc"] = frame["unrealized_plpc"] * 100
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol"),
            "qty": st.column_config.NumberColumn("Qty", format="%g"),
            "avg_entry_price": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_value": st.column_config.NumberColumn("Value", format="$%,.2f"),
            "unrealized_pl": st.column_config.NumberColumn("Unrealized", format="$%,.2f"),
            "unrealized_plpc": st.column_config.NumberColumn("Unrealized %", format="%.2f%%"),
        },
    )


def page_account(mode: str) -> None:
    """One tab: the sidebar picks the strategy, the page shows its account's P&L per day and its open positions."""
    st.title(mode.capitalize())
    strategies = account_reader.strategies_for_mode(mode, account_reader.default_env_dir())
    if not strategies:
        st.info(f"No strategy trades {mode} on Alpaca: add an env/.env.<strategy>.{mode} file with BROKER=alpaca.")
        return

    # The choice is kept under a plain session key: Streamlit drops the widget's own state on a run that does not
    # draw it, i.e. whenever another tab is open.
    remembered = f"account_strategy_{mode}"
    chosen = st.session_state.get(remembered)
    strategy = st.sidebar.selectbox(
        "Strategy",
        strategies,
        index=strategies.index(chosen) if chosen in strategies else 0,
        key=f"account_strategy_select_{mode}",
    )
    st.session_state[remembered] = strategy
    if st.sidebar.button("Refresh", icon=":material/refresh:", width="stretch", key=f"account_refresh_{mode}"):
        _account_pnl.clear()
    st.caption(f"{strategy} · {mode} Alpaca account")

    try:
        pnl = _account_pnl(strategy, mode, datetime.now(MARKET_TZ).date())
    except ConfigurationError as exc:
        st.error(f"env/.env.{strategy}.{mode}: {exc}")
        return
    except (APIError, RequestException) as exc:
        st.error(f"Could not reach Alpaca: {exc}")
        return

    if not pnl.days and not pnl.positions:
        st.info("No trades yet.")
        return

    _render_metrics(pnl)
    if pnl.days:
        st.plotly_chart(account_pnl_chart(pnl.days), width="stretch")
        st.caption(
            "Realized: the P&L of that day's sales, net of fees, against the average cost. "
            "Unrealized (hatched): the open positions at that day's close; today, at the current price."
        )
    else:
        st.info("No fill history in this account: only today's unrealized P&L is known.")
    problems = _problems(pnl)
    if problems:
        st.warning("The fills do not fully explain the account:\n\n" + "\n".join(f"- {p}" for p in problems))
    _render_positions(pnl)


def page_live() -> None:
    page_account("live")


def page_paper() -> None:
    page_account("paper")
```

- [ ] **Step 4: Wire the tabs**

In `src/trading_agent_framework/dashboard/app.py`:

- Add the import `from trading_agent_framework.dashboard._pages.account import page_live, page_paper`.
- Replace the `NAV_PAGES` comment and tuple with:

```python
# (title, url path, page function), in tab order; DEFAULT_PAGE opens first. Plain data, so tests can
# check the tabs without a Streamlit runtime (st.Page cannot be built outside one).
NAV_PAGES = (
    ("Live", "live", page_live),
    ("Paper", "paper", page_paper),
    ("Backtesting", "backtesting", page_backtesting),
    ("Models", "models", page_models),
)
DEFAULT_PAGE = "Backtesting"
```

- In `main()`, build the pages with `default=(title == DEFAULT_PAGE)`:

```python
    pages = [st.Page(page, title=title, url_path=url_path, default=(title == DEFAULT_PAGE)) for title, url_path, page in NAV_PAGES]
```

- Update the module docstring's first line to `"""Streamlit dashboard: Live, Paper, Backtesting and Models tabs.`

- [ ] **Step 5: Run the page and smoke tests**

Run: `uv run pytest tests/dashboard/test_account_page.py tests/dashboard/test_app_smoke.py -v`
Expected: all passed. If `at.dataframe[0].value` is not a DataFrame in this Streamlit version, read the symbol column the way `test_scorecard_shows_the_runs_slippage_right_of_the_broker_column` reads its dataframe.

- [ ] **Step 6: Docs**

In `src/trading_agent_framework/dashboard/CLAUDE.md`, change "over two tabs" to "over four tabs" and add before "**Backtesting**":

```markdown
**Live** and **Paper** (`_pages/account.py`, one `page_account(mode)`): the sidebar `selectbox` (keyed per mode) lists the strategies with an `env/.env.<strategy>.<mode>` file whose `BROKER` is alpaca; `account_reader.py` (Streamlit-free) reads that file's keys with `dotenv_values` -- never `load_dotenv`, so no key reaches `os.environ` -- and calls three Alpaca GETs through read-only Protocols (`/account/activities` FILL+FEE, IEX **RAW** daily bars since fills are unadjusted, positions); `pnl.py` (pure) replays the fills at average cost -- `AverageCostBook`, shared with `reader.load_session_pnl` -- into realized P&L per US/Eastern day and the unrealized P&L at each day's close (today: the positions' `unrealized_pl`), and reports fills that do not explain the positions. Fetched on demand (`st.cache_data`, Refresh clears it), no polling. Backtesting stays the default tab (`app.DEFAULT_PAGE`).
```

In the root `CLAUDE.md`, change the dashboard command comment to:

```
uv run dashboard [--benchmark-dir PATH]   # Streamlit dashboard: Live and Paper tabs (P&L per day from the strategy's Alpaca account), Backtesting tab (logs/ and memory/ run history) and Models tab (vLLM benchmark results)
```

- [ ] **Step 7: Full suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: all passed, `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/dashboard/_pages/account.py src/trading_agent_framework/dashboard/app.py tests/dashboard/test_account_page.py tests/dashboard/test_app_smoke.py src/trading_agent_framework/dashboard/CLAUDE.md CLAUDE.md
git commit -m "feat(dashboard): Live and Paper tabs with a strategy's daily P&L from Alpaca

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Manual check against a real paper account (needs the user's go-ahead)

**Files:** none (no commit).

- [ ] **Step 1: Ask the user** which paper strategy to fetch (it calls Alpaca's GET endpoints with that strategy's paper keys; nothing is written).

- [ ] **Step 2: Run a read-only fetch** (scratchpad script, not committed):

```bash
uv run python -c "
from datetime import datetime
from trading_agent_framework.dashboard import account_reader
from trading_agent_framework.utils.clock import MARKET_TZ
pnl = account_reader.load_strategy_pnl('<strategy>', 'paper', account_reader.default_env_dir(), datetime.now(MARKET_TZ).date())
print(len(pnl.days), 'days; realized', pnl.realized_total, '; unrealized now', pnl.unrealized_now)
print('last days', pnl.days[-3:])
print('positions', [(p.symbol, p.qty, p.unrealized_pl) for p in pnl.positions])
print('mismatches', pnl.mismatches, 'unmatched', pnl.unmatched_sales, 'unpriced', pnl.unpriced)
"
```

Expected: no exception; `unrealized now` equals the sum of the positions' `unrealized_pl`; mismatches empty for an account whose history starts with this strategy (report any to the user rather than "fixing" them).

- [ ] **Step 3: Report** the numbers to the user and offer to start `uv run dashboard` for a look in the browser.

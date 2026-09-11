# Market Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stock market data (`get_last_price(s)`, `get_quote`, `get_bars` / `get_historical_prices(_for_assets)`) and a pandas-ta-classic indicators accessor to the broker and strategy layers, after renaming the `strategies/` package to `core/`.

**Architecture:** `Broker` gains four abstract market-data methods. `AlpacaBroker` implements them with an injected `StockHistoricalDataClient` plus a new pure translation module, `brokers/alpaca/market_data.py` (same rules as `orders.py`: no I/O, no state). The bars window starts at the Nth-previous trading session, taken from one `get_calendar` call. `Strategy` adds thin delegates and a lazily built `Indicators` accessor that fetches bars through `get_historical_prices` and runs pandas-ta-classic on them. Nothing is cached.

**Tech Stack:** Python 3.14, `uv`, `alpaca-py` 0.44 (`alpaca.data`), pandas 3 (already installed through alpaca-py), `pandas-ta-classic` 0.6, `pytest`, `ruff`, `uv check`.

**Spec:** `docs/superpowers/specs/2026-09-11-data-layer-design.md`. Read it before starting any task, **including §12 (amendments made while planning), which overrides the sections it names**.

## Global Constraints

- Python `>=3.14`, package manager `uv`. The only new dependency is `pandas-ta-classic>=0.6.52,<0.7` (added in Task 7). pandas is already installed (both alpaca-py and pandas-ta-classic require it). Don't import numpy directly.
- Money is `Decimal`. Scalar prices (`get_last_price(s)`, `Quote.bid/ask/bid_size/ask_size`) are converted with `orders._to_decimal` (through `str`, never `Decimal(float)`). The one new float boundary is `Bars.df` (float64 OHLCV), built only in `market_data._bars_frame`.
- Stocks only. No `quote=`/`exchange=` parameters. Timesteps are exactly `"minute"` and `"day"`; anything else raises `ValueError`.
- Every Alpaca data request sets `feed=DataFeed.IEX`. Bar requests also set `adjustment=Adjustment.ALL`.
- `brokers/alpaca/market_data.py` is pure (no I/O, no state, no client instances). It is the only module that imports `alpaca.data.requests`. `core/`, `entities/` and `brokers/base.py` never import `alpaca`. `entities/` and `core/` never import pandas at import time (only under `TYPE_CHECKING` or inside functions).
- Broker data methods wrap every client exception in `BrokerError` (`raise BrokerError(...) from exc`). "No data" is `None` or a missing dict entry, never an exception. `Strategy` facade methods don't catch `BrokerError`.
- No caching anywhere in this layer.
- Tests never touch the network. They use hand-written fakes from `tests/fakes.py` built from real alpaca-py objects, not `MagicMock`. (The one exception: `tests/brokers/alpaca/test_client.py` already patches SDK constructors with `MagicMock`; its new test follows that file's pattern.)
- Test file basenames must be unique across the whole repo: there is no `__init__.py` under `tests/`.
- Every task ends green on `uv run pytest`, `uv run ruff check` and `uv check`. Ruff allows 200-character lines, but the code stays near 100; match the surrounding file.
- Commit once per task with the message `Task N: <summary>`, ending with the attribution trailer lines from your instructions. Work on branch `feature/data-layer`.

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/trading_agent_framework/core/` (was `strategies/`) | 1 | Package rename only |
| `src/trading_agent_framework/entities/quote.py` | 2 | `Quote` (bid/ask/sizes/timestamp, `.mid`) |
| `src/trading_agent_framework/entities/bars.py` | 2 | `Bars` (asset, timestep, float64 OHLCV `df`) |
| `src/trading_agent_framework/brokers/alpaca/market_data.py` | 3, 4 | Pure: timesteps, fetch window, requests (3); response parsing (4) |
| `src/trading_agent_framework/brokers/alpaca/client.py` | 5 | `+ build_stock_data_client` |
| `src/trading_agent_framework/brokers/alpaca/broker.py` | 5 | `AlpacaBroker` market-data I/O |
| `src/trading_agent_framework/brokers/base.py` | 6 | Abstract market-data methods |
| `src/trading_agent_framework/core/strategy.py` | 6, 7 | Facade methods (6); `indicators` property (7) |
| `src/trading_agent_framework/core/indicators.py` | 7 | `Indicators`, `IndicatorRow` |
| `tests/fakes.py` | 4, 5, 6 | Data factories (4), `FakeStockHistoricalDataClient` (5), `FakeBroker` data + `make_bars_frame` (6) |
| `scripts/tests/smoke_alpaca_data.py` | 8 | Manual live-feed check |
| `CLAUDE.md` | 1, 8 | Rename (1); new modules and rules (8) |

---

### Task 1: Rename the `strategies` package to `core`

**Files:**
- Move: `src/trading_agent_framework/strategies/` → `src/trading_agent_framework/core/`
- Move: `tests/strategies/` → `tests/core/`
- Modify: every file importing `trading_agent_framework.strategies` (`core/__init__.py`, `core/strategy.py`, `core/executor.py`, the six test files, `scripts/tests/smoke_strategy_paper.py`)
- Modify: `CLAUDE.md` (Architecture bullet)

**Interfaces:**
- Produces: package `trading_agent_framework.core` exporting `Strategy` and `StrategyExecutor`, with modules `core.strategy`, `core.executor`, `core.events` and `core.timing`. Every class and function name stays the same.

- [ ] **Step 1: Point the package re-export test at `core`**

In `tests/strategies/test_runners.py`, replace the import line

```python
from trading_agent_framework import strategies
```

with

```python
from trading_agent_framework import core
```

and replace the test at the bottom of the file:

```python
def test_core_package_reexports() -> None:
    assert core.Strategy is strategy_module.Strategy
    assert core.StrategyExecutor is executor_module.StrategyExecutor
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/strategies/test_runners.py -q`
Expected: collection error, `ImportError: cannot import name 'core' from 'trading_agent_framework'`.

- [ ] **Step 3: Move both directories**

```bash
git mv src/trading_agent_framework/strategies src/trading_agent_framework/core
git mv tests/strategies tests/core
```

- [ ] **Step 4: Rewrite the imports**

```bash
grep -rl "trading_agent_framework\.strategies" src tests scripts \
  | xargs sed -i 's/trading_agent_framework\.strategies/trading_agent_framework.core/g'
```

- [ ] **Step 5: Remove leftover cache-only directories**

`git mv` moves tracked files only, so the old paths may survive holding just `__pycache__`. Python would then treat `trading_agent_framework.strategies` as an empty namespace package. Check that nothing but caches is left:

```bash
find src/trading_agent_framework/strategies tests/strategies -type f -not -path "*/__pycache__/*" 2>/dev/null
```

Expected: no output. Then delete the leftovers:

```bash
rm -rf src/trading_agent_framework/strategies tests/strategies
```

- [ ] **Step 6: Re-sort imports**

`config` < `core` < `entities` alphabetically, so the renamed imports move in the isort order:

```bash
uv run ruff check --fix
uv run ruff check
```

Expected: `All checks passed!`

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q && uv check`
Expected: every test passes, with the same test count as before the rename, and `uv check` is clean.

- [ ] **Step 8: Confirm no stale references remain**

Run: `grep -rn "strategies" src tests scripts --include=*.py`
Expected: only prose, e.g. the `Strategy` docstring "Base class for strategies, ...". No imports and no module paths.

- [ ] **Step 9: Update `CLAUDE.md`**

In the Architecture list, replace the bullet that starts with `` - `strategies/` -- `` so that it starts with `` - `core/` -- `` instead. Keep the rest of the bullet unchanged.

- [ ] **Step 10: Commit**

```bash
git add -A src/trading_agent_framework/core tests/core scripts/tests/smoke_strategy_paper.py CLAUDE.md
git status --short
git commit -m "Task 1: rename the strategies package to core"
```

`git status --short` should list only renames (`R`) and modifications (`M`) under those paths.

---

### Task 2: `Quote` and `Bars` entities

**Files:**
- Create: `src/trading_agent_framework/entities/quote.py`
- Create: `src/trading_agent_framework/entities/bars.py`
- Modify: `src/trading_agent_framework/entities/__init__.py`
- Test: `tests/entities/test_quote.py`, `tests/entities/test_bars.py`, `tests/entities/test_entities_public_api.py`

**Interfaces:**
- Produces: `Quote(asset: Asset, bid: Decimal | None, ask: Decimal | None, bid_size: Decimal | None, ask_size: Decimal | None, timestamp: datetime)`, frozen, with property `mid -> Decimal | None`.
- Produces: `Bars(asset: Asset, timestep: str, df: pd.DataFrame)`, frozen, `eq=False`. `df` has a tz-aware `America/New_York` index (oldest first) and float64 `open, high, low, close, volume` columns.

- [ ] **Step 1: Write the failing tests**

`tests/entities/test_quote.py`:

```python
from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote

_TS = datetime(2026, 9, 10, 13, 30, tzinfo=UTC)


def _quote(bid: Decimal | None, ask: Decimal | None) -> Quote:
    return Quote(
        asset=Asset("AAPL"),
        bid=bid,
        ask=ask,
        bid_size=Decimal(3),
        ask_size=Decimal(4),
        timestamp=_TS,
    )


def test_mid_is_the_average_of_bid_and_ask() -> None:
    assert _quote(Decimal("100.10"), Decimal("100.20")).mid == Decimal("100.15")


@pytest.mark.parametrize(
    ("bid", "ask"),
    [(None, Decimal("100.20")), (Decimal("100.10"), None), (None, None)],
)
def test_mid_is_none_without_both_sides(bid: Decimal | None, ask: Decimal | None) -> None:
    assert _quote(bid, ask).mid is None


def test_quote_is_frozen() -> None:
    quote = _quote(Decimal(1), Decimal(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        quote.bid = Decimal(5)  # ty: ignore[invalid-assignment]
```

`tests/entities/test_bars.py`:

```python
from __future__ import annotations

import subprocess
import sys

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars


def _frame() -> pd.DataFrame:
    index = pd.date_range(
        "2026-09-01", periods=3, freq="1D", tz="America/New_York", name="timestamp"
    )
    close = [1.0, 2.0, 3.0]
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": [10.0] * 3},
        index=index,
    )


def test_bars_hold_asset_timestep_and_frame() -> None:
    df = _frame()
    bars = Bars(asset=Asset("AAPL"), timestep="day", df=df)

    assert bars.asset == Asset("AAPL")
    assert bars.timestep == "day"
    assert bars.df is df


def test_bars_compare_by_identity_not_by_frame() -> None:
    df = _frame()
    first = Bars(asset=Asset("AAPL"), timestep="day", df=df)
    second = Bars(asset=Asset("AAPL"), timestep="day", df=df.copy())

    # A dataclass-generated __eq__ would compare the DataFrames and raise
    # "The truth value of a DataFrame is ambiguous".
    assert first == first
    assert first != second


def test_importing_entities_does_not_import_pandas() -> None:
    """Subprocess: other tests in this session have already imported pandas."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.entities\n"
            "import sys\n"
            "assert 'pandas' not in sys.modules\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

In `tests/entities/test_entities_public_api.py`, add these two imports next to the other module imports:

```python
from trading_agent_framework.entities import bars as bars_module
from trading_agent_framework.entities import quote as quote_module
```

and these two lines at the end of `test_entities_reexports_match_source_modules`:

```python
    assert entities.Bars is bars_module.Bars
    assert entities.Quote is quote_module.Quote
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/entities -q`
Expected: collection errors, `ModuleNotFoundError: No module named 'trading_agent_framework.entities.quote'` (and `.bars`).

- [ ] **Step 3: Implement the entities**

`src/trading_agent_framework/entities/quote.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.entities.asset import Asset


@dataclass(frozen=True, slots=True)
class Quote:
    """Latest bid/ask for an asset. A side is None when the book is empty on that side."""

    asset: Asset
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    timestamp: datetime

    @property
    def mid(self) -> Decimal | None:
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2
```

`src/trading_agent_framework/entities/bars.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_agent_framework.entities.asset import Asset

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True, eq=False)
class Bars:
    """OHLCV history for one asset, oldest first, indexed in market time (America/New_York).

    The columns are float64 on purpose, as the codebase's second float boundary: bars feed
    indicator maths, never order sizing. `eq=False` because a generated `__eq__` would compare
    the DataFrames, which raises.
    """

    asset: Asset
    timestep: str
    df: pd.DataFrame
```

In `src/trading_agent_framework/entities/__init__.py`, add the imports (alphabetical position among the existing ones)

```python
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
```

and add `"Bars"` and `"Quote"` to `__all__`, keeping it alphabetical (`"Bars"` after `"AssetType"`, `"Quote"` after `"PositionSide"`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/entities -q`
Expected: all pass.

- [ ] **Step 5: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add src/trading_agent_framework/entities tests/entities
git commit -m "Task 2: add Quote and Bars entities"
```

---

### Task 3: `market_data.py`: timesteps, fetch window and requests

**Files:**
- Create: `src/trading_agent_framework/brokers/alpaca/market_data.py`
- Test: `tests/brokers/alpaca/test_market_data_requests.py`

**Interfaces:**
- Consumes: `MarketSession(open, close)` from `trading_agent_framework.clock`; `BrokerError`.
- Produces (all in `trading_agent_framework.brokers.alpaca.market_data`):
  - constants `MARKET_TZ: ZoneInfo`, `FEED = DataFeed.IEX`, `ADJUSTMENT = Adjustment.ALL`, `MAX_SYMBOLS_PER_REQUEST = 150`, `TIMESTEPS = ("minute", "day")`
  - `class AlpacaStockDataClient(Protocol)` with `get_stock_bars(request_params: StockBarsRequest) -> BarSet`, `get_stock_latest_trade(request_params: StockLatestTradeRequest) -> dict[str, Trade]`, `get_stock_latest_quote(request_params: StockLatestQuoteRequest) -> dict[str, Quote]`
  - `parse_timestep(timestep: str) -> TimeFrame`
  - `sessions_needed(length: int, timestep: str) -> int`
  - `calendar_lookback_start(end: datetime, length: int, timestep: str) -> date`
  - `bars_start(end: datetime, length: int, timestep: str, sessions: Sequence[MarketSession]) -> datetime`
  - `chunk_assets(assets: Iterable[Asset]) -> Iterator[list[Asset]]`
  - `build_bars_request(assets: Sequence[Asset], timestep: str, start: datetime, end: datetime) -> StockBarsRequest`
  - `build_latest_trade_request(assets: Sequence[Asset]) -> StockLatestTradeRequest`
  - `build_latest_quote_request(assets: Sequence[Asset]) -> StockLatestQuoteRequest`

Two alpaca-py facts the tests rely on: `TimeFrame` has no `__eq__` (compare `.value`, `"1Min"`/`"1Day"`), and request models store `start`/`end` as **naive UTC**.

- [ ] **Step 1: Write the failing tests**

`tests/brokers/alpaca/test_market_data_requests.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from alpaca.data.enums import Adjustment, DataFeed
from tests.fakes import ET, et, weekday_sessions

from trading_agent_framework.brokers.alpaca.market_data import (
    MAX_SYMBOLS_PER_REQUEST,
    bars_start,
    build_bars_request,
    build_latest_quote_request,
    build_latest_trade_request,
    calendar_lookback_start,
    chunk_assets,
    parse_timestep,
    sessions_needed,
)
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.errors import BrokerError

_END = et(2026, 9, 10, 9, 31)  # Thursday, one minute after the open
_SESSIONS = weekday_sessions(date(2026, 9, 8), 3)  # Tue 8, Wed 9, Thu 10


@pytest.mark.parametrize(("timestep", "value"), [("minute", "1Min"), ("day", "1Day")])
def test_parse_timestep(timestep: str, value: str) -> None:
    assert parse_timestep(timestep).value == value


@pytest.mark.parametrize("timestep", ["hour", "5minute", "Day", ""])
def test_parse_timestep_rejects_everything_else(timestep: str) -> None:
    with pytest.raises(ValueError, match="Unsupported timestep"):
        parse_timestep(timestep)


@pytest.mark.parametrize(
    ("length", "timestep", "expected"),
    [(20, "day", 21), (1, "day", 2), (30, "minute", 2), (390, "minute", 2), (391, "minute", 3)],
)
def test_sessions_needed_adds_one_for_a_partial_current_session(
    length: int, timestep: str, expected: int
) -> None:
    assert sessions_needed(length, timestep) == expected


def test_sessions_needed_rejects_a_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length"):
        sessions_needed(0, "day")


def test_sessions_needed_rejects_an_unknown_timestep() -> None:
    with pytest.raises(ValueError, match="Unsupported timestep"):
        sessions_needed(5, "hour")


@pytest.mark.parametrize(
    ("length", "timestep", "expected"),
    [(30, "minute", date(2026, 8, 28)), (20, "day", date(2026, 7, 30))],
)
def test_calendar_lookback_leaves_room_for_weekends_and_holidays(
    length: int, timestep: str, expected: date
) -> None:
    # ceil(sessions * 1.5) + 10 calendar days: 2 sessions -> 13 days, 21 sessions -> 42 days
    assert calendar_lookback_start(_END, length, timestep) == expected


def test_bars_start_is_midnight_of_the_earliest_session_needed() -> None:
    assert bars_start(_END, 30, "minute", _SESSIONS) == datetime(2026, 9, 9, tzinfo=ET)


def test_bars_start_falls_back_to_the_earliest_session_available() -> None:
    assert bars_start(_END, 10, "day", _SESSIONS) == datetime(2026, 9, 8, tzinfo=ET)


def test_bars_start_ignores_sessions_after_end() -> None:
    sessions = weekday_sessions(date(2026, 9, 8), 5)  # through Monday the 14th
    assert bars_start(_END, 1, "day", sessions) == datetime(2026, 9, 9, tzinfo=ET)


def test_bars_start_without_any_past_session_raises() -> None:
    with pytest.raises(BrokerError, match="no session"):
        bars_start(_END, 1, "day", weekday_sessions(date(2026, 9, 14), 2))


def test_chunk_assets_dedupes_and_batches() -> None:
    assets = [Asset(f"S{i}") for i in range(MAX_SYMBOLS_PER_REQUEST * 2 + 1)] + [Asset("S0")]

    chunks = list(chunk_assets(assets))

    assert [len(chunk) for chunk in chunks] == [150, 150, 1]
    assert chunks[0][0] == Asset("S0")


def test_bars_request_asks_for_adjusted_iex_bars_of_every_symbol() -> None:
    start = datetime(2026, 9, 9, tzinfo=ET)

    request = build_bars_request([Asset("AAPL"), Asset("MSFT")], "minute", start, _END)

    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.timeframe.value == "1Min"
    assert request.feed == DataFeed.IEX
    assert request.adjustment == Adjustment.ALL
    # alpaca-py normalises start/end to naive UTC
    assert request.start == start.astimezone(UTC).replace(tzinfo=None)
    assert request.end == _END.astimezone(UTC).replace(tzinfo=None)


@pytest.mark.parametrize("build", [build_latest_trade_request, build_latest_quote_request])
def test_latest_requests_use_the_iex_feed(build) -> None:
    request = build([Asset("AAPL"), Asset("MSFT")])

    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.feed == DataFeed.IEX
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_requests.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.alpaca.market_data'`.

- [ ] **Step 3: Implement**

`src/trading_agent_framework/brokers/alpaca/market_data.py`:

```python
"""Pure Alpaca market-data translation: timesteps, the fetch window, requests, responses.

Same rules as `orders.py` and `account.py`: no I/O, no state, no client instances. The only
module allowed to import `alpaca.data.requests`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame

from trading_agent_framework.clock import MarketSession
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.errors import BrokerError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.data.models import Quote as AlpacaQuote
    from alpaca.data.models import Trade as AlpacaTrade

MARKET_TZ = ZoneInfo("America/New_York")
FEED = DataFeed.IEX
ADJUSTMENT = Adjustment.ALL  # split- and dividend-adjusted, like lumibot's default
MAX_SYMBOLS_PER_REQUEST = 150
TIMESTEPS = ("minute", "day")
_MINUTES_PER_SESSION = 390


class AlpacaStockDataClient(Protocol):
    """What `AlpacaBroker` calls on a `StockHistoricalDataClient` (or a test fake).

    Narrower than the SDK's `... | RawData` return types: this project never enables raw_data.
    """

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...
    def get_stock_latest_trade(
        self, request_params: StockLatestTradeRequest
    ) -> dict[str, AlpacaTrade]: ...
    def get_stock_latest_quote(
        self, request_params: StockLatestQuoteRequest
    ) -> dict[str, AlpacaQuote]: ...


def parse_timestep(timestep: str) -> TimeFrame:
    if timestep == "minute":
        return TimeFrame.Minute
    if timestep == "day":
        return TimeFrame.Day
    raise ValueError(f"Unsupported timestep {timestep!r}; expected one of {TIMESTEPS}")


def sessions_needed(length: int, timestep: str) -> int:
    """Trading sessions to fetch for `length` bars; the +1 covers a partial current session."""
    parse_timestep(timestep)
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}")
    if timestep == "day":
        return length + 1
    return math.ceil(length / _MINUTES_PER_SESSION) + 1


def calendar_lookback_start(end: datetime, length: int, timestep: str) -> date:
    """First day of the calendar request: comfortably more calendar days than sessions needed."""
    days = math.ceil(sessions_needed(length, timestep) * 1.5) + 10
    return (end.astimezone(MARKET_TZ) - timedelta(days=days)).date()


def bars_start(
    end: datetime, length: int, timestep: str, sessions: Sequence[MarketSession]
) -> datetime:
    """Midnight (market time) of the earliest session needed, so its pre-market bars count."""
    needed = sessions_needed(length, timestep)
    today = end.astimezone(MARKET_TZ).date()
    past = [s for s in sessions if s.open.astimezone(MARKET_TZ).date() <= today]
    if not past:
        raise BrokerError(f"The Alpaca calendar has no session on or before {today}")
    first = past[max(0, len(past) - needed)]
    return datetime.combine(first.open.astimezone(MARKET_TZ).date(), time(0), tzinfo=MARKET_TZ)


def chunk_assets(assets: Iterable[Asset]) -> Iterator[list[Asset]]:
    """Unique assets, in order, in batches small enough for one Alpaca request."""
    unique = list(dict.fromkeys(assets))
    for i in range(0, len(unique), MAX_SYMBOLS_PER_REQUEST):
        yield unique[i : i + MAX_SYMBOLS_PER_REQUEST]


def _symbols(assets: Sequence[Asset]) -> list[str]:
    return [asset.symbol for asset in assets]


def build_bars_request(
    assets: Sequence[Asset], timestep: str, start: datetime, end: datetime
) -> StockBarsRequest:
    return StockBarsRequest(
        symbol_or_symbols=_symbols(assets),
        timeframe=parse_timestep(timestep),
        start=start,
        end=end,
        feed=FEED,
        adjustment=ADJUSTMENT,
    )


def build_latest_trade_request(assets: Sequence[Asset]) -> StockLatestTradeRequest:
    return StockLatestTradeRequest(symbol_or_symbols=_symbols(assets), feed=FEED)


def build_latest_quote_request(assets: Sequence[Asset]) -> StockLatestQuoteRequest:
    return StockLatestQuoteRequest(symbol_or_symbols=_symbols(assets), feed=FEED)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_requests.py -q`
Expected: all pass.

- [ ] **Step 5: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add src/trading_agent_framework/brokers/alpaca/market_data.py tests/brokers/alpaca/test_market_data_requests.py
git commit -m "Task 3: add pure market-data timesteps, fetch window and request builders"
```

---

### Task 4: `market_data.py`: response parsing

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/market_data.py`
- Modify: `tests/fakes.py` (data factories)
- Test: `tests/brokers/alpaca/test_market_data_parse.py`

**Interfaces:**
- Consumes: `Bars`, `Quote` (Task 2); `orders._field`, `orders._to_decimal`; `MARKET_TZ` (Task 3).
- Produces:
  - `parse_bars(barset: object, assets: Sequence[Asset], timestep: str, length: int, *, sessions: Sequence[MarketSession] | None = None) -> dict[Asset, Bars]`
  - `parse_latest_trades(response: Mapping[str, object], assets: Sequence[Asset]) -> dict[Asset, Decimal | None]`
  - `parse_quote(response: Mapping[str, object], asset: Asset) -> Quote | None`
  - in `tests/fakes.py`: `bar_payload(timestamp: str, close: float, volume: float = 1000.0) -> dict[str, object]`, `make_alpaca_barset(bars: dict[str, list[dict[str, object]]]) -> BarSet`, `make_alpaca_trade(symbol="AAPL", price=100.15, timestamp="2026-09-10T13:30:00Z") -> Trade`, `make_alpaca_quote(symbol="AAPL", bid=100.1, ask=100.2, bid_size=3.0, ask_size=4.0, timestamp="2026-09-10T13:30:00Z") -> Quote`

alpaca-py builds `BarSet`, `Bar`, `Trade` and `Quote` from the raw API payload (short keys such as `"t"`, `"o"`, `"bp"`), which is why the factories take raw values.

- [ ] **Step 1: Add the data factories to `tests/fakes.py`**

Add an import next to the existing `alpaca.trading.models` import:

```python
import alpaca.data.models as alpaca_data_models
```

Add these functions after `make_api_error`:

```python
def bar_payload(timestamp: str, close: float, volume: float = 1000.0) -> dict[str, object]:
    """One bar as Alpaca's API sends it; `BarSet` is built from these raw payloads."""
    return {
        "t": timestamp,
        "o": close - 0.5,
        "h": close + 1.0,
        "l": close - 1.0,
        "c": close,
        "v": volume,
        "n": 10,
        "vw": close,
    }


def make_alpaca_barset(bars: dict[str, list[dict[str, object]]]) -> alpaca_data_models.BarSet:
    return alpaca_data_models.BarSet(bars)


def make_alpaca_trade(
    symbol: str = "AAPL", price: float = 100.15, timestamp: str = "2026-09-10T13:30:00Z"
) -> alpaca_data_models.Trade:
    payload = {"t": timestamp, "x": "V", "p": price, "s": 50, "i": 1, "c": ["@"], "z": "C"}
    return alpaca_data_models.Trade(symbol, payload)


def make_alpaca_quote(
    symbol: str = "AAPL",
    bid: float = 100.1,
    ask: float = 100.2,
    bid_size: float = 3.0,
    ask_size: float = 4.0,
    timestamp: str = "2026-09-10T13:30:00Z",
) -> alpaca_data_models.Quote:
    payload = {
        "t": timestamp,
        "bp": bid,
        "bs": bid_size,
        "bx": "V",
        "ap": ask,
        "as": ask_size,
        "ax": "V",
        "c": ["R"],
        "z": "C",
    }
    return alpaca_data_models.Quote(symbol, payload)
```

- [ ] **Step 2: Write the failing tests**

`tests/brokers/alpaca/test_market_data_parse.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal

from tests.fakes import (
    bar_payload,
    et,
    make_alpaca_barset,
    make_alpaca_quote,
    make_alpaca_trade,
    make_session,
)

from trading_agent_framework.brokers.alpaca.market_data import (
    parse_bars,
    parse_latest_trades,
    parse_quote,
)
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars

AAPL = Asset("AAPL")
MSFT = Asset("MSFT")

# Thursday 2026-09-10, daylight saving time: 13:30Z is 09:30 in New York.
_MINUTES = [
    bar_payload("2026-09-10T13:29:00Z", 99.0),  # 09:29, pre-market
    bar_payload("2026-09-10T13:30:00Z", 100.0),  # 09:30
    bar_payload("2026-09-10T13:31:00Z", 101.0),  # 09:31
    bar_payload("2026-09-10T20:00:00Z", 102.0),  # 16:00, after hours
]
_SESSION = [make_session(date(2026, 9, 10))]


def test_day_bars_become_a_float_frame_indexed_in_market_time() -> None:
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-09-09T04:00:00Z", 10.0), bar_payload("2026-09-10T04:00:00Z", 11.0)]}
    )

    bars = parse_bars(barset, [AAPL], "day", 5)[AAPL]

    assert isinstance(bars, Bars)
    assert (bars.asset, bars.timestep) == (AAPL, "day")
    assert list(bars.df.columns) == ["open", "high", "low", "close", "volume"]
    assert all(str(dtype) == "float64" for dtype in bars.df.dtypes)
    assert list(bars.df["close"]) == [10.0, 11.0]
    assert bars.df.index[0] == et(2026, 9, 9)
    assert str(bars.df.index.tz) == "America/New_York"


def test_only_the_last_length_bars_are_kept_oldest_first() -> None:
    barset = make_alpaca_barset({"AAPL": list(reversed(_MINUTES))})

    bars = parse_bars(barset, [AAPL], "minute", 2)[AAPL]

    assert list(bars.df["close"]) == [101.0, 102.0]


def test_duplicate_timestamps_keep_the_first_bar() -> None:
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-09-10T13:30:00Z", 100.0), bar_payload("2026-09-10T13:30:00Z", 555.0)]}
    )

    assert list(parse_bars(barset, [AAPL], "minute", 5)[AAPL].df["close"]) == [100.0]


def test_sessions_filter_extended_hours_before_truncating() -> None:
    barset = make_alpaca_barset({"AAPL": _MINUTES})

    bars = parse_bars(barset, [AAPL], "minute", 2, sessions=_SESSION)[AAPL]

    # Truncating first would keep [101, 102] and then filter down to [101].
    assert list(bars.df["close"]) == [100.0, 101.0]


def test_without_sessions_extended_hours_bars_are_kept() -> None:
    bars = parse_bars(make_alpaca_barset({"AAPL": _MINUTES}), [AAPL], "minute", 10)[AAPL]

    assert len(bars.df) == 4


def test_early_close_sessions_drop_bars_after_the_early_close() -> None:
    # Friday 2026-11-27 closes at 13:00 in New York (standard time: 18:00Z).
    early = [make_session(date(2026, 11, 27), close_at=time(13, 0))]
    barset = make_alpaca_barset(
        {"AAPL": [bar_payload("2026-11-27T17:59:00Z", 1.0), bar_payload("2026-11-27T18:05:00Z", 2.0)]}
    )

    bars = parse_bars(barset, [AAPL], "minute", 5, sessions=early)[AAPL]

    assert list(bars.df["close"]) == [1.0]


def test_assets_without_bars_are_left_out() -> None:
    barset = make_alpaca_barset({"AAPL": _MINUTES, "MSFT": []})

    assert set(parse_bars(barset, [AAPL, MSFT, Asset("TSLA")], "minute", 5)) == {AAPL}


def test_assets_filtered_down_to_nothing_are_left_out() -> None:
    barset = make_alpaca_barset({"AAPL": [_MINUTES[0]]})  # pre-market only

    assert parse_bars(barset, [AAPL], "minute", 5, sessions=_SESSION) == {}


def test_latest_trades_become_decimal_prices_and_missing_symbols_none() -> None:
    response = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    assert parse_latest_trades(response, [AAPL, MSFT]) == {AAPL: Decimal("100.15"), MSFT: None}


def test_quote_maps_bid_ask_sizes_and_timestamp() -> None:
    quote = parse_quote({"AAPL": make_alpaca_quote(bid=100.1, ask=100.2)}, AAPL)

    assert quote is not None
    assert quote.asset == AAPL
    assert (quote.bid, quote.ask) == (Decimal("100.1"), Decimal("100.2"))
    assert (quote.bid_size, quote.ask_size) == (Decimal(3), Decimal(4))
    assert quote.timestamp == datetime(2026, 9, 10, 13, 30, tzinfo=UTC)
    assert str(quote.timestamp.tzinfo) == "America/New_York"


def test_an_empty_book_side_is_none_not_zero() -> None:
    quote = parse_quote({"AAPL": make_alpaca_quote(bid=0.0, ask=100.2)}, AAPL)

    assert quote is not None
    assert quote.bid is None
    assert quote.ask == Decimal("100.2")
    assert quote.mid is None


def test_no_quote_for_the_symbol_is_none() -> None:
    assert parse_quote({}, AAPL) is None
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_parse.py -q`
Expected: collection error, `ImportError: cannot import name 'parse_bars'`.

- [ ] **Step 4: Implement the parsers**

In `market_data.py`, extend the imports to:

```python
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame

from trading_agent_framework.brokers.alpaca.orders import _field, _to_decimal
from trading_agent_framework.clock import MarketSession
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError
```

add `_OHLCV = ("open", "high", "low", "close", "volume")` next to the other constants, and append:

```python
# --- response parsing ------------------------------------------------------------


def parse_bars(
    barset: object,
    assets: Sequence[Asset],
    timestep: str,
    length: int,
    *,
    sessions: Sequence[MarketSession] | None = None,
) -> dict[Asset, Bars]:
    """The last `length` bars per asset, oldest first. Assets without bars are left out.

    When `sessions` is given, only bars inside them are kept (so early closes are handled).
    This filter runs before truncating, so the caller gets `length` in-session bars
    whenever the feed has that many.
    """
    data = cast(Mapping[str, Sequence[object]], _field(barset, "data") or {})
    result: dict[Asset, Bars] = {}
    for asset in assets:
        df = _bars_frame(data.get(asset.symbol) or [])
        if sessions is not None:
            df = _within_sessions(df, sessions)
        df = df.iloc[-length:]
        if not df.empty:
            result[asset] = Bars(asset=asset, timestep=timestep, df=df)
    return result


def _bars_frame(rows: Sequence[object]) -> pd.DataFrame:
    """The float64 boundary for bars (see `entities/bars.py`): indicators want native floats."""
    index = pd.to_datetime([_field(row, "timestamp") for row in rows], utc=True)
    columns = {name: [float(cast(float, _field(row, name))) for row in rows] for name in _OHLCV}
    df = pd.DataFrame(columns, index=index.tz_convert(MARKET_TZ), dtype="float64")
    df.index.name = "timestamp"
    return df[~df.index.duplicated(keep="first")].sort_index()


def _within_sessions(df: pd.DataFrame, sessions: Sequence[MarketSession]) -> pd.DataFrame:
    keep = pd.Series(False, index=df.index)
    for session in sessions:
        keep |= (df.index >= session.open) & (df.index < session.close)
    return df[keep]


def parse_latest_trades(
    response: Mapping[str, object], assets: Sequence[Asset]
) -> dict[Asset, Decimal | None]:
    """Last traded price per asset; None when Alpaca returned no trade for the symbol."""
    return {asset: _to_decimal(_field(response.get(asset.symbol), "price")) for asset in assets}


def parse_quote(response: Mapping[str, object], asset: Asset) -> Quote | None:
    raw = response.get(asset.symbol)
    if raw is None:
        return None
    return Quote(
        asset=asset,
        bid=_book_price(_field(raw, "bid_price")),
        ask=_book_price(_field(raw, "ask_price")),
        bid_size=_to_decimal(_field(raw, "bid_size")),
        ask_size=_to_decimal(_field(raw, "ask_size")),
        timestamp=cast(datetime, _field(raw, "timestamp")).astimezone(MARKET_TZ),
    )


def _book_price(value: object) -> Decimal | None:
    """Alpaca reports an empty book side as 0: that means "no price", not a price of zero."""
    price = _to_decimal(value)
    return price if price is not None and price > 0 else None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_parse.py tests/brokers/alpaca/test_market_data_requests.py -q`
Expected: all pass.

- [ ] **Step 6: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add src/trading_agent_framework/brokers/alpaca/market_data.py tests/fakes.py tests/brokers/alpaca/test_market_data_parse.py
git commit -m "Task 4: parse Alpaca bars, latest trades and quotes"
```

---

### Task 5: `AlpacaBroker` market data

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/client.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/broker.py`
- Modify: `tests/fakes.py` (`FakeStockHistoricalDataClient`)
- Test: `tests/brokers/alpaca/test_client.py`, `tests/brokers/alpaca/test_broker_market_data.py`

**Interfaces:**
- Consumes: everything from Tasks 3–4; `account.build_calendar_request(start: date, end: date)`, `account.parse_calendar(responses, tz) -> list[MarketSession]`; `FakeTradingClient.calendar_response` / `.calendar_requests` / `.raises`; `make_alpaca_calendar(day)`.
- Produces:
  - `build_stock_data_client(creds: AlpacaCredentials) -> StockHistoricalDataClient`
  - `AlpacaBroker.__init__(..., *, clock=None, is_paper=True, data_client: market_data.AlpacaStockDataClient | None = None)`. `from_credentials` passes `build_stock_data_client(creds)`.
  - `AlpacaBroker.get_last_price(asset: Asset) -> Decimal | None`
  - `AlpacaBroker.get_last_prices(assets: Sequence[Asset]) -> dict[Asset, Decimal | None]`
  - `AlpacaBroker.get_quote(asset: Asset) -> Quote | None`
  - `AlpacaBroker.get_bars(assets: Sequence[Asset], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]`
  - `tests.fakes.FakeStockHistoricalDataClient` with `bars: dict[str, list[dict]]` (raw payloads), `trades: dict[str, Trade]`, `quotes: dict[str, Quote]`, `bars_requests`, `trade_requests`, `quote_requests`, `raises: dict[str, BaseException]`

- [ ] **Step 1: Add `FakeStockHistoricalDataClient` to `tests/fakes.py`**

Add to the imports:

```python
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
```

Add after `FakeTradingClient`:

```python
def _requested_symbols(request: StockBarsRequest | StockLatestTradeRequest | StockLatestQuoteRequest) -> list[str]:
    symbols = request.symbol_or_symbols
    return [symbols] if isinstance(symbols, str) else list(symbols)


class FakeStockHistoricalDataClient:
    """A hand-written stand-in for `alpaca.data.historical.StockHistoricalDataClient`.

    Answers from `bars` (raw payloads, see `bar_payload`), `trades` and `quotes`, all keyed by
    symbol and filtered to the requested symbols. Like Alpaca, a symbol without data is simply
    absent from the response.
    """

    def __init__(self) -> None:
        self.bars: dict[str, list[dict[str, object]]] = {}
        self.trades: dict[str, alpaca_data_models.Trade] = {}
        self.quotes: dict[str, alpaca_data_models.Quote] = {}
        self.bars_requests: list[StockBarsRequest] = []
        self.trade_requests: list[StockLatestTradeRequest] = []
        self.quote_requests: list[StockLatestQuoteRequest] = []
        self.raises: dict[str, BaseException] = {}

    def _maybe_raise(self, method: str) -> None:
        error = self.raises.get(method)
        if error is not None:
            raise error

    def get_stock_bars(self, request_params: StockBarsRequest) -> alpaca_data_models.BarSet:
        self.bars_requests.append(request_params)
        self._maybe_raise("get_stock_bars")
        wanted = _requested_symbols(request_params)
        return alpaca_data_models.BarSet({s: self.bars[s] for s in wanted if s in self.bars})

    def get_stock_latest_trade(
        self, request_params: StockLatestTradeRequest
    ) -> dict[str, alpaca_data_models.Trade]:
        self.trade_requests.append(request_params)
        self._maybe_raise("get_stock_latest_trade")
        wanted = _requested_symbols(request_params)
        return {s: self.trades[s] for s in wanted if s in self.trades}

    def get_stock_latest_quote(
        self, request_params: StockLatestQuoteRequest
    ) -> dict[str, alpaca_data_models.Quote]:
        self.quote_requests.append(request_params)
        self._maybe_raise("get_stock_latest_quote")
        wanted = _requested_symbols(request_params)
        return {s: self.quotes[s] for s in wanted if s in self.quotes}
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/brokers/alpaca/test_client.py` (add `build_stock_data_client` to its import from `client`):

```python
def test_build_stock_data_client_uses_the_same_credentials(monkeypatch) -> None:
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)
    mock_data_client = MagicMock()
    monkeypatch.setattr("alpaca.data.historical.StockHistoricalDataClient", mock_data_client)

    result = build_stock_data_client(creds)

    mock_data_client.assert_called_once_with(api_key="key", secret_key="secret")
    assert result is mock_data_client.return_value
```

`tests/brokers/alpaca/test_broker_market_data.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal

import pytest
from alpaca.data.enums import DataFeed
from tests.fakes import (
    FakeClock,
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    bar_payload,
    et,
    make_alpaca_calendar,
    make_alpaca_quote,
    make_alpaca_trade,
)

from trading_agent_framework.brokers.alpaca import broker as broker_module
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError

AAPL = Asset("AAPL")
MSFT = Asset("MSFT")
_NOW = et(2026, 9, 10, 9, 32)  # Thursday, two minutes after the open
_MINUTES = {
    "AAPL": [
        bar_payload("2026-09-10T13:29:00Z", 99.0),  # 09:29, pre-market
        bar_payload("2026-09-10T13:30:00Z", 100.0),
        bar_payload("2026-09-10T13:31:00Z", 101.0),
    ]
}


def _broker(
    data: FakeStockHistoricalDataClient | None, trading: FakeTradingClient | None = None
) -> AlpacaBroker:
    return AlpacaBroker(
        "momentum", trading or FakeTradingClient(), clock=FakeClock(_NOW), data_client=data
    )


def _trading_with_calendar() -> FakeTradingClient:
    trading = FakeTradingClient()
    trading.calendar_response = [make_alpaca_calendar("2026-09-09"), make_alpaca_calendar("2026-09-10")]
    return trading


def _data_with_minutes() -> FakeStockHistoricalDataClient:
    data = FakeStockHistoricalDataClient()
    data.bars = _MINUTES
    return data


# --- last prices and quotes ------------------------------------------------------


def test_get_last_prices_fetches_every_symbol_in_one_iex_request() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    prices = _broker(data).get_last_prices([AAPL, MSFT])

    assert prices == {AAPL: Decimal("100.15"), MSFT: None}
    [request] = data.trade_requests
    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.feed == DataFeed.IEX


def test_get_last_prices_splits_large_universes_into_150_symbol_requests() -> None:
    data = FakeStockHistoricalDataClient()

    prices = _broker(data).get_last_prices([Asset(f"S{i}") for i in range(151)])

    assert len(prices) == 151
    assert [len(r.symbol_or_symbols) for r in data.trade_requests] == [150, 1]


def test_get_last_price_returns_the_single_price() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 99.5)}

    assert _broker(data).get_last_price(AAPL) == Decimal("99.5")


def test_get_quote_returns_bid_and_ask_from_the_iex_feed() -> None:
    data = FakeStockHistoricalDataClient()
    data.quotes = {"AAPL": make_alpaca_quote(bid=100.1, ask=100.2)}

    quote = _broker(data).get_quote(AAPL)

    assert isinstance(quote, Quote)
    assert quote.mid == Decimal("100.15")
    [request] = data.quote_requests
    assert request.feed == DataFeed.IEX


def test_get_quote_is_none_when_alpaca_has_no_quote() -> None:
    assert _broker(FakeStockHistoricalDataClient()).get_quote(AAPL) is None


# --- bars ------------------------------------------------------------------------


def test_get_bars_starts_the_window_at_the_earliest_session_needed() -> None:
    data = _data_with_minutes()
    trading = _trading_with_calendar()

    _broker(data, trading).get_bars([AAPL, MSFT], 30, "minute")

    [calendar_request] = trading.calendar_requests
    assert (calendar_request.start, calendar_request.end) == (date(2026, 8, 28), date(2026, 9, 10))
    [bars_request] = data.bars_requests
    assert bars_request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert bars_request.timeframe.value == "1Min"
    # Midnight 2026-09-09 and 09:32 2026-09-10 in New York, stored by alpaca-py as naive UTC.
    assert bars_request.start == datetime(2026, 9, 9, 4, 0)
    assert bars_request.end == datetime(2026, 9, 10, 13, 32)


def test_get_bars_keeps_extended_hours_by_default() -> None:
    result = _broker(_data_with_minutes(), _trading_with_calendar()).get_bars([AAPL], 30, "minute")

    assert list(result[AAPL].df["close"]) == [99.0, 100.0, 101.0]


def test_get_bars_drops_extended_hours_on_request() -> None:
    broker = _broker(_data_with_minutes(), _trading_with_calendar())

    result = broker.get_bars([AAPL], 30, "minute", include_after_hours=False)

    assert list(result[AAPL].df["close"]) == [100.0, 101.0]


def test_get_bars_leaves_out_assets_without_data() -> None:
    broker = _broker(_data_with_minutes(), _trading_with_calendar())

    assert set(broker.get_bars([AAPL, MSFT], 30, "minute")) == {AAPL}


def test_get_bars_rejects_an_unknown_timestep_before_any_request() -> None:
    data = _data_with_minutes()
    trading = _trading_with_calendar()

    with pytest.raises(ValueError, match="Unsupported timestep"):
        _broker(data, trading).get_bars([AAPL], 5, "hour")

    assert trading.calendar_requests == []
    assert data.bars_requests == []


# --- failures --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "call", "match"),
    [
        ("get_stock_latest_trade", lambda b: b.get_last_prices([AAPL]), "latest trades"),
        ("get_stock_latest_quote", lambda b: b.get_quote(AAPL), "latest quote"),
        ("get_stock_bars", lambda b: b.get_bars([AAPL], 5, "day"), "bars"),
    ],
)
def test_data_client_failures_become_broker_errors(
    method: str, call: Callable[[AlpacaBroker], object], match: str
) -> None:
    data = FakeStockHistoricalDataClient()
    data.raises[method] = RuntimeError("boom")

    with pytest.raises(BrokerError, match=match):
        call(_broker(data, _trading_with_calendar()))


def test_calendar_failures_become_broker_errors() -> None:
    trading = FakeTradingClient()
    trading.raises["get_calendar"] = RuntimeError("boom")

    with pytest.raises(BrokerError, match="calendar"):
        _broker(FakeStockHistoricalDataClient(), trading).get_bars([AAPL], 5, "day")


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.get_last_price(AAPL),
        lambda b: b.get_last_prices([AAPL]),
        lambda b: b.get_quote(AAPL),
        lambda b: b.get_bars([AAPL], 5, "day"),
    ],
)
def test_market_data_without_a_data_client_raises(call: Callable[[AlpacaBroker], object]) -> None:
    with pytest.raises(BrokerError, match="no market data client"):
        call(_broker(None))


def test_from_credentials_wires_the_stock_data_client(monkeypatch: pytest.MonkeyPatch) -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 101.0)}
    monkeypatch.setattr(broker_module, "build_trading_client", lambda creds: FakeTradingClient())
    monkeypatch.setattr(broker_module, "build_stock_data_client", lambda creds: data)
    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)

    broker = AlpacaBroker.from_credentials("momentum", creds, with_stream=False)

    assert broker.get_last_price(AAPL) == Decimal("101.0")
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_client.py tests/brokers/alpaca/test_broker_market_data.py -q`
Expected: collection errors, `ImportError: cannot import name 'build_stock_data_client'` (and `TypeError: ... unexpected keyword argument 'data_client'` once that import exists).

- [ ] **Step 4: Implement the client factory**

In `client.py`, add under `TYPE_CHECKING`:

```python
    from alpaca.data.historical import StockHistoricalDataClient
```

update the module docstring's "Both functions" to "All three functions", and append:

```python
def build_stock_data_client(creds: AlpacaCredentials) -> StockHistoricalDataClient:
    """Market data uses the trading credentials; paper and live share one data endpoint."""
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(api_key=creds.api_key, secret_key=creds.api_secret)
```

- [ ] **Step 5: Implement the broker methods**

In `broker.py`:

1. Change the imports:

```python
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, cast

from alpaca.common.exceptions import APIError

from trading_agent_framework.brokers.alpaca import account, market_data, orders
from trading_agent_framework.brokers.alpaca.client import (
    build_stock_data_client,
    build_trading_client,
    build_trading_stream,
)
from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError
```

2. Add the `data_client` keyword to `__init__` and store it:

```python
    def __init__(
        self,
        strategy_name: str,
        client: orders.AlpacaTradingClient,
        tracker: OrderTracker | None = None,
        stream: TradingStream | None = None,
        *,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        data_client: market_data.AlpacaStockDataClient | None = None,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(client),
            is_paper=is_paper,
        )
        self._client = client
        self._data_client = data_client
        self._stream = stream
        self._alpaca_stream: AlpacaTradeStream | None = None
```

3. In `from_credentials`, build the data client (the same `cast` reasoning as the trading client applies):

```python
        client = cast("orders.AlpacaTradingClient", build_trading_client(creds))
        data_client = cast("market_data.AlpacaStockDataClient", build_stock_data_client(creds))
        stream = build_trading_stream(creds) if with_stream else None
        return cls(
            strategy_name, client, stream=stream, is_paper=creds.is_paper, data_client=data_client
        )
```

4. Add the market-data section after `sync_open_orders`:

```python
    # --- market data -------------------------------------------------------------------

    def _require_data_client(self) -> market_data.AlpacaStockDataClient:
        if self._data_client is None:
            raise BrokerError(
                "no market data client configured; construct the broker with data_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return self._data_client

    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self.get_last_prices([asset])[asset]

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        client = self._require_data_client()
        prices: dict[Asset, Decimal | None] = {}
        for chunk in market_data.chunk_assets(assets):
            request = market_data.build_latest_trade_request(chunk)
            try:
                response = client.get_stock_latest_trade(request)
            except Exception as exc:
                raise BrokerError(
                    f"Failed to fetch latest trades ({len(chunk)} symbols): {exc}"
                ) from exc
            prices.update(market_data.parse_latest_trades(response, chunk))
        return prices

    def get_quote(self, asset: Asset) -> Quote | None:
        client = self._require_data_client()
        request = market_data.build_latest_quote_request([asset])
        try:
            response = client.get_stock_latest_quote(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the latest quote for {asset.symbol}: {exc}") from exc
        return market_data.parse_quote(response, asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        client = self._require_data_client()
        end = self.clock.now()
        sessions = self._sessions_before(end, length, timestep)
        start = market_data.bars_start(end, length, timestep, sessions)
        in_session = None if include_after_hours else sessions
        bars: dict[Asset, Bars] = {}
        for chunk in market_data.chunk_assets(assets):
            request = market_data.build_bars_request(chunk, timestep, start, end)
            try:
                barset = client.get_stock_bars(request)
            except Exception as exc:
                raise BrokerError(
                    f"Failed to fetch {timestep} bars ({len(chunk)} symbols): {exc}"
                ) from exc
            bars.update(market_data.parse_bars(barset, chunk, timestep, length, sessions=in_session))
        return bars

    def _sessions_before(self, end: datetime, length: int, timestep: str) -> list[MarketSession]:
        first_day = market_data.calendar_lookback_start(end, length, timestep)
        last_day = end.astimezone(market_data.MARKET_TZ).date()
        request = account.build_calendar_request(first_day, last_day)
        try:
            days = self._client.get_calendar(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca calendar: {exc}") from exc
        return account.parse_calendar(days, market_data.MARKET_TZ)
```

`calendar_lookback_start` validates `length` and `timestep` first, so a bad argument raises `ValueError` before any request goes out.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers -q`
Expected: all pass, including the existing broker tests. Existing tests build `AlpacaBroker` without `data_client`, which is fine because it's optional.

- [ ] **Step 7: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add src/trading_agent_framework/brokers/alpaca/client.py src/trading_agent_framework/brokers/alpaca/broker.py tests/fakes.py tests/brokers/alpaca/test_client.py tests/brokers/alpaca/test_broker_market_data.py
git commit -m "Task 5: fetch last prices, quotes and calendar-windowed bars in AlpacaBroker"
```

---

### Task 6: `Broker` interface, `FakeBroker` and the `Strategy` facade

**Files:**
- Modify: `src/trading_agent_framework/brokers/base.py`
- Modify: `src/trading_agent_framework/core/strategy.py`
- Modify: `tests/fakes.py` (`FakeBroker` data, `make_bars_frame`)
- Test: `tests/brokers/test_base.py`, `tests/core/test_strategy_market_data.py`

**Interfaces:**
- Consumes: `AlpacaBroker`'s four methods (Task 5) already match the abstract signatures below.
- Produces:
  - abstract `Broker.get_last_price(asset: Asset) -> Decimal | None`, `get_last_prices(assets: Sequence[Asset]) -> dict[Asset, Decimal | None]`, `get_quote(asset: Asset) -> Quote | None`, `get_bars(assets: Sequence[Asset], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]`
  - `Strategy.get_last_price(asset: Asset | str) -> Decimal | None`, `get_last_prices(assets: Iterable[Asset | str]) -> dict[Asset, Decimal | None]`, `get_quote(asset: Asset | str) -> Quote | None`, `get_historical_prices(asset: Asset | str, length: int, timestep: str = "day", *, include_after_hours: bool = True) -> Bars | None`, `get_historical_prices_for_assets(assets: Iterable[Asset | str], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]`
  - `tests.fakes.make_bars_frame(closes: Sequence[float], *, start: datetime | None = None, freq: str = "1D") -> pd.DataFrame` (high = close + 1, low = close − 1, volume 1000)
  - `FakeBroker` fields `last_prices: dict[str, Decimal]`, `quotes: dict[str, Quote]`, `bar_frames: dict[str, pd.DataFrame]`, `bars_calls: list[tuple[tuple[str, ...], int, str, bool]]`, `market_data_error: BrokerError | None`. `get_bars` returns the last `length` rows of `bar_frames[symbol]`.

- [ ] **Step 1: Write the failing interface test**

Append to `tests/brokers/test_base.py`:

```python
def test_market_data_methods_are_part_of_the_broker_interface() -> None:
    expected = {"get_last_price", "get_last_prices", "get_quote", "get_bars"}
    assert expected <= Broker.__abstractmethods__
```

- [ ] **Step 2: Write the failing facade tests**

`tests/core/test_strategy_market_data.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def test_get_last_price_accepts_a_symbol() -> None:
    strategy, broker = _strategy()
    broker.last_prices["SPY"] = Decimal("450.10")

    assert strategy.get_last_price("SPY") == Decimal("450.10")


def test_get_last_price_is_none_without_data() -> None:
    strategy, _ = _strategy()

    assert strategy.get_last_price("SPY") is None


def test_get_last_prices_keys_results_by_asset() -> None:
    strategy, broker = _strategy()
    broker.last_prices = {"SPY": Decimal("450.10"), "QQQ": Decimal("380.5")}

    prices = strategy.get_last_prices(["SPY", Asset("QQQ"), "TLT"])

    assert prices == {
        Asset("SPY"): Decimal("450.10"),
        Asset("QQQ"): Decimal("380.5"),
        Asset("TLT"): None,
    }


def test_get_quote_delegates_to_the_broker() -> None:
    strategy, broker = _strategy()
    quote = Quote(
        asset=Asset("SPY"),
        bid=Decimal(1),
        ask=Decimal(2),
        bid_size=None,
        ask_size=None,
        timestamp=datetime(2026, 9, 14, 14, tzinfo=UTC),
    )
    broker.quotes["SPY"] = quote

    assert strategy.get_quote("SPY") is quote


def test_get_historical_prices_returns_the_last_length_bars() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["AAPL"] = make_bars_frame([1, 2, 3, 4])

    bars = strategy.get_historical_prices("AAPL", 2, "day")

    assert bars is not None
    assert bars.asset == Asset("AAPL")
    assert list(bars.df["close"]) == [3.0, 4.0]
    assert broker.bars_calls == [(("AAPL",), 2, "day", True)]


def test_get_historical_prices_is_none_without_data() -> None:
    strategy, _ = _strategy()

    assert strategy.get_historical_prices("AAPL", 2) is None


def test_get_historical_prices_for_assets_forwards_every_option() -> None:
    strategy, broker = _strategy()
    broker.bar_frames = {"AAPL": make_bars_frame([1, 2]), "MSFT": make_bars_frame([3, 4])}

    result = strategy.get_historical_prices_for_assets(
        ["AAPL", Asset("MSFT"), "TLT"], 2, "minute", include_after_hours=False
    )

    assert set(result) == {Asset("AAPL"), Asset("MSFT")}
    assert broker.bars_calls == [(("AAPL", "MSFT", "TLT"), 2, "minute", False)]


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get_last_price("SPY"),
        lambda s: s.get_last_prices(["SPY"]),
        lambda s: s.get_quote("SPY"),
        lambda s: s.get_historical_prices("SPY", 5),
        lambda s: s.get_historical_prices_for_assets(["SPY"], 5),
    ],
)
def test_broker_errors_reach_the_strategy(call: Callable[[Strategy], object]) -> None:
    strategy, broker = _strategy()
    broker.market_data_error = BrokerError("feed down")

    with pytest.raises(BrokerError, match="feed down"):
        call(strategy)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/brokers/test_base.py tests/core/test_strategy_market_data.py -q`
Expected: the interface test fails with `assert {...} <= frozenset({...})`, and the facade file fails to import `make_bars_frame`.

- [ ] **Step 4: Add the abstract methods to `Broker`**

In `brokers/base.py`, add the imports

```python
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
```

and, after `sync_open_orders`, add:

```python
    # --- market data -------------------------------------------------------------------

    @abstractmethod
    def get_last_price(self, asset: Asset) -> Decimal | None:
        """Last traded price; None when the data source has no trade for the asset."""

    @abstractmethod
    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]: ...

    @abstractmethod
    def get_quote(self, asset: Asset) -> Quote | None: ...

    @abstractmethod
    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        """The last `length` bars per asset, oldest first; assets without data are left out."""
```

Update the module docstring's list of what subclasses provide to include "market data".

- [ ] **Step 5: Implement the fake market data in `tests/fakes.py`**

Add the imports:

```python
from collections.abc import Callable, Iterable, Sequence

import pandas as pd

from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError
```

Add after `weekday_sessions`:

```python
def make_bars_frame(
    closes: Sequence[float], *, start: datetime | None = None, freq: str = "1D"
) -> pd.DataFrame:
    """An OHLCV frame shaped like `Bars.df`: high = close + 1, low = close - 1."""
    index = pd.date_range(
        start if start is not None else et(2026, 1, 5),
        periods=len(closes),
        freq=freq,
        name="timestamp",
    )
    close = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=index,
    )
```

In `FakeBroker.__init__`, add:

```python
        self.last_prices: dict[str, Decimal] = {}
        self.quotes: dict[str, Quote] = {}
        self.bar_frames: dict[str, pd.DataFrame] = {}
        self.bars_calls: list[tuple[tuple[str, ...], int, str, bool]] = []
        self.market_data_error: BrokerError | None = None
```

and add these methods to `FakeBroker` (before `start_stream`):

```python
    def _check_market_data(self) -> None:
        if self.market_data_error is not None:
            raise self.market_data_error

    def get_last_price(self, asset: Asset) -> Decimal | None:
        self._check_market_data()
        return self.last_prices.get(asset.symbol)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        self._check_market_data()
        return {asset: self.last_prices.get(asset.symbol) for asset in assets}

    def get_quote(self, asset: Asset) -> Quote | None:
        self._check_market_data()
        return self.quotes.get(asset.symbol)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        self._check_market_data()
        requested = list(assets)
        symbols = tuple(asset.symbol for asset in requested)
        self.bars_calls.append((symbols, length, timestep, include_after_hours))
        return {
            asset: Bars(asset=asset, timestep=timestep, df=self.bar_frames[asset.symbol].iloc[-length:])
            for asset in requested
            if asset.symbol in self.bar_frames
        }
```

(`Callable` and `Iterable` were already imported: extend the existing `collections.abc` import line rather than duplicating it.)

- [ ] **Step 6: Add the `Strategy` facade**

In `core/strategy.py`, add the imports

```python
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
```

and add this section between the accounting section and the trading section:

```python
    # --- market data -----------------------------------------------------------------

    def get_last_price(self, asset: Asset | str) -> Decimal | None:
        """Last traded price; for the quote midpoint use `get_quote(asset).mid`."""
        return self.broker.get_last_price(_to_asset(asset))

    def get_last_prices(self, assets: Iterable[Asset | str]) -> dict[Asset, Decimal | None]:
        return self.broker.get_last_prices([_to_asset(asset) for asset in assets])

    def get_quote(self, asset: Asset | str) -> Quote | None:
        return self.broker.get_quote(_to_asset(asset))

    def get_historical_prices(
        self,
        asset: Asset | str,
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> Bars | None:
        """The last `length` "minute" or "day" bars, oldest first; None without data."""
        target = _to_asset(asset)
        bars = self.broker.get_bars(
            [target], length, timestep, include_after_hours=include_after_hours
        )
        return bars.get(target)

    def get_historical_prices_for_assets(
        self,
        assets: Iterable[Asset | str],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self.broker.get_bars(
            [_to_asset(asset) for asset in assets],
            length,
            timestep,
            include_after_hours=include_after_hours,
        )
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/test_base.py tests/core -q`
Expected: all pass.

- [ ] **Step 8: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add src/trading_agent_framework/brokers/base.py src/trading_agent_framework/core/strategy.py tests/fakes.py tests/brokers/test_base.py tests/core/test_strategy_market_data.py
git commit -m "Task 6: add market data to the Broker interface and the Strategy facade"
```

---

### Task 7: Indicators

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Create: `src/trading_agent_framework/core/indicators.py`
- Modify: `src/trading_agent_framework/core/strategy.py` (`indicators` property)
- Test: `tests/core/test_indicators.py`

**Interfaces:**
- Consumes: `Strategy.get_historical_prices` (Task 6); `FakeBroker.bar_frames` / `.bars_calls`, `make_bars_frame` (Task 6).
- Produces:
  - `Strategy.indicators -> Indicators` (built once per strategy)
  - `Indicators.<pandas_ta_name>(asset: Asset | str, timestep: str = "day", **kwargs) -> float | IndicatorRow | None`. The reserved kwargs `bars` (bar count to fetch) and `include_after_hours` are consumed here; everything else goes to pandas-ta.
  - `Indicators.custom(name: str, fn: Callable[..., object], asset: Asset | str, timestep: str = "day", **kwargs) -> float | IndicatorRow | None`
  - `IndicatorRow(values: Mapping[str, float | None])` with attribute access (`.` and `-` normalised to `_`), `__getitem__`, `as_dict()`
  - `default_bars(kwargs: Mapping[str, Any]) -> int`, which is `max(50, int(kwargs.get("length", 50)) * 3)`

pandas-ta-classic facts verified on Python 3.14 + pandas 3.0.5 (0.6.52): module-level functions accept `open/high/low/close/volume` Series as keyword arguments and ignore the ones they don't need. With too few bars, `sma` returns **`None`** (not a NaN Series). `bbands(length=20, std=2)` returns the columns `BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0`. On closes 1…30 with high = close + 1 and low = close − 1: `sma(length=5)` = 28.0, `rsi(length=14)` = 100.0, `atr(length=14)` = 2.0.

- [ ] **Step 1: Add the dependency**

```bash
uv add "pandas-ta-classic>=0.6.52,<0.7"
uv run python -c "import pandas_ta_classic; print('ok')"
```

Expected: `pyproject.toml` `dependencies` now lists `pandas-ta-classic>=0.6.52,<0.7`, and the import prints `ok`.

- [ ] **Step 2: Write the failing tests**

`tests/core/test_indicators.py`:

```python
from __future__ import annotations

import subprocess
import sys

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.core.indicators import IndicatorRow, Indicators, default_bars
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset

_CLOSES = [float(c) for c in range(1, 31)]  # 1.0 .. 30.0, steadily rising


def _strategy(closes: list[float] = _CLOSES) -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    broker.bar_frames["SPY"] = make_bars_frame(closes)
    return Strategy(broker), broker


def test_strategy_builds_one_indicators_accessor() -> None:
    strategy, _ = _strategy()

    assert isinstance(strategy.indicators, Indicators)
    assert strategy.indicators is strategy.indicators


def test_single_column_indicators_return_the_latest_value() -> None:
    strategy, _ = _strategy()

    assert strategy.indicators.sma("SPY", length=5) == pytest.approx(28.0)
    assert strategy.indicators.rsi(Asset("SPY"), length=14) == pytest.approx(100.0)


def test_indicators_receive_every_ohlcv_column() -> None:
    strategy, _ = _strategy()

    assert strategy.indicators.atr("SPY", length=14) == pytest.approx(2.0)


def test_multi_column_indicators_return_an_indicator_row() -> None:
    strategy, _ = _strategy()

    row = strategy.indicators.bbands("SPY", length=20, std=2)

    assert isinstance(row, IndicatorRow)
    assert row.BBM_20_2_0 == pytest.approx(20.5)  # mean of closes 11..30
    assert row["BBM_20_2.0"] == pytest.approx(20.5)
    assert set(row.as_dict()) == {"BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0"}


@pytest.mark.parametrize(("kwargs", "expected"), [({"length": 40}, 120), ({"length": 5}, 50), ({}, 150)])
def test_default_bars_is_three_times_the_length_with_a_floor_of_50(
    kwargs: dict[str, int], expected: int
) -> None:
    assert default_bars(kwargs) == expected


def test_the_default_lookback_decides_how_many_bars_are_fetched() -> None:
    strategy, broker = _strategy()

    strategy.indicators.sma("SPY", length=40)
    strategy.indicators.macd("SPY")

    assert [call[1] for call in broker.bars_calls] == [120, 150]


def test_bars_and_include_after_hours_control_the_fetch() -> None:
    strategy, broker = _strategy()

    value = strategy.indicators.sma(
        "SPY", timestep="minute", length=5, bars=10, include_after_hours=False
    )

    assert value == pytest.approx(28.0)  # the last 10 closes still end at 26..30
    assert broker.bars_calls == [(("SPY",), 10, "minute", False)]


def test_too_few_bars_is_none() -> None:
    strategy, _ = _strategy(_CLOSES[:3])

    assert strategy.indicators.sma("SPY", length=5) is None


def test_a_nan_latest_value_is_none() -> None:
    strategy, _ = _strategy()

    slow = strategy.indicators.custom("slow", lambda df: df["close"].rolling(100).mean(), "SPY")

    assert slow is None


def test_no_bars_is_none() -> None:
    strategy, _ = _strategy()

    assert strategy.indicators.sma("AAPL", length=5) is None


def test_custom_indicators_get_the_bars_frame_and_their_kwargs() -> None:
    strategy, _ = _strategy()

    value = strategy.indicators.custom(
        "shifted", lambda df, offset: df["close"] - offset, "SPY", offset=5
    )

    assert value == pytest.approx(25.0)


def test_custom_rejects_a_non_callable() -> None:
    strategy, _ = _strategy()

    with pytest.raises(TypeError, match="callable"):
        strategy.indicators.custom("bad", 42, "SPY")  # ty: ignore[invalid-argument-type]


def test_custom_indicators_must_return_pandas() -> None:
    strategy, _ = _strategy()

    with pytest.raises(TypeError, match="Series or DataFrame"):
        strategy.indicators.custom("scalar", lambda df: 1.0, "SPY")


def test_an_unknown_indicator_name_raises_attribute_error() -> None:
    strategy, _ = _strategy()

    with pytest.raises(AttributeError, match="no indicator named 'not_an_indicator'"):
        strategy.indicators.not_an_indicator  # noqa: B018


def test_indicator_row_normalises_names_and_rejects_unknown_ones() -> None:
    row = IndicatorRow({"BBL_20_2.0": 1.5, "MACD-x": None})

    assert row.BBL_20_2_0 == 1.5
    assert row.MACD_x is None
    with pytest.raises(AttributeError):
        row.missing  # noqa: B018


def test_importing_core_imports_neither_pandas_nor_alpaca() -> None:
    """Subprocess: other tests in this session have already imported both."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import trading_agent_framework.core\n"
            "import sys\n"
            "assert 'pandas' not in sys.modules, 'pandas'\n"
            "assert 'alpaca' not in sys.modules, 'alpaca'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/core/test_indicators.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.core.indicators'`.

- [ ] **Step 4: Implement `core/indicators.py`**

```python
"""Technical indicators computed with pandas-ta-classic over the strategy's own bars.

`strategy.indicators.sma(asset, length=200)` fetches recent bars through
`strategy.get_historical_prices`, runs the pandas-ta-classic function of the same name, and
returns its latest value. There is no cache and no look-ahead guard: every call re-fetches,
and the latest bar is the current one (live and paper trading only, for now).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.asset import Asset

if TYPE_CHECKING:
    import pandas as pd

    from trading_agent_framework.core.strategy import Strategy

_OHLCV = ("open", "high", "low", "close", "volume")
_MIN_BARS = 50
_BARS_PER_LENGTH = 3


def default_bars(kwargs: Mapping[str, Any]) -> int:
    """Bars fetched when the caller passes no `bars`: 3x the indicator's own `length`
    (warm-up for smoothed indicators), never fewer than 50."""
    return max(_MIN_BARS, int(kwargs.get("length", _MIN_BARS)) * _BARS_PER_LENGTH)


class IndicatorRow:
    """Latest row of a multi-column indicator: `row.BBL_20_2_0` or `row["BBL_20_2.0"]`."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, float | None]) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> float | None:
        if name.startswith("_"):
            raise AttributeError(name)
        for key, value in self._values.items():
            if name in (key, key.replace(".", "_").replace("-", "_")):
                return value
        raise AttributeError(f"no column {name!r}; columns are {list(self._values)}")

    def __getitem__(self, key: str) -> float | None:
        return self._values[key]

    def as_dict(self) -> dict[str, float | None]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"IndicatorRow({self._values!r})"


type IndicatorValue = float | IndicatorRow | None


class Indicators:
    """`strategy.indicators`: any pandas-ta-classic indicator by name, plus `custom`."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def __getattr__(self, name: str) -> Callable[..., IndicatorValue]:
        if name.startswith("_"):
            raise AttributeError(name)
        function = getattr(_pandas_ta(), name, None)
        if not callable(function):
            raise AttributeError(
                f"pandas_ta_classic has no indicator named {name!r}; "
                "use indicators.custom(name, fn, asset, ...) for your own"
            )

        def compute(df: pd.DataFrame, **kwargs: Any) -> object:
            return function(**{column: df[column] for column in _OHLCV}, **kwargs)

        def indicator(asset: Asset | str, timestep: str = "day", **kwargs: Any) -> IndicatorValue:
            return self._evaluate(name, compute, asset, timestep, kwargs)

        indicator.__name__ = name
        return indicator

    def custom(
        self,
        name: str,
        fn: Callable[..., object],
        asset: Asset | str,
        timestep: str = "day",
        **kwargs: Any,
    ) -> IndicatorValue:
        """Run `fn(df, **kwargs) -> Series | DataFrame` over the fetched bars."""
        if not callable(fn):
            raise TypeError(f"custom indicator {name!r}: fn must be callable, got {type(fn).__name__}")
        return self._evaluate(name, fn, asset, timestep, kwargs)

    def _evaluate(
        self,
        name: str,
        compute: Callable[..., object],
        asset: Asset | str,
        timestep: str,
        kwargs: Mapping[str, Any],
    ) -> IndicatorValue:
        options = dict(kwargs)
        count = options.pop("bars", None)
        include_after_hours = options.pop("include_after_hours", True)
        if count is None:
            count = default_bars(options)
        bars = self._strategy.get_historical_prices(
            asset, count, timestep, include_after_hours=include_after_hours
        )
        if bars is None or bars.df.empty:
            return None
        return _latest(name, compute(bars.df, **options))


def _pandas_ta() -> Any:
    import pandas_ta_classic  # deferred: keeps `import trading_agent_framework.core` light

    return pandas_ta_classic


def _latest(name: str, result: object) -> IndicatorValue:
    import pandas as pd

    if result is None:  # pandas-ta returns None when there are too few bars
        return None
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return None
        return IndicatorRow({str(column): _number(v) for column, v in result.iloc[-1].items()})
    if isinstance(result, pd.Series):
        return None if result.empty else _number(result.iloc[-1])
    raise TypeError(
        f"indicator {name!r} returned {type(result).__name__}; expected a pandas Series or DataFrame"
    )


def _number(value: object) -> float | None:
    import pandas as pd

    return None if pd.isna(value) else float(value)  # ty: ignore[invalid-argument-type]
```

(Drop the `ty: ignore` comment if `uv check` doesn't need it.)

- [ ] **Step 5: Add the `indicators` property to `Strategy`**

In `core/strategy.py`, add the import

```python
from trading_agent_framework.core.indicators import Indicators
```

in `__init__`, before `self.executor = StrategyExecutor(self)`, add

```python
        self._indicators: Indicators | None = None
```

and add this property directly after the `is_backtesting` property:

```python
    @property
    def indicators(self) -> Indicators:
        """pandas-ta-classic indicators over this strategy's bars, e.g. `indicators.sma(a, length=20)`."""
        if self._indicators is None:
            self._indicators = Indicators(self)
        return self._indicators
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/core -q`
Expected: all pass. If pandas-ta-classic emits a warning on import, note it in the task report but don't silence it.

- [ ] **Step 7: Full check and commit**

```bash
uv run pytest -q && uv run ruff check && uv check
git add pyproject.toml uv.lock src/trading_agent_framework/core/indicators.py src/trading_agent_framework/core/strategy.py tests/core/test_indicators.py
git commit -m "Task 7: add pandas-ta-classic indicators to the strategy"
```

---

### Task 8: Manual data smoke script and documentation

**Files:**
- Create: `scripts/tests/smoke_alpaca_data.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `AlpacaBroker.from_credentials`, `Strategy` market-data methods and `indicators` (Tasks 5–7).

- [ ] **Step 1: Write the smoke script**

`scripts/tests/smoke_alpaca_data.py`:

```python
#!/usr/bin/env python3
"""Manual paper-account check of the market data layer against Alpaca's live IEX feed.

NOT part of the automated test suite: the suite never touches the network. Run it by hand:

    uv run python scripts/tests/smoke_alpaca_data.py

Credentials come from env/.env.alpaca.integration-tests, as in smoke_alpaca_orders.py. The
script is read-only (it places no orders) and works at any time of day. It fails only on things
that must always hold: SPY has a last trade, daily bars come back full-length, regular-hours
minute bars stay inside 09:30-16:00, and the indicators return values. It also prints two things
no fake can show: what Alpaca does with an unknown symbol, and how many minute bars the
include_after_hours=False filter keeps.
"""

from __future__ import annotations

import sys
from datetime import time
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.errors import ConfigurationError, TradingFrameworkError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-data"
SYMBOLS = ["SPY", "QQQ", "AAPL"]
UNKNOWN_SYMBOL = "ZZZZZZ"
DAY_BARS = 30
MINUTE_BARS = 60


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(
            f"Credentials file not found: {ENV_FILE}\n"
            "Create it with ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_IS_PAPER=true "
            "before running this script."
        )
    load_dotenv(ENV_FILE, override=True)
    try:
        creds = AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc
    if not creds.is_paper:
        raise SmokeTestFailure("ALPACA_IS_PAPER is not true in the credentials file; refusing to run.")
    return creds


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)

    price = strategy.get_last_price("SPY")
    print(f"get_last_price(SPY) = {price}")
    _check(isinstance(price, Decimal) and price > 0, "no last trade price for SPY")

    prices = strategy.get_last_prices(SYMBOLS)
    print(f"get_last_prices({SYMBOLS}) = {prices}")
    _check(all(value is not None for value in prices.values()), f"missing last prices: {prices}")

    try:
        mixed = strategy.get_last_prices([UNKNOWN_SYMBOL, "SPY"])
        print(f"unknown symbol in a batch -> {mixed}")
    except TradingFrameworkError as exc:
        print(f"unknown symbol in a batch -> raised {type(exc).__name__}: {exc}")

    quote = strategy.get_quote("SPY")
    mid = quote.mid if quote is not None else None
    print(f"get_quote(SPY) = {quote} (mid={mid}); an empty side is normal outside market hours")

    day = strategy.get_historical_prices("SPY", DAY_BARS, "day")
    _check(day is not None and len(day.df) == DAY_BARS, f"expected {DAY_BARS} daily bars for SPY")
    assert day is not None
    print(f"daily bars: {day.df.index[0]} .. {day.df.index[-1]}\n{day.df.tail(3)}")

    extended = strategy.get_historical_prices("SPY", MINUTE_BARS, "minute")
    regular = strategy.get_historical_prices(
        "SPY", MINUTE_BARS, "minute", include_after_hours=False
    )
    _check(extended is not None and regular is not None, "no minute bars for SPY")
    assert extended is not None and regular is not None
    print(f"minute bars, extended hours: {len(extended.df)} ({extended.df.index[0]} .. {extended.df.index[-1]})")
    print(f"minute bars, regular hours:  {len(regular.df)} ({regular.df.index[0]} .. {regular.df.index[-1]})")
    _check(
        all(time(9, 30) <= ts.time() < time(16, 0) for ts in regular.df.index),
        "include_after_hours=False returned bars outside 09:30-16:00",
    )

    many = strategy.get_historical_prices_for_assets(SYMBOLS, DAY_BARS, "day")
    print(f"get_historical_prices_for_assets: {({a.symbol: len(b.df) for a, b in many.items()})}")
    _check({asset.symbol for asset in many} == set(SYMBOLS), "missing daily bars in the batch")

    sma = strategy.indicators.sma("SPY", length=20)
    bbands = strategy.indicators.bbands("SPY", length=20, std=2)
    rsi = strategy.indicators.rsi("SPY", timestep="minute", length=14)
    print(f"sma(20) = {sma}\nbbands(20, 2) = {bbands}\nrsi(14, minute) = {rsi}")
    _check(sma is not None and bbands is not None and rsi is not None, "an indicator returned None")

    print("\nPASS: last prices, quote, day/minute bars, batch bars and indicators all returned data.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
```

- [ ] **Step 2: Lint it**

Run: `uv run ruff check scripts/tests/smoke_alpaca_data.py`
Expected: `All checks passed!`

- [ ] **Step 3: Update `CLAUDE.md`**

Make these exact edits:

1. In **Commands**, after the `smoke_strategy_paper.py` line, add:
   ```
   uv run python scripts/tests/smoke_alpaca_data.py       # manual paper-account smoke test: market data + indicators (read-only)
   ```
2. Replace the `entities/` bullet with:
   `` - `entities/` -- pure data: `Order`, `Position`, `Asset`, `Quote`, `Bars`, enums. No I/O, no broker knowledge (`Bars` imports pandas only for type checking). ``
3. Replace the end of the `brokers/alpaca/orders.py` bullet, "Together with `account.py`, the only modules allowed to import `alpaca.trading.requests`.", with "Together with `account.py`, the only modules allowed to import `alpaca.trading.requests` (`market_data.py` is the only one allowed to import `alpaca.data.requests`)."
4. After the `brokers/alpaca/account.py` bullet, add:
   `` - `brokers/alpaca/market_data.py` -- **pure** market-data translation (same rules as `orders.py`): timesteps (`"minute"`/`"day"` only), the calendar-based bars window, IEX request builders, and bar/trade/quote parsing. ``
5. Replace the `brokers/alpaca/broker.py` bullet with:
   `` - `brokers/alpaca/broker.py` -- `AlpacaBroker`: wires the real `TradingClient` and `StockHistoricalDataClient` I/O to the pure modules plus tracker bookkeeping. Contains no translation logic itself. ``
6. Extend the `core/` bullet with: `` `indicators.py` (`strategy.indicators.<pandas-ta name>(asset, ...)`, no cache). ``
7. Replace the **Money is `Decimal`** bullet with:
   `` - **Money is `Decimal`** everywhere except two deliberate float boundaries: `orders.py` (Alpaca's SDK wants floats for some request fields) and `Bars.df` (float64 OHLCV for indicator maths, built only in `market_data._bars_frame`). Don't add a third. ``
8. Replace "**`orders.py` and `account.py` stay pure.**" with "**`orders.py`, `account.py` and `market_data.py` stay pure.**" (rest of that bullet unchanged).
9. Add a bullet to **Key patterns / gotchas**:
   `` - **Market data:** `get_last_price` is the last *trade* (not the quote midpoint; use `get_quote(...).mid`). Every data request uses the IEX feed; bars are split/dividend-adjusted. `get_bars` makes one `get_calendar` call to start its window at the right session. Nothing is cached: every call hits the API (200 requests/min on the free IEX plan). ``

- [ ] **Step 4: Final verification**

```bash
uv run pytest -q
uv run ruff check
uv check
grep -rn "alpaca.data.requests" src | grep -v market_data.py
```

Expected: all tests pass; ruff and `uv check` are clean; the grep prints nothing.

- [ ] **Step 5: Run the smoke script if credentials exist**

If `env/.env.alpaca.integration-tests` exists, run `uv run python scripts/tests/smoke_alpaca_data.py` and include its full output in the task report. Report the two printed behaviours (unknown symbol, regular-hours minute bar count). If the file doesn't exist, don't create it: report that the user must run the script by hand.

- [ ] **Step 6: Commit**

```bash
git add scripts/tests/smoke_alpaca_data.py CLAUDE.md
git commit -m "Task 8: add the market data smoke script; document the data layer"
```

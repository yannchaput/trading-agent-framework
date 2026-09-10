# Strategy Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lean, lumibot-compatible strategy layer — lifecycle and order-event hooks, `sleeptime`-driven iterations, an accounting/trading facade, live/paper runners, and coloured lumibot-style run logs — on top of the existing broker layer.

**Architecture:** A single-threaded `StrategyExecutor` drives a `Strategy`'s hooks through an injectable `MarketClock` (never `time.sleep`/`datetime.now` directly). Order events from the Alpaca stream thread are queued by an `OrderTracker` listener and dispatched on the executor thread while it waits. `Broker` gains account, calendar-clock, replace and close capabilities; `AlpacaBroker` implements them through pure translation functions.

**Tech Stack:** Python 3.14, `uv`, `alpaca-py` 0.44, `python-dotenv`, stdlib `zoneinfo`/`logging`/`threading`, `pytest`, `ruff`, `uv check`.

**Spec:** `docs/superpowers/specs/2026-09-10-strategy-framework-design.md` — read it before starting any task.

## Global Constraints

- Python `>=3.14`, package manager `uv`. **No new dependencies** — `pyproject.toml` `dependencies` stay exactly `alpaca-py>=0.44.0,<0.45` and `python-dotenv>=1.2`. No `termcolor`, no APScheduler.
- Money is `Decimal` everywhere. The only `Decimal → float` conversion point stays `orders._to_api_number` in `brokers/alpaca/orders.py`.
- `strategies/`, `clock.py`, `log.py`, `entities/`, `config/`, `brokers/base.py`, `brokers/tracker.py` never import `alpaca`.
- Only `brokers/alpaca/orders.py` and `brokers/alpaca/account.py` may import `alpaca.trading.requests`. Both stay pure: no I/O, no state, no client instances.
- New broker methods never let a raw SDK/pydantic exception escape: request-building failures raise `OrderValidationError`, client-call failures raise `BrokerError` (`raise BrokerError(...) from exc`).
- All datetimes are tz-aware. Market time zone is `ZoneInfo("America/New_York")`.
- Log colours are hand-written ANSI escape codes and **stay in the log files** (read with VS Code's "ANSI Colors" plugin).
- Tests never touch the network and use hand-written fakes (`tests/fakes.py`), never `MagicMock`.
- Every task ends green on: `uv run pytest`, `uv run ruff check`, `uv check` (line length 100, ruff rules `E,F,I,UP,B`).
- Commit per task with message `Task N: <summary>`, ending with the attribution trailer lines from your instructions. Work happens on branch `feature/strategy-framework`.

## Spec deviations (decided while planning — keep them)

1. `MarketClock.next_session()` returns `MarketSession | None`; `None` ends the run normally. Live clocks never return `None`; fakes and the future backtest clock do.
2. The executor waits in slices of at most **60 s** (spec said 1 s). The `wake` event already interrupts waits for order events and `stop()`; the slice only bounds clock drift.
3. `next_tick(last_tick, interval, now)` takes the last tick (on the same grid as the anchor) rather than the anchor.
4. `Strategy` has no `name` argument: `strategy.name` is `broker.strategy_name`, so the `client_order_id` prefix, adopted orders and log names can never disagree.
5. `Strategy.__init__` takes `project_root: Path | None = None`, forwarded to `setup_strategy_logging` (tests point it at `tmp_path`).
6. A `BrokerError` from `clock.next_session()` is logged and retried every 60 s instead of ending the run.
7. Exceptions in any lifecycle hook (not only `on_trading_iteration`) are logged, passed to `on_bot_crash`, and trading continues.
8. `wait_for_order_execution` / `wait_for_orders_execution` return `bool` (`True` when every order reached a final status).

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/trading_agent_framework/config/env.py` | modify | add `TradingMode` StrEnum; derive `TRADING_MODES` |
| `src/trading_agent_framework/entities/account.py` | create | `AccountBalances` frozen dataclass |
| `src/trading_agent_framework/clock.py` | create | `MarketSession`, `MarketClock` ABC, `MARKET_TZ` |
| `src/trading_agent_framework/log.py` | create | `setup_strategy_logging`, `reset_strategy_logging`, `ColorLogger` |
| `src/trading_agent_framework/strategies/timing.py` | create | `SleepTime`, `parse_sleeptime`, `next_tick` (pure) |
| `src/trading_agent_framework/brokers/tracker.py` | modify | `OrderTracker.mark_replaced` |
| `src/trading_agent_framework/brokers/alpaca/orders.py` | modify | open-only orders request, replace/close request builders, close-all parsing, Protocol growth |
| `src/trading_agent_framework/brokers/alpaca/account.py` | create | pure `parse_account`, `build_calendar_request`, `parse_calendar` |
| `src/trading_agent_framework/brokers/alpaca/clock.py` | create | `AlpacaMarketClock` |
| `src/trading_agent_framework/brokers/base.py` | modify | `clock`, `is_paper`, new abstract methods, no-op stream hooks |
| `src/trading_agent_framework/brokers/alpaca/broker.py` | modify | implement new `Broker` methods |
| `src/trading_agent_framework/strategies/events.py` | create | `QueuedOrderEvent`, `OrderEventQueue` |
| `src/trading_agent_framework/strategies/strategy.py` | create | `Strategy` hooks, config, facade, runners |
| `src/trading_agent_framework/strategies/executor.py` | create | `StrategyExecutor` |
| `src/trading_agent_framework/strategies/__init__.py` | create | public exports |
| `tests/conftest.py` | create | autouse logging reset |
| `tests/fakes.py` | modify | `FakeClock`, `FakeBroker`, Alpaca account/calendar builders, fake client growth |
| `scripts/tests/smoke_strategy_paper.py` | create | manual paper smoke test |
| `CLAUDE.md` | modify | architecture + rules update |

Task order: 1 entities/config → 2 clock → 3 logging → 4 timing → 5 pure Alpaca translation + tracker → 6 `AlpacaMarketClock` → 7 `Broker` ABC + `AlpacaBroker` + `FakeBroker` → 8 event queue → 9 `Strategy` facade → 10 executor lifecycle → 11 executor events/waits/interrupts → 12 runners + exports → 13 smoke script + docs.

---

### Task 1: `TradingMode` enum and `AccountBalances` entity

**Files:**
- Modify: `src/trading_agent_framework/config/env.py` (top of module, `TRADING_MODES` definition)
- Modify: `src/trading_agent_framework/config/__init__.py`
- Create: `src/trading_agent_framework/entities/account.py`
- Modify: `src/trading_agent_framework/entities/__init__.py`
- Test: `tests/config/test_env.py`, `tests/config/test_config_public_api.py`, `tests/entities/test_account.py`, `tests/entities/test_entities_public_api.py`

**Interfaces:**
- Produces: `TradingMode(StrEnum)` with `LIVE = "live"`, `PAPER = "paper"`, `BACKTESTING = "backtesting"` in `trading_agent_framework.config.env` (re-exported from `trading_agent_framework.config`). `TRADING_MODES: frozenset[str]` unchanged in value.
- Produces: `AccountBalances(cash: Decimal, portfolio_value: Decimal, buying_power: Decimal)` — frozen, slots — in `trading_agent_framework.entities.account` (re-exported from `trading_agent_framework.entities`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/config/test_env.py`:

```python
from trading_agent_framework.config.env import TRADING_MODES, TradingMode


def test_trading_modes_are_derived_from_the_enum() -> None:
    assert TRADING_MODES == frozenset({"live", "paper", "backtesting"})
    assert {mode.value for mode in TradingMode} == TRADING_MODES


def test_trading_mode_is_a_plain_string() -> None:
    assert TradingMode("paper") is TradingMode.PAPER
    assert f"{TradingMode.LIVE}" == "live"
```

(Move the new import next to the file's existing imports so ruff's `I` rule stays happy.)

Append to `tests/config/test_config_public_api.py` inside `test_config_reexports_match_source_module`:

```python
    assert config.TradingMode is env_module.TradingMode
```

Create `tests/entities/test_account.py`:

```python
from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from trading_agent_framework.entities.account import AccountBalances


def test_account_balances_holds_decimals() -> None:
    balances = AccountBalances(
        cash=Decimal("1000.50"), portfolio_value=Decimal("2500"), buying_power=Decimal("4000")
    )
    assert balances.cash == Decimal("1000.50")
    assert balances.portfolio_value == Decimal("2500")
    assert balances.buying_power == Decimal("4000")


def test_account_balances_is_frozen() -> None:
    balances = AccountBalances(
        cash=Decimal("1"), portfolio_value=Decimal("1"), buying_power=Decimal("1")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        balances.cash = Decimal("2")  # ty: ignore[invalid-assignment]
```

Append to `tests/entities/test_entities_public_api.py`: add the import `from trading_agent_framework.entities import account as account_module` next to the other module imports, and inside the test:

```python
    assert entities.AccountBalances is account_module.AccountBalances
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/config tests/entities -q`
Expected: FAIL — `ImportError: cannot import name 'TradingMode'` and `ModuleNotFoundError: ... entities.account`.

- [ ] **Step 3: Implement**

In `src/trading_agent_framework/config/env.py`, add `from enum import StrEnum` to the imports and replace the `TRADING_MODES` line with:

```python
class TradingMode(StrEnum):
    """The three ways a strategy can run; values match the env-file suffixes."""

    LIVE = "live"
    PAPER = "paper"
    BACKTESTING = "backtesting"


TRADING_MODES: frozenset[str] = frozenset(mode.value for mode in TradingMode)
```

In `src/trading_agent_framework/config/__init__.py`, import `TradingMode` alongside the existing names and add `"TradingMode"` to `__all__` (keep `__all__` sorted).

Create `src/trading_agent_framework/entities/account.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class AccountBalances:
    """Broker account snapshot. `portfolio_value` is equity: cash + positions."""

    cash: Decimal
    portfolio_value: Decimal
    buying_power: Decimal
```

In `src/trading_agent_framework/entities/__init__.py`, add `from trading_agent_framework.entities.account import AccountBalances` and `"AccountBalances"` to `__all__` (sorted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass (175 existing + new tests).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/config src/trading_agent_framework/entities tests/config tests/entities
git commit -m "Task 1: add TradingMode enum and AccountBalances entity"
```

---

### Task 2: `MarketClock` seam and `FakeClock`

**Files:**
- Create: `src/trading_agent_framework/clock.py`
- Modify: `tests/fakes.py` (append clock helpers)
- Test: `tests/test_clock.py`

**Interfaces:**
- Produces (`trading_agent_framework.clock`):
  - `MARKET_TZ: ZoneInfo` = `ZoneInfo("America/New_York")`
  - `MarketSession(open: datetime, close: datetime)` — frozen; both tz-aware, `close > open`, else `ValueError`.
  - `MarketClock` ABC: class attribute `tz: ZoneInfo = MARKET_TZ`; `now() -> datetime` (default `datetime.now(self.tz)`); `wait(seconds: float, wake: threading.Event) -> None` (default `wake.wait(seconds)`); abstract `next_session() -> MarketSession | None` = first session whose `close > now()`, `None` = no more sessions.
- Produces (`tests/fakes.py`): `ET`, `et(year, month, day, hour=0, minute=0, second=0) -> datetime`, `make_session(day: date, open_at: time = time(9, 30), close_at: time = time(16, 0)) -> MarketSession`, `weekday_sessions(first_day: date, count: int) -> list[MarketSession]`, `FakeClock(now: datetime, sessions: Iterable[MarketSession] = ())` with attributes `sessions`, `waits: list[float]`, `on_wait: Callable[[], None] | None`, `next_session_errors: list[BaseException]`, and method `advance(seconds: float)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clock.py`:

```python
from __future__ import annotations

import threading
import time as time_module
from datetime import UTC, date, datetime, timedelta

import pytest
from tests.fakes import ET, FakeClock, et, make_session, weekday_sessions

from trading_agent_framework.clock import MARKET_TZ, MarketClock, MarketSession


class _NoSessionClock(MarketClock):
    def next_session(self) -> MarketSession | None:
        return None


def test_market_session_requires_tz_aware_times() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        MarketSession(open=datetime(2026, 9, 14, 9, 30), close=et(2026, 9, 14, 16))


def test_market_session_requires_close_after_open() -> None:
    with pytest.raises(ValueError, match="after"):
        MarketSession(open=et(2026, 9, 14, 16), close=et(2026, 9, 14, 9, 30))


def test_default_now_is_aware_in_market_time_zone() -> None:
    now = _NoSessionClock().now()
    assert now.tzinfo is MARKET_TZ
    assert abs(now - datetime.now(UTC)) < timedelta(seconds=5)


def test_default_wait_returns_early_when_woken() -> None:
    wake = threading.Event()
    wake.set()
    started = time_module.monotonic()
    _NoSessionClock().wait(30, wake)
    assert time_module.monotonic() - started < 1


def test_fake_clock_wait_advances_time() -> None:
    clock = FakeClock(et(2026, 9, 14, 9, 0))
    clock.wait(90, threading.Event())
    assert clock.now() == et(2026, 9, 14, 9, 1, 30)
    assert clock.waits == [90]


def test_fake_clock_wait_does_not_advance_when_woken() -> None:
    clock = FakeClock(et(2026, 9, 14, 9, 0))
    wake = threading.Event()
    wake.set()
    clock.wait(90, wake)
    assert clock.now() == et(2026, 9, 14, 9, 0)


def test_fake_clock_next_session_skips_closed_sessions_then_ends() -> None:
    monday, tuesday = weekday_sessions(date(2026, 9, 14), 2)
    clock = FakeClock(et(2026, 9, 14, 16, 30), [monday, tuesday])
    assert clock.next_session() == tuesday
    clock.advance(timedelta(days=1).total_seconds())
    assert clock.next_session() is None


def test_fake_clock_returns_current_session_while_open() -> None:
    session = make_session(date(2026, 9, 14))
    clock = FakeClock(et(2026, 9, 14, 11, 0), [session])
    assert clock.next_session() == session


def test_weekday_sessions_skip_weekends() -> None:
    sessions = weekday_sessions(date(2026, 9, 18), 2)  # Friday
    assert [s.open.date() for s in sessions] == [date(2026, 9, 18), date(2026, 9, 21)]
    assert sessions[0].open.tzinfo is ET
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_clock.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_agent_framework.clock'`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/clock.py`:

```python
"""Broker-agnostic market clock seam.

The strategy executor never calls `datetime.now()` or `time.sleep()` itself:
it asks a `MarketClock` what time it is, when the next session runs, and to
wait. Live clocks use the wall clock; a future backtesting clock will advance
simulated time instead, with no change to the executor.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class MarketSession:
    """One regular trading session (early closes included)."""

    open: datetime
    close: datetime

    def __post_init__(self) -> None:
        if self.open.tzinfo is None or self.close.tzinfo is None:
            raise ValueError("MarketSession open/close must be tz-aware")
        if self.close <= self.open:
            raise ValueError("MarketSession close must be after open")


class MarketClock(ABC):
    """Time source and session calendar for the strategy executor."""

    tz: ZoneInfo = MARKET_TZ

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def wait(self, seconds: float, wake: threading.Event) -> None:
        """Block for up to `seconds`, returning early as soon as `wake` is set."""
        if seconds > 0:
            wake.wait(seconds)

    @abstractmethod
    def next_session(self) -> MarketSession | None:
        """The first session whose close is after `now()`; None when there are no more."""
```

Append to `tests/fakes.py` (merge the new imports into the file's import block; `MarketClock`/`MarketSession` come from `trading_agent_framework.clock`):

```python
import threading
from collections.abc import Callable, Iterable
from datetime import date, time, timedelta
from zoneinfo import ZoneInfo

from trading_agent_framework.clock import MarketClock, MarketSession

ET = ZoneInfo("America/New_York")


def et(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """A tz-aware datetime in market (Eastern) time."""
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def make_session(
    day: date, open_at: time = time(9, 30), close_at: time = time(16, 0)
) -> MarketSession:
    return MarketSession(
        open=datetime.combine(day, open_at, tzinfo=ET),
        close=datetime.combine(day, close_at, tzinfo=ET),
    )


def weekday_sessions(first_day: date, count: int) -> list[MarketSession]:
    """`count` regular 09:30-16:00 sessions on consecutive weekdays from `first_day`."""
    sessions: list[MarketSession] = []
    day = first_day
    while len(sessions) < count:
        if day.weekday() < 5:
            sessions.append(make_session(day))
        day += timedelta(days=1)
    return sessions


class FakeClock(MarketClock):
    """Manual `MarketClock`: `wait` advances fake time instantly unless `wake` is set.

    `on_wait` runs at the start of every `wait` call -- tests use it to inject
    order events or call `stop()` from "outside" the executor loop.
    `next_session_errors` are raised (in order) by `next_session` before any
    session is returned.
    """

    def __init__(self, now: datetime, sessions: Iterable[MarketSession] = ()) -> None:
        self._now = now
        self.sessions = list(sessions)
        self.waits: list[float] = []
        self.on_wait: Callable[[], None] | None = None
        self.next_session_errors: list[BaseException] = []

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def wait(self, seconds: float, wake: threading.Event) -> None:
        self.waits.append(seconds)
        if self.on_wait is not None:
            self.on_wait()
        if not wake.is_set():
            self.advance(seconds)

    def next_session(self) -> MarketSession | None:
        if self.next_session_errors:
            raise self.next_session_errors.pop(0)
        return next((s for s in self.sessions if s.close > self._now), None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/clock.py tests/fakes.py tests/test_clock.py
git commit -m "Task 2: add MarketClock seam and FakeClock test double"
```

---

### Task 3: Logging module (`log.py`)

**Files:**
- Create: `src/trading_agent_framework/log.py`
- Create: `tests/conftest.py`
- Test: `tests/test_log.py`

**Interfaces:**
- Consumes: `TradingMode`, `find_project_root` from `trading_agent_framework.config.env`.
- Produces (`trading_agent_framework.log`):
  - constants `PACKAGE_LOGGER_NAME = "trading_agent_framework"`, `ANSI_RESET`, `ANSI_GREY`, `ANSI_BLUE`, `ANSI_YELLOW`, `ANSI_RED`, `LOG_FILE_NAMES: dict[TradingMode, str]`
  - `setup_strategy_logging(strategy_name: str, mode: TradingMode, *, project_root: Path | None = None, level: int = logging.INFO, started_at: datetime | None = None) -> Path` (returns the log file path)
  - `reset_strategy_logging() -> None`
  - `ColorLogger(logger: logging.Logger, prefix: str)` with `log_debug/log_info/log_warning/log_error/log_critical(message: object, *, stacklevel: int = 1) -> str`. `stacklevel=1` attributes the record to the caller of the method; a wrapper that delegates passes `stacklevel=2`.

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
from __future__ import annotations

from collections.abc import Iterator

import pytest

from trading_agent_framework.log import reset_strategy_logging


@pytest.fixture(autouse=True)
def _reset_strategy_logging() -> Iterator[None]:
    """setup_strategy_logging turns off propagation on the package logger, which
    would starve other tests' `caplog`; always undo it after each test."""
    yield
    reset_strategy_logging()
```

Create `tests/test_log.py`:

```python
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.log import (
    ANSI_BLUE,
    ANSI_GREY,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    PACKAGE_LOGGER_NAME,
    ColorLogger,
    reset_strategy_logging,
    setup_strategy_logging,
)

_STARTED = datetime(2026, 9, 10, 14, 30, 5)


def _setup(tmp_path: Path, **kwargs: object) -> Path:
    return setup_strategy_logging(
        "momentum",
        TradingMode.PAPER,
        project_root=tmp_path,
        started_at=_STARTED,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def _color_logger() -> ColorLogger:
    return ColorLogger(logging.getLogger(f"{PACKAGE_LOGGER_NAME}.tests"), "momentum")


def _last_line(log_file: Path) -> str:
    for handler in logging.getLogger(PACKAGE_LOGGER_NAME).handlers:
        handler.flush()
    return log_file.read_text(encoding="utf-8").splitlines()[-1]


def test_setup_creates_lumibot_style_run_directory(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    expected = tmp_path / "logs" / "momentum" / "paper" / "2026-09-10_143005_paper" / "paper.log"
    assert log_file == expected
    assert log_file.is_file()


@pytest.mark.parametrize(
    ("mode", "file_name"),
    [
        (TradingMode.LIVE, "live.log"),
        (TradingMode.PAPER, "paper.log"),
        (TradingMode.BACKTESTING, "backtest.log"),
    ],
)
def test_log_file_name_per_mode(tmp_path: Path, mode: TradingMode, file_name: str) -> None:
    log_file = setup_strategy_logging("momentum", mode, project_root=tmp_path, started_at=_STARTED)
    assert log_file.name == file_name
    assert log_file.parent.name == f"2026-09-10_143005_{mode.value}"


def test_info_line_keeps_ansi_colour_in_file(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    returned = _color_logger().log_info("hello")
    line = _last_line(log_file)
    assert returned == "hello"
    pattern = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \| INFO \| \[momentum\] "
    assert re.fullmatch(pattern + re.escape(f"{ANSI_BLUE}hello{ANSI_RESET}"), line)


def test_warning_line_names_the_calling_site(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    _color_logger().log_warning("careful")
    line = _last_line(log_file)
    assert "| WARNING | test_log.py:test_warning_line_names_the_calling_site:" in line
    assert line.endswith(f"[momentum] {ANSI_YELLOW}careful{ANSI_RESET}")


@pytest.mark.parametrize(
    ("method", "level", "color"),
    [
        ("log_debug", "DEBUG", ANSI_GREY),
        ("log_info", "INFO", ANSI_BLUE),
        ("log_warning", "WARNING", ANSI_YELLOW),
        ("log_error", "ERROR", ANSI_RED),
        ("log_critical", "CRITICAL", ANSI_RED),
    ],
)
def test_each_level_uses_its_colour(tmp_path: Path, method: str, level: str, color: str) -> None:
    log_file = _setup(tmp_path, level=logging.DEBUG)
    getattr(_color_logger(), method)("msg")
    line = _last_line(log_file)
    assert f"| {level} |" in line
    assert line.endswith(f"{color}msg{ANSI_RESET}")


def test_debug_is_dropped_at_info_level(tmp_path: Path) -> None:
    log_file = _setup(tmp_path)
    _color_logger().log_debug("hidden")
    _color_logger().log_info("shown")
    _last_line(log_file)
    content = log_file.read_text(encoding="utf-8")
    assert "hidden" not in content
    assert "shown" in content


def test_setup_is_idempotent_and_closes_previous_file(tmp_path: Path) -> None:
    first = _setup(tmp_path)
    second = setup_strategy_logging(
        "momentum", TradingMode.PAPER, project_root=tmp_path, started_at=datetime(2026, 9, 10, 15)
    )
    assert len(logging.getLogger(PACKAGE_LOGGER_NAME).handlers) == 2
    _color_logger().log_info("only in second")
    _last_line(second)
    assert "only in second" not in first.read_text(encoding="utf-8")
    assert "only in second" in second.read_text(encoding="utf-8")


def test_setup_isolates_package_logger_and_quiets_noisy_libraries(tmp_path: Path) -> None:
    _setup(tmp_path)
    assert logging.getLogger(PACKAGE_LOGGER_NAME).propagate is False
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("websockets").level == logging.WARNING


def test_reset_restores_default_logger_state(tmp_path: Path) -> None:
    _setup(tmp_path)
    reset_strategy_logging()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    assert package_logger.propagate is True
    assert package_logger.handlers == []
    assert package_logger.level == logging.NOTSET
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_log.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_agent_framework.log'` (the conftest import fails the same way).

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/log.py`:

```python
"""Coloured strategy logging, written to lumibot-style run log files.

`setup_strategy_logging` mirrors lumibot's `Trader(logfile=...)` combined with
the user's `build_logs` helper: one directory per run,
`logs/<strategy>/<mode>/<YYYY-mm-dd_HHMMSS>_<mode>/<mode>.log`.

`ColorLogger` reproduces `WrappingStrategy.log_*`: each message is wrapped in
an ANSI colour code inside the message itself, so the colour is kept in the
log file (read with VS Code's "ANSI Colors" plugin). The escape codes are
hand-written on purpose: `termcolor` disables colour whenever stdout is not a
TTY (nohup, systemd, cron), which would strip it from background runs' files.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from trading_agent_framework.config.env import TradingMode, find_project_root

PACKAGE_LOGGER_NAME = "trading_agent_framework"

ANSI_RESET = "\x1b[0m"
ANSI_GREY = "\x1b[90m"
ANSI_BLUE = "\x1b[34m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED = "\x1b[31m"

LOG_FILE_NAMES: dict[TradingMode, str] = {
    TradingMode.LIVE: "live.log",
    TradingMode.PAPER: "paper.log",
    TradingMode.BACKTESTING: "backtest.log",
}

_NOISY_LOGGERS = ("urllib3", "websockets")
_installed_handlers: list[logging.Handler] = []


class LumibotStyleFormatter(logging.Formatter):
    """lumibot's line format: the source location is shown for WARNING and above only."""

    _short = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    _long = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._long if record.levelno >= logging.WARNING else self._short
        return formatter.format(record)


def setup_strategy_logging(
    strategy_name: str,
    mode: TradingMode,
    *,
    project_root: Path | None = None,
    level: int = logging.INFO,
    started_at: datetime | None = None,
) -> Path:
    """Send the package's logs to the console and a fresh run log file; return its path.

    Idempotent: a second call replaces the handlers installed by the first.
    """
    root = project_root if project_root is not None else find_project_root()
    stamp = (started_at or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    log_dir = root / "logs" / strategy_name / mode.value / f"{stamp}_{mode.value}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAMES[mode]

    reset_strategy_logging()
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    formatter = LumibotStyleFormatter()
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        package_logger.addHandler(handler)
        _installed_handlers.append(handler)
    package_logger.setLevel(level)
    package_logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_file


def reset_strategy_logging() -> None:
    """Remove and close the handlers installed by `setup_strategy_logging`."""
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in _installed_handlers:
        package_logger.removeHandler(handler)
        handler.close()
    _installed_handlers.clear()
    package_logger.setLevel(logging.NOTSET)
    package_logger.propagate = True


class ColorLogger:
    """`WrappingStrategy`-style coloured log methods bound to one logger and prefix.

    Each method logs `[<prefix>] <colour>message<reset>` and returns the plain
    message. `stacklevel` works like `logging.Logger.log`'s: 1 attributes the
    record to whoever called the method.
    """

    def __init__(self, logger: logging.Logger, prefix: str) -> None:
        self._logger = logger
        self._prefix = prefix

    def log_debug(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.DEBUG, ANSI_GREY, message, stacklevel)

    def log_info(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.INFO, ANSI_BLUE, message, stacklevel)

    def log_warning(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.WARNING, ANSI_YELLOW, message, stacklevel)

    def log_error(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.ERROR, ANSI_RED, message, stacklevel)

    def log_critical(self, message: object, *, stacklevel: int = 1) -> str:
        return self._log(logging.CRITICAL, ANSI_RED, message, stacklevel)

    def _log(self, level: int, color: str, message: object, stacklevel: int) -> str:
        text = str(message)
        # +2 skips this helper and the public log_* method.
        self._logger.log(
            level,
            "[%s] %s%s%s",
            self._prefix,
            color,
            text,
            ANSI_RESET,
            stacklevel=stacklevel + 2,
        )
        return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass. The existing `caplog`-based tests (tracker, stream, orders) must still pass — this is what the conftest fixture protects.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/log.py tests/conftest.py tests/test_log.py
git commit -m "Task 3: add coloured strategy logging with lumibot-style run log files"
```

---

### Task 4: Sleeptime parsing and tick maths (`strategies/timing.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/__init__.py` (docstring only for now; exports land in Task 12)
- Create: `src/trading_agent_framework/strategies/timing.py`
- Test: `tests/strategies/test_timing.py` (do **not** add `__init__.py` files under `tests/` — the existing tree has none; `tests.fakes` resolves through `pythonpath = ["."]`)

**Interfaces:**
- Produces (`trading_agent_framework.strategies.timing`):
  - `SleepTime(interval: timedelta | None = None, sessions: int | None = None)` — frozen; exactly one field set.
  - `parse_sleeptime(value: int | str) -> SleepTime`. An int means minutes. A string is `<n><unit>`, where the unit is `S`, `M`/`T`, `H` or `D` (case-insensitive, surrounding spaces allowed). `D` gives `SleepTime(sessions=n)`, the others give `SleepTime(interval=...)`. A bool, float or other type, a malformed string, or n ≤ 0 raises `ConfigurationError`.
  - `next_tick(last_tick: datetime, interval: timedelta, now: datetime) -> tuple[datetime, int]` returns `(last_tick + j·interval, j − 1)`, where `j ≥ 1` is the smallest value putting the tick strictly after `now`. The second value counts the missed ticks.

- [ ] **Step 1: Write the failing tests**

Create `src/trading_agent_framework/strategies/__init__.py`:

```python
"""Strategy framework: lifecycle hooks, executor, and the lumibot-style Strategy base."""
```

Create `tests/strategies/test_timing.py`:

```python
from __future__ import annotations

from datetime import timedelta

import pytest
from tests.fakes import et

from trading_agent_framework.errors import ConfigurationError
from trading_agent_framework.strategies.timing import SleepTime, next_tick, parse_sleeptime


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, SleepTime(interval=timedelta(minutes=5))),
        ("30S", SleepTime(interval=timedelta(seconds=30))),
        ("30s", SleepTime(interval=timedelta(seconds=30))),
        ("5M", SleepTime(interval=timedelta(minutes=5))),
        ("5m", SleepTime(interval=timedelta(minutes=5))),
        ("2T", SleepTime(interval=timedelta(minutes=2))),
        ("2H", SleepTime(interval=timedelta(hours=2))),
        (" 3h ", SleepTime(interval=timedelta(hours=3))),
        ("1D", SleepTime(sessions=1)),
        ("2d", SleepTime(sessions=2)),
    ],
)
def test_parse_sleeptime_valid(value: int | str, expected: SleepTime) -> None:
    assert parse_sleeptime(value) == expected


@pytest.mark.parametrize(
    "value", ["", "M", "5", "5X", "1.5H", "-5M", "0M", "5 M M", 0, -1, True, 5.0, None]
)
def test_parse_sleeptime_invalid(value: object) -> None:
    with pytest.raises(ConfigurationError):
        parse_sleeptime(value)  # ty: ignore[invalid-argument-type]


def test_sleeptime_requires_exactly_one_field() -> None:
    with pytest.raises(ValueError):
        SleepTime()
    with pytest.raises(ValueError):
        SleepTime(interval=timedelta(minutes=1), sessions=1)


def test_next_tick_without_overrun() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 31))
    assert tick == et(2026, 9, 14, 9, 35)
    assert skipped == 0


def test_next_tick_counts_skipped_ticks_after_overrun() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 43))
    assert tick == et(2026, 9, 14, 9, 45)
    assert skipped == 2  # 09:35 and 09:40 were missed


def test_next_tick_is_strictly_after_now() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 35))
    assert tick == et(2026, 9, 14, 9, 40)
    assert skipped == 1


def test_next_tick_when_now_is_before_last_tick() -> None:
    tick, skipped = next_tick(et(2026, 9, 14, 9, 30), timedelta(minutes=5), et(2026, 9, 14, 9, 0))
    assert tick == et(2026, 9, 14, 9, 35)
    assert skipped == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_timing.py -q`
Expected: FAIL — `ModuleNotFoundError: ... strategies.timing`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/strategies/timing.py`:

```python
"""Pure timing helpers for the strategy executor: `sleeptime` parsing and tick maths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_agent_framework.errors import ConfigurationError

_PATTERN = re.compile(r"^\s*(\d+)\s*([smthd])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "t": 60, "h": 3600}
_FORMAT_HELP = "an int (minutes) or '<n><S|M|T|H|D>', e.g. '30S', '5M', '2H', '1D'"


@dataclass(frozen=True, slots=True)
class SleepTime:
    """Either a fixed interval between iterations, or one iteration every N sessions."""

    interval: timedelta | None = None
    sessions: int | None = None

    def __post_init__(self) -> None:
        if (self.interval is None) == (self.sessions is None):
            raise ValueError("SleepTime needs exactly one of interval or sessions")


def parse_sleeptime(value: int | str) -> SleepTime:
    """Parse lumibot's `sleeptime` attribute."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ConfigurationError(f"Invalid sleeptime {value!r}; expected {_FORMAT_HELP}")
    if isinstance(value, int):
        count, unit = value, "m"
    else:
        match = _PATTERN.match(value)
        if match is None:
            raise ConfigurationError(f"Invalid sleeptime {value!r}; expected {_FORMAT_HELP}")
        count, unit = int(match.group(1)), match.group(2).lower()
    if count <= 0:
        raise ConfigurationError(f"sleeptime must be positive, got {value!r}")
    if unit == "d":
        return SleepTime(sessions=count)
    return SleepTime(interval=timedelta(seconds=count * _UNIT_SECONDS[unit]))


def next_tick(last_tick: datetime, interval: timedelta, now: datetime) -> tuple[datetime, int]:
    """The first tick on `last_tick`'s grid strictly after `now`, and how many ticks were missed."""
    steps = max(1, (now - last_tick) // interval + 1)
    return last_tick + steps * interval, steps - 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies tests/strategies
git commit -m "Task 4: add sleeptime parsing and tick maths"
```

---

### Task 5: Pure Alpaca translation (account, calendar, replace, close) and `OrderTracker.mark_replaced`

**Files:**
- Create: `src/trading_agent_framework/brokers/alpaca/account.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/orders.py` (requests import block, `AlpacaTradingClient` Protocol, `build_get_orders_request`, new functions at the end of the request-building section)
- Modify: `src/trading_agent_framework/brokers/tracker.py` (new method after `get_active_orders`)
- Modify: `tests/fakes.py` (new Alpaca model builders, `FakeTradingClient` growth)
- Test: `tests/brokers/alpaca/test_account.py`, `tests/brokers/alpaca/test_orders_requests.py`, `tests/brokers/test_tracker.py`

**Interfaces:**
- Consumes: `orders._field`, `orders._to_decimal`, `orders._to_api_number`, `orders.round_price`, `orders.round_stop_price`, `orders.parse_broker_order` (existing); `MarketSession` (Task 2); `AccountBalances` (Task 1).
- Produces (`brokers/alpaca/account.py`): `parse_account(response: object) -> AccountBalances`, `build_calendar_request(start: date, end: date) -> GetCalendarRequest`, `parse_calendar(responses: Iterable[object], tz: ZoneInfo) -> list[MarketSession]`.
- Produces (`brokers/alpaca/orders.py`): `build_get_orders_request(limit: int = 100, *, open_only: bool = False)`, `build_replace_order_request(*, limit_price: Decimal | None = None, stop_price: Decimal | None = None) -> ReplaceOrderRequest`, `build_close_position_request(fraction: Decimal) -> ClosePositionRequest`, `parse_close_all_responses(responses: Iterable[object], strategy_name: str) -> list[Order]`. The `AlpacaTradingClient` Protocol gains `get_account`, `get_calendar(filters=)`, `replace_order_by_id(order_id, order_data=)`, `close_position(symbol_or_asset_id, close_options=)`, `close_all_positions(cancel_orders=)`.
- Produces (`brokers/tracker.py`): `OrderTracker.mark_replaced(old: Order, new: Order) -> None`.
- Produces (`tests/fakes.py`): `make_alpaca_account(**overrides)`, `make_alpaca_calendar(day: str, open_at: str = "09:30", close_at: str = "16:00")`, `make_close_position_response(body, symbol="AAPL")`, `make_failed_close_details(symbol="AAPL")`, and on `FakeTradingClient`: `raises: dict[str, BaseException]` plus `account_response`, `calendar_response`, `calendar_requests`, `replace_calls`, `replace_response`, `close_position_calls`, `close_position_response`, `close_all_calls`, `close_all_response`.

- [ ] **Step 1: Extend the fakes**

In `tests/fakes.py`, add `GetCalendarRequest`, `ReplaceOrderRequest`, `ClosePositionRequest` to the `from alpaca.trading.requests import ...` line, then add these builders after `make_alpaca_position`:

```python
def make_alpaca_account(**overrides: object) -> alpaca_models.TradeAccount:
    """Build a real `alpaca.trading.models.TradeAccount` (string money fields, as Alpaca sends)."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "account_number": "PA0000000",
        "status": alpaca_enums.AccountStatus.ACTIVE,
        "cash": "10000.50",
        "equity": "25000.25",
        "portfolio_value": "25000.25",
        "buying_power": "20000",
    }
    defaults.update(overrides)
    return alpaca_models.TradeAccount(**defaults)  # ty: ignore[invalid-argument-type]


def make_alpaca_calendar(
    day: str, open_at: str = "09:30", close_at: str = "16:00"
) -> alpaca_models.Calendar:
    """Build a real `Calendar` the way the SDK does from the API payload (naive times)."""
    return alpaca_models.Calendar(date=day, open=open_at, close=close_at)  # ty: ignore[invalid-argument-type]


def make_failed_close_details(symbol: str = "AAPL") -> alpaca_models.FailedClosePositionDetails:
    return alpaca_models.FailedClosePositionDetails(
        code=40310000, message="insufficient qty available for order", symbol=symbol
    )


def make_close_position_response(
    body: alpaca_models.Order | alpaca_models.FailedClosePositionDetails, symbol: str = "AAPL"
) -> alpaca_models.ClosePositionResponse:
    is_order = isinstance(body, alpaca_models.Order)
    return alpaca_models.ClosePositionResponse(
        order_id=body.id if isinstance(body, alpaca_models.Order) else None,
        status=200 if is_order else 403,
        symbol=symbol,
        body=body,
    )
```

In `FakeTradingClient.__init__`, append:

```python
        self.raises: dict[str, BaseException] = {}

        self.account_response: alpaca_models.TradeAccount | None = None

        self.calendar_response: list[alpaca_models.Calendar] = []
        self.calendar_requests: list[GetCalendarRequest] = []

        self.replace_calls: list[tuple[str, ReplaceOrderRequest]] = []
        self.replace_response: alpaca_models.Order | None = None

        self.close_position_calls: list[tuple[str, ClosePositionRequest]] = []
        self.close_position_response: alpaca_models.Order | None = None

        self.close_all_calls: list[bool] = []
        self.close_all_response: list[alpaca_models.ClosePositionResponse] = []
```

Add `self._maybe_raise("get_orders")` as the first line of the existing `get_orders`, then append these methods to the class:

```python
    def _maybe_raise(self, method: str) -> None:
        error = self.raises.get(method)
        if error is not None:
            raise error

    def get_account(self) -> alpaca_models.TradeAccount:
        self._maybe_raise("get_account")
        assert self.account_response is not None, "test must set client.account_response"
        return self.account_response

    def get_calendar(self, filters: GetCalendarRequest) -> list[alpaca_models.Calendar]:
        self.calendar_requests.append(filters)
        self._maybe_raise("get_calendar")
        return self.calendar_response

    def replace_order_by_id(
        self, order_id: str, order_data: ReplaceOrderRequest
    ) -> alpaca_models.Order:
        self.replace_calls.append((order_id, order_data))
        self._maybe_raise("replace_order_by_id")
        assert self.replace_response is not None, "test must set client.replace_response"
        return self.replace_response

    def close_position(
        self, symbol_or_asset_id: str, close_options: ClosePositionRequest
    ) -> alpaca_models.Order:
        self.close_position_calls.append((symbol_or_asset_id, close_options))
        self._maybe_raise("close_position")
        assert self.close_position_response is not None, "test must set close_position_response"
        return self.close_position_response

    def close_all_positions(self, cancel_orders: bool) -> list[alpaca_models.ClosePositionResponse]:
        self.close_all_calls.append(cancel_orders)
        self._maybe_raise("close_all_positions")
        return self.close_all_response
```

- [ ] **Step 2: Write the failing tests**

Create `tests/brokers/alpaca/test_account.py`:

```python
from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.fakes import ET, make_alpaca_account, make_alpaca_calendar, make_session

from trading_agent_framework.brokers.alpaca import account
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.errors import BrokerError


def test_parse_account_maps_equity_to_portfolio_value() -> None:
    assert account.parse_account(make_alpaca_account()) == AccountBalances(
        cash=Decimal("10000.50"),
        portfolio_value=Decimal("25000.25"),
        buying_power=Decimal("20000"),
    )


def test_parse_account_falls_back_to_portfolio_value_without_equity() -> None:
    response = make_alpaca_account(equity=None, portfolio_value="123.45")
    assert account.parse_account(response).portfolio_value == Decimal("123.45")


def test_parse_account_rejects_missing_cash() -> None:
    with pytest.raises(BrokerError, match="cash"):
        account.parse_account(make_alpaca_account(cash=None))


def test_build_calendar_request_sets_the_date_range() -> None:
    request = account.build_calendar_request(date(2026, 9, 14), date(2026, 9, 28))
    assert request.start == date(2026, 9, 14)
    assert request.end == date(2026, 9, 28)


def test_parse_calendar_localizes_sorts_and_keeps_early_closes() -> None:
    responses = [
        make_alpaca_calendar("2024-12-02"),
        make_alpaca_calendar("2024-11-29", close_at="13:00"),  # day after Thanksgiving
    ]
    sessions = account.parse_calendar(responses, ET)
    assert sessions == [
        make_session(date(2024, 11, 29), close_at=time(13, 0)),
        make_session(date(2024, 12, 2)),
    ]
    assert sessions[0].open.tzinfo is ET


def test_parse_calendar_converts_aware_times() -> None:
    day = SimpleNamespace(
        open=datetime(2024, 11, 29, 14, 30, tzinfo=UTC),
        close=datetime(2024, 11, 29, 18, 0, tzinfo=UTC),
    )
    [session] = account.parse_calendar([day], ET)
    assert session == make_session(date(2024, 11, 29), close_at=time(13, 0))
```

Create `tests/brokers/alpaca/test_orders_requests.py`:

```python
from __future__ import annotations

import logging
from decimal import Decimal

import pytest
from alpaca.trading.enums import QueryOrderStatus
from tests.fakes import (
    make_alpaca_order,
    make_close_position_response,
    make_failed_close_details,
)

from trading_agent_framework.brokers.alpaca import orders
from trading_agent_framework.errors import OrderValidationError

_ORDER_ID = "22222222-2222-2222-2222-222222222222"


def test_get_orders_request_defaults_to_all_statuses() -> None:
    request = orders.build_get_orders_request(50)
    assert request.status == QueryOrderStatus.ALL
    assert request.limit == 50


def test_get_orders_request_can_ask_for_open_orders_only() -> None:
    request = orders.build_get_orders_request(500, open_only=True)
    assert request.status == QueryOrderStatus.OPEN
    assert request.limit == 500


def test_replace_request_rounds_prices_to_alpaca_ticks() -> None:
    request = orders.build_replace_order_request(
        limit_price=Decimal("101.005"), stop_price=Decimal("99.999")
    )
    assert request.limit_price == 101.01
    assert request.stop_price == 100.0


def test_replace_request_needs_a_price() -> None:
    with pytest.raises(OrderValidationError, match="limit_price"):
        orders.build_replace_order_request()


def test_replace_request_wraps_sdk_validation_errors() -> None:
    with pytest.raises(OrderValidationError):
        orders.build_replace_order_request(stop_price=Decimal("-1"))


@pytest.mark.parametrize(
    ("fraction", "percentage"), [(Decimal(1), "100"), (Decimal("0.25"), "25.00")]
)
def test_close_position_request_uses_a_percentage(fraction: Decimal, percentage: str) -> None:
    request = orders.build_close_position_request(fraction)
    assert request.percentage == percentage
    assert request.qty is None


@pytest.mark.parametrize("fraction", [Decimal(0), Decimal("-0.5"), Decimal("1.5")])
def test_close_position_request_rejects_fractions_outside_zero_one(fraction: Decimal) -> None:
    with pytest.raises(OrderValidationError, match="fraction"):
        orders.build_close_position_request(fraction)


def test_parse_close_all_responses_keeps_orders_and_logs_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    responses = [
        make_close_position_response(make_alpaca_order(id=_ORDER_ID, symbol="AAPL"), "AAPL"),
        make_close_position_response(make_failed_close_details("TSLA"), "TSLA"),
    ]
    with caplog.at_level(logging.WARNING):
        closed = orders.parse_close_all_responses(responses, "momentum")
    assert [order.identifier for order in closed] == [_ORDER_ID]
    assert closed[0].strategy_name == "momentum"
    assert "TSLA" in caplog.text
```

Append to `tests/brokers/test_tracker.py`:

```python
def test_mark_replaced_cancels_old_tracks_new_and_notifies_nobody() -> None:
    tracker = OrderTracker()
    notified: list[tuple[Order, OrderEvent]] = []
    tracker.listeners.append(lambda order, event: notified.append((order, event)))
    old = make_order()
    tracker.track_unprocessed(old)
    tracker.process_trade_event(old, OrderEvent.NEW)
    notified.clear()
    new = make_order()

    tracker.mark_replaced(old, new)

    assert old.status == OrderStatus.CANCELED
    assert old in tracker.canceled
    assert old not in tracker.new
    assert new in tracker.unprocessed
    assert tracker.get_tracked_order(new.identifier) is new
    assert notified == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/brokers -q`
Expected: FAIL — `ImportError` for `brokers.alpaca.account`, `AttributeError` for `build_replace_order_request` / `mark_replaced`, and `TypeError: build_get_orders_request() got an unexpected keyword argument 'open_only'`.

- [ ] **Step 4: Implement `orders.py` additions**

Extend the requests import:

```python
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderRequest,
    ReplaceOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
```

Extend the `TYPE_CHECKING` block:

```python
    from alpaca.trading.models import Calendar as AlpacaCalendarModel
    from alpaca.trading.models import ClosePositionResponse as AlpacaClosePositionResponse
    from alpaca.trading.models import TradeAccount as AlpacaTradeAccount
```

Replace the Protocol body's method list (and change "only needs to implement these five methods" in its docstring to "only needs to implement the methods below"):

```python
    def submit_order(self, order_data: OrderRequest) -> AlpacaOrderModel: ...
    def cancel_order_by_id(self, order_id: str) -> None: ...
    def get_orders(self, filter: GetOrdersRequest) -> list[AlpacaOrderModel]: ...
    def get_order_by_id(self, order_id: str) -> AlpacaOrderModel: ...
    def get_all_positions(self) -> list[AlpacaPositionModel]: ...
    def get_account(self) -> AlpacaTradeAccount: ...
    def get_calendar(self, filters: GetCalendarRequest) -> list[AlpacaCalendarModel]: ...
    def replace_order_by_id(
        self, order_id: str, order_data: ReplaceOrderRequest
    ) -> AlpacaOrderModel: ...
    def close_position(
        self, symbol_or_asset_id: str, close_options: ClosePositionRequest
    ) -> AlpacaOrderModel: ...
    def close_all_positions(self, cancel_orders: bool) -> list[AlpacaClosePositionResponse]: ...
```

Replace `build_get_orders_request`:

```python
def build_get_orders_request(limit: int = 100, *, open_only: bool = False) -> GetOrdersRequest:
    """Build a GetOrdersRequest for every order (default) or only the open ones."""
    status = QueryOrderStatus.OPEN if open_only else QueryOrderStatus.ALL
    return GetOrdersRequest(status=status, limit=limit)


def build_replace_order_request(
    *, limit_price: Decimal | None = None, stop_price: Decimal | None = None
) -> ReplaceOrderRequest:
    """Build the PATCH /orders/{id} body; prices are rounded to Alpaca's ticks first."""
    if limit_price is None and stop_price is None:
        raise OrderValidationError("modify_order needs a new limit_price and/or stop_price")
    try:
        return ReplaceOrderRequest(
            limit_price=_to_api_number(None if limit_price is None else round_price(limit_price)),
            stop_price=_to_api_number(
                None if stop_price is None else round_stop_price(stop_price)
            ),
        )
    except (ValidationError, ValueError) as exc:
        raise OrderValidationError(str(exc)) from exc


def build_close_position_request(fraction: Decimal) -> ClosePositionRequest:
    """Close `fraction` (0 < fraction <= 1) of a position, sent as Alpaca's percentage string."""
    if not Decimal(0) < fraction <= Decimal(1):
        raise OrderValidationError(f"close fraction must be in (0, 1], got {fraction}")
    return ClosePositionRequest(percentage=format(fraction * 100, "f"))
```

Append after `parse_broker_position`:

```python
def parse_close_all_responses(responses: Iterable[object], strategy_name: str) -> list[Order]:
    """Orders placed by close_all_positions. Each response `body` is either an order or a
    FailedClosePositionDetails (code/message, no id); failures are logged and skipped."""
    closed: list[Order] = []
    for response in responses:
        body = _field(response, "body")
        if _field(body, "id") is None:
            logger.warning(
                "Alpaca could not close position %s: %s",
                _field(response, "symbol"),
                _field(body, "message"),
            )
            continue
        order = parse_broker_order(body, strategy_name)
        if order is not None:
            closed.append(order)
    return closed
```

- [ ] **Step 5: Implement `account.py`**

Create `src/trading_agent_framework/brokers/alpaca/account.py`:

```python
"""Pure Alpaca account and calendar translation.

Same rules as `orders.py`: no I/O, no state, no client instances. Together
with `orders.py`, the only module allowed to import `alpaca.trading.requests`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import cast
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from trading_agent_framework.brokers.alpaca.orders import _field, _to_decimal
from trading_agent_framework.clock import MarketSession
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.errors import BrokerError


def parse_account(response: object) -> AccountBalances:
    """Map an Alpaca TradeAccount to AccountBalances; equity is the portfolio value."""
    cash = _to_decimal(_field(response, "cash"))
    equity = _to_decimal(_field(response, "equity"))
    if equity is None:
        equity = _to_decimal(_field(response, "portfolio_value"))
    buying_power = _to_decimal(_field(response, "buying_power"))
    if cash is None or equity is None or buying_power is None:
        raise BrokerError("Alpaca account response is missing cash, equity or buying_power")
    return AccountBalances(cash=cash, portfolio_value=equity, buying_power=buying_power)


def build_calendar_request(start: date, end: date) -> GetCalendarRequest:
    return GetCalendarRequest(start=start, end=end)


def parse_calendar(responses: Iterable[object], tz: ZoneInfo) -> list[MarketSession]:
    """Alpaca calendar days -> sessions sorted by open. The SDK returns naive market times."""
    sessions = [
        MarketSession(
            open=_localize(cast(datetime, _field(day, "open")), tz),
            close=_localize(cast(datetime, _field(day, "close")), tz),
        )
        for day in responses
    ]
    return sorted(sessions, key=lambda session: session.open)


def _localize(value: datetime, tz: ZoneInfo) -> datetime:
    return value.replace(tzinfo=tz) if value.tzinfo is None else value.astimezone(tz)
```

- [ ] **Step 6: Implement `OrderTracker.mark_replaced`**

Add to `OrderTracker` in `src/trading_agent_framework/brokers/tracker.py`, after `get_active_orders`:

```python
    def mark_replaced(self, old: Order, new: Order) -> None:
        """Record that the broker replaced `old` with `new` (Alpaca PATCH /orders/{id}).

        No listener fires: the stream's own `replaced` event for `old` maps to
        MODIFIED (a no-op), and `new` reports its own events once tracked.
        """
        with self._transition_lock:
            self._process_canceled(old)
            self.unprocessed.append(new)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/brokers tests/fakes.py tests/brokers
git commit -m "Task 5: add pure Alpaca account/calendar/replace/close translation and mark_replaced"
```

---

### Task 6: `AlpacaMarketClock`

**Files:**
- Create: `src/trading_agent_framework/brokers/alpaca/clock.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/__init__.py` (export `AlpacaMarketClock`)
- Test: `tests/brokers/alpaca/test_clock.py`, `tests/brokers/alpaca/test_alpaca_public_api.py`

**Interfaces:**
- Consumes: `account.build_calendar_request`, `account.parse_calendar` (Task 5); `orders.AlpacaTradingClient` Protocol; `MarketClock`, `MarketSession` (Task 2).
- Produces: `AlpacaMarketClock(client: AlpacaTradingClient, now: Callable[[], datetime] | None = None)` — a `MarketClock` whose `next_session() -> MarketSession` never returns `None`. It raises `BrokerError` when the calendar call fails or returns no upcoming session. Class attribute `LOOKAHEAD_DAYS = 14`. Nothing is fetched until the first `next_session()` call.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/alpaca/test_clock.py`:

```python
from __future__ import annotations

from datetime import date, datetime

import pytest
from tests.fakes import (
    FakeTradingClient,
    et,
    make_alpaca_calendar,
    make_api_error,
    make_session,
)

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.clock import MARKET_TZ
from trading_agent_framework.errors import BrokerError


class _Now:
    """Mutable wall clock for tests."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def _client_with_days(*days: str) -> FakeTradingClient:
    client = FakeTradingClient()
    client.calendar_response = [make_alpaca_calendar(day) for day in days]
    return client


def test_construction_does_not_fetch_the_calendar() -> None:
    client = _client_with_days("2026-09-14")
    AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))
    assert client.calendar_requests == []


def test_next_session_fetches_a_lookahead_window_from_today() -> None:
    client = _client_with_days("2026-09-14", "2026-09-15")
    clock = AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))

    assert clock.next_session() == make_session(date(2026, 9, 14))
    [request] = client.calendar_requests
    assert request.start == date(2026, 9, 14)
    assert request.end == date(2026, 9, 28)


def test_next_session_returns_the_open_session_and_uses_the_cache() -> None:
    client = _client_with_days("2026-09-14", "2026-09-15")
    now = _Now(et(2026, 9, 14, 7))
    clock = AlpacaMarketClock(client, now=now)
    clock.next_session()

    now.value = et(2026, 9, 14, 11)
    assert clock.next_session() == make_session(date(2026, 9, 14))
    now.value = et(2026, 9, 14, 17)
    assert clock.next_session() == make_session(date(2026, 9, 15))
    assert len(client.calendar_requests) == 1


def test_next_session_refetches_once_every_cached_session_closed() -> None:
    client = _client_with_days("2026-09-14")
    now = _Now(et(2026, 9, 14, 7))
    clock = AlpacaMarketClock(client, now=now)
    clock.next_session()

    client.calendar_response = [make_alpaca_calendar("2026-09-15")]
    now.value = et(2026, 9, 14, 17)
    assert clock.next_session() == make_session(date(2026, 9, 15))
    assert [r.start for r in client.calendar_requests] == [date(2026, 9, 14), date(2026, 9, 14)]


def test_next_session_raises_when_the_calendar_is_empty() -> None:
    clock = AlpacaMarketClock(_client_with_days(), now=_Now(et(2026, 9, 14, 7)))
    with pytest.raises(BrokerError, match="no session"):
        clock.next_session()


def test_next_session_wraps_client_errors() -> None:
    client = _client_with_days("2026-09-14")
    client.raises["get_calendar"] = make_api_error(500)
    clock = AlpacaMarketClock(client, now=_Now(et(2026, 9, 14, 7)))
    with pytest.raises(BrokerError, match="calendar"):
        clock.next_session()


def test_default_now_uses_market_time_zone() -> None:
    clock = AlpacaMarketClock(FakeTradingClient())
    assert clock.now().tzinfo is MARKET_TZ
```

In `tests/brokers/alpaca/test_alpaca_public_api.py`, add `from trading_agent_framework.brokers.alpaca import clock as clock_module` and inside the test:

```python
    assert alpaca.AlpacaMarketClock is clock_module.AlpacaMarketClock
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_clock.py tests/brokers/alpaca/test_alpaca_public_api.py -q`
Expected: FAIL — `ModuleNotFoundError: ... brokers.alpaca.clock`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/brokers/alpaca/clock.py`:

```python
"""`MarketClock` backed by Alpaca's market calendar (GET /calendar)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta

from trading_agent_framework.brokers.alpaca import account
from trading_agent_framework.brokers.alpaca.orders import AlpacaTradingClient
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.errors import BrokerError


class AlpacaMarketClock(MarketClock):
    """Sessions (early closes included) come from Alpaca's calendar, fetched about ten
    trading days at a time and refetched once every cached session has closed.
    Nothing is fetched until the first `next_session()` call."""

    LOOKAHEAD_DAYS = 14

    def __init__(
        self, client: AlpacaTradingClient, now: Callable[[], datetime] | None = None
    ) -> None:
        self._client = client
        self._now = now
        self._sessions: list[MarketSession] = []

    def now(self) -> datetime:
        return self._now() if self._now is not None else super().now()

    def next_session(self) -> MarketSession:
        now = self.now()
        session = self._first_session_closing_after(now)
        if session is None:
            self._refresh(now.date())
            session = self._first_session_closing_after(now)
        if session is None:
            raise BrokerError(f"Alpaca calendar has no session after {now.isoformat()}")
        return session

    def _first_session_closing_after(self, now: datetime) -> MarketSession | None:
        return next((s for s in self._sessions if s.close > now), None)

    def _refresh(self, start: date) -> None:
        request = account.build_calendar_request(
            start, start + timedelta(days=self.LOOKAHEAD_DAYS)
        )
        try:
            responses = self._client.get_calendar(filters=request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca market calendar: {exc}") from exc
        self._sessions = account.parse_calendar(responses, self.tz)
```

In `src/trading_agent_framework/brokers/alpaca/__init__.py`, add `from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock` and `"AlpacaMarketClock"` to `__all__` (sorted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass (including `tests/brokers/test_lazy_imports.py` — `import trading_agent_framework.brokers` must still not import `alpaca`).

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/alpaca tests/brokers/alpaca
git commit -m "Task 6: add AlpacaMarketClock backed by the Alpaca calendar"
```

---

### Task 7: `Broker` interface growth, `AlpacaBroker` implementation, `FakeBroker`

**Files:**
- Modify: `src/trading_agent_framework/brokers/base.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/broker.py`
- Modify: `tests/fakes.py` (append `FakeBroker`)
- Test: `tests/brokers/test_base.py` (create), `tests/brokers/alpaca/test_broker_account.py` (create)

**Interfaces:**
- Consumes: Tasks 1, 2, 5, 6.
- Produces (`Broker`):
  - `__init__(strategy_name: str, tracker: OrderTracker | None = None, *, clock: MarketClock, is_paper: bool = True)`, with attributes `clock` and `is_paper`;
  - abstract `get_account() -> AccountBalances`;
  - abstract `modify_order(order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None) -> Order`, which returns the replacement order;
  - abstract `close_position(asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None`, which returns `None` when there is no position;
  - abstract `close_all_positions(cancel_orders: bool = True) -> list[Order]`;
  - abstract `sync_open_orders() -> list[Order]`;
  - concrete no-op `start_stream() -> None` and `stop_stream(timeout: float = 5.0) -> None`.
- Produces (`AlpacaBroker`): `__init__(strategy_name, client, tracker=None, stream=None, *, clock: MarketClock | None = None, is_paper: bool = True)`. The default clock is `AlpacaMarketClock(client)`, and `from_credentials` passes `is_paper=creds.is_paper`.
- Produces (`tests/fakes.py`): `FakeBroker(clock: MarketClock, strategy_name: str = "momentum", *, is_paper: bool = True)`.
  - Attributes: `account: AccountBalances`, `positions: list[Position]`, `remote_orders: dict[str, Order]`, `orders_to_sync: list[Order]`, `submitted`, `canceled`, `modified: list[tuple[Order, Decimal | None, Decimal | None]]`, `closed: list[tuple[Asset, Decimal]]`, `close_all_calls: list[bool]`.
  - `calls: list[str]` records `"sync_open_orders"`, `"start_stream"` and `"stop_stream"` in order.
  - Submitted and synced orders go through the real `tracker`.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/test_base.py`:

```python
from __future__ import annotations

from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order


def _limit_order() -> Order:
    return Order(
        strategy_name="momentum",
        asset=Asset("AAPL"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal(1),
        limit_price=Decimal("100"),
    )


def test_broker_stores_clock_and_account_kind() -> None:
    clock = FakeClock(et(2026, 9, 14, 9))
    broker = FakeBroker(clock, is_paper=False)
    assert broker.clock is clock
    assert broker.is_paper is False
    assert broker.strategy_name == "momentum"


def test_default_stream_hooks_are_no_ops() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 9)))
    assert Broker.start_stream(broker) is None
    assert Broker.stop_stream(broker) is None


def test_fake_broker_modify_marks_the_original_replaced() -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 9)))
    order = broker.submit_order(_limit_order())

    replacement = broker.modify_order(order, limit_price=Decimal("101"))

    assert replacement.identifier != order.identifier
    assert replacement.limit_price == Decimal("101")
    assert order.status == OrderStatus.CANCELED
    assert broker.tracker.get_tracked_order(replacement.identifier) is replacement
```

Create `tests/brokers/alpaca/test_broker_account.py`:

```python
from __future__ import annotations

from decimal import Decimal
from typing import cast

import pytest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from tests.fakes import (
    FakeTradingClient,
    make_alpaca_account,
    make_alpaca_order,
    make_api_error,
    make_close_position_response,
    make_failed_close_details,
)

from trading_agent_framework.brokers.alpaca import broker as broker_module
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import BrokerError, OrderValidationError

_OLD_ID = "11111111-1111-1111-1111-111111111111"
_NEW_ID = "22222222-2222-2222-2222-222222222222"


def _broker(client: FakeTradingClient) -> AlpacaBroker:
    return AlpacaBroker("momentum", client)


def _tracked_limit_order(broker: AlpacaBroker) -> Order:
    order = Order(
        strategy_name="momentum",
        asset=Asset("AAPL"),
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal(1),
        limit_price=Decimal("100"),
        identifier=_OLD_ID,
    )
    broker.tracker.track_unprocessed(order)
    return order


def test_default_clock_is_an_alpaca_market_clock() -> None:
    assert isinstance(_broker(FakeTradingClient()).clock, AlpacaMarketClock)


def test_from_credentials_records_the_account_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_module, "build_trading_client", lambda creds: FakeTradingClient())
    creds = AlpacaCredentials(api_key="k", api_secret="s", is_paper=False)
    broker = AlpacaBroker.from_credentials("momentum", creds, with_stream=False)
    assert broker.is_paper is False


def test_get_account_parses_balances() -> None:
    client = FakeTradingClient()
    client.account_response = make_alpaca_account()
    assert _broker(client).get_account() == AccountBalances(
        cash=Decimal("10000.50"),
        portfolio_value=Decimal("25000.25"),
        buying_power=Decimal("20000"),
    )


def test_get_account_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["get_account"] = make_api_error(500)
    with pytest.raises(BrokerError, match="account"):
        _broker(client).get_account()


def test_modify_order_replaces_and_tracks_the_new_order() -> None:
    client = FakeTradingClient()
    client.replace_response = make_alpaca_order(
        id=_NEW_ID, type="limit", limit_price="101.00", client_order_id="momentum:x"
    )
    broker = _broker(client)
    old = _tracked_limit_order(broker)

    new = broker.modify_order(old, limit_price=Decimal("101"))

    [(order_id, request)] = client.replace_calls
    assert order_id == _OLD_ID
    assert request.limit_price == 101.0
    assert new.identifier == _NEW_ID
    assert old.status == OrderStatus.CANCELED
    assert broker.tracker.get_tracked_order(_NEW_ID) is new


def test_modify_order_without_prices_is_a_validation_error() -> None:
    broker = _broker(FakeTradingClient())
    with pytest.raises(OrderValidationError):
        broker.modify_order(_tracked_limit_order(broker))


def test_modify_order_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["replace_order_by_id"] = make_api_error(422)
    broker = _broker(client)
    with pytest.raises(BrokerError, match=_OLD_ID):
        broker.modify_order(_tracked_limit_order(broker), limit_price=Decimal("101"))


def test_close_position_sends_a_percentage_and_tracks_the_order() -> None:
    client = FakeTradingClient()
    client.close_position_response = make_alpaca_order(id=_NEW_ID, side="sell")
    broker = _broker(client)

    order = broker.close_position(Asset("AAPL"), Decimal("0.5"))

    [(symbol, request)] = client.close_position_calls
    assert symbol == "AAPL"
    assert request.percentage == "50.0"
    assert order is not None
    assert broker.tracker.get_tracked_order(_NEW_ID) is order


def test_close_position_returns_none_without_a_position() -> None:
    client = FakeTradingClient()
    client.raises["close_position"] = make_api_error(404)
    assert _broker(client).close_position(Asset("AAPL")) is None


def test_close_position_wraps_other_errors() -> None:
    client = FakeTradingClient()
    client.raises["close_position"] = make_api_error(403)
    with pytest.raises(BrokerError, match="AAPL"):
        _broker(client).close_position(Asset("AAPL"))


def test_close_all_positions_tracks_placed_orders_and_skips_failures() -> None:
    client = FakeTradingClient()
    client.close_all_response = [
        make_close_position_response(make_alpaca_order(id=_NEW_ID, side="sell"), "AAPL"),
        make_close_position_response(make_failed_close_details("TSLA"), "TSLA"),
    ]
    broker = _broker(client)

    closed = broker.close_all_positions(cancel_orders=False)

    assert client.close_all_calls == [False]
    assert [o.identifier for o in closed] == [_NEW_ID]
    assert broker.tracker.get_tracked_order(_NEW_ID) is closed[0]


def test_close_all_positions_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["close_all_positions"] = make_api_error(500)
    with pytest.raises(BrokerError):
        _broker(client).close_all_positions()


def test_sync_open_orders_adopts_only_this_strategys_untracked_orders() -> None:
    client = FakeTradingClient()
    mine = make_alpaca_order(id=_NEW_ID, client_order_id="momentum:abc")
    already_tracked = make_alpaca_order(id=_OLD_ID, client_order_id="momentum:def")
    other = make_alpaca_order(client_order_id="other:ghi")
    manual = make_alpaca_order(client_order_id="manual-order")
    client.orders_response = [mine, already_tracked, other, manual]
    broker = _broker(client)
    _tracked_limit_order(broker)  # identifier _OLD_ID

    adopted = broker.sync_open_orders()

    request = cast(GetOrdersRequest, client.last_orders_request)
    assert request.status == QueryOrderStatus.OPEN
    assert [o.identifier for o in adopted] == [_NEW_ID]
    assert broker.tracker.get_tracked_order(_NEW_ID) is adopted[0]
    assert len(broker.tracker.get_all_tracked_orders()) == 2


def test_sync_open_orders_wraps_client_errors() -> None:
    client = FakeTradingClient()
    client.raises["get_orders"] = make_api_error(500)
    with pytest.raises(BrokerError, match="open orders"):
        _broker(client).sync_open_orders()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/brokers -q`
Expected: FAIL — `ImportError: cannot import name 'FakeBroker'` and `AttributeError`s on `AlpacaBroker` (`get_account`, `clock`, …).

- [ ] **Step 3: Implement the `Broker` ABC changes**

In `src/trading_agent_framework/brokers/base.py`, update the module docstring's first paragraph to mention account, clock, and stream hooks. Add these imports:

```python
from decimal import Decimal

from trading_agent_framework.clock import MarketClock
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
```

Replace `__init__`:

```python
    def __init__(
        self,
        strategy_name: str,
        tracker: OrderTracker | None = None,
        *,
        clock: MarketClock,
        is_paper: bool = True,
    ) -> None:
        self.strategy_name = strategy_name
        self.tracker = tracker if tracker is not None else OrderTracker()
        self.clock = clock
        self.is_paper = is_paper
```

Append to the class:

```python
    @abstractmethod
    def get_account(self) -> AccountBalances: ...

    @abstractmethod
    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Replace `order`'s prices at the broker; returns the replacement order."""

    @abstractmethod
    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        """Close `fraction` of the position in `asset`; None when there is no position."""

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]: ...

    @abstractmethod
    def sync_open_orders(self) -> list[Order]:
        """Track this strategy's open broker orders (e.g. after a restart); returns the adopted ones."""

    def start_stream(self) -> None:  # noqa: B027 -- optional hook, deliberately not abstract
        """Start pushing order events into `tracker`. No-op for brokers without a stream."""

    def stop_stream(self, timeout: float = 5.0) -> None:  # noqa: B027 -- optional hook
        """Stop the order-event stream. No-op for brokers without a stream."""
```

- [ ] **Step 4: Implement the `AlpacaBroker` changes**

In `src/trading_agent_framework/brokers/alpaca/broker.py`:

Change the import `from trading_agent_framework.brokers.alpaca import orders` to `from trading_agent_framework.brokers.alpaca import account, orders`, and add:

```python
from decimal import Decimal

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.clock import MarketClock
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
```

Replace `__init__` and the `from_credentials` return line:

```python
    _SYNC_LIMIT = 500  # Alpaca's maximum page size for GET /orders

    def __init__(
        self,
        strategy_name: str,
        client: orders.AlpacaTradingClient,
        tracker: OrderTracker | None = None,
        stream: TradingStream | None = None,
        *,
        clock: MarketClock | None = None,
        is_paper: bool = True,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(client),
            is_paper=is_paper,
        )
        self._client = client
        self._stream = stream
        self._alpaca_stream: AlpacaTradeStream | None = None
```

```python
        return cls(strategy_name, client, stream=stream, is_paper=creds.is_paper)
```

Add the new methods after `pull_positions`:

```python
    def get_account(self) -> AccountBalances:
        try:
            response = self._client.get_account()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca account: {exc}") from exc
        return account.parse_account(response)

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        request = orders.build_replace_order_request(limit_price=limit_price, stop_price=stop_price)
        try:
            response = self._client.replace_order_by_id(order.identifier, order_data=request)
        except Exception as exc:
            raise BrokerError(f"Failed to modify order {order.identifier}: {exc}") from exc
        replacement = orders.parse_broker_order(response, self.strategy_name)
        if replacement is None:
            raise BrokerError(f"Alpaca returned no usable replacement for order {order.identifier}")
        self.tracker.mark_replaced(order, replacement)
        return replacement

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        request = orders.build_close_position_request(fraction)
        try:
            response = self._client.close_position(asset.symbol, close_options=request)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError(f"Failed to close position {asset.symbol}: {exc}") from exc
        except Exception as exc:
            raise BrokerError(f"Failed to close position {asset.symbol}: {exc}") from exc
        order = orders.parse_broker_order(response, self.strategy_name)
        if order is not None:
            self.tracker.track_unprocessed(order)
        return order

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        try:
            responses = self._client.close_all_positions(cancel_orders=cancel_orders)
        except Exception as exc:
            raise BrokerError(f"Failed to close all positions: {exc}") from exc
        closed = orders.parse_close_all_responses(responses, self.strategy_name)
        for order in closed:
            self.tracker.track_unprocessed(order)
        return closed

    def sync_open_orders(self) -> list[Order]:
        request = orders.build_get_orders_request(self._SYNC_LIMIT, open_only=True)
        try:
            responses = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch open orders: {exc}") from exc
        prefix = f"{self.strategy_name}:"
        adopted: list[Order] = []
        for order in orders.parse_broker_orders(responses, self.strategy_name):
            if not (order.client_order_id or "").startswith(prefix):
                continue
            if self.tracker.get_tracked_order(order.identifier) is not None:
                continue
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted
```

- [ ] **Step 5: Add `FakeBroker` to `tests/fakes.py`**

Merge these imports into the file's import block: `import dataclasses`, `from decimal import Decimal`, `from typing import ClassVar`, `from trading_agent_framework.brokers.base import Broker`, `from trading_agent_framework.entities.account import AccountBalances`, `from trading_agent_framework.entities.asset import Asset`, `from trading_agent_framework.entities.enums import OrderStatus`, `from trading_agent_framework.entities.order import Order`, `from trading_agent_framework.entities.position import Position`. Then append:

```python
class FakeBroker(Broker):
    """In-memory `Broker` for strategy/executor tests: records every call, no I/O.

    Submitted and synced orders go through the real `OrderTracker`, so tests can
    drive fills with `broker.tracker.process_trade_event(...)`.
    """

    name: ClassVar[str] = "fake"

    def __init__(
        self, clock: MarketClock, strategy_name: str = "momentum", *, is_paper: bool = True
    ) -> None:
        super().__init__(strategy_name, clock=clock, is_paper=is_paper)
        self.account = AccountBalances(
            cash=Decimal("10000"), portfolio_value=Decimal("25000"), buying_power=Decimal("20000")
        )
        self.positions: list[Position] = []
        self.remote_orders: dict[str, Order] = {}
        self.orders_to_sync: list[Order] = []
        self.submitted: list[Order] = []
        self.canceled: list[Order] = []
        self.modified: list[tuple[Order, Decimal | None, Decimal | None]] = []
        self.closed: list[tuple[Asset, Decimal]] = []
        self.close_all_calls: list[bool] = []
        self.calls: list[str] = []

    def _conform_order(self, order: Order) -> Order:
        return order

    def _submit_order(self, order: Order) -> Order:
        self.submitted.append(order)
        self.tracker.track_unprocessed(order)
        return order

    def cancel_order(self, order: Order) -> None:
        self.canceled.append(order)

    def pull_order(self, identifier: str) -> Order | None:
        return self.remote_orders.get(identifier)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        return list(self.remote_orders.values())[:limit]

    def pull_positions(self) -> list[Position]:
        return list(self.positions)

    def get_account(self) -> AccountBalances:
        return self.account

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        self.modified.append((order, limit_price, stop_price))
        replacement = dataclasses.replace(
            order,
            identifier=uuid4().hex,
            limit_price=limit_price if limit_price is not None else order.limit_price,
            stop_price=stop_price if stop_price is not None else order.stop_price,
            status=OrderStatus.UNPROCESSED,
            transactions=[],
            filled_quantity=Decimal(0),
        )
        self.tracker.mark_replaced(order, replacement)
        return replacement

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        self.closed.append((asset, fraction))
        return None

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        self.close_all_calls.append(cancel_orders)
        return []

    def sync_open_orders(self) -> list[Order]:
        self.calls.append("sync_open_orders")
        for order in self.orders_to_sync:
            self.tracker.track_unprocessed(order)
        return list(self.orders_to_sync)

    def start_stream(self) -> None:
        self.calls.append("start_stream")

    def stop_stream(self, timeout: float = 5.0) -> None:
        self.calls.append("stop_stream")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass — including every pre-existing `tests/brokers/alpaca/test_broker.py` test, which still builds `AlpacaBroker("momentum", client)` and now gets a lazy `AlpacaMarketClock`.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/brokers tests/fakes.py tests/brokers
git commit -m "Task 7: extend Broker with account, clock, replace/close and order sync"
```

---

### Task 8: Order-event queue (`strategies/events.py`)

**Files:**
- Create: `src/trading_agent_framework/strategies/events.py`
- Test: `tests/strategies/test_events.py`

**Interfaces:**
- Consumes: `OrderTracker.listeners` (existing: `list[Callable[[Order, OrderEvent], None]]`, called on the stream thread inside the tracker lock, after the order's status and transactions are updated).
- Produces:
  - `QueuedOrderEvent(order: Order, event: OrderEvent, price: Decimal | None = None, quantity: Decimal | None = None)`, a frozen dataclass.
  - `OrderEventQueue(wake: threading.Event)`. Calling the instance, `queue(order, event)`, is the tracker listener: it enqueues the event and sets `wake`. For FILLED and PARTIALLY_FILLED events it copies `price` and `quantity` from `order.transactions[-1]` at call time.
  - `OrderEventQueue.drain() -> Iterator[QueuedOrderEvent]` yields the pending events in FIFO order until the queue is empty.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_events.py`:

```python
from __future__ import annotations

import threading
from decimal import Decimal

from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.strategies.events import OrderEventQueue, QueuedOrderEvent


def _order() -> Order:
    return Order(
        strategy_name="momentum", asset=Asset("AAPL"), side=OrderSide.BUY, quantity=Decimal(10)
    )


def test_listener_queues_the_event_and_sets_wake() -> None:
    wake = threading.Event()
    queue = OrderEventQueue(wake)
    order = _order()

    queue(order, OrderEvent.NEW)

    assert wake.is_set()
    assert list(queue.drain()) == [QueuedOrderEvent(order, OrderEvent.NEW)]


def test_fill_events_capture_the_fill_at_call_time() -> None:
    queue = OrderEventQueue(threading.Event())
    order = _order()
    order.add_transaction(Decimal("100.5"), Decimal("4"))

    queue(order, OrderEvent.PARTIALLY_FILLED)
    order.add_transaction(Decimal("101"), Decimal("6"))  # a later fill must not leak in

    [item] = queue.drain()
    assert item.price == Decimal("100.5")
    assert item.quantity == Decimal("4")


def test_fill_event_without_transactions_has_no_price() -> None:
    queue = OrderEventQueue(threading.Event())
    queue(_order(), OrderEvent.FILLED)
    [item] = queue.drain()
    assert item.price is None
    assert item.quantity is None


def test_drain_is_fifo_and_empties_the_queue() -> None:
    queue = OrderEventQueue(threading.Event())
    order = _order()
    queue(order, OrderEvent.NEW)
    queue(order, OrderEvent.CANCELED)

    assert [item.event for item in queue.drain()] == [OrderEvent.NEW, OrderEvent.CANCELED]
    assert list(queue.drain()) == []


def test_events_posted_by_the_tracker_on_another_thread_are_queued() -> None:
    wake = threading.Event()
    queue = OrderEventQueue(wake)
    tracker = OrderTracker()
    tracker.listeners.append(queue)
    order = _order()
    tracker.track_unprocessed(order)

    def stream_thread() -> None:
        tracker.process_trade_event(order, OrderEvent.NEW)
        tracker.process_trade_event(
            order, OrderEvent.FILLED, price=Decimal("99"), filled_quantity=Decimal("10")
        )

    thread = threading.Thread(target=stream_thread)
    thread.start()
    thread.join()

    assert wake.is_set()
    items = list(queue.drain())
    assert [item.event for item in items] == [OrderEvent.NEW, OrderEvent.FILLED]
    assert (items[1].price, items[1].quantity) == (Decimal("99"), Decimal("10"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_events.py -q`
Expected: FAIL — `ModuleNotFoundError: ... strategies.events`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/strategies/events.py`:

```python
"""Hand-off of order events from the broker's stream thread to the executor thread.

`OrderEventQueue` is registered as an `OrderTracker` listener. The tracker calls it
on the stream thread (inside its transition lock), so it only records the event
and wakes the executor; the strategy's order hooks run later, on the executor
thread, and therefore never concurrently with other strategy code.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal

from trading_agent_framework.entities.enums import OrderEvent
from trading_agent_framework.entities.order import Order

_FILL_EVENTS = frozenset({OrderEvent.FILLED, OrderEvent.PARTIALLY_FILLED})


@dataclass(frozen=True, slots=True)
class QueuedOrderEvent:
    order: Order
    event: OrderEvent
    price: Decimal | None = None
    quantity: Decimal | None = None


class OrderEventQueue:
    """Thread-safe FIFO of order events, doubling as an `OrderTracker` listener."""

    def __init__(self, wake: threading.Event) -> None:
        self._queue: queue.SimpleQueue[QueuedOrderEvent] = queue.SimpleQueue()
        self._wake = wake

    def __call__(self, order: Order, event: OrderEvent) -> None:
        price = quantity = None
        if event in _FILL_EVENTS and order.transactions:
            # Read now: later fills append more transactions before the executor drains.
            last_fill = order.transactions[-1]
            price, quantity = last_fill.price, last_fill.quantity
        self._queue.put(QueuedOrderEvent(order, event, price, quantity))
        self._wake.set()

    def drain(self) -> Iterator[QueuedOrderEvent]:
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/events.py tests/strategies/test_events.py
git commit -m "Task 8: add OrderEventQueue bridging tracker events to the executor thread"
```

---

### Task 9: `Strategy` base — hooks, config, logging and accounting/trading facade

**Files:**
- Create: `src/trading_agent_framework/strategies/strategy.py`
- Test: `tests/strategies/test_strategy.py`

**Interfaces:**
- Consumes: `Broker` (Task 7), `MarketClock` (Task 2), `TradingMode` (Task 1), `ColorLogger` (Task 3), entities.
- Produces (`trading_agent_framework.strategies.strategy.Strategy`):
  - Class attributes: `sleeptime: int | str = "1M"`, `minutes_before_opening = 60`, `minutes_before_closing = 1`, `minutes_after_closing = 0`, `parameters: Mapping[str, Any]` (empty by default).
  - `__init__(broker: Broker, *, mode: TradingMode = TradingMode.PAPER, parameters: Mapping[str, Any] | None = None, clock: MarketClock | None = None, project_root: Path | None = None)`.
  - Instance attributes: `broker`, `trading_mode`, `parameters` (class defaults overlaid by the constructor values), `clock` (defaults to `broker.clock`), `project_root`, `vars: SimpleNamespace`, `first_iteration: bool`.
  - Properties: `name` (= `broker.strategy_name`), `is_backtesting`, `cash`, `portfolio_value`.
  - No-op hooks with the signatures from spec §3: `initialize()`, `on_trading_iteration()`, `before_market_opens()`, `before_starting_trading()`, `before_market_closes()`, `after_market_closes()`, `on_strategy_end()`, `on_bot_crash(error: BaseException)` (the default calls `on_abrupt_closing()`), `on_abrupt_closing()`, `on_new_order(order)`, `on_canceled_order(order)`, `on_partially_filled_order(position, order, price, quantity, multiplier)`, `on_filled_order(position, order, price, quantity, multiplier)`.
  - Logging: `log_debug/log_info/log_warning/log_error/log_critical(message: object) -> str`.
  - Accounting: `get_cash() -> Decimal`, `get_portfolio_value() -> Decimal`, `get_positions() -> list[Position]`, `get_position(asset: Asset | str) -> Position | None`, `get_orders() -> list[Order]`, `get_order(identifier: str) -> Order | None`, `get_datetime() -> datetime`.
  - Trading:
    - `create_order(asset, quantity, side, *, limit_price=None, stop_price=None, time_in_force=TimeInForce.DAY) -> Order`;
    - `submit_order(order) -> Order`, `submit_orders(orders) -> list[Order]`;
    - `cancel_order(order) -> None`, `cancel_orders(orders) -> None`, `cancel_open_orders() -> None`;
    - `modify_order(order, limit_price=None, stop_price=None) -> Order`;
    - `sell_all(cancel_open_orders: bool = True) -> list[Order]`;
    - `close_position(asset, fraction=1) -> Order | None`;
    - `close_positions(assets: Iterable[Asset | str] | None = None) -> list[Order]`.
  - Module helpers: the alias `Number = Decimal | int | float | str`, and `_to_asset`, `_to_decimal`, `_to_optional_decimal`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_strategy.py`:

```python
from __future__ import annotations

import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


def _broker(**kwargs: object) -> FakeBroker:
    return FakeBroker(FakeClock(_START), **kwargs)  # ty: ignore[invalid-argument-type]


def _position(symbol: str) -> Position:
    return Position(
        strategy_name="momentum",
        asset=Asset(symbol),
        quantity=Decimal(10),
        side=PositionSide.LONG,
    )


class ParamStrategy(Strategy):
    parameters = {"symbol": "SPY", "quantity": 1}


class CrashRecorder(Strategy):
    def __init__(self, broker: FakeBroker) -> None:
        super().__init__(broker)
        self.abrupt_closings = 0

    def on_abrupt_closing(self) -> None:
        self.abrupt_closings += 1


# --- configuration ------------------------------------------------------------


def test_name_comes_from_the_broker() -> None:
    assert Strategy(_broker(strategy_name="momo")).name == "momo"


def test_defaults_match_lumibot() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    assert strategy.sleeptime == "1M"
    assert (strategy.minutes_before_opening, strategy.minutes_before_closing) == (60, 1)
    assert strategy.minutes_after_closing == 0
    assert strategy.parameters == {}
    assert strategy.vars == SimpleNamespace()
    assert strategy.first_iteration is True
    assert strategy.trading_mode is TradingMode.PAPER
    assert strategy.is_backtesting is False
    assert strategy.clock is broker.clock
    assert strategy.get_datetime() == _START


def test_parameters_merge_class_defaults_with_constructor_values() -> None:
    strategy = ParamStrategy(_broker(), parameters={"quantity": 5})
    assert strategy.parameters == {"symbol": "SPY", "quantity": 5}
    assert ParamStrategy.parameters == {"symbol": "SPY", "quantity": 1}


def test_explicit_clock_overrides_the_broker_clock() -> None:
    clock = FakeClock(et(2026, 9, 15, 10))
    assert Strategy(_broker(), clock=clock).get_datetime() == et(2026, 9, 15, 10)


def test_is_backtesting_follows_the_mode() -> None:
    assert Strategy(_broker(), mode=TradingMode.BACKTESTING).is_backtesting is True


# --- hooks ----------------------------------------------------------------------


def test_hooks_default_to_no_ops() -> None:
    strategy = Strategy(_broker())
    order = strategy.create_order("AAPL", 1, "buy")
    strategy.initialize()
    strategy.on_trading_iteration()
    strategy.before_market_opens()
    strategy.before_starting_trading()
    strategy.before_market_closes()
    strategy.after_market_closes()
    strategy.on_strategy_end()
    strategy.on_abrupt_closing()
    strategy.on_new_order(order)
    strategy.on_canceled_order(order)
    strategy.on_partially_filled_order(None, order, Decimal(1), Decimal(1), 1)
    strategy.on_filled_order(None, order, Decimal(1), Decimal(1), 1)


def test_default_on_bot_crash_calls_on_abrupt_closing() -> None:
    strategy = CrashRecorder(_broker())
    strategy.on_bot_crash(RuntimeError("boom"))
    assert strategy.abrupt_closings == 1


# --- logging --------------------------------------------------------------------


def test_log_info_returns_the_message_and_names_the_caller(
    caplog: pytest.LogCaptureFixture,
) -> None:
    strategy = Strategy(_broker())
    with caplog.at_level(logging.INFO, logger="trading_agent_framework"):
        returned = strategy.log_info("hello")
    record = caplog.records[-1]
    assert returned == "hello"
    assert record.filename == "test_strategy.py"
    assert "[momentum]" in record.getMessage()
    assert "hello" in record.getMessage()


# --- accounting -----------------------------------------------------------------


def test_cash_and_portfolio_value_come_from_the_account() -> None:
    strategy = Strategy(_broker())
    assert strategy.get_cash() == strategy.cash == Decimal("10000")
    assert strategy.get_portfolio_value() == strategy.portfolio_value == Decimal("25000")


def test_get_position_accepts_a_symbol_or_an_asset() -> None:
    broker = _broker()
    aapl = _position("AAPL")
    broker.positions = [aapl]
    strategy = Strategy(broker)
    assert strategy.get_positions() == [aapl]
    assert strategy.get_position("AAPL") is aapl
    assert strategy.get_position(Asset("AAPL")) is aapl
    assert strategy.get_position("TSLA") is None


def test_get_order_checks_the_tracker_then_the_broker() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    tracked = strategy.submit_order(strategy.create_order("AAPL", 1, "buy"))
    remote = strategy.create_order("TSLA", 1, "buy")
    broker.remote_orders[remote.identifier] = remote
    assert strategy.get_orders() == [tracked]
    assert strategy.get_order(tracked.identifier) is tracked
    assert strategy.get_order(remote.identifier) is remote
    assert strategy.get_order("missing") is None


# --- create_order -----------------------------------------------------------------


def test_create_market_order_from_plain_values() -> None:
    order = Strategy(_broker()).create_order("AAPL", 10, "buy")
    assert order.asset == Asset("AAPL")
    assert order.side is OrderSide.BUY
    assert order.order_type is OrderType.MARKET
    assert order.quantity == Decimal(10)
    assert order.time_in_force is TimeInForce.DAY
    assert order.strategy_name == "momentum"
    assert order.status is OrderStatus.UNPROCESSED


def test_create_limit_order() -> None:
    order = Strategy(_broker()).create_order(
        Asset("AAPL"), "2.5", OrderSide.SELL, limit_price=101.25, time_in_force="gtc"
    )
    assert order.order_type is OrderType.LIMIT
    assert order.limit_price == Decimal("101.25")
    assert order.quantity == Decimal("2.5")
    assert order.time_in_force is TimeInForce.GTC


def test_create_stop_order() -> None:
    order = Strategy(_broker()).create_order("AAPL", 1, "sell", stop_price=95)
    assert order.order_type is OrderType.STOP
    assert order.stop_price == Decimal(95)


def test_create_stop_limit_order_maps_limit_to_stop_limit_price() -> None:
    order = Strategy(_broker()).create_order("AAPL", 1, "sell", limit_price=94, stop_price=95)
    assert order.order_type is OrderType.STOP_LIMIT
    assert order.stop_price == Decimal(95)
    assert order.stop_limit_price == Decimal(94)
    assert order.limit_price is None


# --- trading ------------------------------------------------------------------------


def test_submit_orders_submits_each() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    orders = [strategy.create_order("AAPL", 1, "buy"), strategy.create_order("TSLA", 1, "buy")]
    assert strategy.submit_orders(orders) == orders
    assert broker.submitted == orders


def test_cancel_open_orders_skips_finished_orders() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    open_order = strategy.submit_order(strategy.create_order("AAPL", 1, "buy"))
    filled = strategy.submit_order(strategy.create_order("TSLA", 1, "buy"))
    broker.tracker.process_trade_event(
        filled, OrderEvent.FILLED, price=Decimal("100"), filled_quantity=Decimal(1)
    )

    strategy.cancel_open_orders()

    assert broker.canceled == [open_order]


def test_cancel_orders_cancels_each() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    orders = [strategy.create_order("AAPL", 1, "buy"), strategy.create_order("TSLA", 1, "buy")]
    strategy.cancel_orders(orders)
    assert broker.canceled == orders


def test_modify_order_converts_prices_to_decimal() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    order = strategy.submit_order(strategy.create_order("AAPL", 1, "buy", limit_price=100))

    replacement = strategy.modify_order(order, limit_price=101.5)

    assert broker.modified == [(order, Decimal("101.5"), None)]
    assert replacement is not order


def test_sell_all_closes_every_position() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.sell_all()
    strategy.sell_all(cancel_open_orders=False)
    assert broker.close_all_calls == [True, False]


def test_close_position_converts_symbol_and_fraction() -> None:
    broker = _broker()
    strategy = Strategy(broker)
    strategy.close_position("AAPL", 0.5)
    strategy.close_position(Asset("TSLA"))
    assert broker.closed == [(Asset("AAPL"), Decimal("0.5")), (Asset("TSLA"), Decimal(1))]


def test_close_positions_defaults_to_every_held_position() -> None:
    broker = _broker()
    broker.positions = [_position("AAPL"), _position("TSLA")]
    strategy = Strategy(broker)

    strategy.close_positions()
    strategy.close_positions(["MSFT"])

    assert [asset.symbol for asset, _ in broker.closed] == ["AAPL", "TSLA", "MSFT"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_strategy.py -q`
Expected: FAIL — `ModuleNotFoundError: ... strategies.strategy`.

- [ ] **Step 3: Implement**

Create `src/trading_agent_framework/strategies/strategy.py`:

```python
"""Lean port of lumibot's `Strategy` template.

Subclasses override the lifecycle hooks (`initialize`, `on_trading_iteration`,
`before_market_opens`, ...) and the order-event hooks. Everything else is a thin
facade over the broker, so strategy code reads like lumibot strategy code.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.clock import MarketClock
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.log import ColorLogger

logger = logging.getLogger(__name__)

Number = Decimal | int | float | str


def _to_asset(asset: Asset | str) -> Asset:
    return asset if isinstance(asset, Asset) else Asset(asset)


def _to_decimal(value: Number) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _to_optional_decimal(value: Number | None) -> Decimal | None:
    return None if value is None else _to_decimal(value)


class Strategy:
    """Base class for strategies, with lumibot's hook names and signatures."""

    sleeptime: int | str = "1M"
    minutes_before_opening: int = 60
    minutes_before_closing: int = 1
    minutes_after_closing: int = 0
    parameters: Mapping[str, Any] = MappingProxyType({})

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        parameters: Mapping[str, Any] | None = None,
        clock: MarketClock | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.broker = broker
        self.trading_mode = mode
        self.parameters = {**type(self).parameters, **(parameters or {})}
        self.clock = clock if clock is not None else broker.clock
        self.project_root = project_root
        self.vars = SimpleNamespace()
        self.first_iteration = True
        self._log = ColorLogger(logger, self.name)

    @property
    def name(self) -> str:
        return self.broker.strategy_name

    @property
    def is_backtesting(self) -> bool:
        return self.trading_mode is TradingMode.BACKTESTING

    # --- lifecycle hooks -------------------------------------------------------

    def initialize(self) -> None:
        """Called once before trading starts; receives matching `parameters` as kwargs."""

    def on_trading_iteration(self) -> None:
        """Called every `sleeptime` while the market is open."""

    def before_market_opens(self) -> None:
        """Called `minutes_before_opening` before each session opens."""

    def before_starting_trading(self) -> None:
        """Called when each session opens, before its first iteration."""

    def before_market_closes(self) -> None:
        """Called `minutes_before_closing` before each session closes."""

    def after_market_closes(self) -> None:
        """Called `minutes_after_closing` after each session closes."""

    def on_strategy_end(self) -> None:
        """Called once when the run ends (stop, interrupt, or no more sessions)."""

    def on_bot_crash(self, error: BaseException) -> None:
        """Called when a hook raises; like lumibot, defaults to `on_abrupt_closing()`."""
        self.on_abrupt_closing()

    def on_abrupt_closing(self) -> None:
        """Called on Ctrl+C / SIGTERM, and by the default `on_bot_crash`."""

    # --- order-event hooks -----------------------------------------------------

    def on_new_order(self, order: Order) -> None:
        """Called when the broker accepts an order."""

    def on_canceled_order(self, order: Order) -> None:
        """Called when an order is canceled or expires."""

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called on each partial fill; `quantity` is the size of this fill."""

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called when an order is completely filled."""

    # --- logging ---------------------------------------------------------------

    def log_debug(self, message: object) -> str:
        return self._log.log_debug(message, stacklevel=2)

    def log_info(self, message: object) -> str:
        return self._log.log_info(message, stacklevel=2)

    def log_warning(self, message: object) -> str:
        return self._log.log_warning(message, stacklevel=2)

    def log_error(self, message: object) -> str:
        return self._log.log_error(message, stacklevel=2)

    def log_critical(self, message: object) -> str:
        return self._log.log_critical(message, stacklevel=2)

    # --- accounting --------------------------------------------------------------

    def get_datetime(self) -> datetime:
        return self.clock.now()

    def get_cash(self) -> Decimal:
        return self.broker.get_account().cash

    def get_portfolio_value(self) -> Decimal:
        return self.broker.get_account().portfolio_value

    @property
    def cash(self) -> Decimal:
        return self.get_cash()

    @property
    def portfolio_value(self) -> Decimal:
        return self.get_portfolio_value()

    def get_positions(self) -> list[Position]:
        return self.broker.pull_positions()

    def get_position(self, asset: Asset | str) -> Position | None:
        symbol = _to_asset(asset).symbol
        return next((p for p in self.get_positions() if p.asset.symbol == symbol), None)

    def get_orders(self) -> list[Order]:
        return self.broker.tracker.get_all_tracked_orders()

    def get_order(self, identifier: str) -> Order | None:
        tracked = self.broker.get_tracked_order(identifier)
        return tracked if tracked is not None else self.broker.pull_order(identifier)

    # --- trading -----------------------------------------------------------------

    def create_order(
        self,
        asset: Asset | str,
        quantity: Number,
        side: OrderSide | str,
        *,
        limit_price: Number | None = None,
        stop_price: Number | None = None,
        time_in_force: TimeInForce | str = TimeInForce.DAY,
    ) -> Order:
        """Build (not submit) an order; the type follows from the prices given."""
        limit = _to_optional_decimal(limit_price)
        stop = _to_optional_decimal(stop_price)
        if limit is not None and stop is not None:
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
        )

    def submit_order(self, order: Order) -> Order:
        return self.broker.submit_order(order)

    def submit_orders(self, orders: Sequence[Order]) -> list[Order]:
        return self.broker.submit_orders(orders)

    def cancel_order(self, order: Order) -> None:
        self.broker.cancel_order(order)

    def cancel_orders(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self.cancel_order(order)

    def cancel_open_orders(self) -> None:
        self.cancel_orders(self.broker.tracker.get_active_orders())

    def modify_order(
        self, order: Order, limit_price: Number | None = None, stop_price: Number | None = None
    ) -> Order:
        """Change an open order's prices; returns the replacement order (new identifier)."""
        return self.broker.modify_order(
            order,
            limit_price=_to_optional_decimal(limit_price),
            stop_price=_to_optional_decimal(stop_price),
        )

    def sell_all(self, cancel_open_orders: bool = True) -> list[Order]:
        return self.broker.close_all_positions(cancel_orders=cancel_open_orders)

    def close_position(self, asset: Asset | str, fraction: Number = 1) -> Order | None:
        return self.broker.close_position(_to_asset(asset), _to_decimal(fraction))

    def close_positions(self, assets: Iterable[Asset | str] | None = None) -> list[Order]:
        """Close each given asset's position; `None` means every position currently held."""
        targets = [p.asset for p in self.get_positions()] if assets is None else assets
        orders = [self.close_position(asset) for asset in targets]
        return [order for order in orders if order is not None]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/strategies/strategy.py tests/strategies/test_strategy.py
git commit -m "Task 9: add Strategy base with lumibot hooks and broker facade"
```

---

### Task 10: `StrategyExecutor` lifecycle loop

**Files:**
- Create: `src/trading_agent_framework/strategies/executor.py`
- Modify: `src/trading_agent_framework/strategies/strategy.py` (executor wiring, `sleep`, `stop`)
- Test: `tests/strategies/test_executor.py`

**Interfaces:**
- Consumes: `parse_sleeptime`, `next_tick` (Task 4); `OrderEventQueue` (Task 8); `Strategy` (Task 9); `FakeBroker`, `FakeClock`, `weekday_sessions`, `et` (fakes).
- Produces (`trading_agent_framework.strategies.executor`):
  - Constants `MAX_WAIT_SLICE_SECONDS = 60.0` and `CALENDAR_RETRY_SECONDS = 60.0`.
  - `StrategyExecutor(strategy: Strategy)` with:
    - `run() -> None`;
    - `stop() -> None`, safe from any thread or hook;
    - property `stopped -> bool`;
    - `wait_until(deadline: datetime) -> None` (Task 11 widens this signature).
- Produces (`Strategy`):
  - attribute `executor: StrategyExecutor`, created in `__init__`;
  - `sleep(seconds: float) -> None`;
  - `stop() -> None`.
- Behaviour contract: spec §5.1–§5.2 and §5.4, including Spec deviations 1, 2, 6 and 7 from this plan's header.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_executor.py`:

```python
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.errors import BrokerError, ConfigurationError
from trading_agent_framework.strategies.strategy import Strategy

MONDAY = date(2026, 9, 14)


class Recorder(Strategy):
    """Records every lifecycle hook with the clock time it ran at."""

    sleeptime = "2H"

    def __init__(self, broker: FakeBroker, **kwargs: object) -> None:
        super().__init__(broker, **kwargs)  # ty: ignore[invalid-argument-type]
        self.calls: list[tuple[str, datetime]] = []
        self.errors: list[BaseException] = []

    def _record(self, hook: str) -> None:
        self.calls.append((hook, self.get_datetime()))

    def initialize(self) -> None:
        self._record("initialize")

    def before_market_opens(self) -> None:
        self._record("before_market_opens")

    def before_starting_trading(self) -> None:
        self._record("before_starting_trading")

    def on_trading_iteration(self) -> None:
        self._record("on_trading_iteration")

    def before_market_closes(self) -> None:
        self._record("before_market_closes")

    def after_market_closes(self) -> None:
        self._record("after_market_closes")

    def on_strategy_end(self) -> None:
        self._record("on_strategy_end")

    def on_abrupt_closing(self) -> None:
        self._record("on_abrupt_closing")

    def on_bot_crash(self, error: BaseException) -> None:
        self.errors.append(error)
        super().on_bot_crash(error)

    @property
    def fake_clock(self) -> FakeClock:
        return cast(FakeClock, self.clock)

    def hooks(self) -> list[str]:
        return [hook for hook, _ in self.calls]

    def times(self, hook: str) -> list[datetime]:
        return [when for name, when in self.calls if name == hook]


def _strategy(
    cls: type[Recorder] = Recorder,
    *,
    start: datetime | None = None,
    sessions: int = 1,
    **kwargs: object,
) -> Recorder:
    clock = FakeClock(start or et(2026, 9, 14, 7), weekday_sessions(MONDAY, sessions))
    return cls(FakeBroker(clock), **kwargs)


def _run(cls: type[Recorder] = Recorder, **kwargs: object) -> Recorder:
    strategy = _strategy(cls, **kwargs)  # ty: ignore[invalid-argument-type]
    strategy.executor.run()
    return strategy


def _day(day: int) -> list[tuple[str, datetime]]:
    return [
        ("before_market_opens", et(2026, 9, day, 8, 30)),
        ("before_starting_trading", et(2026, 9, day, 9, 30)),
        ("on_trading_iteration", et(2026, 9, day, 9, 30)),
        ("on_trading_iteration", et(2026, 9, day, 11, 30)),
        ("on_trading_iteration", et(2026, 9, day, 13, 30)),
        ("on_trading_iteration", et(2026, 9, day, 15, 30)),
        ("before_market_closes", et(2026, 9, day, 15, 59)),
        ("after_market_closes", et(2026, 9, day, 16, 0)),
    ]


# --- lifecycle order and timing ------------------------------------------------------


def test_hook_order_and_times_over_two_sessions() -> None:
    strategy = _run(sessions=2)
    assert strategy.calls == [
        ("initialize", et(2026, 9, 14, 7)),
        *_day(14),
        *_day(15),
        ("on_strategy_end", et(2026, 9, 15, 16, 0)),
    ]


def test_mid_session_start_skips_before_market_opens_and_anchors_on_start() -> None:
    strategy = _run(start=et(2026, 9, 14, 10, 15))
    assert "before_market_opens" not in strategy.hooks()
    assert strategy.times("before_starting_trading") == [et(2026, 9, 14, 10, 15)]
    assert strategy.times("on_trading_iteration") == [
        et(2026, 9, 14, 10, 15),
        et(2026, 9, 14, 12, 15),
        et(2026, 9, 14, 14, 15),
    ]


def test_before_market_opens_runs_with_zero_minutes_before_opening() -> None:
    class NoLead(Recorder):
        minutes_before_opening = 0

    strategy = _run(NoLead)
    assert strategy.times("before_market_opens") == [et(2026, 9, 14, 9, 30)]


def test_five_minute_sleeptime_stops_before_the_closing_window() -> None:
    class FiveMinutes(Recorder):
        sleeptime = "5M"

    iterations = _run(FiveMinutes).times("on_trading_iteration")
    assert len(iterations) == 78
    assert iterations[0] == et(2026, 9, 14, 9, 30)
    assert iterations[-1] == et(2026, 9, 14, 15, 55)


def test_daily_sleeptime_runs_once_per_session() -> None:
    class Daily(Recorder):
        sleeptime = "1D"

    assert _run(Daily, sessions=2).times("on_trading_iteration") == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 15, 9, 30),
    ]


def test_two_day_sleeptime_skips_every_other_session() -> None:
    class EveryOtherDay(Recorder):
        sleeptime = "2D"

    strategy = _run(EveryOtherDay, sessions=3)
    assert strategy.times("on_trading_iteration") == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 16, 9, 30),
    ]
    assert len(strategy.times("before_starting_trading")) == 3


def test_overrun_skips_missed_ticks_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    class Slow(Recorder):
        sleeptime = "10M"

        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            if self.first_iteration:
                self.fake_clock.advance(25 * 60)

    with caplog.at_level(logging.WARNING, logger="trading_agent_framework"):
        strategy = _run(Slow)
    assert strategy.times("on_trading_iteration")[:3] == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 14, 10, 0),
        et(2026, 9, 14, 10, 10),
    ]
    assert "skipped 2 tick(s)" in caplog.text


def test_sleeptime_is_reread_after_each_iteration() -> None:
    class Changer(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            self.sleeptime = "1H"

    assert _run(Changer).times("on_trading_iteration")[:3] == [
        et(2026, 9, 14, 9, 30),
        et(2026, 9, 14, 10, 30),
        et(2026, 9, 14, 11, 30),
    ]


def test_first_iteration_is_true_only_once() -> None:
    class FirstFlag(Recorder):
        def __init__(self, broker: FakeBroker) -> None:
            super().__init__(broker)
            self.flags: list[bool] = []

        def on_trading_iteration(self) -> None:
            self.flags.append(self.first_iteration)

    strategy = _run(FirstFlag)
    assert cast(FirstFlag, strategy).flags == [True, False, False, False]


def test_sleep_advances_the_clock_inside_a_hook() -> None:
    class Sleeper(Recorder):
        def on_trading_iteration(self) -> None:
            if self.first_iteration:
                self.sleep(90)
                self._record("after_sleep")

    assert _run(Sleeper).times("after_sleep") == [et(2026, 9, 14, 9, 31, 30)]


def test_no_sessions_runs_initialize_and_end_only() -> None:
    assert _run(sessions=0).hooks() == ["initialize", "on_strategy_end"]


# --- errors -----------------------------------------------------------------------


def test_iteration_crash_calls_on_bot_crash_and_trading_continues() -> None:
    class Crashy(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            if self.first_iteration:
                raise RuntimeError("boom")

    strategy = _run(Crashy)
    assert [str(e) for e in strategy.errors] == ["boom"]
    assert "on_abrupt_closing" in strategy.hooks()  # lumibot's default on_bot_crash
    assert len(strategy.times("on_trading_iteration")) == 4
    assert strategy.hooks()[-1] == "on_strategy_end"


def test_lifecycle_hook_crash_is_reported_and_the_session_goes_on() -> None:
    class BadOpen(Recorder):
        def before_market_opens(self) -> None:
            raise RuntimeError("open failed")

    strategy = _run(BadOpen)
    assert [str(e) for e in strategy.errors] == ["open failed"]
    assert "before_starting_trading" in strategy.hooks()


def test_initialize_crash_propagates_without_on_strategy_end() -> None:
    class BadInit(Recorder):
        def initialize(self) -> None:
            raise RuntimeError("bad init")

    strategy = _strategy(BadInit)
    with pytest.raises(RuntimeError, match="bad init"):
        strategy.executor.run()
    assert [str(e) for e in strategy.errors] == ["bad init"]
    assert "on_strategy_end" not in strategy.hooks()
    assert cast(FakeBroker, strategy.broker).calls[-1] == "stop_stream"


def test_invalid_sleeptime_fails_fast_after_initialize() -> None:
    class Bad(Recorder):
        sleeptime = "soon"

    strategy = _strategy(Bad)
    with pytest.raises(ConfigurationError):
        strategy.executor.run()
    assert strategy.hooks() == ["initialize", "on_strategy_end"]


def test_calendar_errors_are_retried(caplog: pytest.LogCaptureFixture) -> None:
    strategy = _strategy()
    strategy.fake_clock.next_session_errors = [BrokerError("calendar down")]
    with caplog.at_level(logging.ERROR, logger="trading_agent_framework"):
        strategy.executor.run()
    assert strategy.times("before_market_opens") == [et(2026, 9, 14, 8, 30)]
    assert "retrying" in caplog.text


# --- stop, parameters, setup/teardown ------------------------------------------------


def test_stop_ends_the_run_after_the_current_hook() -> None:
    class Stopper(Recorder):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            self.stop()

    strategy = _run(Stopper, sessions=2)
    assert strategy.executor.stopped is True
    assert strategy.hooks() == [
        "initialize",
        "before_market_opens",
        "before_starting_trading",
        "on_trading_iteration",
        "on_strategy_end",
    ]


def test_initialize_receives_matching_parameters() -> None:
    class WithParams(Recorder):
        def initialize(self, symbol: str = "", quantity: int = 1) -> None:
            self.received = (symbol, quantity)

    strategy = _run(WithParams, parameters={"symbol": "SPY", "unused": 3})
    assert cast(WithParams, strategy).received == ("SPY", 1)


def test_orders_are_synced_and_the_stream_started_before_initialize() -> None:
    class SeesSetup(Recorder):
        def initialize(self) -> None:
            self.calls_at_init = list(cast(FakeBroker, self.broker).calls)

    strategy = _run(SeesSetup)
    broker = cast(FakeBroker, strategy.broker)
    assert cast(SeesSetup, strategy).calls_at_init == ["sync_open_orders", "start_stream"]
    assert broker.calls == ["sync_open_orders", "start_stream", "stop_stream"]
    assert broker.tracker.listeners == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_executor.py -q`
Expected: FAIL — `AttributeError: 'Recorder' object has no attribute 'executor'` (and `ModuleNotFoundError` for `strategies.executor` once wired).

- [ ] **Step 3: Implement the executor**

Create `src/trading_agent_framework/strategies/executor.py`:

```python
"""Single-threaded driver of a `Strategy`'s lifecycle.

`StrategyExecutor.run()` walks the market calendar one session at a time:
before_market_opens -> before_starting_trading -> on_trading_iteration every
`sleeptime` -> before_market_closes -> after_market_closes. Every wait goes
through the strategy's `MarketClock`, so a simulated clock can later drive the
very same loop for backtesting.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from trading_agent_framework.clock import MarketSession
from trading_agent_framework.errors import BrokerError
from trading_agent_framework.strategies.events import OrderEventQueue
from trading_agent_framework.strategies.timing import next_tick, parse_sleeptime

if TYPE_CHECKING:
    from trading_agent_framework.strategies.strategy import Strategy

logger = logging.getLogger(__name__)

# Waits are sliced so clock drift (e.g. a suspended laptop) is corrected at least this often;
# order events and stop() interrupt a slice through the wake event anyway.
MAX_WAIT_SLICE_SECONDS = 60.0
CALENDAR_RETRY_SECONDS = 60.0

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


class StrategyExecutor:
    """Runs one strategy's hooks on the calling thread, session after session."""

    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._events = OrderEventQueue(self._wake)
        self._session_index = 0
        self._last_iteration_session: int | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def stop(self) -> None:
        """Ask the run to end at the next check; safe from any thread or hook."""
        self._stop.set()
        self._wake.set()

    def run(self) -> None:
        strategy = self.strategy
        broker = strategy.broker
        self._stop.clear()
        broker.tracker.listeners.append(self._events)
        initialized = False
        try:
            broker.sync_open_orders()
            broker.start_stream()
            self._initialize()
            initialized = True
            parse_sleeptime(strategy.sleeptime)  # fail fast on a bad sleeptime
            self._run_sessions()
        finally:
            if initialized:
                self._call_hook(strategy.on_strategy_end)
            broker.stop_stream()
            broker.tracker.listeners.remove(self._events)

    def wait_until(self, deadline: datetime) -> None:
        """Wait on the clock until `deadline` or `stop()`."""
        clock = self.strategy.clock
        while not self._stop.is_set():
            remaining = (deadline - clock.now()).total_seconds()
            if remaining <= 0:
                return
            clock.wait(min(remaining, MAX_WAIT_SLICE_SECONDS), self._wake)
            self._wake.clear()

    # --- run phases ----------------------------------------------------------------

    def _initialize(self) -> None:
        strategy = self.strategy
        signature = inspect.signature(strategy.initialize)
        kwargs = {
            name: value
            for name, value in strategy.parameters.items()
            if name in signature.parameters and signature.parameters[name].kind in _KEYWORD_KINDS
        }
        logger.info("Initializing strategy %s", strategy.name)
        try:
            strategy.initialize(**kwargs)
        except Exception as exc:
            logger.exception("initialize failed for strategy %s", strategy.name)
            self._on_bot_crash(exc)
            raise

    def _run_sessions(self) -> None:
        while not self._stop.is_set():
            session = self._next_session()
            if session is None:
                return
            self._run_session(session)
            self._session_index += 1

    def _next_session(self) -> MarketSession | None:
        clock = self.strategy.clock
        while not self._stop.is_set():
            try:
                return clock.next_session()
            except BrokerError:
                logger.exception(
                    "Could not get the next market session; retrying in %.0fs",
                    CALENDAR_RETRY_SECONDS,
                )
                self.wait_until(clock.now() + timedelta(seconds=CALENDAR_RETRY_SECONDS))
        return None

    def _run_session(self, session: MarketSession) -> None:
        strategy = self.strategy
        logger.info(
            "Next session for %s: %s -> %s",
            strategy.name,
            session.open.isoformat(),
            session.close.isoformat(),
        )
        started_before_open = self._now() < session.open
        self.wait_until(session.open - timedelta(minutes=strategy.minutes_before_opening))
        if self._stop.is_set():
            return
        if started_before_open:
            self._call_hook(strategy.before_market_opens)
            self.wait_until(session.open)
            if self._stop.is_set():
                return
        self._call_hook(strategy.before_starting_trading)
        self._trade(session)
        if self._stop.is_set():
            return
        self._call_hook(strategy.before_market_closes)
        self.wait_until(session.close + timedelta(minutes=strategy.minutes_after_closing))
        if self._stop.is_set():
            return
        self._call_hook(strategy.after_market_closes)

    def _trade(self, session: MarketSession) -> None:
        strategy = self.strategy
        stop_at = session.close - timedelta(minutes=strategy.minutes_before_closing)
        tick = max(session.open, self._now())
        while not self._stop.is_set():
            self.wait_until(min(tick, stop_at))
            if self._stop.is_set() or self._now() >= stop_at:
                return
            sleeptime = parse_sleeptime(strategy.sleeptime)
            if sleeptime.sessions is not None and not self._iteration_due(sleeptime.sessions):
                return
            self._iterate()
            sleeptime = parse_sleeptime(strategy.sleeptime)  # the iteration may change it
            if sleeptime.interval is None:
                return  # session-based sleeptime: one iteration per due session
            tick, skipped = next_tick(tick, sleeptime.interval, self._now())
            if skipped:
                logger.warning(
                    "Iteration of %s overran its %s sleeptime; skipped %d tick(s)",
                    strategy.name,
                    strategy.sleeptime,
                    skipped,
                )

    def _iteration_due(self, sessions: int) -> bool:
        last = self._last_iteration_session
        return last is None or self._session_index - last >= sessions

    def _iterate(self) -> None:
        strategy = self.strategy
        self._last_iteration_session = self._session_index
        logger.debug("Trading iteration of %s at %s", strategy.name, self._now().isoformat())
        try:
            strategy.on_trading_iteration()
        except Exception as exc:
            logger.exception("on_trading_iteration failed for strategy %s", strategy.name)
            self._on_bot_crash(exc)
        finally:
            strategy.first_iteration = False

    # --- helpers -------------------------------------------------------------------

    def _call_hook(self, hook: Callable[[], None]) -> None:
        try:
            hook()
        except Exception as exc:
            name = getattr(hook, "__name__", repr(hook))
            logger.exception("%s failed for strategy %s", name, self.strategy.name)
            self._on_bot_crash(exc)

    def _on_bot_crash(self, error: BaseException) -> None:
        try:
            self.strategy.on_bot_crash(error)
        except Exception:
            logger.exception("on_bot_crash raised for strategy %s", self.strategy.name)

    def _now(self) -> datetime:
        return self.strategy.clock.now()
```

- [ ] **Step 4: Wire the executor into `Strategy`**

In `src/trading_agent_framework/strategies/strategy.py`:
- change `from datetime import datetime` to `from datetime import datetime, timedelta`;
- add `from trading_agent_framework.strategies.executor import StrategyExecutor`;
- append `self.executor = StrategyExecutor(self)` as the last line of `__init__`;
- add a control section after the logging section:

```python
    # --- control -------------------------------------------------------------------

    def sleep(self, seconds: float) -> None:
        """Pause for `seconds` of clock time (returns early if the run is stopped)."""
        self.executor.wait_until(self.get_datetime() + timedelta(seconds=seconds))

    def stop(self) -> None:
        """End the run once the current hook returns; `on_strategy_end` still runs."""
        self.executor.stop()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass. If `test_hook_order_and_times_over_two_sessions` fails, compare the recorded list with `_day()`: the most likely cause is waiting past `stop_at` (the `min(tick, stop_at)` guard) or a skipped `before_market_opens` (the `started_before_open` check must run before the first wait).

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies tests/strategies/test_executor.py
git commit -m "Task 10: add StrategyExecutor session loop with sleeptime-driven iterations"
```

---

### Task 11: Order-event dispatch, order waits, and Ctrl+C / SIGTERM

**Files:**
- Modify: `src/trading_agent_framework/strategies/executor.py`
- Modify: `src/trading_agent_framework/strategies/strategy.py`
- Test: `tests/strategies/test_executor_events.py`

**Interfaces:**
- Consumes: `OrderEventQueue.drain()`, `QueuedOrderEvent` (Task 8); the executor from Task 10.
- Produces (`StrategyExecutor`):
  - `wait_until(deadline: datetime, until: Callable[[], bool] | None = None) -> bool` replaces the Task 10 version. It returns `True` as soon as `until()` holds, and `False` at the deadline or on `stop()`. Order events are dispatched on every loop.
  - `wait_for(until: Callable[[], bool], timeout: float | None = None) -> bool`. With no timeout, the horizon is 365 days.
  - `run()` also handles `KeyboardInterrupt` (calls `on_abrupt_closing`, then `on_strategy_end` via `finally`), and turns SIGTERM into `KeyboardInterrupt` when running on the main thread.
- Produces (`Strategy`):
  - `wait_for_order_execution(order: Order, timeout: float | None = None) -> bool`;
  - `wait_for_orders_execution(orders: Sequence[Order], timeout: float | None = None) -> bool`.
  - An order counts as final when its status is FILL, CANCELED, ERROR or EXPIRED.
- Dispatch mapping: spec §5.3. Fill hooks receive `position = strategy.get_position(order.asset)` and `multiplier = 1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_executor_events.py`:

```python
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.strategy import Strategy

MONDAY = date(2026, 9, 14)


class OrderHooks(Strategy):
    """Records order hooks (with the thread they ran on) and lifecycle milestones."""

    sleeptime = "1D"

    def __init__(self, broker: FakeBroker) -> None:
        super().__init__(broker)
        self.events: list[tuple[object, ...]] = []
        self.threads: set[str] = set()
        self.milestones: list[str] = []
        self.errors: list[BaseException] = []

    def _note(self, *event: object) -> None:
        self.events.append(event)
        self.threads.add(threading.current_thread().name)

    def on_new_order(self, order: Order) -> None:
        self._note("new", order.identifier)

    def on_canceled_order(self, order: Order) -> None:
        self._note("canceled", order.identifier)

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        self._note("partial", order.identifier, position, price, quantity, multiplier)

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        self._note("filled", order.identifier, position, price, quantity, multiplier)

    def on_trading_iteration(self) -> None:
        self.milestones.append("on_trading_iteration")

    def on_abrupt_closing(self) -> None:
        self.milestones.append("on_abrupt_closing")

    def on_strategy_end(self) -> None:
        self.milestones.append("on_strategy_end")

    def on_bot_crash(self, error: BaseException) -> None:
        self.errors.append(error)

    @property
    def fake_clock(self) -> FakeClock:
        return cast(FakeClock, self.clock)

    @property
    def fake_broker(self) -> FakeBroker:
        return cast(FakeBroker, self.broker)


def _strategy(cls: type[OrderHooks] = OrderHooks, sessions: int = 1) -> OrderHooks:
    clock = FakeClock(et(2026, 9, 14, 7), weekday_sessions(MONDAY, sessions))
    return cls(FakeBroker(clock))


def _aapl_position() -> Position:
    return Position(
        strategy_name="momentum", asset=Asset("AAPL"), quantity=Decimal(10), side=PositionSide.LONG
    )


def _on_first_wait(clock: FakeClock, action: Callable[[], None]) -> None:
    """Run `action` on a separate "stream" thread during the executor's first wait."""

    def hook() -> None:
        clock.on_wait = None
        thread = threading.Thread(target=action, name="fake-stream")
        thread.start()
        thread.join()

    clock.on_wait = hook


def test_events_from_the_stream_thread_run_hooks_on_the_executor_thread() -> None:
    strategy = _strategy()
    broker = strategy.fake_broker
    broker.positions = [_aapl_position()]
    order = broker.submit_order(strategy.create_order("AAPL", 10, "buy"))

    def stream() -> None:
        broker.tracker.process_trade_event(order, OrderEvent.NEW)
        broker.tracker.process_trade_event(
            order, OrderEvent.PARTIALLY_FILLED, price=Decimal(100), filled_quantity=Decimal(4)
        )
        broker.tracker.process_trade_event(
            order, OrderEvent.FILLED, price=Decimal(101), filled_quantity=Decimal(6)
        )

    _on_first_wait(strategy.fake_clock, stream)
    strategy.executor.run()

    position = _aapl_position()
    assert strategy.events == [
        ("new", order.identifier),
        ("partial", order.identifier, position, Decimal(100), Decimal(4), 1),
        ("filled", order.identifier, position, Decimal(101), Decimal(6), 1),
    ]
    assert strategy.threads == {threading.current_thread().name}


def test_cancel_error_and_modified_events(caplog: pytest.LogCaptureFixture) -> None:
    strategy = _strategy()
    broker = strategy.fake_broker
    canceled = broker.submit_order(strategy.create_order("AAPL", 1, "buy"))
    rejected = broker.submit_order(strategy.create_order("TSLA", 1, "buy"))
    modified = broker.submit_order(strategy.create_order("MSFT", 1, "buy"))

    def stream() -> None:
        broker.tracker.process_trade_event(canceled, OrderEvent.CANCELED)
        broker.tracker.process_trade_event(rejected, OrderEvent.ERROR)
        broker.tracker.process_trade_event(modified, OrderEvent.MODIFIED)

    _on_first_wait(strategy.fake_clock, stream)
    with caplog.at_level(logging.WARNING, logger="trading_agent_framework"):
        strategy.executor.run()

    assert strategy.events == [("canceled", canceled.identifier)]
    assert rejected.identifier in caplog.text


def test_order_hook_crash_goes_to_on_bot_crash_and_trading_continues() -> None:
    class BadHook(OrderHooks):
        def on_new_order(self, order: Order) -> None:
            raise RuntimeError("hook failed")

    strategy = _strategy(BadHook)
    broker = strategy.fake_broker
    order = broker.submit_order(strategy.create_order("AAPL", 1, "buy"))
    _on_first_wait(
        strategy.fake_clock, lambda: broker.tracker.process_trade_event(order, OrderEvent.NEW)
    )

    strategy.executor.run()

    assert [str(e) for e in strategy.errors] == ["hook failed"]
    assert strategy.milestones == ["on_trading_iteration", "on_strategy_end"]


class Waiter(OrderHooks):
    """Submits an order in its first iteration and waits for it."""

    fill_it = True

    def on_trading_iteration(self) -> None:
        broker = self.fake_broker
        order = self.submit_order(self.create_order("AAPL", 1, "buy"))
        if self.fill_it:
            _on_first_wait(
                self.fake_clock,
                lambda: broker.tracker.process_trade_event(
                    order, OrderEvent.FILLED, price=Decimal(100), filled_quantity=Decimal(1)
                ),
            )
        started = self.get_datetime()
        self.vars.result = self.wait_for_order_execution(order, timeout=600)
        self.vars.waited = self.get_datetime() - started
        self.vars.events_at_return = list(self.events)


def test_wait_for_order_execution_returns_once_filled() -> None:
    strategy = _strategy(Waiter)
    strategy.executor.run()
    assert strategy.vars.result is True
    assert strategy.vars.waited.total_seconds() == 0
    assert [event[0] for event in strategy.vars.events_at_return] == ["filled"]


def test_wait_for_order_execution_times_out() -> None:
    class NeverFilled(Waiter):
        fill_it = False

    strategy = _strategy(NeverFilled)
    strategy.executor.run()
    assert strategy.vars.result is False
    assert strategy.vars.waited.total_seconds() == 600


def test_keyboard_interrupt_runs_abrupt_closing_then_strategy_end() -> None:
    class Interrupted(OrderHooks):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            raise KeyboardInterrupt

    strategy = _strategy(Interrupted, sessions=2)
    strategy.executor.run()  # returns normally
    assert strategy.milestones == ["on_trading_iteration", "on_abrupt_closing", "on_strategy_end"]
    assert strategy.fake_broker.calls[-1] == "stop_stream"


def test_sigterm_is_handled_like_ctrl_c_and_the_handler_restored() -> None:
    class Terminated(OrderHooks):
        def on_trading_iteration(self) -> None:
            super().on_trading_iteration()
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # interrupted by the handler raising KeyboardInterrupt

    previous = signal.getsignal(signal.SIGTERM)
    strategy = _strategy(Terminated)
    strategy.executor.run()
    assert strategy.milestones == ["on_trading_iteration", "on_abrupt_closing", "on_strategy_end"]
    assert signal.getsignal(signal.SIGTERM) == previous


def test_run_off_the_main_thread_skips_signal_handling() -> None:
    strategy = _strategy()
    errors: list[BaseException] = []

    def target() -> None:
        try:
            strategy.executor.run()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=10)
    assert errors == []
    assert strategy.milestones == ["on_trading_iteration", "on_strategy_end"]


def test_waiting_is_cut_short_by_stop_from_another_thread() -> None:
    strategy = _strategy()
    started: list[datetime] = []

    def stop_during_pre_open_wait() -> None:
        started.append(strategy.get_datetime())
        strategy.stop()

    _on_first_wait(strategy.fake_clock, stop_during_pre_open_wait)
    strategy.executor.run()
    assert strategy.milestones == ["on_strategy_end"]
    assert strategy.get_datetime() == started[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_executor_events.py -q`
Expected: FAIL — order hooks never called, `AttributeError: 'OrderHooks' object has no attribute 'wait_for_order_execution'`, and `KeyboardInterrupt` escaping `run()`.

- [ ] **Step 3: Implement the executor changes**

In `src/trading_agent_framework/strategies/executor.py`, add `import signal` and `from types import FrameType` to the imports, `from trading_agent_framework.entities.enums import OrderEvent`, and extend the events import to `from trading_agent_framework.strategies.events import OrderEventQueue, QueuedOrderEvent`. Update the module docstring's last sentence to also mention that order events are dispatched on the executor thread while it waits.

Replace `run` with:

```python
    def run(self) -> None:
        strategy = self.strategy
        broker = strategy.broker
        self._stop.clear()
        broker.tracker.listeners.append(self._events)
        restore_sigterm = self._install_sigterm_handler()
        initialized = False
        try:
            broker.sync_open_orders()
            broker.start_stream()
            self._initialize()
            initialized = True
            parse_sleeptime(strategy.sleeptime)  # fail fast on a bad sleeptime
            self._run_sessions()
        except KeyboardInterrupt:
            logger.warning("Strategy %s interrupted; running on_abrupt_closing", strategy.name)
            self._call_hook(strategy.on_abrupt_closing)
        finally:
            if initialized:
                self._call_hook(strategy.on_strategy_end)
            broker.stop_stream()
            broker.tracker.listeners.remove(self._events)
            restore_sigterm()
```

Replace `wait_until` with:

```python
    def wait_until(self, deadline: datetime, until: Callable[[], bool] | None = None) -> bool:
        """Wait on the clock until `deadline`, dispatching order events meanwhile.

        Returns True as soon as `until()` holds; False at the deadline or on `stop()`.
        """
        clock = self.strategy.clock
        while True:
            self._dispatch_events()
            if until is not None and until():
                return True
            remaining = (deadline - clock.now()).total_seconds()
            if remaining <= 0 or self._stop.is_set():
                return False
            clock.wait(min(remaining, MAX_WAIT_SLICE_SECONDS), self._wake)
            self._wake.clear()

    def wait_for(self, until: Callable[[], bool], timeout: float | None = None) -> bool:
        """`wait_until` with a relative timeout; no timeout means a one-year horizon."""
        horizon = timedelta(days=365) if timeout is None else timedelta(seconds=timeout)
        return self.wait_until(self._now() + horizon, until)
```

Add to the helpers section:

```python
    def _dispatch_events(self) -> None:
        for item in self._events.drain():
            self._dispatch(item)

    def _dispatch(self, item: QueuedOrderEvent) -> None:
        strategy = self.strategy
        order = item.order
        try:
            if item.event is OrderEvent.NEW:
                strategy.on_new_order(order)
            elif item.event is OrderEvent.CANCELED:
                strategy.on_canceled_order(order)
            elif item.event in (OrderEvent.FILLED, OrderEvent.PARTIALLY_FILLED):
                if item.price is None or item.quantity is None:
                    logger.warning("Fill event for order %s has no fill data", order.identifier)
                    return
                hook = (
                    strategy.on_filled_order
                    if item.event is OrderEvent.FILLED
                    else strategy.on_partially_filled_order
                )
                position = strategy.get_position(order.asset)
                hook(position, order, item.price, item.quantity, 1)
            elif item.event is OrderEvent.ERROR:
                logger.warning(
                    "Order %s for %s was rejected: %s",
                    order.identifier,
                    strategy.name,
                    order.error_message,
                )
        except Exception as exc:
            logger.exception("Order hook failed for %s (%s)", strategy.name, item.event)
            self._on_bot_crash(exc)

    def _install_sigterm_handler(self) -> Callable[[], None]:
        """Turn SIGTERM into KeyboardInterrupt for this run; returns the undo function."""
        if threading.current_thread() is not threading.main_thread():
            return lambda: None  # signal handlers can only be installed on the main thread
        previous = signal.getsignal(signal.SIGTERM)

        def interrupt(signum: int, frame: FrameType | None) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)

        def restore() -> None:
            signal.signal(signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)

        return restore
```

(MODIFIED events need no branch: the tracker already treats them as a no-op.)

- [ ] **Step 4: Add the order waits to `Strategy`**

In `src/trading_agent_framework/strategies/strategy.py`, add `OrderStatus` to the enums import, define below the `Number` alias:

```python
_FINAL_STATUSES = frozenset(
    {OrderStatus.FILL, OrderStatus.CANCELED, OrderStatus.ERROR, OrderStatus.EXPIRED}
)
```

and add to the control section:

```python
    def wait_for_order_execution(self, order: Order, timeout: float | None = None) -> bool:
        """Wait until `order` is filled, canceled, expired or rejected; False on timeout/stop.

        Needs the broker's trade stream (the runners start it). Order hooks keep firing
        while waiting. A modified order is replaced: wait on the order `modify_order` returned.
        """
        return self.wait_for_orders_execution([order], timeout)

    def wait_for_orders_execution(
        self, orders: Sequence[Order], timeout: float | None = None
    ) -> bool:
        return self.executor.wait_for(
            lambda: all(order.status in _FINAL_STATUSES for order in orders), timeout
        )
```

Update the `sleep` docstring to: `"""Pause for `seconds` of clock time; order hooks still fire meanwhile."""`

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass, including every Task 10 test.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies tests/strategies/test_executor_events.py
git commit -m "Task 11: dispatch order events on the executor thread; add order waits and interrupt handling"
```

---

### Task 12: Runners (`run_strategy`, `run_paper_trading`, `run_live_trading`) and package exports

**Files:**
- Modify: `src/trading_agent_framework/strategies/strategy.py`
- Modify: `src/trading_agent_framework/strategies/__init__.py`
- Test: `tests/strategies/test_runners.py`

**Interfaces:**
- Consumes: `setup_strategy_logging` (Task 3), `StrategyExecutor.run` (Tasks 10–11), `ConfigurationError`, `BrokerError`.
- Produces (`Strategy`):
  - `run_strategy() -> None` dispatches on `trading_mode`;
  - `run_paper_trading() -> None` and `run_live_trading() -> None` raise `ConfigurationError` on a mode/account mismatch, then set up logging, log the banner, and run the executor;
  - `run_backtesting() -> None` raises `NotImplementedError`.
- Produces (`trading_agent_framework.strategies`): re-exports `Strategy` and `StrategyExecutor`, listed in `__all__`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_runners.py`:

```python
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et, weekday_sessions

from trading_agent_framework import strategies
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.errors import BrokerError, ConfigurationError
from trading_agent_framework.log import ANSI_BLUE, ANSI_RESET
from trading_agent_framework.strategies import executor as executor_module
from trading_agent_framework.strategies import strategy as strategy_module
from trading_agent_framework.strategies.strategy import Strategy


class Hello(Strategy):
    sleeptime = "1D"

    def on_trading_iteration(self) -> None:
        self.log_info("hello from iteration")


def _strategy(
    tmp_path: Path,
    *,
    is_paper: bool = True,
    mode: TradingMode = TradingMode.PAPER,
    cls: type[Strategy] = Hello,
) -> Strategy:
    clock = FakeClock(et(2026, 9, 14, 4, 30), weekday_sessions(date(2026, 9, 14), 1))
    return cls(FakeBroker(clock, is_paper=is_paper), mode=mode, project_root=tmp_path)


def _log_content(tmp_path: Path, mode: str) -> str:
    [log_file] = (tmp_path / "logs" / "momentum" / mode).glob(f"*_{mode}/{mode}.log")
    return log_file.read_text(encoding="utf-8")


def test_run_paper_trading_logs_the_banner_and_the_iterations(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert isinstance(strategy.broker, FakeBroker)
    strategy.broker.positions = [
        Position(
            strategy_name="momentum",
            asset=Asset("AAPL"),
            quantity=Decimal(10),
            side=PositionSide.LONG,
        )
    ]

    strategy.run_paper_trading()

    content = _log_content(tmp_path, "paper")
    assert "PAPER TRADING MODE" in content
    assert "Broker account: PAPER" in content
    assert "05:00:00 until market opens" in content
    assert "11:30:00 until market closes" in content
    assert "Initial cash: 10000" in content
    assert "Position: 10 AAPL" in content
    assert f"{ANSI_BLUE}hello from iteration{ANSI_RESET}" in content


def test_run_live_trading_writes_the_live_log(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, is_paper=False, mode=TradingMode.LIVE)
    strategy.run_live_trading()
    assert "LIVE TRADING MODE" in _log_content(tmp_path, "live")
    assert strategy.trading_mode is TradingMode.LIVE


def test_live_trading_refuses_a_paper_account(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="live mode against a paper"):
        _strategy(tmp_path).run_live_trading()
    assert not (tmp_path / "logs").exists()


def test_paper_trading_refuses_a_live_account(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="paper mode against a live"):
        _strategy(tmp_path, is_paper=False).run_paper_trading()
    assert not (tmp_path / "logs").exists()


def test_banner_survives_calendar_errors(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert isinstance(strategy.clock, FakeClock)
    strategy.clock.next_session_errors = [BrokerError("calendar down")]

    strategy.run_paper_trading()

    content = _log_content(tmp_path, "paper")
    assert "Market calendar unavailable: calendar down" in content
    assert "hello from iteration" in content


class Dispatch(Strategy):
    def __init__(self, broker: FakeBroker, **kwargs: object) -> None:
        super().__init__(broker, **kwargs)  # ty: ignore[invalid-argument-type]
        self.ran: list[str] = []

    def run_paper_trading(self) -> None:
        self.ran.append("paper")

    def run_live_trading(self) -> None:
        self.ran.append("live")

    def run_backtesting(self) -> None:
        self.ran.append("backtesting")


@pytest.mark.parametrize("mode", list(TradingMode))
def test_run_strategy_dispatches_on_the_trading_mode(tmp_path: Path, mode: TradingMode) -> None:
    strategy = _strategy(tmp_path, mode=mode, cls=Dispatch)
    strategy.run_strategy()
    assert isinstance(strategy, Dispatch)
    assert strategy.ran == [mode.value]


def test_run_backtesting_is_not_implemented_yet(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="backtesting"):
        _strategy(tmp_path, mode=TradingMode.BACKTESTING).run_strategy()


def test_strategies_package_reexports() -> None:
    assert strategies.Strategy is strategy_module.Strategy
    assert strategies.StrategyExecutor is executor_module.StrategyExecutor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/strategies/test_runners.py -q`
Expected: FAIL — `AttributeError: 'Hello' object has no attribute 'run_paper_trading'` and `AttributeError: module 'trading_agent_framework.strategies' has no attribute 'Strategy'`.

- [ ] **Step 3: Implement the runners**

In `src/trading_agent_framework/strategies/strategy.py`, add these imports: `from trading_agent_framework.errors import BrokerError, ConfigurationError` and `from trading_agent_framework.log import ColorLogger, setup_strategy_logging` (replacing the existing `ColorLogger` import). Then append to the class:

```python
    # --- runners (the lumibot-agent `WrappingStrategy` entry points) -------------------

    def run_strategy(self) -> None:
        if self.trading_mode is TradingMode.LIVE:
            self.run_live_trading()
        elif self.trading_mode is TradingMode.PAPER:
            self.run_paper_trading()
        else:
            self.run_backtesting()

    def run_paper_trading(self) -> None:
        self._run_trading(TradingMode.PAPER)

    def run_live_trading(self) -> None:
        self._run_trading(TradingMode.LIVE)

    def run_backtesting(self) -> None:
        raise NotImplementedError(
            "backtesting is not implemented yet; it ships with the backtesting subproject"
        )

    def _run_trading(self, mode: TradingMode) -> None:
        if self.broker.is_paper != (mode is TradingMode.PAPER):
            account_kind = "paper" if self.broker.is_paper else "live"
            raise ConfigurationError(
                f"Refusing to run strategy {self.name!r} in {mode} mode "
                f"against a {account_kind} broker account"
            )
        self.trading_mode = mode
        log_file = setup_strategy_logging(self.name, mode, project_root=self.project_root)
        self._log_startup_banner(mode, log_file)
        self.executor.run()

    def _log_startup_banner(self, mode: TradingMode, log_file: Path) -> None:
        self.log_info(f"======== {mode.value.upper()} TRADING MODE ========")
        self.log_info(f"Logs will be saved to: {log_file.parent}")
        self.log_info(f"Broker account: {'PAPER' if self.broker.is_paper else 'LIVE'}")
        self.log_info(f"Parameters: {dict(self.parameters)}")
        self._log_market_conditions()
        self.log_info(f"Initial cash: {self.get_cash()}")
        for position in self.get_positions():
            self.log_info(f"Position: {position.quantity} {position.asset}")

    def _log_market_conditions(self) -> None:
        try:
            session = self.clock.next_session()
        except BrokerError as exc:
            self.log_warning(f"Market calendar unavailable: {exc}")
            return
        if session is None:
            self.log_warning("No upcoming market session")
            return
        now = self.clock.now()
        if session.open > now:
            self.log_info(f"{_format_duration(session.open - now)} until market opens")
        else:
            self.log_info("Market is open")
        self.log_info(f"{_format_duration(session.close - now)} until market closes")
```

and at module level, after `_to_optional_decimal`:

```python
def _format_duration(delta: timedelta) -> str:
    """HH:MM:SS, with hours allowed past 24 (a weekend wait reads 65:30:00)."""
    total = max(0, int(delta.total_seconds()))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
```

- [ ] **Step 4: Export the public API**

Replace `src/trading_agent_framework/strategies/__init__.py` with:

```python
"""Strategy framework: lifecycle hooks, executor, and the lumibot-style Strategy base."""

from __future__ import annotations

from trading_agent_framework.strategies.executor import StrategyExecutor
from trading_agent_framework.strategies.strategy import Strategy

__all__ = [
    "Strategy",
    "StrategyExecutor",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/strategies tests/strategies/test_runners.py
git commit -m "Task 12: add paper/live runners with account guard and startup banner"
```

---

### Task 13: Manual paper smoke script and documentation

**Files:**
- Create: `scripts/tests/smoke_strategy_paper.py`
- Modify: `scripts/tests/smoke_alpaca_orders.py` (project-root path bug)
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything above; `env/.env.alpaca.integration-tests` (the existing git-ignored paper credentials file).
- Produces: a runnable manual check, `uv run python scripts/tests/smoke_strategy_paper.py`. It exits with 0 on PASS and 1 on FAIL. It is not part of `pytest`.

- [ ] **Step 1: Fix the existing smoke script's project root**

`scripts/tests/smoke_alpaca_orders.py` sits two levels below the repo root, but it computes `PROJECT_ROOT = Path(__file__).resolve().parent.parent`, which is `scripts/`, so its `ENV_FILE` points at `scripts/env/...`. Replace that line with:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
```

Then update the usage line in its docstring to `uv run python scripts/tests/smoke_alpaca_orders.py`.

- [ ] **Step 2: Write the strategy smoke script**

Create `scripts/tests/smoke_strategy_paper.py`:

```python
#!/usr/bin/env python3
"""Manual paper-trading sanity check for the strategy framework.

NOT part of the automated test suite (the suite never touches the network).
Run by hand with real *paper* Alpaca credentials:

    uv run python scripts/tests/smoke_strategy_paper.py

Credentials come from env/.env.alpaca.integration-tests, the same git-ignored
file smoke_alpaca_orders.py uses. A script-local always-open clock replaces
Alpaca's calendar so the check also runs outside market hours. Outside regular
hours Alpaca may refuse to replace a queued order; the script then reports FAIL
with the broker's message -- rerun during regular hours before suspecting code.

The strategy (sleeptime 30S):
  1. iteration 1 -- logs cash and portfolio value, submits a 1-share SPY limit
     buy far below market so it rests
  2. iteration 2 -- modifies that order's limit price
  3. iteration 3 -- cancels it, waits for the cancel, then stops the run

PASS requires three iterations, a successful modify, no hook errors,
on_new_order and on_canceled_order to have fired, and on_strategy_end last. The run log lands in
logs/smoke/paper/<timestamp>_paper/paper.log (open it with VS Code's ANSI
Colors plugin). Refuses to run against a live account.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.config.env import AlpacaCredentials, TradingMode, find_project_root
from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import TradingFrameworkError
from trading_agent_framework.strategies import Strategy

PROJECT_ROOT = find_project_root()
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke"
SYMBOL = "SPY"
RESTING_LIMIT_PRICE = Decimal("100.00")  # nowhere near SPY's real price
MODIFIED_LIMIT_PRICE = Decimal("101.00")
CANCEL_TIMEOUT_SECONDS = 20


class AlwaysOpenClock(MarketClock):
    """One session opening now and closing in an hour, whatever the real calendar says."""

    def __init__(self) -> None:
        start = self.now()
        self._session = MarketSession(open=start, close=start + timedelta(hours=1))

    def next_session(self) -> MarketSession | None:
        return self._session if self._session.close > self.now() else None


class SmokeStrategy(Strategy):
    sleeptime = "30S"
    minutes_before_closing = 0

    def initialize(self) -> None:
        self.vars.hooks = []
        self.vars.errors = []
        self.vars.order = None
        self.vars.modified = False
        self.vars.cancel_confirmed = False

    def _record(self, hook: str) -> None:
        self.vars.hooks.append(hook)
        self.log_info(f"hook: {hook}")

    def on_trading_iteration(self) -> None:
        self._record("on_trading_iteration")
        iteration = self.vars.hooks.count("on_trading_iteration")
        if iteration == 1:
            self.log_info(f"cash={self.get_cash()} portfolio_value={self.get_portfolio_value()}")
            order = self.create_order(SYMBOL, 1, OrderSide.BUY, limit_price=RESTING_LIMIT_PRICE)
            self.vars.order = self.submit_order(order)
        elif iteration == 2:
            self.vars.order = self.modify_order(self.vars.order, limit_price=MODIFIED_LIMIT_PRICE)
            self.vars.modified = True
        elif iteration == 3:
            self.cancel_order(self.vars.order)
            self.vars.cancel_confirmed = self.wait_for_order_execution(
                self.vars.order, timeout=CANCEL_TIMEOUT_SECONDS
            )
            self.stop()

    def on_new_order(self, order: Order) -> None:
        self._record("on_new_order")

    def on_canceled_order(self, order: Order) -> None:
        self._record("on_canceled_order")

    def on_strategy_end(self) -> None:
        self._record("on_strategy_end")

    def on_bot_crash(self, error: BaseException) -> None:
        # The executor catches hook failures and keeps trading; record them so they fail the run.
        self.vars.errors.append(repr(error))
        self.log_error(f"hook failed: {error!r}")


def _failures(strategy: SmokeStrategy) -> list[str]:
    hooks: list[str] = strategy.vars.hooks
    failures = []
    if strategy.vars.errors:
        failures.append(f"hook errors: {strategy.vars.errors}")
    if hooks.count("on_trading_iteration") != 3:
        failures.append(f"expected 3 iterations, got {hooks.count('on_trading_iteration')}")
    if not strategy.vars.modified:
        failures.append("modify_order did not succeed")
    for hook in ("on_new_order", "on_canceled_order"):
        if hook not in hooks:
            failures.append(f"{hook} never fired")
    if not strategy.vars.cancel_confirmed:
        failures.append("cancel was not confirmed by the trade stream")
    if not hooks or hooks[-1] != "on_strategy_end":
        failures.append("on_strategy_end did not run last")
    return failures


def main() -> int:
    load_dotenv(ENV_FILE, override=True)
    creds = AlpacaCredentials.from_env()
    if not creds.is_paper:
        print("Refusing to run: ALPACA_IS_PAPER is false (live account).", file=sys.stderr)
        return 1

    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds)
    strategy = SmokeStrategy(
        broker, mode=TradingMode.PAPER, clock=AlwaysOpenClock(), project_root=PROJECT_ROOT
    )
    started = datetime.now()
    try:
        strategy.run_paper_trading()
    except TradingFrameworkError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        order = strategy.vars.__dict__.get("order")
        if order is not None and order.is_active():
            broker.cancel_order(order)  # never leave a resting order behind

    failures = _failures(strategy)
    elapsed = (datetime.now() - started).total_seconds()
    if failures:
        print(f"FAIL after {elapsed:.0f}s: " + "; ".join(failures), file=sys.stderr)
        print(f"hooks: {strategy.vars.hooks}", file=sys.stderr)
        return 1
    print(f"PASS after {elapsed:.0f}s: hooks {strategy.vars.hooks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Lint and type-check the script**

Run: `uv run ruff check scripts && uv check`
Expected: clean. (`uv run pytest -q` must still pass untouched.)

- [ ] **Step 4: Update `CLAUDE.md`**

In the **Commands** block, replace the smoke line with:

```bash
uv run python scripts/tests/smoke_alpaca_orders.py     # manual paper-trading smoke test: broker orders
uv run python scripts/tests/smoke_strategy_paper.py    # manual paper-trading smoke test: strategy lifecycle
```

In **Architecture**:
- change the `brokers/alpaca/orders.py` bullet's last sentence to "Together with `account.py`, the only modules allowed to import `alpaca.trading.requests`."
- add these bullets:

```markdown
- `clock.py` -- `MarketClock` ABC + `MarketSession`: the executor's only source of time and waiting (the seam a future backtest clock plugs into).
- `brokers/alpaca/account.py` -- **pure** account and calendar translation (same rules as `orders.py`).
- `brokers/alpaca/clock.py` -- `AlpacaMarketClock`: sessions (early closes included) from Alpaca's calendar, cached ~10 trading days.
- `strategies/` -- `Strategy` (lumibot hook names/signatures, broker facade, paper/live runners), `StrategyExecutor` (single-threaded session loop), `timing.py` (pure `sleeptime` parsing and tick maths), `events.py` (stream-thread → executor-thread order-event queue).
- `log.py` -- `ColorLogger` (`log_info`/`log_warning`/... with ANSI colours) and `setup_strategy_logging` (lumibot-style `logs/<strategy>/<mode>/<ts>_<mode>/<mode>.log`).
```

In **Key patterns / gotchas**, change "**`orders.py` stays pure.**" to "**`orders.py` and `account.py` stay pure.**" and add:

```markdown
- **Strategy code runs on one thread.** Order hooks (`on_filled_order`, ...) are dispatched by the executor while it waits (between ticks, in `strategy.sleep`, in `wait_for_order_execution`) -- never on the Alpaca stream thread. Broker/stream code must only feed `OrderTracker`; it never calls strategy hooks.
- **No `time.sleep` / `datetime.now` in strategy or executor code** -- go through `strategy.sleep()` / `strategy.clock` so a simulated clock can drive backtests later.
- **ANSI colour codes stay in log files on purpose** (read with VS Code's "ANSI Colors" plugin). Don't strip them and don't switch to `termcolor` (it drops colour when stdout isn't a TTY).
- `strategy.name` is `broker.strategy_name` (it prefixes every `client_order_id` and decides which open orders `sync_open_orders` adopts after a restart).
- `tests/conftest.py` resets package logging after every test; `setup_strategy_logging` disables propagation, which would otherwise starve other tests' `caplog`.
```

- [ ] **Step 5: Run the full verification**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all pass.

Then — only if paper credentials are available in `env/.env.alpaca.integration-tests` — run `uv run python scripts/tests/smoke_strategy_paper.py` and expect `PASS after ~60-90s`. Open the printed run's `logs/smoke/paper/<ts>_paper/paper.log` in VS Code and check the lines are coloured by the ANSI Colors plugin. If credentials are not available, say so in the task report instead of claiming the smoke test passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/tests CLAUDE.md
git commit -m "Task 13: add strategy paper smoke script; document the strategy layer"
```


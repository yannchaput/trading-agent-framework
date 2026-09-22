# IBKR Broker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Interactive Brokers (TWS API via IB Gateway) a second paper/live broker next to Alpaca, selected by `BROKER=alpaca|ibkr` in the strategy env file, with market data, calendar and news always coming from Alpaca.

**Architecture:** Sub-project A reworks configuration (explicit credential groups, `BROKER`/`BROKER_API_IS_PAPER`), extracts Alpaca market data into a reusable `AlpacaMarketData`, adds a `build_broker` factory, and stops backtests from building a live broker (`PlaceholderBroker`). Sub-project B adds `brokers/ibkr/`: pure translation modules (`orders.py`, `account.py`), an `IbkrConnection` that owns `ib_async.IB` on its own event-loop thread, order-event handlers feeding `OrderTracker`, and `IbkrBroker`, which composes IBKR trading with `AlpacaMarketData`, `AlpacaMarketClock` and `AlpacaNewsProvider`.

**Tech Stack:** Python 3.14, `uv`, `pytest`, `ruff`, `alpaca-py` 0.44, `ib_async` 2.1 (and its `eventkit` dependency).

**Spec:** `docs/superpowers/specs/2026-09-22-ibkr-broker-design.md`

## Global Constraints

- Env variable names are exactly: `BROKER`, `BROKER_API_IS_PAPER`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `ALPACA_NEWS_API_KEY`, `ALPACA_NEWS_API_SECRET`, `ALPACA_DATA_API_KEY`, `ALPACA_DATA_API_SECRET`, `ALPACA_DATA_IS_PAPER`.
- Defaults: `BROKER=alpaca`, `BROKER_API_IS_PAPER=true`, `ALPACA_DATA_IS_PAPER=true`, `IBKR_HOST=127.0.0.1`, `IBKR_PORT=4002` (paper) / `4001` (live), `IBKR_CLIENT_ID=1`.
- `ALPACA_IS_PAPER` present in the environment raises `ConfigurationError("ALPACA_IS_PAPER was renamed to BROKER_API_IS_PAPER; rename it in your env file")`. No alias.
- Credential groups never fall back to each other.
- `ib_async` is imported only by `brokers/ibkr/orders.py` and `brokers/ibkr/account.py` (module level), `brokers/ibkr/client.py` (inside `_default_ib_factory`), and `brokers/ibkr/broker.py` (under `TYPE_CHECKING` only). `brokers/ibkr/events.py` does not import it. `import trading_agent_framework.brokers` never imports it.
- `brokers/ibkr/orders.py` and `brokers/ibkr/account.py` are pure: no I/O, no state, no client instances.
- `brokers/ibkr/orders.py` is the IBKR counterpart of the Alpaca `orders.py` float boundary (`ib_async` order fields are floats). No other new float boundary: convert with `Decimal(str(x))` via `orders.to_decimal`.
- No raw `ib_async`/SDK exception escapes a broker: wrap in `BrokerError` / `OrderValidationError` / `ConfigurationError`. `order.set_error(...)` happens BEFORE re-raising on a failed submit.
- Broker/event code never calls strategy hooks; it only feeds `OrderTracker`.
- Tests: hand-written fakes (no `MagicMock`), no network. `ib_async` objects in tests are real `ib_async` dataclasses/namedtuples.
- Ruff config: `line-length = 200`, rules `E, F, I, UP, B` (no naming rules: camelCase fake methods need no `noqa`).
- Before every commit: `uv run pytest && uv run ruff check`, both passing.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Never edit the user's real `env/.env.*` files (they hold secrets). Only `env/.env.example` is edited.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/trading_agent_framework/config/env.py` | modify | `BrokerKind`, `BrokerSettings`, `IbkrSettings`, `AlpacaCredentials.for_trading/for_news/for_data`, rename guard |
| `src/trading_agent_framework/config/__init__.py` | modify | re-export new names |
| `src/trading_agent_framework/brokers/alpaca/data.py` | create | `AlpacaMarketData`: prices, quotes, bars, calendar client |
| `src/trading_agent_framework/brokers/alpaca/news.py` | modify | `lazy_news_provider()` factory |
| `src/trading_agent_framework/brokers/alpaca/broker.py` | modify | delegate market data; `from_credentials(trading, data, news)` |
| `src/trading_agent_framework/backtesting/placeholder.py` | create | `PlaceholderBroker` |
| `src/trading_agent_framework/brokers/factory.py` | create | `build_broker()` |
| `src/trading_agent_framework/main.py` | modify | factory for paper/live, placeholder for backtesting |
| `src/trading_agent_framework/brokers/ibkr/__init__.py` | create | package docstring |
| `src/trading_agent_framework/brokers/ibkr/orders.py` | create | pure order/contract/status/position translation |
| `src/trading_agent_framework/brokers/ibkr/account.py` | create | pure account-summary parsing and start-up checks |
| `src/trading_agent_framework/brokers/ibkr/client.py` | create | `IbkrConnection` (loop thread, call timeout, reconnect) |
| `src/trading_agent_framework/brokers/ibkr/events.py` | create | `IbkrOrderEvents` (status/fill/error handlers) |
| `src/trading_agent_framework/brokers/ibkr/broker.py` | create | `IbkrBroker` |
| `src/trading_agent_framework/brokers/__init__.py` | modify | lazy `IbkrBroker` export |
| `tests/fakes.py` | modify | `ib_async` object builders and `FakeIB` |
| `scripts/tests/smoke_ibkr_account.py`, `scripts/tests/smoke_ibkr_orders.py` | create | manual paper checks |
| `env/.env.example`, `README.md`, `CLAUDE.md` | modify | docs |

---

### Task 1: Env settings and explicit Alpaca credential groups

**Files:**
- Modify: `src/trading_agent_framework/config/env.py:27-115`
- Modify: `src/trading_agent_framework/config/__init__.py`
- Modify: `src/trading_agent_framework/main.py:60` (temporary `for_trading()`)
- Modify: `src/trading_agent_framework/backtesting/data/alpaca.py:209-221`
- Modify: `src/trading_agent_framework/backtesting/broker.py:133-146`
- Modify: `src/trading_agent_framework/core/strategy.py:496-527` (docstring only)
- Test: `tests/config/test_env.py`, `tests/config/test_config_public_api.py`, `tests/backtesting/data/test_alpaca.py:384-420`, `tests/backtesting/test_broker_news.py:30-60`

**Interfaces:**
- Produces:
  - `BrokerKind(StrEnum)`: `ALPACA = "alpaca"`, `IBKR = "ibkr"`
  - `BrokerSettings(kind: BrokerKind, is_paper: bool)`; `BrokerSettings.from_env(env: Mapping[str, str] | None = None) -> BrokerSettings`
  - `IbkrSettings(host: str, port: int, client_id: int, is_paper: bool)`; `IbkrSettings.from_env(is_paper: bool, env: Mapping[str, str] | None = None) -> IbkrSettings`
  - `IBKR_PAPER_PORT = 4002`, `IBKR_LIVE_PORT = 4001`
  - `AlpacaCredentials.for_trading(env=None)`, `.for_news(env=None)`, `.for_data(env=None)`, each `-> AlpacaCredentials`. `AlpacaCredentials.from_env` is REMOVED.

Note: `scripts/tests/*.py` still call `AlpacaCredentials.from_env()` after this task; Task 2 migrates them (they are not part of the test suite).

- [ ] **Step 1: Replace the `AlpacaCredentials.from_env` tests with failing tests for the new API**

In `tests/config/test_env.py`, delete every test that calls `AlpacaCredentials.from_env` (the `test_from_env_*` tests, around lines 90-130) and change the import block to:

```python
from trading_agent_framework.config.env import (
    IBKR_LIVE_PORT,
    IBKR_PAPER_PORT,
    TRADING_MODES,
    AlpacaCredentials,
    BrokerKind,
    BrokerSettings,
    FredCredentials,
    IbkrSettings,
    TradingMode,
    load_strategy_env,
)
```

Append:

```python
# --- broker selection ---------------------------------------------------------------


def test_broker_settings_default_to_alpaca_paper() -> None:
    assert BrokerSettings.from_env({}) == BrokerSettings(kind=BrokerKind.ALPACA, is_paper=True)


@pytest.mark.parametrize("raw", ["ibkr", "IBKR", " ibkr "])
def test_broker_settings_read_broker_case_insensitively(raw: str) -> None:
    assert BrokerSettings.from_env({"BROKER": raw}).kind is BrokerKind.IBKR


def test_broker_settings_reject_an_unknown_broker() -> None:
    with pytest.raises(ConfigurationError, match="Unknown BROKER 'schwab'"):
        BrokerSettings.from_env({"BROKER": "schwab"})


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("0", False), ("no", False), ("true", True), ("anything", True)])
def test_broker_api_is_paper_is_fail_safe_toward_paper(raw: str, expected: bool) -> None:
    assert BrokerSettings.from_env({"BROKER_API_IS_PAPER": raw}).is_paper is expected


@pytest.mark.parametrize(
    "read",
    [
        BrokerSettings.from_env,
        AlpacaCredentials.for_trading,
        AlpacaCredentials.for_news,
        AlpacaCredentials.for_data,
        lambda env: IbkrSettings.from_env(True, env),
    ],
)
def test_the_renamed_alpaca_is_paper_variable_is_rejected_everywhere(read) -> None:
    env = {"ALPACA_IS_PAPER": "true", "ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}

    with pytest.raises(ConfigurationError, match="ALPACA_IS_PAPER was renamed to BROKER_API_IS_PAPER"):
        read(env)


# --- Alpaca credential groups --------------------------------------------------------


def test_trading_credentials_read_alpaca_api_and_the_broker_paper_flag() -> None:
    env = {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "BROKER_API_IS_PAPER": "false"}

    creds = AlpacaCredentials.for_trading(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("k", "s", False)


def test_news_credentials_read_only_the_news_pair() -> None:
    env = {"ALPACA_NEWS_API_KEY": "nk", "ALPACA_NEWS_API_SECRET": "ns", "ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}

    creds = AlpacaCredentials.for_news(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("nk", "ns", True)


def test_data_credentials_read_the_data_pair_and_their_own_paper_flag() -> None:
    env = {"ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds", "ALPACA_DATA_IS_PAPER": "no", "BROKER_API_IS_PAPER": "true"}

    creds = AlpacaCredentials.for_data(env)

    assert (creds.api_key, creds.api_secret, creds.is_paper) == ("dk", "ds", False)


def test_data_credentials_default_to_paper() -> None:
    assert AlpacaCredentials.for_data({"ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}).is_paper is True


@pytest.mark.parametrize(
    ("read", "missing"),
    [
        (AlpacaCredentials.for_trading, "ALPACA_API_KEY / ALPACA_API_SECRET"),
        (AlpacaCredentials.for_news, "ALPACA_NEWS_API_KEY / ALPACA_NEWS_API_SECRET"),
        (AlpacaCredentials.for_data, "ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET"),
    ],
)
def test_each_group_names_its_own_missing_variables(read, missing: str) -> None:
    with pytest.raises(ConfigurationError, match=missing):
        read({})


def test_groups_never_fall_back_to_the_trading_pair() -> None:
    with pytest.raises(ConfigurationError, match="ALPACA_NEWS_API_KEY"):
        AlpacaCredentials.for_news({"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"})


def test_a_blank_secret_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="ALPACA_API_SECRET"):
        AlpacaCredentials.for_trading({"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "   "})


# --- IBKR settings -------------------------------------------------------------------


def test_ibkr_settings_defaults_depend_on_the_paper_flag() -> None:
    assert IbkrSettings.from_env(True, {}) == IbkrSettings(host="127.0.0.1", port=IBKR_PAPER_PORT, client_id=1, is_paper=True)
    assert IbkrSettings.from_env(False, {}).port == IBKR_LIVE_PORT


def test_ibkr_settings_read_host_port_and_client_id() -> None:
    env = {"IBKR_HOST": "10.0.0.5", "IBKR_PORT": "7497", "IBKR_CLIENT_ID": "7"}

    assert IbkrSettings.from_env(True, env) == IbkrSettings(host="10.0.0.5", port=7497, client_id=7, is_paper=True)


@pytest.mark.parametrize("variable", ["IBKR_PORT", "IBKR_CLIENT_ID"])
def test_ibkr_settings_reject_non_integer_numbers(variable: str) -> None:
    with pytest.raises(ConfigurationError, match=f"{variable} must be an integer"):
        IbkrSettings.from_env(True, {variable: "abc"})
```

Keep `test_repr_does_not_leak_api_key_or_secret` and every `load_strategy_env`/`FredCredentials` test unchanged.

In `tests/config/test_config_public_api.py`, after the existing `AlpacaCredentials` assertion add:

```python
    assert config.BrokerKind is env_module.BrokerKind
    assert config.BrokerSettings is env_module.BrokerSettings
    assert config.IbkrSettings is env_module.IbkrSettings
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/config -v`
Expected: FAIL with `ImportError: cannot import name 'IBKR_LIVE_PORT'` (or `BrokerKind`).

- [ ] **Step 3: Implement the settings in `config/env.py`**

Replace line 27 (`_FALSE_PAPER_VALUES = ...`) with:

```python
_FALSE_PAPER_VALUES = frozenset({"false", "0", "no"})

# Old name -> new name. Still present = a stale env file: fail loudly, no alias.
_RENAMED_VARIABLES = {"ALPACA_IS_PAPER": "BROKER_API_IS_PAPER"}

IBKR_PAPER_PORT = 4002  # IB Gateway's default paper-trading API port
IBKR_LIVE_PORT = 4001  # IB Gateway's default live-trading API port
```

Keep `find_project_root`, `resolve_env_file` and `load_strategy_env` unchanged. Replace the whole `AlpacaCredentials` class (lines 90-115) with:

```python
def _reject_renamed(source: Mapping[str, str]) -> None:
    for old, new in _RENAMED_VARIABLES.items():
        if old in source:
            raise ConfigurationError(f"{old} was renamed to {new}; rename it in your env file")


def _is_paper(raw: str | None) -> bool:
    """Fail-safe toward paper: only an explicit false/0/no means live."""
    return (raw or "true").strip().lower() not in _FALSE_PAPER_VALUES


def _int_setting(source: Mapping[str, str], name: str, default: int) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from None


class BrokerKind(StrEnum):
    """Which broker trades in paper/live mode (`BROKER`); meaningless in backtesting."""

    ALPACA = "alpaca"
    IBKR = "ibkr"


@dataclass(frozen=True, slots=True)
class BrokerSettings:
    kind: BrokerKind
    is_paper: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BrokerSettings:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        raw_kind = (source.get("BROKER") or BrokerKind.ALPACA.value).strip().lower()
        try:
            kind = BrokerKind(raw_kind)
        except ValueError:
            valid = ", ".join(k.value for k in BrokerKind)
            raise ConfigurationError(f"Unknown BROKER {raw_kind!r}; expected one of: {valid}") from None
        return cls(kind=kind, is_paper=_is_paper(source.get("BROKER_API_IS_PAPER")))


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """One Alpaca key pair. Each component reads its OWN group -- trading, news or data --
    and groups never fall back to each other."""

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    is_paper: bool = True

    @classmethod
    def for_trading(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_API_KEY`/`ALPACA_API_SECRET`: the Alpaca trading broker (`BROKER=alpaca`)."""
        return cls._read(env, "ALPACA_API", paper_variable="BROKER_API_IS_PAPER")

    @classmethod
    def for_news(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_NEWS_API_*`: the news tool, in every mode and for every broker."""
        return cls._read(env, "ALPACA_NEWS_API", paper_variable=None)

    @classmethod
    def for_data(cls, env: Mapping[str, str] | None = None) -> AlpacaCredentials:
        """`ALPACA_DATA_API_*`: prices, bars and the calendar (paper/live, both brokers) and
        `AlpacaBacktestData`. `ALPACA_DATA_IS_PAPER` picks the endpoint the calendar call uses."""
        return cls._read(env, "ALPACA_DATA_API", paper_variable="ALPACA_DATA_IS_PAPER")

    @classmethod
    def _read(cls, env: Mapping[str, str] | None, prefix: str, *, paper_variable: str | None) -> AlpacaCredentials:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        key_name, secret_name = f"{prefix}_KEY", f"{prefix}_SECRET"
        api_key = source.get(key_name) or ""
        api_secret = source.get(secret_name) or ""
        if not api_key.strip() or not api_secret.strip():
            raise ConfigurationError(f"Missing or blank {key_name} / {secret_name} environment variables")
        is_paper = True if paper_variable is None else _is_paper(source.get(paper_variable))
        return cls(api_key=api_key, api_secret=api_secret, is_paper=is_paper)


@dataclass(frozen=True, slots=True)
class IbkrSettings:
    """Where IB Gateway listens. No key/secret: the Gateway holds the IBKR login."""

    host: str
    port: int
    client_id: int
    is_paper: bool

    @classmethod
    def from_env(cls, is_paper: bool, env: Mapping[str, str] | None = None) -> IbkrSettings:
        source = env if env is not None else os.environ
        _reject_renamed(source)
        host = (source.get("IBKR_HOST") or "127.0.0.1").strip()
        port = _int_setting(source, "IBKR_PORT", IBKR_PAPER_PORT if is_paper else IBKR_LIVE_PORT)
        client_id = _int_setting(source, "IBKR_CLIENT_ID", 1)
        return cls(host=host, port=port, client_id=client_id, is_paper=is_paper)
```

In `src/trading_agent_framework/config/__init__.py`, import `BrokerKind`, `BrokerSettings`, `IbkrSettings` from `config.env` and add them to `__all__` next to `AlpacaCredentials`.

- [ ] **Step 4: Migrate the in-package callers of the removed `from_env`**

- `src/trading_agent_framework/main.py:60`: `creds = AlpacaCredentials.for_trading()` (temporary; Tasks 2-4 replace this block).
- `src/trading_agent_framework/backtesting/data/alpaca.py`, in `_real_client` and `_real_trading_client`: `AlpacaCredentials.for_data()` instead of `AlpacaCredentials.from_env()`.
- `src/trading_agent_framework/backtesting/broker.py:143`: `AlpacaNewsProvider.from_credentials(AlpacaCredentials.for_news())`; its docstring (line 134) becomes "The injected news source, else an Alpaca provider built from the `ALPACA_NEWS_*` credentials on first use."
- `src/trading_agent_framework/core/strategy.py`, `run_backtesting` docstring: replace "`AlpacaCredentials.from_env()`" with "`AlpacaCredentials.for_data()` (`ALPACA_DATA_*`)" in the `data_source` paragraph and with "`AlpacaCredentials.for_news()` (`ALPACA_NEWS_*`)" in the `news_source` paragraph.

Then run `grep -rn "AlpacaCredentials.from_env\|ALPACA_IS_PAPER" src tests` and fix every remaining hit in `src/` and `tests/` (scripts are Task 2; the rename guard in `config/env.py` and its tests are expected hits).

- [ ] **Step 5: Update the two env-reading tests**

`tests/backtesting/data/test_alpaca.py::test_default_clients_are_built_from_env_credentials_when_not_injected`: replace its two `setenv` lines with

```python
    monkeypatch.delenv("ALPACA_IS_PAPER", raising=False)
    monkeypatch.setenv("ALPACA_DATA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_DATA_API_SECRET", "secret")
```

and "`AlpacaCredentials.from_env()`" in its docstring with "`AlpacaCredentials.for_data()`".

`tests/backtesting/test_broker_news.py`:
- `test_the_default_news_source_is_built_lazily_from_env_credentials_and_memoized`: set `ALPACA_NEWS_API_KEY`/`ALPACA_NEWS_API_SECRET` instead of the old names, plus `monkeypatch.delenv("ALPACA_IS_PAPER", raising=False)`.
- `test_missing_alpaca_credentials_surface_as_a_broker_error`: delete `ALPACA_NEWS_API_KEY`/`ALPACA_NEWS_API_SECRET` (and `ALPACA_IS_PAPER`) instead of the old names.

- [ ] **Step 6: Run the suite and lint**

Run: `uv run pytest && uv run ruff check`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/config src/trading_agent_framework/main.py src/trading_agent_framework/backtesting src/trading_agent_framework/core/strategy.py tests/config tests/backtesting
git commit -m "feat: BROKER/BROKER_API_IS_PAPER settings and explicit Alpaca credential groups (Task 1)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Extract `AlpacaMarketData`; `AlpacaBroker.from_credentials(trading, data, news)`

**Files:**
- Create: `src/trading_agent_framework/brokers/alpaca/data.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/news.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/broker.py` (constructor, `from_credentials`, market-data section lines 251-341, `news_provider`, `get_news`)
- Modify: `src/trading_agent_framework/main.py:60-64`
- Modify: `scripts/tests/smoke_alpaca_orders.py`, `smoke_alpaca_data.py`, `smoke_news.py`, `smoke_macro.py`, `smoke_fundamentals.py`, `smoke_strategy_paper.py`, `smoke_strategy_news.py`
- Test: `tests/brokers/alpaca/test_alpaca_data.py` (create), `tests/brokers/alpaca/test_broker_market_data.py`, `tests/brokers/alpaca/test_broker_account.py`, `tests/brokers/alpaca/test_alpaca_news.py`

**Interfaces:**
- Consumes: `AlpacaCredentials.for_trading/for_data/for_news` (Task 1).
- Produces:
  - `AlpacaMarketData(data_client: market_data.AlpacaStockDataClient | None, calendar_client: orders.AlpacaTradingClient)`
  - `AlpacaMarketData.from_credentials(creds: AlpacaCredentials) -> AlpacaMarketData`
  - `AlpacaMarketData.calendar_client` property
  - `AlpacaMarketData.get_last_price(asset) -> Decimal | None`, `.get_last_prices(assets) -> dict[Asset, Decimal | None]`, `.get_quote(asset) -> Quote | None`, `.get_bars(assets, length, timestep="day", *, end: datetime, include_after_hours=True) -> dict[Asset, Bars]`
  - `lazy_news_provider(credentials: Callable[[], AlpacaCredentials]) -> Callable[[], AlpacaNewsProvider]` (in `brokers/alpaca/news.py`); missing credentials raise `BrokerError("no news source available: ...")`
  - `AlpacaBroker.__init__(..., data_client=None, news_client=None, market_data: AlpacaMarketData | None = None, news_provider_factory: Callable[[], NewsProvider] | None = None)`
  - `AlpacaBroker.from_credentials(strategy_name, *, trading: AlpacaCredentials, data: AlpacaCredentials, news: Callable[[], AlpacaCredentials] | None = None, with_stream: bool = True) -> AlpacaBroker`

- [ ] **Step 1: Write failing tests for `AlpacaMarketData`**

Create `tests/brokers/alpaca/test_alpaca_data.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import (
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    bar_payload,
    et,
    make_alpaca_calendar,
    make_alpaca_trade,
)

from trading_agent_framework.brokers.alpaca import data as data_module
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.utils.errors import BrokerError

AAPL = Asset("AAPL")
_NOW = et(2026, 9, 10, 9, 32)


def test_get_bars_ends_at_the_given_end_and_windows_on_the_calendar_client() -> None:
    data = FakeStockHistoricalDataClient()
    data.bars = {"AAPL": [bar_payload("2026-09-09T04:00:00Z", 100.0), bar_payload("2026-09-10T04:00:00Z", 101.0)]}
    calendar = FakeTradingClient()
    calendar.calendar_response = [make_alpaca_calendar("2026-09-09"), make_alpaca_calendar("2026-09-10")]

    bars = AlpacaMarketData(data, calendar).get_bars([AAPL], 2, "day", end=_NOW)

    assert AAPL in bars
    [request] = data.bars_requests
    assert request.end == _NOW
    assert calendar.calendar_requests  # the session window came from the calendar client


def test_get_last_price_reads_the_latest_trade() -> None:
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.15)}

    assert AlpacaMarketData(data, FakeTradingClient()).get_last_price(AAPL) == Decimal("100.15")


def test_market_data_without_a_data_client_raises() -> None:
    with pytest.raises(BrokerError, match="no market data client"):
        AlpacaMarketData(None, FakeTradingClient()).get_last_price(AAPL)


def test_from_credentials_builds_both_clients_from_the_data_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, AlpacaCredentials]] = []
    calendar = FakeTradingClient()

    def fake_data(creds: AlpacaCredentials) -> FakeStockHistoricalDataClient:
        seen.append(("data", creds))
        return FakeStockHistoricalDataClient()

    def fake_trading(creds: AlpacaCredentials) -> FakeTradingClient:
        seen.append(("calendar", creds))
        return calendar

    monkeypatch.setattr(data_module, "build_stock_data_client", fake_data)
    monkeypatch.setattr(data_module, "build_trading_client", fake_trading)
    creds = AlpacaCredentials(api_key="dk", api_secret="ds", is_paper=True)

    market_data = AlpacaMarketData.from_credentials(creds)

    assert seen == [("data", creds), ("calendar", creds)]
    assert market_data.calendar_client is calendar
```

Append to `tests/brokers/alpaca/test_alpaca_news.py` (add `FakeNewsClient`, `AlpacaNewsProvider`, `AlpacaCredentials` to its imports if missing):

```python
def test_lazy_news_provider_resolves_credentials_only_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.alpaca import news as news_module
    from trading_agent_framework.brokers.alpaca.news import lazy_news_provider

    calls: list[str] = []
    monkeypatch.setattr(news_module, "build_news_client", lambda creds: FakeNewsClient())

    def creds() -> AlpacaCredentials:
        calls.append("read")
        return AlpacaCredentials(api_key="k", api_secret="s")

    factory = lazy_news_provider(creds)
    assert calls == []
    assert isinstance(factory(), AlpacaNewsProvider)
    assert calls == ["read"]


def test_lazy_news_provider_turns_missing_credentials_into_a_broker_error() -> None:
    from trading_agent_framework.brokers.alpaca.news import lazy_news_provider
    from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

    def missing() -> AlpacaCredentials:
        raise ConfigurationError("Missing or blank ALPACA_NEWS_API_KEY / ALPACA_NEWS_API_SECRET environment variables")

    with pytest.raises(BrokerError, match="no news source available: Missing or blank ALPACA_NEWS_API_KEY"):
        lazy_news_provider(missing)()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_alpaca_data.py tests/brokers/alpaca/test_alpaca_news.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.alpaca.data'` and `ImportError: cannot import name 'lazy_news_provider'`.

- [ ] **Step 3: Create `brokers/alpaca/data.py` by moving the market-data code out of `broker.py`**

Start the file with:

```python
"""`AlpacaMarketData`: Alpaca IEX prices, quotes and bars, plus the calendar they are windowed on.

Extracted from `AlpacaBroker` so a broker that trades elsewhere (`IbkrBroker`) reuses the exact
same data path. I/O only: every request and parse lives in the pure `market_data`/`account` modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, cast

from trading_agent_framework.brokers.alpaca import account, market_data, orders
from trading_agent_framework.brokers.alpaca.client import build_stock_data_client, build_trading_client
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials


class AlpacaMarketData:
    def __init__(
        self,
        data_client: market_data.AlpacaStockDataClient | None,
        calendar_client: orders.AlpacaTradingClient,
    ) -> None:
        self._data_client = data_client
        self._calendar_client = calendar_client

    @classmethod
    def from_credentials(cls, creds: AlpacaCredentials) -> AlpacaMarketData:
        """Both clients from the `ALPACA_DATA_*` credentials (the calendar lives on the trading API)."""
        data_client = cast("market_data.AlpacaStockDataClient", build_stock_data_client(creds))
        calendar_client = cast("orders.AlpacaTradingClient", build_trading_client(creds))
        return cls(data_client, calendar_client)

    @property
    def calendar_client(self) -> orders.AlpacaTradingClient:
        return self._calendar_client

    def _require_data_client(self) -> market_data.AlpacaStockDataClient:
        if self._data_client is None:
            raise BrokerError(
                "no market data client configured; construct the broker with data_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return self._data_client
```

Then MOVE (cut from `broker.py`, paste into the class, otherwise unchanged) `get_last_price`, `get_last_prices`, `get_quote`, `get_bars` and `_sessions_before`, with two edits:
- `get_bars` signature becomes `def get_bars(self, assets: Sequence[Asset], length: int, timestep: str = "day", *, end: datetime, include_after_hours: bool = True) -> dict[Asset, Bars]:` and its body uses the `end` parameter instead of `end = self.clock.now()`.
- `_sessions_before` calls `self._calendar_client.get_calendar(request)` instead of `self._client.get_calendar(request)`.

- [ ] **Step 4: Add `lazy_news_provider` to `brokers/alpaca/news.py`**

Change its imports to `from collections.abc import Callable, Sequence` and `from trading_agent_framework.utils.errors import BrokerError, ConfigurationError`, then append:

```python
def lazy_news_provider(credentials: Callable[[], AlpacaCredentials]) -> Callable[[], AlpacaNewsProvider]:
    """A factory reading the news credentials only when first called, so a strategy that never
    searches news needs no `ALPACA_NEWS_*`. Missing credentials become `BrokerError`, which the
    news tool turns into `{"error": ...}`."""

    def build() -> AlpacaNewsProvider:
        try:
            creds = credentials()
        except ConfigurationError as exc:
            raise BrokerError(f"no news source available: {exc}") from exc
        return AlpacaNewsProvider.from_credentials(creds)

    return build
```

- [ ] **Step 5: Make `AlpacaBroker` delegate**

In `brokers/alpaca/broker.py`:
- Imports: add `from collections.abc import Callable, Sequence`, `from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData`, and `lazy_news_provider` to the `brokers.alpaca.news` import. Remove imports ruff reports unused afterwards (`build_news_client`, `build_stock_data_client`, `market_data`, `MarketSession`, ...; `account` stays for `configure_account`).
- Constructor: add keyword-only `market_data: AlpacaMarketData | None = None` and `news_provider_factory: Callable[[], NewsProvider] | None = None`; replace the `self._data_client = ...` and `self._news_provider = ...` lines with:

```python
        self._market_data = market_data if market_data is not None else AlpacaMarketData(data_client, client)
        self._news_provider: NewsProvider | None = AlpacaNewsProvider(news_client) if news_client is not None else None
        self._news_provider_factory = news_provider_factory
```

- `from_credentials` becomes:

```python
    @classmethod
    def from_credentials(
        cls,
        strategy_name: str,
        *,
        trading: AlpacaCredentials,
        data: AlpacaCredentials,
        news: Callable[[], AlpacaCredentials] | None = None,
        with_stream: bool = True,
    ) -> AlpacaBroker:
        # TradingClient's own signatures declare `T | RawData` (a dict) because
        # the SDK supports a raw_data mode; this project never enables it, so
        # every call actually returns the parsed model. Narrow once, here, at
        # the single point a real client is constructed.
        client = cast("orders.AlpacaTradingClient", build_trading_client(trading))
        stream = build_trading_stream(trading) if with_stream else None
        broker = cls(
            strategy_name,
            client,
            stream=stream,
            is_paper=trading.is_paper,
            market_data=AlpacaMarketData.from_credentials(data),
            news_provider_factory=lazy_news_provider(news) if news is not None else None,
        )
        broker.configure_account()
        return broker
```

- `news_provider` and `get_news`:

```python
    def news_provider(self) -> NewsProvider | None:
        if self._news_provider is None and self._news_provider_factory is not None:
            self._news_provider = self._news_provider_factory()
        return self._news_provider
```

In `get_news`, start with `provider = self.news_provider()`, keep the existing "no news client configured" `BrokerError` when it is `None`, and call `provider.get_news(...)`.

- Replace the removed market-data methods with delegates:

```python
    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self._market_data.get_last_price(asset)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return self._market_data.get_last_prices(assets)

    def get_quote(self, asset: Asset) -> Quote | None:
        return self._market_data.get_quote(asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self._market_data.get_bars(
            assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours
        )
```

- [ ] **Step 6: Update the existing `from_credentials` tests**

In `tests/brokers/alpaca/test_broker_market_data.py::test_from_credentials_wires_the_stock_data_client` and `tests/brokers/alpaca/test_broker_account.py` (`test_from_credentials_records_the_account_kind`, `test_from_credentials_configures_account_restrictions`):
- add `from trading_agent_framework.brokers.alpaca import data as data_module`;
- patch `data_module.build_stock_data_client` (instead of `broker_module.build_stock_data_client`) and also `data_module.build_trading_client` (returning a separate `FakeTradingClient()`), so no real client is built;
- call `AlpacaBroker.from_credentials("momentum", trading=creds, data=creds, with_stream=False)`.

Append to `test_broker_market_data.py`:

```python
def test_from_credentials_builds_the_news_provider_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.alpaca import data as data_module
    from trading_agent_framework.brokers.alpaca import news as news_module

    client = FakeTradingClient()
    client.account_configuration_response = make_alpaca_account_configuration()
    monkeypatch.setattr(broker_module, "build_trading_client", lambda creds: client)
    monkeypatch.setattr(data_module, "build_trading_client", lambda creds: FakeTradingClient())
    monkeypatch.setattr(data_module, "build_stock_data_client", lambda creds: FakeStockHistoricalDataClient())
    monkeypatch.setattr(news_module, "build_news_client", lambda creds: FakeNewsClient())
    reads: list[str] = []

    def news_creds() -> AlpacaCredentials:
        reads.append("news")
        return AlpacaCredentials(api_key="nk", api_secret="ns")

    creds = AlpacaCredentials(api_key="k", api_secret="s")
    broker = AlpacaBroker.from_credentials("momentum", trading=creds, data=creds, news=news_creds, with_stream=False)

    assert reads == []
    assert broker.news_provider() is broker.news_provider()
    assert reads == ["news"]
```

- [ ] **Step 7: Run the broker/backtesting/core tests**

Run: `uv run pytest tests/brokers tests/backtesting tests/core tests/agents && uv run ruff check`
Expected: PASS. If a test still reaches `broker._data_client` or `broker._sessions_before`, point it at `broker._market_data`.

- [ ] **Step 8: Migrate `main.py` and the smoke scripts**

`main.py` (temporary until Task 3/4):

```python
    broker = AlpacaBroker.from_credentials(
        strategy_name,
        trading=AlpacaCredentials.for_trading(),
        data=AlpacaCredentials.for_data(),
        news=AlpacaCredentials.for_news,
    )
```

and drop the `creds.api_key` check above it (`for_trading()` already raises `ConfigurationError`).

In each of `scripts/tests/smoke_alpaca_orders.py`, `smoke_alpaca_data.py`, `smoke_news.py`, `smoke_macro.py`, `smoke_fundamentals.py`, `smoke_strategy_paper.py`, `smoke_strategy_news.py`:
- `AlpacaCredentials.from_env()` -> `AlpacaCredentials.for_trading()`;
- `AlpacaBroker.from_credentials(NAME, creds, with_stream=X)` -> `AlpacaBroker.from_credentials(NAME, trading=creds, data=AlpacaCredentials.for_data(), news=AlpacaCredentials.for_news, with_stream=X)` (keep each script's `with_stream` value; omit it where the script omits it);
- user-facing strings: `ALPACA_IS_PAPER` -> `BROKER_API_IS_PAPER`; "Create it with ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_IS_PAPER=true" -> "Create it with ALPACA_API_KEY / ALPACA_API_SECRET, ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET, ALPACA_NEWS_API_KEY / ALPACA_NEWS_API_SECRET and BROKER_API_IS_PAPER=true".

Verify: `grep -rn "AlpacaCredentials.from_env\|ALPACA_IS_PAPER" scripts src` shows only the rename guard in `config/env.py`. Then run `uv run ruff check scripts` and `uv run python -m py_compile scripts/tests/*.py`.

- [ ] **Step 9: Run the full suite and commit**

Run: `uv run pytest && uv run ruff check`
Expected: PASS.

```bash
git add src/trading_agent_framework/brokers/alpaca src/trading_agent_framework/main.py scripts/tests tests/brokers/alpaca
git commit -m "refactor: extract AlpacaMarketData and split AlpacaBroker trading/data/news credentials (Task 2)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `PlaceholderBroker`; backtests stop building a live broker

**Files:**
- Create: `src/trading_agent_framework/backtesting/placeholder.py`
- Modify: `src/trading_agent_framework/main.py`
- Modify: `scripts/tests/smoke_backtest.py:54-90`
- Test: `tests/backtesting/test_placeholder.py` (create), `tests/test_main.py`

**Interfaces:**
- Produces: `PlaceholderBroker(strategy_name: str)` — a `Broker`, `name = "placeholder"`, `clock` a `BacktestClock`, every operation raising `BrokerError("not available before the backtest starts")`. `main.build_broker(strategy_name) -> Broker` (a module-level name, temporary body here; Task 4 replaces it with the factory import).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/test_placeholder.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.backtesting.placeholder import PlaceholderBroker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError

AAPL = Asset("AAPL")


def _order() -> Order:
    return Order("s", AAPL, OrderSide.BUY, quantity=Decimal(1))


def test_it_carries_the_strategy_name_and_a_simulated_clock() -> None:
    broker = PlaceholderBroker("news_binary")

    assert broker.strategy_name == "news_binary"
    assert isinstance(broker.clock, BacktestClock)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.submit_order(_order()),
        lambda b: b.cancel_order(_order()),
        lambda b: b.pull_order("x"),
        lambda b: b.pull_orders(),
        lambda b: b.pull_positions(),
        lambda b: b.get_account(),
        lambda b: b.modify_order(_order(), limit_price=Decimal(1)),
        lambda b: b.close_position(AAPL),
        lambda b: b.close_all_positions(),
        lambda b: b.sync_open_orders(),
        lambda b: b.get_last_price(AAPL),
        lambda b: b.get_last_prices([AAPL]),
        lambda b: b.get_quote(AAPL),
        lambda b: b.get_bars([AAPL], 5),
    ],
)
def test_every_operation_raises(call) -> None:
    with pytest.raises(BrokerError, match="not available before the backtest starts"):
        call(PlaceholderBroker("s"))
```

Append to `tests/test_main.py`:

```python
def test_backtesting_mode_builds_a_placeholder_and_never_a_live_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from trading_agent_framework.backtesting.placeholder import PlaceholderBroker

    seen: list[object] = []
    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", lambda *a, **k: pytest.fail("a live broker was built for a backtest"))
    monkeypatch.setitem(main_module.AGENT_STRATEGIES, "probe", lambda broker, mode: seen.append(broker))

    main_module._run_strategy(Console(), TradingMode.BACKTESTING, "probe")

    [broker] = seen
    assert isinstance(broker, PlaceholderBroker)
    assert broker.strategy_name == "probe"


def test_a_configuration_error_exits_cleanly_in_paper_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.console import Console

    from trading_agent_framework.utils.errors import ConfigurationError

    def refuse(strategy_name: str):
        raise ConfigurationError("Missing or blank ALPACA_API_KEY / ALPACA_API_SECRET environment variables")

    monkeypatch.setattr(main_module, "find_project_root", lambda: None)
    monkeypatch.setattr(main_module, "load_strategy_env", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "build_broker", refuse)

    with pytest.raises(SystemExit) as excinfo:
        main_module._run_strategy(Console(), TradingMode.PAPER, "news_binary")

    assert excinfo.value.code == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backtesting/test_placeholder.py tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.backtesting.placeholder'`.

- [ ] **Step 3: Implement `backtesting/placeholder.py`**

```python
"""`PlaceholderBroker`: what `main.py` builds a strategy with in backtesting mode, before
`run_backtest` swaps in the real `BacktestBroker`. It only carries the strategy name and a
simulated clock and touches no network -- building a live broker just to start a backtest used
to reconfigure the live Alpaca account, and would need IB Gateway running under `BROKER=ibkr`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar, NoReturn

from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.errors import BrokerError

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)  # never read: run_backtest replaces the clock too


def _unavailable() -> NoReturn:
    raise BrokerError("not available before the backtest starts")


class PlaceholderBroker(Broker):
    name: ClassVar[str] = "placeholder"

    def __init__(self, strategy_name: str) -> None:
        super().__init__(strategy_name, clock=BacktestClock(start=_EPOCH, sessions=[]), is_paper=True)

    def _conform_order(self, order: Order) -> Order:
        _unavailable()

    def _submit_order(self, order: Order) -> Order:
        _unavailable()

    def cancel_order(self, order: Order) -> None:
        _unavailable()

    def pull_order(self, identifier: str) -> Order | None:
        _unavailable()

    def pull_orders(self, limit: int = 100) -> list[Order]:
        _unavailable()

    def pull_positions(self) -> list[Position]:
        _unavailable()

    def get_account(self) -> AccountBalances:
        _unavailable()

    def modify_order(self, order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None) -> Order:
        _unavailable()

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        _unavailable()

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        _unavailable()

    def sync_open_orders(self) -> list[Order]:
        _unavailable()

    def get_last_price(self, asset: Asset) -> Decimal | None:
        _unavailable()

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        _unavailable()

    def get_quote(self, asset: Asset) -> Quote | None:
        _unavailable()

    def get_bars(self, assets: Sequence[Asset], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]:
        _unavailable()
```

- [ ] **Step 4: Rewire `main.py`**

- `StrategyBuilder = Callable[[Broker, TradingMode], Strategy | None]`; retype both builders' `broker` parameter to `Broker` (`from trading_agent_framework.brokers.base import Broker`); drop the two `# ty: ignore[invalid-argument-type]` comments in `tests/test_main.py` if the type checker no longer needs them.
- Add below the imports (temporary; Task 4 replaces it with the factory import):

```python
def build_broker(strategy_name: str) -> Broker:
    return AlpacaBroker.from_credentials(
        strategy_name,
        trading=AlpacaCredentials.for_trading(),
        data=AlpacaCredentials.for_data(),
        news=AlpacaCredentials.for_news,
    )
```

- In `_run_strategy`, replace the credentials/broker lines with:

```python
    if trading_mode is TradingMode.BACKTESTING:
        broker: Broker = PlaceholderBroker(strategy_name)
    else:
        try:
            broker = build_broker(strategy_name)
        except (ConfigurationError, BrokerError) as exc:
            console.print(str(exc), style="bold red")
            raise SystemExit(1) from exc
```

(import `PlaceholderBroker` from `trading_agent_framework.backtesting.placeholder`, `BrokerError`/`ConfigurationError` from `trading_agent_framework.utils.errors`).

- [ ] **Step 5: Update `scripts/tests/smoke_backtest.py`**

Remove `_load_credentials` and the `AlpacaBroker.from_credentials(...)` placeholder; build `placeholder_broker = PlaceholderBroker("smoke_backtest")`. If the script's backtest uses `AlpacaBacktestData`, update its docstring and missing-file message to name `ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET` (and keep loading the env file with `load_dotenv` as it does today). Then `uv run ruff check scripts && uv run python -m py_compile scripts/tests/smoke_backtest.py`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest && uv run ruff check`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/backtesting/placeholder.py src/trading_agent_framework/main.py scripts/tests/smoke_backtest.py tests/backtesting/test_placeholder.py tests/test_main.py
git commit -m "feat: backtests run on an inert PlaceholderBroker instead of a live one (Task 3)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: `build_broker` factory, entry point wiring, env docs

**Files:**
- Create: `src/trading_agent_framework/brokers/factory.py`
- Modify: `src/trading_agent_framework/main.py`
- Modify: `env/.env.example`, `README.md` (env section)
- Test: `tests/brokers/test_factory.py` (create), `tests/brokers/test_lazy_imports.py`

**Interfaces:**
- Consumes: `BrokerSettings`, `BrokerKind`, `AlpacaCredentials` (Task 1); `AlpacaBroker.from_credentials` (Task 2).
- Produces:
  - `BrokerBuilder = Callable[[str, BrokerSettings, Mapping[str, str] | None], Broker]`
  - `BUILDERS: dict[BrokerKind, BrokerBuilder]` (module-level; Task 11 adds IBKR)
  - `build_broker(strategy_name: str, *, env: Mapping[str, str] | None = None) -> Broker`

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/test_factory.py`:

```python
from __future__ import annotations

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.brokers import factory
from trading_agent_framework.config.env import BrokerKind, BrokerSettings
from trading_agent_framework.utils.errors import ConfigurationError


def _fake_broker(strategy_name: str) -> FakeBroker:
    return FakeBroker(FakeClock(et(2026, 9, 22, 10)), strategy_name=strategy_name)


def test_build_broker_dispatches_on_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, BrokerSettings]] = []

    def build(strategy_name, settings, env):
        calls.append((strategy_name, settings))
        return _fake_broker(strategy_name)

    monkeypatch.setitem(factory.BUILDERS, BrokerKind.ALPACA, build)

    broker = factory.build_broker("news_binary", env={"BROKER_API_IS_PAPER": "false"})

    assert broker.strategy_name == "news_binary"
    assert calls == [("news_binary", BrokerSettings(kind=BrokerKind.ALPACA, is_paper=False))]


def test_build_broker_reports_a_broker_without_a_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(factory.BUILDERS, BrokerKind.IBKR, raising=False)

    with pytest.raises(ConfigurationError, match="BROKER=ibkr is not supported"):
        factory.build_broker("s", env={"BROKER": "ibkr"})


def test_the_alpaca_builder_reads_trading_and_data_credentials_but_not_news(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker

    captured: dict[str, object] = {}

    def fake_from_credentials(cls, strategy_name, *, trading, data, news, with_stream=True):
        captured.update(trading=trading, data=data, news=news)
        return _fake_broker(strategy_name)

    monkeypatch.setattr(AlpacaBroker, "from_credentials", classmethod(fake_from_credentials))
    env = {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}

    factory.build_broker("s", env=env)

    assert captured["trading"].api_key == "k"
    assert captured["data"].api_key == "dk"
    with pytest.raises(ConfigurationError, match="ALPACA_NEWS_API_KEY"):
        captured["news"]()  # read lazily: missing news credentials only fail when news is used
```

Append to `tests/brokers/test_lazy_imports.py`:

```python
def test_importing_the_factory_does_not_import_alpaca() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import trading_agent_framework.brokers.factory\nimport sys\nassert 'alpaca' not in sys.modules\nprint('OK')\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers/test_factory.py tests/brokers/test_lazy_imports.py -v`
Expected: FAIL with `ImportError: cannot import name 'factory'`.

- [ ] **Step 3: Implement `brokers/factory.py`**

```python
"""Paper/live broker selection from the strategy env file (`BROKER=alpaca|ibkr`).

Backtests never come here (`main.py` gives them a `PlaceholderBroker`). Broker classes are
imported inside the builders, so importing this module stays as light as `brokers/__init__.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.config.env import AlpacaCredentials, BrokerKind, BrokerSettings
from trading_agent_framework.utils.errors import ConfigurationError

BrokerBuilder = Callable[[str, BrokerSettings, Mapping[str, str] | None], Broker]


def _build_alpaca(strategy_name: str, settings: BrokerSettings, env: Mapping[str, str] | None) -> Broker:
    from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker

    return AlpacaBroker.from_credentials(
        strategy_name,
        trading=AlpacaCredentials.for_trading(env),
        data=AlpacaCredentials.for_data(env),
        news=lambda: AlpacaCredentials.for_news(env),
    )


BUILDERS: dict[BrokerKind, BrokerBuilder] = {
    BrokerKind.ALPACA: _build_alpaca,
}


def build_broker(strategy_name: str, *, env: Mapping[str, str] | None = None) -> Broker:
    """The paper/live broker named by `BROKER` (default `alpaca`), connected and checked."""
    settings = BrokerSettings.from_env(env)
    builder = BUILDERS.get(settings.kind)
    if builder is None:
        raise ConfigurationError(f"BROKER={settings.kind.value} is not supported yet")
    return builder(strategy_name, settings, env)
```

In `main.py`: delete the temporary `build_broker` function (Task 3) and the now-unused `AlpacaBroker`/`AlpacaCredentials` imports; add `from trading_agent_framework.brokers.factory import build_broker`. The `test_main.py` patches of `main_module.build_broker` keep working.

- [ ] **Step 4: Run tests**

Run: `uv run pytest && uv run ruff check`
Expected: PASS.

- [ ] **Step 5: Rewrite the env docs**

Replace everything above `# LLM connection` in `env/.env.example` with:

```dotenv
# --- Broker selection (paper/live only; ignored when backtesting) -------------------
# Which broker trades: alpaca (default) or ibkr.
BROKER=alpaca

# true (default) -> paper trading. false -> LIVE TRADING WITH REAL MONEY.
# With BROKER=ibkr it also picks the default IB Gateway port and is checked against the
# logged-in account (paper accounts start with DU). Replaces ALPACA_IS_PAPER, which is
# now rejected.
BROKER_API_IS_PAPER=true

# --- Alpaca trading (required when BROKER=alpaca) ------------------------------------
# Key pair from https://app.alpaca.markets/ (paper and live dashboards give different keys).
ALPACA_API_KEY=
ALPACA_API_SECRET=

# --- IBKR trading (used when BROKER=ibkr; all optional) ------------------------------
# IB Gateway holds the IBKR login; this app only connects to its API socket.
# IBKR_PORT defaults to 4002 (paper) or 4001 (live), IB Gateway's defaults.
# Give each strategy running at the same time its own IBKR_CLIENT_ID and keep it stable:
# only the client id that placed an order can modify or cancel it.
IBKR_HOST=127.0.0.1
IBKR_PORT=
IBKR_CLIENT_ID=1

# --- Alpaca market data (required for paper/live with either broker, and for
# --- backtests using AlpacaBacktestData; Yahoo backtests need nothing) --------------
# Prices, quotes, bars and the market calendar always come from Alpaca's IEX feed.
# ALPACA_DATA_IS_PAPER says whether this is a paper key pair (used for the calendar call).
ALPACA_DATA_API_KEY=
ALPACA_DATA_API_SECRET=
ALPACA_DATA_IS_PAPER=true

# --- Alpaca news (required when a strategy uses the news tool, in any mode) ----------
ALPACA_NEWS_API_KEY=
ALPACA_NEWS_API_SECRET=
```

In `README.md`'s env-file section: replace every `ALPACA_IS_PAPER` with `BROKER_API_IS_PAPER`, and add this table with the sentence below it:

```markdown
| Variables | Used by | Required when |
|---|---|---|
| `BROKER`, `BROKER_API_IS_PAPER` | broker factory | optional (`alpaca`, `true`) |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Alpaca trading | `BROKER=alpaca`, paper/live |
| `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID` | IBKR trading | `BROKER=ibkr` (all have defaults) |
| `ALPACA_DATA_API_KEY`, `ALPACA_DATA_API_SECRET`, `ALPACA_DATA_IS_PAPER` | market data and calendar | paper/live (both brokers); `AlpacaBacktestData` backtests |
| `ALPACA_NEWS_API_KEY`, `ALPACA_NEWS_API_SECRET` | news tool | a strategy uses the news tool |

Groups never fall back to each other; with one Alpaca key pair, repeat it in each group you need.
```

Run `grep -n "ALPACA_" README.md` and fix every stale reference.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/brokers/factory.py src/trading_agent_framework/main.py tests/brokers env/.env.example README.md
git commit -m "feat: build_broker factory selects the paper/live broker from BROKER (Task 4)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `ib_async` dependency and IBKR order building (`brokers/ibkr/orders.py`, build side)

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `src/trading_agent_framework/brokers/ibkr/__init__.py`, `src/trading_agent_framework/brokers/ibkr/orders.py`
- Test: `tests/brokers/ibkr/test_orders_build.py` (plus `tests/brokers/ibkr/__init__.py` only if `tests/brokers/alpaca/__init__.py` exists — mirror it)

**Interfaces:**
- Produces (in `brokers/ibkr/orders.py`):
  - `EXCHANGE = "SMART"`, `CURRENCY = "USD"`
  - `build_contract(asset: Asset) -> ib_async.Stock`
  - `client_order_id_for(strategy_name: str, identifier: str) -> str` -> `"{strategy_name}:{identifier}"`
  - `conform_order(order: Order) -> Order` (rejects notional, floors fractional quantity)
  - `build_order(order: Order) -> ib_async.Order`

- [ ] **Step 1: Add the dependency**

Run: `uv add "ib_async>=2.1,<3"`
Then: `uv run python -c "import ib_async; print(ib_async.__version__)"`
Expected: prints `2.1.x`; `pyproject.toml` and `uv.lock` changed.

- [ ] **Step 2: Write the failing tests**

Create `tests/brokers/ibkr/test_orders_build.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

AAPL = Asset("AAPL")


def _order(**overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": Decimal(10)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


def test_contract_is_a_smart_routed_usd_stock() -> None:
    contract = orders.build_contract(AAPL)

    assert (contract.secType, contract.symbol, contract.exchange, contract.currency) == ("STK", "AAPL", "SMART", "USD")


def test_client_order_id_prefixes_the_strategy() -> None:
    assert orders.client_order_id_for("news_binary", "abc") == "news_binary:abc"


def test_conform_rejects_notional_orders() -> None:
    with pytest.raises(OrderValidationError, match="notional"):
        orders.conform_order(_order(quantity=None, notional=Decimal(100)))


def test_conform_floors_fractional_quantities(caplog: pytest.LogCaptureFixture) -> None:
    order = orders.conform_order(_order(quantity=Decimal("3.7")))

    assert order.quantity == Decimal(3)
    assert "whole shares" in caplog.text


@pytest.mark.parametrize("quantity", [Decimal("0.4"), Decimal(0), Decimal(-1), None])
def test_conform_rejects_quantities_below_one_share(quantity: Decimal | None) -> None:
    with pytest.raises(OrderValidationError):
        orders.conform_order(_order(quantity=quantity))


def test_market_day_order() -> None:
    ib_order = orders.build_order(_order(client_order_id="s:1"))

    assert (ib_order.action, ib_order.totalQuantity, ib_order.orderType, ib_order.tif) == ("BUY", 10.0, "MKT", "DAY")
    assert ib_order.orderRef == "s:1"
    assert ib_order.outsideRth is False
    assert ib_order.transmit is True


def test_limit_sell_gtc_outside_regular_hours() -> None:
    ib_order = orders.build_order(
        _order(side=OrderSide.SELL, order_type=OrderType.LIMIT, limit_price=Decimal("101.25"), time_in_force=TimeInForce.GTC, extended_hours=True)
    )

    assert (ib_order.action, ib_order.orderType, ib_order.lmtPrice, ib_order.tif, ib_order.outsideRth) == ("SELL", "LMT", 101.25, "GTC", True)


def test_stop_and_stop_limit() -> None:
    stop = orders.build_order(_order(order_type=OrderType.STOP, stop_price=Decimal(95)))
    stop_limit = orders.build_order(_order(order_type=OrderType.STOP_LIMIT, stop_price=Decimal(95), stop_limit_price=Decimal("94.5")))

    assert (stop.orderType, stop.auxPrice) == ("STP", 95.0)
    assert (stop_limit.orderType, stop_limit.auxPrice, stop_limit.lmtPrice) == ("STP LMT", 95.0, 94.5)


def test_trailing_stop_by_amount_or_percent() -> None:
    by_amount = orders.build_order(_order(order_type=OrderType.TRAIL, trail_price=Decimal(2)))
    by_percent = orders.build_order(_order(order_type=OrderType.TRAIL, trail_percent=Decimal("1.5")))

    assert (by_amount.orderType, by_amount.auxPrice) == ("TRAIL", 2.0)
    assert (by_percent.orderType, by_percent.trailingPercent) == ("TRAIL", 1.5)


@pytest.mark.parametrize(("order_type", "expected"), [(OrderType.MARKET, "MOC"), (OrderType.LIMIT, "LOC")])
def test_at_the_close_orders(order_type: OrderType, expected: str) -> None:
    ib_order = orders.build_order(_order(order_type=order_type, limit_price=Decimal(100), time_in_force=TimeInForce.CLS))

    assert (ib_order.orderType, ib_order.tif) == (expected, "DAY")


@pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.FOK, TimeInForce.OPG])
def test_other_time_in_force_values_map_directly(tif: TimeInForce) -> None:
    assert orders.build_order(_order(time_in_force=tif)).tif == tif.value.upper()


@pytest.mark.parametrize(
    "overrides",
    [
        {"order_type": OrderType.LIMIT},
        {"order_type": OrderType.STOP},
        {"order_type": OrderType.STOP_LIMIT, "stop_price": Decimal(1)},
        {"order_type": OrderType.TRAIL},
        {"order_type": OrderType.STOP, "stop_price": Decimal(1), "time_in_force": TimeInForce.CLS},
    ],
)
def test_missing_prices_and_unsupported_combinations_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(OrderValidationError):
        orders.build_order(_order(**overrides))
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_orders_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.ibkr'`.

- [ ] **Step 4: Implement**

`src/trading_agent_framework/brokers/ibkr/__init__.py`:

```python
"""Interactive Brokers (TWS API via IB Gateway, `ib_async`): trading, account and positions.
Market data, the calendar and news come from Alpaca (see `IbkrBroker`)."""
```

`src/trading_agent_framework/brokers/ibkr/orders.py`:

```python
"""Pure IBKR order translation: our `Order` <-> `ib_async` `Contract`/`Order`/`Trade`.

No I/O, no state, no client instances (same rules as `brokers/alpaca/orders.py`). This is the
IBKR counterpart of Alpaca's float boundary: `ib_async` order fields are floats; everything
handed back to the framework is `Decimal`.
"""

from __future__ import annotations

import logging
from decimal import ROUND_FLOOR, Decimal

from ib_async import Order as IbOrder
from ib_async import Stock

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

logger = logging.getLogger(__name__)

EXCHANGE = "SMART"
CURRENCY = "USD"

_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
    OrderType.TRAIL: "TRAIL",
}
_AT_THE_CLOSE: dict[OrderType, str] = {OrderType.MARKET: "MOC", OrderType.LIMIT: "LOC"}
_TIME_IN_FORCE: dict[TimeInForce, str] = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.OPG: "OPG",
}


def build_contract(asset: Asset) -> Stock:
    return Stock(asset.symbol, EXCHANGE, CURRENCY)


def client_order_id_for(strategy_name: str, identifier: str) -> str:
    return f"{strategy_name}:{identifier}"


def conform_order(order: Order) -> Order:
    """IBKR's API trades whole shares: reject dollar amounts, floor fractional share counts."""
    if order.notional is not None:
        raise OrderValidationError("IBKR does not accept notional (dollar-amount) orders; pass a share quantity")
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    whole = order.quantity.to_integral_value(rounding=ROUND_FLOOR)
    if whole < 1:
        raise OrderValidationError(
            f"order {order.identifier} for {order.quantity} {order.asset.symbol} is below one whole share (IBKR trades whole shares only)"
        )
    if whole != order.quantity:
        logger.warning("IBKR trades whole shares only: %s quantity %s floored to %s", order.asset.symbol, order.quantity, whole)
        order.quantity = whole
    return order


def _require(value: Decimal | None, name: str, order: Order) -> float:
    if value is None:
        raise OrderValidationError(f"{order.order_type.value} order {order.identifier} needs {name}")
    return float(value)


def build_order(order: Order) -> IbOrder:
    """The `ib_async` order for an already-conformed `order` (quantity in whole shares)."""
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    ib_order = IbOrder()
    ib_order.action = "BUY" if order.side is OrderSide.BUY else "SELL"
    ib_order.totalQuantity = float(order.quantity)
    if order.time_in_force is TimeInForce.CLS:
        at_the_close = _AT_THE_CLOSE.get(order.order_type)
        if at_the_close is None:
            raise OrderValidationError("an at-the-close (CLS) order must be a market or limit order")
        ib_order.orderType = at_the_close
        ib_order.tif = "DAY"
    else:
        ib_order.orderType = _ORDER_TYPES[order.order_type]
        ib_order.tif = _TIME_IN_FORCE[order.time_in_force]
    if order.order_type is OrderType.LIMIT:
        ib_order.lmtPrice = _require(order.limit_price, "limit_price", order)
    elif order.order_type is OrderType.STOP:
        ib_order.auxPrice = _require(order.stop_price, "stop_price", order)
    elif order.order_type is OrderType.STOP_LIMIT:
        ib_order.auxPrice = _require(order.stop_price, "stop_price", order)
        ib_order.lmtPrice = _require(order.stop_limit_price, "stop_limit_price", order)
    elif order.order_type is OrderType.TRAIL:
        if order.trail_price is not None:
            ib_order.auxPrice = float(order.trail_price)
        elif order.trail_percent is not None:
            ib_order.trailingPercent = float(order.trail_percent)
        else:
            raise OrderValidationError(f"trailing stop order {order.identifier} needs trail_price or trail_percent")
    ib_order.outsideRth = order.extended_hours
    ib_order.orderRef = order.client_order_id or ""
    ib_order.transmit = True
    return ib_order
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/trading_agent_framework/brokers/ibkr tests/brokers/ibkr
git commit -m "feat: ib_async dependency and pure IBKR order building (Task 5)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: IBKR response parsing (`brokers/ibkr/orders.py`, parse side) and `ib_async` test builders

**Files:**
- Modify: `src/trading_agent_framework/brokers/ibkr/orders.py`
- Modify: `tests/fakes.py`
- Test: `tests/brokers/ibkr/test_orders_parse.py`

**Interfaces:**
- Consumes: Task 5's `orders` module.
- Produces (in `brokers/ibkr/orders.py`):
  - `PENDING_STATUSES: frozenset[str]` = `{"", "PendingSubmit", "ApiPending"}`
  - `map_status(status: str) -> OrderStatus`
  - `map_status_event(status: str) -> OrderEvent | None`
  - `identifier_from_order_ref(order_ref: str, strategy_name: str) -> str | None`
  - `to_decimal(value: float | None) -> Decimal | None` (None for `None`, NaN, `ib_async.util.UNSET_DOUBLE`)
  - `parse_trade(trade: Trade, strategy_name: str) -> Order`
  - `parse_portfolio_item(item: PortfolioItem, strategy_name: str) -> Position | None`
  - `rejection_message(trade: Trade) -> str | None`
- Produces (in `tests/fakes.py`): `make_ib_trade(...) -> Trade`, `make_ib_fill(...) -> Fill`, `make_ib_portfolio_item(...) -> PortfolioItem`, `make_ib_summary(...) -> list[AccountValue]` with the keyword arguments shown below.

- [ ] **Step 1: Add the `ib_async` builders to `tests/fakes.py`**

Add next to the other third-party imports:

```python
import ib_async
from ib_async import AccountValue, Execution, Fill, PortfolioItem, Stock, Trade, TradeLogEntry
from ib_async import CommissionReport as IbCommissionReport
from ib_async import Order as IbOrder
from ib_async import OrderStatus as IbOrderStatus
```

Append:

```python
# --- IBKR (ib_async) objects ------------------------------------------------------

_IB_TIME = datetime(2026, 9, 22, 14, 0, tzinfo=UTC)


def make_ib_trade(
    *,
    symbol: str = "AAPL",
    action: str = "BUY",
    quantity: float = 10.0,
    order_type: str = "MKT",
    tif: str = "DAY",
    lmt_price: float = ib_async.util.UNSET_DOUBLE,
    aux_price: float = ib_async.util.UNSET_DOUBLE,
    order_ref: str = "s:abc",
    status: str = "Submitted",
    filled: float = 0.0,
    avg_fill_price: float = 0.0,
    order_id: int = 1,
    perm_id: int = 1001,
    client_id: int = 1,
    log_message: str = "",
    log_error_code: int = 0,
) -> Trade:
    order = IbOrder(
        orderId=order_id, clientId=client_id, permId=perm_id, action=action, totalQuantity=quantity,
        orderType=order_type, lmtPrice=lmt_price, auxPrice=aux_price, tif=tif, orderRef=order_ref,
    )
    order_status = IbOrderStatus(
        orderId=order_id, status=status, filled=filled, remaining=quantity - filled,
        avgFillPrice=avg_fill_price, permId=perm_id, clientId=client_id,
    )
    log = [TradeLogEntry(time=_IB_TIME, status=status, message=log_message, errorCode=log_error_code)]
    return Trade(contract=Stock(symbol, "SMART", "USD"), order=order, orderStatus=order_status, fills=[], log=log)


def make_ib_fill(
    *,
    order_ref: str = "s:abc",
    exec_id: str = "e1",
    shares: float = 10.0,
    price: float = 100.0,
    cum_qty: float = 10.0,
    avg_price: float = 100.0,
    symbol: str = "AAPL",
) -> Fill:
    execution = Execution(
        execId=exec_id, time=_IB_TIME, shares=shares, price=price, cumQty=cum_qty,
        avgPrice=avg_price, orderRef=order_ref, side="BOT",
    )
    return Fill(contract=Stock(symbol, "SMART", "USD"), execution=execution, commissionReport=IbCommissionReport(), time=_IB_TIME)


def make_ib_portfolio_item(
    *,
    symbol: str = "AAPL",
    position: float = 10.0,
    market_price: float = 101.0,
    market_value: float = 1010.0,
    average_cost: float = 100.0,
    unrealized_pnl: float = 10.0,
    account: str = "DU123",
) -> PortfolioItem:
    return PortfolioItem(
        contract=Stock(symbol, "SMART", "USD"), position=position, marketPrice=market_price, marketValue=market_value,
        averageCost=average_cost, unrealizedPNL=unrealized_pnl, realizedPNL=0.0, account=account,
    )


def make_ib_summary(
    *,
    account: str = "DU123",
    cash: str = "10000",
    net_liquidation: str = "25000",
    buying_power: str = "10000",
    currency: str = "USD",
) -> list[AccountValue]:
    def value(tag: str, amount: str) -> AccountValue:
        return AccountValue(account=account, tag=tag, value=amount, currency=currency, modelCode="")

    return [
        value("TotalCashValue", cash),
        value("NetLiquidation", net_liquidation),
        value("BuyingPower", buying_power),
        AccountValue(account=account, tag="AccountType", value="INDIVIDUAL", currency="", modelCode=""),
    ]
```

(`datetime` and `UTC` are already imported in `tests/fakes.py`.)

- [ ] **Step 2: Write the failing tests**

Create `tests/brokers/ibkr/test_orders_parse.py`:

```python
from __future__ import annotations

import math
from decimal import Decimal

import ib_async
import pytest
from tests.fakes import make_ib_portfolio_item, make_ib_trade

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PendingSubmit", OrderStatus.NEW),
        ("PreSubmitted", OrderStatus.NEW),
        ("Submitted", OrderStatus.NEW),
        ("Filled", OrderStatus.FILL),
        ("Cancelled", OrderStatus.CANCELED),
        ("ApiCancelled", OrderStatus.CANCELED),
        ("Inactive", OrderStatus.ERROR),
        ("PendingCancel", OrderStatus.CANCELLING),
        ("ApiPending", OrderStatus.UNKNOWN),
    ],
)
def test_map_status(status: str, expected: OrderStatus) -> None:
    assert orders.map_status(status) is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PreSubmitted", OrderEvent.NEW),
        ("Submitted", OrderEvent.NEW),
        ("Cancelled", OrderEvent.CANCELED),
        ("ApiCancelled", OrderEvent.CANCELED),
        ("Inactive", OrderEvent.ERROR),
        ("Filled", None),  # fills come from executions, with price and quantity
        ("PendingSubmit", None),
        ("PendingCancel", None),
    ],
)
def test_map_status_event(status: str, expected: OrderEvent | None) -> None:
    assert orders.map_status_event(status) is expected


def test_identifier_from_order_ref() -> None:
    assert orders.identifier_from_order_ref("news_binary:abc", "news_binary") == "abc"
    assert orders.identifier_from_order_ref("other:abc", "news_binary") is None
    assert orders.identifier_from_order_ref("news_binary:", "news_binary") is None
    assert orders.identifier_from_order_ref("", "news_binary") is None


@pytest.mark.parametrize("value", [None, math.nan, ib_async.util.UNSET_DOUBLE])
def test_to_decimal_treats_unset_values_as_none(value: float | None) -> None:
    assert orders.to_decimal(value) is None


def test_to_decimal_is_exact_for_the_printed_float() -> None:
    assert orders.to_decimal(101.25) == Decimal("101.25")


def test_parse_trade_for_this_strategy() -> None:
    trade = make_ib_trade(order_ref="s:abc", order_type="LMT", lmt_price=99.5, tif="GTC", status="Submitted", filled=4.0, avg_fill_price=99.4)

    order = orders.parse_trade(trade, "s")

    assert order.identifier == "abc"
    assert order.client_order_id == "s:abc"
    assert (order.asset, order.side, order.order_type, order.time_in_force) == (Asset("AAPL"), OrderSide.BUY, OrderType.LIMIT, TimeInForce.GTC)
    assert (order.quantity, order.limit_price, order.filled_quantity, order.avg_fill_price) == (Decimal(10), Decimal("99.5"), Decimal(4), Decimal("99.4"))
    assert order.status is OrderStatus.NEW
    assert order.raw is trade


def test_parse_trade_of_another_client_uses_the_perm_id() -> None:
    order = orders.parse_trade(make_ib_trade(order_ref="", perm_id=555), "s")

    assert order.identifier == "ibkr:555"
    assert order.client_order_id is None


def test_parse_trade_at_the_close_and_stop_limit() -> None:
    moc = orders.parse_trade(make_ib_trade(order_type="MOC"), "s")
    stop_limit = orders.parse_trade(make_ib_trade(order_type="STP LMT", aux_price=95.0, lmt_price=94.5), "s")

    assert (moc.order_type, moc.time_in_force) == (OrderType.MARKET, TimeInForce.CLS)
    assert (stop_limit.order_type, stop_limit.stop_price, stop_limit.stop_limit_price) == (OrderType.STOP_LIMIT, Decimal(95), Decimal("94.5"))


def test_parse_portfolio_item() -> None:
    position = orders.parse_portfolio_item(make_ib_portfolio_item(), "s")

    assert position is not None
    assert (position.asset, position.quantity, position.side) == (Asset("AAPL"), Decimal(10), PositionSide.LONG)
    assert (position.avg_fill_price, position.current_price, position.market_value, position.unrealized_pnl) == (
        Decimal(100), Decimal(101), Decimal(1010), Decimal(10)
    )


def test_a_flat_portfolio_item_is_no_position() -> None:
    assert orders.parse_portfolio_item(make_ib_portfolio_item(position=0.0), "s") is None


def test_rejection_message() -> None:
    rejected = make_ib_trade(status="Inactive", log_message="Order rejected - reason: no trading permissions", log_error_code=201)

    assert orders.rejection_message(rejected) == "Order rejected - reason: no trading permissions (IBKR error 201)"
    assert orders.rejection_message(make_ib_trade(status="Submitted")) is None
    assert orders.rejection_message(make_ib_trade(status="Cancelled")) == "order Cancelled by IBKR"
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_orders_parse.py -v`
Expected: FAIL with `AttributeError: module 'trading_agent_framework.brokers.ibkr.orders' has no attribute 'map_status'`.

- [ ] **Step 4: Implement (extend `brokers/ibkr/orders.py`)**

Extend the imports:

```python
import math

from ib_async import PortfolioItem, Trade
from ib_async.util import UNSET_DOUBLE

from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.entities.position import Position
```

Append:

```python
PENDING_STATUSES: frozenset[str] = frozenset({"", "PendingSubmit", "ApiPending"})
_REJECTED_STATUSES = frozenset({"Inactive", "Cancelled", "ApiCancelled"})

_STATUS: dict[str, OrderStatus] = {
    "PendingSubmit": OrderStatus.NEW,
    "PreSubmitted": OrderStatus.NEW,
    "Submitted": OrderStatus.NEW,
    "Filled": OrderStatus.FILL,
    "Cancelled": OrderStatus.CANCELED,
    "ApiCancelled": OrderStatus.CANCELED,
    "Inactive": OrderStatus.ERROR,
    "PendingCancel": OrderStatus.CANCELLING,
}
_STATUS_EVENT: dict[str, OrderEvent] = {
    "PreSubmitted": OrderEvent.NEW,
    "Submitted": OrderEvent.NEW,
    "Cancelled": OrderEvent.CANCELED,
    "ApiCancelled": OrderEvent.CANCELED,
    "Inactive": OrderEvent.ERROR,
}
_PARSED_ORDER_TYPES: dict[str, tuple[OrderType, TimeInForce | None]] = {
    **{ib: (ours, None) for ours, ib in _ORDER_TYPES.items()},
    "MOC": (OrderType.MARKET, TimeInForce.CLS),
    "LOC": (OrderType.LIMIT, TimeInForce.CLS),
}
_PARSED_TIME_IN_FORCE: dict[str, TimeInForce] = {ib: ours for ours, ib in _TIME_IN_FORCE.items()}


def map_status(status: str) -> OrderStatus:
    return _STATUS.get(status, OrderStatus.UNKNOWN)


def map_status_event(status: str) -> OrderEvent | None:
    """The tracker event an order-status update means; fills come from executions instead."""
    return _STATUS_EVENT.get(status)


def identifier_from_order_ref(order_ref: str, strategy_name: str) -> str | None:
    prefix = f"{strategy_name}:"
    if order_ref.startswith(prefix) and len(order_ref) > len(prefix):
        return order_ref[len(prefix):]
    return None


def to_decimal(value: float | None) -> Decimal | None:
    if value is None or math.isnan(value) or value == UNSET_DOUBLE:
        return None
    return Decimal(str(value))


def parse_trade(trade: Trade, strategy_name: str) -> Order:
    ib_order = trade.order
    order_type, forced_tif = _PARSED_ORDER_TYPES.get(ib_order.orderType, (OrderType.MARKET, None))
    identifier = identifier_from_order_ref(ib_order.orderRef, strategy_name)
    limit = to_decimal(ib_order.lmtPrice)
    aux = to_decimal(ib_order.auxPrice)
    return Order(
        strategy_name=strategy_name,
        asset=Asset(trade.contract.symbol),
        side=OrderSide.BUY if ib_order.action == "BUY" else OrderSide.SELL,
        order_type=order_type,
        quantity=to_decimal(ib_order.totalQuantity),
        time_in_force=forced_tif or _PARSED_TIME_IN_FORCE.get(ib_order.tif, TimeInForce.DAY),
        limit_price=limit if order_type is OrderType.LIMIT else None,
        stop_price=aux if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) else None,
        stop_limit_price=limit if order_type is OrderType.STOP_LIMIT else None,
        trail_price=aux if order_type is OrderType.TRAIL else None,
        trail_percent=to_decimal(ib_order.trailingPercent) if order_type is OrderType.TRAIL else None,
        extended_hours=bool(ib_order.outsideRth),
        status=map_status(trade.orderStatus.status),
        identifier=identifier if identifier is not None else f"ibkr:{ib_order.permId}",
        client_order_id=ib_order.orderRef if identifier is not None else None,
        filled_quantity=to_decimal(trade.orderStatus.filled) or Decimal(0),
        avg_fill_price=to_decimal(trade.orderStatus.avgFillPrice) or None,  # IBKR reports 0.0 before any fill
        raw=trade,
    )


def parse_portfolio_item(item: PortfolioItem, strategy_name: str) -> Position | None:
    quantity = to_decimal(item.position) or Decimal(0)
    if quantity == 0:
        return None
    return Position(
        strategy_name=strategy_name,
        asset=Asset(item.contract.symbol),
        quantity=abs(quantity),
        side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
        avg_fill_price=to_decimal(item.averageCost),
        current_price=to_decimal(item.marketPrice),
        market_value=to_decimal(item.marketValue),
        unrealized_pnl=to_decimal(item.unrealizedPNL),
        raw=item,
    )


def rejection_message(trade: Trade) -> str | None:
    """Why IBKR refused `trade`, or None while it is pending or working."""
    status = trade.orderStatus.status
    if status not in _REJECTED_STATUSES:
        return None
    for entry in reversed(trade.log):
        if entry.message:
            suffix = f" (IBKR error {entry.errorCode})" if entry.errorCode else ""
            return f"{entry.message}{suffix}"
    return f"order {status} by IBKR"
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/orders.py tests/fakes.py tests/brokers/ibkr/test_orders_parse.py
git commit -m "feat: pure IBKR status, trade and position parsing (Task 6)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: IBKR account translation and checks (`brokers/ibkr/account.py`)

**Files:**
- Create: `src/trading_agent_framework/brokers/ibkr/account.py`
- Test: `tests/brokers/ibkr/test_account.py`

**Interfaces:**
- Consumes: `make_ib_summary` (Task 6).
- Produces:
  - `PAPER_ACCOUNT_PREFIX = "DU"`, `BASE_CURRENCY = "USD"`
  - `is_paper_account(account_id: str) -> bool`
  - `parse_account(values: Iterable[AccountValue], account_id: str) -> AccountBalances` (`BrokerError` when a tag is missing or not a number)
  - `check_account(values: Iterable[AccountValue], account_id: str, *, is_paper: bool) -> list[str]` — raises `ConfigurationError` for fatal problems, returns warnings to log.

Cash-vs-margin detection (deviation from the spec's "account type", recorded at the end of this plan): IBKR's `AccountType` summary tag is `INDIVIDUAL`/`IRA`/..., not cash/margin. An account is treated as **margin** when `BuyingPower > TotalCashValue * 1.05 + 1`: a cash account's buying power is its settled cash, a margin account's is 2-4x its equity. An empty new account reads as cash.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/ibkr/test_account.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import make_ib_summary

from trading_agent_framework.brokers.ibkr import account
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError


def test_parse_account() -> None:
    balances = account.parse_account(make_ib_summary(cash="10000", net_liquidation="25000", buying_power="10000"), "DU123")

    assert balances == AccountBalances(cash=Decimal(10000), portfolio_value=Decimal(25000), buying_power=Decimal(10000))


def test_parse_account_ignores_other_accounts_values() -> None:
    values = make_ib_summary(account="U999", cash="1") + make_ib_summary(account="DU123", cash="10000")

    assert account.parse_account(values, "DU123").cash == Decimal(10000)


def test_parse_account_reports_a_missing_tag() -> None:
    values = [v for v in make_ib_summary() if v.tag != "BuyingPower"]

    with pytest.raises(BrokerError, match="BuyingPower"):
        account.parse_account(values, "DU123")


def test_is_paper_account() -> None:
    assert account.is_paper_account("DU1234567")
    assert not account.is_paper_account("U1234567")


def test_a_paper_cash_account_passes_quietly() -> None:
    assert account.check_account(make_ib_summary(), "DU123", is_paper=True) == []


@pytest.mark.parametrize(("account_id", "is_paper"), [("DU123", False), ("U123", True)])
def test_the_paper_flag_must_match_the_account(account_id: str, is_paper: bool) -> None:
    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        account.check_account(make_ib_summary(account=account_id), account_id, is_paper=is_paper)


def test_a_margin_account_is_refused_live() -> None:
    values = make_ib_summary(account="U123", cash="10000", buying_power="40000")

    with pytest.raises(ConfigurationError, match="margin"):
        account.check_account(values, "U123", is_paper=False)


def test_a_margin_account_only_warns_on_paper() -> None:
    [warning] = account.check_account(make_ib_summary(cash="10000", buying_power="40000"), "DU123", is_paper=True)

    assert "margin" in warning


def test_a_non_usd_base_currency_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="USD"):
        account.check_account(make_ib_summary(currency="EUR"), "DU123", is_paper=True)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_account.py -v`
Expected: FAIL with `ImportError: cannot import name 'account'`.

- [ ] **Step 3: Implement**

```python
"""Pure IBKR account translation and start-up checks (same rules as `brokers/alpaca/account.py`).

IBKR cannot switch margin or shorting off through the API, so `IbkrBroker.configure_account`
only CHECKS the account with `check_account` and refuses to trade live on a margin account.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

from ib_async import AccountValue

from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

PAPER_ACCOUNT_PREFIX = "DU"
BASE_CURRENCY = "USD"
_MARGIN_TOLERANCE = Decimal("1.05")  # a cash account's buying power is its settled cash


def is_paper_account(account_id: str) -> bool:
    return account_id.startswith(PAPER_ACCOUNT_PREFIX)


def _tags(values: Iterable[AccountValue], account_id: str) -> dict[str, AccountValue]:
    return {v.tag: v for v in values if v.account == account_id}


def _amount(tags: dict[str, AccountValue], tag: str) -> Decimal:
    value = tags.get(tag)
    if value is None:
        raise BrokerError(f"IBKR account summary has no {tag}")
    try:
        return Decimal(value.value)
    except InvalidOperation:
        raise BrokerError(f"IBKR account summary {tag} is not a number: {value.value!r}") from None


def parse_account(values: Iterable[AccountValue], account_id: str) -> AccountBalances:
    tags = _tags(values, account_id)
    return AccountBalances(
        cash=_amount(tags, "TotalCashValue"),
        portfolio_value=_amount(tags, "NetLiquidation"),
        buying_power=_amount(tags, "BuyingPower"),
    )


def check_account(values: Iterable[AccountValue], account_id: str, *, is_paper: bool) -> list[str]:
    """Raise `ConfigurationError` when the account must not trade; return warnings to log."""
    if is_paper_account(account_id) != is_paper:
        kind = "a paper" if is_paper_account(account_id) else "a live"
        raise ConfigurationError(
            f"IB Gateway is logged into {kind} account ({account_id}) but BROKER_API_IS_PAPER={str(is_paper).lower()}"
        )
    tags = _tags(values, account_id)
    currency = tags["NetLiquidation"].currency if "NetLiquidation" in tags else ""
    if currency != BASE_CURRENCY:
        raise ConfigurationError(f"IBKR account {account_id} has base currency {currency!r}; only USD is supported")
    cash = _amount(tags, "TotalCashValue")
    buying_power = _amount(tags, "BuyingPower")
    if buying_power > cash * _MARGIN_TOLERANCE + 1:
        message = (
            f"IBKR account {account_id} looks like a margin account (buying power {buying_power} > cash {cash}); "
            "this framework expects a cash account (no margin, no shorting)"
        )
        if not is_paper:
            raise ConfigurationError(message)
        return [message]
    return []
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/account.py tests/brokers/ibkr/test_account.py
git commit -m "feat: pure IBKR account parsing and start-up checks (Task 7)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `FakeIB` and `IbkrConnection` (event-loop thread, call timeout, reconnect)

**Files:**
- Modify: `tests/fakes.py` (add `FakeIB`)
- Create: `src/trading_agent_framework/brokers/ibkr/client.py`
- Test: `tests/brokers/ibkr/test_client.py`

**Interfaces:**
- Consumes: `IbkrSettings` (Task 1); `make_ib_summary`, `make_ib_trade` (Task 6).
- Produces:
  - `IbkrConnection(settings: IbkrSettings, *, ib_factory: Callable[[], Any] | None = None, call_timeout: float = 30.0, connect_timeout: float = 15.0, reconnect_attempts: int = 3, sleep: Callable[[float], None] = time.sleep)`
  - `.start() -> None`, `.connect() -> None`, `.call(fn: Callable[[Any], T | Awaitable[T]], *, timeout: float | None = None) -> T`, `.disconnect() -> None`
  - `.on_reconnect: Callable[[], None] | None` (set by the broker)
  - `FakeIB(*, accounts=("DU123",))` in `tests/fakes.py`, with test-facing attributes: `connected`, `connect_calls`, `connect_errors`, `accounts`, `summary`, `portfolio_items`, `all_trades`, `foreign_open_trades`, `executions`, `placed`, `canceled`, `global_cancels`, `unknown_symbols`, `place_status`, `place_log_message`, `place_log_error_code`; events `orderStatusEvent`, `execDetailsEvent`, `errorEvent`, `disconnectedEvent`.

- [ ] **Step 1: Add `FakeIB` to `tests/fakes.py`**

Add `from eventkit import Event as IbEvent` to the imports and append:

```python
class FakeIB:
    """Stand-in for `ib_async.IB`: the methods `IbkrConnection`/`IbkrBroker` call, real `eventkit`
    events, and real `ib_async` objects in and out. Tests configure and drive it directly."""

    def __init__(self, *, accounts: Sequence[str] = ("DU123",)) -> None:
        self.orderStatusEvent = IbEvent("orderStatusEvent")
        self.execDetailsEvent = IbEvent("execDetailsEvent")
        self.errorEvent = IbEvent("errorEvent")
        self.disconnectedEvent = IbEvent("disconnectedEvent")
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.connect_errors: list[Exception] = []  # one popped per connect attempt
        self.client_id = 0
        self.accounts = list(accounts)
        self.summary: list[AccountValue] = make_ib_summary(account=self.accounts[0] if self.accounts else "DU123")
        self.portfolio_items: list[PortfolioItem] = []
        self.all_trades: list[Trade] = []
        self.foreign_open_trades: list[Trade] = []
        self.executions: list[Fill] = []
        self.placed: list[tuple[Stock, IbOrder]] = []
        self.canceled: list[IbOrder] = []
        self.global_cancels = 0
        self.unknown_symbols: set[str] = set()
        self.place_status = "Submitted"
        self.place_log_message = ""
        self.place_log_error_code = 0
        self._next_order_id = 1

    async def connectAsync(self, host: str, port: int, clientId: int, timeout: float | None = None, **_: object) -> None:
        self.connect_calls.append((host, port, clientId))
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        self.connected = True
        self.client_id = clientId

    def isConnected(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.connected = False
        self.disconnectedEvent.emit()

    def managedAccounts(self) -> list[str]:
        return list(self.accounts)

    async def accountSummaryAsync(self, account: str = "") -> list[AccountValue]:
        return list(self.summary)

    async def qualifyContractsAsync(self, *contracts: Stock) -> list[Stock | None]:
        return [
            None if c.symbol in self.unknown_symbols else Stock(c.symbol, c.exchange, c.currency, conId=100 + len(c.symbol))
            for c in contracts
        ]

    def placeOrder(self, contract: Stock, order: IbOrder) -> Trade:
        self.placed.append((contract, order))
        existing = next((t for t in self.all_trades if order.orderId and t.order.orderId == order.orderId), None)
        if existing is not None:  # a modification: IBKR edits the order in place
            existing.order = order
            return existing
        order.orderId = self._next_order_id
        order.permId = 1000 + self._next_order_id
        order.clientId = self.client_id
        self._next_order_id += 1
        trade = make_ib_trade(
            symbol=contract.symbol, action=order.action, quantity=order.totalQuantity, order_type=order.orderType,
            order_ref=order.orderRef, status=self.place_status, order_id=order.orderId, perm_id=order.permId,
            client_id=self.client_id, log_message=self.place_log_message, log_error_code=self.place_log_error_code,
        )
        trade.order = order
        self.all_trades.append(trade)
        return trade

    def cancelOrder(self, order: IbOrder) -> None:
        self.canceled.append(order)

    def reqGlobalCancel(self) -> None:
        self.global_cancels += 1

    def openTrades(self) -> list[Trade]:
        return [t for t in self.all_trades if not t.isDone()]

    def trades(self) -> list[Trade]:
        return list(self.all_trades)

    async def reqAllOpenOrdersAsync(self) -> list[Trade]:
        return self.openTrades() + list(self.foreign_open_trades)

    async def reqExecutionsAsync(self, execFilter: object = None) -> list[Fill]:
        return list(self.executions)

    def portfolio(self, account: str = "") -> list[PortfolioItem]:
        return list(self.portfolio_items)
```

- [ ] **Step 2: Write the failing tests**

Create `tests/brokers/ibkr/test_client.py`:

```python
from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator

import pytest
from tests.fakes import FakeIB

from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.config.env import IbkrSettings
from trading_agent_framework.utils.errors import BrokerError

SETTINGS = IbkrSettings(host="127.0.0.1", port=4002, client_id=7, is_paper=True)


@pytest.fixture
def ib() -> FakeIB:
    return FakeIB()


@pytest.fixture
def connection(ib: FakeIB) -> Iterator[IbkrConnection]:
    conn = IbkrConnection(SETTINGS, ib_factory=lambda: ib, call_timeout=1.0, sleep=lambda seconds: None)
    conn.start()
    yield conn
    conn.disconnect()


def test_connect_uses_the_settings(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()

    assert ib.connect_calls == [("127.0.0.1", 4002, 7)]


def test_call_returns_plain_and_awaited_results(connection: IbkrConnection) -> None:
    connection.connect()

    assert connection.call(lambda ib: ib.managedAccounts()) == ["DU123"]
    assert connection.call(lambda ib: ib.accountSummaryAsync())[0].tag == "TotalCashValue"


def test_call_runs_on_the_loop_thread(connection: IbkrConnection) -> None:
    connection.connect()

    assert connection.call(lambda ib: threading.current_thread().name) == "ibkr-event-loop"


def test_call_timeout_is_a_broker_error(connection: IbkrConnection) -> None:
    connection.connect()

    with pytest.raises(BrokerError, match="IB Gateway did not answer within 0.1s"):
        connection.call(lambda ib: asyncio.sleep(5), timeout=0.1)


def test_call_wraps_exceptions(connection: IbkrConnection) -> None:
    connection.connect()

    def boom(ib: FakeIB) -> None:
        raise RuntimeError("socket closed")

    with pytest.raises(BrokerError, match="IB Gateway call failed: socket closed"):
        connection.call(boom)


def test_a_failed_connect_is_a_broker_error(connection: IbkrConnection, ib: FakeIB) -> None:
    ib.connect_errors = [ConnectionRefusedError("refused")]

    with pytest.raises(BrokerError, match="Could not connect to IB Gateway at 127.0.0.1:4002"):
        connection.connect()


def test_call_before_connect_is_refused(connection: IbkrConnection) -> None:
    with pytest.raises(BrokerError, match="not connected"):
        connection.call(lambda ib: ib.managedAccounts())


def test_a_dropped_connection_reconnects_then_runs_the_reconcile_hook(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()
    reconciled: list[str] = []
    connection.on_reconnect = lambda: reconciled.append("sync")
    ib.disconnect()  # the Gateway restarted

    assert connection.call(lambda ib: ib.managedAccounts()) == ["DU123"]
    assert len(ib.connect_calls) == 2
    assert reconciled == ["sync"]


def test_reconnect_gives_up_after_the_configured_attempts(connection: IbkrConnection, ib: FakeIB) -> None:
    connection.connect()
    ib.disconnect()
    ib.connect_errors = [ConnectionRefusedError("down")] * 3

    with pytest.raises(BrokerError, match="lost the IB Gateway connection"):
        connection.call(lambda ib: ib.managedAccounts())
    assert len(ib.connect_calls) == 4


def test_disconnect_is_idempotent(ib: FakeIB) -> None:
    conn = IbkrConnection(SETTINGS, ib_factory=lambda: ib)
    conn.start()
    conn.connect()

    conn.disconnect()
    conn.disconnect()

    assert ib.connected is False
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.ibkr.client'`.

- [ ] **Step 4: Implement `brokers/ibkr/client.py`**

```python
"""`IbkrConnection`: owns an `ib_async.IB` on a dedicated asyncio event-loop thread.

`ib_async` is asyncio-native and runs its event handlers on its loop, while strategy code runs
on the executor thread. Every IB access therefore goes through `call()`, which runs a function
on the loop and blocks the caller for the result -- with a timeout, so a hung Gateway can never
block the strategy forever. This is the only threads/asyncio code in `brokers/ibkr/`.

Never call `call()` from inside a function passed to `call()`: it already runs on the loop
thread and would wait on itself.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from trading_agent_framework.config.env import IbkrSettings
from trading_agent_framework.utils.errors import BrokerError

logger = logging.getLogger(__name__)


def _default_ib_factory() -> Any:
    from ib_async import IB

    return IB()


class IbkrConnection:
    def __init__(
        self,
        settings: IbkrSettings,
        *,
        ib_factory: Callable[[], Any] | None = None,
        call_timeout: float = 30.0,
        connect_timeout: float = 15.0,
        reconnect_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._ib_factory = ib_factory or _default_ib_factory
        self._call_timeout = call_timeout
        self._connect_timeout = connect_timeout
        self._reconnect_attempts = reconnect_attempts
        self._sleep = sleep
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ib: Any = None
        self._has_connected = False
        self.on_reconnect: Callable[[], None] | None = None

    def start(self) -> None:
        """Start the loop thread and create the `IB` object on it."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="ibkr-event-loop")
        self._thread.start()
        self._ib = self._run(lambda _: self._ib_factory(), self._call_timeout)

    def connect(self) -> None:
        settings = self._settings
        try:
            self._run(
                lambda ib: ib.connectAsync(settings.host, settings.port, clientId=settings.client_id, timeout=self._connect_timeout),
                self._connect_timeout + 5,
            )
        except BrokerError as exc:
            raise BrokerError(
                f"Could not connect to IB Gateway at {settings.host}:{settings.port} (client id {settings.client_id}): {exc}"
            ) from exc
        self._has_connected = True
        logger.info("Connected to IB Gateway at %s:%s (client id %s)", settings.host, settings.port, settings.client_id)

    def call[T](self, fn: Callable[[Any], T | Awaitable[T]], *, timeout: float | None = None) -> T:
        """Run `fn(ib)` on the loop thread, awaiting it if it returns an awaitable."""
        self._ensure_connected()
        return self._run(fn, timeout if timeout is not None else self._call_timeout)

    def disconnect(self) -> None:
        loop = self._loop
        if loop is None:
            return
        if self._ib is not None:
            try:
                self._run(lambda ib: ib.disconnect() if ib.isConnected() else None, 5.0)
            except BrokerError:
                logger.exception("error disconnecting from IB Gateway")
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(5.0)
        loop.close()
        self._loop = None
        self._thread = None
        self._has_connected = False

    def _ensure_connected(self) -> None:
        if not self._has_connected:
            raise BrokerError("not connected to IB Gateway; call connect() first")
        if self._run(lambda ib: ib.isConnected(), self._call_timeout):
            return
        logger.warning("IB Gateway connection lost; reconnecting")
        last_error: BrokerError | None = None
        for attempt in range(self._reconnect_attempts):
            if attempt:
                self._sleep(2.0**attempt)
            try:
                self.connect()
            except BrokerError as exc:
                last_error = exc
                continue
            if self.on_reconnect is not None:
                self.on_reconnect()  # connected again, so its own call()s do not recurse
            return
        raise BrokerError(f"lost the IB Gateway connection and could not reconnect: {last_error}") from last_error

    def _run[T](self, fn: Callable[[Any], T | Awaitable[T]], timeout: float) -> T:
        loop = self._loop
        if loop is None:
            raise BrokerError("IBKR connection not started; call start() first")

        async def runner() -> T:
            result = fn(self._ib)
            if inspect.isawaitable(result):
                return await result
            return result

        future = asyncio.run_coroutine_threadsafe(runner(), loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise BrokerError(f"IB Gateway did not answer within {timeout}s") from exc
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"IB Gateway call failed: {exc}") from exc
```

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS. A hanging test means a `call()`/`_run()` ran inside a function already on the loop thread.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/client.py tests/fakes.py tests/brokers/ibkr/test_client.py
git commit -m "feat: IbkrConnection runs ib_async on its own loop thread with timeouts and reconnect (Task 8)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Order events (`brokers/ibkr/events.py`)

**Files:**
- Create: `src/trading_agent_framework/brokers/ibkr/events.py`
- Test: `tests/brokers/ibkr/test_events.py`

**Interfaces:**
- Consumes: `orders.map_status_event`, `orders.to_decimal`, `orders.rejection_message` (Task 6); `OrderTracker`; `FakeIB`, `make_ib_trade`, `make_ib_fill` (Tasks 6, 8).
- Produces:
  - `INFO_CODES: frozenset[int]`
  - `IbkrOrderEvents(tracker: OrderTracker)` with `.register(ib)`, `.unregister(ib)`, `.on_order_status(trade) -> bool`, `.on_exec_details(trade, fill) -> bool`, `.apply_fill(fill) -> bool`, `.on_error(req_id, error_code, error_string, contract) -> None`

Rules:
- Orders are found by `orderRef` through `tracker.get_tracked_order_by_client_order_id`.
- `NEW` applies only while the order is `UNPROCESSED` or `SUBMITTED` (IBKR sends `PreSubmitted` then `Submitted`; a second NEW would move a partially filled order back).
- `ERROR`/`CANCELED` are ignored while the order is still `UNPROCESSED`: `IbkrBroker._submit_order` owns that window and raises itself, so the failure is not reported twice. When `_submit_order` returns it marks the order `SUBMITTED`; from then on this module owns every transition.
- A repeated `CANCELED` is ignored.
- Fills: per-execution `shares` and `price`; `FILLED` when `cumQty >= order.quantity`, else `PARTIALLY_FILLED`; each `execId` applies once.

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/ibkr/test_events.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest
from tests.fakes import FakeIB, make_ib_fill, make_ib_trade

from trading_agent_framework.brokers.ibkr.events import IbkrOrderEvents
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order


def _tracked(tracker: OrderTracker, status: OrderStatus = OrderStatus.SUBMITTED) -> Order:
    order = Order("s", Asset("AAPL"), OrderSide.BUY, quantity=Decimal(10), identifier="abc", client_order_id="s:abc")
    tracker.track_unprocessed(order)
    order.status = status
    return order


def test_submitted_moves_the_order_to_new_once() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="PreSubmitted")) is True
    assert events.on_order_status(make_ib_trade(status="Submitted")) is False
    assert order.status is OrderStatus.NEW
    assert tracker.new.snapshot() == [order]


def test_untracked_orders_are_ignored() -> None:
    assert IbkrOrderEvents(OrderTracker()).on_order_status(make_ib_trade(order_ref="other:x")) is False


def test_inactive_records_the_reason_as_an_error() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)

    IbkrOrderEvents(tracker).on_order_status(make_ib_trade(status="Inactive", log_message="margin required", log_error_code=201))

    assert order.status is OrderStatus.ERROR
    assert order.error_message == "margin required (IBKR error 201)"


def test_errors_and_cancels_during_submission_are_left_to_submit_order() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker, status=OrderStatus.UNPROCESSED)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="Inactive")) is False
    assert events.on_order_status(make_ib_trade(status="Cancelled")) is False
    assert order.status is OrderStatus.UNPROCESSED


def test_cancel_is_applied_once() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    assert events.on_order_status(make_ib_trade(status="Cancelled")) is True
    assert events.on_order_status(make_ib_trade(status="ApiCancelled")) is False
    assert order.status is OrderStatus.CANCELED


def test_partial_then_full_fill() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)

    events.on_exec_details(make_ib_trade(), make_ib_fill(exec_id="e1", shares=4.0, price=100.0, cum_qty=4.0, avg_price=100.0))
    assert order.status is OrderStatus.PARTIAL_FILL
    events.on_exec_details(make_ib_trade(), make_ib_fill(exec_id="e2", shares=6.0, price=101.0, cum_qty=10.0, avg_price=100.6))

    assert order.status is OrderStatus.FILL
    assert order.filled_quantity == Decimal(10)
    assert [(t.quantity, t.price) for t in order.transactions] == [(Decimal(4), Decimal(100)), (Decimal(6), Decimal(101))]
    assert order.avg_fill_price == Decimal("100.6")


def test_a_replayed_execution_is_ignored() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)
    fill = make_ib_fill(exec_id="e1", shares=4.0, cum_qty=4.0)

    assert events.apply_fill(fill) is True
    assert events.apply_fill(fill) is False
    assert order.filled_quantity == Decimal(4)


def test_informational_codes_are_debug_only(caplog: pytest.LogCaptureFixture) -> None:
    events = IbkrOrderEvents(OrderTracker())

    events.on_error(-1, 2104, "Market data farm connection is OK:usfarm", None)
    events.on_error(12, 201, "Order rejected", None)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == ["IB Gateway error 201 (request 12): Order rejected"]


def test_register_wires_and_unregister_unwires_the_handlers() -> None:
    tracker = OrderTracker()
    order = _tracked(tracker)
    events = IbkrOrderEvents(tracker)
    ib = FakeIB()

    events.register(ib)
    ib.orderStatusEvent.emit(make_ib_trade(status="Submitted"))
    assert order.status is OrderStatus.NEW

    events.unregister(ib)
    ib.orderStatusEvent.emit(make_ib_trade(status="Cancelled"))
    assert order.status is OrderStatus.NEW
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.ibkr.events'`.

- [ ] **Step 3: Implement**

```python
"""IBKR order-event handlers: `ib_async` order-status, execution and error events -> `OrderTracker`.

They run on `IbkrConnection`'s loop thread and only feed the tracker (never strategy hooks),
exactly like `AlpacaTradeStream`. Fills come from executions (price and quantity per fill);
status updates carry NEW / CANCELED / ERROR only.
"""

from __future__ import annotations

import logging
import threading
from decimal import Decimal
from typing import Any

from trading_agent_framework.brokers.ibkr import orders
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus
from trading_agent_framework.entities.order import Order

logger = logging.getLogger(__name__)

# "connection OK"-style notices IB Gateway sends with an error code; never order failures.
INFO_CODES: frozenset[int] = frozenset({2100, 2104, 2106, 2107, 2108, 2119, 2150, 2158})

_NEW_FROM = frozenset({OrderStatus.UNPROCESSED, OrderStatus.SUBMITTED})


class IbkrOrderEvents:
    def __init__(self, tracker: OrderTracker) -> None:
        self._tracker = tracker
        self._applied_exec_ids: set[str] = set()
        self._lock = threading.Lock()

    def register(self, ib: Any) -> None:
        ib.orderStatusEvent += self.on_order_status
        ib.execDetailsEvent += self.on_exec_details
        ib.errorEvent += self.on_error

    def unregister(self, ib: Any) -> None:
        ib.orderStatusEvent -= self.on_order_status
        ib.execDetailsEvent -= self.on_exec_details
        ib.errorEvent -= self.on_error

    def _find(self, order_ref: str) -> Order | None:
        if not order_ref:
            return None
        return self._tracker.get_tracked_order_by_client_order_id(order_ref)

    def on_order_status(self, trade: Any) -> bool:
        event = orders.map_status_event(trade.orderStatus.status)
        order = self._find(trade.order.orderRef)
        if event is None or order is None:
            return False
        if event is OrderEvent.NEW and order.status not in _NEW_FROM:
            return False
        if event in (OrderEvent.ERROR, OrderEvent.CANCELED) and order.status is OrderStatus.UNPROCESSED:
            return False  # _submit_order is still waiting on this order and reports it itself
        if event is OrderEvent.CANCELED and order.status is OrderStatus.CANCELED:
            return False
        if event is OrderEvent.ERROR:
            order.set_error(orders.rejection_message(trade) or "rejected by IBKR")
        order.update_raw(trade)
        self._tracker.process_trade_event(order, event)
        return True

    def on_exec_details(self, trade: Any, fill: Any) -> bool:
        return self.apply_fill(fill)

    def apply_fill(self, fill: Any) -> bool:
        """Apply one execution (live, or replayed by the post-reconnect reconcile) exactly once."""
        execution = fill.execution
        order = self._find(execution.orderRef)
        if order is None:
            return False
        with self._lock:
            if execution.execId in self._applied_exec_ids:
                return False
            self._applied_exec_ids.add(execution.execId)
        price = orders.to_decimal(execution.price) or Decimal(0)
        shares = orders.to_decimal(execution.shares) or Decimal(0)
        cumulative = orders.to_decimal(execution.cumQty) or Decimal(0)
        average = orders.to_decimal(execution.avgPrice)
        if average:
            order.avg_fill_price = average
        complete = order.quantity is not None and cumulative >= order.quantity
        event = OrderEvent.FILLED if complete else OrderEvent.PARTIALLY_FILLED
        self._tracker.process_trade_event(order, event, price=price, filled_quantity=shares)
        return True

    def on_error(self, req_id: int, error_code: int, error_string: str, contract: Any) -> None:
        if error_code in INFO_CODES:
            logger.debug("IB Gateway notice %s: %s", error_code, error_string)
            return
        logger.warning("IB Gateway error %s (request %s): %s", error_code, req_id, error_string)
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/events.py tests/brokers/ibkr/test_events.py
git commit -m "feat: IBKR order-status and execution events feed the OrderTracker (Task 9)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: `IbkrBroker` trading and account operations

**Files:**
- Create: `src/trading_agent_framework/brokers/ibkr/broker.py`
- Test: `tests/brokers/ibkr/test_broker.py`

**Interfaces:**
- Consumes: `IbkrConnection` (Task 8), `IbkrOrderEvents` (Task 9), `orders`/`account` (Tasks 5-7), `AlpacaMarketData` (Task 2), `AlpacaMarketClock`.
- Produces:
  - `IbkrBroker(strategy_name: str, connection: IbkrConnection, *, market_data: AlpacaMarketData, client_id: int, tracker: OrderTracker | None = None, clock: MarketClock | None = None, is_paper: bool = True, news_provider_factory: Callable[[], NewsProvider] | None = None, ack_timeout: float = 5.0)`, `name = "ibkr"`
  - every `Broker` abstract method, plus `configure_account() -> None`, `news_provider()`, `start_stream()`, `stop_stream(timeout: float = 5.0)`, `reconcile() -> None` (public; installed as `connection.on_reconnect`)
  - `account_id` property (cached `managedAccounts()[0]`; `ConfigurationError` unless exactly one account)

- [ ] **Step 1: Write the failing tests**

Create `tests/brokers/ibkr/test_broker.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest
from tests.fakes import (
    FakeClock,
    FakeIB,
    FakeStockHistoricalDataClient,
    FakeTradingClient,
    et,
    make_alpaca_trade,
    make_ib_fill,
    make_ib_portfolio_item,
    make_ib_summary,
    make_ib_trade,
)

from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.config.env import IbkrSettings
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, OrderValidationError

AAPL = Asset("AAPL")
SETTINGS = IbkrSettings(host="127.0.0.1", port=4002, client_id=1, is_paper=True)


@pytest.fixture
def ib() -> FakeIB:
    return FakeIB()


@pytest.fixture
def broker(ib: FakeIB) -> Iterator[IbkrBroker]:
    connection = IbkrConnection(SETTINGS, ib_factory=lambda: ib, call_timeout=2.0, sleep=lambda s: None)
    connection.start()
    connection.connect()
    data = FakeStockHistoricalDataClient()
    data.trades = {"AAPL": make_alpaca_trade("AAPL", 100.5)}
    built = IbkrBroker(
        "s", connection, market_data=AlpacaMarketData(data, FakeTradingClient()), client_id=1,
        clock=FakeClock(et(2026, 9, 22, 10)), ack_timeout=0.2,
    )
    yield built
    connection.disconnect()


def _buy(quantity: str = "10", **overrides: object) -> Order:
    fields: dict[str, object] = {"strategy_name": "s", "asset": AAPL, "side": OrderSide.BUY, "quantity": Decimal(quantity)}
    fields.update(overrides)
    return Order(**fields)  # ty: ignore[invalid-argument-type]


# --- submit -----------------------------------------------------------------------------


def test_submit_places_a_smart_order_tagged_with_the_client_order_id(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())

    [(contract, ib_order)] = ib.placed
    assert (contract.symbol, contract.exchange) == ("AAPL", "SMART")
    assert contract.conId  # qualified
    assert ib_order.orderRef == f"s:{order.identifier}" == order.client_order_id
    assert order.status is OrderStatus.SUBMITTED
    assert broker.tracker.get_tracked_order(order.identifier) is order


def test_contracts_are_qualified_once_per_symbol(broker: IbkrBroker, ib: FakeIB) -> None:
    calls: list[str] = []
    original = ib.qualifyContractsAsync

    async def counting(*contracts):
        calls.append(contracts[0].symbol)
        return await original(*contracts)

    ib.qualifyContractsAsync = counting  # ty: ignore[invalid-assignment]
    broker.submit_order(_buy())
    broker.submit_order(_buy())

    assert calls == ["AAPL"]


def test_an_unknown_symbol_is_a_validation_error(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.unknown_symbols = {"AAPL"}

    with pytest.raises(OrderValidationError, match="does not know the US stock AAPL"):
        broker.submit_order(_buy())


def test_a_rejection_sets_the_error_before_raising_and_untracks(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.place_status = "Inactive"
    ib.place_log_message = "No trading permissions"
    ib.place_log_error_code = 201
    order = _buy()

    with pytest.raises(BrokerError, match="IBKR rejected order .*No trading permissions"):
        broker.submit_order(order)

    assert order.status is OrderStatus.ERROR
    assert order.error_message == "No trading permissions (IBKR error 201)"
    assert broker.tracker.get_tracked_order(order.identifier) is None


def test_notional_orders_are_rejected(broker: IbkrBroker) -> None:
    with pytest.raises(OrderValidationError, match="notional"):
        broker.submit_order(Order(strategy_name="s", asset=AAPL, side=OrderSide.BUY, notional=Decimal(100)))


def test_fractional_quantities_are_floored(broker: IbkrBroker, ib: FakeIB) -> None:
    broker.submit_order(_buy("2.6"))

    assert ib.placed[-1][1].totalQuantity == 2.0


# --- cancel / modify ----------------------------------------------------------------------


def test_cancel_cancels_the_matching_ib_order(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())

    broker.cancel_order(order)

    assert [o.orderRef for o in ib.canceled] == [order.client_order_id]


def test_cancel_of_an_unknown_order_is_a_broker_error(broker: IbkrBroker) -> None:
    with pytest.raises(BrokerError, match="no open IBKR order"):
        broker.cancel_order(_buy(client_order_id="s:missing"))


def test_orders_of_another_client_id_cannot_be_cancelled_or_modified(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.all_trades.append(make_ib_trade(order_ref="s:old", client_id=9, status="Submitted"))
    order = _buy(identifier="old", client_order_id="s:old")

    with pytest.raises(BrokerError, match="placed by client id 9"):
        broker.cancel_order(order)
    with pytest.raises(BrokerError, match="placed by client id 9"):
        broker.modify_order(order, limit_price=Decimal(99))


def test_modify_edits_the_order_in_place(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy(order_type=OrderType.LIMIT, limit_price=Decimal(100)))

    modified = broker.modify_order(order, limit_price=Decimal("99.5"))

    assert modified is order
    assert order.limit_price == Decimal("99.5")
    assert len(ib.all_trades) == 1
    assert ib.placed[-1][1].lmtPrice == 99.5
    assert ib.placed[-1][1].orderId == ib.placed[0][1].orderId


# --- pulls and positions ------------------------------------------------------------------


def test_pull_positions_reads_the_portfolio(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0), make_ib_portfolio_item(symbol="MSFT", position=0.0)]

    [position] = broker.pull_positions()

    assert (position.asset, position.quantity, position.side) == (AAPL, Decimal(10), PositionSide.LONG)


def test_pull_orders_and_pull_order(broker: IbkrBroker) -> None:
    order = broker.submit_order(_buy())

    assert [o.identifier for o in broker.pull_orders()] == [order.identifier]
    pulled = broker.pull_order(order.identifier)
    assert pulled is not None and pulled.client_order_id == order.client_order_id
    assert broker.pull_order("nope") is None


def test_close_position_sells_the_floored_fraction(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0)]

    order = broker.close_position(AAPL, Decimal("0.35"))

    assert order is not None
    assert (order.side, order.quantity, order.order_type) == (OrderSide.SELL, Decimal(3), OrderType.MARKET)


def test_close_position_without_a_position_is_none(broker: IbkrBroker) -> None:
    assert broker.close_position(AAPL) is None


def test_close_all_positions_cancels_then_sells_everything(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.portfolio_items = [make_ib_portfolio_item(symbol="AAPL", position=10.0), make_ib_portfolio_item(symbol="MSFT", position=5.0)]

    closed = broker.close_all_positions()

    assert ib.global_cancels == 1
    assert sorted((o.asset.symbol, o.quantity) for o in closed) == [("AAPL", Decimal(10)), ("MSFT", Decimal(5))]


def test_sync_open_orders_adopts_this_strategys_untracked_orders(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.foreign_open_trades = [make_ib_trade(order_ref="s:old", client_id=9, perm_id=77), make_ib_trade(order_ref="other:x", perm_id=78)]

    adopted = broker.sync_open_orders()

    assert [o.identifier for o in adopted] == ["old"]
    assert broker.tracker.get_tracked_order("old") is adopted[0]
    assert broker.sync_open_orders() == []


# --- account ------------------------------------------------------------------------------


def test_get_account(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.summary = make_ib_summary(cash="1000", net_liquidation="2500", buying_power="1000")

    assert broker.get_account() == AccountBalances(cash=Decimal(1000), portfolio_value=Decimal(2500), buying_power=Decimal(1000))


def test_configure_account_refuses_a_paper_mismatch(broker: IbkrBroker) -> None:
    broker.is_paper = False

    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        broker.configure_account()


def test_more_than_one_managed_account_is_refused(broker: IbkrBroker, ib: FakeIB) -> None:
    ib.accounts = ["DU1", "DU2"]

    with pytest.raises(ConfigurationError, match="exactly one account"):
        broker.get_account()


# --- data, events, reconcile ----------------------------------------------------------------


def test_market_data_comes_from_alpaca(broker: IbkrBroker) -> None:
    assert broker.get_last_price(AAPL) == Decimal("100.5")


def test_the_stream_feeds_the_tracker(broker: IbkrBroker, ib: FakeIB) -> None:
    broker.start_stream()
    order = broker.submit_order(_buy())
    trade = ib.all_trades[0]

    # Emitted on the loop thread, where ib_async emits them in production.
    broker._connection.call(lambda _: ib.orderStatusEvent.emit(trade))
    broker._connection.call(lambda _: ib.execDetailsEvent.emit(trade, make_ib_fill(order_ref=order.client_order_id, cum_qty=10.0)))

    assert order.status is OrderStatus.FILL


def test_reconcile_applies_missed_fills_once(broker: IbkrBroker, ib: FakeIB) -> None:
    order = broker.submit_order(_buy())
    ib.executions = [make_ib_fill(order_ref=order.client_order_id, exec_id="e9", shares=10.0, cum_qty=10.0)]

    broker.reconcile()
    broker.reconcile()

    assert order.status is OrderStatus.FILL
    assert order.filled_quantity == Decimal(10)
```

(Task 11 adds `AlpacaCredentials` to this file's `config.env` import when its tests need it.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers/ibkr/test_broker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trading_agent_framework.brokers.ibkr.broker'`.

- [ ] **Step 3: Implement `brokers/ibkr/broker.py`**

```python
"""`IbkrBroker`: IBKR (TWS API via IB Gateway) for trading, account and positions; Alpaca for
market data, the calendar and news.

Wiring only: translation lives in the pure `orders`/`account` modules, threads and asyncio in
`IbkrConnection`, and event handling in `IbkrOrderEvents`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, ClassVar

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.ibkr import account, orders
from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.brokers.ibkr.events import IbkrOrderEvents
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, OrderValidationError

if TYPE_CHECKING:
    from ib_async import Contract, Trade

logger = logging.getLogger(__name__)


async def _place_and_wait(ib: Any, contract: Contract, ib_order: Any, timeout: float) -> Trade:
    """Place the order, then wait (up to `timeout`) until IBKR acknowledges or rejects it."""
    trade = ib.placeOrder(contract, ib_order)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while trade.orderStatus.status in orders.PENDING_STATUSES and loop.time() < deadline:
        await asyncio.sleep(0.05)
    return trade


class IbkrBroker(Broker):
    name: ClassVar[str] = "ibkr"

    def __init__(
        self,
        strategy_name: str,
        connection: IbkrConnection,
        *,
        market_data: AlpacaMarketData,
        client_id: int,
        tracker: OrderTracker | None = None,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        news_provider_factory: Callable[[], NewsProvider] | None = None,
        ack_timeout: float = 5.0,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(market_data.calendar_client),
            is_paper=is_paper,
        )
        self._connection = connection
        self._market_data = market_data
        self._client_id = client_id
        self._news_provider: NewsProvider | None = None
        self._news_provider_factory = news_provider_factory
        self._ack_timeout = ack_timeout
        self._events = IbkrOrderEvents(self.tracker)
        self._contracts: dict[str, Contract] = {}
        self._account_id: str | None = None
        connection.on_reconnect = self.reconcile

    # --- account ---------------------------------------------------------------------

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            accounts = self._connection.call(lambda ib: ib.managedAccounts())
            if len(accounts) != 1:
                raise ConfigurationError(
                    f"IB Gateway manages {len(accounts)} accounts ({', '.join(accounts)}); this framework needs exactly one account"
                )
            self._account_id = accounts[0]
        return self._account_id

    def _summary(self) -> list[Any]:
        account_id = self.account_id
        return self._connection.call(lambda ib: ib.accountSummaryAsync(account_id))

    def get_account(self) -> AccountBalances:
        return account.parse_account(self._summary(), self.account_id)

    def configure_account(self) -> None:
        """IBKR cannot switch margin/shorting off via the API: check the account instead."""
        for warning in account.check_account(self._summary(), self.account_id, is_paper=self.is_paper):
            logger.warning(warning)
        logger.info("IBKR account %s checked (paper=%s)", self.account_id, self.is_paper)

    # --- orders ----------------------------------------------------------------------

    def _conform_order(self, order: Order) -> Order:
        return orders.conform_order(order)

    def _qualified(self, asset: Asset) -> Contract:
        contract = self._contracts.get(asset.symbol)
        if contract is None:
            results = self._connection.call(lambda ib: ib.qualifyContractsAsync(orders.build_contract(asset)))
            contract = results[0] if results else None
            if contract is None:
                raise OrderValidationError(f"IBKR does not know the US stock {asset.symbol}")
            self._contracts[asset.symbol] = contract
        return contract

    def _submit_order(self, order: Order) -> Order:
        if not order.client_order_id:
            order.client_order_id = orders.client_order_id_for(self.strategy_name, order.identifier)
        ib_order = orders.build_order(order)
        contract = self._qualified(order.asset)
        # Tracked before placing: the event handlers can see this order before the call returns.
        self.tracker.track_unprocessed(order)
        try:
            trade = self._connection.call(lambda ib: _place_and_wait(ib, contract, ib_order, self._ack_timeout))
        except Exception as exc:
            order.set_error(exc)
            logger.exception("Failed to submit order %s", order.identifier)
            self.tracker.untrack(order)
            raise  # set_error BEFORE re-raising -- lumibot's contract
        rejection = orders.rejection_message(trade)
        if rejection is not None:
            order.set_error(rejection)
            self.tracker.untrack(order)
            raise BrokerError(f"IBKR rejected order {order.identifier}: {rejection}")
        order.update_raw(trade)
        if order.status is OrderStatus.UNPROCESSED:
            order.status = OrderStatus.SUBMITTED  # from here on, IbkrOrderEvents owns its status
        return order

    def _open_trade(self, order: Order) -> Trade:
        client_order_id = order.client_order_id
        trade = self._connection.call(
            lambda ib: next((t for t in ib.openTrades() if client_order_id and t.order.orderRef == client_order_id), None)
        )
        if trade is None:
            raise BrokerError(f"no open IBKR order for {order.identifier}")
        if trade.order.clientId != self._client_id:
            raise BrokerError(
                f"order {order.identifier} was placed by client id {trade.order.clientId}; "
                f"only that client id can change it (this broker is client id {self._client_id})"
            )
        return trade

    def cancel_order(self, order: Order) -> None:
        trade = self._open_trade(order)
        self._connection.call(lambda ib: ib.cancelOrder(trade.order))

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """IBKR edits the order in place: the same `Order` comes back with the new prices."""
        trade = self._open_trade(order)

        def replace(ib: Any) -> Trade:
            if limit_price is not None:
                trade.order.lmtPrice = float(limit_price)
            if stop_price is not None:
                trade.order.auxPrice = float(stop_price)
            return ib.placeOrder(trade.contract, trade.order)

        self._connection.call(replace)
        if limit_price is not None:
            if order.order_type is OrderType.STOP_LIMIT:
                order.stop_limit_price = limit_price
            else:
                order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        self.tracker.process_trade_event(order, OrderEvent.MODIFIED)
        return order

    def pull_order(self, identifier: str) -> Order | None:
        return next((o for o in self.pull_orders(limit=10_000) if o.identifier == identifier), None)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        """Open orders plus those completed this session (IBKR's API keeps no deeper history)."""
        trades = self._connection.call(lambda ib: ib.trades())
        return [orders.parse_trade(t, self.strategy_name) for t in trades][-limit:]

    def pull_positions(self) -> list[Position]:
        items = self._connection.call(lambda ib: ib.portfolio())
        parsed = (orders.parse_portfolio_item(item, self.strategy_name) for item in items)
        return [p for p in parsed if p is not None]

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = next((p for p in self.pull_positions() if p.asset == asset), None)
        if position is None:
            return None
        quantity = (position.quantity * fraction).to_integral_value(rounding=ROUND_FLOOR)
        if quantity < 1:
            logger.warning("close_position(%s, %s): below one whole share, nothing sent", asset.symbol, fraction)
            return None
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        return self.submit_order(Order(self.strategy_name, asset, side, OrderType.MARKET, quantity=quantity))

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            self._connection.call(lambda ib: ib.reqGlobalCancel())
        closed = (self.close_position(p.asset) for p in self.pull_positions())
        return [o for o in closed if o is not None]

    def sync_open_orders(self) -> list[Order]:
        trades = self._connection.call(lambda ib: ib.reqAllOpenOrdersAsync())
        adopted: list[Order] = []
        for trade in trades:
            identifier = orders.identifier_from_order_ref(trade.order.orderRef, self.strategy_name)
            if identifier is None or self.tracker.get_tracked_order(identifier) is not None:
                continue
            order = orders.parse_trade(trade, self.strategy_name)
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted

    def reconcile(self) -> None:
        """After a reconnect: adopt open orders, then apply fills missed while disconnected."""
        self.sync_open_orders()
        fills = self._connection.call(lambda ib: ib.reqExecutionsAsync())
        for fill in fills:
            self._events.apply_fill(fill)

    # --- market data, clock, news (Alpaca) ---------------------------------------------

    def news_provider(self) -> NewsProvider | None:
        if self._news_provider is None and self._news_provider_factory is not None:
            self._news_provider = self._news_provider_factory()
        return self._news_provider

    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self._market_data.get_last_price(asset)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return self._market_data.get_last_prices(assets)

    def get_quote(self, asset: Asset) -> Quote | None:
        return self._market_data.get_quote(asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self._market_data.get_bars(
            assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours
        )

    # --- stream ----------------------------------------------------------------------

    def start_stream(self) -> None:
        self._connection.call(lambda ib: self._events.register(ib))

    def stop_stream(self, timeout: float = 5.0) -> None:
        try:
            self._connection.call(lambda ib: self._events.unregister(ib), timeout=timeout)
        except BrokerError:
            logger.exception("error unregistering IBKR order events")
        self._connection.disconnect()
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/brokers/ibkr -v && uv run ruff check`
Expected: PASS.

Debugging hint: `test_submit_...` expects `SUBMITTED` because the fixture never calls `start_stream()`; if it sees `NEW`, the handlers were registered somewhere else.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers/ibkr/broker.py tests/brokers/ibkr/test_broker.py
git commit -m "feat: IbkrBroker trading, account and position operations (Task 10)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: `IbkrBroker.from_settings`, factory wiring, lazy export

**Files:**
- Modify: `src/trading_agent_framework/brokers/ibkr/broker.py`
- Modify: `src/trading_agent_framework/brokers/factory.py`
- Modify: `src/trading_agent_framework/brokers/__init__.py`
- Test: `tests/brokers/ibkr/test_broker.py`, `tests/brokers/test_factory.py`, `tests/brokers/test_lazy_imports.py`

**Interfaces:**
- Consumes: Task 10's `IbkrBroker`; `lazy_news_provider` (Task 2); `IbkrSettings`, `AlpacaCredentials` (Task 1).
- Produces: `IbkrBroker.from_settings(strategy_name: str, settings: IbkrSettings, *, data: AlpacaCredentials, news: Callable[[], AlpacaCredentials] | None = None, connection: IbkrConnection | None = None, market_data: AlpacaMarketData | None = None) -> IbkrBroker`; `factory.BUILDERS[BrokerKind.IBKR]`; `trading_agent_framework.brokers.IbkrBroker` (lazy).

- [ ] **Step 1: Write the failing tests**

In `tests/brokers/ibkr/test_broker.py`, change the import to `from trading_agent_framework.config.env import AlpacaCredentials, IbkrSettings`, then append:

```python
def test_from_settings_connects_and_checks_the_account(ib: FakeIB) -> None:
    connection = IbkrConnection(SETTINGS, ib_factory=lambda: ib)

    built = IbkrBroker.from_settings(
        "s", SETTINGS, data=AlpacaCredentials("k", "s"), connection=connection,
        market_data=AlpacaMarketData(FakeStockHistoricalDataClient(), FakeTradingClient()),
    )
    try:
        assert ib.connect_calls == [("127.0.0.1", 4002, 1)]
        assert built.is_paper is True
        assert built.account_id == "DU123"
    finally:
        connection.disconnect()


def test_from_settings_disconnects_when_the_account_check_fails(ib: FakeIB) -> None:
    live = IbkrSettings(host="127.0.0.1", port=4001, client_id=1, is_paper=False)  # but the account is DU123
    connection = IbkrConnection(live, ib_factory=lambda: ib)

    with pytest.raises(ConfigurationError, match="BROKER_API_IS_PAPER"):
        IbkrBroker.from_settings(
            "s", live, data=AlpacaCredentials("k", "s"), connection=connection,
            market_data=AlpacaMarketData(FakeStockHistoricalDataClient(), FakeTradingClient()),
        )

    assert ib.connected is False
```

Append to `tests/brokers/test_factory.py`:

```python
def test_the_ibkr_builder_passes_settings_and_alpaca_data_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    captured: dict[str, object] = {}

    def fake_from_settings(cls, strategy_name, settings, *, data, news, connection=None, market_data=None):
        captured.update(settings=settings, data=data, news=news)
        return _fake_broker(strategy_name)

    monkeypatch.setattr(IbkrBroker, "from_settings", classmethod(fake_from_settings))
    env = {"BROKER": "ibkr", "BROKER_API_IS_PAPER": "false", "IBKR_CLIENT_ID": "3", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"}

    factory.build_broker("s", env=env)

    settings = captured["settings"]
    assert (settings.port, settings.client_id, settings.is_paper) == (4001, 3, False)
    assert captured["data"].api_key == "dk"


def test_the_ibkr_builder_does_not_need_alpaca_trading_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    monkeypatch.setattr(IbkrBroker, "from_settings", classmethod(lambda cls, name, settings, **kwargs: _fake_broker(name)))

    broker = factory.build_broker("s", env={"BROKER": "ibkr", "ALPACA_DATA_API_KEY": "dk", "ALPACA_DATA_API_SECRET": "ds"})

    assert broker.strategy_name == "s"
```

Append to `tests/brokers/test_lazy_imports.py`:

```python
def test_importing_brokers_and_the_factory_does_not_import_ib_async() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import trading_agent_framework.brokers\nimport trading_agent_framework.brokers.factory\nimport sys\nassert 'ib_async' not in sys.modules\nprint('OK')\n"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_ibkr_broker_is_a_lazy_export() -> None:
    import trading_agent_framework.brokers as brokers
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    assert brokers.IbkrBroker is IbkrBroker
    assert "IbkrBroker" in dir(brokers)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/brokers -v`
Expected: FAIL (`IbkrBroker` has no `from_settings`; `BrokerKind.IBKR` has no builder; `brokers.IbkrBroker` missing).

- [ ] **Step 3: Implement**

In `brokers/ibkr/broker.py` add `from trading_agent_framework.brokers.alpaca.news import lazy_news_provider` and, under `TYPE_CHECKING`, `from trading_agent_framework.config.env import AlpacaCredentials, IbkrSettings`. Add to the class:

```python
    @classmethod
    def from_settings(
        cls,
        strategy_name: str,
        settings: IbkrSettings,
        *,
        data: AlpacaCredentials,
        news: Callable[[], AlpacaCredentials] | None = None,
        connection: IbkrConnection | None = None,
        market_data: AlpacaMarketData | None = None,
    ) -> IbkrBroker:
        """Connect to IB Gateway and check the account; any failure disconnects before raising."""
        market_data = market_data if market_data is not None else AlpacaMarketData.from_credentials(data)
        connection = connection if connection is not None else IbkrConnection(settings)
        connection.start()
        try:
            connection.connect()
            broker = cls(
                strategy_name,
                connection,
                market_data=market_data,
                client_id=settings.client_id,
                is_paper=settings.is_paper,
                news_provider_factory=lazy_news_provider(news) if news is not None else None,
            )
            broker.configure_account()
        except BaseException:
            connection.disconnect()
            raise
        return broker
```

In `brokers/factory.py`, import `IbkrSettings` from `config.env` and add:

```python
def _build_ibkr(strategy_name: str, settings: BrokerSettings, env: Mapping[str, str] | None) -> Broker:
    from trading_agent_framework.brokers.ibkr.broker import IbkrBroker

    return IbkrBroker.from_settings(
        strategy_name,
        IbkrSettings.from_env(settings.is_paper, env),
        data=AlpacaCredentials.for_data(env),
        news=lambda: AlpacaCredentials.for_news(env),
    )
```

and register it: `BUILDERS = {BrokerKind.ALPACA: _build_alpaca, BrokerKind.IBKR: _build_ibkr}`.

In `brokers/__init__.py`: add `from trading_agent_framework.brokers.ibkr.broker import IbkrBroker` under `TYPE_CHECKING`, `"IbkrBroker"` to `__all__`, and `"IbkrBroker": (".ibkr.broker", "IbkrBroker")` to `_LAZY`; extend the module docstring: "`IbkrBroker` lives in `brokers.ibkr` and imports `ib_async`; it is lazy for the same reason."

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest && uv run ruff check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/trading_agent_framework/brokers tests/brokers
git commit -m "feat: BROKER=ibkr builds a connected, checked IbkrBroker (Task 11)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Smoke scripts, comments and documentation

**Files:**
- Create: `scripts/tests/smoke_ibkr_account.py`, `scripts/tests/smoke_ibkr_orders.py`
- Modify: `src/trading_agent_framework/agents/tools/trading.py:98` (comment)
- Modify: `src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py:117` (comment)
- Modify: `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `build_broker` (Tasks 4, 11), `IbkrBroker.account_id`, `ibkr.account.is_paper_account`, `OrderTracker.listeners` (`list[Callable[[Order, OrderEvent], None]]`).

- [ ] **Step 1: Write `scripts/tests/smoke_ibkr_account.py` (read-only)**

```python
"""Manual smoke test: connect to a PAPER IB Gateway, check the account, print balances and positions.

Read-only: places no orders. Needs IB Gateway running and logged into a paper account, and
`env/.env.ibkr.integration-tests` with BROKER=ibkr, BROKER_API_IS_PAPER=true, optional IBKR_*,
and ALPACA_DATA_API_KEY / ALPACA_DATA_API_SECRET.

    uv run python scripts/tests/smoke_ibkr_account.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.config.env import BrokerKind, BrokerSettings, find_project_root
from trading_agent_framework.entities.asset import Asset

STRATEGY_NAME = "smoke_ibkr"
ENV_FILE = find_project_root() / "env" / ".env.ibkr.integration-tests"


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"Credentials file not found: {ENV_FILE}", file=sys.stderr)
        return 1
    load_dotenv(ENV_FILE, override=True)
    settings = BrokerSettings.from_env()
    if settings.kind is not BrokerKind.IBKR or not settings.is_paper:
        print("Refusing to run: needs BROKER=ibkr and BROKER_API_IS_PAPER=true.", file=sys.stderr)
        return 1
    broker = build_broker(STRATEGY_NAME)  # connects and runs configure_account()
    assert isinstance(broker, IbkrBroker)
    try:
        print(f"Account {broker.account_id}: {broker.get_account()}")
        for position in broker.pull_positions():
            print(f"  {position.asset.symbol}: {position.quantity} @ {position.avg_fill_price}")
        print(f"SPY last price (Alpaca IEX): {broker.get_last_price(Asset('SPY'))}")
        print(f"Next session: {broker.clock.next_session()}")
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write `scripts/tests/smoke_ibkr_orders.py` (paper only)**

```python
"""Manual smoke test: order round trip against a PAPER IB Gateway.

Refuses to run unless the logged-in account is a paper (DU...) account. Places a far-from-market
limit buy, modifies it and cancels it; then buys and sells 1 share at market (fills only during
market hours). Same env file as smoke_ibkr_account.py.

    uv run python scripts/tests/smoke_ibkr_orders.py
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal

from dotenv import load_dotenv

from trading_agent_framework.brokers.factory import build_broker
from trading_agent_framework.brokers.ibkr.account import is_paper_account
from trading_agent_framework.brokers.ibkr.broker import IbkrBroker
from trading_agent_framework.config.env import BrokerKind, BrokerSettings, find_project_root
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType
from trading_agent_framework.entities.order import Order

STRATEGY_NAME = "smoke_ibkr"
ENV_FILE = find_project_root() / "env" / ".env.ibkr.integration-tests"
SPY = Asset("SPY")


def _wait_for(order: Order, statuses: set[OrderStatus], seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while order.status not in statuses and time.monotonic() < deadline:
        time.sleep(0.2)
    print(f"  {order.identifier}: {order.status}")


def main() -> int:
    if not ENV_FILE.is_file():
        print(f"Credentials file not found: {ENV_FILE}", file=sys.stderr)
        return 1
    load_dotenv(ENV_FILE, override=True)
    settings = BrokerSettings.from_env()
    if settings.kind is not BrokerKind.IBKR or not settings.is_paper:
        print("Refusing to run: needs BROKER=ibkr and BROKER_API_IS_PAPER=true.", file=sys.stderr)
        return 1
    broker = build_broker(STRATEGY_NAME)
    assert isinstance(broker, IbkrBroker)
    try:
        if not is_paper_account(broker.account_id):
            print("Refusing to run: IB Gateway is not logged into a paper account.", file=sys.stderr)
            return 1
        broker.tracker.listeners.append(lambda order, event: print(f"  event {event} for {order.identifier}"))
        broker.sync_open_orders()
        broker.start_stream()
        last = broker.get_last_price(SPY)
        if last is None:
            print("No SPY price from Alpaca.", file=sys.stderr)
            return 1
        far = (last * Decimal("0.5")).quantize(Decimal("0.01"))
        print(f"1) limit buy 1 SPY @ {far} (last {last})")
        limit = broker.submit_order(Order(STRATEGY_NAME, SPY, OrderSide.BUY, OrderType.LIMIT, quantity=Decimal(1), limit_price=far))
        _wait_for(limit, {OrderStatus.NEW})
        print(f"2) modify to {far - 1}")
        broker.modify_order(limit, limit_price=far - 1)
        print("3) cancel")
        broker.cancel_order(limit)
        _wait_for(limit, {OrderStatus.CANCELED})
        print("4) market buy 1, then sell it (market hours only)")
        buy = broker.submit_order(Order(STRATEGY_NAME, SPY, OrderSide.BUY, quantity=Decimal(1)))
        _wait_for(buy, {OrderStatus.FILL}, 60)
        if buy.status is OrderStatus.FILL:
            sell = broker.close_position(SPY)
            if sell is not None:
                _wait_for(sell, {OrderStatus.FILL}, 60)
    finally:
        broker.stop_stream()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Lint and compile the scripts**

Run: `uv run ruff check scripts && uv run python -m py_compile scripts/tests/smoke_ibkr_account.py scripts/tests/smoke_ibkr_orders.py`
Expected: no output, exit 0.

- [ ] **Step 4: Update the two comments**

- `agents/tools/trading.py:98`: `# AlpacaBroker._submit_order re-raises a raw SDK exception on rejection` -> `# a broker's _submit_order may re-raise the underlying failure after order.set_error (lumibot contract)`.
- `strategies/cross_momentum/agent_cross_momentum.py:117`: append to the comment: "Market data is Alpaca's under every broker (BROKER=ibkr included), so this limiter always applies."

- [ ] **Step 5: Update `README.md`**

Add after the env-file section:

````markdown
## Interactive Brokers (IBKR)

Set `BROKER=ibkr` in the strategy's paper/live env file. IBKR handles orders, account and
positions; prices, bars, the market calendar and news still come from Alpaca (`ALPACA_DATA_*`,
`ALPACA_NEWS_*`). Backtests are unaffected (Yahoo or Alpaca data).

1. Install and start **IB Gateway**, and log in with your paper (or live) IBKR username.
2. In IB Gateway, *Configure > Settings > API > Settings*: enable "ActiveX and Socket Clients",
   untick "Read-Only API", keep socket port 4002 (paper) / 4001 (live), trust 127.0.0.1.
3. Use a **cash** account: the broker refuses to trade live on a margin account (IBKR cannot
   turn off margin or shorting through the API).
4. Give each strategy running at the same time its own `IBKR_CLIENT_ID`, and keep it stable
   across restarts: only the client id that placed an order can modify or cancel it.
5. IB Gateway logs out and restarts daily. For unattended runs use [IBC](https://github.com/IbcAlpha/IBC)
   to restart and log in automatically; the broker reconnects on its next call.
6. IBKR trades whole shares through the API: fractional quantities are rounded down, and
   notional (dollar-amount) orders are rejected.

Manual checks against a paper Gateway (env file `env/.env.ibkr.integration-tests`):

```bash
uv run python scripts/tests/smoke_ibkr_account.py   # read-only
uv run python scripts/tests/smoke_ibkr_orders.py    # places paper orders
```
````

- [ ] **Step 6: Update `CLAUDE.md`**

- Commands block: add
  - `uv run python scripts/tests/smoke_ibkr_account.py      # manual paper IB Gateway smoke test: account (read-only)`
  - `uv run python scripts/tests/smoke_ibkr_orders.py       # manual paper IB Gateway smoke test: orders`
- Architecture, add entries:
  - `brokers/factory.py` -- `build_broker()`: the paper/live broker named by `BROKER` (`alpaca` default, `ibkr`); broker classes are imported inside the builders.
  - `brokers/alpaca/data.py` -- `AlpacaMarketData`: Alpaca IEX prices/quotes/bars and the calendar client, shared by `AlpacaBroker` and `IbkrBroker`.
  - `brokers/ibkr/` -- `IbkrBroker` over the TWS API (`ib_async`, IB Gateway): **pure** `orders.py`/`account.py`; `client.py` (`IbkrConnection`, the only thread/asyncio code: every `IB` access goes through `call()` on its loop thread, with timeouts and reconnect); `events.py` (`IbkrOrderEvents`, feeds `OrderTracker` only); `broker.py` (wiring; market data, clock and news come from Alpaca).
  - `backtesting/placeholder.py` -- `PlaceholderBroker`: what `main.py` builds a strategy with in backtesting mode; no network.
- Update the `config/env.py` entry: "strategy/mode env file resolution, `BrokerSettings` (`BROKER`, `BROKER_API_IS_PAPER`), `IbkrSettings`, and `AlpacaCredentials.for_trading/for_news/for_data`."
- In the "Money is `Decimal`" gotcha, change "`orders.py` (Alpaca's SDK wants floats for some request fields)" to "`orders.py` (Alpaca's SDK wants floats for some request fields; `brokers/ibkr/orders.py` is the same boundary for `ib_async`)".
- Gotchas, add:
  - **Credential groups never fall back.** `ALPACA_API_*` (Alpaca trading), `ALPACA_DATA_*` (market data and calendar for both brokers, and `AlpacaBacktestData`), `ALPACA_NEWS_*` (the news tool). `ALPACA_IS_PAPER` is rejected (renamed `BROKER_API_IS_PAPER`).
  - **`ib_async` stays inside `brokers/ibkr/`** and `import trading_agent_framework.brokers` never imports it. Never call `IbkrConnection.call()` from a function passed to `call()`: it already runs on the loop thread and would deadlock.
  - **IBKR order identity is ours.** `identifier` is our id and `orderRef` carries `{strategy}:{identifier}`; `modify_order` edits in place and returns the same `Order`. Orders placed by another `IBKR_CLIENT_ID` are tracked but cannot be cancelled or modified.
  - **IBKR order-status hand-off.** While `_submit_order` waits (order `UNPROCESSED`) it owns rejections; once it returns, the order is `SUBMITTED` and `IbkrOrderEvents` owns every later transition.
  - **IBKR margin check is a heuristic.** IBKR's API does not say cash vs margin; `ibkr/account.check_account` treats buying power above ~1.05x cash as margin (refused live, warned on paper).

- [ ] **Step 7: Run everything and commit**

Run: `uv run pytest && uv run ruff check`
Expected: PASS.

```bash
git add scripts/tests/smoke_ibkr_account.py scripts/tests/smoke_ibkr_orders.py src/trading_agent_framework/agents/tools/trading.py src/trading_agent_framework/strategies/cross_momentum/agent_cross_momentum.py README.md CLAUDE.md
git commit -m "docs: IBKR smoke scripts, IB Gateway setup and architecture notes (Task 12)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec section | Task |
|---|---|
| 2 Env contract, rename guard | 1 (code), 4 (docs) |
| 3.1 `config/env.py` | 1 |
| 3.2 factory | 4, 11 |
| 3.3 call-site migration | 1, 2, 3 |
| 3.4 `main.py`, `PlaceholderBroker` | 3, 4 |
| 3.5 `AlpacaMarketData` | 2 |
| 4.1 package layout, `ib_async` confinement | 5-11 |
| 4.2 `IbkrConnection` | 8 |
| 4.3 order mapping | 5, 6 |
| 4.4 broker operations, `configure_account` | 7, 10 |
| 4.5 order events | 9 |
| 4.6 lifecycle, reconnect | 8, 10 (`reconcile`), 11 (`from_settings`) |
| 5 other touched code, docs | 4, 12 |
| 6 testing (automated + manual) | 1-12 |

**Deviation from the spec:** cash-vs-margin detection uses `BuyingPower > TotalCashValue * 1.05 + 1` (Task 7), because IBKR's `AccountType` summary tag does not distinguish cash from margin accounts. Recorded in `CLAUDE.md` in Task 12.

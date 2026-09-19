# Agent Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a LangChain trading agent everything it needs: `PrebuiltTools.all(strategy)` (orders, account, market data, indicators, memory in one call) plus explicitly-wired research tools (Alpaca news, FRED macro, SEC fundamentals), each gated on `strategy.clock.now()` so backtests never see future information.

**Architecture:** New package `agents/tools/` — one file per domain, each exposing a `<domain>_tools(strategy) -> list[Callable]` factory (mirrors `memory/tools.py`). `PrebuiltTools.all()` concatenates trading + account + market-data + indicator + memory tools. News extends the existing Alpaca broker layer (`market_data.py`/`client.py`/`broker.py`). Macro and fundamentals are broker-independent: macro lazily wraps `fredapi`; fundamentals gets a new top-level `fundamentals/` package (`sec.py` pure translation, `edgar_client.py` cached I/O), a trimmed port of lumibot's `SECFundamentals`.

**Tech Stack:** Python 3.14, `uv`, `langchain` (already a dependency), new: `fredapi` (macro), `httpx` (SEC EDGAR). `pytest`, `ruff`, `uv check`.

**Spec:** `docs/superpowers/specs/2026-09-19-agent-tools-design.md`. Read it before starting any task.

## Global Constraints

- Python `>=3.14`, package manager `uv`.
- Every tool factory is `<name>_tools(strategy) -> list[Callable[..., dict[str, Any]]]`: full type hints, a **one-line docstring** (sent to the model as the tool description), plain values in and out (`str`/`float`/`int`/`bool`/`list`/`dict` — no `Decimal`, no entities), errors as `{"error": ...}` rather than raised, mirroring `memory/tools.py`.
- Order-mutating tools (`submit_order`, `cancel_order`, `cancel_open_orders`, `close_position`, `sell_all`) get the function attribute `mutates_trading = True` (`vars(fn)["mutates_trading"] = True`), same convention as `memory/tools.py`'s `remember_decision`.
- `agents/tools/*.py` do **not** use `from __future__ import annotations` for any module whose functions LangChain will introspect via `inspect.signature` at tool-registration time — this repo's `memory/tools.py` already follows this rule; match it in every new `agents/tools/*.py` file (the `TYPE_CHECKING`-only `Strategy` import is fine since it's only used as a factory *parameter* type, not a returned tool's parameter type).
- `agents/tools/macro.py`'s `fredapi` import and `fundamentals/edgar_client.py`'s `httpx` import: `fredapi` is deferred (imported inside a function body, mirroring `core/indicators.py`'s `pandas_ta_classic` and `agents/manager.py`'s `langchain`) so a strategy that never wires in the macro tool doesn't pay for it; `httpx` is a normal top-level import (small, and `fundamentals/edgar_client.py` is only imported when a strategy wires in the fundamentals tool, same effect via Python's own lazy module loading).
- Every framework-domain exception (`BrokerError`, `OrderValidationError`, `MacroDataError`, `FundamentalsError`) is caught at the tool boundary and turned into `{"error": str(exc)}`; a raw library exception never escapes a tool, and a *tool's own* internal client/parsing code never lets a raw `httpx`/`fredapi`/SDK exception escape either — it's wrapped into the matching framework exception first (repo rule, see `utils/errors.py`).
- Tests never touch the network; hand-written fakes only (`tests/fakes.py`), never `MagicMock` — except `tests/brokers/alpaca/test_client.py`'s existing `MagicMock` pattern for asserting SDK-constructor call args, which Task 7 matches for `build_news_client`. `httpx.MockTransport` (real httpx, no sockets) is used for `fundamentals/edgar_client.py` tests.
- Every task ends green on `uv run pytest`, `uv run ruff check` and `uv check`. Ruff allows 200-character lines.
- Work on branch `feature/agent-tools`. Commit once per task with the message `Task N: <summary>`, ending with the attribution trailer lines from your instructions. **Never stage `TODO.md`**: it carries the user's own uncommitted edits (the user strikes "Tools" from the MIGRATION list themselves once this ships).
- **Footnoted judgment calls** (spec was silent or this plan trims scope for time — flagged here per writing-plans convention, not asked as questions since there's no user in this loop):
  1. Test depth is narrower than the `2026-09-11-agent-memory.md` plan's per-task exhaustiveness (that plan often has 15-20 tests per task; this one targets ~5-10) given the spec covers four subsystems in one pass. Each task's tests still cover every tool's happy path, its error path, and its lean-shape contract — just not every edge case lumibot's original code handles (e.g. `news`'s `start`/`end` parsing assumes tz-aware ISO strings; a naive-string input is an unhandled `TypeError`, not covered).
  2. `fundamentals/sec.py`'s `parse_filings` includes an internal `"primary_document"` key in each row so `get_filing_document` can resolve a document without a second network round-trip; `agents/tools/fundamentals.py`'s `get_filings` tool strips that key before returning to the model (the spec's `get_filings` shape has no `primary_document` field).
  3. `agents/tools/news.py` and `agents/tools/macro.py` accept the `Strategy` (not a raw client) per spec §3.2, but `macro_tools` also accepts an optional keyword-only `fred_client_factory` (defaulting to the real lazy `fredapi` factory) purely as a test seam — the spec doesn't mention this parameter but it's necessary to test `get_fred_series` without installing/calling real `fredapi`. Same pattern for `fundamentals_tools`'s optional `client: SecEdgarClient | None` parameter.

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/trading_agent_framework/agents/tools/trading.py` | 1 | `trading_tools(strategy)` |
| `src/trading_agent_framework/agents/tools/account.py` | 2 | `account_tools(strategy)` |
| `src/trading_agent_framework/agents/tools/market_data.py` | 3 | `market_data_tools(strategy)` |
| `src/trading_agent_framework/agents/tools/indicators.py` | 4 | `indicator_tools(strategy)` |
| `src/trading_agent_framework/agents/tools/prebuilt.py` | 5 | `PrebuiltTools.all(strategy)` |
| `src/trading_agent_framework/agents/tools/__init__.py` | 5, 14 | Exports (core factories in 5; news/macro/fundamentals added in 14) |
| `src/trading_agent_framework/brokers/alpaca/market_data.py` | 6 | `+ build_news_request`, `parse_news`, `AlpacaNewsClient` protocol |
| `src/trading_agent_framework/brokers/alpaca/client.py` | 7 | `+ build_news_client` |
| `src/trading_agent_framework/brokers/alpaca/broker.py` | 7 | `+ AlpacaBroker.get_news(...)` |
| `src/trading_agent_framework/agents/tools/news.py` | 8 | `news_tools(strategy)` |
| `src/trading_agent_framework/config/env.py` | 9 | `+ FredCredentials` |
| `src/trading_agent_framework/utils/errors.py` | 9, 12 | `+ MacroDataError` (9), `+ FundamentalsError` (12) |
| `pyproject.toml` | 9, 12 | `+ fredapi` (9), `+ httpx` (12) |
| `env/.env.example` | 9, 12 | `+ FRED_API_KEY` (9), `+ SEC_EDGAR_USER_AGENT` (12) |
| `src/trading_agent_framework/agents/tools/macro.py` | 10 | `macro_tools(strategy)` |
| `src/trading_agent_framework/fundamentals/sec.py` | 11 | Pure SEC translation |
| `src/trading_agent_framework/fundamentals/edgar_client.py` | 12 | `SecEdgarClient` (I/O) |
| `src/trading_agent_framework/fundamentals/__init__.py` | 11, 12 | Package docstring (11); exports (12) |
| `.gitignore` | 12 | `+ /cache/` |
| `src/trading_agent_framework/agents/tools/fundamentals.py` | 13 | `fundamentals_tools(strategy)` |
| `CLAUDE.md` | 14 | Architecture bullets + no-look-ahead gotcha |
| `README.md` | 14 | `FRED_API_KEY` / `SEC_EDGAR_USER_AGENT` docs |
| `scripts/tests/smoke_news.py`, `smoke_macro.py`, `smoke_fundamentals.py` | 14 | Manual network smoke scripts |
| `tests/agents/tools/test_trading_tools.py` | 1 | |
| `tests/agents/tools/test_account_tools.py` | 2 | |
| `tests/agents/tools/test_market_data_tools.py` | 3 | |
| `tests/agents/tools/test_indicator_tools.py` | 4 | |
| `tests/agents/tools/test_prebuilt.py` | 5 | |
| `tests/brokers/alpaca/test_market_data_requests.py`, `test_market_data_parse.py` | 6 | additions |
| `tests/brokers/alpaca/test_client.py`, `test_broker_market_data.py` | 7 | additions |
| `tests/fakes.py` | 7 | `+ FakeNewsClient`, `make_alpaca_news_article` |
| `tests/agents/tools/test_news_tools.py` | 8 | |
| `tests/config/test_env.py` | 9 | additions |
| `tests/test_errors.py` | 9, 12 | additions |
| `tests/agents/tools/test_macro_tools.py` | 10 | |
| `tests/fundamentals/test_sec.py` | 11 | |
| `tests/fundamentals/test_edgar_client.py` | 12 | |
| `tests/agents/tools/test_fundamentals_tools.py` | 13 | |

---

### Task 1: `trading_tools(strategy)` — submit, cancel, close orders

**Files:**
- Create: `src/trading_agent_framework/agents/tools/__init__.py` (empty package docstring only; Task 5 adds exports)
- Create: `src/trading_agent_framework/agents/tools/trading.py`
- Test: `tests/agents/tools/test_trading_tools.py`

**Interfaces:**
- Consumes: `Strategy` (`core/strategy.py`): `.create_order`, `.submit_order`, `.get_order`, `.cancel_order`, `.cancel_open_orders`, `.close_position`, `.sell_all`, `.get_orders`. `Order` entity fields. `OrderValidationError`, `BrokerError` from `utils/errors.py`.
- Produces (used by Task 5): `trading.trading_tools(strategy: Strategy) -> list[Callable[..., dict[str, Any]]]` returning, in order: `submit_order`, `cancel_order`, `cancel_open_orders`, `close_position`, `sell_all`, `get_orders`, `get_order`. `trading._lean_order(order: Order) -> dict[str, Any]` (used nowhere outside this module, but its shape — `identifier, symbol, side, order_type, status`, plus `quantity`/`limit_price`/`stop_price`/`filled_quantity`/`avg_fill_price` when not `None`/zero — is the contract Task 5's `test_prebuilt.py` and later tasks assume for "a lean order dict").

- [ ] **Step 1: Create the branch**

```bash
git switch -c feature/agent-tools
```

- [ ] **Step 2: Write the failing test**

Create `tests/agents/tools/test_trading_tools.py`:

```python
from __future__ import annotations

from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.trading import trading_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.utils.errors import BrokerError


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def _tools(strategy: Strategy) -> dict[str, object]:
    return {tool.__name__: tool for tool in trading_tools(strategy)}


def test_returns_seven_tools_with_one_line_docstrings() -> None:
    strategy, _ = _strategy()
    tools = trading_tools(strategy)
    assert [t.__name__ for t in tools] == [
        "submit_order", "cancel_order", "cancel_open_orders",
        "close_position", "sell_all", "get_orders", "get_order",
    ]
    for tool in tools:
        assert tool.__doc__ is not None
        assert len(tool.__doc__.splitlines()) == 1


def test_mutates_trading_flag_on_every_order_affecting_tool() -> None:
    tools = _tools(_strategy()[0])
    for name in ("submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all"):
        assert getattr(tools[name], "mutates_trading") is True
    for name in ("get_orders", "get_order"):
        assert not hasattr(tools[name], "mutates_trading")


def test_submit_order_builds_and_submits_a_market_order() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)

    result = tools["submit_order"]("SPY", 10, "buy")

    assert result["symbol"] == "SPY"
    assert result["side"] == "buy"
    assert result["order_type"] == "market"
    assert result["quantity"] == 10.0
    assert "error" not in result
    [submitted] = broker.submitted
    assert submitted.asset == Asset("SPY")


def test_submit_order_with_a_limit_price_builds_a_limit_order() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)

    result = tools["submit_order"]("SPY", 5, "sell", limit_price=450.5)

    assert (result["order_type"], result["limit_price"]) == ("limit", 450.5)


def test_submit_order_returns_an_error_dict_on_broker_failure() -> None:
    strategy, broker = _strategy()
    broker.market_data_error = None  # submit doesn't use market data
    tools = _tools(strategy)

    def _boom(order: Order) -> Order:
        raise BrokerError("submit failed")

    broker._submit_order = _boom  # type: ignore[method-assign]

    result = tools["submit_order"]("SPY", 1, "buy")

    assert result == {"error": "submit failed"}


def test_cancel_order_unknown_id_returns_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["cancel_order"]("nope") == {"error": "unknown order_id 'nope'"}


def test_cancel_order_known_id_cancels_it() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)
    submitted = tools["submit_order"]("SPY", 1, "buy")

    result = tools["cancel_order"](submitted["identifier"])

    assert result == {"identifier": submitted["identifier"], "status": "cancel_requested"}
    assert len(broker.canceled) == 1


def test_cancel_open_orders_delegates_to_the_strategy() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)
    tools["submit_order"]("SPY", 1, "buy")

    assert tools["cancel_open_orders"]() == {"status": "ok"}
    assert len(broker.canceled) == 1


def test_close_position_without_a_position_reports_no_position() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["close_position"]("SPY") == {"status": "no position"}


def test_sell_all_returns_the_closed_orders() -> None:
    strategy, broker = _strategy()
    broker.close_all_positions = lambda cancel_orders=True: [
        Order(strategy_name="momentum", asset=Asset("SPY"), side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=Decimal(1))
    ]
    tools = _tools(strategy)

    result = tools["sell_all"]()

    assert len(result["orders"]) == 1
    assert result["orders"][0]["symbol"] == "SPY"


def test_get_orders_lists_every_tracked_order() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)
    tools["submit_order"]("SPY", 1, "buy")
    tools["submit_order"]("QQQ", 2, "sell")

    result = tools["get_orders"]()

    assert {o["symbol"] for o in result["orders"]} == {"SPY", "QQQ"}


def test_get_order_unknown_id_returns_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_order"]("nope") == {"error": "unknown order_id 'nope'"}


def test_get_order_known_id_returns_the_lean_order() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)
    submitted = tools["submit_order"]("SPY", 1, "buy")

    result = tools["get_order"](submitted["identifier"])

    assert result["identifier"] == submitted["identifier"]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_trading_tools.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'trading_agent_framework.agents.tools'`.

- [ ] **Step 4: Create the package and `trading.py`**

Create `src/trading_agent_framework/agents/tools/__init__.py`:

```python
"""Agent tools: PrebuiltTools.all() plus individually-wired tool factories."""
```

Create `src/trading_agent_framework/agents/tools/trading.py`:

```python
"""Plain typed trading tools for a LangChain agent: submit, cancel, and close orders.

Each docstring is a single line on purpose: it becomes the tool description sent to the model
on every call. Broker/order-validation failures come back as `{"error": ...}` so the model can
correct itself, mirroring `memory/tools.py`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def _lean_order(order: Order) -> dict[str, Any]:
    lean: dict[str, Any] = {
        "identifier": order.identifier,
        "symbol": order.asset.symbol,
        "side": order.side.value,
        "order_type": order.order_type.value,
        "status": order.status.value,
    }
    if order.quantity is not None:
        lean["quantity"] = float(order.quantity)
    if order.limit_price is not None:
        lean["limit_price"] = float(order.limit_price)
    if order.stop_price is not None:
        lean["stop_price"] = float(order.stop_price)
    if order.filled_quantity:
        lean["filled_quantity"] = float(order.filled_quantity)
    if order.avg_fill_price is not None:
        lean["avg_fill_price"] = float(order.avg_fill_price)
    return lean


def trading_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:
    """Order tools bound to `strategy`."""

    def submit_order(
        symbol: str,
        quantity: float,
        side: str,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit a market, limit, stop or stop-limit order; the type follows from the prices given."""
        try:
            order = strategy.create_order(
                symbol, quantity, side,
                limit_price=limit_price, stop_price=stop_price, time_in_force=time_in_force,
            )
            submitted = strategy.submit_order(order)
        except (OrderValidationError, BrokerError) as exc:
            return {"error": str(exc)}
        return _lean_order(submitted)

    def cancel_order(order_id: str) -> dict[str, Any]:
        """Cancel a tracked order by its identifier."""
        order = strategy.get_order(order_id)
        if order is None:
            return {"error": f"unknown order_id {order_id!r}"}
        try:
            strategy.cancel_order(order)
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"identifier": order.identifier, "status": "cancel_requested"}

    def cancel_open_orders() -> dict[str, Any]:
        """Cancel every open order."""
        try:
            strategy.cancel_open_orders()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"status": "ok"}

    def close_position(symbol: str, fraction: float = 1.0) -> dict[str, Any]:
        """Close (or partially close) the position in a symbol."""
        try:
            order = strategy.close_position(symbol, fraction)
        except BrokerError as exc:
            return {"error": str(exc)}
        return _lean_order(order) if order is not None else {"status": "no position"}

    def sell_all() -> dict[str, Any]:
        """Close every open position."""
        try:
            orders = strategy.sell_all()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"orders": [_lean_order(order) for order in orders]}

    def get_orders() -> dict[str, Any]:
        """List every tracked order."""
        return {"orders": [_lean_order(order) for order in strategy.get_orders()]}

    def get_order(order_id: str) -> dict[str, Any]:
        """Look up one order by its identifier."""
        order = strategy.get_order(order_id)
        return _lean_order(order) if order is not None else {"error": f"unknown order_id {order_id!r}"}

    for tool in (submit_order, cancel_order, cancel_open_orders, close_position, sell_all):
        vars(tool)["mutates_trading"] = True

    return [submit_order, cancel_order, cancel_open_orders, close_position, sell_all, get_orders, get_order]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_trading_tools.py -q`
Expected: all pass.

- [ ] **Step 6: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/trading_agent_framework/agents/tools/__init__.py src/trading_agent_framework/agents/tools/trading.py tests/agents/tools/test_trading_tools.py
git commit -m "Task 1: add trading_tools (submit, cancel, close orders)"
```

(Add the attribution trailer lines to the message.)

---

### Task 2: `account_tools(strategy)` — balances and positions

**Files:**
- Create: `src/trading_agent_framework/agents/tools/account.py`
- Test: `tests/agents/tools/test_account_tools.py`

**Interfaces:**
- Consumes: `Strategy.broker.get_account()` (`AccountBalances`), `Strategy.get_positions()`, `Strategy.get_position()` (`Position`), `BrokerError`.
- Produces (used by Task 5): `account.account_tools(strategy) -> list[Callable]` returning, in order: `get_account_balance`, `get_positions`, `get_position`. `account._lean_position(position: Position) -> dict[str, Any]`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_account_tools.py`:

```python
from __future__ import annotations

from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def _tools(strategy: Strategy) -> dict[str, object]:
    return {tool.__name__: tool for tool in account_tools(strategy)}


def test_returns_three_tools_with_one_line_docstrings() -> None:
    tools = account_tools(_strategy()[0])
    assert [t.__name__ for t in tools] == ["get_account_balance", "get_positions", "get_position"]
    for tool in tools:
        assert len(tool.__doc__.splitlines()) == 1


def test_get_account_balance_reports_cash_portfolio_value_and_buying_power() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    result = tools["get_account_balance"]()

    assert result == {"cash": 10000.0, "portfolio_value": 25000.0, "buying_power": 20000.0}


def test_get_account_balance_returns_error_on_broker_failure() -> None:
    strategy, broker = _strategy()

    def _boom() -> None:
        raise BrokerError("account unavailable")

    broker.get_account = _boom  # type: ignore[method-assign]
    tools = _tools(strategy)

    assert tools["get_account_balance"]() == {"error": "account unavailable"}


def test_get_positions_lists_every_position() -> None:
    strategy, broker = _strategy()
    broker.positions = [
        Position(strategy_name="momentum", asset=Asset("SPY"), quantity=Decimal(10), side=PositionSide.LONG,
                 current_price=Decimal("450.5"), market_value=Decimal("4505"))
    ]
    tools = _tools(strategy)

    result = tools["get_positions"]()

    assert result["positions"] == [
        {"symbol": "SPY", "quantity": 10.0, "side": "long", "current_price": 450.5, "market_value": 4505.0}
    ]


def test_get_position_without_a_position_returns_none() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_position"]("SPY") == {"position": None}


def test_get_position_with_a_position_returns_the_lean_dict() -> None:
    strategy, broker = _strategy()
    broker.positions = [
        Position(strategy_name="momentum", asset=Asset("SPY"), quantity=Decimal(5), side=PositionSide.LONG)
    ]
    tools = _tools(strategy)

    result = tools["get_position"]("SPY")

    assert result == {"symbol": "SPY", "quantity": 5.0, "side": "long"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_account_tools.py -q`
Expected: `ModuleNotFoundError: No module named 'trading_agent_framework.agents.tools.account'`.

- [ ] **Step 3: Create `account.py`**

Create `src/trading_agent_framework/agents/tools/account.py`:

```python
"""Plain typed account tools for a LangChain agent: balances and positions."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def _lean_position(position: Position) -> dict[str, Any]:
    lean: dict[str, Any] = {
        "symbol": position.asset.symbol,
        "quantity": float(position.quantity),
        "side": position.side.value,
    }
    if position.avg_fill_price is not None:
        lean["avg_fill_price"] = float(position.avg_fill_price)
    if position.current_price is not None:
        lean["current_price"] = float(position.current_price)
    if position.market_value is not None:
        lean["market_value"] = float(position.market_value)
    if position.unrealized_pnl is not None:
        lean["unrealized_pnl"] = float(position.unrealized_pnl)
    return lean


def account_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:
    """Account tools bound to `strategy`."""

    def get_account_balance() -> dict[str, Any]:
        """Get cash, portfolio value and buying power."""
        try:
            account = strategy.broker.get_account()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
        }

    def get_positions() -> dict[str, Any]:
        """List every open position."""
        try:
            positions = strategy.get_positions()
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"positions": [_lean_position(p) for p in positions]}

    def get_position(symbol: str) -> dict[str, Any]:
        """Look up the open position in one symbol."""
        try:
            position = strategy.get_position(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        return _lean_position(position) if position is not None else {"position": None}

    return [get_account_balance, get_positions, get_position]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_account_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/account.py tests/agents/tools/test_account_tools.py
git commit -m "Task 2: add account_tools (balances and positions)"
```

(Add the attribution trailer lines to the message.)

---

### Task 3: `market_data_tools(strategy)` — last price, quote, bars

**Files:**
- Create: `src/trading_agent_framework/agents/tools/market_data.py`
- Test: `tests/agents/tools/test_market_data_tools.py`

**Interfaces:**
- Consumes: `Strategy.get_last_price`, `Strategy.get_quote` (`Quote`), `Strategy.get_historical_prices` (`Bars`), `BrokerError`.
- Produces (used by Task 5): `market_data.market_data_tools(strategy) -> list[Callable]` returning, in order: `get_last_price`, `get_quote`, `get_bars`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_market_data_tools.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.entities.asset import Asset


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def _tools(strategy: Strategy) -> dict[str, object]:
    return {tool.__name__: tool for tool in market_data_tools(strategy)}


def test_returns_three_tools_with_one_line_docstrings() -> None:
    tools = market_data_tools(_strategy()[0])
    assert [t.__name__ for t in tools] == ["get_last_price", "get_quote", "get_bars"]
    for tool in tools:
        assert len(tool.__doc__.splitlines()) == 1


def test_get_last_price_returns_the_price() -> None:
    strategy, broker = _strategy()
    broker.last_prices["SPY"] = Decimal("450.10")
    tools = _tools(strategy)

    assert tools["get_last_price"]("SPY") == {"symbol": "SPY", "price": 450.10}


def test_get_last_price_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_last_price"]("SPY") == {"error": "no trade data for 'SPY'"}


def test_get_quote_returns_bid_ask_and_mid() -> None:
    strategy, broker = _strategy()
    broker.quotes["SPY"] = Quote(
        asset=Asset("SPY"), bid=Decimal("450"), ask=Decimal("451"),
        bid_size=None, ask_size=None, timestamp=datetime(2026, 9, 14, 14, tzinfo=UTC),
    )
    tools = _tools(strategy)

    result = tools["get_quote"]("SPY")

    assert (result["bid"], result["ask"], result["mid"]) == (450.0, 451.0, 450.5)
    assert result["timestamp"] == "2026-09-14T14:00:00+00:00"


def test_get_quote_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_quote"]("SPY") == {"error": "no quote for 'SPY'"}


def test_get_bars_returns_oldest_first_ohlcv_rows() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["SPY"] = make_bars_frame([100.0, 101.0, 102.0], start=et(2026, 9, 10))
    tools = _tools(strategy)

    result = tools["get_bars"]("SPY", length=3)

    assert result["symbol"] == "SPY"
    assert result["timestep"] == "day"
    assert [row["close"] for row in result["bars"]] == [100.0, 101.0, 102.0]
    assert set(result["bars"][0]) == {"date", "open", "high", "low", "close", "volume"}


def test_get_bars_clamps_length_to_the_allowed_range() -> None:
    strategy, broker = _strategy()
    broker.bar_frames["SPY"] = make_bars_frame([100.0] * 250, start=et(2026, 9, 10))
    tools = _tools(strategy)

    tools["get_bars"]("SPY", length=1000)

    [(_, length, _, _)] = broker.bars_calls
    assert length == 200


def test_get_bars_without_data_returns_an_error() -> None:
    strategy, _ = _strategy()
    tools = _tools(strategy)

    assert tools["get_bars"]("SPY") == {"error": "no bars for 'SPY'"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_market_data_tools.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `market_data.py`**

Create `src/trading_agent_framework/agents/tools/market_data.py`:

```python
"""Plain typed market-data tools for a LangChain agent: last price, quote, bars."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

_MIN_BARS = 1
_MAX_BARS = 200


def market_data_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:
    """Market-data tools bound to `strategy`."""

    def get_last_price(symbol: str) -> dict[str, Any]:
        """Get the last traded price for a symbol."""
        try:
            price = strategy.get_last_price(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        if price is None:
            return {"error": f"no trade data for {symbol!r}"}
        return {"symbol": symbol.upper(), "price": float(price)}

    def get_quote(symbol: str) -> dict[str, Any]:
        """Get the latest bid/ask quote for a symbol."""
        try:
            quote = strategy.get_quote(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        if quote is None:
            return {"error": f"no quote for {symbol!r}"}
        return {
            "symbol": symbol.upper(),
            "bid": float(quote.bid) if quote.bid is not None else None,
            "ask": float(quote.ask) if quote.ask is not None else None,
            "mid": float(quote.mid) if quote.mid is not None else None,
            "timestamp": quote.timestamp.isoformat(),
        }

    def get_bars(symbol: str, length: int = 30, timestep: str = "day") -> dict[str, Any]:
        """Get recent OHLCV bars for a symbol, oldest first."""
        clamped_length = min(max(int(length), _MIN_BARS), _MAX_BARS)
        try:
            bars = strategy.get_historical_prices(symbol, clamped_length, timestep)
        except BrokerError as exc:
            return {"error": str(exc)}
        if bars is None:
            return {"error": f"no bars for {symbol!r}"}
        rows = [
            {
                "date": index.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for index, row in bars.df.iterrows()
        ]
        return {"symbol": symbol.upper(), "timestep": timestep, "bars": rows}

    return [get_last_price, get_quote, get_bars]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_market_data_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/market_data.py tests/agents/tools/test_market_data_tools.py
git commit -m "Task 3: add market_data_tools (last price, quote, bars)"
```

(Add the attribution trailer lines to the message.)

---

### Task 4: `indicator_tools(strategy)` — generic pandas-ta-classic dispatch

**Files:**
- Create: `src/trading_agent_framework/agents/tools/indicators.py`
- Test: `tests/agents/tools/test_indicator_tools.py`

**Interfaces:**
- Consumes: `Strategy.indicators` (`core/indicators.py`'s `Indicators.__getattr__`, `IndicatorRow`).
- Produces (used by Task 5): `indicators.indicator_tools(strategy) -> list[Callable]` returning `[get_indicator]`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_indicator_tools.py`:

```python
from __future__ import annotations

from tests.fakes import FakeBroker, FakeClock, et, make_bars_frame

from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.core.strategy import Strategy


def _strategy_with_bars(symbol: str = "SPY") -> Strategy:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    broker.bar_frames[symbol] = make_bars_frame([100.0 + i for i in range(60)], start=et(2026, 7, 1))
    return Strategy(broker)


def _tool(strategy: Strategy):
    [tool] = indicator_tools(strategy)
    return tool


def test_returns_one_tool_with_a_one_line_docstring() -> None:
    tools = indicator_tools(_strategy_with_bars())
    assert [t.__name__ for t in tools] == ["get_indicator"]
    assert len(tools[0].__doc__.splitlines()) == 1


def test_get_indicator_computes_a_scalar_indicator() -> None:
    tool = _tool(_strategy_with_bars())

    result = tool("sma", "SPY", params={"length": 20})

    assert result["indicator"] == "sma"
    assert result["symbol"] == "SPY"
    assert isinstance(result["value"], float)


def test_get_indicator_computes_a_multi_column_indicator() -> None:
    tool = _tool(_strategy_with_bars())

    result = tool("bbands", "SPY", params={"length": 20, "std": 2})

    assert isinstance(result["value"], dict)
    assert result["value"]  # at least one BB* column


def test_get_indicator_without_params_uses_the_indicator_defaults() -> None:
    tool = _tool(_strategy_with_bars())

    result = tool("rsi", "SPY")

    assert result["indicator"] == "rsi"


def test_get_indicator_unknown_name_returns_an_error() -> None:
    tool = _tool(_strategy_with_bars())

    result = tool("not_a_real_indicator", "SPY")

    assert "error" in result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_indicator_tools.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `indicators.py`**

Create `src/trading_agent_framework/agents/tools/indicators.py`:

```python
"""Plain typed indicator tool for a LangChain agent: any pandas-ta-classic indicator by name."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.core.indicators import IndicatorRow

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


def indicator_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:
    """Indicator tool bound to `strategy`."""

    def get_indicator(
        name: str, symbol: str, timestep: str = "day", params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute a pandas-ta-classic indicator (e.g. sma, rsi, bbands) for a symbol."""
        try:
            indicator = getattr(strategy.indicators, name)
            value = indicator(symbol, timestep, **(params or {}))
        except AttributeError as exc:
            return {"error": str(exc)}
        result = value.as_dict() if isinstance(value, IndicatorRow) else value
        return {"indicator": name, "symbol": symbol.upper(), "value": result}

    return [get_indicator]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_indicator_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/indicators.py tests/agents/tools/test_indicator_tools.py
git commit -m "Task 4: add indicator_tools (generic pandas-ta-classic dispatch)"
```

(Add the attribution trailer lines to the message.)

---

### Task 5: `PrebuiltTools.all(strategy)`

**Files:**
- Create: `src/trading_agent_framework/agents/tools/prebuilt.py`
- Modify: `src/trading_agent_framework/agents/tools/__init__.py`
- Test: `tests/agents/tools/test_prebuilt.py`

**Interfaces:**
- Consumes: Tasks 1-4's `trading_tools`, `account_tools`, `market_data_tools`, `indicator_tools`; `memory.tools.memory_tools`.
- Produces (used by Task 14, and by any strategy): `prebuilt.PrebuiltTools.all(strategy) -> list[Callable[..., Any]]`. `agents/tools/__init__.py` exports `PrebuiltTools`, `trading_tools`, `account_tools`, `market_data_tools`, `indicator_tools` (Task 14 adds `news_tools`, `macro_tools`, `fundamentals_tools`).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_prebuilt.py`:

```python
from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools import PrebuiltTools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.memory.tools import memory_tools


def _strategy(tmp_path: Path) -> Strategy:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker, project_root=tmp_path)


def test_all_combines_every_domain_in_order(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    tools = PrebuiltTools.all(strategy)
    names = [tool.__name__ for tool in tools]

    memory_names = [tool.__name__ for tool in memory_tools(strategy.memory)]
    expected = memory_names + [
        "submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all",
        "get_orders", "get_order",
        "get_account_balance", "get_positions", "get_position",
        "get_last_price", "get_quote", "get_bars",
        "get_indicator",
    ]
    assert names == expected


def test_all_tool_names_are_unique(tmp_path: Path) -> None:
    names = [tool.__name__ for tool in PrebuiltTools.all(_strategy(tmp_path))]
    assert len(names) == len(set(names))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_prebuilt.py -q`
Expected: `ImportError: cannot import name 'PrebuiltTools'`.

- [ ] **Step 3: Create `prebuilt.py` and update `__init__.py`**

Create `src/trading_agent_framework/agents/tools/prebuilt.py`:

```python
"""`PrebuiltTools.all(strategy)`: trading, account, market-data, indicator and memory tools in one call."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.trading import trading_tools
from trading_agent_framework.memory.tools import memory_tools

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy


class PrebuiltTools:
    """Everything a trading agent needs out of the box, bundled in one call."""

    @staticmethod
    def all(strategy: "Strategy") -> list[Callable[..., Any]]:
        """Trading, account, market-data, indicator and memory tools for `strategy`."""
        return [
            *memory_tools(strategy.memory),
            *trading_tools(strategy),
            *account_tools(strategy),
            *market_data_tools(strategy),
            *indicator_tools(strategy),
        ]
```

Replace `src/trading_agent_framework/agents/tools/__init__.py` with:

```python
"""Agent tools: PrebuiltTools.all() plus individually-wired tool factories."""

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.prebuilt import PrebuiltTools
from trading_agent_framework.agents.tools.trading import trading_tools

__all__ = [
    "PrebuiltTools",
    "account_tools",
    "indicator_tools",
    "market_data_tools",
    "trading_tools",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_prebuilt.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/prebuilt.py src/trading_agent_framework/agents/tools/__init__.py tests/agents/tools/test_prebuilt.py
git commit -m "Task 5: add PrebuiltTools.all()"
```

(Add the attribution trailer lines to the message.)

---

### Task 6: Alpaca news — pure request/response translation

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/market_data.py`
- Test: `tests/brokers/alpaca/test_market_data_requests.py`, `tests/brokers/alpaca/test_market_data_parse.py` (append to both)

**Interfaces:**
- Consumes: `alpaca.data.requests.NewsRequest`, `alpaca.data.models.news.NewsSet`.
- Produces (used by Task 7): `market_data.AlpacaNewsClient` (Protocol), `market_data.MAX_NEWS_LIMIT = 50`, `market_data.build_news_request(symbols: Sequence[str], *, start: datetime | None, end: datetime, limit: int, include_content: bool) -> NewsRequest`, `market_data.parse_news(news_set: object) -> list[dict[str, object]]` (each item: `id, headline, summary, source, created_at, symbols`, plus `content` only when non-empty).

- [ ] **Step 1: Write the failing tests**

Append to `tests/brokers/alpaca/test_market_data_requests.py` (add `NewsRequest`-adjacent imports; keep the existing import block, add):

```python
from trading_agent_framework.brokers.alpaca.market_data import (
    MAX_NEWS_LIMIT,
    build_news_request,
)
```

Append at the end of the file:

```python
def test_build_news_request_joins_symbols_and_clamps_limit() -> None:
    start = et(2026, 9, 8)
    end = et(2026, 9, 10)

    request = build_news_request(["SPY", "QQQ"], start=start, end=end, limit=999, include_content=True)

    assert request.symbols == "SPY,QQQ"
    assert request.limit == MAX_NEWS_LIMIT
    assert request.include_content is True
    assert request.start == start.astimezone(UTC).replace(tzinfo=None)
    assert request.end == end.astimezone(UTC).replace(tzinfo=None)


def test_build_news_request_with_no_symbols_omits_them() -> None:
    request = build_news_request([], start=None, end=_END, limit=5, include_content=False)

    assert request.symbols is None
    assert request.start is None
```

Append to `tests/brokers/alpaca/test_market_data_parse.py`'s import block:

```python
from alpaca.data.models.news import NewsSet

from trading_agent_framework.brokers.alpaca.market_data import parse_news
```

Append at the end of the file:

```python
def test_parse_news_extracts_lean_articles() -> None:
    news_set = NewsSet(
        {
            "news": [
                {
                    "id": 123,
                    "headline": "Fed cuts rates",
                    "source": "benzinga",
                    "url": "https://example.com/a",
                    "summary": "Rates fall.",
                    "created_at": "2026-09-10T13:30:00Z",
                    "updated_at": "2026-09-10T13:30:00Z",
                    "symbols": ["SPY", "QQQ"],
                    "author": "Jane Doe",
                    "content": "",
                }
            ],
            "next_page_token": None,
        }
    )

    articles = parse_news(news_set)

    assert articles == [
        {
            "id": 123,
            "headline": "Fed cuts rates",
            "summary": "Rates fall.",
            "source": "benzinga",
            "created_at": "2026-09-10T13:30:00+00:00",
            "symbols": ["SPY", "QQQ"],
        }
    ]


def test_parse_news_includes_content_only_when_present() -> None:
    news_set = NewsSet(
        {
            "news": [
                {
                    "id": 1, "headline": "h", "source": "s", "url": None, "summary": "sum",
                    "created_at": "2026-09-10T00:00:00Z", "updated_at": "2026-09-10T00:00:00Z",
                    "symbols": [], "author": "a", "content": "full article text",
                }
            ],
            "next_page_token": None,
        }
    )

    [article] = parse_news(news_set)

    assert article["content"] == "full article text"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_requests.py tests/brokers/alpaca/test_market_data_parse.py -q`
Expected: `ImportError: cannot import name 'build_news_request'` / `'parse_news'`.

- [ ] **Step 3: Extend `market_data.py`**

In `src/trading_agent_framework/brokers/alpaca/market_data.py`, change the `alpaca.data.requests` import to add `NewsRequest`:

```python
from alpaca.data.requests import (
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
```

Add to the `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.data.models import Quote as AlpacaQuote
    from alpaca.data.models import Trade as AlpacaTrade
    from alpaca.data.models.news import NewsSet
```

Add near `MAX_SYMBOLS_PER_REQUEST`:

```python
MAX_NEWS_LIMIT = 50
```

Add after the `AlpacaStockDataClient` protocol:

```python
class AlpacaNewsClient(Protocol):
    """What `AlpacaBroker` calls on a `NewsClient` (or a test fake)."""

    def get_news(self, request_params: NewsRequest) -> NewsSet: ...
```

Add near `build_latest_quote_request`:

```python
def build_news_request(
    symbols: Sequence[str],
    *,
    start: datetime | None,
    end: datetime,
    limit: int,
    include_content: bool,
) -> NewsRequest:
    return NewsRequest(
        symbols=",".join(symbols) if symbols else None,
        start=start,
        end=end,
        limit=min(max(int(limit), 1), MAX_NEWS_LIMIT),
        include_content=include_content,
    )
```

Add at the end of the file (response parsing section):

```python
def parse_news(news_set: object) -> list[dict[str, object]]:
    """Lean articles: id, headline, summary, source, created_at, symbols, and content when present."""
    articles = cast(Mapping[str, Sequence[object]], _field(news_set, "data") or {}).get("news", [])
    parsed: list[dict[str, object]] = []
    for article in articles:
        item: dict[str, object] = {
            "id": _field(article, "id"),
            "headline": _field(article, "headline"),
            "summary": _field(article, "summary"),
            "source": _field(article, "source"),
            "created_at": cast(datetime, _field(article, "created_at")).isoformat(),
            "symbols": list(cast(Sequence[str], _field(article, "symbols") or [])),
        }
        content = _field(article, "content")
        if content:
            item["content"] = content
        parsed.append(item)
    return parsed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/alpaca/test_market_data_requests.py tests/brokers/alpaca/test_market_data_parse.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/brokers/alpaca/market_data.py tests/brokers/alpaca/test_market_data_requests.py tests/brokers/alpaca/test_market_data_parse.py
git commit -m "Task 6: add pure Alpaca news request/response translation"
```

(Add the attribution trailer lines to the message.)

---

### Task 7: Alpaca news — client factory and `AlpacaBroker.get_news`

**Files:**
- Modify: `src/trading_agent_framework/brokers/alpaca/client.py`
- Modify: `src/trading_agent_framework/brokers/alpaca/broker.py`
- Modify: `tests/fakes.py`
- Test: `tests/brokers/alpaca/test_client.py`, `tests/brokers/alpaca/test_broker_market_data.py` (append to both)

**Interfaces:**
- Consumes: Task 6's `market_data.build_news_request`, `market_data.parse_news`, `market_data.AlpacaNewsClient`.
- Produces (used by Task 8): `client.build_news_client(creds: AlpacaCredentials) -> NewsClient`. `AlpacaBroker.get_news(symbols: Sequence[str] = (), *, start: datetime | None = None, end: datetime, limit: int = 10, include_content: bool = False) -> list[dict[str, object]]`, raising `BrokerError` on failure or if no news client is configured. `AlpacaBroker.__init__`'s new keyword-only `news_client: market_data.AlpacaNewsClient | None = None`. `tests.fakes.FakeNewsClient` and `tests.fakes.make_alpaca_news_article(**overrides) -> dict[str, object]`.

Note: `start` defaults to `None` (unlike `end`, which is required) so `broker.get_news(end=...)` alone is valid — the "no client configured" and "wraps client failures" tests below call it that way.

- [ ] **Step 1: Add `FakeNewsClient` to `tests/fakes.py`**

Add `NewsRequest` to the `alpaca.data.requests` import block near the top of `tests/fakes.py`:

```python
from alpaca.data.requests import (
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
```

Append after `FakeStockHistoricalDataClient` (before the `ET = ZoneInfo(...)` line):

```python
def make_alpaca_news_article(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": 123,
        "headline": "Fed cuts rates",
        "source": "benzinga",
        "url": "https://example.com/a",
        "summary": "Rates fall.",
        "created_at": "2026-09-10T13:30:00Z",
        "updated_at": "2026-09-10T13:30:00Z",
        "symbols": ["SPY"],
        "author": "Jane Doe",
        "content": "",
    }
    return payload | overrides


class FakeNewsClient:
    """A hand-written stand-in for `alpaca.data.historical.news.NewsClient`."""

    def __init__(self) -> None:
        self.articles: list[dict[str, object]] = []
        self.news_requests: list[NewsRequest] = []
        self.raises: BaseException | None = None

    def get_news(self, request_params: NewsRequest) -> alpaca_data_models.news.NewsSet:
        self.news_requests.append(request_params)
        if self.raises is not None:
            raise self.raises
        return alpaca_data_models.news.NewsSet({"news": self.articles, "next_page_token": None})
```

`alpaca_data_models.news` needs its submodule imported explicitly (the top-level `import alpaca.data.models as alpaca_data_models` doesn't pull in the `news` submodule automatically); add near the top import block:

```python
import alpaca.data.models.news as _alpaca_news_models  # noqa: F401  -- registers alpaca_data_models.news
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/brokers/alpaca/test_client.py`:

```python
def test_build_news_client_uses_the_same_credentials(monkeypatch) -> None:
    from trading_agent_framework.brokers.alpaca.client import build_news_client

    creds = AlpacaCredentials(api_key="key", api_secret="secret", is_paper=True)
    mock_news_client = MagicMock()
    monkeypatch.setattr("alpaca.data.historical.news.NewsClient", mock_news_client)

    result = build_news_client(creds)

    mock_news_client.assert_called_once_with(api_key="key", secret_key="secret")
    assert result is mock_news_client.return_value
```

Append to `tests/brokers/alpaca/test_broker_market_data.py` (add `FakeNewsClient`, `make_alpaca_news_article` to the `tests.fakes` import):

```python
def _broker_with_news(news: FakeNewsClient) -> AlpacaBroker:
    return AlpacaBroker(
        "momentum", FakeTradingClient(), clock=FakeClock(_NOW), news_client=news
    )


def test_get_news_builds_a_request_and_parses_the_response() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    broker = _broker_with_news(news)

    articles = broker.get_news(["SPY"], start=None, end=_NOW, limit=5, include_content=False)

    assert articles[0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY"


def test_get_news_without_a_configured_client_raises() -> None:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_NOW))

    with pytest.raises(BrokerError, match="no news client configured"):
        broker.get_news(end=_NOW)


def test_get_news_wraps_client_failures_as_broker_error() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("rate limited")
    broker = _broker_with_news(news)

    with pytest.raises(BrokerError, match="Failed to fetch news"):
        broker.get_news(end=_NOW)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/brokers/alpaca/test_client.py tests/brokers/alpaca/test_broker_market_data.py -q`
Expected: `ImportError: cannot import name 'build_news_client'` and `TypeError: AlpacaBroker.get_news`-related failures.

- [ ] **Step 4: Extend `client.py`**

Add to `src/trading_agent_framework/brokers/alpaca/client.py`'s `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream
```

Append:

```python
def build_news_client(creds: AlpacaCredentials) -> NewsClient:
    """News uses the trading credentials, same as market data."""
    from alpaca.data.historical.news import NewsClient

    return NewsClient(api_key=creds.api_key, secret_key=creds.api_secret)
```

- [ ] **Step 5: Extend `broker.py`**

In `src/trading_agent_framework/brokers/alpaca/broker.py`, add `build_news_client` to the `client` import:

```python
from trading_agent_framework.brokers.alpaca.client import (
    build_news_client,
    build_stock_data_client,
    build_trading_client,
    build_trading_stream,
)
```

Change `__init__`'s signature and body:

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
        news_client: market_data.AlpacaNewsClient | None = None,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(client),
            is_paper=is_paper,
        )
        self._client = client
        self._data_client = data_client
        self._news_client = news_client
        self._stream = stream
        self._alpaca_stream: AlpacaTradeStream | None = None
```

Change `from_credentials`:

```python
    @classmethod
    def from_credentials(
        cls,
        strategy_name: str,
        creds: AlpacaCredentials,
        with_stream: bool = True,
    ) -> AlpacaBroker:
        client = cast("orders.AlpacaTradingClient", build_trading_client(creds))
        data_client = cast("market_data.AlpacaStockDataClient", build_stock_data_client(creds))
        news_client = cast("market_data.AlpacaNewsClient", build_news_client(creds))
        stream = build_trading_stream(creds) if with_stream else None
        return cls(
            strategy_name, client, stream=stream, is_paper=creds.is_paper,
            data_client=data_client, news_client=news_client,
        )
```

Add near `_require_data_client` (in the "market data" section):

```python
    def _require_news_client(self) -> market_data.AlpacaNewsClient:
        if self._news_client is None:
            raise BrokerError(
                "no news client configured; construct the broker with news_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return self._news_client

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]:
        client = self._require_news_client()
        request = market_data.build_news_request(
            symbols, start=start, end=end, limit=limit, include_content=include_content
        )
        try:
            response = client.get_news(request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch news: {exc}") from exc
        return market_data.parse_news(response)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/brokers/alpaca/test_client.py tests/brokers/alpaca/test_broker_market_data.py -q`
Expected: all pass.

- [ ] **Step 7: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/trading_agent_framework/brokers/alpaca/client.py src/trading_agent_framework/brokers/alpaca/broker.py tests/fakes.py tests/brokers/alpaca/test_client.py tests/brokers/alpaca/test_broker_market_data.py
git commit -m "Task 7: add build_news_client and AlpacaBroker.get_news"
```

(Add the attribution trailer lines to the message.)

---

### Task 8: `news_tools(strategy)`

**Files:**
- Create: `src/trading_agent_framework/agents/tools/news.py`
- Test: `tests/agents/tools/test_news_tools.py`

**Interfaces:**
- Consumes: `AlpacaBroker.get_news` (Task 7), `Strategy.clock.now()`, `BrokerError`.
- Produces (used by Task 14): `news.news_tools(strategy) -> list[Callable]` returning `[search_news]`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_news_tools.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from tests.fakes import FakeClock, FakeNewsClient, FakeTradingClient, et, make_alpaca_news_article

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import BrokerError


def _now() -> datetime:
    return et(2026, 9, 14, 10)


def _alpaca_strategy(news: FakeNewsClient) -> Strategy:
    broker = AlpacaBroker("momentum", FakeTradingClient(), clock=FakeClock(_now()), news_client=news)
    return Strategy(broker)


def _tool(strategy: Strategy):
    [tool] = news_tools(strategy)
    return tool


def test_search_news_returns_articles_from_the_broker() -> None:
    news = FakeNewsClient()
    news.articles = [make_alpaca_news_article(headline="Rates cut")]
    tool = _tool(_alpaca_strategy(news))

    result = tool(symbols="SPY,QQQ")

    assert result["count"] == 1
    assert result["articles"][0]["headline"] == "Rates cut"
    [request] = news.news_requests
    assert request.symbols == "SPY,QQQ"


def test_search_news_defaults_end_to_the_strategy_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool()

    [request] = news.news_requests
    # alpaca-py normalises start/end to naive UTC (same as test_market_data_requests.py's
    # test_bars_request_asks_for_adjusted_iex_bars_of_every_symbol)
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_clamps_end_that_is_after_the_clock() -> None:
    news = FakeNewsClient()
    tool = _tool(_alpaca_strategy(news))

    tool(end=et(2026, 9, 20).isoformat())

    [request] = news.news_requests
    assert request.end == _now().astimezone(UTC).replace(tzinfo=None)


def test_search_news_returns_an_error_dict_on_broker_failure() -> None:
    news = FakeNewsClient()
    news.raises = RuntimeError("boom")
    tool = _tool(_alpaca_strategy(news))

    result = tool()

    assert "error" in result


def test_search_news_without_an_alpaca_broker_returns_an_error() -> None:
    class _OtherBroker(Broker):
        name = "other"
        def _conform_order(self, order): return order
        def _submit_order(self, order): return order
        def cancel_order(self, order): pass
        def pull_order(self, identifier): return None
        def pull_orders(self, limit=100): return []
        def pull_positions(self): return []
        def get_account(self): raise NotImplementedError
        def modify_order(self, order, *, limit_price=None, stop_price=None): raise NotImplementedError
        def close_position(self, asset, fraction=1): return None
        def close_all_positions(self, cancel_orders=True): return []
        def sync_open_orders(self): return []

    strategy = Strategy(_OtherBroker("momentum", clock=FakeClock(_now())))
    tool = _tool(strategy)

    assert tool() == {"error": "news requires an Alpaca broker"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_news_tools.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `news.py`**

Create `src/trading_agent_framework/agents/tools/news.py`:

```python
"""Plain typed news tool for a LangChain agent: Alpaca news search, gated on the strategy clock."""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 50
_DEFAULT_LOOKBACK_DAYS = 7


def news_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:
    """Alpaca news tool bound to `strategy`."""

    def search_news(
        symbols: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Search recent news headlines and summaries, optionally filtered to symbols."""
        if not isinstance(strategy.broker, AlpacaBroker):
            return {"error": "news requires an Alpaca broker"}
        now = strategy.clock.now()
        end_dt = min(datetime.fromisoformat(end), now) if end else now
        start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        clamped_limit = min(max(int(limit), MIN_LIMIT), MAX_LIMIT)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        try:
            articles = strategy.broker.get_news(
                symbol_list, start=start_dt, end=end_dt, limit=clamped_limit, include_content=include_content
            )
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"count": len(articles), "articles": articles}

    return [search_news]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_news_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/news.py tests/agents/tools/test_news_tools.py
git commit -m "Task 8: add news_tools (Alpaca news search)"
```

(Add the attribution trailer lines to the message.)

---

### Task 9: FRED configuration — `FredCredentials`, `MacroDataError`, dependency, env doc

**Files:**
- Modify: `src/trading_agent_framework/config/env.py`
- Modify: `src/trading_agent_framework/utils/errors.py`
- Modify: `pyproject.toml`
- Modify: `env/.env.example`
- Test: `tests/config/test_env.py`, `tests/test_errors.py` (append to both)

**Interfaces:**
- Produces (used by Task 10): `config.env.FredCredentials(api_key: str)`, `FredCredentials.from_env(env=None) -> FredCredentials` (raises `ConfigurationError` on missing/blank `FRED_API_KEY`). `utils.errors.MacroDataError(TradingFrameworkError)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/config/test_env.py`'s imports:

```python
from trading_agent_framework.config.env import FredCredentials
```

Append at the end of the file:

```python
def test_fred_credentials_from_env() -> None:
    creds = FredCredentials.from_env({"FRED_API_KEY": "abc123"})
    assert creds.api_key == "abc123"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_fred_credentials_missing_or_blank_key_raises(value) -> None:
    env = {} if value is None else {"FRED_API_KEY": value}
    with pytest.raises(ConfigurationError, match="FRED_API_KEY"):
        FredCredentials.from_env(env)


def test_fred_credentials_repr_does_not_leak_the_key() -> None:
    assert "supersecret" not in repr(FredCredentials(api_key="supersecret"))
```

Append to `tests/test_errors.py`'s import and body:

```python
from trading_agent_framework.utils.errors import MacroDataError


def test_macro_data_error_is_a_trading_framework_error() -> None:
    assert issubclass(MacroDataError, TradingFrameworkError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/config/test_env.py tests/test_errors.py -q`
Expected: `ImportError: cannot import name 'FredCredentials'` / `'MacroDataError'`.

- [ ] **Step 3: Add `MacroDataError`**

Append to `src/trading_agent_framework/utils/errors.py`:

```python


class MacroDataError(TradingFrameworkError):
    """Raised when a macro (FRED) data lookup fails."""
```

- [ ] **Step 4: Add `FredCredentials`**

Append to `src/trading_agent_framework/config/env.py`:

```python


@dataclass(frozen=True, slots=True)
class FredCredentials:
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FredCredentials:
        source = env if env is not None else os.environ
        api_key = source.get("FRED_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError("Missing or blank FRED_API_KEY environment variable")
        return cls(api_key=api_key)
```

- [ ] **Step 5: Add the dependency and env doc**

In `pyproject.toml`'s `dependencies` list, insert alphabetically:

```toml
    "fredapi>=0.5,<0.6",
```

(full list becomes `alpaca-py`, `fredapi`, `langchain`, `langchain-openai`, `pandas-ta-classic`, ...)

Append to `env/.env.example`:

```
# FRED (Federal Reserve Economic Data) API key for agents/tools/macro.py.
# Get one at https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=
```

- [ ] **Step 6: Install the new dependency**

```bash
uv sync
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/config/test_env.py tests/test_errors.py -q`
Expected: all pass.

- [ ] **Step 8: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/config/env.py src/trading_agent_framework/utils/errors.py pyproject.toml uv.lock env/.env.example tests/config/test_env.py tests/test_errors.py
git commit -m "Task 9: add FredCredentials, MacroDataError and the fredapi dependency"
```

(Add the attribution trailer lines to the message.)

---

### Task 10: `macro_tools(strategy)` — FRED series lookup

**Files:**
- Create: `src/trading_agent_framework/agents/tools/macro.py`
- Test: `tests/agents/tools/test_macro_tools.py`

**Interfaces:**
- Consumes: `FredCredentials.from_env` (Task 9), `MacroDataError` (Task 9), `Strategy.clock.now()`.
- Produces (used by Task 14): `macro.macro_tools(strategy, *, fred_client_factory=<default>) -> list[Callable]` returning `[get_fred_series]`. `macro.FredSeriesClient` (Protocol) — the test seam's shape: `get_series(series_id, observation_start=None, observation_end=None, realtime_start=None, realtime_end=None) -> pandas.Series`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_macro_tools.py`:

```python
from __future__ import annotations

from datetime import date

import pandas as pd
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.core.strategy import Strategy


class _FakeFred:
    def __init__(self, series: pd.Series, calls: list[dict[str, object]]) -> None:
        self._series = series
        self._calls = calls

    def get_series(self, series_id, observation_start=None, observation_end=None, realtime_start=None, realtime_end=None):
        self._calls.append(
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
            }
        )
        return self._series


def _strategy() -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))


def _tool(series: pd.Series, calls: list[dict[str, object]]):
    strategy = _strategy()
    [tool] = macro_tools(strategy, fred_client_factory=lambda: _FakeFred(series, calls))
    return tool, strategy


def test_get_fred_series_returns_observations() -> None:
    series = pd.Series(
        [21000.0, 21050.0],
        index=pd.to_datetime(["2026-08-01", "2026-09-01"]),
    )
    calls: list[dict[str, object]] = []
    tool, _ = _tool(series, calls)

    result = tool("M2SL")

    assert result["series_id"] == "M2SL"
    assert result["observations"] == [
        {"date": "2026-08-01", "value": 21000.0},
        {"date": "2026-09-01", "value": 21050.0},
    ]


def test_get_fred_series_pins_realtime_and_observation_end_to_the_clock() -> None:
    calls: list[dict[str, object]] = []
    tool, strategy = _tool(pd.Series(dtype=float), calls)

    tool("FEDFUNDS", start_date="2020-01-01")

    [call] = calls
    cutoff = strategy.clock.now().date()
    assert call["observation_start"] == "2020-01-01"
    assert call["observation_end"] == cutoff
    assert call["realtime_end"] == cutoff


def test_get_fred_series_clamps_limit_to_the_last_n_observations() -> None:
    series = pd.Series([float(i) for i in range(10)], index=pd.date_range("2026-01-01", periods=10, freq="D"))
    calls: list[dict[str, object]] = []
    tool, _ = _tool(series, calls)

    result = tool("CPIAUCSL", limit=3)

    assert len(result["observations"]) == 3
    assert result["observations"][-1]["value"] == 9.0


def test_get_fred_series_returns_an_error_dict_on_failure() -> None:
    class _Boom:
        def get_series(self, *args, **kwargs):
            raise RuntimeError("network down")

    strategy = _strategy()
    [tool] = macro_tools(strategy, fred_client_factory=lambda: _Boom())

    result = tool("M2SL")

    assert "error" in result
    assert "M2SL" in result["error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_macro_tools.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `macro.py`**

Create `src/trading_agent_framework/agents/tools/macro.py`:

```python
"""Plain typed macro tool for a LangChain agent: FRED time series, gated on the strategy clock."""

from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

from trading_agent_framework.config.env import FredCredentials
from trading_agent_framework.utils.errors import MacroDataError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 250


class FredSeriesClient(Protocol):
    def get_series(
        self,
        series_id: str,
        observation_start: object = None,
        observation_end: object = None,
        realtime_start: object = None,
        realtime_end: object = None,
    ) -> Any: ...


def _default_fred_client() -> FredSeriesClient:
    import fredapi  # deferred: a strategy that never wires in the macro tool doesn't pay for it

    return fredapi.Fred(api_key=FredCredentials.from_env().api_key)


def _fetch_series(client: FredSeriesClient, series_id: str, start_date: str | None, cutoff: date) -> Any:
    try:
        return client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=cutoff,
            realtime_start=None,
            realtime_end=cutoff,
        )
    except Exception as exc:
        raise MacroDataError(f"Failed to fetch FRED series {series_id!r}: {exc}") from exc


def macro_tools(
    strategy: "Strategy", *, fred_client_factory: Callable[[], FredSeriesClient] = _default_fred_client
) -> list[Callable[..., dict[str, Any]]]:
    """FRED macro tool bound to `strategy`."""

    def get_fred_series(series_id: str, start_date: str | None = None, limit: int = 60) -> dict[str, Any]:
        """Fetch a FRED macro time series (e.g. M2SL, FEDFUNDS, CPIAUCSL, UNRATE) up to the current date."""
        clamped_limit = min(max(int(limit), MIN_LIMIT), MAX_LIMIT)
        cutoff = strategy.clock.now().date()
        try:
            series = _fetch_series(fred_client_factory(), series_id, start_date, cutoff)
        except MacroDataError as exc:
            return {"error": str(exc)}
        observations = [
            {"date": index.date().isoformat() if hasattr(index, "date") else str(index), "value": float(value)}
            for index, value in series.items()
        ]
        return {
            "series_id": series_id,
            "as_of": cutoff.isoformat(),
            "observations": observations[-clamped_limit:],
        }

    return [get_fred_series]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_macro_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/macro.py tests/agents/tools/test_macro_tools.py
git commit -m "Task 10: add macro_tools (FRED series lookup)"
```

(Add the attribution trailer lines to the message.)

---

### Task 11: `fundamentals/sec.py` — pure SEC EDGAR translation

**Files:**
- Create: `src/trading_agent_framework/fundamentals/__init__.py`
- Create: `src/trading_agent_framework/fundamentals/sec.py`
- Test: `tests/fundamentals/test_sec.py`

**Interfaces:**
- Consumes: nothing internal (stdlib only: `html`, `re`, `datetime`).
- Produces (used by Task 12, 13): `sec.INCOME_STATEMENT_TAGS`, `sec.BALANCE_SHEET_TAGS` (`dict[str, list[str]]`); `sec.parse_company_tickers(payload: dict, symbol: str) -> str` (raises `ValueError`); `sec.filter_facts_as_of(facts: dict, tags: list[str], as_of: datetime) -> list[dict]`; `sec.latest_fact(candidates: list[dict]) -> dict | None`; `sec.compact_company_facts(payload: dict, *, as_of: datetime, max_facts: int = 80) -> dict` (`{"facts", "fact_count", "truncated"}`); `sec.statement_values(payload: dict, tag_map: dict[str, list[str]], *, as_of: datetime) -> dict[str, dict]`; `sec.parse_filings(submissions_payload: dict, *, cik: str, as_of: datetime, form: str | None = None, limit: int = 10) -> list[dict]` (each row has `form, accession_number, filing_date, report_date, primary_document, document_url`); `sec.filing_url(cik: str, accession_number: str, primary_document: str) -> str`; `sec.strip_html(raw: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fundamentals/test_sec.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_agent_framework.fundamentals import sec

_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


def test_parse_company_tickers_finds_the_cik_case_insensitively() -> None:
    assert sec.parse_company_tickers(_TICKERS_PAYLOAD, "aapl") == "0000320193"


def test_parse_company_tickers_raises_for_an_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="ZZZZ"):
        sec.parse_company_tickers(_TICKERS_PAYLOAD, "ZZZZ")


def _facts(tag: str, *rows: dict[str, object]) -> dict[str, object]:
    return {tag: {"units": {"USD": list(rows)}}}


def test_filter_facts_as_of_excludes_post_cutoff_facts() -> None:
    facts = _facts(
        "NetIncomeLoss",
        {"val": 100, "filed": "2026-01-01", "form": "10-K", "accn": "a1"},
        {"val": 200, "filed": "2026-06-01", "form": "10-K", "accn": "a2"},
    )
    as_of = datetime(2026, 3, 1, tzinfo=UTC)

    candidates = sec.filter_facts_as_of(facts, ["NetIncomeLoss"], as_of)

    assert [c["value"] for c in candidates] == [100]


def test_filter_facts_as_of_prefers_10k_then_most_recent() -> None:
    facts = _facts(
        "NetIncomeLoss",
        {"val": 50, "filed": "2026-01-01", "form": "8-K", "accn": "a1"},
        {"val": 60, "filed": "2026-01-01", "form": "10-K", "accn": "a2"},
    )
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    candidates = sec.filter_facts_as_of(facts, ["NetIncomeLoss"], as_of)

    assert sec.latest_fact(candidates)["value"] == 60


def test_latest_fact_of_no_candidates_is_none() -> None:
    assert sec.latest_fact([]) is None


def test_compact_company_facts_orders_priority_tags_first_and_caps_at_max_facts() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K"}]}},
                "ZzzCustomTag": {"units": {"USD": [{"val": 1, "filed": "2026-01-01", "form": "10-K"}]}},
                "Assets": {"units": {"USD": [{"val": 20, "filed": "2026-01-01", "form": "10-K"}]}},
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    result = sec.compact_company_facts(payload, as_of=as_of, max_facts=2)

    assert result["fact_count"] == 2
    assert result["truncated"] is True
    assert set(result["facts"]) <= {"NetIncomeLoss", "Assets", "ZzzCustomTag"}


def test_statement_values_omits_a_field_whose_only_candidate_mismatches_the_anchor_period() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}
                },
                "GrossProfit": {
                    "units": {"USD": [{"val": 99, "filed": "2026-01-01", "form": "10-K", "end": "2024-12-31", "accn": "a2"}]}
                },
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    values = sec.statement_values(payload, {"net_income": ["NetIncomeLoss"], "gross_profit": ["GrossProfit"]}, as_of=as_of)

    # anchor is whichever candidate sorts first (NetIncomeLoss, filed/end used as sort keys);
    # gross_profit's different `end` means it can't match that anchor and is omitted.
    assert "net_income" in values
    assert "gross_profit" not in values


def test_parse_filings_filters_by_form_and_as_of() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-K"],
                "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002", "0000320193-26-000003"],
                "filingDate": ["2026-01-01", "2026-02-01", "2026-12-01"],
                "reportDate": ["2025-12-31", "2026-01-31", "2026-11-30"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z", "2026-02-01T20:00:00Z", "2026-12-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm", "aapl-8k.htm", "aapl-10k2.htm"],
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    rows = sec.parse_filings(submissions, cik="0000320193", as_of=as_of, form="10-K", limit=10)

    assert [row["accession_number"] for row in rows] == ["0000320193-26-000001"]
    assert rows[0]["document_url"] == sec.filing_url("0000320193", "0000320193-26-000001", "aapl-10k.htm")


def test_parse_filings_respects_the_limit() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q", "10-Q"],
                "accessionNumber": ["a1", "a2", "a3"],
                "filingDate": ["2026-01-01", "2026-02-01", "2026-03-01"],
                "reportDate": [None, None, None],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z", "2026-02-01T20:00:00Z", "2026-03-01T20:00:00Z"],
                "primaryDocument": ["d1.htm", "d2.htm", "d3.htm"],
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    rows = sec.parse_filings(submissions, cik="1", as_of=as_of, limit=2)

    assert len(rows) == 2


def test_filing_url_strips_dashes_and_leading_zeros() -> None:
    url = sec.filing_url("0000320193", "0000320193-26-000001", "aapl-10k.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl-10k.htm"


def test_strip_html_removes_tags_and_collapses_whitespace() -> None:
    raw = "<html><body><p>Hello &amp; welcome</p><script>bad()</script></body></html>"
    assert sec.strip_html(raw) == "Hello & welcome"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_sec.py -q`
Expected: `ModuleNotFoundError: No module named 'trading_agent_framework.fundamentals'`.

- [ ] **Step 3: Create the package and `sec.py`**

Create `src/trading_agent_framework/fundamentals/__init__.py`:

```python
"""SEC EDGAR fundamentals: `sec.py` (pure translation) and `edgar_client.py` (cached I/O)."""
```

Create `src/trading_agent_framework/fundamentals/sec.py`:

```python
"""Pure SEC EDGAR translation: CIK lookup, income-statement/balance-sheet tag maps, as-of
candidate filtering and statement-period matching, filings parsing, URL building, HTML stripping.

No I/O, no state, no strategy/clock knowledge (same rules as `brokers/alpaca/market_data.py`).
Ported and trimmed from lumibot's `SECFundamentals` (`lumibot/fundamentals/sec.py`).
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime
from typing import Any

SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data/"

INCOME_STATEMENT_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

BALANCE_SHEET_TAGS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "debt": ["LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent", "LongTermDebt"],
    "equity": ["StockholdersEquity"],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStocksIncludingAdditionalPaidInCapital",
    ],
}

_PRIORITY_COMPANY_FACT_TAGS = tuple(
    dict.fromkeys(
        tag for tags in (*INCOME_STATEMENT_TAGS.values(), *BALANCE_SHEET_TAGS.values()) for tag in tags
    )
)

_FORM_PRIORITY = {"10-K": 5, "20-F": 5, "40-F": 5, "10-Q": 4, "8-K": 2}


def parse_company_tickers(payload: dict[str, Any], symbol: str) -> str:
    """The zero-padded 10-digit CIK for `symbol`; raises `ValueError` if not found."""
    symbol_upper = symbol.upper().strip()
    for entry in payload.values():
        if str(entry.get("ticker", "")).upper() == symbol_upper:
            return f"{int(entry['cik_str']):010d}"
    raise ValueError(f"No SEC CIK found for ticker {symbol!r}")


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _same_tz(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo is None and reference.tzinfo is not None:
        return value.replace(tzinfo=reference.tzinfo)
    if value.tzinfo is not None and reference.tzinfo is None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _candidate_sort_key(row: dict[str, Any], tags: list[str] | None = None) -> tuple[Any, ...]:
    form_priority = _FORM_PRIORITY.get(str(row.get("form") or "").upper(), 1)
    tag_priority = 0
    if tags:
        try:
            tag_priority = len(tags) - tags.index(str(row.get("tag") or ""))
        except ValueError:
            tag_priority = 0
    return (
        str(row.get("filed") or ""),
        str(row.get("end") or ""),
        str(row.get("start") or ""),
        form_priority,
        tag_priority,
    )


def filter_facts_as_of(facts: dict[str, Any], tags: list[str], as_of: datetime) -> list[dict[str, Any]]:
    """Candidates for `tags` filed on or before `as_of`, best (most recent, highest-priority form) first."""
    candidates: list[dict[str, Any]] = []
    for tag in tags:
        tag_payload = facts.get(tag)
        if not tag_payload:
            continue
        for unit, unit_facts in tag_payload.get("units", {}).items():
            if not isinstance(unit_facts, list):
                continue
            for fact in unit_facts:
                if not isinstance(fact, dict):
                    continue
                filed = _parse_dt(fact.get("filed") or fact.get("acceptanceDateTime"))
                if filed is None or _same_tz(filed, as_of) > as_of:
                    continue
                candidates.append(
                    {
                        "tag": tag,
                        "value": fact.get("val"),
                        "unit": unit,
                        "filed": fact.get("filed"),
                        "form": fact.get("form"),
                        "fy": fact.get("fy"),
                        "fp": fact.get("fp"),
                        "start": fact.get("start"),
                        "end": fact.get("end"),
                        "accession_number": fact.get("accn"),
                    }
                )
    candidates.sort(key=lambda row: _candidate_sort_key(row, tags), reverse=True)
    return candidates


def latest_fact(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most relevant candidate; `filter_facts_as_of` already sorts best-first."""
    return candidates[0] if candidates else None


def compact_company_facts(payload: dict[str, Any], *, as_of: datetime, max_facts: int = 80) -> dict[str, Any]:
    """The latest as-of value for each priority tag present, capped at `max_facts`."""
    facts = payload.get("facts", {}).get("us-gaap", {})
    ordered_tags = [tag for tag in _PRIORITY_COMPANY_FACT_TAGS if tag in facts]
    ordered_tags.extend(tag for tag in sorted(facts) if tag not in ordered_tags)
    limit = max(int(max_facts), 1)
    compact: dict[str, Any] = {}
    truncated = False
    for tag in ordered_tags:
        if len(compact) >= limit:
            truncated = True
            break
        fact = latest_fact(filter_facts_as_of(facts, [tag], as_of))
        if fact is not None:
            compact[tag] = {"value": fact["value"], "unit": fact["unit"], "filed": fact["filed"], "form": fact["form"]}
    return {"facts": compact, "fact_count": len(compact), "truncated": truncated}


def _statement_anchor(field_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = [row for rows in field_candidates.values() for row in rows]
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_sort_key, reverse=True)[0]


def _same_statement_period(candidate: dict[str, Any], anchor: dict[str, Any]) -> bool:
    candidate_accn = str(candidate.get("accession_number") or "").strip()
    anchor_accn = str(anchor.get("accession_number") or "").strip()
    if candidate_accn and anchor_accn and candidate_accn == anchor_accn:
        return True
    candidate_end = str(candidate.get("end") or "").strip()
    anchor_end = str(anchor.get("end") or "").strip()
    if not candidate_end or not anchor_end or candidate_end != anchor_end:
        return False
    for key in ("start", "fy", "fp", "form"):
        candidate_value = str(candidate.get(key) or "").strip()
        anchor_value = str(anchor.get(key) or "").strip()
        if candidate_value and anchor_value and candidate_value != anchor_value:
            return False
    return True


def statement_values(
    payload: dict[str, Any], tag_map: dict[str, list[str]], *, as_of: datetime
) -> dict[str, dict[str, Any]]:
    """Income-statement / balance-sheet field values as of `as_of`, from a raw company-facts payload.

    A field whose best candidate doesn't match the statement's period anchor is omitted, rather
    than mixing facts pulled from different SEC filings or periods.
    """
    facts = payload.get("facts", {}).get("us-gaap", {})
    field_candidates = {
        field: candidates
        for field, tags in tag_map.items()
        if (candidates := filter_facts_as_of(facts, tags, as_of))
    }
    anchor = _statement_anchor(field_candidates)
    if anchor is None:
        return {}
    values: dict[str, dict[str, Any]] = {}
    for field, candidates in field_candidates.items():
        for candidate in candidates:
            if _same_statement_period(candidate, anchor):
                values[field] = {
                    "value": candidate["value"], "unit": candidate["unit"],
                    "filed": candidate["filed"], "form": candidate["form"],
                }
                break
    return values


def parse_filings(
    submissions_payload: dict[str, Any],
    *,
    cik: str,
    as_of: datetime,
    form: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Recent filings filed on or before `as_of`, optionally restricted to one `form`."""
    recent = submissions_payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    primary_docs = recent.get("primaryDocument", [])
    rows: list[dict[str, Any]] = []
    for idx, filing_form in enumerate(forms):
        if form and str(filing_form).upper() != form.upper():
            continue
        filed_raw = acceptances[idx] if idx < len(acceptances) and acceptances[idx] else filing_dates[idx]
        filed_dt = _parse_dt(filed_raw)
        if filed_dt is None or _same_tz(filed_dt, as_of) > as_of:
            continue
        accession = accession_numbers[idx]
        primary_document = primary_docs[idx] if idx < len(primary_docs) else ""
        rows.append(
            {
                "form": filing_form,
                "accession_number": accession,
                "filing_date": filing_dates[idx] if idx < len(filing_dates) else None,
                "report_date": report_dates[idx] if idx < len(report_dates) else None,
                "primary_document": primary_document,
                "document_url": filing_url(cik, accession, primary_document),
            }
        )
        if len(rows) >= max(int(limit), 1):
            break
    return rows


def filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(str(cik).lstrip("0") or "0"))
    accession_clean = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES_BASE_URL}{cik_int}/{accession_clean}/{primary_document}"


def strip_html(raw: str) -> str:
    """Filing HTML/XBRL to readable text."""
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    text = re.sub(r"(?is)<ix:hidden.*?</ix:hidden>", " ", text)
    text = re.sub(r"(?is)</?(?:p|div|br|tr|table|section|article|h[1-6])\b[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_sec.py -q`
Expected: all pass. If `test_statement_values_omits_a_field_whose_only_candidate_mismatches_the_anchor_period`'s exact anchor choice doesn't match the assertion (the anchor is whichever single candidate sorts highest across *all* fields, and with one candidate per field here it's a coin flip between `net_income`'s and `gross_profit`'s row), adjust the test's `filed`/`end` values so `net_income`'s row sorts first deterministically (e.g. give it a later `filed` date) rather than changing `statement_values` — the behavior (a field with a non-matching period is dropped) is what's under test, not which specific field wins the anchor race.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/fundamentals/__init__.py src/trading_agent_framework/fundamentals/sec.py tests/fundamentals/test_sec.py
git commit -m "Task 11: add fundamentals/sec.py (pure SEC EDGAR translation)"
```

(Add the attribution trailer lines to the message.)

---

### Task 12: `fundamentals/edgar_client.py` — cached SEC EDGAR I/O

**Files:**
- Modify: `src/trading_agent_framework/fundamentals/__init__.py`
- Create: `src/trading_agent_framework/fundamentals/edgar_client.py`
- Modify: `src/trading_agent_framework/utils/errors.py`
- Modify: `pyproject.toml`
- Modify: `env/.env.example`
- Modify: `.gitignore`
- Test: `tests/fundamentals/test_edgar_client.py`, `tests/test_errors.py` (append)

**Interfaces:**
- Consumes: `fundamentals.sec.parse_company_tickers`, `fundamentals.sec.filing_url` (Task 11).
- Produces (used by Task 13): `edgar_client.SecEdgarClient(user_agent: str, cache_dir: Path, *, min_request_interval_seconds: float = 0.2, transport: httpx.BaseTransport | None = None)`, raising `ConfigurationError` on blank `user_agent`. Methods: `.get_json(url, cache_key) -> dict`, `.get_text(url, cache_key) -> str`, `.ticker_to_cik(symbol) -> str`, `.get_company_facts_payload(cik) -> dict`, `.get_submissions_payload(cik) -> dict`, `.get_filing_text(cik, accession_number, primary_document) -> str`. All raise `FundamentalsError` on request failure. `utils.errors.FundamentalsError(TradingFrameworkError)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fundamentals/test_edgar_client.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL"}}


def _client(tmp_path: Path, handler) -> SecEdgarClient:
    transport = httpx.MockTransport(handler)
    return SecEdgarClient("TestApp test@example.com", tmp_path, min_request_interval_seconds=0.0, transport=transport)


def test_blank_user_agent_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="SEC_EDGAR_USER_AGENT"):
        SecEdgarClient("  ", tmp_path)


def test_get_json_fetches_and_caches(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["User-Agent"] == "TestApp test@example.com"
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, handler)

    first = client.get_json("https://data.sec.gov/x.json", ("x.json",))
    second = client.get_json("https://data.sec.gov/x.json", ("x.json",))

    assert first == second == {"ok": True}
    assert len(calls) == 1  # second call was a cache hit
    assert (tmp_path / "x.json").exists()


def test_get_json_wraps_http_errors_as_fundamentals_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _client(tmp_path, handler)

    with pytest.raises(FundamentalsError):
        client.get_json("https://data.sec.gov/missing.json", ("missing.json",))


def test_get_text_fetches_and_caches(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>filing</html>")

    client = _client(tmp_path, handler)

    text = client.get_text("https://www.sec.gov/f.htm", ("filings", "f.htm"))

    assert text == "<html>filing</html>"
    assert (tmp_path / "filings" / "f.htm").exists()


def test_ticker_to_cik_uses_sec_parse_company_tickers(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_TICKERS)

    client = _client(tmp_path, handler)

    assert client.ticker_to_cik("AAPL") == "0000320193"


def test_ticker_to_cik_unknown_symbol_raises_fundamentals_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_TICKERS)

    client = _client(tmp_path, handler)

    with pytest.raises(FundamentalsError, match="ZZZZ"):
        client.ticker_to_cik("ZZZZ")


def test_get_company_facts_payload_hits_the_xbrl_endpoint(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"facts": {}})

    client = _client(tmp_path, handler)

    client.get_company_facts_payload("0000320193")

    assert seen_urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]


def test_get_submissions_payload_hits_the_submissions_endpoint(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"filings": {}})

    client = _client(tmp_path, handler)

    client.get_submissions_payload("0000320193")

    assert seen_urls == ["https://data.sec.gov/submissions/CIK0000320193.json"]
```

Append to `tests/test_errors.py`:

```python
from trading_agent_framework.utils.errors import FundamentalsError


def test_fundamentals_error_is_a_trading_framework_error() -> None:
    assert issubclass(FundamentalsError, TradingFrameworkError)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/fundamentals/test_edgar_client.py tests/test_errors.py -q`
Expected: `ModuleNotFoundError: No module named 'trading_agent_framework.fundamentals.edgar_client'` (and `ImportError: cannot import name 'FundamentalsError'` for the errors test).

- [ ] **Step 3: Add `FundamentalsError`**

Append to `src/trading_agent_framework/utils/errors.py`:

```python


class FundamentalsError(TradingFrameworkError):
    """Raised when a SEC EDGAR fundamentals lookup fails."""
```

- [ ] **Step 4: Create `edgar_client.py`**

Create `src/trading_agent_framework/fundamentals/edgar_client.py`:

```python
"""SEC EDGAR I/O: `data.sec.gov` requests, cached to disk, rate-limited.

The only module allowed to import `httpx` for SEC access. Wraps every network failure as
`FundamentalsError` (repo rule: no raw library exception escapes) and requires a non-blank
`user_agent` (SEC's fair-access policy blocks generic/missing ones).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


class SecEdgarClient:
    """Cached, rate-limited SEC EDGAR REST client."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        min_request_interval_seconds: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise ConfigurationError("Missing or blank SEC_EDGAR_USER_AGENT environment variable")
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_request_interval_seconds = max(float(min_request_interval_seconds), 0.0)
        self._last_request_at = 0.0
        self._client = httpx.Client(transport=transport, timeout=30.0)

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def _cache_path(self, *parts: str) -> Path:
        safe = [_SAFE_CHARS.sub("_", str(part)).strip("_") for part in parts]
        return self.cache_dir.joinpath(*safe)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def get_json(self, url: str, cache_key: tuple[str, ...]) -> dict[str, Any]:
        cache_path = self._cache_path(*cache_key)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        self._rate_limit()
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def get_text(self, url: str, cache_key: tuple[str, ...]) -> str:
        cache_path = self._cache_path(*cache_key)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")
        self._rate_limit()
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            text = response.text
        except httpx.HTTPError as exc:
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8", errors="replace")
        return text

    def ticker_to_cik(self, symbol: str) -> str:
        payload = self.get_json(SEC_COMPANY_TICKERS_URL, ("company_tickers.json",))
        try:
            return sec.parse_company_tickers(payload, symbol)
        except ValueError as exc:
            raise FundamentalsError(str(exc)) from exc

    def get_company_facts_payload(self, cik: str) -> dict[str, Any]:
        url = f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        return self.get_json(url, ("companyfacts", f"CIK{cik}.json"))

    def get_submissions_payload(self, cik: str) -> dict[str, Any]:
        url = f"{SEC_DATA_BASE_URL}/submissions/CIK{cik}.json"
        return self.get_json(url, ("submissions", f"CIK{cik}.json"))

    def get_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        url = sec.filing_url(cik, accession_number, primary_document)
        return self.get_text(url, ("filings", cik, accession_number, primary_document))
```

- [ ] **Step 5: Export `SecEdgarClient` and add the dependency**

Replace `src/trading_agent_framework/fundamentals/__init__.py`:

```python
"""SEC EDGAR fundamentals: `sec.py` (pure translation) and `edgar_client.py` (cached I/O)."""

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient

__all__ = ["SecEdgarClient"]
```

In `pyproject.toml`'s `dependencies` list, insert alphabetically:

```toml
    "httpx>=0.27",
```

Append to `env/.env.example`:

```
# Identity string SEC EDGAR requires on every request (fair-access policy):
# "<app or project name> <contact email>". Requests without one may be blocked.
SEC_EDGAR_USER_AGENT=
```

Append to `.gitignore`:

```

# SEC EDGAR fundamentals cache (company-level, not strategy/mode scoped)
/cache/
```

- [ ] **Step 6: Install the new dependency**

```bash
uv sync
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/fundamentals/test_edgar_client.py tests/test_errors.py -q`
Expected: all pass.

- [ ] **Step 8: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/fundamentals/__init__.py src/trading_agent_framework/fundamentals/edgar_client.py src/trading_agent_framework/utils/errors.py pyproject.toml uv.lock env/.env.example .gitignore tests/fundamentals/test_edgar_client.py tests/test_errors.py
git commit -m "Task 12: add SecEdgarClient (cached SEC EDGAR I/O) and the httpx dependency"
```

(Add the attribution trailer lines to the message.)

---

### Task 13: `fundamentals_tools(strategy)`

**Files:**
- Create: `src/trading_agent_framework/agents/tools/fundamentals.py`
- Test: `tests/agents/tools/test_fundamentals_tools.py`

**Interfaces:**
- Consumes: `fundamentals.sec.*` (Task 11), `fundamentals.edgar_client.SecEdgarClient` (Task 12), `FundamentalsError`, `ConfigurationError`.
- Produces (used by Task 14): `fundamentals.fundamentals_tools(strategy, *, client=None) -> list[Callable]` returning, in order: `get_company_facts`, `get_income_statement`, `get_balance_sheet`, `get_filings`, `get_filing_document`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/tools/test_fundamentals_tools.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.utils.errors import FundamentalsError

_CIK = "0000320193"


class _FakeEdgarClient:
    def __init__(self) -> None:
        self.company_facts: dict[str, object] = {"facts": {"us-gaap": {}}}
        self.submissions: dict[str, object] = {"filings": {"recent": {}}}
        self.filing_text = "<p>Annual report text</p>"
        self.raises: FundamentalsError | None = None

    def ticker_to_cik(self, symbol: str) -> str:
        if self.raises is not None:
            raise self.raises
        return _CIK

    def get_company_facts_payload(self, cik: str) -> dict[str, object]:
        return self.company_facts

    def get_submissions_payload(self, cik: str) -> dict[str, object]:
        return self.submissions

    def get_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        return self.filing_text


def _strategy() -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 14, 10))))


def _tools(client: _FakeEdgarClient) -> dict[str, object]:
    return {tool.__name__: tool for tool in fundamentals_tools(_strategy(), client=client)}


def test_returns_five_tools_with_one_line_docstrings() -> None:
    tools = fundamentals_tools(_strategy(), client=_FakeEdgarClient())
    assert [t.__name__ for t in tools] == [
        "get_company_facts", "get_income_statement", "get_balance_sheet", "get_filings", "get_filing_document",
    ]
    for tool in tools:
        assert len(tool.__doc__.splitlines()) == 1


def test_get_company_facts_returns_symbol_cik_and_compact_facts() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_company_facts"]("aapl")

    assert result["symbol"] == "AAPL"
    assert result["cik"] == _CIK
    assert "NetIncomeLoss" in result["facts"]


def test_get_company_facts_returns_error_on_unknown_ticker() -> None:
    client = _FakeEdgarClient()
    client.raises = FundamentalsError("No SEC CIK found for ticker 'ZZZZ'")
    tools = _tools(client)

    assert tools["get_company_facts"]("ZZZZ") == {"error": "No SEC CIK found for ticker 'ZZZZ'"}


def test_get_income_statement_returns_values_for_the_symbol() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_income_statement"]("AAPL")

    assert result["symbol"] == "AAPL"
    assert result["values"]["net_income"]["value"] == 10


def test_get_balance_sheet_returns_values_for_the_symbol() -> None:
    client = _FakeEdgarClient()
    client.company_facts = {
        "facts": {"us-gaap": {"Assets": {"units": {"USD": [{"val": 999, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}}}}
    }
    tools = _tools(client)

    result = tools["get_balance_sheet"]("AAPL")

    assert result["values"]["assets"]["value"] == 999


def test_get_filings_omits_primary_document_from_the_returned_rows() -> None:
    client = _FakeEdgarClient()
    client.submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["0000320193-26-000001"],
                "filingDate": ["2026-01-01"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm"],
            }
        }
    }
    tools = _tools(client)

    result = tools["get_filings"]("AAPL", form="10-K")

    [filing] = result["filings"]
    assert "primary_document" not in filing
    assert filing["accession_number"] == "0000320193-26-000001"


def test_get_filing_document_strips_html_and_truncates() -> None:
    client = _FakeEdgarClient()
    client.submissions = {
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": ["acc-1"],
                "filingDate": ["2026-01-01"],
                "reportDate": ["2025-12-31"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm"],
            }
        }
    }
    client.filing_text = "<p>" + "x" * 20 + "</p>"
    tools = _tools(client)

    result = tools["get_filing_document"]("AAPL", "acc-1", max_chars=5)

    assert result["text"] == "xxxxx"
    assert result["truncated"] is True


def test_get_filing_document_unknown_accession_number_returns_error() -> None:
    tools = _tools(_FakeEdgarClient())

    result = tools["get_filing_document"]("AAPL", "does-not-exist")

    assert "error" in result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agents/tools/test_fundamentals_tools.py -q`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `fundamentals.py`**

Create `src/trading_agent_framework/agents/tools/fundamentals.py`:

```python
"""Plain typed SEC fundamentals tools for a LangChain agent, gated on the strategy clock."""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_FILINGS_LIMIT = 1
MAX_FILINGS_LIMIT = 25


def _default_client(strategy: "Strategy") -> SecEdgarClient:
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "")
    if not user_agent.strip():
        raise ConfigurationError("Missing or blank SEC_EDGAR_USER_AGENT environment variable")
    return SecEdgarClient(user_agent, strategy.project_root / "cache" / "sec")


def fundamentals_tools(
    strategy: "Strategy", *, client: SecEdgarClient | None = None
) -> list[Callable[..., dict[str, Any]]]:
    """SEC fundamentals tools bound to `strategy`, backed by one shared `SecEdgarClient`."""
    edgar = client if client is not None else _default_client(strategy)

    def _statement(symbol: str, tag_map: dict[str, list[str]]) -> dict[str, Any]:
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            payload = edgar.get_company_facts_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        values = sec.statement_values(payload, tag_map, as_of=as_of)
        return {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "values": values}

    def get_company_facts(symbol: str, max_facts: int = 40) -> dict[str, Any]:
        """Get a company's latest SEC XBRL facts (revenue, assets, EPS, ...) as of now."""
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            payload = edgar.get_company_facts_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        compact = sec.compact_company_facts(payload, as_of=as_of, max_facts=max_facts)
        return {"symbol": symbol.upper(), "cik": cik, "as_of": as_of.isoformat(), **compact}

    def get_income_statement(symbol: str) -> dict[str, Any]:
        """Get a company's income statement (revenue, net income, EPS, ...) as of now."""
        return _statement(symbol, sec.INCOME_STATEMENT_TAGS)

    def get_balance_sheet(symbol: str) -> dict[str, Any]:
        """Get a company's balance sheet (assets, liabilities, equity, ...) as of now."""
        return _statement(symbol, sec.BALANCE_SHEET_TAGS)

    def get_filings(symbol: str, form: str | None = None, limit: int = 10) -> dict[str, Any]:
        """List a company's recent SEC filings (10-K, 10-Q, 8-K, ...) as of now."""
        clamped_limit = min(max(int(limit), MIN_FILINGS_LIMIT), MAX_FILINGS_LIMIT)
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            submissions = edgar.get_submissions_payload(cik)
        except FundamentalsError as exc:
            return {"error": str(exc)}
        rows = sec.parse_filings(submissions, cik=cik, as_of=as_of, form=form, limit=clamped_limit)
        filings = [{k: v for k, v in row.items() if k != "primary_document"} for row in rows]
        return {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "filings": filings}

    def get_filing_document(symbol: str, accession_number: str, max_chars: int = 8000) -> dict[str, Any]:
        """Fetch the readable text of one SEC filing document."""
        as_of = strategy.clock.now()
        try:
            cik = edgar.ticker_to_cik(symbol)
            submissions = edgar.get_submissions_payload(cik)
            rows = sec.parse_filings(submissions, cik=cik, as_of=as_of, limit=1000)
            match = next((row for row in rows if row["accession_number"] == accession_number), None)
            if match is None:
                return {"error": f"unknown accession_number {accession_number!r}"}
            raw = edgar.get_filing_text(cik, accession_number, match["primary_document"])
        except FundamentalsError as exc:
            return {"error": str(exc)}
        text = sec.strip_html(raw)
        truncated = len(text) > max_chars
        return {
            "symbol": symbol.upper(),
            "accession_number": accession_number,
            "document_url": match["document_url"],
            "text": text[:max_chars],
            "truncated": truncated,
        }

    return [get_company_facts, get_income_statement, get_balance_sheet, get_filings, get_filing_document]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agents/tools/test_fundamentals_tools.py -q`
Expected: all pass.

- [ ] **Step 5: Full checks**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/trading_agent_framework/agents/tools/fundamentals.py tests/agents/tools/test_fundamentals_tools.py
git commit -m "Task 13: add fundamentals_tools (SEC company facts, statements, filings)"
```

(Add the attribution trailer lines to the message.)

---

### Task 14: Final integration — exports, docs, smoke scripts

**Files:**
- Modify: `src/trading_agent_framework/agents/tools/__init__.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Create: `scripts/tests/smoke_news.py`, `scripts/tests/smoke_macro.py`, `scripts/tests/smoke_fundamentals.py`
- Test: extend `tests/agents/tools/test_prebuilt.py` (import-surface assertion only; no new automated test file for the smoke scripts, per the spec — they are manual/network)

**Interfaces:**
- Consumes: everything from Tasks 1-13.
- Produces: the final public surface of `trading_agent_framework.agents.tools` (`PrebuiltTools`, `trading_tools`, `account_tools`, `market_data_tools`, `indicator_tools`, `news_tools`, `macro_tools`, `fundamentals_tools`).

- [ ] **Step 1: Write the failing test**

Append to `tests/agents/tools/test_prebuilt.py`:

```python
def test_package_exports_every_tool_factory() -> None:
    import trading_agent_framework.agents.tools as tools_package

    assert set(tools_package.__all__) == {
        "PrebuiltTools",
        "trading_tools",
        "account_tools",
        "market_data_tools",
        "indicator_tools",
        "news_tools",
        "macro_tools",
        "fundamentals_tools",
    }
    for name in tools_package.__all__:
        assert hasattr(tools_package, name)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/agents/tools/test_prebuilt.py::test_package_exports_every_tool_factory -q`
Expected: `AssertionError` (the current `__all__` is missing `news_tools`, `macro_tools`, `fundamentals_tools`).

- [ ] **Step 3: Update `agents/tools/__init__.py`**

Replace `src/trading_agent_framework/agents/tools/__init__.py`:

```python
"""Agent tools: PrebuiltTools.all() plus individually-wired tool factories."""

from trading_agent_framework.agents.tools.account import account_tools
from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.agents.tools.indicators import indicator_tools
from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.agents.tools.market_data import market_data_tools
from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.agents.tools.prebuilt import PrebuiltTools
from trading_agent_framework.agents.tools.trading import trading_tools

__all__ = [
    "PrebuiltTools",
    "account_tools",
    "fundamentals_tools",
    "indicator_tools",
    "macro_tools",
    "market_data_tools",
    "news_tools",
    "trading_tools",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/agents/tools/test_prebuilt.py -q`
Expected: all pass.

- [ ] **Step 5: Update `CLAUDE.md`**

In the `## Architecture` bullet list, after the `memory/` bullet and before the `agents/` bullet (or immediately after the existing `agents/` bullet — place it directly after the line describing `agents/`), insert:

```markdown
- `agents/tools/` -- LangChain tool factories for a trading agent. `trading.py`/`account.py`/`market_data.py`/`indicators.py` wrap `Strategy` methods; `prebuilt.py`'s `PrebuiltTools.all(strategy)` bundles those four plus `memory_tools` in one call. `news.py`/`macro.py`/`fundamentals.py` are wired in explicitly (not part of `PrebuiltTools.all()`): Alpaca news (via `AlpacaBroker.get_news`), FRED macro series (`fredapi`, lazy import), and SEC EDGAR fundamentals (`fundamentals/`). Every tool factory takes `strategy`, not a raw client, so research tools can gate on `strategy.clock.now()`.
- `fundamentals/` -- SEC EDGAR client, trimmed and ported from lumibot's `SECFundamentals`: `sec.py` (**pure**: tag maps, as-of candidate filtering, statement-period matching, filings parsing, URL building, HTML stripping) and `edgar_client.py` (`SecEdgarClient`, the only module allowed to import `httpx` for SEC access -- cached to `<project_root>/cache/sec/`, rate-limited, requires `SEC_EDGAR_USER_AGENT`).
```

In the `## Key patterns / gotchas` list, after the existing "No data tool may use wall-clock time" bullet, insert:

```markdown
- **Research agent tools (news/macro/fundamentals) gate on `strategy.clock.now()`, not the price-bar chokepoint.** `BacktestDataSource.bars()` is the no-look-ahead gate for prices; nothing enforces the equivalent for news, FRED series or SEC filings. Each of `agents/tools/news.py`, `macro.py` and `fundamentals.py` takes the `Strategy` (not a raw client) specifically so it can pass `strategy.clock.now()` as an explicit `end`/`realtime_end`/`as_of` cutoff to its provider. A new research tool that skips this leaks future information into backtests exactly like a data tool that calls `datetime.now()` would.
```

- [ ] **Step 6: Update `README.md`**

In the section documenting env vars (near where `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` are described, alongside the `ALPACA_*` vars), add two short paragraphs:

```markdown
`FRED_API_KEY` is needed only if a strategy wires in `agents.tools.macro_tools` (FRED macro series). Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html.

`SEC_EDGAR_USER_AGENT` is needed only if a strategy wires in `agents.tools.fundamentals_tools` (SEC company facts/filings). SEC's fair-access policy requires a real identity string on every request: `"<app or project name> <contact email>"`.
```

(If `README.md` doesn't have an obvious existing env-var section near those variables, add this next to wherever `ALPACA_API_KEY`/`ALPACA_API_SECRET` are documented — check the file for the right spot before inserting rather than guessing a line number.)

- [ ] **Step 7: Create the three smoke scripts**

Read `scripts/tests/smoke_alpaca_data.py` first (already read during planning) for the exact structure to mirror: module docstring explaining it's manual/network-only, `PROJECT_ROOT`/`ENV_FILE` constants, a `SmokeTestFailure` exception, a `_check` helper, a `main() -> int`, and a `__main__` block that prints `PASS`/`FAIL`.

Create `scripts/tests/smoke_news.py`:

```python
#!/usr/bin/env python3
"""Manual check of the Alpaca news tool against Alpaca's live news feed.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_news.py

Credentials come from env/.env.alpaca.integration-tests, as in smoke_alpaca_data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.news import news_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-news"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)
    [search_news] = news_tools(strategy)

    result = search_news(symbols="SPY,QQQ", limit=5)
    print(f"search_news(SPY,QQQ) -> count={result.get('count')}")
    _check("error" not in result, f"search_news returned an error: {result.get('error')}")
    _check(result["count"] >= 0, "search_news returned a negative count")
    if result["articles"]:
        print(f"first headline: {result['articles'][0]['headline']}")

    print("\nPASS: search_news returned without error.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
```

Create `scripts/tests/smoke_macro.py`:

```python
#!/usr/bin/env python3
"""Manual check of the FRED macro tool against the live FRED API.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_macro.py

Needs FRED_API_KEY in env/.env.alpaca.integration-tests (or another loaded env file) and a
paper-trading Alpaca broker (only used to build a Strategy/clock; no orders are placed).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.macro import macro_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-macro"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)
    [get_fred_series] = macro_tools(strategy)

    result = get_fred_series("M2SL", limit=5)
    print(f"get_fred_series(M2SL) -> {result}")
    _check("error" not in result, f"get_fred_series returned an error: {result.get('error')}")
    _check(len(result["observations"]) > 0, "no observations returned")

    print("\nPASS: get_fred_series returned observations.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
```

Create `scripts/tests/smoke_fundamentals.py`:

```python
#!/usr/bin/env python3
"""Manual check of the SEC fundamentals tools against the live SEC EDGAR API.

NOT part of the automated test suite. Run it by hand:

    uv run python scripts/tests/smoke_fundamentals.py

Needs SEC_EDGAR_USER_AGENT in env/.env.alpaca.integration-tests (or another loaded env file) and
a paper-trading Alpaca broker (only used to build a Strategy/clock; no orders are placed). First
run writes to <project_root>/cache/sec/; delete that directory to force a fresh fetch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from trading_agent_framework.agents.tools.fundamentals import fundamentals_tools
from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.config.env import AlpacaCredentials
from trading_agent_framework.core import Strategy
from trading_agent_framework.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / "env" / ".env.alpaca.integration-tests"

STRATEGY_NAME = "smoke-fundamentals"
SYMBOL = "AAPL"


class SmokeTestFailure(Exception):
    """Raised for any check that didn't hold."""


def _load_credentials() -> AlpacaCredentials:
    if not ENV_FILE.is_file():
        raise SmokeTestFailure(f"Credentials file not found: {ENV_FILE}")
    load_dotenv(ENV_FILE, override=True)
    try:
        return AlpacaCredentials.from_env()
    except ConfigurationError as exc:
        raise SmokeTestFailure(f"Invalid credentials in {ENV_FILE}: {exc}") from exc


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def main() -> int:
    creds = _load_credentials()
    broker = AlpacaBroker.from_credentials(STRATEGY_NAME, creds, with_stream=False)
    strategy = Strategy(broker)
    tools = {tool.__name__: tool for tool in fundamentals_tools(strategy)}

    facts = tools["get_company_facts"](SYMBOL)
    print(f"get_company_facts({SYMBOL}) -> cik={facts.get('cik')} fact_count={facts.get('fact_count')}")
    _check("error" not in facts, f"get_company_facts returned an error: {facts.get('error')}")

    income = tools["get_income_statement"](SYMBOL)
    print(f"get_income_statement({SYMBOL}) -> {list(income.get('values', {}))}")
    _check("error" not in income, f"get_income_statement returned an error: {income.get('error')}")

    filings = tools["get_filings"](SYMBOL, form="10-K", limit=1)
    print(f"get_filings({SYMBOL}, 10-K) -> {filings.get('filings')}")
    _check("error" not in filings, f"get_filings returned an error: {filings.get('error')}")
    _check(len(filings["filings"]) > 0, "no 10-K filings found for AAPL")

    accession = filings["filings"][0]["accession_number"]
    document = tools["get_filing_document"](SYMBOL, accession, max_chars=500)
    print(f"get_filing_document({SYMBOL}, {accession}) -> {len(document.get('text', ''))} chars")
    _check("error" not in document, f"get_filing_document returned an error: {document.get('error')}")

    print("\nPASS: company facts, income statement, filings and a filing document all returned data.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeTestFailure as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
```

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest -q && uv run ruff check && uv check`
Expected: all green. (The three smoke scripts are not run here — they need real credentials and network; run them by hand later if you have `env/.env.alpaca.integration-tests` with `FRED_API_KEY`/`SEC_EDGAR_USER_AGENT` set.)

- [ ] **Step 9: Commit**

```bash
git add src/trading_agent_framework/agents/tools/__init__.py CLAUDE.md README.md scripts/tests/smoke_news.py scripts/tests/smoke_macro.py scripts/tests/smoke_fundamentals.py tests/agents/tools/test_prebuilt.py
git commit -m "Task 14: export news/macro/fundamentals tools, document them, add smoke scripts"
```

(Add the attribution trailer lines to the message.)

---

## After this plan

Once all 14 tasks are merged, the `Tools` line in `TODO.md`'s MIGRATION checklist is the only remaining unstruck item from the original list — the user strikes it themselves (per this plan's Global Constraints, `TODO.md` is never staged by implementation work). Wiring `PrebuiltTools.all()` or individual research tools into a specific strategy's `agents.create(...)` call is explicitly out of scope (spec §1) and is a separate follow-up per strategy.

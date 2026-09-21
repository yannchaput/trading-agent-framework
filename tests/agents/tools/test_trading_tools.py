from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.tools.trading import trading_tools
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType
from trading_agent_framework.entities.order import Order
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.utils.errors import BrokerError


def _strategy() -> tuple[Strategy, FakeBroker]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)))
    return Strategy(broker), broker


def _tools(strategy: Strategy) -> dict[str, Callable[..., dict[str, Any]]]:
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
        assert tools[name].mutates_trading is True
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


# Same defect class as the get_order/cancel_order final-review fix above: AlpacaBroker._submit_order
# deliberately ends in a bare `raise` after order.set_error(exc) (lumibot compatibility contract), so a
# raw Alpaca SDK exception (e.g. a rejected order) must not escape submit_order and crash
# on_trading_iteration().
def test_submit_order_returns_an_error_dict_when_the_broker_raises_a_raw_exception() -> None:
    strategy, broker = _strategy()
    broker.market_data_error = None  # submit doesn't use market data
    tools = _tools(strategy)

    def _boom(order: Order) -> Order:
        raise RuntimeError("insufficient buying power")

    broker._submit_order = _boom  # ty: ignore[invalid-assignment]

    result = tools["submit_order"]("SPY", 1, "buy")

    assert "error" in result
    assert "insufficient buying power" in result["error"]


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


def test_cancel_order_refuses_an_order_submitted_in_the_same_run() -> None:
    # The agent placed an order before it had finished deciding, then cancelled it (24 dropped trades in one
    # backtest) -- or cancelled and resubmitted it. Live, the market order may have filled by then.
    strategy, broker = _strategy()
    tools = _tools(strategy)
    with agent_call_context(run_id="run-1"):
        submitted = tools["submit_order"]("SPY", 1, "buy")
        result = tools["cancel_order"](submitted["identifier"])

    assert "error" in result
    assert "this run" in result["error"]
    assert broker.canceled == []


def test_cancel_order_still_cancels_an_order_from_an_earlier_run() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)
    with agent_call_context(run_id="run-1"):
        submitted = tools["submit_order"]("SPY", 1, "buy")
    with agent_call_context(run_id="run-2"):
        result = tools["cancel_order"](submitted["identifier"])

    assert result == {"identifier": submitted["identifier"], "status": "cancel_requested"}
    assert len(broker.canceled) == 1


def test_cancel_order_refuses_an_order_closed_out_by_close_position_in_the_same_run() -> None:
    strategy, broker = _strategy()
    order = Order(strategy_name="momentum", asset=Asset("SPY"), side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=Decimal(1))
    strategy.close_position = lambda symbol, fraction=1.0: order  # type: ignore[method-assign]
    tools = _tools(strategy)
    broker.submit_order(order)
    with agent_call_context(run_id="run-1"):
        tools["close_position"]("SPY")
        result = tools["cancel_order"](order.identifier)

    assert "this run" in result["error"]
    assert broker.canceled == []


def test_cancel_open_orders_leaves_orders_submitted_in_the_same_run_alone() -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)
    with agent_call_context(run_id="run-1"):
        older = tools["submit_order"]("SPY", 1, "buy")
    with agent_call_context(run_id="run-2"):
        tools["submit_order"]("QQQ", 1, "buy")
        assert tools["cancel_open_orders"]() == {"status": "ok"}

    assert [o.identifier for o in broker.canceled] == [older["identifier"]]


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


# Final-review fix: strategy.get_order can fall through to a raw SDK lookup
# (AlpacaBroker.pull_order re-raises a raw APIError for a non-404 failure); the tool
# must not let that escape and crash on_trading_iteration().
def test_get_order_returns_an_error_dict_when_the_lookup_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)

    def _boom(identifier: str) -> Order:
        raise RuntimeError("malformed order_id")

    monkeypatch.setattr(broker, "pull_order", _boom)

    result = tools["get_order"]("not-a-real-id")

    assert "error" in result
    assert "not-a-real-id" in result["error"]


def test_cancel_order_returns_an_error_dict_when_the_lookup_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, broker = _strategy()
    tools = _tools(strategy)

    def _boom(identifier: str) -> Order:
        raise RuntimeError("malformed order_id")

    monkeypatch.setattr(broker, "pull_order", _boom)

    result = tools["cancel_order"]("not-a-real-id")

    assert "error" in result
    assert "not-a-real-id" in result["error"]

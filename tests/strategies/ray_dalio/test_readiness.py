from __future__ import annotations

import inspect
from typing import Any

import pytest

from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.ray_dalio.readiness import require_readiness_before_orders


def _tools(calls: list[str], *, positions: dict[str, Any] | None = None, order: dict[str, Any] | None = None):
    def get_account_balance() -> dict[str, Any]:
        """Fake balance."""
        calls.append("get_account_balance")
        return {"cash": 1}

    def get_positions() -> dict[str, Any]:
        """Fake positions."""
        calls.append("get_positions")
        return positions if positions is not None else {"positions": []}

    def get_last_price(symbol: str) -> dict[str, Any]:
        """Fake price."""
        calls.append("get_last_price")
        return {"price": 1}

    def submit_order(symbol: str, quantity: int = 1, side: str = "buy") -> dict[str, Any]:
        """Fake order."""
        calls.append("submit_order")
        return order if order is not None else {"identifier": "o1"}

    def other() -> dict[str, Any]:
        """Untouched."""
        return {}

    raw = [get_account_balance, get_positions, get_last_price, submit_order, other]
    return raw, {t.__name__: t for t in require_readiness_before_orders(raw)}


def _check(tools: dict[str, Any], symbol: str = "TQQQ") -> None:
    tools["get_account_balance"]()
    tools["get_positions"]()
    tools["get_last_price"](symbol=symbol)


def test_an_order_without_any_check_is_refused_and_names_all_three() -> None:
    calls: list[str] = []
    _, tools = _tools(calls)
    with agent_call_context(run_id="r1"):
        result = tools["submit_order"](symbol="TQQQ")
    assert result["error"].startswith("ORDER_READINESS_REQUIRED")
    for name in ("get_account_balance", "get_positions", "get_last_price(symbol='TQQQ')"):
        assert name in result["error"]
    assert calls == []


def test_an_order_after_all_three_checks_is_placed() -> None:
    calls: list[str] = []
    _, tools = _tools(calls)
    with agent_call_context(run_id="r1"):
        _check(tools)
        assert tools["submit_order"](symbol="TQQQ") == {"identifier": "o1"}


def test_the_price_check_is_per_symbol() -> None:
    _, tools = _tools([])
    with agent_call_context(run_id="r1"):
        _check(tools, "UPRO")
        result = tools["submit_order"](symbol="TQQQ")
    assert "get_last_price(symbol='TQQQ')" in result["error"]
    assert "get_account_balance" not in result["error"]
    assert "get_positions" not in result["error"]


def test_the_price_check_ignores_case() -> None:
    _, tools = _tools([])
    with agent_call_context(run_id="r1"):
        _check(tools, "tqqq")
        assert tools["submit_order"](symbol="TQQQ") == {"identifier": "o1"}


def test_a_failed_check_does_not_count() -> None:
    _, tools = _tools([], positions={"error": "x"})
    with agent_call_context(run_id="r1"):
        _check(tools)
        result = tools["submit_order"](symbol="TQQQ")
    assert "get_positions" in result["error"]
    assert "get_account_balance" not in result["error"]


def test_an_accepted_order_requires_fresh_account_and_positions() -> None:
    _, tools = _tools([])
    with agent_call_context(run_id="r1"):
        _check(tools)
        tools["submit_order"](symbol="TQQQ")
        result = tools["submit_order"](symbol="TQQQ")
    assert "get_account_balance" in result["error"]
    assert "get_positions" in result["error"]
    assert "get_last_price" not in result["error"]


def test_a_refused_order_keeps_the_checks() -> None:
    calls: list[str] = []
    _, tools = _tools(calls, order={"error": "insufficient"})
    with agent_call_context(run_id="r1"):
        _check(tools)
        tools["submit_order"](symbol="TQQQ")
        result = tools["submit_order"](symbol="TQQQ")
    assert result == {"error": "insufficient"}
    assert calls.count("submit_order") == 2


def test_checks_do_not_carry_over_to_a_new_run() -> None:
    _, tools = _tools([])
    with agent_call_context(run_id="r1"):
        _check(tools)
    with agent_call_context(run_id="r2"):
        result = tools["submit_order"](symbol="TQQQ")
    assert "ORDER_READINESS_REQUIRED" in result["error"]


def test_missing_tool_raises() -> None:
    raw, _ = _tools([])
    with pytest.raises(ValueError, match="get_positions"):
        require_readiness_before_orders([t for t in raw if t.__name__ != "get_positions"])


def test_wrapped_tools_keep_their_names_and_signatures() -> None:
    raw, _ = _tools([])
    guarded = require_readiness_before_orders(raw)
    assert [t.__name__ for t in guarded] == [t.__name__ for t in raw]
    assert inspect.signature(guarded[3]) == inspect.signature(raw[3])
    assert guarded[3].__doc__ == "Fake order."
    assert guarded[4] is raw[4]

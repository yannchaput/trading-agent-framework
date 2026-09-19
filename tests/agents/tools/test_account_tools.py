from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

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


def _tools(strategy: Strategy) -> dict[str, Callable[..., dict[str, Any]]]:
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

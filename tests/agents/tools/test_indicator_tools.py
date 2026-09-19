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
    assert [t.__name__ for t in tools] == ["get_indicator"]  # ty: ignore[unresolved-attribute]
    assert len(tools[0].__doc__.splitlines()) == 1  # ty: ignore[unresolved-attribute]


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

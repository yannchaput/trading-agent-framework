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
    names = [tool.__name__ for tool in tools]  # ty: ignore[unresolved-attribute]

    memory_names = [tool.__name__ for tool in memory_tools(strategy.memory)]  # ty: ignore[unresolved-attribute]
    expected = memory_names + [
        "submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all",
        "get_orders", "get_order",
        "get_account_balance", "get_positions", "get_position",
        "get_last_price", "get_quote", "get_bars",
        "get_indicator",
    ]
    assert names == expected


def test_all_tool_names_are_unique(tmp_path: Path) -> None:
    names = [tool.__name__ for tool in PrebuiltTools.all(_strategy(tmp_path))]  # ty: ignore[unresolved-attribute]
    assert len(names) == len(set(names))


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

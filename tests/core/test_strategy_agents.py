from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents import AgentManager
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


def _strategy(tmp_path: Path, name: str = "momentum") -> Strategy:
    return Strategy(
        FakeBroker(FakeClock(_START), strategy_name=name),
        mode=TradingMode.PAPER,
        project_root=tmp_path,
    )


def test_agents_is_lazy_and_returns_an_agent_manager(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    manager = strategy.agents

    assert isinstance(manager, AgentManager)


def test_agents_is_memoized_across_accesses(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)

    assert strategy.agents is strategy.agents


def test_agents_is_independent_per_strategy_instance(tmp_path: Path) -> None:
    first = _strategy(tmp_path, name="momentum")
    second = _strategy(tmp_path, name="momentum")

    assert first.agents is not second.agents

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.ray_dalio import RayDalioStrategy
from trading_agent_framework.strategies.ray_dalio.prompts import ANALYSTS, CONTEXT, DISAGREEMENT, SYSTEM_PROMPTS, TASK_PROMPTS, TRADER, UNIVERSE
from trading_agent_framework.utils.errors import AgentError, ConfigurationError

_NAMES = (*ANALYSTS, DISAGREEMENT, TRADER)
_TOOLS = ("get_fred_snapshot", "get_fred_latest", "get_fred_series", "search_news", "get_account_balance", "get_positions", "get_last_price", "submit_order")


class _FakeHandle:
    def __init__(self, name: str, order: list[str]) -> None:
        self.name, self.order = name, order
        self.runs: list[tuple[str, dict[str, Any]]] = []
        self.thread: int | None = None
        self.error: Exception | None = None

    def run(self, task_prompt: str, *, context: dict[str, Any]) -> AgentRunResult:
        self.runs.append((task_prompt, context))
        self.thread = threading.get_ident()
        self.order.append(self.name)
        if self.error is not None:
            raise self.error
        return AgentRunResult(output=f"{self.name} view", tool_calls=[])


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.handles = {name: _FakeHandle(name, self.order) for name in _NAMES}

    def create(self, **kwargs: Any) -> _FakeHandle:
        self.created.append(kwargs)
        return self.handles[kwargs["name"]]

    def __getitem__(self, name: str) -> _FakeHandle:
        return self.handles[name]


def _strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, news: bool = True) -> tuple[RayDalioStrategy, _FakeAgents]:
    monkeypatch.setenv("FRED_API_KEY", "key")
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test test@example.com")
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="ray_dalio")
    broker.news = FakeNewsProvider() if news else None
    strategy = RayDalioStrategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tools(agents: _FakeAgents) -> dict[str, dict[str, Any]]:
    return {c["name"]: {t.__name__: t for t in c["tools"]} for c in agents.created}


def test_initialize_creates_the_five_agents_with_lumibots_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)

    strategy.initialize()

    assert [c["name"] for c in agents.created] == list(_NAMES)
    assert all(c["system_prompt"] == SYSTEM_PROMPTS[c["name"]] for c in agents.created)
    assert strategy.sleeptime == "1D"


def test_only_the_trader_has_order_tools_and_they_are_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    tools = _tools(agents)

    assert all("submit_order" not in tools[name] for name in _NAMES if name != TRADER)
    with agent_call_context(run_id="run-1"):
        refused = tools[TRADER]["submit_order"](symbol="SHV", quantity=1, side="buy")
    assert "ORDER_READINESS_REQUIRED" in str(refused)


def test_each_agent_gets_its_own_news_and_macro_tool_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    tools = _tools(agents)

    for tool in ("search_news", "get_fred_snapshot"):
        instances = [tools[name][tool] for name in _NAMES]
        assert all(a is not b for i, a in enumerate(instances) for b in instances[i + 1 :])


def test_every_tool_a_prompt_names_is_in_that_agents_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    tools = _tools(agents)

    for name in _NAMES:
        text = SYSTEM_PROMPTS[name] + TASK_PROMPTS[name]
        assert all(tool in tools[name] for tool in _TOOLS if tool in text), name


def test_an_iteration_runs_analysts_then_disagreement_then_trader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()

    strategy.on_trading_iteration()

    base = {"date": "2026-09-14", **CONTEXT}
    views = {"growth": "growth_agent view", "inflation": "inflation_agent view", "liquidity": "debt_liquidity_agent view"}
    for name in _NAMES:
        assert len(agents[name].runs) == 1 and agents[name].runs[0][0] == TASK_PROMPTS[name]
    for name in ANALYSTS:
        assert agents[name].runs[0][1] == base
    assert agents[DISAGREEMENT].runs[0][1] == {**base, **views}
    assert agents[TRADER].runs[0][1] == {**base, **views, "disagreement": "thoughtful_disagreement view"}
    assert agents.order[-2:] == [DISAGREEMENT, TRADER] and set(agents.order[:3]) == set(ANALYSTS)


def test_an_analyst_error_stops_the_iteration_before_the_trader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents["inflation_agent"].error = AgentError("boom")

    with pytest.raises(AgentError, match="boom"):
        strategy.on_trading_iteration()

    assert agents[TRADER].runs == []


def test_initialize_fails_without_a_news_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch, news=False)

    with pytest.raises(ConfigurationError, match="news provider"):
        strategy.initialize()


def test_backtesting_preloads_the_universe_and_spy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: seen.update(kwargs))

    strategy.run_backtesting()

    assert [a.symbol for a in seen["preload_assets"]] == [*UNIVERSE, "SPY"]
    assert seen["timestep"] == "day" and seen["agent_telemetry"] is True


def test_the_trader_has_no_unguarded_order_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()

    trader = set(_tools(agents)[TRADER])

    assert "submit_order" in trader
    assert not {"close_position", "sell_all"} & trader


def test_disagreement_and_trader_run_on_the_strategy_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()

    strategy.on_trading_iteration()

    assert agents[DISAGREEMENT].thread == threading.get_ident()
    assert agents[TRADER].thread == threading.get_ident()

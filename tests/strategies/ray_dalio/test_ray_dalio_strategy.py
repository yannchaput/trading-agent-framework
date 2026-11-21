from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.ray_dalio import RayDalioStrategy
from trading_agent_framework.strategies.ray_dalio.agent_ray_dalio import _size
from trading_agent_framework.strategies.ray_dalio.prompts import ANALYSTS, CONTEXT, DISAGREEMENT, SYSTEM_PROMPTS, TASK_PROMPTS, UNIVERSE
from trading_agent_framework.utils.errors import AgentError, ConfigurationError

_NAMES = (*ANALYSTS, DISAGREEMENT)
_TOOLS = ("get_fred_snapshot", "get_fred_latest", "get_fred_series", "search_news", "get_account_balance", "get_positions", "get_last_price", "submit_basket")


class _FakeHandle:
    def __init__(self, name: str, order: list[str]) -> None:
        self.name, self.order = name, order
        self.runs: list[tuple[str, dict[str, Any]]] = []
        self.thread: int | None = None
        self.error: Exception | None = None
        self.tools: dict[str, Callable[..., Any]] = {}
        self.script: Callable[[dict[str, Callable[..., Any]]], None] | None = None  # what the agent does with its tools

    def run(self, task_prompt: str, *, context: dict[str, Any]) -> AgentRunResult:
        self.runs.append((task_prompt, context))
        self.thread = threading.get_ident()
        self.order.append(self.name)
        if self.error is not None:
            raise self.error
        if self.script is not None:
            self.script(self.tools)
        return AgentRunResult(output=f"{self.name} view", tool_calls=[])


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.order: list[str] = []
        self.handles = {name: _FakeHandle(name, self.order) for name in _NAMES}

    def create(self, **kwargs: Any) -> _FakeHandle:
        self.created.append(kwargs)
        handle = self.handles[kwargs["name"]]
        handle.tools = {tool.__name__: tool for tool in kwargs["tools"]}
        return handle

    def __getitem__(self, name: str) -> _FakeHandle:
        return self.handles[name]


def _strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, news: bool = True, held: dict[str, int] | None = None) -> tuple[RayDalioStrategy, _FakeAgents]:
    """A paper strategy with fake agents. The book is worth 10,000; every fund costs 50 and SHV 100; `held` maps a symbol to shares."""
    monkeypatch.setenv("FRED_API_KEY", "key")
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test test@example.com")
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="ray_dalio")
    broker.news = FakeNewsProvider() if news else None
    broker.last_prices = {symbol: Decimal(50) for symbol in UNIVERSE} | {"SHV": Decimal(100)}
    broker.positions = [Position(strategy_name="ray_dalio", asset=Asset(symbol), quantity=Decimal(shares), side=PositionSide.LONG) for symbol, shares in (held or {}).items()]
    invested = sum(shares * 50 for shares in (held or {}).values())
    broker.account = AccountBalances(cash=Decimal(10_000 - invested), portfolio_value=Decimal(10_000), buying_power=Decimal(1_000_000))
    strategy = RayDalioStrategy(broker, mode=TradingMode.PAPER, project_root=tmp_path)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tools(agents: _FakeAgents) -> dict[str, dict[str, Any]]:
    return {c["name"]: {t.__name__: t for t in c["tools"]} for c in agents.created}


def _submits(basket: dict[str, float]) -> Callable[[dict[str, Callable[..., Any]]], None]:
    """An agent script: call `submit_basket` with `basket`."""

    def run(tools: dict[str, Callable[..., Any]]) -> None:
        tools["submit_basket"](weights=basket, reason="test")

    return run


def _bought(strategy: RayDalioStrategy) -> set[str]:
    return {order.asset.symbol for order in cast(FakeBroker, strategy.broker).submitted if order.side.value == "buy"}


def _sold(strategy: RayDalioStrategy) -> set[str]:
    return {order.asset.symbol for order in cast(FakeBroker, strategy.broker).submitted if order.side.value == "sell"}


def test_initialize_creates_the_four_agents_with_lumibots_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)

    strategy.initialize()

    assert [c["name"] for c in agents.created] == list(_NAMES)
    assert all(c["system_prompt"] == SYSTEM_PROMPTS[c["name"]] for c in agents.created)
    assert strategy.sleeptime == "1D"


def test_only_the_disagreement_agent_can_submit_a_basket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    tools = _tools(agents)

    assert [name for name, agent_tools in tools.items() if "submit_basket" in agent_tools] == [DISAGREEMENT]
    assert not any("submit_order" in agent_tools for agent_tools in tools.values())


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


def test_an_iteration_runs_analysts_then_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert agents.order[-1] == DISAGREEMENT and set(agents.order[:3]) == set(ANALYSTS)


def test_an_analyst_error_stops_the_iteration_before_any_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents["inflation_agent"].error = AgentError("boom")
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})

    with pytest.raises(AgentError, match="boom"):
        strategy.on_trading_iteration()

    assert agents[DISAGREEMENT].runs == [] and cast(FakeBroker, strategy.broker).submitted == []


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


def test_disagreement_and_rebalance_run_on_the_strategy_thread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})
    threads: list[int] = []
    rebalance = strategy._rebalancer.rebalance
    monkeypatch.setattr(strategy._rebalancer, "rebalance", lambda target: (threads.append(threading.get_ident()), rebalance(target))[1])

    strategy.on_trading_iteration()

    assert agents[DISAGREEMENT].thread == threading.get_ident()
    assert threads == [threading.get_ident()]


def test_a_holding_the_basket_leaves_out_is_sold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch, held={"TMF": 40})  # 2,000 of TMF
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})

    strategy.on_trading_iteration()

    assert _sold(strategy) == {"TMF"}


def test_the_basket_is_bought_and_the_rest_parked_in_shv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})

    strategy.on_trading_iteration()

    assert _bought(strategy) == {"TQQQ", "UPRO", "SHV"} and _sold(strategy) == set()


def test_shv_in_a_basket_is_the_remainder_not_a_stock_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.5, "UPRO": 0.4, "SHV": 0.3})  # with SHV counted as a stock this is 120% and the target is refused

    strategy.on_trading_iteration()

    assert _bought(strategy) == {"TQQQ", "UPRO", "SHV"}


def test_an_accepted_basket_and_its_reason_are_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})

    with caplog.at_level(logging.INFO):
        strategy.on_trading_iteration()

    assert any("basket" in text and "TQQQ" in text and "test" in text for text in caplog.messages)  # `_submits` gives the reason "test"


def test_no_basket_means_no_order_and_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch, held={"TMF": 40})
    strategy.initialize()

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert cast(FakeBroker, strategy.broker).submitted == [] and "no basket submitted" in caplog.text


def test_a_basket_is_not_reused_the_next_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, agents = _strategy(tmp_path, monkeypatch)
    strategy.initialize()
    agents[DISAGREEMENT].script = _submits({"TQQQ": 0.4, "UPRO": 0.3, "SHV": 0.3})
    strategy.on_trading_iteration()
    first_session = len(cast(FakeBroker, strategy.broker).submitted)
    agents[DISAGREEMENT].script = None  # the agent submits nothing this time

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert len(cast(FakeBroker, strategy.broker).submitted) == first_session and "no basket submitted" in caplog.text


def _order_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "order" in record.getMessage()]


def test_a_new_order_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch)
    order = strategy.create_order("TQQQ", Decimal("7.842749"), "sell")

    with caplog.at_level(logging.INFO):
        strategy.on_new_order(order)

    (line,) = _order_lines(caplog)
    assert "new order: sell 7.842749 TQQQ" in line and order.identifier in line


def test_a_partial_fill_is_logged_with_its_size_price_and_the_orders_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch)
    order = strategy.create_order("UPRO", 10, "buy")

    with caplog.at_level(logging.INFO):
        strategy.on_partially_filled_order(None, order, Decimal("77.5"), Decimal(4), 1)

    (line,) = _order_lines(caplog)
    assert "order partially filled: buy 4 of 10 UPRO @ $77.50" in line and order.identifier in line


def test_a_full_fill_is_logged_with_its_size_price_and_the_orders_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path, monkeypatch)
    order = strategy.create_order("SQQQ", Decimal("7.842749"), "sell")

    with caplog.at_level(logging.INFO):
        strategy.on_filled_order(None, order, Decimal("96.84"), Decimal("7.842749"), 1)

    (line,) = _order_lines(caplog)
    assert "order filled: sell 7.842749 of 7.842749 SQQQ @ $96.84" in line and order.identifier in line


def test_a_notional_order_has_no_share_count_to_log() -> None:
    assert _size(None) == "?"
    assert _size(Decimal("10.500")) == "10.5"

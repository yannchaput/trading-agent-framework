"""SectorPodsStrategy: weekday gate, startup refusals, agent wiring, escaped errors and the factory."""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.sector_pods import SectorPodsStrategy
from trading_agent_framework.strategies.sector_pods.agent_sector_pods import _size
from trading_agent_framework.strategies.sector_pods.pods import PODS
from trading_agent_framework.strategies.sector_pods.store import WeekState
from trading_agent_framework.utils.errors import BrokerError, FatalStrategyError
from trading_agent_framework.utils.strategy_factory import Strategies, build_strategy

TUESDAY, WEDNESDAY = et(2026, 11, 10, 12), et(2026, 11, 11, 12)


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()


class _FakeGraph:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result, self.invocations = result, 0

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.invocations += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _strategy(tmp_path: Path, now: datetime = TUESDAY, *, news: bool = True, mode: TradingMode = TradingMode.BACKTESTING) -> tuple[SectorPodsStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(now), strategy_name="sector_pods")
    broker.news = FakeNewsProvider() if news else None
    strategy = SectorPodsStrategy(broker, mode=mode, project_root=tmp_path, fred_client_factory=lambda: object())
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def test_skips_other_weekdays(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, WEDNESDAY)
    strategy.initialize()
    graph = strategy._graph = _FakeGraph({"abandoned": None})

    strategy.on_trading_iteration()

    assert graph.invocations == 0


def test_completed_week_not_run_again(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    graph = strategy._graph = _FakeGraph({"abandoned": None})
    assert strategy._store is not None
    strategy._store.save(WeekState(last_completed="2026-11-10"))

    strategy.on_trading_iteration()

    assert graph.invocations == 0


def test_initialize_without_news_provider_is_fatal(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, news=False)

    with pytest.raises(FatalStrategyError, match="news provider"):
        strategy.initialize()


def test_initialize_without_fred_key_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    strategy, _ = _strategy(tmp_path)
    strategy._fred_client_factory = None

    with pytest.raises(FatalStrategyError):
        strategy.initialize()


def test_each_pod_gets_its_own_news_and_macro_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    by_name = {created["name"]: created for created in agents.created}
    assert list(by_name) == [*(pod.key for pod in PODS), "risk_manager", "portfolio_manager"] and len(by_name) == 7
    tools = {name: {tool.__name__: tool for tool in created["tools"]} for name, created in by_name.items()}
    for tool_name in ("search_news", "get_fred_series"):
        objects = [tools[pod.key][tool_name] for pod in PODS]
        assert all(a is not b for i, a in enumerate(objects) for b in objects[i + 1 :])
    assert set(tools["portfolio_manager"]) == {"search_memory", "remember_lesson", "count_characters", "submit_target_weights"}
    assert set(tools["risk_manager"]) == {"count_characters", "submit_risk_review"}
    assert all("count_characters" in agent_tools for agent_tools in tools.values())
    assert by_name["tech_pod"]["final_tools"] == ["submit_pod_view"]
    assert by_name["tech_pod"]["budget"].exempt == ("submit_pod_view", "count_characters")
    # remember_lesson is outside the memory budget: three searches used to spend it, so no lesson was ever stored
    assert by_name["portfolio_manager"]["budget"].exempt == ("submit_target_weights", "count_characters", "remember_lesson")
    assert by_name["portfolio_manager"]["final_tools"] == ["submit_target_weights"]


def test_every_submit_tool_refuses_an_over_long_text_through_its_own_gate(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()
    assert strategy._store is not None
    parts = strategy._graph_parts(strategy._store)  # a fresh set of agents, so each node's recorder can be armed here
    for node in (*parts.pods.values(), parts.risk_manager, parts.portfolio_manager):
        node.recorder.arm({})

    tools = {created["name"]: {tool.__name__: tool for tool in created["tools"]} for created in agents.created}  # the latest agent of a name wins
    params = strategy.settings
    refusal = tools["risk_manager"]["submit_risk_review"](objections=[], summary="x" * (params.summary_max_chars + 1))
    assert refusal["error"].startswith(f"too long: summary has {params.summary_max_chars + 1} characters (limit {params.summary_max_chars}).")
    assert "call submit_risk_review again" in refusal["error"]
    long_book = [{"symbol": "XLK", "weight": 0.1, "reason": "r" * (params.reason_max_chars + 1)}]
    assert "reason for XLK has" in tools["portfolio_manager"]["submit_target_weights"](weights=long_book, overrides="", summary="s")["error"]
    long_scores = [{"symbol": "XLK", "score": 1, "reason": "r" * (params.reason_max_chars + 1)}]
    assert "call submit_pod_view again" in tools["tech_pod"]["submit_pod_view"](scores=long_scores, sector_view="v")["error"]


def test_an_error_escaping_the_graph_is_an_abandoned_week_that_keeps_the_previous_views(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    assert strategy._store is not None
    strategy._store.save(WeekState(last_completed="2026-11-03", previous={"date": "2026-11-03"}))
    graph = strategy._graph = _FakeGraph(BrokerError("holdings unavailable"))

    strategy.on_trading_iteration()

    saved = strategy._store.load()
    assert graph.invocations == 1 and saved.abandoned_streak == 1
    assert saved.previous == {"date": "2026-11-03"} and saved.last_completed == "2026-11-03"


def test_three_escaped_errors_in_a_row_abort_a_backtest(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    strategy._graph = _FakeGraph(BrokerError("down"))

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    with pytest.raises(FatalStrategyError, match="3 reviews abandoned"):
        strategy.on_trading_iteration()


def test_three_escaped_errors_do_not_abort_a_paper_run(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.PAPER)
    strategy.initialize()
    graph = strategy._graph = _FakeGraph(BrokerError("down"))

    for _ in range(4):
        strategy.on_trading_iteration()

    assert strategy._store is not None and strategy._store.load().abandoned_streak == 4 and graph.invocations == 4


def _order_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "order" in record.getMessage()]


def test_a_new_order_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path)
    order = strategy.create_order("XLK", Decimal("7.842749"), "sell")

    with caplog.at_level(logging.INFO):
        strategy.on_new_order(order)

    (line,) = _order_lines(caplog)
    assert "new order: sell 7.842749 XLK" in line and order.identifier in line


def test_a_partial_fill_is_logged_with_its_size_price_and_the_orders_size(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path)
    order = strategy.create_order("XLV", 10, "buy")

    with caplog.at_level(logging.INFO):
        strategy.on_partially_filled_order(None, order, Decimal("77.5"), Decimal(4), 1)

    (line,) = _order_lines(caplog)
    assert "order partially filled: buy 4 of 10 XLV @ $77.50" in line and order.identifier in line


def test_a_full_fill_is_logged_with_its_size_price_and_the_orders_size(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path)
    order = strategy.create_order("XLK", Decimal("7.842749"), "sell")

    with caplog.at_level(logging.INFO):
        strategy.on_filled_order(None, order, Decimal("96.84"), Decimal("7.842749"), 1)

    (line,) = _order_lines(caplog)
    assert "order filled: sell 7.842749 of 7.842749 XLK @ $96.84" in line and order.identifier in line


def test_a_notional_order_has_no_share_count_to_log() -> None:
    assert _size(None) == "?"
    assert _size(Decimal("10.500")) == "10.5"


def test_factory_builds_sector_pods() -> None:
    strategy = build_strategy(Strategies.SECTOR_PODS, FakeBroker(FakeClock(TUESDAY), strategy_name="sector_pods"), TradingMode.BACKTESTING)

    assert isinstance(strategy, SectorPodsStrategy)

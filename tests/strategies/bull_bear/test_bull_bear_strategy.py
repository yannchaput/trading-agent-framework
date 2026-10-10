from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.bull_bear import BullBearStrategy
from trading_agent_framework.strategies.bull_bear.parameters import BullBearParams
from trading_agent_framework.strategies.bull_bear.pipeline import ReviewOutcome
from trading_agent_framework.strategies.bull_bear.state import BullBearState, StateStore, state_path
from trading_agent_framework.utils.errors import BrokerError, FatalStrategyError
from trading_agent_framework.utils.strategy_factory import Strategies

UNIVERSE = ["AAA", "BBB", "CCC", "SHV"]
TUESDAY, MONDAY = et(2026, 10, 6, 12), et(2026, 10, 5, 12)


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()

    def __getitem__(self, name: str) -> object:
        return object()


class _NoBars:
    def bars(self, symbols, today):  # noqa: ANN001, ANN201
        raise AssertionError("not called in these tests")


class _FakePipeline:
    def __init__(self, outcomes: list[ReviewOutcome], *, completed: bool = False) -> None:
        self.outcomes = list(outcomes)
        self.completed = completed
        self.runs = 0

    def completed_today(self) -> bool:
        return self.completed

    def run(self) -> ReviewOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, now: datetime = TUESDAY, **kwargs: Any) -> tuple[BullBearStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(now), strategy_name="bull_bear")
    broker.news = FakeNewsProvider()  # the researcher's news tool needs a source (`initialize` checks)
    strategy = BullBearStrategy(broker, mode=mode, universe=UNIVERSE, project_root=tmp_path, bars_source=_NoBars(), sector_of=lambda symbol: "Technology", **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


def test_it_is_registered_as_bull_bear() -> None:
    assert Strategies("bull_bear") is Strategies.BULL_BEAR


def test_it_iterates_every_session_at_noon_and_never_scores_the_parking_symbol(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)

    assert strategy.sleeptime == "1D" and strategy.iteration_start_time == time(12, 0)
    assert strategy.universe == ["AAA", "BBB", "CCC"]


def test_initialize_creates_the_four_agents_with_only_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    tools = {created["name"]: _tool_names(created) for created in agents.created}
    assert tools == {
        "researcher": {"search_news", "get_income_statement", "get_balance_sheet", "submit_note"},
        "bull": {"submit_bull_case"},
        "bear": {"submit_bear_case"},
        "judge": {"submit_picks"},
    }


def test_only_the_researcher_has_a_budget_exempt_tool_and_every_agent_uses_the_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=BullBearParams(agent_temperature=0.1))

    strategy.initialize()

    assert {created["name"]: created.get("exempt_tools") for created in agents.created} == {"researcher": ["submit_note"], "bull": None, "bear": None, "judge": None}
    assert {created["temperature"] for created in agents.created} == {0.1}


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, expected in [(TradingMode.BACKTESTING, BullBearState()), (TradingMode.PAPER, BullBearState(abandoned_streak=2))]:
        store = StateStore(state_path(tmp_path, mode))
        store.save(BullBearState(abandoned_streak=2))
        strategy, _ = _strategy(tmp_path, mode=mode)

        strategy.initialize()

        assert store.load() == expected


def test_a_missing_sec_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy, _ = _strategy(tmp_path)

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_a_broker_without_a_news_source_refuses_to_start(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)
    strategy.broker.news = None  # type: ignore[attr-defined]

    with pytest.raises(FatalStrategyError, match="news provider"):
        strategy.initialize()

    assert agents.created == []


def test_a_news_source_that_cannot_be_built_refuses_to_start_with_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)

    def missing() -> None:
        raise BrokerError("no news source available for backtesting: ALPACA_NEWS_API_KEY is not set")

    monkeypatch.setattr(strategy.broker, "news_provider", missing)

    with pytest.raises(FatalStrategyError, match="ALPACA_NEWS_API_KEY"):
        strategy.initialize()


def test_a_completed_review_of_another_day_does_not_block_todays(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    assert strategy.pipeline is not None
    store = StateStore(state_path(tmp_path, TradingMode.PAPER))
    store.save(BullBearState(last_completed_review="2026-10-06"))

    assert strategy.pipeline.completed_today()  # TUESDAY is 2026-10-06
    store.save(BullBearState(last_completed_review="2026-09-29"))
    assert not strategy.pipeline.completed_today()


def test_the_review_runs_on_tuesdays_only(tmp_path: Path) -> None:
    monday, _ = _strategy(tmp_path, now=MONDAY)
    monday.pipeline = _FakePipeline([ReviewOutcome(True, 0)])  # type: ignore[assignment]
    tuesday, _ = _strategy(tmp_path, now=TUESDAY)
    tuesday.pipeline = _FakePipeline([ReviewOutcome(True, 0)])  # type: ignore[assignment]

    monday.on_trading_iteration()
    tuesday.on_trading_iteration()

    assert (monday.pipeline.runs, tuesday.pipeline.runs) == (0, 1)  # type: ignore[union-attr]


def test_a_review_already_completed_today_is_not_run_again(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.pipeline = _FakePipeline([], completed=True)  # type: ignore[assignment]

    strategy.on_trading_iteration()

    assert strategy.pipeline.runs == 0  # type: ignore[union-attr]


def test_a_backtest_aborts_after_three_abandoned_reviews_in_a_row(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 3)])  # type: ignore[assignment]

    with pytest.raises(FatalStrategyError, match="3 reviews abandoned in a row"):
        strategy.on_trading_iteration()


def test_paper_never_aborts_on_abandoned_reviews(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 5)])  # type: ignore[assignment]

    strategy.on_trading_iteration()  # logs and carries on


def test_run_backtesting_passes_the_class_window_the_daily_yahoo_source_and_the_preloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _ = _strategy(tmp_path, mode=TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["start"] == BullBearStrategy.parameters["backtesting_start"]
    assert captured["data_source"] is YahooBacktestData and captured["timestep"] == "day"
    assert {asset.symbol for asset in captured["preload_assets"]} == {"AAA", "BBB", "CCC", "SHV", "SPY"}
    assert captured["budget"] == Decimal(10000) and captured["warmup_trading_days"] == 300

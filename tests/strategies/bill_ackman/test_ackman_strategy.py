from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.backtesting.time_window import PredefinedWindow, backtest_window
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.bill_ackman import BillAckmanStrategy
from trading_agent_framework.strategies.bill_ackman.fact_sheet import TRADING_DAYS_PER_YEAR
from trading_agent_framework.strategies.bill_ackman.parameters import AckmanParams
from trading_agent_framework.strategies.bill_ackman.pipeline import ReviewOutcome
from trading_agent_framework.strategies.bill_ackman.state import ReviewState, StateStore, state_path
from trading_agent_framework.utils.errors import FatalStrategyError

UNIVERSE = ["AAA", "BBB", "CCC"]


class _FakeAgents:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.created.append(kwargs)
        return object()

    def __getitem__(self, name: str) -> object:
        return object()


class _FakeScreen:
    def run(self, symbols, *, as_of, price_of, top_n=None):  # noqa: ANN001, ANN201
        raise AssertionError("not called in these tests")


class _FakePipeline:
    def __init__(self, outcomes: list[ReviewOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.runs = 0

    def run(self) -> ReviewOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, **kwargs: Any) -> tuple[BillAckmanStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman")
    screen = kwargs.pop("screen", _FakeScreen())
    strategy = BillAckmanStrategy(broker, mode=mode, universe=UNIVERSE, project_root=tmp_path, screen=screen, **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


def test_the_strategy_runs_once_a_day() -> None:
    assert BillAckmanStrategy.sleeptime == "1D"


def test_the_defaults_use_the_bi_month_window_and_the_spec_budget() -> None:
    parameters = BillAckmanStrategy.parameters

    assert (parameters["backtesting_start"], parameters["backtesting_end"]) == backtest_window(PredefinedWindow.BI_MONTH)
    assert (parameters["benchmark_symbol"], parameters["budget"], parameters["warmup_trading_days"]) == ("SPY", 100000, 260)


def test_initialize_creates_the_three_agents_with_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    researcher, short_seller, trader = agents.created
    assert [created["name"] for created in agents.created] == ["researcher", "short_seller", "trader"]
    assert {"get_company_facts", "get_filings", "get_bars", "get_last_price", "submit_ranking"} <= _tool_names(researcher)
    assert {"get_company_facts", "search_news", "get_bars", "submit_verdicts"} <= _tool_names(short_seller)
    assert _tool_names(trader) == {"submit_portfolio"}


def test_no_agent_has_an_order_account_indicator_or_memory_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    forbidden = {"submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all", "get_positions", "get_account_balance", "get_orders", "remember_decision", "search_memory"}
    for created in agents.created:
        assert not forbidden & _tool_names(created)


def test_each_agent_gets_only_its_own_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    submits = [{name for name in _tool_names(created) if name.startswith("submit_")} for created in agents.created]
    assert submits == [{"submit_ranking"}, {"submit_verdicts"}, {"submit_portfolio"}]


def test_every_prompt_requires_english_and_names_its_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    for created, tool in zip(agents.created, ["submit_ranking", "submit_verdicts", "submit_portfolio"], strict=True):
        assert "English" in created["system_prompt"]
        assert tool in created["system_prompt"]
        assert "exactly once" in created["system_prompt"]


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, kept in ((TradingMode.BACKTESTING, False), (TradingMode.PAPER, True)):
        StateStore(state_path(tmp_path, mode)).save(ReviewState(fail_counts={"HHH": 1}))
        strategy, _ = _strategy(tmp_path, mode)

        strategy.initialize()

        assert state_path(tmp_path, mode).exists() is kept


def test_a_missing_sec_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy, _ = _strategy(tmp_path)

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_without_an_injected_screen_the_real_one_is_built_and_needs_the_user_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    strategy = BillAckmanStrategy(FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="bill_ackman"), mode=TradingMode.PAPER, universe=UNIVERSE, project_root=tmp_path)
    strategy._agents = cast(AgentManager, _FakeAgents())

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_a_review_runs_on_every_iteration(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    pipeline = _FakePipeline([ReviewOutcome(True, 0)] * 3)
    strategy.pipeline = pipeline  # type: ignore[assignment]

    for _ in range(3):
        strategy.on_trading_iteration()

    assert pipeline.runs == 3


def test_a_backtest_aborts_after_three_abandoned_reviews_in_a_row(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 1), ReviewOutcome(False, 2), ReviewOutcome(False, 3)])  # type: ignore[assignment]

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    with pytest.raises(FatalStrategyError, match="3 reviews abandoned in a row"):
        strategy.on_trading_iteration()


def test_the_abort_threshold_comes_from_the_parameters(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING, settings=AckmanParams(max_consecutive_abandoned=1))
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, 1)])  # type: ignore[assignment]

    with pytest.raises(FatalStrategyError):
        strategy.on_trading_iteration()


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_never_abort_on_abandoned_reviews(tmp_path: Path, mode: TradingMode) -> None:
    strategy, _ = _strategy(tmp_path, mode)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([ReviewOutcome(False, streak) for streak in range(1, 8)])  # type: ignore[assignment]

    for _ in range(7):
        strategy.on_trading_iteration()


def test_the_review_log_lives_in_the_run_directory_once_there_is_a_run_id(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    assert strategy._review_log_path() is None
    strategy.run_id = "2026-09-14_100000_backtesting"

    assert strategy._review_log_path() == tmp_path / "logs" / "bill_ackman" / "backtesting" / "2026-09-14_100000_backtesting" / "reviews.jsonl"


def test_run_backtesting_passes_the_window_the_daily_yahoo_source_and_the_preloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: seen.update(kwargs))

    strategy.run_backtesting()

    start, end = backtest_window(PredefinedWindow.BI_MONTH)
    assert (seen["start"], seen["end"]) == (start, end)
    assert seen["data_source"] is YahooBacktestData
    assert seen["timestep"] == "day" and seen["benchmark"] == "SPY" and seen["warmup_trading_days"] == 260
    assert seen["budget"] == Decimal("100000") and seen["agent_telemetry"] is True
    assert [asset.symbol for asset in seen["preload_assets"]] == ["AAA", "BBB", "CCC", "SHV", "SPY"]


def test_the_warmup_covers_a_year_of_daily_bars_so_the_12_month_return_exists_from_day_one() -> None:
    assert BillAckmanStrategy.parameters["warmup_trading_days"] > TRADING_DAYS_PER_YEAR

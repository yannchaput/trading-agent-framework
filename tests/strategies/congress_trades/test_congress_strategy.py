from __future__ import annotations

from datetime import UTC, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.congress_trades import CongressTradesStrategy
from trading_agent_framework.strategies.congress_trades.parameters import CongressParams
from trading_agent_framework.strategies.congress_trades.pipeline import RunOutcome
from trading_agent_framework.strategies.congress_trades.state import CongressState, StateStore, state_path
from trading_agent_framework.utils.errors import ConfigurationError, FatalStrategyError


class _FakeAgents:
    def __init__(self, error: Exception | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self.error = error

    def create(self, **kwargs: Any) -> object:
        if self.error is not None:
            raise self.error
        self.created.append(kwargs)
        return object()

    def __getitem__(self, name: str) -> object:
        return object()


class _FakeSource:
    def known(self, as_of: Any) -> Any:
        raise AssertionError("not called in these tests")

    def new_since(self, known: Any, processed: Any) -> Any:
        raise AssertionError("not called in these tests")


class _FakePipeline:
    def __init__(self, outcomes: list[RunOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.runs = 0

    def run(self) -> RunOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, **kwargs: Any) -> tuple[CongressTradesStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="congress_trades")
    kwargs.setdefault("source", _FakeSource())
    strategy = CongressTradesStrategy(broker, mode=mode, project_root=tmp_path, **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


# --- cadence and agents ----------------------------------------------------------------------------------


def test_the_strategy_checks_once_every_session_at_ten_in_the_morning() -> None:
    assert CongressTradesStrategy.sleeptime == "1D"
    assert CongressTradesStrategy.iteration_start_time == time(10, 0)


def test_initialize_creates_the_three_agents_with_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    assert [created["name"] for created in agents.created] == ["researcher", "portfolio_manager", "trader"]
    researcher, portfolio, trader = agents.created
    assert _tool_names(researcher) == {"list_filings", "read_filing", "get_last_price", "submit_holdings"}
    assert _tool_names(portfolio) == {"submit_target"}
    assert _tool_names(trader) == {"get_account_balance", "get_positions", "get_last_price", "place_order", "check_orders", "submit_trade_report"}


def test_only_the_trading_agent_can_place_orders_and_nobody_gets_an_unbounded_order_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    can_order = [created["name"] for created in agents.created if "place_order" in _tool_names(created)]
    assert can_order == ["trader"]
    forbidden = {"submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all", "remember_decision", "search_memory", "search_news"}
    for created in agents.created:
        assert not forbidden & _tool_names(created)


def test_each_agent_gets_only_its_own_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    submits = [{name for name in _tool_names(created) if name.startswith("submit_")} for created in agents.created]
    assert submits == [{"submit_holdings"}, {"submit_target"}, {"submit_trade_report"}]


def test_every_prompt_requires_english_and_names_its_submit_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    for created, tool in zip(agents.created, ["submit_holdings", "submit_target", "submit_trade_report"], strict=True):
        assert "English" in created["system_prompt"] and tool in created["system_prompt"] and "exactly once" in created["system_prompt"]


def test_the_followed_member_comes_from_the_settings(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=CongressParams(politician="Jane Doe"))

    strategy.initialize()

    assert "Jane Doe" in agents.created[0]["system_prompt"] and "Jane Doe" in agents.created[1]["system_prompt"]


def test_every_agent_is_sampled_at_the_configured_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=CongressParams(agent_temperature=0.2))

    strategy.initialize()

    assert [created["temperature"] for created in agents.created] == [0.2, 0.2, 0.2]


# --- refusing to start -----------------------------------------------------------------------------------


def test_a_missing_congress_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONGRESS_USER_AGENT", raising=False)
    strategy, _ = _strategy(tmp_path, source=None)

    with pytest.raises(FatalStrategyError, match="CONGRESS_USER_AGENT"):
        strategy.initialize()


def test_with_a_user_agent_the_real_clerk_source_is_built_without_any_network_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONGRESS_USER_AGENT", "TestApp test@example.com")
    strategy, agents = _strategy(tmp_path, source=None)

    strategy.initialize()

    assert len(agents.created) == 3 and strategy.pipeline is not None


def test_no_llm_model_configured_refuses_to_start(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy._agents = cast(AgentManager, _FakeAgents(ConfigurationError("LLM_MODEL is not set")))

    with pytest.raises(FatalStrategyError, match="LLM_MODEL"):
        strategy.initialize()


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, kept in ((TradingMode.BACKTESTING, False), (TradingMode.PAPER, True)):
        StateStore(state_path(tmp_path, mode)).save(CongressState(processed=["10075701"]))
        strategy, _ = _strategy(tmp_path, mode)

        strategy.initialize()

        assert state_path(tmp_path, mode).exists() is kept


# --- the daily check -------------------------------------------------------------------------------------


def test_a_check_runs_on_every_iteration(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy.initialize()
    pipeline = _FakePipeline([RunOutcome(True, 0)] * 3)
    strategy.pipeline = pipeline  # type: ignore[assignment]

    for _ in range(3):
        strategy.on_trading_iteration()

    assert pipeline.runs == 3


def test_a_backtest_aborts_after_three_abandoned_runs_in_a_row(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([RunOutcome(False, 1), RunOutcome(False, 2), RunOutcome(False, 3)])  # type: ignore[assignment]

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    with pytest.raises(FatalStrategyError, match="3 runs abandoned in a row"):
        strategy.on_trading_iteration()


def test_the_abort_threshold_comes_from_the_parameters(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING, settings=CongressParams(max_consecutive_abandoned=1))
    strategy.initialize()
    strategy.pipeline = _FakePipeline([RunOutcome(False, 1)])  # type: ignore[assignment]

    with pytest.raises(FatalStrategyError):
        strategy.on_trading_iteration()


@pytest.mark.parametrize("mode", [TradingMode.PAPER, TradingMode.LIVE])
def test_paper_and_live_never_abort_on_abandoned_runs(tmp_path: Path, mode: TradingMode) -> None:
    strategy, _ = _strategy(tmp_path, mode)
    strategy.initialize()
    strategy.pipeline = _FakePipeline([RunOutcome(False, streak) for streak in range(1, 8)])  # type: ignore[assignment]

    for _ in range(7):
        strategy.on_trading_iteration()


def test_the_run_log_lives_in_the_run_directory_once_there_is_a_run_id(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    assert strategy._run_log_path() is None
    strategy.run_id = "2026-09-14_100000_backtesting"

    assert strategy._run_log_path() == tmp_path / "logs" / "congress_trades" / "backtesting" / "2026-09-14_100000_backtesting" / "runs.jsonl"


# --- backtesting -----------------------------------------------------------------------------------------


def test_run_backtesting_passes_the_class_window_the_daily_yahoo_source_and_only_the_benchmark_preloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: seen.update(kwargs))

    strategy.run_backtesting()

    parameters = CongressTradesStrategy.parameters
    assert (seen["start"], seen["end"]) == (parameters["backtesting_start"], parameters["backtesting_end"])
    assert seen["data_source"] is YahooBacktestData
    assert seen["timestep"] == "day" and seen["benchmark"] == "SPY" and seen["warmup_trading_days"] == 10
    assert seen["budget"] == Decimal(str(parameters["budget"])) and seen["agent_telemetry"] is True
    assert [asset.symbol for asset in seen["preload_assets"]] == ["SPY"]  # the traded tickers are known only once the filings are read


def test_the_default_backtest_window_is_timezone_aware_and_ends_in_the_past() -> None:
    """Only what any window must satisfy: its length is a choice that changes (`backtest_window(...)` in the class parameters)."""
    start, end = CongressTradesStrategy.parameters["backtesting_start"], CongressTradesStrategy.parameters["backtesting_end"]

    assert start.tzinfo is not None and end.tzinfo is not None  # a backtest window must be timezone-aware
    assert start < end <= datetime.now(UTC)

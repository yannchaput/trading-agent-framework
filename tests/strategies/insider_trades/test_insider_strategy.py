from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.insider_trades import InsiderTradesStrategy
from trading_agent_framework.strategies.insider_trades.parameters import InsiderParams
from trading_agent_framework.strategies.insider_trades.pipeline import RunOutcome
from trading_agent_framework.strategies.insider_trades.state import InsiderState, StateStore, state_path
from trading_agent_framework.utils.clock import MARKET_TZ
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
    loaded_on = None

    def load(self, *, as_of: Any, since: Any) -> Any:
        raise AssertionError("not called in these tests")

    def known(self, as_of: Any) -> Any:
        raise AssertionError("not called in these tests: initialize must not read the filings")


class _FakePipeline:
    def __init__(self, outcomes: list[RunOutcome]) -> None:
        self.outcomes = list(outcomes)
        self.runs = 0

    def run(self) -> RunOutcome:
        self.runs += 1
        return self.outcomes.pop(0)


def _strategy(tmp_path: Path, mode: TradingMode = TradingMode.PAPER, **kwargs: Any) -> tuple[InsiderTradesStrategy, _FakeAgents]:
    broker = FakeBroker(FakeClock(et(2026, 9, 14, 10)), strategy_name="insider_trades")
    kwargs.setdefault("source", _FakeSource())
    strategy = InsiderTradesStrategy(broker, mode=mode, project_root=tmp_path, **kwargs)
    agents = _FakeAgents()
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents


def _tool_names(created: dict[str, Any]) -> set[str]:
    return {tool.__name__ for tool in created["tools"]}


# --- cadence and agents ----------------------------------------------------------------------------------


def test_the_strategy_checks_once_every_session_at_ten_in_the_morning() -> None:
    assert InsiderTradesStrategy.sleeptime == "1D"
    assert InsiderTradesStrategy.iteration_start_time == time(10, 0)


def test_initialize_creates_the_three_agents_with_their_own_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    assert [created["name"] for created in agents.created] == ["researcher", "trader", "trader_retry"]
    researcher, trader, retry = agents.created
    assert _tool_names(researcher) == {"get_insider_trades", "read_form4", "submit_note"}
    trading = {"get_account_balance", "get_positions", "get_last_price", "place_order", "check_orders", "submit_trade_report"}
    assert _tool_names(trader) == trading | {"submit_target"}
    assert _tool_names(retry) == trading


def test_only_the_two_trading_agents_can_place_orders_and_nobody_gets_an_unbounded_order_tool(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    can_order = [created["name"] for created in agents.created if "place_order" in _tool_names(created)]
    assert can_order == ["trader", "trader_retry"]
    forbidden = {"submit_order", "cancel_order", "cancel_open_orders", "close_position", "sell_all", "remember_decision", "search_memory", "search_news", "get_last_price_researcher"}
    for created in agents.created:
        assert not forbidden & _tool_names(created)
    assert not {"get_last_price", "get_positions", "get_account_balance"} & _tool_names(agents.created[0])  # the researcher has no market-data or account tool


def test_only_the_trader_can_submit_a_target(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    submits = [{name for name in _tool_names(created) if name.startswith("submit_")} for created in agents.created]
    assert submits == [{"submit_note"}, {"submit_target", "submit_trade_report"}, {"submit_trade_report"}]


def test_every_prompt_requires_english_and_names_its_submit_tools(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path)

    strategy.initialize()

    for created in agents.created:
        text = created["system_prompt"]
        assert "English" in text and "exactly once" in text
        for tool in (name for name in _tool_names(created) if name.startswith("submit_")):
            assert tool in text


def test_every_agent_is_sampled_at_the_configured_temperature(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=InsiderParams(agent_temperature=0.2))

    strategy.initialize()

    assert [created["temperature"] for created in agents.created] == [0.2, 0.2, 0.2]


def test_the_trader_prompt_takes_its_numbers_from_the_settings(tmp_path: Path) -> None:
    strategy, agents = _strategy(tmp_path, settings=InsiderParams(watchlist=("AAA", "BBB", "CCC", "DDD"), max_total_weight=0.8, min_weight=0.05, max_position_weight=0.3))

    strategy.initialize()

    text = agents.created[1]["system_prompt"]
    assert "between 0.05 and 0.3" in text and "at most 0.8" in text and "0.2000 each" in text


def test_initialize_reads_no_filing(tmp_path: Path) -> None:
    """The source's `known` raises in these tests: nothing may call it before the pipeline has loaded the source."""
    strategy, _ = _strategy(tmp_path)

    strategy.initialize()

    assert strategy.pipeline is not None


# --- refusing to start -----------------------------------------------------------------------------------


def test_a_missing_sec_user_agent_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    strategy, _ = _strategy(tmp_path, source=None)

    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_with_a_user_agent_the_real_edgar_source_is_built_without_any_network_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")
    strategy, agents = _strategy(tmp_path, source=None)

    strategy.initialize()

    assert len(agents.created) == 3 and strategy.pipeline is not None
    assert (tmp_path / "cache" / "sec").is_dir()


def test_no_llm_model_configured_refuses_to_start(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    strategy._agents = cast(AgentManager, _FakeAgents(ConfigurationError("LLM_MODEL is not set")))

    with pytest.raises(FatalStrategyError, match="LLM_MODEL"):
        strategy.initialize()


def test_a_backtest_starts_with_no_saved_state_and_paper_keeps_it(tmp_path: Path) -> None:
    for mode, kept in ((TradingMode.BACKTESTING, False), (TradingMode.PAPER, True)):
        StateStore(state_path(tmp_path, mode)).save(InsiderState(processed=["0000000000-26-000001"]))
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
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING, settings=InsiderParams(max_consecutive_abandoned=1))
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

    assert strategy._run_log_path() == tmp_path / "logs" / "insider_trades" / "backtesting" / "2026-09-14_100000_backtesting" / "runs.jsonl"


# --- the Form 4 load window ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "since"),
    [
        (datetime(2025, 9, 29, tzinfo=MARKET_TZ), date(2022, 1, 1)),  # the default window
        (datetime(2026, 1, 5, tzinfo=MARKET_TZ), date(2022, 1, 1)),  # a January start still sees December trades: 40 days back is in 2025
        (datetime(2026, 2, 20, tzinfo=MARKET_TZ), date(2023, 1, 1)),  # 40 days back is Jan 11, 2026
    ],
)
def test_a_backtest_reads_history_back_to_the_look_back_of_its_earliest_visible_trade(tmp_path: Path, start: datetime, since: date) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy._backtest_start, strategy._backtest_end = start, datetime(2026, 10, 17, tzinfo=MARKET_TZ)

    as_of, loaded_since = strategy.load_window()

    assert as_of == datetime(2026, 10, 17, tzinfo=MARKET_TZ)
    assert loaded_since == since


def test_paper_and_live_load_as_of_now_and_reach_back_from_today(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.PAPER)

    as_of, since = strategy.load_window()

    assert as_of == et(2026, 9, 14, 10)
    assert since == date(2023, 1, 1)  # 40 days before 2026-09-14 is in 2026, 3 years back


def test_paper_in_early_january_reaches_back_from_the_december_trades(tmp_path: Path) -> None:
    broker = FakeBroker(FakeClock(et(2027, 1, 6, 10)), strategy_name="insider_trades")
    strategy = InsiderTradesStrategy(broker, mode=TradingMode.PAPER, project_root=tmp_path, source=_FakeSource())

    assert strategy.load_window()[1] == date(2023, 1, 1)  # 40 days before 2027-01-06 is in 2026


def test_the_look_back_follows_the_settings(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING, settings=InsiderParams(history_years=5, signal_window_days=60))
    strategy._backtest_start, strategy._backtest_end = datetime(2026, 2, 20, tzinfo=MARKET_TZ), datetime(2026, 10, 17, tzinfo=MARKET_TZ)

    assert strategy.load_window()[1] == date(2020, 1, 1)  # 70 days before 2026-02-20 is Dec 12, 2025


# --- backtesting -----------------------------------------------------------------------------------------


def test_run_backtesting_passes_the_class_window_the_daily_yahoo_source_and_the_watchlist_preloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: seen.update(kwargs))

    strategy.run_backtesting()

    parameters = InsiderTradesStrategy.parameters
    assert (seen["start"], seen["end"]) == (parameters["backtesting_start"], parameters["backtesting_end"])
    assert seen["data_source"] is YahooBacktestData
    assert seen["timestep"] == "day" and seen["benchmark"] == "SPY" and seen["warmup_trading_days"] == 10
    assert seen["budget"] == Decimal(str(parameters["budget"])) and seen["agent_telemetry"] is True
    assert [asset.symbol for asset in seen["preload_assets"]] == [*strategy.settings.watchlist, "SPY"]


def test_run_backtesting_records_the_merged_window_for_the_form_4_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _strategy(tmp_path, TradingMode.BACKTESTING)
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: None)

    strategy.run_backtesting()
    assert strategy._backtest_start == InsiderTradesStrategy.parameters["backtesting_start"]
    assert strategy._backtest_end == InsiderTradesStrategy.parameters["backtesting_end"] + timedelta(days=1)  # a bare-midnight end covers that whole day
    assert strategy.load_window()[1] == date(2022, 1, 1)

    strategy.run_backtesting(start=datetime(2026, 1, 5, tzinfo=MARKET_TZ), end=datetime(2026, 3, 2, 16, tzinfo=MARKET_TZ))
    assert strategy._backtest_start == datetime(2026, 1, 5, tzinfo=MARKET_TZ)
    assert strategy._backtest_end == datetime(2026, 3, 2, 16, tzinfo=MARKET_TZ)


def test_the_default_backtest_window_is_timezone_aware_and_ends_in_the_past() -> None:
    """Only what any window must satisfy: its length is a choice that changes (`backtest_window(...)` in the class parameters)."""
    start, end = InsiderTradesStrategy.parameters["backtesting_start"], InsiderTradesStrategy.parameters["backtesting_end"]

    assert start.tzinfo is not None and end.tzinfo is not None  # a backtest window must be timezone-aware
    assert start < end <= datetime.now(UTC)

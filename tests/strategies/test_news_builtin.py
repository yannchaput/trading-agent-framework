from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.news_builtin.agent_news_binary import MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, BacktestError, ConfigurationError, FatalStrategyError

_START = et(2026, 9, 14, 9, 0)


class _FakeHandle:
    def __init__(self) -> None:
        self.runs: list[tuple[str, object]] = []
        self.error: Exception | None = None
        self.script: list[Exception | None] = []  # per-run outcome (None = success); overrides `error` while non-empty

    def run(self, task_prompt: str, *, context: object = None) -> AgentRunResult:
        self.runs.append((task_prompt, context))
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
            return AgentRunResult(output="Hold SHV.", tool_calls=[])
        if self.error is not None:
            raise self.error
        return AgentRunResult(output="Hold SHV.", tool_calls=[])


class _FakeAgents:
    def __init__(self, handle: _FakeHandle) -> None:
        self.handle = handle
        self.created: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeHandle:
        self.created.append(kwargs)
        return self.handle

    def __getitem__(self, name: str) -> _FakeHandle:
        return self.handle


def _strategy(tmp_path: Path, mode: TradingMode, parameters: dict[str, object] | None = None) -> tuple[NewsBinaryStrategy, _FakeAgents, _FakeHandle]:
    strategy = NewsBinaryStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=mode, project_root=tmp_path, parameters=parameters)
    handle = _FakeHandle()
    agents = _FakeAgents(handle)
    strategy._agents = agents  # ty: ignore[invalid-assignment]
    return strategy, agents, handle


def test_backtest_defaults_match_the_original_strategy() -> None:
    parameters = NewsBinaryStrategy.parameters
    assert parameters["backtesting_start"] == datetime(2025, 1, 1, tzinfo=MARKET_TZ)
    assert parameters["backtesting_end"] == datetime(2026, 8, 14, tzinfo=MARKET_TZ)
    assert parameters["budget"] == 10000
    assert parameters["benchmark_symbol"] == "SPY"


@pytest.mark.parametrize(("mode", "expected"), [(TradingMode.BACKTESTING, "1D"), (TradingMode.PAPER, "1H"), (TradingMode.LIVE, "1H")])
def test_initialize_sets_sleeptime_per_mode(tmp_path: Path, mode: TradingMode, expected: str) -> None:
    strategy, _, _ = _strategy(tmp_path, mode)

    strategy.initialize()

    assert strategy.sleeptime == expected


def test_initialize_creates_the_agent_with_prebuilt_memory_and_news_tools(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)

    strategy.initialize()

    [created] = agents.created
    names = {tool.__name__ for tool in created["tools"]}  # ty: ignore[unresolved-attribute]
    assert created["name"] == NewsBinaryStrategy.AGENT_NAME
    assert {"search_news", "remember_decision", "search_memory", "submit_order", "get_positions", "get_bars", "get_last_price"} <= names
    assert "SHV" in created["system_prompt"]  # ty: ignore[unsupported-operator]


def test_backtest_runs_the_agent_on_the_first_and_every_fifth_iteration(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.strategy_parameters = {**NewsBinaryStrategy.strategy_parameters, "backtest_every_n_iterations": 5}
    strategy.initialize()

    for _ in range(10):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 3  # iterations 1, 5 and 10


def test_paper_runs_the_agent_on_every_iteration(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()

    for _ in range(3):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 3
    _, context = handle.runs[0]
    assert context["current_datetime"] == _START.isoformat()  # ty: ignore[invalid-argument-type, not-subscriptable]


def test_an_agent_error_is_logged_and_does_not_stop_the_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = AgentError("llm exploded")

    with caplog.at_level(logging.ERROR):
        strategy.on_trading_iteration()

    assert "llm exploded" in caplog.text


def test_backtest_fails_fast_after_three_consecutive_agent_errors(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()
    handle.error = AgentError("bad LLM_BASE_URL")

    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    with pytest.raises(FatalStrategyError, match="bad LLM_BASE_URL") as excinfo:
        strategy.on_trading_iteration()

    assert isinstance(excinfo.value.__cause__, AgentError)
    assert MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS == 3
    assert len(handle.runs) == 3


def test_a_dead_llm_aborts_a_real_backtest_instead_of_writing_a_flat_report(tmp_path: Path) -> None:
    sessions = weekday_sessions(date(2026, 1, 5), 6)
    closes = [100.0 + i for i in range(len(sessions))]
    bars = pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in sessions], name="timestamp"),
    )
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    source.set_bars(Asset("SPY"), bars)
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    handle.error = AgentError("llm down")

    with pytest.raises(BacktestError, match="llm down"):
        Strategy.run_backtesting(strategy, start=sessions[0].open - timedelta(hours=1), end=sessions[-1].close, data_source=source, benchmark="SPY")

    assert len(handle.runs) == MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS  # it stopped; it did not grind through all six sessions
    assert list(tmp_path.rglob("metrics.json")) == []  # and left no report that could pass for a real (flat) run


def test_backtest_agent_error_counter_is_reset_by_a_success(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    strategy.initialize()
    handle.script = [AgentError("x"), None, AgentError("x"), AgentError("x")]

    for _ in range(4):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 4


def test_paper_never_raises_on_repeated_agent_errors(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = AgentError("llm down")

    for _ in range(5):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 5


def test_a_configuration_error_propagates(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = ConfigurationError("no model")

    with pytest.raises(ConfigurationError):
        strategy.on_trading_iteration()


def test_a_real_agent_builds_with_every_tool_and_logs_its_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage("Hold SHV.")]))
    monkeypatch.setattr(AgentManager, "_resolve_model", lambda self, model_arg, timeout: model)
    strategy = NewsBinaryStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=TradingMode.PAPER, project_root=tmp_path)

    with caplog.at_level(logging.INFO):
        strategy.initialize()
        strategy.on_trading_iteration()

    assert NewsBinaryStrategy.AGENT_NAME in strategy.agents
    assert "Hold SHV." in caplog.text


def test_run_backtesting_wires_alpaca_data_preload_and_fees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["data_source"] is AlpacaBacktestData
    assert [asset.symbol for asset in captured["preload_assets"]] == ["SPY", "QQQ", "SHV"]  # ty: ignore[not-iterable]
    assert captured["commission"] == Decimal("0.001")
    assert captured["warmup_trading_days"] == 300


def test_system_prompt_sizes_within_cash_and_only_names_real_tools(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    [created] = agents.created
    prompt = str(created["system_prompt"])
    tool_names = {tool.__name__ for tool in created["tools"]}  # ty: ignore[unresolved-attribute]

    assert "95%" in prompt
    assert "get_account_balance" in prompt
    assert "Orders fill on a later bar" in prompt
    # `buying_power` is a multiple of cash on a live margin account, so it may only ever be named as
    # the smaller-of bound alongside cash -- never as the amount to size against on its own.
    assert "SMALLER of 'buying_power' and ('cash' + the proceeds of the sells you submitted in this run)" in prompt
    assert "buying on margin is forbidden" in prompt
    assert "'error'" in prompt
    assert "SHV" in prompt
    mentioned = {"search_memory", "search_news", "get_positions", "get_orders", "remember_decision", "open_thesis", "close_thesis", "get_account_balance"}
    assert all(name in prompt for name in mentioned)
    assert mentioned <= tool_names


def test_system_prompt_rotates_in_one_run_instead_of_waiting_for_the_sell_to_fill(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    # The old rule stranded the portfolio in cash between the sell and the next run's buy, and sized the
    # buy against the settled cash that a sell does not yet fund (7 shares of SHV instead of ~95).
    assert "do not buy in the same run" not in prompt
    assert "the next run completes the rotation" not in prompt
    assert "submit the sell first, then the buy in the same run" in prompt


def test_system_prompt_requires_checking_positions_and_topping_up_an_undersized_holding(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    assert "Every run, call get_positions and get_account_balance" in prompt
    assert "never rely on memory for what you hold" in prompt
    # A leftover from a partial rotation (7 SHV shares = 7% of equity) must not count as "already aligned".
    assert "less than 90% of portfolio_value" in prompt
    assert "buy the shortfall" in prompt
    assert "pct_of_portfolio" in prompt  # the snapshot field that answers "is this holding under 90%?"


def test_the_run_context_carries_a_portfolio_snapshot(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    broker = strategy.broker
    broker.positions = [Position(strategy_name="news_builtin", asset=Asset("SHV"), quantity=Decimal("7"), side=PositionSide.LONG, avg_fill_price=Decimal("103.34"), market_value=Decimal("723.38"))]  # ty: ignore[unresolved-attribute]
    strategy.initialize()

    strategy.on_trading_iteration()

    _, context = handle.runs[0]
    portfolio = context["portfolio"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert portfolio["cash"] == 10000.0
    assert portfolio["portfolio_value"] == 25000.0
    assert portfolio["buying_power"] == 20000.0
    assert portfolio["positions"] == [{"symbol": "SHV", "quantity": 7.0, "side": "long", "avg_fill_price": 103.34, "market_value": 723.38, "pct_of_portfolio": 2.9}]


def test_the_portfolio_snapshot_shows_an_empty_book_as_an_empty_list(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()

    strategy.on_trading_iteration()

    _, context = handle.runs[0]
    assert context["portfolio"]["positions"] == []  # ty: ignore[invalid-argument-type, not-subscriptable]


def test_the_snapshot_prices_a_position_the_broker_reports_without_a_market_value(tmp_path: Path) -> None:
    # BacktestBroker positions carry only quantity and avg_fill_price; the agent must not be left to do
    # quantity x price / equity itself to tell that 7 SHV shares are 7% of the book, not a full rotation.
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    broker = strategy.broker
    broker.positions = [Position(strategy_name="news_builtin", asset=Asset("SHV"), quantity=Decimal("50"), side=PositionSide.LONG, avg_fill_price=Decimal("100"))]  # ty: ignore[unresolved-attribute]
    broker.last_prices["SHV"] = Decimal("100")  # ty: ignore[unresolved-attribute]
    strategy.initialize()

    strategy.on_trading_iteration()

    [position] = handle.runs[0][1]["portfolio"]["positions"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert position["market_value"] == 5000.0
    assert position["pct_of_portfolio"] == 20.0  # 5000 of the fake account's 25000

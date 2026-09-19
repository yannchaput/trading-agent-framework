from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.yahoo import YahooBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.news_builtin import NewsBuiltinStrategy
from trading_agent_framework.strategies.news_builtin.agent_news_builtin import AGENT_NAME
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import AgentError, ConfigurationError

_START = et(2026, 9, 14, 9, 0)


class _FakeHandle:
    def __init__(self) -> None:
        self.runs: list[tuple[str, object]] = []
        self.error: Exception | None = None

    def run(self, task_prompt: str, *, context: object = None) -> AgentRunResult:
        self.runs.append((task_prompt, context))
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


def _strategy(tmp_path: Path, mode: TradingMode) -> tuple[NewsBuiltinStrategy, _FakeAgents, _FakeHandle]:
    strategy = NewsBuiltinStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=mode, project_root=tmp_path)
    handle = _FakeHandle()
    agents = _FakeAgents(handle)
    strategy._agents = agents  # ty: ignore[invalid-assignment]
    return strategy, agents, handle


def test_backtest_defaults_match_the_original_strategy() -> None:
    assert NewsBuiltinStrategy.backtesting_start == datetime(2025, 1, 1, tzinfo=MARKET_TZ)
    assert NewsBuiltinStrategy.backtesting_end == datetime(2026, 4, 1, tzinfo=MARKET_TZ)
    assert NewsBuiltinStrategy.budget == Decimal("10000")
    assert NewsBuiltinStrategy.benchmark_symbol == "SPY"


@pytest.mark.parametrize(("mode", "expected"), [(TradingMode.BACKTESTING, "1D"), (TradingMode.PAPER, "2H"), (TradingMode.LIVE, "2H")])
def test_initialize_sets_sleeptime_per_mode(tmp_path: Path, mode: TradingMode, expected: str) -> None:
    strategy, _, _ = _strategy(tmp_path, mode)

    strategy.initialize()

    assert strategy.sleeptime == expected


def test_initialize_creates_the_agent_with_prebuilt_memory_and_news_tools(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)

    strategy.initialize()

    [created] = agents.created
    names = {tool.__name__ for tool in created["tools"]}  # ty: ignore[unresolved-attribute]
    assert created["name"] == AGENT_NAME
    assert {"search_news", "remember_decision", "search_memory", "submit_order", "get_positions", "get_bars", "get_last_price"} <= names
    assert "SHV" in created["system_prompt"]  # ty: ignore[unsupported-operator]


def test_backtest_runs_the_agent_on_the_first_and_every_fifth_iteration(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
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
    assert context == {"current_datetime": _START.isoformat()}


def test_an_agent_error_is_logged_and_does_not_stop_the_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = AgentError("llm exploded")

    with caplog.at_level(logging.ERROR):
        strategy.on_trading_iteration()

    assert "llm exploded" in caplog.text


def test_a_configuration_error_propagates(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.error = ConfigurationError("no model")

    with pytest.raises(ConfigurationError):
        strategy.on_trading_iteration()


def test_a_real_agent_builds_with_every_tool_and_logs_its_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage("Hold SHV.")]))
    monkeypatch.setattr(AgentManager, "_resolve_model", lambda self, model_arg, timeout: model)
    strategy = NewsBuiltinStrategy(FakeBroker(FakeClock(_START), strategy_name="news_builtin"), mode=TradingMode.PAPER, project_root=tmp_path)

    with caplog.at_level(logging.INFO):
        strategy.initialize()
        strategy.on_trading_iteration()

    assert AGENT_NAME in strategy.agents
    assert "Hold SHV." in caplog.text


def test_run_backtesting_wires_yahoo_data_preload_and_fees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(Strategy, "run_backtesting", lambda self, **kwargs: captured.update(kwargs))
    strategy, _, _ = _strategy(tmp_path, TradingMode.BACKTESTING)

    strategy.run_backtesting()

    assert captured["data_source"] is YahooBacktestData
    assert [asset.symbol for asset in captured["preload_assets"]] == ["SPY", "QQQ", "SHV"]  # ty: ignore[not-iterable]
    assert captured["commission"] == Decimal("0.001")
    assert captured["warmup_trading_days"] == 0

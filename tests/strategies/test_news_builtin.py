from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolCall
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position
from trading_agent_framework.memory.tools import agent_call_context
from trading_agent_framework.strategies.news_builtin import NewsBinaryStrategy
from trading_agent_framework.strategies.news_builtin.agent_news_binary import MAX_CONSECUTIVE_BACKTEST_AGENT_ERRORS
from trading_agent_framework.utils.errors import AgentError, BacktestDataError, BacktestError, BrokerError, ConfigurationError, FatalStrategyError

_START = et(2026, 9, 14, 9, 0)


class _StubNewsProvider:
    def get_news(self, symbols=(), *, start=None, end=None, limit=10, include_content=False):
        if include_content:
            return [{"id": 1, "headline": "h", "content": "full article text"}]
        return []


def _default_result() -> AgentRunResult:
    # A real run always ends with a recorded decision; tests that don't care about that specifically
    # (sleeptime, portfolio snapshot content, regime tracking, error counting, ...) should not
    # accidentally exercise the no-decision retry path just by using the fake's placeholder result.
    return AgentRunResult(
        output="Hold SHV.",
        tool_calls=[ToolCallRecord(name="remember_decision", args={"text": "Hold SHV."}, result='{"id": "decision_default", "kind": "decision", "status": "recorded"}')],
    )


class _FakeHandle:
    def __init__(self) -> None:
        self.runs: list[tuple[str, object]] = []
        self.run_ids: list[str | None] = []
        self.error: Exception | None = None
        self.script: list[Exception | None] = []  # per-run outcome (None = success); overrides `error` while non-empty
        self.result: AgentRunResult | None = None  # overrides the default successful result when set
        self.results: list[AgentRunResult | Exception] = []  # per-call queue (result or raised exception), checked first

    def run(self, task_prompt: str, *, context: object = None, run_id: str | None = None) -> AgentRunResult:
        self.runs.append((task_prompt, context))
        self.run_ids.append(run_id)
        if self.results:
            outcome = self.results.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
            return _default_result()
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return _default_result()


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
    strategy._agents = cast(AgentManager, agents)
    return strategy, agents, handle


def test_backtest_defaults_match_the_original_strategy() -> None:
    parameters = NewsBinaryStrategy.parameters
    assert parameters


@pytest.mark.parametrize(("mode", "expected"), [(TradingMode.BACKTESTING, "3H"), (TradingMode.PAPER, "1H"), (TradingMode.LIVE, "1H")])
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


def test_initialize_wires_remember_decision_and_submit_order_through_the_news_grounding_gate(tmp_path: Path) -> None:
    # Regression: the agent recorded decisions (citing invented report details) or submitted orders
    # without ever calling search_news that run. See NewsBinaryStrategy's grounding.py.
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.broker.news_provider = lambda: _StubNewsProvider()  # type: ignore[method-assign]
    strategy.initialize()
    [created] = agents.created
    tools = {tool.__name__: tool for tool in created["tools"]}  # ty: ignore[unresolved-attribute]

    with agent_call_context(run_id="run-1"):
        premature = tools["remember_decision"](text="KEEP")
        tools["search_news"](symbols="SPY")  # headline-only scan: still not grounded
        headline_only = tools["remember_decision"](text="KEEP")
        tools["search_news"](symbols="SPY", include_content=True)
        after = tools["remember_decision"](text="KEEP")

    assert "error" in premature
    assert "search_news" in premature["error"]
    assert "error" in headline_only
    assert "include_content" in headline_only["error"]
    assert "error" not in after


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


def test_a_run_that_never_records_a_decision_logs_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Regression: a stuck grounding-gate retry loop was observed ending with the model emitting a
    # malformed pseudo tool-call as plain text -- no real remember_decision call ever executes, and
    # the run ends with no error and no warning, invisible outside the raw agent-message DEBUG dump.
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.result = AgentRunResult(
        output="...",
        tool_calls=[ToolCallRecord(name="search_news", args={}, result='{"count": 30, "articles": []}')],
    )

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert "remember_decision" in caplog.text


def test_a_run_where_remember_decision_was_refused_still_logs_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.result = AgentRunResult(
        output="...",
        tool_calls=[ToolCallRecord(name="remember_decision", args={"text": "KEEP"}, result='{"error": "call search_news..."}')],
    )

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert "remember_decision" in caplog.text


def test_a_run_that_records_a_decision_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.result = AgentRunResult(
        output="...",
        tool_calls=[ToolCallRecord(name="remember_decision", args={"text": "KEEP"}, result='{"id": "decision_1", "kind": "decision", "status": "recorded"}')],
    )

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert "remember_decision" not in caplog.text


def test_a_run_that_records_a_decision_is_not_retried(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.result = AgentRunResult(
        output="KEEP.",
        tool_calls=[ToolCallRecord(name="remember_decision", args={"text": "KEEP"}, result='{"id": "decision_1", "kind": "decision", "status": "recorded"}')],
    )

    strategy.on_trading_iteration()

    assert len(handle.runs) == 1


def test_a_run_that_never_records_a_decision_is_retried_once_and_the_retry_can_succeed(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # Regression: the model sometimes places a real trade via submit_order, then just writes a prose
    # summary and stops -- no remember_decision call at all, real or malformed, anywhere in the run.
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.results = [
        AgentRunResult(
            output="Bought SPY at 680.26 to enter bullish regime.",
            tool_calls=[ToolCallRecord(name="submit_order", args={"symbol": "SPY", "quantity": 13, "side": "buy"}, result='{"identifier": "x"}')],
        ),
        AgentRunResult(
            output="Recorded.",
            tool_calls=[ToolCallRecord(name="remember_decision", args={"text": "Bought SPY."}, result='{"id": "decision_2", "kind": "decision", "status": "recorded"}')],
        ),
    ]

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 2
    retry_prompt = handle.runs[1][0]
    assert "remember_decision" in retry_prompt
    assert "Bought SPY at 680.26 to enter bullish regime." in retry_prompt  # carries the first run's own words forward
    # Same logical run: the retry must not need to re-ground itself with another search_news call.
    assert handle.run_ids[0] == handle.run_ids[1]
    assert handle.run_ids[0] is not None
    assert "retry" in caplog.text.lower()


def test_a_retry_that_also_fails_logs_a_final_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.result = AgentRunResult(output="Bought SPY.", tool_calls=[])  # every call, including the retry, skips it

    with caplog.at_level(logging.WARNING):
        strategy.on_trading_iteration()

    assert len(handle.runs) == 2  # the retry was attempted exactly once, not looped
    assert "no decision was recorded" in caplog.text.lower()


def test_a_retry_that_itself_raises_is_logged_like_any_agent_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    handle.results = [
        AgentRunResult(output="Bought SPY.", tool_calls=[]),  # 1st run: no decision recorded
        AgentError("retry server down"),  # retry: raises
    ]

    with caplog.at_level(logging.ERROR):
        strategy.on_trading_iteration()

    assert "retry server down" in caplog.text
    assert len(handle.runs) == 2


def test_a_real_agent_builds_with_every_tool_and_logs_its_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    # Two scripted messages: the run records no decision, so the retry path consumes a second one.
    model = FakeToolCallingChatModel(
        messages=iter(
            [
                AIMessage("Hold SHV."),
                AIMessage(
                    content="",
                    tool_calls=[ToolCall(name="remember_decision", args={"text": "Hold SHV."}, id="call_1")],
                ),
                AIMessage("Recorded."),
            ]
        )
    )
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
    assert captured["commission"] == Decimal(0)  # Alpaca charges no commission on US ETFs
    assert captured["warmup_trading_days"] == 10


def test_system_prompt_sizes_within_cash_and_only_names_real_tools(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    [created] = agents.created
    prompt = str(created["system_prompt"])
    tool_names = {tool.__name__ for tool in created["tools"]}  # ty: ignore[unresolved-attribute]

    assert "95%" in prompt
    assert "get_account_balance" in prompt
    # `buying_power` is a multiple of cash on a live margin account, so it may only ever be named as
    # the smaller-of bound alongside cash -- never as the amount to size against on its own.
    assert "SMALLER of 'buying_power' and ('cash' + the proceeds of the sells you submitted in this run" in prompt
    assert "buying on margin is forbidden" in prompt
    assert "'error'" in prompt
    assert "SHV" in prompt
    mentioned = {"search_memory", "search_news", "get_positions", "get_orders", "remember_decision", "open_thesis", "close_thesis", "get_account_balance"}
    assert all(name in prompt for name in mentioned)
    assert mentioned <= tool_names


def test_system_prompt_tells_the_agent_to_record_once_and_stop(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    # Without it the local model re-called remember_decision (and even submit_order) after success,
    # because the tool result gives it no sign that the run is over.
    assert "Call remember_decision exactly once per run" in prompt
    assert "make no further tool call" in prompt


def test_system_prompt_explains_the_search_news_grounding_requirement(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    assert "remember_decision and submit_order are refused" in prompt
    assert "search_news" in prompt.split("remember_decision and submit_order are refused")[1][:200]


def test_system_prompt_makes_an_order_final_for_the_run(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    # The agent submitted a sell, reconsidered, cancelled it and recorded "no trade" (24 runs), or cancelled
    # and resubmitted the same order; cancel_order now refuses same-run orders, and the prompt says why.
    assert "settle the decision before the first order" in prompt
    assert "never cancel an order you submitted in this run" in prompt


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


def test_system_prompt_does_not_promise_a_sell_funds_the_buy_in_every_mode(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    # Only the backtest broker fills in submission order and credits a pending sell; a live cash account
    # can refuse the buy, and the 'error' rule is what covers that -- so the prompt must not claim otherwise.
    assert "Orders fill on a later bar" not in prompt
    assert "so the sell funds the buy" not in prompt
    assert "is credited toward the buy" in prompt


def test_system_prompt_counts_pending_buys_as_held_and_excludes_refused_sells(tmp_path: Path) -> None:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.initialize()
    prompt = str(agents.created[0]["system_prompt"])

    assert "count an open buy order from get_orders as already held" in prompt
    assert "sells you submitted in this run that were accepted (no 'error')" in prompt
    assert "combined value of the instruments for the regime" in prompt


def test_a_price_lookup_failure_leaves_the_position_unpriced_but_still_runs_the_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # BacktestDataError is a BacktestError, not a BrokerError: uncaught, it would skip the tick without
    # ever reaching the agent-error counter that aborts a dead backtest.
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    broker = strategy.broker
    broker.positions = [Position(strategy_name="news_builtin", asset=Asset("SHV"), quantity=Decimal("50"), side=PositionSide.LONG, avg_fill_price=Decimal("100"))]  # ty: ignore[unresolved-attribute]
    monkeypatch.setattr(broker, "get_last_price", lambda asset: (_ for _ in ()).throw(BacktestDataError("no bars")))
    strategy.initialize()

    strategy.on_trading_iteration()

    [position] = handle.runs[0][1]["portfolio"]["positions"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert position["symbol"] == "SHV"
    assert "market_value" not in position
    assert "pct_of_portfolio" not in position


def test_snapshot_errors_are_namespaced_so_a_failed_positions_call_is_not_read_as_an_empty_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.BACKTESTING)
    broker = strategy.broker
    monkeypatch.setattr(broker, "get_account", lambda: (_ for _ in ()).throw(BacktestDataError("no bars")))
    monkeypatch.setattr(broker, "pull_positions", lambda: (_ for _ in ()).throw(BrokerError("down")))
    strategy.initialize()

    strategy.on_trading_iteration()

    portfolio = handle.runs[0][1]["portfolio"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert portfolio == {"balance_error": "no bars", "positions_error": "down"}


def _hold(symbol: str, quantity: str = "10") -> Position:
    return Position(strategy_name="news_builtin", asset=Asset(symbol), quantity=Decimal(quantity), side=PositionSide.LONG, avg_fill_price=Decimal("100"))


def _prompt(tmp_path: Path, **strategy_parameters: object) -> str:
    strategy, agents, _ = _strategy(tmp_path, TradingMode.PAPER)
    strategy.strategy_parameters = {**NewsBinaryStrategy.strategy_parameters, **strategy_parameters}
    strategy.initialize()
    return str(agents.created[0]["system_prompt"])


def test_system_prompt_keeps_the_current_holding_when_the_news_is_unclear(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path)

    # "negative or unclear evidence means hold <defensive>" made every headline-poor day a rotation into SHV.
    assert "negative or unclear evidence means hold" not in prompt
    assert "Unclear, mixed, stale or no relevant news means KEEP the current holding" in prompt


def test_system_prompt_makes_one_data_print_insufficient_to_leave_the_risk_instruments(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path)

    assert "two INDEPENDENT bearish signals" in prompt
    assert "ONE signal however many headlines cover it" in prompt
    assert "the same bearish call already recorded on the previous run" in prompt
    assert "PENDING bearish flip" in prompt  # what makes the previous-run confirmation findable via search_memory
    assert "one clear bullish signal" in prompt  # the way back is deliberately easier


def test_system_prompt_reads_the_regime_from_the_snapshot(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path)

    assert "current_regime" in prompt
    assert "sessions_in_regime" in prompt


def test_system_prompt_prefers_fresh_on_topic_news_over_stale_or_already_cited_articles(tmp_path: Path) -> None:
    # Regression: the live agent picked an 11-hour-old article over a fresh on-topic headline sitting at
    # the top of the same run's broad scan, then re-cited that same stale article across four straight
    # runs (search_memory already showed it as a prior decision's basis) to keep justifying no trade.
    prompt = _prompt(tmp_path)

    assert "prefer the most recent one that is actually about the regime call" in prompt
    assert "already appears in a decision from search_memory" in prompt
    assert "it is not new evidence" in prompt


def test_system_prompt_names_only_the_configured_symbols(tmp_path: Path) -> None:
    prompt = _prompt(tmp_path, symbols=("VOO", "IWY"), defensive_symbol="BIL", news_symbols="VOO,IWY")

    assert "BIL" in prompt
    assert "VOO or IWY" in prompt
    assert "empty book" in prompt
    for literal in ("SHV", "SPY", "QQQ"):
        assert literal not in prompt


@pytest.mark.parametrize(
    ("held", "expected"),
    [([], "none"), (["SPY"], "risk_on"), (["QQQ"], "risk_on"), (["SHV"], "defensive"), (["SPY", "SHV"], "mixed"), (["AAPL"], "none")],
)
def test_the_snapshot_derives_the_regime_from_what_is_held(tmp_path: Path, held: list[str], expected: str) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.broker.positions = [_hold(symbol) for symbol in held]  # ty: ignore[unresolved-attribute]
    strategy.initialize()

    strategy.on_trading_iteration()

    portfolio = handle.runs[0][1]["portfolio"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert portfolio["current_regime"] == expected
    assert portfolio["sessions_in_regime"] == 1


def test_the_regime_follows_the_configured_symbols_not_literals(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    strategy.strategy_parameters = {**NewsBinaryStrategy.strategy_parameters, "symbols": ("VOO",), "defensive_symbol": "BIL"}
    strategy.broker.positions = [_hold("BIL")]  # ty: ignore[unresolved-attribute]
    strategy.initialize()

    strategy.on_trading_iteration()

    assert handle.runs[0][1]["portfolio"]["current_regime"] == "defensive"  # ty: ignore[invalid-argument-type, not-subscriptable]


def test_sessions_in_regime_counts_distinct_trading_days_and_resets_on_a_flip(tmp_path: Path) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    broker = strategy.broker
    broker.positions = [_hold("SHV")]  # ty: ignore[unresolved-attribute]
    strategy.initialize()

    strategy.on_trading_iteration()  # day 1, first run
    strategy.clock.advance(3600)  # day 1, an hourly re-run must not count as a second session
    strategy.on_trading_iteration()
    strategy.clock.advance(86400)  # day 2
    strategy.on_trading_iteration()
    broker.positions = [_hold("SPY")]  # ty: ignore[unresolved-attribute]
    strategy.clock.advance(86400)  # day 3, the book flipped
    strategy.on_trading_iteration()
    strategy.clock.advance(86400)  # day 4
    strategy.on_trading_iteration()

    seen = [(run[1]["portfolio"]["current_regime"], run[1]["portfolio"]["sessions_in_regime"]) for run in handle.runs]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert seen == [("defensive", 1), ("defensive", 1), ("defensive", 2), ("risk_on", 1), ("risk_on", 2)]


def test_a_failed_positions_read_neither_claims_a_regime_nor_moves_the_counter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _, handle = _strategy(tmp_path, TradingMode.PAPER)
    broker = strategy.broker
    broker.positions = [_hold("SHV")]  # ty: ignore[unresolved-attribute]
    strategy.initialize()
    strategy.on_trading_iteration()
    monkeypatch.setattr(broker, "pull_positions", lambda: (_ for _ in ()).throw(BrokerError("down")))
    strategy.clock.advance(86400)
    strategy.on_trading_iteration()
    monkeypatch.undo()
    strategy.clock.advance(86400)
    strategy.on_trading_iteration()

    failed = handle.runs[1][1]["portfolio"]  # ty: ignore[invalid-argument-type, not-subscriptable]
    assert "current_regime" not in failed and "sessions_in_regime" not in failed
    # the unread day is not a session in the regime: day 1 -> 1, day 3 -> 2
    assert handle.runs[2][1]["portfolio"]["sessions_in_regime"] == 2  # ty: ignore[invalid-argument-type, not-subscriptable]

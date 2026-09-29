from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et, make_session

from trading_agent_framework.agents.results import AgentRunResult, ToolCallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState
from trading_agent_framework.utils.errors import AgentError, FatalStrategyError

DAY = date(2026, 9, 1)
TEN_AM = et(2026, 9, 1, 10, 0)


def _strategy(tmp_path: Path, *, mode: TradingMode = TradingMode.PAPER, now=TEN_AM) -> VwapPullbackStrategy:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path)
    strategy.vars.session = None
    strategy.vars.consecutive_agent_errors = 0
    strategy._build_components()
    return strategy


def _session() -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"))


class _Handle:
    def __init__(self, outcome: AgentRunResult | Exception) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, dict]] = []

    def run(self, task_prompt: str, *, context=None, run_id=None, force_tool=None) -> AgentRunResult:
        self.calls.append((task_prompt, context))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Agents(dict):
    pass


def test_initialize_creates_both_agents_and_the_graph(tmp_path: Path) -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 8, 0), [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, universe=["AAA"], project_root=tmp_path, chat_model=FakeToolCallingChatModel(messages=iter([])))
    strategy.initialize()
    assert "vwap_entry" in strategy.agents and "vwap_exit" in strategy.agents
    assert strategy._graph is not None


def test_a_session_is_prepared_once_per_day_and_the_graph_runs_each_tick(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    prepared: list[date] = []
    strategy.scanner.prepare_session = lambda: prepared.append(DAY) or _session()
    invoked: list[object] = []
    strategy._graph = type("G", (), {"invoke": lambda self, state: invoked.append(state["now"])})()
    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    assert prepared == [DAY]
    assert len(invoked) == 2
    assert strategy.vars.session.unknown_positions_checked


def test_order_hooks_and_the_close_reach_the_desk(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    seen: list[str] = []
    strategy.desk.on_order_filled = lambda order, price, quantity: seen.append("filled")
    strategy.desk.on_order_canceled = lambda order: seen.append("canceled")
    strategy.desk.flatten_all = lambda reason: seen.append(reason)
    strategy.on_filled_order(None, object(), Decimal(1), Decimal(1), 1)
    strategy.on_canceled_order(object())
    strategy.before_market_closes()
    assert seen == ["filled", "canceled", "end-of-day flatten"]


def test_the_entry_node_runs_the_entry_agent_with_the_setups(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    strategy.vars.session.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED)
    handle = _Handle(AgentRunResult(output="passed", tool_calls=[ToolCallRecord(name="pass_on_setup", args={"symbol": "AAA"}, result="{}")]))
    strategy._agents = _Agents({"vwap_entry": handle})
    update = strategy._entry_node({"now": et(2026, 9, 1, 10, 0)})
    assert update["runs"] == [{"agent": "vwap_entry", "ok": True, "tool_calls": 1}]
    task, context = handle.calls[0]
    assert context["setups"][0]["symbol"] == "AAA" and "free_slots" in context


def test_three_agent_failures_in_a_row_abort_a_backtest(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.vars.session = _session()
    strategy._agents = _Agents({"vwap_entry": _Handle(AgentError("model down"))})
    strategy._entry_node({"now": et(2026, 9, 1, 10, 0)})
    strategy._entry_node({"now": et(2026, 9, 1, 10, 5)})
    with pytest.raises(FatalStrategyError, match="3 runs in a row"):
        strategy._entry_node({"now": et(2026, 9, 1, 10, 10)})


def test_main_registers_the_strategy() -> None:
    from trading_agent_framework.main import AGENT_STRATEGIES

    assert "vwap_pullback_continuation" in AGENT_STRATEGIES

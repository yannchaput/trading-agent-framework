from __future__ import annotations

from datetime import date, timedelta
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


def test_a_backtest_waits_in_one_minute_slices_so_the_stop_follows_the_entry_bar_by_bar(tmp_path: Path) -> None:
    # Final review I1: with the backtest clock's default infinite slice, one 5-minute tick jumps 10:00 -> 10:05 at
    # once, so the entry filled on the 10:01 bar gets its stop only at 10:05 and the 10:03 break is never checked.
    from tests.fakes import FrameDataSource, minute_ohlc
    from tests.strategies.vwap_pullback.test_vwap_desk_entries import ROWS

    from trading_agent_framework.backtesting.broker import BacktestBroker
    from trading_agent_framework.backtesting.clock import BacktestClock
    from trading_agent_framework.strategies.vwap_pullback.features import BarContext
    from trading_agent_framework.strategies.vwap_pullback.session import CandidateInfo
    from trading_agent_framework.strategies.vwap_pullback.trades import TradeStatus

    clock = BacktestClock(start=TEN_AM, sessions=[make_session(DAY)])
    frames = {("AAA", "minute"): minute_ohlc(et(2026, 9, 1, 9, 31), ROWS)}
    broker = BacktestBroker("vwap_pullback_continuation", data_source=FrameDataSource(frames), clock=clock, budget=Decimal("100000"), timestep="minute")
    clock.on_advance = broker.on_advance
    strategy = VwapPullbackStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], project_root=tmp_path, chat_model=FakeToolCallingChatModel(messages=iter([])))
    strategy.initialize()
    assert strategy.clock.max_wait_slice == 60.0
    assert BacktestClock.max_wait_slice == float("inf")  # set on this run's clock only
    broker.tracker.listeners.append(strategy.executor._events)  # what executor.run() wires
    state = SessionState(day=DAY, session=make_session(DAY), bar_stamp="close", session_open_equity=Decimal("100000"))
    state.candidates["AAA"] = CandidateInfo(symbol="AAA", daily_atr=2.0, beta=1.0)
    state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
    state.contexts["AAA"] = [BarContext(time=TEN_AM, open=100, high=100.2, low=99.8, close=100, volume=5000, vwap=99.9, rs=0.01, rvol=2.0, session_open=99.0, session_high=100.2)]
    strategy.vars.session = state
    strategy.desk.enter_long("AAA", "earnings", "clean pullback")
    trade = state.book.get("AAA")
    strategy.executor.wait_until(TEN_AM + timedelta(seconds=300))  # one 5-minute tick
    assert trade.status is TradeStatus.CLOSED and trade.exit_reason == "stop"
    stop_fill = next(f for f in broker.ledger.fills if f.side.value == "sell")
    assert stop_fill.time == et(2026, 9, 1, 10, 3) and stop_fill.price == Decimal("99.30")  # the breaking bar, at the stop


def test_a_failed_exit_run_does_not_mark_the_trades_reviewed(tmp_path: Path) -> None:
    # Final review M2: the triggers that called the exit agent must bring the trades back next tick.
    strategy = _strategy(tmp_path)
    strategy.vars.session = _session()
    reviewed: list[object] = []
    strategy.desk.mark_reviewed = reviewed.append
    strategy._agents = _Agents({"vwap_exit": _Handle(AgentError("model down"))})
    strategy._exit_node({"now": TEN_AM})
    assert reviewed == []
    strategy._agents = _Agents({"vwap_exit": _Handle(AgentRunResult(output="held", tool_calls=[]))})
    strategy._exit_node({"now": TEN_AM})
    assert reviewed == [TEN_AM]

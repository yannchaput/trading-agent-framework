from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from tests.fakes import FakeBroker, FakeClock, et, make_session

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.vwap_pullback import VwapPullbackStrategy
from trading_agent_framework.strategies.vwap_pullback.session import SessionState
from trading_agent_framework.strategies.vwap_pullback.setups import Setup, SetupState

DAY = date(2026, 9, 1)
TEN_AM = et(2026, 9, 1, 10, 0)


def _strategy(tmp_path: Path, *, mode: TradingMode = TradingMode.PAPER, now=TEN_AM) -> VwapPullbackStrategy:
    broker = FakeBroker(FakeClock(now, [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path)
    strategy.vars.session = None
    strategy._build_components()
    return strategy


def _session() -> SessionState:
    return SessionState(day=DAY, session=make_session(DAY), bar_stamp="open", session_open_equity=Decimal("25000"))


def _record_steps(strategy: VwapPullbackStrategy) -> list[str]:
    """Replace the three tick steps by recorders; the list fills in call order."""
    steps: list[str] = []
    strategy.desk.reconcile = lambda now: steps.append("reconcile")
    strategy.scanner.scan = lambda state: steps.append("scan")
    strategy.desk.enter_triggered = lambda: steps.append("enter") or []
    return steps


def test_initialize_builds_the_desk_and_the_scanner_and_no_agent(tmp_path: Path) -> None:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 8, 0), [make_session(DAY)]), "vwap_pullback_continuation")
    strategy = VwapPullbackStrategy(broker, universe=["AAA"], project_root=tmp_path)
    strategy.initialize()
    assert strategy.desk is not None and strategy.scanner is not None
    assert strategy._agents is None  # the lazy agent manager was never touched


def test_a_tick_reconciles_then_scans_then_enters_and_the_session_is_prepared_once(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    prepared: list[date] = []
    strategy.scanner.prepare_session = lambda: prepared.append(DAY) or _session()
    steps = _record_steps(strategy)
    strategy.on_trading_iteration()
    strategy.on_trading_iteration()
    assert prepared == [DAY]
    assert steps == ["reconcile", "scan", "enter"] * 2
    assert strategy.vars.session.unknown_positions_checked


def test_a_flattened_session_runs_no_tick_step(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    strategy.vars.session = _session()
    strategy.vars.session.flattened = True
    steps = _record_steps(strategy)
    strategy.on_trading_iteration()
    assert steps == []


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


def test_main_registers_the_strategy() -> None:
    from trading_agent_framework.main import AGENT_STRATEGIES

    assert "vwap_pullback_continuation" in AGENT_STRATEGIES


def test_the_strategy_package_imports_no_llm_library() -> None:
    # A fresh interpreter: this process has already imported LangChain through other tests.
    code = (
        "import sys; import trading_agent_framework.strategies.vwap_pullback; "
        "bad = [m for m in ('langchain', 'langchain_openai', 'langgraph') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


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
    strategy = VwapPullbackStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], project_root=tmp_path)
    strategy.initialize()
    assert strategy.clock.max_wait_slice == 60.0
    assert BacktestClock.max_wait_slice == float("inf")  # set on this run's clock only
    broker.tracker.listeners.append(strategy.executor._events)  # what executor.run() wires
    state = SessionState(day=DAY, session=make_session(DAY), bar_stamp="close", session_open_equity=Decimal("100000"))
    state.candidates["AAA"] = CandidateInfo(symbol="AAA", daily_atr=2.0, beta=1.0)
    state.setups["AAA"] = Setup(symbol="AAA", state=SetupState.TRIGGERED, pullback_low=99.5, trigger_close=100.0, last_close=100.0)
    state.contexts["AAA"] = [BarContext(time=TEN_AM, open=100, high=100.2, low=99.8, close=100, volume=5000, vwap=99.9, rs=0.01, rvol=2.0, session_open=99.0, session_high=100.2)]
    strategy.vars.session = state
    assert strategy.desk.enter_triggered() == ["AAA"]
    trade = state.book.get("AAA")
    strategy.executor.wait_until(TEN_AM + timedelta(seconds=300))  # one 5-minute tick
    assert trade.status is TradeStatus.CLOSED and trade.exit_reason == "stop"
    stop_fill = next(f for f in broker.ledger.fills if f.side.value == "sell")
    assert stop_fill.time == et(2026, 9, 1, 10, 3) and stop_fill.price == Decimal("99.30")  # the breaking bar, at the stop


class _FakeIndicators:
    def __init__(self, adx, rsi) -> None:
        self._adx, self._rsi = adx, rsi

    def adx(self, asset, timestep, **kwargs):
        assert (asset, timestep, kwargs) == ("SPY", "minute", {"include_after_hours": False})
        return self._adx

    def rsi(self, asset, timestep, **kwargs):
        return self._rsi


def _charting(tmp_path: Path, adx, rsi, vix) -> tuple[VwapPullbackStrategy, list[tuple]]:
    from trading_agent_framework.core.indicators import IndicatorRow

    strategy = _strategy(tmp_path, mode=TradingMode.BACKTESTING)
    lines: list[tuple] = []
    strategy.add_line = lambda name, value, **kw: lines.append((name, value, kw["plot_name"]))
    strategy._indicators = _FakeIndicators(None if adx is None else IndicatorRow({"ADX_14": adx}), rsi)
    strategy._vix = type("V", (), {"previous_close": lambda self, day: vix})()
    return strategy, lines


def test_a_tick_charts_adx_rsi_and_the_previous_vix_close(tmp_path: Path) -> None:
    strategy, lines = _charting(tmp_path, adx=22.0, rsi=61.5, vix=17.2)
    strategy._chart_indicators(_session())
    assert lines == [("ADX", 22.0, "ADX / RSI"), ("RSI", 61.5, "ADX / RSI"), ("VIX", 17.2, "VIX")]


def test_a_value_that_is_not_ready_is_skipped_not_charted(tmp_path: Path) -> None:
    strategy, lines = _charting(tmp_path, adx=None, rsi=None, vix=None)
    strategy._chart_indicators(_session())
    assert lines == []


def test_nothing_is_charted_outside_backtesting(tmp_path: Path) -> None:
    strategy, lines = _charting(tmp_path, adx=22.0, rsi=61.5, vix=17.2)
    strategy.trading_mode = TradingMode.PAPER
    strategy._chart_indicators(_session())
    assert lines == []

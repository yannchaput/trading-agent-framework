from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from tests.fakes import FakeBroker, FakeClock, et, make_session
from tests.strategies.earnings_drift.drift_helpers import make_candidate
from tests.strategies.earnings_drift.test_drift_scanner import StaticEvents

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.backtesting.data.alpaca import AlpacaBacktestData
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams
from trading_agent_framework.strategies.earnings_drift.scanner import ScanResult
from trading_agent_framework.strategies.earnings_drift.tools import ORDER_TOOLS
from trading_agent_framework.utils.errors import AgentError, FatalStrategyError

DAY = date(2026, 9, 1)


class _Handle:
    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.runs = error, []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs.append((context, tool_budget))
        if self.error is not None:
            raise self.error
        return AgentRunResult(output="ok", tool_calls=[])


class _Manager:
    def __init__(self, handle: _Handle) -> None:
        self.handle, self.created = handle, []

    def create(self, **kwargs: Any) -> _Handle:
        self.created.append(kwargs)
        return self.handle

    def __getitem__(self, name: str) -> _Handle:
        return self.handle

    def telemetry_summary(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _strategy(
    tmp_path: Path, *, mode: TradingMode = TradingMode.BACKTESTING, settings: DriftParams | None = None, handle: _Handle | None = None, name: str = "earnings_drift"
) -> tuple[EarningsDriftStrategy, _Manager]:
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 16, 0), [make_session(DAY)]), name)
    strategy = EarningsDriftStrategy(broker, mode=mode, universe=["AAA"], project_root=tmp_path, settings=settings or DriftParams(live_bar_delay_seconds=0), event_source=StaticEvents([]))
    manager = _Manager(handle or _Handle())
    strategy._agents = cast(AgentManager, manager)
    strategy.initialize()
    return strategy, manager


def _stub_cycle(strategy: EarningsDriftStrategy, *, candidates: int = 1, holdings: int = 0, hollow: bool = False) -> list[str]:
    steps: list[str] = []
    assert strategy.scanner is not None and strategy.desk is not None
    desk = strategy.desk
    strategy.scanner.prepare = lambda held: steps.append("prepare") or ScanResult(DAY, [DAY], [make_candidate()] * candidates, {}, hollow=hollow)  # type: ignore[method-assign]
    desk.exposed_symbols = lambda: set()  # type: ignore[method-assign]
    desk.begin_session = lambda *args: steps.append("begin")  # type: ignore[method-assign]
    desk.reconcile = lambda: steps.append("reconcile") or []  # type: ignore[method-assign]
    desk.candidate_sheets = lambda: [{"symbol": "AAA"}] * candidates  # type: ignore[method-assign]
    desk.holdings_context = lambda: [{"symbol": "HHH"}] * holdings  # type: ignore[method-assign]
    desk.balances = lambda: {}  # type: ignore[method-assign]
    desk.baseline_entries = lambda: steps.append("baseline") or []  # type: ignore[method-assign]
    desk.ensure_stops = lambda: steps.append("ensure")  # type: ignore[method-assign]
    desk.record_undecided = lambda: steps.append("undecided")  # type: ignore[method-assign]
    desk.save = lambda: steps.append("save")  # type: ignore[method-assign]
    return steps


def test_initialize_creates_the_agent_with_order_and_research_tools(tmp_path: Path) -> None:
    _, manager = _strategy(tmp_path)
    [created] = manager.created
    names = [tool.__name__ for tool in created["tools"]]
    assert created["name"] == "drift" and names[:4] == list(ORDER_TOOLS) and "search_news" in names and "get_filing_document" in names
    assert created["exempt_tools"] == list(ORDER_TOOLS) and created["temperature"] == 0.3


def test_baseline_mode_creates_no_agent_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _, manager = _strategy(tmp_path, settings=DriftParams(agent_enabled=False))
    assert manager.created == [] and "guardrail baseline" in caplog.text


def test_a_missing_sec_identity_refuses_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT")
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 16, 0), [make_session(DAY)]), "earnings_drift")
    strategy = EarningsDriftStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA"], project_root=tmp_path)
    strategy._agents = cast(AgentManager, _Manager(_Handle()))
    with pytest.raises(FatalStrategyError, match="SEC_EDGAR_USER_AGENT"):
        strategy.initialize()


def test_the_cycle_runs_in_order_after_the_close(tmp_path: Path) -> None:
    strategy, manager = _strategy(tmp_path)
    steps = _stub_cycle(strategy, candidates=2, holdings=1)
    strategy.after_market_closes()
    assert steps == ["prepare", "begin", "reconcile", "ensure", "undecided", "save"]
    [(context, budget)] = manager.handle.runs
    assert set(context) == {"date", "candidates", "holdings", "balances"} and budget == 4 * 3


def test_no_candidates_and_no_holdings_means_no_agent_call(tmp_path: Path) -> None:
    strategy, manager = _strategy(tmp_path)
    _stub_cycle(strategy, candidates=0)
    strategy.after_market_closes()
    assert manager.handle.runs == []


def test_baseline_mode_buys_instead_of_asking_the_agent(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, settings=DriftParams(agent_enabled=False))
    steps = _stub_cycle(strategy)
    strategy.after_market_closes()
    assert "baseline" in steps


def test_three_agent_failures_end_a_backtest_the_next_morning(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, handle=_Handle(AgentError("model down")))
    _stub_cycle(strategy)
    for _ in range(2):
        strategy.after_market_closes()
    strategy.on_trading_iteration()  # two failures: carry on
    strategy.after_market_closes()
    with pytest.raises(FatalStrategyError, match="3 agent runs failed in a row"):
        strategy.on_trading_iteration()


def test_agent_failures_never_end_a_paper_run(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.PAPER, handle=_Handle(AgentError("model down")))
    _stub_cycle(strategy)
    for _ in range(4):
        strategy.after_market_closes()
    strategy.on_trading_iteration()


def test_three_hollow_scans_end_a_backtest_the_next_morning(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy, hollow=True)
    for _ in range(3):
        strategy.after_market_closes()
    with pytest.raises(FatalStrategyError, match="3 failed or hollow scans in a row"):
        strategy.on_trading_iteration()


def test_order_hooks_reach_the_desk(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    assert strategy.desk is not None
    seen: list[str] = []
    strategy.desk.on_order_filled = lambda order: seen.append("filled")  # type: ignore[method-assign]
    strategy.desk.on_order_canceled = lambda order: seen.append("canceled")  # type: ignore[method-assign]
    order = object()
    strategy.on_filled_order(None, order, None, None, 1)  # type: ignore[arg-type]
    strategy.on_canceled_order(order)  # type: ignore[arg-type]
    assert seen == ["filled", "canceled"]


def test_a_failing_scan_still_runs_the_guardrails(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, manager = _strategy(tmp_path)
    steps = _stub_cycle(strategy, candidates=0)
    assert strategy.scanner is not None

    def boom(held: set[str]) -> ScanResult:
        raise RuntimeError("boom")

    strategy.scanner.prepare = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR):
        strategy.after_market_closes()
    assert steps == ["begin", "reconcile", "ensure", "undecided", "save"]
    assert "RuntimeError" in caplog.text and "boom" in caplog.text
    assert manager.handle.runs == []


def test_an_unexpected_agent_exception_still_runs_the_tail(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    strategy, _ = _strategy(tmp_path, handle=_Handle(RuntimeError("boom")))
    steps = _stub_cycle(strategy)
    with caplog.at_level(logging.ERROR):
        strategy.after_market_closes()
    assert steps == ["prepare", "begin", "reconcile", "ensure", "undecided", "save"]
    assert "RuntimeError" in caplog.text and "boom" in caplog.text
    strategy.on_trading_iteration()  # an unexpected exception is not an agent failure: no fatal error is pending


def _scans(strategy: EarningsDriftStrategy, kinds: str) -> None:
    """Each cycle's scan in turn: `f` raises, `h` is hollow, `s` is sound."""
    assert strategy.scanner is not None
    pending = list(kinds)

    def prepare(held: set[str]) -> ScanResult:
        kind = pending.pop(0)
        if kind == "f":
            raise RuntimeError("benchmark fetch failed")
        return ScanResult(DAY, [DAY], [], {}, hollow=kind == "h")

    strategy.scanner.prepare = prepare  # type: ignore[method-assign]


def _cycles(strategy: EarningsDriftStrategy, count: int) -> None:
    for _ in range(count):
        strategy.after_market_closes()


def test_three_failed_scans_end_a_backtest_the_next_morning(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy)
    _scans(strategy, "fff")
    _cycles(strategy, 3)
    with pytest.raises(FatalStrategyError, match="3 failed or hollow scans in a row"):
        strategy.on_trading_iteration()


def test_two_failed_scans_do_not_end_a_backtest(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy)
    _scans(strategy, "ff")
    _cycles(strategy, 2)
    strategy.on_trading_iteration()


def test_failed_scans_never_end_a_paper_run(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.PAPER)
    _stub_cycle(strategy)
    _scans(strategy, "ffff")
    _cycles(strategy, 4)
    strategy.on_trading_iteration()


def test_failed_and_hollow_scans_share_one_streak(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy)
    _scans(strategy, "fhf")
    _cycles(strategy, 3)
    with pytest.raises(FatalStrategyError, match="3 failed or hollow scans in a row"):
        strategy.on_trading_iteration()


def test_a_sound_scan_resets_the_failed_scan_streak(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path)
    _stub_cycle(strategy)
    _scans(strategy, "ffsff")
    _cycles(strategy, 5)
    strategy.on_trading_iteration()


def test_the_agent_and_the_baseline_keep_separate_state_files(tmp_path: Path) -> None:
    agent, _ = _strategy(tmp_path, mode=TradingMode.PAPER)
    assert agent.desk is not None
    agent._state.traded_symbols.add("AAA")
    agent.desk.save()
    baseline, _ = _strategy(tmp_path, mode=TradingMode.PAPER, name="earnings_drift_baseline", settings=DriftParams(agent_enabled=False, live_bar_delay_seconds=0))
    assert baseline.desk is not None
    assert baseline._state.traded_symbols == set()  # the agent's state is not the baseline's
    baseline._state.traded_symbols.add("BBB")
    baseline.desk.save()
    assert (tmp_path / "data" / "earnings_drift_state_paper.json").exists() and (tmp_path / "data" / "earnings_drift_baseline_state_paper.json").exists()
    again, _ = _strategy(tmp_path, mode=TradingMode.PAPER)
    assert again._state.traded_symbols == {"AAA"}  # the baseline's save did not overwrite it


# --- A4: the backtest override and the SEC freshness as_of ------------------------------------------


def _captured_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: DriftParams, **overrides: Any) -> tuple[EarningsDriftStrategy, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def run_backtesting(self: Strategy, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(Strategy, "run_backtesting", run_backtesting)
    broker = FakeBroker(FakeClock(et(2026, 9, 1, 16, 0), [make_session(DAY)]), "earnings_drift")
    strategy = EarningsDriftStrategy(broker, mode=TradingMode.BACKTESTING, universe=["AAA", "BBB"], project_root=tmp_path, settings=settings, event_source=StaticEvents([]))
    strategy.run_backtesting(**overrides)
    return strategy, captured


@pytest.mark.parametrize("agent_enabled", [True, False])
def test_run_backtesting_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_enabled: bool) -> None:
    strategy, kwargs = _captured_backtest(tmp_path, monkeypatch, DriftParams(agent_enabled=agent_enabled))
    assert kwargs["slippage"] == Decimal("0.0005") and kwargs["data_source"] is AlpacaBacktestData
    assert [asset.symbol for asset in kwargs["preload_assets"]] == ["AAA", "BBB", "SPY"]
    assert kwargs["timestep"] == "day" and kwargs["warmup_trading_days"] == 283 and kwargs["benchmark"] == "SPY"
    assert kwargs["agent_telemetry"] is agent_enabled
    assert kwargs["start"] == EarningsDriftStrategy.parameters["backtesting_start"] and kwargs["budget"] == Decimal(10000)
    end = EarningsDriftStrategy.parameters["backtesting_end"]
    assert kwargs["end"] == end
    assert strategy._backtest_end == end + timedelta(days=1)  # a bare-midnight end covers that whole day


def test_run_backtesting_records_an_overridden_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    end = et(2026, 6, 30, 16, 0)
    strategy, kwargs = _captured_backtest(tmp_path, monkeypatch, DriftParams(), end=end)
    assert kwargs["end"] == end and strategy._backtest_end == end


def test_a_backtest_loads_sec_events_as_of_its_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    strategy, _ = _captured_backtest(tmp_path, monkeypatch, DriftParams(live_bar_delay_seconds=0), end=et(2026, 9, 23, 16, 0))
    strategy._agents = cast(AgentManager, _Manager(_Handle()))
    strategy.initialize()
    assert strategy.scanner is not None
    assert strategy.scanner._load_as_of() == et(2026, 9, 23, 16, 0)


def test_paper_loads_sec_events_as_of_now(tmp_path: Path) -> None:
    strategy, _ = _strategy(tmp_path, mode=TradingMode.PAPER)
    assert strategy.scanner is not None
    assert strategy.scanner._load_as_of() is None  # the scanner then uses its clock's now

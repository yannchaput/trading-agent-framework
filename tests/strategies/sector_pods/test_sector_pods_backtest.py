"""End to end: a backtest over Tuesdays with fake agents (they call the real submit tools) and fake daily bars."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et, weekday_sessions

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.sector_pods import SectorPodsStrategy
from trading_agent_framework.strategies.sector_pods.pods import ALL_SYMBOLS
from trading_agent_framework.utils.errors import BacktestError

PRIOR = weekday_sessions(date(2025, 9, 15), 300)  # 300 completed sessions before the window
Tools = dict[str, Callable[..., dict[str, Any]]]
Script = Callable[[Tools, Any], None]


def _bars(growth: float, sessions: list[Any], base: float = 50.0) -> pd.DataFrame:
    closes = [base * (1 + growth) ** i * (1.01 if i % 2 else 0.99) for i in range(len(sessions))]
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.005 for c in closes], "low": [c * 0.995 for c in closes], "close": closes, "volume": [1e6] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in sessions], name="timestamp"),
    )


class _Handle:
    def __init__(self, tools: Tools, script: Script) -> None:
        self.tools, self.script = tools, script
        self.contexts: list[Any] = []

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None, max_steps: int | None = None) -> AgentRunResult:
        self.contexts.append(context)
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _pod(tools: Tools, ctx: Any) -> None:
    tools["submit_pod_view"](scores=[{"symbol": s, "score": 1, "reason": "trend"} for s in ctx["facts"]], sector_view="fine")


def _risk(tools: Tools, ctx: Any) -> None:
    tools["submit_risk_review"](objections=[], summary="calm")


def _pm(tools: Tools, ctx: Any) -> None:
    tools["submit_target_weights"](weights=[{"symbol": s, "weight": 0.2, "reason": "why"} for s in ("XLK", "XLV")], overrides="", summary="done")


def _silent(tools: Tools, ctx: Any) -> None:
    return None


class _Manager:
    def __init__(self, pm: Script = _pm) -> None:
        self.handles: dict[str, _Handle] = {}
        self.pm = pm

    def create(self, *, name: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        script = {"risk_manager": _risk, "portfolio_manager": self.pm}.get(name, _pod)
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, script)
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "test")


def _run(tmp_path: Path, manager: _Manager, tuesdays: int) -> None:
    sessions = weekday_sessions(date(2026, 11, 9), 5 * tuesdays - 3)
    source = FakeBacktestDataSource()
    source.set_sessions(sessions)
    every = [*PRIOR, *sessions]
    for index, symbol in enumerate(ALL_SYMBOLS):
        source.set_bars(Asset(symbol), _bars(0.003 - index * 0.0001, every))
    source.set_bars(Asset("SHV"), _bars(0.0, every, base=100.0))
    source.set_bars(Asset("SPY"), _bars(0.001, every, base=400.0))
    broker = FakeBroker(FakeClock(et(2026, 11, 9, 9, 0)), strategy_name="sector_pods")
    broker.news = FakeNewsProvider()
    strategy = SectorPodsStrategy(broker, mode=TradingMode.BACKTESTING, project_root=tmp_path, fred_client_factory=lambda: object())
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=sessions[0].open - timedelta(hours=1), end=sessions[-1].close, data_source=source, budget=Decimal(10000), benchmark="SPY", news_source=FakeNewsProvider())


def test_backtest_end_to_end(tmp_path: Path) -> None:
    manager = _Manager()

    _run(tmp_path, manager, 2)

    (path,) = tmp_path.rglob("reviews.jsonl")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["date"] for line in lines] == ["2026-11-10", "2026-11-17"]
    assert all(line["abandoned"] is None for line in lines)
    assert {order["symbol"] for order in lines[0]["orders"]} >= {"XLK", "XLV"}
    first, second = manager.handles["portfolio_manager"].contexts
    assert first["previous"] is None and second["previous"] is not None
    assert list(tmp_path.rglob("metrics.json"))


def test_backtest_aborts_after_three_abandoned_weeks(tmp_path: Path) -> None:
    with pytest.raises(BacktestError, match="3 reviews abandoned"):
        _run(tmp_path, _Manager(pm=_silent), 3)

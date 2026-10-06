"""End to end: a short daily backtest with a scripted agent (it calls the real desk tools), then the baseline."""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeNewsProvider, et, weekday_sessions
from tests.strategies.earnings_drift.test_drift_scanner import StaticEvents

from trading_agent_framework.agents.manager import AgentManager
from trading_agent_framework.agents.results import AgentRunResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.strategies.earnings_drift import EarningsDriftStrategy
from trading_agent_framework.strategies.earnings_drift.events import EarningsEvent
from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams

ALL = weekday_sessions(date(2026, 7, 20), 36)
HISTORY, WINDOW = ALL[:30], ALL[30:]
FLAT = (100.0, 100.5, 99.5, 100.0, 1_000_000.0)
AAA = [FLAT] * 30 + [
    (106.0, 110.0, 105.0, 109.0, 5_000_000.0),  # W0: the reaction
    (109.5, 111.0, 108.0, 110.0, 2_000_000.0),  # W1: the entry fills at 109.5
    (110.0, 112.0, 109.0, 111.0, 1_500_000.0),  # W2: the trail rises to 112
    (105.0, 106.0, 100.0, 101.0, 3_000_000.0),  # W3: 112 x 0.92 = 103.04 is crossed
    (101.0, 102.0, 100.0, 101.0, 1_000_000.0),
    (101.0, 102.0, 100.0, 101.0, 1_000_000.0),
]
SPY = [(400.0, 401.0, 399.0, 400.0, 5e7)] * 30 + [(400.0, 402.0, 399.0, 400.8, 5e7)] + [(400.8, 401.5, 400.0, 400.8, 5e7)] * 5
BBB = [(50.0, 50.5, 49.5, 50.0, 1_000_000.0)] * 36
RELEASE = et(WINDOW[0].open.year, WINDOW[0].open.month, WINDOW[0].open.day, 7, 0)


def _frame(rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([s.close for s in ALL], name="timestamp"))


class _Handle:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.runs = 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        for sheet in context["candidates"]:
            self.tools["buy"](sheet["symbol"], sheet["max_quantity"], 8.0, "real beat, held the gap")
        return AgentRunResult(output="ok", tool_calls=[])


class _Manager:
    def __init__(self) -> None:
        self.handle = _Handle()

    def create(self, *, name: str, tools: list[Any], **_: Any) -> _Handle:
        self.handle.tools = {tool.__name__: tool for tool in tools}
        return self.handle

    def __getitem__(self, name: str) -> _Handle:
        return self.handle

    def telemetry_summary(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, settings: DriftParams) -> tuple[EarningsDriftStrategy, _Manager]:
    source = FakeBacktestDataSource()
    source.set_sessions(WINDOW)
    for symbol, rows in {"AAA": AAA, "BBB": BBB, "SPY": SPY}.items():
        source.set_bars(Asset(symbol), _frame(rows))
    news = FakeNewsProvider(
        {"AAA": [{"headline": "AAA Q3 EPS $1.52 Beats $1.20 Estimate, Sales $1.1B Beat $1B Estimate", "created_at": (RELEASE + timedelta(minutes=1)).isoformat(), "symbols": ["AAA"]}]}
    )
    events = StaticEvents([EarningsEvent("AAA", RELEASE, "acc-AAA", "aaa.htm")])
    strategy = EarningsDriftStrategy(
        FakeBroker(FakeClock(WINDOW[0].open - timedelta(hours=1)), strategy_name="earnings_drift"),
        mode=TradingMode.BACKTESTING,
        universe=["AAA", "BBB"],
        project_root=tmp_path,
        settings=settings,
        event_source=events,
    )
    manager = _Manager()
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=WINDOW[0].open - timedelta(hours=1), end=WINDOW[-1].close, data_source=source, news_source=news, budget=Decimal(10000), benchmark="SPY")
    return strategy, manager


def _lines(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob(name)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_agent_buys_after_the_close_and_the_trail_takes_it_out(tmp_path: Path) -> None:
    _, manager = _run(tmp_path, DriftParams())
    [trade] = _lines(tmp_path, "trades.jsonl")
    assert trade["symbol"] == "AAA" and trade["exit_reason"] == "trail" and trade["quantity"] == "11"
    assert Decimal(trade["entry_price"]) == Decimal("109.5") and Decimal(trade["exit_price"]) == Decimal("103.04")
    assert trade["entry_date"] == WINDOW[1].open.date().isoformat() and trade["sessions_held"] == 3
    assert manager.handle.runs == 3  # W0 (the candidate), W1 and W2 (the holding); nothing to review after W3
    assert [(d["symbol"], d["decision"]) for d in _lines(tmp_path, "decisions.jsonl")] == [("AAA", "buy")]


def test_the_baseline_takes_the_same_trade_without_an_agent(tmp_path: Path) -> None:
    _, manager = _run(tmp_path, DriftParams(agent_enabled=False))
    [trade] = _lines(tmp_path, "trades.jsonl")
    assert trade["exit_reason"] == "trail" and trade["thesis"] == "baseline" and trade["agent_enabled"] is False
    assert manager.handle.runs == 0

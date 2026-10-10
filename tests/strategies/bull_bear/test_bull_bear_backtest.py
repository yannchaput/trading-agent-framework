"""End to end: a backtest over two Tuesdays with fake agents (they call the real submit tools) and fake daily bars."""

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
from trading_agent_framework.strategies.bull_bear import BullBearStrategy

PRIOR = weekday_sessions(date(2025, 7, 21), 300)  # history before the window: 300 completed sessions to score on
SESSIONS = weekday_sessions(date(2026, 9, 14), 7)  # Monday 14th to Tuesday 22nd: two Tuesdays
UNIVERSE = [f"S{i}" for i in range(6)]


def _bars(growth: float, base: float = 50.0) -> pd.DataFrame:
    sessions = [*PRIOR, *SESSIONS]
    closes = [base * (1 + growth) ** i * (1.01 if i % 2 else 0.99) for i in range(len(sessions))]
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.005 for c in closes], "low": [c * 0.995 for c in closes], "close": closes, "volume": [1e6] * len(closes)},
        index=pd.DatetimeIndex([session.close for session in sessions], name="timestamp"),
    )


class _Handle:
    def __init__(self, tools: dict[str, Callable[..., dict[str, Any]]], script: Callable[[dict[str, Callable[..., dict[str, Any]]], Any], None]) -> None:
        self.tools, self.script, self.runs = tools, script, 0

    def run(self, task_prompt: str, *, context: Any = None, run_id: str | None = None, force_tool: str | None = None, tool_budget: int | None = None) -> AgentRunResult:
        self.runs += 1
        self.script(self.tools, context)
        return AgentRunResult(output="ok", tool_calls=[])


def _research(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_note"](context["fact_sheet"]["symbol"], "no news found")


def _bull(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_bull_case"]([{"symbol": s["fact_sheet"]["symbol"], "conviction": "high", "argument": "trend"} for s in context["stocks"]])


def _bear(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    tools["submit_bear_case"]([{"symbol": s["fact_sheet"]["symbol"], "risk": "low", "concern": "none", "argument": "none"} for s in context["stocks"]])


def _judge(tools: dict[str, Callable[..., dict[str, Any]]], context: Any) -> None:
    picked = [s["fact_sheet"]["symbol"] for s in context["stocks"]][:5]
    tools["submit_picks"]([{"symbol": s, "reason": "won"} for s in picked], [{"symbol": s, "reason": "lost"} for s in context["held"] if s not in picked])


class _Manager:
    def __init__(self) -> None:
        self.handles: dict[str, _Handle] = {}
        self.scripts = {"researcher": _research, "bull": _bull, "bear": _bear, "judge": _judge}

    def create(self, *, name: str, system_prompt: str, tools: list[Callable[..., Any]], **_: Any) -> _Handle:
        self.handles[name] = _Handle({tool.__name__: tool for tool in tools}, self.scripts[name])
        return self.handles[name]

    def __getitem__(self, name: str) -> _Handle:
        return self.handles[name]

    def telemetry_summary(self) -> dict[str, dict[str, Any]]:
        return {}  # the runner reads per-agent totals at the end of a run


@pytest.fixture(autouse=True)
def _sec_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "TestApp test@example.com")


def _run(tmp_path: Path, manager: _Manager) -> BullBearStrategy:
    source = FakeBacktestDataSource()
    source.set_sessions(SESSIONS)
    for index, symbol in enumerate(UNIVERSE):
        source.set_bars(Asset(symbol), _bars(0.003 - index * 0.0002))
    source.set_bars(Asset("SHV"), _bars(0.0, base=100.0))
    source.set_bars(Asset("SPY"), _bars(0.001, base=400.0))
    strategy = BullBearStrategy(
        FakeBroker(FakeClock(et(2026, 9, 14, 9, 0)), strategy_name="bull_bear"),
        mode=TradingMode.BACKTESTING,
        universe=UNIVERSE,
        project_root=tmp_path,
        sector_of=lambda symbol: "Technology",
    )
    strategy._agents = cast(AgentManager, manager)
    Strategy.run_backtesting(strategy, start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=source, budget=Decimal(10000), benchmark="SPY", news_source=FakeNewsProvider())
    return strategy


def _review_lines(tmp_path: Path) -> list[dict[str, Any]]:
    (path,) = tmp_path.rglob("reviews.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_backtest_reviews_on_the_two_tuesdays_and_buys_the_judges_picks(tmp_path: Path) -> None:
    manager = _Manager()

    _run(tmp_path, manager)

    lines = _review_lines(tmp_path)
    assert [line["date"] for line in lines] == ["2026-09-15", "2026-09-22"]
    assert not any(line["abandoned"] for line in lines)
    assert [stock["symbol"] for stock in lines[0]["debate_set"]] == UNIVERSE  # ranked by momentum, S0 first
    assert {order["symbol"] for order in lines[0]["orders"]} == set(UNIVERSE[:5])
    assert manager.handles["researcher"].runs == 2 * len(UNIVERSE)
    assert all(manager.handles[name].runs == 2 for name in ("bull", "bear", "judge"))
    assert list(tmp_path.rglob("metrics.json"))  # the run completed and wrote its report

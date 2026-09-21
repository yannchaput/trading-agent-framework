"""Agent telemetry through a real `Strategy.run_backtesting`: totals in settings.json, calls in llm_stats.sqlite."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage
from tests.backtesting.fakes import FakeBacktestDataSource
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et, weekday_sessions

from trading_agent_framework.backtesting.runner import BacktestResult
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy
from trading_agent_framework.entities.asset import Asset

SESSIONS = weekday_sessions(date(2026, 1, 5), 3)


class AgentStrategy(Strategy):
    sleeptime = "1D"

    def initialize(self) -> None:
        replies = iter([AIMessage("ok", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}) for _ in range(20)])
        self.agents.create(name="trader", system_prompt="x", model=FakeToolCallingChatModel(messages=replies))

    def on_trading_iteration(self) -> None:
        self.agents["trader"].run("go")


class PlainStrategy(Strategy):
    sleeptime = "1D"

    def on_trading_iteration(self) -> None:
        pass


def _backtest(tmp_path: Path, cls: type[Strategy], **kwargs: object) -> BacktestResult:
    closes = [100.0, 101.0, 102.0]
    bars = pd.DataFrame(
        {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes, "volume": [1000.0] * 3},
        index=pd.DatetimeIndex([s.close for s in SESSIONS], name="timestamp"),
    )
    source = FakeBacktestDataSource()
    source.set_sessions(SESSIONS)
    source.set_bars(Asset("SPY"), bars)
    strategy = cls(FakeBroker(FakeClock(et(2026, 1, 5, 8, 0)), strategy_name="agent_strat"), mode=TradingMode.BACKTESTING, project_root=tmp_path)
    return strategy.run_backtesting(start=SESSIONS[0].open - timedelta(hours=1), end=SESSIONS[-1].close, data_source=source, benchmark="SPY", **kwargs)  # type: ignore[arg-type]


def _db(tmp_path: Path) -> Path:
    return tmp_path / "memory" / "agent_strat" / "backtesting" / "llm_stats.sqlite"


def test_telemetry_is_on_by_default_in_a_backtest_and_totals_reach_settings_json(tmp_path: Path) -> None:
    result = _backtest(tmp_path, AgentStrategy)

    on_disk = json.loads((result.run_dir / "settings.json").read_text())
    assert on_disk["agents"] == result.settings["agents"]
    trader = on_disk["agents"]["trader"]
    assert (trader["calls"], trader["input_tokens"], trader["output_tokens"], trader["total_tokens"], trader["tool_calls"]) == (3, 30, 6, 36, 0)


def test_every_call_is_saved_in_llm_stats_sqlite_under_the_run_directory_name(tmp_path: Path) -> None:
    result = _backtest(tmp_path, AgentStrategy)

    rows = sqlite3.connect(_db(tmp_path)).execute("SELECT run_id, agent, input_tokens FROM llm_calls ORDER BY id").fetchall()
    assert rows == [(result.run_dir.name, "trader", 10)] * 3


def test_call_timestamps_are_simulated_session_time_not_the_wall_clock(tmp_path: Path) -> None:
    _backtest(tmp_path, AgentStrategy)

    stamps = [ts for (ts,) in sqlite3.connect(_db(tmp_path)).execute("SELECT ts FROM llm_calls ORDER BY id")]
    assert [ts[:10] for ts in stamps] == ["2026-01-05", "2026-01-06", "2026-01-07"]


def test_turning_it_off_writes_no_agents_block_and_no_database(tmp_path: Path) -> None:
    result = _backtest(tmp_path, AgentStrategy, agent_telemetry=False)

    assert "agents" not in json.loads((result.run_dir / "settings.json").read_text())
    assert not _db(tmp_path).exists()


def test_a_strategy_without_agents_gets_no_agents_block_and_no_database(tmp_path: Path) -> None:
    result = _backtest(tmp_path, PlainStrategy)

    assert "agents" not in json.loads((result.run_dir / "settings.json").read_text())
    assert not _db(tmp_path).exists()


def test_a_second_backtest_replaces_the_first_runs_rows(tmp_path: Path) -> None:
    _backtest(tmp_path, AgentStrategy)

    second = _backtest(tmp_path, AgentStrategy)

    run_ids = {run_id for (run_id,) in sqlite3.connect(_db(tmp_path)).execute("SELECT run_id FROM llm_calls")}
    assert run_ids == {second.run_dir.name}

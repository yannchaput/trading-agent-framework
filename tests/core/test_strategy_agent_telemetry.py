from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from tests.fakes import FakeBroker, FakeClock, FakeToolCallingChatModel, et

from trading_agent_framework.agents.stats_store import LLMStatsStore, llm_stats_db_path
from trading_agent_framework.agents.telemetry import CallRecord
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


class TelemetryStrategy(Strategy):
    agent_telemetry = True


def _strategy(tmp_path: Path, mode: TradingMode, cls: type[Strategy] = Strategy, run_id: str | None = "2026-09-14_090000_paper") -> Strategy:
    strategy = cls(FakeBroker(FakeClock(_START), strategy_name="momentum"), mode=mode, project_root=tmp_path)
    strategy.run_id = run_id
    return strategy


def _run_one_agent_call(strategy: Strategy) -> None:
    model = FakeToolCallingChatModel(messages=iter([AIMessage("hi", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})]))
    strategy.agents.create(name="trader", system_prompt="x", model=model).run("go")


def _rows(db: Path) -> list[tuple[str, str, int]]:
    return sqlite3.connect(db).execute("SELECT run_id, agent, input_tokens FROM llm_calls ORDER BY id").fetchall()


def _seed(tmp_path: Path, mode: TradingMode, run_id: str) -> Path:
    db = llm_stats_db_path(tmp_path, "momentum", mode)
    LLMStatsStore(db, run_id=run_id).record(
        CallRecord(ts=datetime(2026, 1, 1, tzinfo=_START.tzinfo), agent="old", model=None, input_tokens=1, output_tokens=1, reasoning_tokens=None,
                   total_tokens=2, latency_ms=1.0, tool_calls=0)
    )
    return db


def test_telemetry_is_off_by_default_so_a_paper_strategy_records_nothing(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER)

    _run_one_agent_call(strategy)

    assert strategy.agent_telemetry_summary() == {}
    assert list(tmp_path.rglob("llm_stats.sqlite")) == []


def test_a_paper_strategy_that_opts_in_logs_to_llm_stats_sqlite_next_to_memory(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER, TelemetryStrategy)

    _run_one_agent_call(strategy)

    assert _rows(tmp_path / "memory" / "momentum" / "paper" / "llm_stats.sqlite") == [("2026-09-14_090000_paper", "trader", 10)]
    assert strategy.agent_telemetry_summary()["trader"]["input_tokens"] == 10


def test_calls_are_stamped_with_the_strategy_clock(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER, TelemetryStrategy)

    _run_one_agent_call(strategy)

    [(ts,)] = sqlite3.connect(tmp_path / "memory" / "momentum" / "paper" / "llm_stats.sqlite").execute("SELECT ts FROM llm_calls").fetchall()
    assert ts == _START.isoformat()


def test_paper_keeps_the_rows_of_earlier_runs(tmp_path: Path) -> None:
    db = _seed(tmp_path, TradingMode.PAPER, "earlier_run")
    strategy = _strategy(tmp_path, TradingMode.PAPER, TelemetryStrategy)

    _run_one_agent_call(strategy)

    assert [row[:2] for row in _rows(db)] == [("earlier_run", "old"), ("2026-09-14_090000_paper", "trader")]


def test_a_backtest_deletes_the_previous_runs_rows_as_soon_as_agents_is_first_used(tmp_path: Path) -> None:
    db = _seed(tmp_path, TradingMode.BACKTESTING, "earlier_run")
    strategy = _strategy(tmp_path, TradingMode.BACKTESTING, TelemetryStrategy, run_id="2026-09-14_090000_backtesting")

    strategy.agents  # noqa: B018 -- first access opens the store; no model call has happened yet

    assert _rows(db) == []


def test_a_strategy_that_never_uses_agents_creates_no_database_and_wipes_nothing(tmp_path: Path) -> None:
    db = _seed(tmp_path, TradingMode.BACKTESTING, "earlier_run")
    strategy = _strategy(tmp_path, TradingMode.BACKTESTING, TelemetryStrategy)

    assert strategy.agent_telemetry_summary() == {}
    assert _rows(db) == [("earlier_run", "old", 1)]
    assert strategy._agents is None  # noqa: SLF001 -- reading the summary must not build the manager


def test_without_a_run_id_calls_are_still_counted_but_nothing_is_written(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER, TelemetryStrategy, run_id=None)

    _run_one_agent_call(strategy)

    assert strategy.agent_telemetry_summary()["trader"]["calls"] == 1
    assert list(tmp_path.rglob("llm_stats.sqlite")) == []


def test_an_unusable_database_location_warns_and_the_run_still_counts_in_memory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "memory").write_text("a file where the memory directory should be")
    strategy = _strategy(tmp_path, TradingMode.PAPER, TelemetryStrategy)

    with caplog.at_level(logging.WARNING):
        _run_one_agent_call(strategy)

    assert strategy.agent_telemetry_summary()["trader"]["calls"] == 1
    assert "llm stats" in caplog.text.lower()

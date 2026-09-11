from __future__ import annotations

from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.strategy import Strategy

_START = et(2026, 9, 14, 9, 0)


def _strategy(
    tmp_path: Path, mode: TradingMode = TradingMode.PAPER, name: str = "momentum"
) -> Strategy:
    return Strategy(
        FakeBroker(FakeClock(_START), strategy_name=name), mode=mode, project_root=tmp_path
    )


def test_memory_is_opened_lazily_and_cached(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    assert not (tmp_path / "memory").exists()

    store = strategy.memory

    assert store.db_path == tmp_path / "memory" / "momentum" / "paper" / "memory.sqlite"
    assert store.db_path.is_file()
    assert store.strategy_name == "momentum"
    assert strategy.memory is store


@pytest.mark.parametrize("mode", list(TradingMode))
def test_each_trading_mode_has_its_own_database(tmp_path: Path, mode: TradingMode) -> None:
    assert _strategy(tmp_path, mode).memory.db_path.parent == tmp_path / "memory" / "momentum" / mode.value


def test_paper_and_live_memory_survive_a_new_run(tmp_path: Path) -> None:
    for mode in (TradingMode.PAPER, TradingMode.LIVE):
        memory_id = _strategy(tmp_path, mode).memory.remember("keep me")["id"]
        assert _strategy(tmp_path, mode).memory.get(memory_id) is not None


def test_backtesting_memory_starts_empty_on_every_run(tmp_path: Path) -> None:
    first_run = _strategy(tmp_path, TradingMode.BACKTESTING)
    memory_id = first_run.memory.remember("would leak into the next backtest")["id"]
    assert first_run.memory.get(memory_id) is not None

    second_run = _strategy(tmp_path, TradingMode.BACKTESTING)
    assert second_run.memory.get(memory_id) is None


def test_memory_reopens_when_the_trading_mode_changes(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path, TradingMode.PAPER)
    paper = strategy.memory
    strategy.trading_mode = TradingMode.LIVE

    live = strategy.memory

    assert live is not paper
    assert live.db_path.parent.name == "live"


def test_memory_events_are_stamped_with_strategy_time(tmp_path: Path) -> None:
    item = _strategy(tmp_path).memory.remember("x")
    assert item["created_at"] == _START.isoformat()


def test_the_memory_folder_name_is_sanitised(tmp_path: Path) -> None:
    store = _strategy(tmp_path, name="orb v2/beta").memory
    assert store.db_path.parent.parent == tmp_path / "memory" / "orb_v2_beta"
    assert store.strategy_name == "orb v2/beta"

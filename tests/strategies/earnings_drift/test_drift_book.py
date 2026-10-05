from __future__ import annotations

import json
import logging
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.earnings_drift.book import (
    STATE_VERSION,
    DriftState,
    JsonlLog,
    StateStore,
    Trade,
    TradeState,
    sessions_held,
    state_path,
)


def _trade() -> Trade:
    return Trade(
        symbol="AAA",
        entry_order_id="e1",
        trail_percent=D("8.0"),
        thesis="beat and raise",
        accession_number="acc",
        state=TradeState.OPEN,
        quantity=D("11"),
        entry_price=D("109.5"),
        opened_on=date(2026, 9, 2),
        stop_order_id="s1",
        reaction_low=D("105"),
    )


def test_a_trade_round_trips_through_json() -> None:
    trade = _trade()
    assert Trade.from_json(json.loads(json.dumps(trade.to_json()))) == trade


def test_state_round_trips_and_is_versioned(tmp_path: Path) -> None:
    path = state_path(tmp_path, TradingMode.PAPER)
    assert path == tmp_path / "data" / "earnings_drift_state_paper.json"
    store = StateStore(path)
    state = DriftState(trades={"AAA": _trade()}, agent_failure_streak=2, hollow_scan_streak=1, traded_symbols={"AAA", "BBB"})
    store.save(state)
    assert json.loads(path.read_text())["version"] == STATE_VERSION
    assert store.load() == state


@pytest.mark.parametrize("content", ["not json", json.dumps({"version": 99, "trades": {}}), json.dumps({"version": 1, "trades": {"AAA": {"symbol": "AAA"}}})])
def test_a_bad_state_file_loads_empty_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture, content: str) -> None:
    path = tmp_path / "state.json"
    path.write_text(content)
    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == DriftState()
    assert "empty state" in caplog.text


def test_a_missing_file_is_an_empty_state_and_wipe_is_safe(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "none.json")
    assert store.load() == DriftState()
    store.wipe()
    store.save(DriftState())
    store.wipe()
    assert not (tmp_path / "none.json").exists()


def test_jsonl_log_appends_and_does_nothing_without_a_path(tmp_path: Path) -> None:
    path = tmp_path / "run" / "trades.jsonl"
    log = JsonlLog(lambda: path)
    log.append({"a": D("1.5")})
    log.append({"b": 2})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"a": "1.5"}, {"b": 2}]
    JsonlLog(lambda: None).append({"ignored": True})


def test_sessions_held_counts_both_ends() -> None:
    dates = [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4), date(2026, 9, 7)]
    assert sessions_held(date(2026, 9, 2), date(2026, 9, 2), dates) == 1
    assert sessions_held(date(2026, 9, 2), date(2026, 9, 7), dates) == 4

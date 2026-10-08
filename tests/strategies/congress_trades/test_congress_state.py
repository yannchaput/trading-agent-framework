from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.congress_trades.state import CongressState, PendingTrade, RunLog, StateStore, state_path


def _state() -> CongressState:
    return CongressState(
        processed=["10075701", "20035553"],
        holdings={"AAPL": {"tier": 8, "value_low": 5_000_001, "value_high": 25_000_000}, "BE": {"tier": 7, "value_low": 1_500_002, "value_high": 6_000_000}},
        target={"AAPL": 0.15, "BE": 0.05},
        traded=["AAPL", "BE", "DIS"],
        pending_trade=PendingTrade(target={"AAPL": 0.15, "BE": 0.05}, days=1),
        abandoned_streak=2,
        last_run="2026-10-07",
    )


def test_the_state_path_is_per_mode_under_the_projects_data_directory(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "congress_trades_state_paper.json"
    assert state_path(tmp_path, TradingMode.BACKTESTING).name == "congress_trades_state_backtesting.json"


def test_the_empty_state_has_no_history() -> None:
    state = CongressState()

    assert (state.processed, state.holdings, state.target, state.traded) == ([], {}, {}, [])
    assert (state.pending_trade, state.abandoned_streak, state.last_run) == (None, 0, None)


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == CongressState()


def test_a_state_round_trips(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")

    store.save(_state())

    assert StateStore(tmp_path / "data" / "state.json").load() == _state()


def test_a_state_without_a_pending_trade_round_trips(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    state = CongressState(processed=["1"], last_run="2026-10-07")

    store.save(state)

    assert store.load() == state
    assert store.load().pending_trade is None


def test_saving_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")

    store.save(_state())
    store.save(CongressState())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


_HOLDING = {"tier": 8, "value_low": 1, "value_high": 2}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2}),
        json.dumps({"version": 1, "processed": "A1"}),
        json.dumps({"version": 1, "processed": [1, 2]}),
        json.dumps({"version": 1, "holdings": {"AAPL": {"tier": 8}}}),  # incomplete record
        json.dumps({"version": 1, "holdings": {"AAPL": {**_HOLDING, "tier": "high"}}}),
        json.dumps({"version": 1, "holdings": {"AAPL": {**_HOLDING, "tier": True}}}),
        json.dumps({"version": 1, "holdings": []}),
        json.dumps({"version": 1, "target": {"AAPL": 1.5}}),  # a weight above 1
        json.dumps({"version": 1, "target": {"AAPL": -0.1}}),
        json.dumps({"version": 1, "target": {"AAPL": "0.1"}}),
        json.dumps({"version": 1, "target": []}),
        json.dumps({"version": 1, "traded": [1]}),
        json.dumps({"version": 1, "pending_trade": {"target": {"AAPL": 0.1}}}),  # no days
        json.dumps({"version": 1, "pending_trade": {"target": {"AAPL": 0.1}, "days": -1}}),
        json.dumps({"version": 1, "pending_trade": {"target": {"AAPL": 2.0}, "days": 0}}),
        json.dumps({"version": 1, "pending_trade": {"target": {"AAPL": 0.1}, "days": 0, "extra": 1}}),
        json.dumps({"version": 1, "pending_trade": "yes"}),
        json.dumps({"version": 1, "abandoned_streak": -1}),
        json.dumps({"version": 1, "abandoned_streak": True}),
        json.dumps({"version": 1, "last_run": 5}),
    ],
)
def test_a_corrupt_or_wrong_shaped_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == CongressState()
    assert "state" in caplog.text.lower()


def test_a_state_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("a file where the directory should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        StateStore(blocker / "state.json").save(_state())

    assert "could not be saved" in caplog.text


def test_wipe_removes_the_file_and_is_safe_when_there_is_none(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(_state())

    store.wipe()
    store.wipe()

    assert not (tmp_path / "state.json").exists()
    assert store.load() == CongressState()


def test_the_run_log_appends_one_json_line_per_run(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "logs" / "run" / "runs.jsonl")

    log.append({"date": "2026-10-07", "new_filings": ["20035553"], "orders": [{"symbol": "AAPL", "side": "buy", "quantity": 1.5}]})
    log.append({"date": "2026-10-08", "outcome": "nothing_new"})

    lines = (tmp_path / "logs" / "run" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-10-07", "2026-10-08"]


def test_the_run_log_writes_decimals_as_strings(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs.jsonl")

    log.append({"weight": Decimal("0.1506")})

    assert json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8")) == {"weight": "0.1506"}


def test_a_run_log_without_a_path_does_nothing(tmp_path: Path) -> None:
    RunLog(None).append({"date": "2026-10-07"})

    assert list(tmp_path.iterdir()) == []


def test_a_run_log_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "logs"
    blocker.write_text("a file where the directory should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        RunLog(blocker / "runs.jsonl").append({"date": "2026-10-07"})

    assert "could not be written" in caplog.text

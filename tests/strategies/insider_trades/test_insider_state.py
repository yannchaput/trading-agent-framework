from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.insider_trades.state import InsiderState, PendingTrade, StateStore, state_path


def _note(**changes: object) -> dict[str, object]:
    note: dict[str, object] = {
        "summary": "Three insiders bought on the open market.",
        "class": "buying",
        "overrides": [{"insider_id": "0001234567", "from_label": "routine", "to_label": "opportunistic", "reason": "first buy in years"}],
        "as_of": "2026-10-07",
    }
    return {**note, **changes}


def _state() -> InsiderState:
    return InsiderState(
        processed=["0001-26-000001", "0002-26-000002"],
        in_window={"AAPL": ["0001-26-000001"], "BE": ["0001-26-000001", "0002-26-000002"]},
        notes={"AAPL": _note(), "BE": _note(**{"class": "none", "overrides": []})},
        target={"AAPL": 0.15, "BE": 0.05},
        traded=["AAPL", "BE", "DIS"],
        pending_trade=PendingTrade(target={"AAPL": 0.15, "BE": 0.05}, days=1),
        abandoned_streak=2,
        last_run="2026-10-07",
    )


def test_the_state_path_is_per_mode_under_the_projects_data_directory(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "insider_trades_state_paper.json"
    assert state_path(tmp_path, TradingMode.BACKTESTING).name == "insider_trades_state_backtesting.json"


def test_the_empty_state_has_no_history() -> None:
    state = InsiderState()

    assert (state.processed, state.in_window, state.notes, state.target, state.traded) == ([], {}, {}, {}, [])
    assert (state.pending_trade, state.abandoned_streak, state.last_run) == (None, 0, None)


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == InsiderState()


def test_a_state_round_trips(tmp_path: Path) -> None:
    StateStore(tmp_path / "data" / "state.json").save(_state())

    assert StateStore(tmp_path / "data" / "state.json").load() == _state()


def test_a_state_without_a_pending_trade_round_trips(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    state = InsiderState(processed=["1"], last_run="2026-10-07")

    store.save(state)

    assert store.load() == state
    assert store.load().pending_trade is None


def test_saving_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")

    store.save(_state())
    store.save(InsiderState())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


def _file(**fields: object) -> str:
    return json.dumps({"version": 1, **fields})


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2}),
        _file(processed="A1"),
        _file(processed=[1, 2]),
        _file(in_window=[]),
        _file(in_window={"AAPL": "A1"}),
        _file(in_window={"AAPL": [1]}),
        _file(notes=[]),
        _file(notes={"AAPL": "text"}),
        _file(notes={"AAPL": _note(summary=5)}),
        _file(notes={"AAPL": _note(**{"class": "bullish"})}),
        _file(notes={"AAPL": _note(overrides="none")}),
        _file(notes={"AAPL": _note(overrides=[{"insider_id": "1", "from_label": "a", "to_label": "b"}])}),  # no reason
        _file(notes={"AAPL": _note(overrides=[{"insider_id": 1, "from_label": "a", "to_label": "b", "reason": "r"}])}),
        _file(notes={"AAPL": _note(as_of=20261007)}),
        _file(notes={"AAPL": {key: value for key, value in _note().items() if key != "as_of"}}),
        _file(target={"AAPL": 1.5}),  # a weight above 1
        _file(target={"AAPL": -0.1}),
        _file(target={"AAPL": "0.1"}),
        _file(target=[]),
        _file(traded=[1]),
        _file(pending_trade={"target": {"AAPL": 0.1}}),  # no days
        _file(pending_trade={"target": {"AAPL": 0.1}, "days": -1}),
        _file(pending_trade={"target": {"AAPL": 2.0}, "days": 0}),
        _file(pending_trade={"target": {"AAPL": 0.1}, "days": 0, "extra": 1}),
        _file(pending_trade="yes"),
        _file(abandoned_streak=-1),
        _file(abandoned_streak=True),
        _file(last_run=5),
    ],
)
def test_a_corrupt_or_wrong_shaped_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == InsiderState()
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
    assert store.load() == InsiderState()

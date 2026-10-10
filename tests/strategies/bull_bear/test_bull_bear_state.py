from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bull_bear.state import STATE_VERSION, BullBearState, ReviewLog, StateStore, state_path


def test_the_state_file_is_per_mode_under_data(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "bull_bear_state_paper.json"


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == BullBearState()


def test_a_saved_state_loads_back(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")
    state = BullBearState(last_completed_review="2026-10-06", abandoned_streak=0, last_picks=[{"symbol": "AAA", "reason": "won", "date": "2026-10-06"}])

    store.save(state)

    assert store.load() == state
    assert json.loads((tmp_path / "data" / "state.json").read_text())["version"] == STATE_VERSION


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 99, "abandoned_streak": 1}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": -1}),
        json.dumps({"version": STATE_VERSION, "last_picks": [{"symbol": "AAA"}]}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": True}),
    ],
)
def test_an_unreadable_or_invalid_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content)

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "starting from an empty state" in caplog.text


def test_wipe_deletes_the_file_and_is_safe_without_one(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    store.save(BullBearState(abandoned_streak=2))

    store.wipe()
    store.wipe()

    assert store.load() == BullBearState()


def test_the_review_log_appends_one_json_line_per_review(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "run" / "reviews.jsonl")

    log.append({"date": "2026-10-06", "abandoned": False})
    log.append({"date": "2026-10-13", "abandoned": True})

    lines = (tmp_path / "run" / "reviews.jsonl").read_text().splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-10-06", "2026-10-13"]


def test_a_review_log_without_a_path_does_nothing(tmp_path: Path) -> None:
    ReviewLog(None).append({"date": "2026-10-06"})

    assert list(tmp_path.iterdir()) == []


def test_a_state_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("a file where the directory should be", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        StateStore(blocker / "state.json").save(BullBearState(abandoned_streak=1))

    assert "could not be saved" in caplog.text


def test_a_review_log_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "logs"
    blocker.write_text("a file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ReviewLog(blocker / "reviews.jsonl").append({"date": "2026-10-06"})

    assert "could not be written" in caplog.text

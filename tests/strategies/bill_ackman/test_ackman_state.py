from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.strategies.bill_ackman.state import ReviewLog, ReviewState, StateStore, state_path


def _state() -> ReviewState:
    return ReviewState(
        last_review="2026-09-14",
        fail_counts={"HLT": 1},
        last_ranking=["AAA", "BBB"],
        last_verdicts={"AAA": "survive", "HLT": "fail"},
        abandoned_streak=2,
    )


def test_the_state_path_is_per_mode_under_the_projects_data_directory(tmp_path: Path) -> None:
    assert state_path(tmp_path, TradingMode.PAPER) == tmp_path / "data" / "bill_ackman_state_paper.json"
    assert state_path(tmp_path, TradingMode.BACKTESTING).name == "bill_ackman_state_backtesting.json"


def test_a_missing_file_is_an_empty_state(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.json").load() == ReviewState()


def test_the_empty_state_has_no_history() -> None:
    state = ReviewState()

    assert (state.last_review, state.fail_counts, state.last_ranking, state.last_verdicts, state.abandoned_streak) == (None, {}, [], {}, 0)


def test_a_state_round_trips(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data" / "state.json")

    store.save(_state())

    assert StateStore(tmp_path / "data" / "state.json").load() == _state()


def test_saving_leaves_no_temporary_file(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")

    store.save(_state())
    store.save(ReviewState())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2}),
        json.dumps({"version": 1, "last_review": 5}),
        json.dumps({"version": 1, "fail_counts": {"HLT": "one"}}),
        json.dumps({"version": 1, "fail_counts": {"HLT": 0}}),
        json.dumps({"version": 1, "fail_counts": []}),
        json.dumps({"version": 1, "last_ranking": "AAA"}),
        json.dumps({"version": 1, "last_ranking": [1, 2]}),
        json.dumps({"version": 1, "last_verdicts": {"AAA": "maybe"}}),
        json.dumps({"version": 1, "abandoned_streak": -1}),
        json.dumps({"version": 1, "abandoned_streak": True}),
    ],
)
def test_a_corrupt_or_wrong_shaped_file_is_an_empty_state_with_a_warning(tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == ReviewState()
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
    assert store.load() == ReviewState()


def test_the_review_log_appends_one_json_line_per_review(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "logs" / "run" / "reviews.jsonl")

    log.append({"date": "2026-09-14", "orders": [{"symbol": "AAA", "side": "buy", "quantity": 1.5}]})
    log.append({"date": "2026-09-15", "abandoned": True, "stage": "researcher", "error": "no submission"})

    lines = (tmp_path / "logs" / "run" / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["date"] for line in lines] == ["2026-09-14", "2026-09-15"]


def test_the_review_log_writes_decimals_as_strings(tmp_path: Path) -> None:
    log = ReviewLog(tmp_path / "reviews.jsonl")

    log.append({"market_cap": Decimal("123.45")})

    assert json.loads((tmp_path / "reviews.jsonl").read_text(encoding="utf-8")) == {"market_cap": "123.45"}


def test_a_review_log_without_a_path_does_nothing(tmp_path: Path) -> None:
    ReviewLog(None).append({"date": "2026-09-14"})

    assert list(tmp_path.iterdir()) == []


def test_a_review_log_that_cannot_be_written_is_logged_and_never_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "logs"
    blocker.write_text("a file", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        ReviewLog(blocker / "reviews.jsonl").append({"date": "2026-09-14"})

    assert "could not be written" in caplog.text

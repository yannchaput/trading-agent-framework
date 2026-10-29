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
    state = BullBearState(
        last_completed_review="2026-10-06",
        abandoned_streak=0,
        verdicts={"AAA": {"verdict": "in", "reason": "bull wins", "date": "2026-10-06", "strength": "strong"}},
        cooldowns={"BBB": 3},
        history=[{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "in", "reason": "bull wins", "close": 51.5}}}],
    )

    store.save(state)

    assert store.load() == state
    assert json.loads((tmp_path / "data" / "state.json").read_text())["version"] == STATE_VERSION


def test_a_version_3_state_with_strengths_loads_back(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.json")
    state = BullBearState(
        last_completed_review="2026-10-06",
        verdicts={
            "AAA": {"verdict": "in", "reason": "bull wins", "date": "2026-10-06", "strength": "standard"},
            "BBB": {"verdict": "out", "reason": "bear wins", "date": "2026-10-06", "strength": "weak"},
        },
    )

    store.save(state)

    assert json.loads((tmp_path / "state.json").read_text())["version"] == 3
    assert store.load() == state


def test_a_version_2_state_loads_without_strengths(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    content = {
        "version": 2,
        "last_completed_review": "2026-10-06",
        "abandoned_streak": 1,
        "verdicts": {"AAA": {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}, "BBB": {"verdict": "out", "reason": "bear wins", "date": "2026-09-29"}},
        "cooldowns": {"BBB": 2},
        "history": [{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "in", "reason": "bull wins", "close": 51.5}}}],
    }
    path.write_text(json.dumps(content))

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == BullBearState(
        last_completed_review="2026-10-06",
        abandoned_streak=1,
        verdicts={"AAA": {"verdict": "in", "reason": "bull wins", "date": "2026-10-06"}, "BBB": {"verdict": "out", "reason": "bear wins", "date": "2026-09-29"}},
        cooldowns={"BBB": 2},
        history=content["history"],
    )
    assert caplog.text == ""


def test_saving_a_state_keeps_a_verdict_that_predates_strengths(tmp_path: Path) -> None:
    # a version-2 verdict no review has judged again has no strength: the file still holds it and loads back whole
    store = StateStore(tmp_path / "state.json")
    state = BullBearState(
        verdicts={
            "AAA": {"verdict": "in", "reason": "old", "date": "2026-09-29"},
            "BBB": {"verdict": "out", "reason": "new", "date": "2026-10-06", "strength": "weak"},
        },
        cooldowns={"AAA": 1},
    )

    store.save(state)

    assert json.loads((tmp_path / "state.json").read_text())["version"] == STATE_VERSION
    assert store.load() == state


def test_a_version_3_entry_without_a_strength_loads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    entry = {"verdict": "in", "reason": "r", "date": "2026-10-06"}
    path.write_text(json.dumps({"version": 3, "last_completed_review": "2026-10-06", "verdicts": {"AAA": entry}}))

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state.verdicts == {"AAA": entry}
    assert caplog.text == ""


@pytest.mark.parametrize(
    "entry",
    [
        {"verdict": "in", "reason": "r", "date": "2026-10-06", "strength": "high"},
        {"verdict": "in", "reason": "r", "date": "2026-10-06", "strength": None},
        {"verdict": "in", "reason": "r", "date": "2026-10-06", "strength": 3},
        {"verdict": "in", "reason": "r", "date": "2026-10-06", "strength": "weak"},
    ],
)
def test_an_unknown_non_string_or_weak_in_strength_is_an_invalid_state(tmp_path: Path, entry: dict[str, object], caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 3, "last_completed_review": "2026-10-06", "verdicts": {"AAA": entry}}))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "not a valid state" in caplog.text


def test_a_version_2_entry_with_a_strength_is_an_invalid_state(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 2, "verdicts": {"AAA": {"verdict": "in", "reason": "r", "date": "2026-10-06", "strength": "strong"}}}))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "not a valid state" in caplog.text


def test_a_state_from_the_picker_or_the_veto_judge_keeps_its_review_date_and_streak(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "last_completed_review": "2026-10-06", "abandoned_streak": 1, "last_book": [{"symbol": "AAA", "reason": "won", "date": "2026-10-06"}]}))

    with caplog.at_level(logging.WARNING):
        state = StateStore(path).load()

    assert state == BullBearState(last_completed_review="2026-10-06", abandoned_streak=1)  # a completed Tuesday is not reviewed again
    assert caplog.text == ""


def test_a_version_1_state_without_a_review_date_keeps_its_streak(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "last_completed_review": None, "abandoned_streak": 2}))

    assert StateStore(path).load() == BullBearState(abandoned_streak=2)


@pytest.mark.parametrize(
    "content",
    [
        {"version": 1, "last_completed_review": 20261006, "abandoned_streak": 0},
        {"version": 1, "last_completed_review": "2026-10-06", "abandoned_streak": -1},
        {"version": 1, "last_completed_review": "2026-10-06", "abandoned_streak": True},
        {"version": 1, "last_completed_review": "2026-10-06", "abandoned_streak": "1"},
        {"version": 4, "last_completed_review": "2026-10-06", "abandoned_streak": 1},
    ],
)
def test_an_invalid_version_1_state_or_another_version_is_an_empty_state_with_a_warning(tmp_path: Path, content: dict[str, object], caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(content))

    with caplog.at_level(logging.WARNING):
        assert StateStore(path).load() == BullBearState()
    assert "starting from an empty state" in caplog.text


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"version": 99, "abandoned_streak": 1}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": -1}),
        json.dumps({"version": STATE_VERSION, "verdicts": {"AAA": {"verdict": "in"}}}),
        json.dumps({"version": STATE_VERSION, "verdicts": ["AAA"]}),
        json.dumps({"version": STATE_VERSION, "abandoned_streak": True}),
        json.dumps({"version": STATE_VERSION, "cooldowns": {"AAA": 0}}),
        json.dumps({"version": STATE_VERSION, "cooldowns": {"AAA": True}}),
        json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "in", "reason": "r"}}}]}),
        json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06"}]}),
        json.dumps({"version": STATE_VERSION, "verdicts": {"AAA": {"verdict": "maybe", "reason": "r", "date": "2026-10-06", "strength": "weak"}}}),
        json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": "veto", "reason": "r", "close": 1.0}}}]}),
        json.dumps({"version": STATE_VERSION, "history": [{"date": "2026-10-06", "verdicts": {"AAA": {"verdict": ["in"], "reason": "r", "close": 1.0}}}]}),
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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trading_agent_framework.dashboard import reviews_reader as rr
from trading_agent_framework.dashboard.models import RunRef

ACKMAN_REVIEW: dict[str, Any] = {
    "date": "2021-09-27",
    "run_id": "2026-10-05_193526_backtesting",
    "abandoned": False,
    "candidates": [{"symbol": "RMD", "rank": 1, "fcf_yield": 0.16}],
    "holdings": ["INTC", "LEN"],
    "forced_exits": ["TSN"],
    "targets": {"LEN": 0.1, "INTC": 0.2, "SHV": 0.08},
    "orders": [{"symbol": "SHV", "side": "sell", "quantity": 21.651797}],
}

BULL_BEAR_REVIEW: dict[str, Any] = {
    "date": "2026-10-06",
    "run_id": "2026-10-09_101500_backtesting",
    "abandoned": False,
    "debate_set": [{"symbol": "AAPL", "rank": 1, "held": True}, {"symbol": "MSFT", "rank": 2, "held": False}],
    "forced_exits": [{"symbol": "KO", "reason": "ranked worse than 35"}],
    "targets": {"AAPL": 0.5, "SHV": 0.48},
    "orders": [],
}

ABANDONED_REVIEW: dict[str, Any] = {
    "date": "2026-10-13",
    "run_id": "2026-10-09_101500_backtesting",
    "abandoned": True,
    "stage": "judge",
    "error": "the agent ended without calling submit_picks",
    "abandoned_streak": 1,
}


RUNS_RECORD: dict[str, Any] = {
    "date": "2026-10-06",
    "run_id": "2026-10-09_101500_backtesting",
    "outcome": "completed_pending",
    "new_filings": [],
    "target": [{"ticker": "AAPL", "weight": 0.095, "reason": None}],
}


def _run_with_runs(tmp_path: Path, lines: list[str]) -> RunRef:
    run_dir = tmp_path / "logs" / "insider_trades" / "backtesting" / "2026-10-09_101500_backtesting"
    run_dir.mkdir(parents=True)
    (run_dir / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunRef.from_path(str(run_dir))


def _run_with_reviews(tmp_path: Path, lines: list[str] | None) -> RunRef:
    run_dir = tmp_path / "logs" / "bull_bear" / "backtesting" / "2026-10-09_101500_backtesting"
    run_dir.mkdir(parents=True)
    if lines is not None:
        (run_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunRef.from_path(str(run_dir))


def test_load_reads_every_record_in_file_order(tmp_path: Path) -> None:
    ref = _run_with_reviews(tmp_path, [json.dumps(BULL_BEAR_REVIEW), json.dumps(ABANDONED_REVIEW)])

    loaded = rr.load_reviews(ref)

    assert [record["date"] for record in loaded.records] == ["2026-10-06", "2026-10-13"]
    assert loaded.malformed == 0


def test_load_skips_and_counts_malformed_lines(tmp_path: Path) -> None:
    ref = _run_with_reviews(tmp_path, [json.dumps(BULL_BEAR_REVIEW), '{"date": "2026-10-1', "[1, 2]", "", json.dumps(ABANDONED_REVIEW)])

    loaded = rr.load_reviews(ref)

    assert [record["date"] for record in loaded.records] == ["2026-10-06", "2026-10-13"]
    assert loaded.malformed == 2  # the truncated line and the non-object; the blank line is not a record


def test_has_reviews_tells_a_run_with_the_file_from_one_without(tmp_path: Path) -> None:
    with_file = _run_with_reviews(tmp_path / "a", [json.dumps(BULL_BEAR_REVIEW)])
    without = _run_with_reviews(tmp_path / "b", None)

    assert rr.has_reviews(with_file)
    assert not rr.has_reviews(without)


def test_has_reviews_also_accepts_a_runs_file(tmp_path: Path) -> None:
    ref = _run_with_runs(tmp_path, [json.dumps(RUNS_RECORD)])

    assert rr.has_reviews(ref)


def test_load_reads_a_runs_file_when_there_is_no_reviews_file(tmp_path: Path) -> None:
    ref = _run_with_runs(tmp_path, [json.dumps(RUNS_RECORD), "not json"])

    loaded = rr.load_reviews(ref)

    assert [record["outcome"] for record in loaded.records] == ["completed_pending"]
    assert loaded.malformed == 1


def test_load_prefers_the_reviews_file_over_a_runs_file(tmp_path: Path) -> None:
    ref = _run_with_reviews(tmp_path, [json.dumps(BULL_BEAR_REVIEW)])
    (Path(ref.path) / "runs.jsonl").write_text(json.dumps(RUNS_RECORD) + "\n", encoding="utf-8")

    assert [record["date"] for record in rr.load_reviews(ref).records] == ["2026-10-06"]


def test_load_of_a_run_with_neither_file_raises_file_not_found(tmp_path: Path) -> None:
    ref = _run_with_reviews(tmp_path, None)

    with pytest.raises(FileNotFoundError):
        rr.load_reviews(ref)


def test_frame_has_one_column_per_key_in_first_seen_order_across_records() -> None:
    frame = rr.reviews_frame([BULL_BEAR_REVIEW, ABANDONED_REVIEW])

    assert list(frame.columns) == ["date", "run_id", "abandoned", "debate_set", "forced_exits", "targets", "orders", "stage", "error", "abandoned_streak"]


def test_frame_keeps_scalars_and_leaves_a_missing_key_empty() -> None:
    frame = rr.reviews_frame([BULL_BEAR_REVIEW, ABANDONED_REVIEW])

    assert list(frame["abandoned"]) == [False, True]
    assert frame["error"].isna().tolist() == [True, False]
    assert frame["error"][1] == "the agent ended without calling submit_picks"
    assert frame["abandoned_streak"][1] == 1


def test_frame_shows_a_list_or_a_dict_as_compact_json() -> None:
    row = rr.reviews_frame([ACKMAN_REVIEW]).iloc[0]

    assert row["holdings"] == '["INTC", "LEN"]'
    assert row["targets"] == '{"LEN": 0.1, "INTC": 0.2, "SHV": 0.08}'
    assert row["orders"] == '[{"symbol": "SHV", "side": "sell", "quantity": 21.651797}]'


def test_frame_shows_the_runs_file_columns() -> None:
    frame = rr.reviews_frame([RUNS_RECORD])

    assert list(frame.columns) == ["date", "run_id", "outcome", "new_filings", "target"]
    assert frame["outcome"][0] == "completed_pending"
    assert frame["target"][0] == '[{"ticker": "AAPL", "weight": 0.095, "reason": null}]'


def test_frame_turns_a_column_of_mixed_types_into_text() -> None:
    frame = rr.reviews_frame([{"date": "2026-10-06", "failed": {}}, {"date": "2026-10-13", "failed": "boom"}])

    assert list(frame["failed"]) == ["{}", "boom"]


def test_frame_of_no_records_is_empty() -> None:
    frame = rr.reviews_frame([])

    assert frame.empty
    assert list(frame.columns) == []

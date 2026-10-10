from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_agent_framework.dashboard import reviews_reader as rr

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


def _write_reviews(logs: Path, strategy: str, mode: str, run_ts: str, lines: list[str]) -> Path:
    run_dir = logs / strategy / mode / f"{run_ts}_{mode}"
    run_dir.mkdir(parents=True)
    (run_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_scan_lists_backtesting_runs_with_a_reviews_file_newest_first_per_strategy(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_reviews(logs, "bull_bear", "backtesting", "2026-10-09_101500", [json.dumps(BULL_BEAR_REVIEW)])
    _write_reviews(logs, "bill_ackman", "backtesting", "2026-10-03_231803", [json.dumps(ACKMAN_REVIEW)])
    _write_reviews(logs, "bill_ackman", "backtesting", "2026-10-05_193526", [json.dumps(ACKMAN_REVIEW)])
    _write_reviews(logs, "bill_ackman", "paper", "2026-10-04_224802", [json.dumps(ACKMAN_REVIEW)])
    (logs / "cross_momentum" / "backtesting" / "2026-10-01_000000_backtesting").mkdir(parents=True)  # no reviews.jsonl

    refs = rr.scan_review_runs(logs)

    assert [(ref.strategy_name, ref.run_ts) for ref in refs] == [
        ("bill_ackman", "2026-10-05_193526"),
        ("bill_ackman", "2026-10-03_231803"),
        ("bull_bear", "2026-10-09_101500"),
    ]


def test_scan_of_a_missing_logs_directory_is_empty(tmp_path: Path) -> None:
    assert rr.scan_review_runs(tmp_path / "nope") == []


def test_load_reads_every_record_in_file_order(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_reviews(logs, "bull_bear", "backtesting", "2026-10-09_101500", [json.dumps(BULL_BEAR_REVIEW), json.dumps(ABANDONED_REVIEW)])
    (ref,) = rr.scan_review_runs(logs)

    loaded = rr.load_reviews(ref)

    assert [record["date"] for record in loaded.records] == ["2026-10-06", "2026-10-13"]
    assert loaded.malformed == 0


def test_load_skips_and_counts_malformed_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write_reviews(logs, "bull_bear", "backtesting", "2026-10-09_101500", [json.dumps(BULL_BEAR_REVIEW), '{"date": "2026-10-1', "[1, 2]", "", json.dumps(ABANDONED_REVIEW)])
    (ref,) = rr.scan_review_runs(logs)

    loaded = rr.load_reviews(ref)

    assert [record["date"] for record in loaded.records] == ["2026-10-06", "2026-10-13"]
    assert loaded.malformed == 2  # the truncated line and the non-object; the blank line is not a record


def test_frame_summarises_a_bill_ackman_review() -> None:
    frame = rr.reviews_frame([ACKMAN_REVIEW])

    assert list(frame.columns) == ["Date", "Status", "Stage", "Holdings", "Targets", "Forced exits", "Orders", "Error"]
    row = frame.iloc[0]
    assert row["Date"] == "2021-09-27"
    assert row["Status"] == "completed"
    assert row["Holdings"] == "INTC, LEN"
    assert row["Targets"] == "INTC 20.0%, LEN 10.0%, SHV 8.0%"  # heaviest first
    assert row["Forced exits"] == "TSN"
    assert row["Orders"] == "SELL SHV 21.651797"


def test_frame_reads_bull_bear_holdings_from_the_debate_set_and_forced_exits_from_dicts() -> None:
    row = rr.reviews_frame([BULL_BEAR_REVIEW]).iloc[0]

    assert row["Holdings"] == "AAPL"
    assert row["Forced exits"] == "KO"
    assert row["Orders"] == ""


def test_frame_shows_an_abandoned_review_with_its_stage_and_error() -> None:
    row = rr.reviews_frame([ABANDONED_REVIEW]).iloc[0]

    assert row["Status"] == "abandoned"
    assert row["Stage"] == "judge"
    assert row["Error"] == "the agent ended without calling submit_picks"
    assert row["Targets"] == ""


def test_frame_of_no_records_keeps_its_columns() -> None:
    frame = rr.reviews_frame([])

    assert frame.empty
    assert "Date" in frame.columns

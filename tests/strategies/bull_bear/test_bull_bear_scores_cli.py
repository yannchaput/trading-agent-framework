"""`uv run bull-bear-scores <run_dir>...`: bull_bear runs scored side by side; a fake download, never the network."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_agent_framework.strategies.bull_bear.batch.scores_cli import main

TODAY = date(2026, 4, 30)
SESSIONS = 60  # business days from Friday 2026-01-02


def _yahoo_frame(opens: list[float]) -> pd.DataFrame:
    """One ticker as yfinance returns it: Open/High/Low/Close/Volume, a naive session-date index."""
    index = pd.bdate_range(start="2026-01-02", periods=len(opens))
    return pd.DataFrame({"Open": opens, "High": [o + 1 for o in opens], "Low": [o - 1 for o in opens], "Close": opens, "Volume": [1e6] * len(opens)}, index=index)


def _batch(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """What `yfinance.download(symbols, group_by="ticker")` returns: (ticker, field) columns."""
    return pd.concat(frames, axis=1)


class FakeDownload:
    def __init__(self, result: pd.DataFrame | None = None, raises: Exception | None = None) -> None:
        self.result, self.raises = result, raises
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, symbols: list[str], **kwargs: Any) -> pd.DataFrame | None:
        self.calls.append((list(symbols), kwargs))
        if self.raises is not None:
            raise self.raises
        return self.result


def _prices(*, spy: bool = True) -> pd.DataFrame:
    frames = {"AAA": _yahoo_frame([100.0 * 1.01**i for i in range(SESSIONS)]), "BBB": _yahoo_frame([100.0 * 0.99**i for i in range(SESSIONS)])}
    if spy:
        frames["SPY"] = _yahoo_frame([500.0] * SESSIONS)
    return _batch(frames)


def _judge_review(day: str) -> dict[str, Any]:
    return {
        "date": day,
        "abandoned": False,
        "debate_set": [{"symbol": "AAA", "rank": 1, "held": False}, {"symbol": "BBB", "rank": 2, "held": False}],
        "bull": [{"symbol": "AAA", "conviction": "high", "argument": "x"}, {"symbol": "BBB", "conviction": "low", "argument": "x"}],
        "bear": [{"symbol": "AAA", "risk": "low", "concern": "none", "argument": "x"}, {"symbol": "BBB", "risk": "high", "concern": "valuation", "argument": "x"}],
        "verdicts": [{"symbol": "AAA", "verdict": "in", "reason": "x"}, {"symbol": "BBB", "verdict": "out", "reason": "x"}],
    }


def _picker_review(day: str) -> dict[str, Any]:
    record = _judge_review(day)
    del record["verdicts"]
    return {**record, "picks": [{"symbol": "AAA", "reason": "x"}]}


def _run(tmp_path: Path, name: str, lines: list[str]) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def test_two_runs_print_the_bull_the_bear_and_the_judge_side_by_side(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    judge = _run(tmp_path, "judge_run", [json.dumps(_judge_review("2026-01-06")), json.dumps(_judge_review("2026-01-13"))])
    picker = _run(tmp_path, "picker_run", [json.dumps(_picker_review("2026-01-06")), "not json"])

    assert main([str(picker), str(judge)], download=FakeDownload(_prices()), today=TODAY) == 0

    out = capsys.readouterr().out
    for title in ("BULL conviction, spread high - low", "BEAR risk, spread low - high", "JUDGE verdict, spread in - out"):
        assert title in out
    assert "concern valuation" in out
    assert "picker_run: 1 reviews scored" in out and "1 malformed lines" in out
    assert "judge_run: 2 reviews scored" in out
    spread_rows = [line for line in out.splitlines() if line.startswith("  spread")]
    assert len(spread_rows) == 3 and all("+" in row for row in spread_rows)  # AAA rises, BBB falls: every agent is right


def test_each_run_reports_the_reviews_without_a_spread_per_agent_and_horizon(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record = _judge_review("2026-01-06")
    record["bull"] = [{"symbol": "AAA", "conviction": "high", "argument": "x"}, {"symbol": "BBB", "conviction": "high", "argument": "x"}]  # no low: no bull spread
    run = _run(tmp_path, "skipping", [json.dumps(record), json.dumps(_judge_review("2026-01-13"))])

    assert main([str(run)], download=FakeDownload(_prices()), today=TODAY) == 0

    out = capsys.readouterr().out
    assert "  reviews without a spread (a group missing): bull 5d 1, 20d 1; bear 5d 0, 20d 0; judge 5d 0, 20d 0" in out


def test_exits_past_the_end_of_the_data_are_counted_apart_from_missing_prices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    late = _run(tmp_path, "late", [json.dumps(_judge_review("2026-03-17"))])  # entry 2026-03-18: the 5d exit exists, the 20d one is past the data

    assert main([str(late)], download=FakeDownload(_prices()), today=TODAY) == 0

    out = capsys.readouterr().out
    assert "  candidate x horizon pairs left out: 0 without prices, 2 whose exit is past the end of the data" in out


def test_the_download_covers_every_run_from_a_week_before_the_first_review_to_forty_days_after_the_last(tmp_path: Path) -> None:
    first = _run(tmp_path, "a", [json.dumps(_judge_review("2026-01-06"))])
    second = _run(tmp_path, "b", [json.dumps(_judge_review("2026-01-13"))])
    download = FakeDownload(_prices())

    main([str(first), str(second)], download=download, today=TODAY)

    ((symbols, kwargs),) = download.calls
    assert sorted(symbols) == ["AAA", "BBB", "SPY"]
    assert kwargs["start"] <= "2025-12-30" and kwargs["end"] == "2026-02-23"  # 2026-01-13 + 40 days, end exclusive


def test_the_download_window_stops_at_today(tmp_path: Path) -> None:
    run = _run(tmp_path, "running", [json.dumps(_judge_review("2026-01-13"))])
    download = FakeDownload(_prices())

    main([str(run)], download=download, today=date(2026, 1, 20))

    assert download.calls[0][1]["end"] == "2026-01-21"


def test_a_run_directory_without_reviews_exits_2(tmp_path: Path) -> None:
    empty = tmp_path / "no_reviews"
    empty.mkdir()

    with pytest.raises(SystemExit) as exit_info:
        main([str(empty)], download=FakeDownload(_prices()), today=TODAY)

    assert exit_info.value.code == 2


def test_a_failing_download_exits_1_with_a_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path, "r", [json.dumps(_judge_review("2026-01-06"))])

    assert main([str(run)], download=FakeDownload(raises=RuntimeError("rate limited")), today=TODAY) == 1

    assert "rate limited" in capsys.readouterr().err


def test_no_spy_in_the_download_exits_1_naming_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = _run(tmp_path, "r", [json.dumps(_judge_review("2026-01-06"))])

    assert main([str(run)], download=FakeDownload(_prices(spy=False)), today=TODAY) == 1

    assert "SPY" in capsys.readouterr().err


def test_a_run_from_another_strategy_prints_a_report_with_no_review_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ackman = _run(tmp_path, "ackman_run", [json.dumps({"date": "2026-01-06", "abandoned": False, "holdings": ["AAA"], "targets": {"AAA": 0.5}})])

    assert main([str(ackman)], download=FakeDownload(_prices()), today=TODAY) == 0

    assert "ackman_run: 0 reviews scored" in capsys.readouterr().out

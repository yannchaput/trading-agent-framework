from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def _agents_script() -> None:
    # Serialized by AppTest.from_function: must be self-contained.
    from trading_agent_framework.dashboard._pages.agents import page_agents

    page_agents()


def _write_reviews(logs: Path, strategy: str, run_ts: str, records: list[dict]) -> None:
    run_dir = logs / strategy / "backtesting" / f"{run_ts}_backtesting"
    run_dir.mkdir(parents=True)
    (run_dir / "reviews.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # the page reads ./logs, like the scorecard
    return tmp_path / "logs"


def _run() -> AppTest:
    at = AppTest.from_function(_agents_script, default_timeout=30)
    at.run()
    return at


def test_without_any_reviews_file_the_page_says_where_they_come_from(logs: Path) -> None:
    at = _run()

    assert not at.exception
    assert [title.value for title in at.title] == ["Agents"]
    assert len(at.dataframe) == 0
    assert any("reviews.jsonl" in info.value for info in at.info)


def test_the_page_shows_one_table_row_per_review(logs: Path) -> None:
    _write_reviews(
        logs,
        "bull_bear",
        "2026-10-09_101500",
        [
            {"date": "2026-10-06", "abandoned": False, "targets": {"AAPL": 0.5, "SHV": 0.48}, "orders": [{"symbol": "AAPL", "side": "buy", "quantity": 3}]},
            {"date": "2026-10-13", "abandoned": True, "stage": "judge", "error": "no valid submission"},
        ],
    )

    at = _run()

    assert not at.exception
    (table,) = at.dataframe
    assert list(table.value["Date"]) == ["2026-10-06", "2026-10-13"]
    assert list(table.value["Status"]) == ["completed", "abandoned"]
    assert table.value["Orders"][0] == "BUY AAPL 3"


def test_the_run_picker_lists_runs_newest_first_and_switches_the_table(logs: Path) -> None:
    _write_reviews(logs, "bill_ackman", "2026-10-03_231803", [{"date": "2021-09-20", "abandoned": False}])
    _write_reviews(logs, "bill_ackman", "2026-10-05_193526", [{"date": "2022-01-03", "abandoned": False}, {"date": "2022-01-10", "abandoned": False}])

    at = _run()

    picker = at.sidebar.selectbox[0]
    assert [option.split(" · ")[1] for option in picker.options] == ["2026-10-05_193526", "2026-10-03_231803"]
    assert list(at.dataframe[0].value["Date"]) == ["2022-01-03", "2022-01-10"]

    picker.select_index(1).run()

    assert list(at.dataframe[0].value["Date"]) == ["2021-09-20"]


def test_the_caption_counts_reviews_abandoned_and_unreadable_lines(logs: Path) -> None:
    run_dir = logs / "bull_bear" / "backtesting" / "2026-10-09_101500_backtesting"
    run_dir.mkdir(parents=True)
    lines = [json.dumps({"date": "2026-10-06", "abandoned": False}), json.dumps({"date": "2026-10-13", "abandoned": True}), "{not json"]
    (run_dir / "reviews.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    caption = " ".join(c.value for c in _run().caption)

    assert "2 reviews" in caption
    assert "1 abandoned" in caption
    assert "1 unreadable line" in caption


def test_a_review_can_be_opened_as_its_raw_record(logs: Path) -> None:
    _write_reviews(
        logs,
        "bill_ackman",
        "2026-10-05_193526",
        [{"date": "2022-01-03", "abandoned": False, "verdicts": [{"symbol": "BBY", "verdict": "survive", "reason": "net cash"}]}],
    )

    at = _run()

    assert not at.exception
    assert at.main.selectbox[0].value == "2022-01-03"
    assert any("net cash" in json_el.value for json_el in at.json)

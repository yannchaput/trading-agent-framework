from __future__ import annotations

from pathlib import Path

from trading_agent_framework.dashboard.discovery import scan_runs


def _make_run_dir(base: Path, strategy: str, run_ts: str) -> Path:
    run_dir = base / strategy / "backtesting" / f"{run_ts}_backtesting"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "settings.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_scan_runs_finds_runs_by_metrics_json(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, "momentum", "2026-06-22_194053")
    _make_run_dir(tmp_path, "mean_reversion", "2026-06-23_090000")

    index = scan_runs(str(tmp_path))

    assert sorted(index.strategy_names()) == ["mean_reversion", "momentum"]
    assert len(index.runs) == 2


def test_scan_runs_ignores_paper_and_live_run_dirs_that_have_no_metrics_json(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, "momentum", "2026-06-22_194053")
    paper_dir = tmp_path / "momentum" / "paper" / "2026-06-23_100000_paper"
    paper_dir.mkdir(parents=True)
    (paper_dir / "some_other_file.log").write_text("x", encoding="utf-8")

    index = scan_runs(str(tmp_path))

    assert len(index.runs) == 1
    assert index.runs[0].mode == "backtesting"


def test_scan_runs_returns_empty_index_for_empty_logs_dir(tmp_path: Path) -> None:
    index = scan_runs(str(tmp_path))
    assert index.runs == []

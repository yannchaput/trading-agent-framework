from __future__ import annotations

from pathlib import Path

from tests.backtesting.dashboard_contract import METRIC_SET_FIELDS

from trading_agent_framework.backtesting import report
from trading_agent_framework.dashboard.models import RunRef
from trading_agent_framework.dashboard.reader import load_metrics


def _run_dir(tmp_path: Path, strategy: str = "momentum", run_ts: str = "2026-06-22_194053") -> Path:
    d = tmp_path / strategy / "backtesting" / f"{run_ts}_backtesting"
    d.mkdir(parents=True)
    return d


def _ref(run_dir: Path) -> RunRef:
    return RunRef.from_path(str(run_dir))


def test_load_metrics_reads_a_full_metrics_json(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    metrics = dict.fromkeys(METRIC_SET_FIELDS - {"raw"}, 0.0)
    metrics["total_return_strategy"] = 0.42
    metrics["sharpe_strategy"] = 1.5
    metrics["raw"] = {"summary_tables": {"eoy_returns_vs_benchmark": [{"year": 2026, "strategy": 0.1, "benchmark": 0.05, "won": True}], "drawdowns": []}}
    report.write_metrics(run_dir, metrics)

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.42
    assert loaded.sharpe_strategy == 1.5
    assert loaded.raw["summary_tables"]["eoy_returns_vs_benchmark"][0]["year"] == 2026


def test_load_metrics_defaults_a_benchmark_less_runs_null_fields_to_zero(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    # A benchmark-less run: compute_metrics only returns strategy-side fields;
    # write_metrics fills every *_benchmark/relative field with JSON null.
    report.write_metrics(run_dir, {"total_return_strategy": 0.1, "raw": {}})

    loaded = load_metrics(_ref(run_dir))

    assert loaded is not None
    assert loaded.total_return_strategy == 0.1
    assert loaded.sharpe_benchmark == 0.0
    assert loaded.beta == 0.0


def test_load_metrics_returns_none_when_metrics_json_is_missing(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path)
    assert load_metrics(_ref(run_dir)) is None

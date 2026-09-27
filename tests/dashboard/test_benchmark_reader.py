from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from tests.dashboard.benchmark_fixtures import IN_PROGRESS, LATEST, OLDER, SCENARIOS, build_results_tree, write_json

from trading_agent_framework.dashboard.benchmark_reader import (
    BenchmarkReadError,
    load_benchmark_run,
    load_scenario_runs,
    resolve_results_dir,
    scan_benchmark_runs,
    split_benchmark_dir,
)
from trading_agent_framework.dashboard.models import BenchmarkRunRef, Check, ScenarioScore


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def _ref(results: Path, run_id: str) -> BenchmarkRunRef:
    return BenchmarkRunRef.from_path(results / run_id)


# --- scan ------------------------------------------------------------------------------


def test_scan_lists_only_complete_timestamped_runs_newest_first(results: Path) -> None:
    refs = scan_benchmark_runs(results)

    assert [ref.run_id for ref in refs] == [LATEST, OLDER]
    assert IN_PROGRESS not in [ref.run_id for ref in refs]
    assert refs[0].started_at == datetime(2026, 9, 27, 13, 22, 43)


def test_scan_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert scan_benchmark_runs(tmp_path / "nope") == []


def test_run_ref_rejects_a_non_timestamp_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        BenchmarkRunRef.from_path(tmp_path / "notes")


# --- load_benchmark_run ----------------------------------------------------------------------------


def test_load_joins_meta_and_summary_by_key(results: Path) -> None:
    run = load_benchmark_run(_ref(results, LATEST))

    q27 = next(m for m in run.models if m.key == "qwen3627b")
    assert q27.display_name == "Qwen3.6-27B-AWQ"
    assert q27.served_name == "qwen3.6-27b-awq"
    assert q27.vllm_version == "0.30.0"
    assert q27.overall == 0.83
    assert q27.categories == {"reasoning": 0.75, "tools": 1.0}
    assert q27.scenarios["reasoning.rsi_signal"] == ScenarioScore(passed=1, runs=2, mean_partial=0.96)
    assert (q27.runs_passed, q27.runs_total, q27.text_tool_calls, q27.timeouts) == (5, 6, 1, 1)
    assert run.repeats == 2
    assert run.timeout_s == 360.0
    assert run.finished_at == datetime(2026, 9, 27, 16, 8, 15)


def test_scenario_and_category_order_follow_meta_not_summary(results: Path) -> None:
    run = load_benchmark_run(_ref(results, LATEST))

    assert run.scenarios == tuple(SCENARIOS)
    assert run.categories == ("reasoning", "tools")


def test_a_model_that_did_not_run_is_kept_with_its_error(results: Path) -> None:
    run = load_benchmark_run(_ref(results, OLDER))

    gptoss = next(m for m in run.models if m.key == "gptoss")
    assert gptoss.ran is False
    assert gptoss.error == "vLLM server failed to start"
    assert gptoss.overall is None
    assert gptoss.median_run_s is None


def test_a_summary_model_missing_from_meta_has_no_served_name_or_version(results: Path) -> None:
    run_dir = results / LATEST
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.append({**summary[0], "key": "ghost", "display_name": "Ghost"})
    write_json(run_dir / "summary.json", summary)

    ghost = next(m for m in load_benchmark_run(_ref(results, LATEST)).models if m.key == "ghost")
    assert ghost.display_name == "Ghost"
    assert ghost.served_name is None
    assert ghost.vllm_version is None


@pytest.mark.parametrize("content", ["{not json", '{"not": "a list"}'])
def test_a_malformed_summary_raises_benchmark_read_error(results: Path, content: str) -> None:
    (results / LATEST / "summary.json").write_text(content, encoding="utf-8")

    with pytest.raises(BenchmarkReadError):
        load_benchmark_run(_ref(results, LATEST))


def test_a_meta_without_scenarios_raises_benchmark_read_error(results: Path) -> None:
    write_json(results / LATEST / "meta.json", {"models": [], "repeats": 2, "timeout_s": 360.0})

    with pytest.raises(BenchmarkReadError):
        load_benchmark_run(_ref(results, LATEST))


# --- results dir -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "rest", "value"),
    [
        (["--server.port", "8502"], ["--server.port", "8502"], None),
        (["--benchmark-dir", "/data/r", "--server.port", "8502"], ["--server.port", "8502"], "/data/r"),
        (["--benchmark-dir=/data/r"], [], "/data/r"),
    ],
)
def test_split_benchmark_dir(argv: list[str], rest: list[str], value: str | None) -> None:
    assert split_benchmark_dir(argv) == (rest, value)


def test_split_benchmark_dir_without_a_value_is_an_error() -> None:
    with pytest.raises(SystemExit):
        split_benchmark_dir(["--benchmark-dir"])


def test_the_flag_wins_and_is_expanded_and_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_results_dir(["--benchmark-dir", "~/bench/results"]) == (tmp_path / "bench" / "results").resolve()


def test_the_default_sits_next_to_the_project_root_not_the_cwd(tmp_path: Path) -> None:
    project = tmp_path / "trading-agent-framework"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert resolve_results_dir([], start=project / "src") == (tmp_path / "benchmark-vllm-models" / "results").resolve()


# --- load_scenario_runs ------------------------------------------------------------------------


def test_scenario_runs_are_filtered_and_sorted_by_repeat(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "reasoning.rsi_signal")

    assert [run.repeat for run in result.runs] == [1, 2]
    first = result.runs[0]
    assert (first.status, first.passed, first.partial) == ("ok", True, 1.0)
    assert first.checks[0] == Check(type="called", passed=True, reason="found")
    assert (first.total_s, first.tool_calls, first.model_calls, first.tokens_per_s, first.completion_tokens) == (
        6.906, 4, 5, 121.35, 882,
    )


def test_a_timed_out_run_is_kept_with_its_error(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "reasoning.rsi_signal")

    timed_out = result.runs[1]
    assert timed_out.status == "timeout"
    assert timed_out.error == "run exceeded 360s"
    assert timed_out.checks == ()


def test_malformed_lines_are_skipped_and_counted(results: Path) -> None:
    result = load_scenario_runs(_ref(results, LATEST), "qwen3627b", "tools.limit_order")

    assert len(result.runs) == 1
    assert result.skipped_lines == 1


def test_the_message_trace_is_not_kept(results: Path) -> None:
    run = load_scenario_runs(_ref(results, LATEST), "glm", "reasoning.rsi_signal").runs[0]

    assert not hasattr(run, "trace")


def test_a_missing_model_file_raises_benchmark_read_error(results: Path) -> None:
    with pytest.raises(BenchmarkReadError):
        load_scenario_runs(_ref(results, LATEST), "nope", "reasoning.rsi_signal")

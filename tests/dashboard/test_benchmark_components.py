from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from tests.dashboard.benchmark_fixtures import LATEST, OLDER, SCENARIOS, build_results_tree

from trading_agent_framework.dashboard.benchmark_reader import load_benchmark_run, load_scenario_runs
from trading_agent_framework.dashboard.components.charts import (
    CHART_TEMPLATE,
    benchmark_category_chart,
    benchmark_scenario_heatmap,
    benchmark_speed_quality_chart,
)
from trading_agent_framework.dashboard.components.tables import benchmark_summary_frame, scenario_runs_frame
from trading_agent_framework.dashboard.models import BenchmarkRun, BenchmarkRunRef


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return build_results_tree(tmp_path / "results")


def _run(results: Path, run_id: str = LATEST) -> BenchmarkRun:
    return load_benchmark_run(BenchmarkRunRef.from_path(results / run_id))


def test_category_chart_has_one_horizontal_bar_trace_per_model(results: Path) -> None:
    run = _run(results)
    fig = benchmark_category_chart(run.models, run.categories)

    assert [trace.name for trace in fig.data] == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert all(trace.orientation == "h" for trace in fig.data)
    assert list(fig.data[1].y) == ["Reasoning", "Tools"]
    assert list(fig.data[1].x) == [75.0, 100.0]
    assert fig.layout.template.layout.paper_bgcolor == CHART_TEMPLATE.layout.paper_bgcolor


def test_speed_quality_chart_plots_run_time_against_overall_score(results: Path) -> None:
    fig = benchmark_speed_quality_chart(_run(results).models)

    (points,) = fig.data
    assert list(points.x) == [4.9, 29.7]
    assert list(points.y) == [50.0, 83.0]
    assert list(points.text) == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert fig.layout.xaxis.type == "log"


def test_speed_quality_chart_skips_models_without_timings(results: Path) -> None:
    fig = benchmark_speed_quality_chart(_run(results, OLDER).models)

    assert list(fig.data[0].text) == ["GLM-4.7-Flash"]


def test_heatmap_rows_are_scenarios_in_meta_order_and_cells_show_passed_over_runs(results: Path) -> None:
    run = _run(results)
    (heatmap,) = benchmark_scenario_heatmap(run.models, run.scenarios).data

    assert list(heatmap.y) == SCENARIOS
    assert list(heatmap.x) == ["GLM-4.7-Flash", "Qwen3.6-27B-AWQ"]
    assert tuple(heatmap.text[0]) == ("0/2", "1/2")  # reasoning.rsi_signal
    assert tuple(heatmap.z[0]) == (80.0, 96.0)


def test_heatmap_shows_a_dash_for_a_scenario_a_model_has_no_score_for(results: Path) -> None:
    run = _run(results)
    glm = dataclasses.replace(run.models[0], scenarios={})
    (heatmap,) = benchmark_scenario_heatmap([glm], run.scenarios).data

    assert tuple(heatmap.text[0]) == ("—",)
    assert tuple(heatmap.z[0]) == (None,)


def test_summary_frame_is_sorted_by_overall_with_the_winner_marked(results: Path) -> None:
    run = _run(results)
    frame = benchmark_summary_frame(run.models, run.categories)

    assert list(frame["Model"]) == ["🏆 Qwen3.6-27B-AWQ", "GLM-4.7-Flash"]
    assert list(frame.columns[:4]) == ["Model", "Overall", "Reasoning", "Tools"]
    assert frame.loc[0, "Overall"] == pytest.approx(83.0)
    assert frame.loc[0, "Runs passed"] == "5/6"
    assert frame.loc[0, "Timeouts / errors"] == "1/0"


def test_summary_frame_puts_a_model_that_did_not_run_last_with_empty_scores(results: Path) -> None:
    run = _run(results, OLDER)
    frame = benchmark_summary_frame(run.models, run.categories)

    assert list(frame["Model"]) == ["🏆 GLM-4.7-Flash", "Gpt-OSS-20b (did not run)"]
    assert frame["Overall"].isna().iloc[1]


def test_scenario_runs_frame_has_one_row_per_repeat(results: Path) -> None:
    runs = load_scenario_runs(BenchmarkRunRef.from_path(results / LATEST), "qwen3627b", "reasoning.rsi_signal").runs
    frame = scenario_runs_frame(runs)

    assert list(frame["Repeat"]) == [1, 2]
    assert list(frame["Passed"]) == ["✅", "❌"]
    assert list(frame["Status"]) == ["ok", "timeout"]
    assert frame.loc[0, "Partial %"] == pytest.approx(100.0)

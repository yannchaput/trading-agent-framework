"""Models tab: compare the vLLM models of a benchmark-vllm-models run (read-only)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard import benchmark_reader as br
from trading_agent_framework.dashboard.components.charts import (
    benchmark_category_chart,
    benchmark_scenario_heatmap,
    benchmark_speed_quality_chart,
)
from trading_agent_framework.dashboard.components.metric_cards import render_metric_card
from trading_agent_framework.dashboard.components.tables import render_benchmark_summary_table, scenario_runs_frame
from trading_agent_framework.dashboard.models import BenchmarkModel, BenchmarkRun, BenchmarkRunRef, ScenarioRuns
from trading_agent_framework.utils.errors import ConfigurationError


@st.cache_data(show_spinner=False)
def _load_run(path: str) -> BenchmarkRun:
    return br.load_benchmark_run(BenchmarkRunRef.from_path(Path(path)))


@st.cache_data(show_spinner=False)
def _load_scenario_runs(path: str, model_key: str, scenario_id: str) -> ScenarioRuns:
    return br.load_scenario_runs(BenchmarkRunRef.from_path(Path(path)), model_key, scenario_id)


def page_models() -> None:
    """Benchmark run picker in the sidebar; scores, charts, heatmap and drill-down in the page."""
    if st.sidebar.button("🔄 Refresh Data", width="stretch", key="models_refresh"):
        _load_run.clear()
        _load_scenario_runs.clear()
        st.rerun()

    with st.container(key="models_page_marker"):
        st.title("Model benchmark")
    try:
        results_dir = br.resolve_results_dir(sys.argv[1:])
    except ConfigurationError as exc:
        st.error(f"Cannot locate the benchmark results: {exc}")
        return

    refs = br.scan_benchmark_runs(results_dir)
    if not refs:
        st.info(f"No complete benchmark run found in `{results_dir}`. Run the benchmark, or point the dashboard at its results with `uv run dashboard --benchmark-dir PATH`.")
        return

    loaded: dict[str, BenchmarkRun | br.BenchmarkReadError] = {}
    for ref in refs:
        try:
            loaded[ref.run_id] = _load_run(str(ref.path))
        except br.BenchmarkReadError as exc:
            loaded[ref.run_id] = exc

    run_id = st.sidebar.selectbox(
        "Benchmark run",
        [ref.run_id for ref in refs],
        format_func=lambda rid: _run_label(rid, loaded[rid]),
        key="benchmark_run",
    )
    run = loaded[run_id]
    if isinstance(run, br.BenchmarkReadError):
        st.error(f"This benchmark run cannot be read: {run}")
        return

    _render_header(run)
    ran = [model for model in run.models if model.ran]
    if not ran:
        st.warning("No model ran in this benchmark run.")
        render_benchmark_summary_table(run.models, run.categories)
        return

    _render_headline_cards(ran)
    st.subheader("Summary")
    render_benchmark_summary_table(run.models, run.categories)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(benchmark_category_chart(ran, run.categories), width="stretch")
    with right:
        st.plotly_chart(benchmark_speed_quality_chart(ran), width="stretch")
    st.subheader("Per-scenario results")
    st.plotly_chart(benchmark_scenario_heatmap(ran, run.scenarios), width="stretch")
    _render_drill_down(run, ran)


def _run_label(run_id: str, run: BenchmarkRun | br.BenchmarkReadError) -> str:
    stamp = BenchmarkRunRef.from_path(Path(run_id)).started_at.strftime("%Y-%m-%d %H:%M")
    if isinstance(run, br.BenchmarkReadError):
        return f"⚠ {stamp} · unreadable"
    return f"{stamp} · {len(run.models)} models · {run.repeats} repeats"


def _fmt_duration(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _render_header(run: BenchmarkRun) -> None:
    parts = [f"Started {run.started_at:%Y-%m-%d %H:%M}"]
    if run.finished_at is not None:
        parts.append(f"finished {run.finished_at:%Y-%m-%d %H:%M} ({_fmt_duration(run.finished_at - run.started_at)})")
    parts += [f"{run.repeats} repeats", f"timeout {run.timeout_s:.0f} s"]
    versions = sorted({model.vllm_version for model in run.models if model.vllm_version})
    if versions:
        parts.append("vLLM " + ", ".join(versions))
    st.caption(" · ".join(parts))
    for model in run.models:
        if not model.ran:
            st.warning(f"{model.display_name} did not run: {model.error or 'no error recorded'}")


def _best(models: Sequence[BenchmarkModel]) -> BenchmarkModel:
    return max(models, key=lambda m: m.overall or 0.0)


def _render_headline_cards(ran: Sequence[BenchmarkModel]) -> None:
    best = _best(ran)
    timed = [m for m in ran if m.median_tokens_per_s is not None]
    quick = [m for m in ran if m.median_run_s is not None]
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("🏆 Best overall", f"{best.display_name} · {(best.overall or 0.0):.0%}")
    with cols[1]:
        fastest = max(timed, key=lambda m: m.median_tokens_per_s) if timed else None
        render_metric_card("⚡ Fastest (tokens/s)", f"{fastest.display_name} · {fastest.median_tokens_per_s:.0f}" if fastest else "—")
    with cols[2]:
        quickest = min(quick, key=lambda m: m.median_run_s) if quick else None
        render_metric_card("⏱ Lowest median run", f"{quickest.display_name} · {quickest.median_run_s:.1f} s" if quickest else "—")


def _worst_scenario(model: BenchmarkModel, scenarios: Sequence[str]) -> str:
    """The model's lowest mean-partial scenario (first in meta order on ties)."""
    scored = [s for s in scenarios if s in model.scenarios]
    return min(scored, key=lambda s: model.scenarios[s].mean_partial) if scored else scenarios[0]


def _render_drill_down(run: BenchmarkRun, ran: Sequence[BenchmarkModel]) -> None:
    if not run.scenarios:
        return
    st.subheader("Drill-down")
    names = {model.key: model.display_name for model in ran}
    keys = list(names)
    best = _best(ran)
    left, right = st.columns(2)
    with left:
        # Keys are scoped to the run, so switching runs resets both pickers.
        model_key = st.selectbox("Model", keys, index=keys.index(best.key), format_func=names.__getitem__, key=f"drill_model_{run.ref.run_id}")
    model = next(m for m in ran if m.key == model_key)
    scenarios = list(run.scenarios)
    with right:
        scenario_id = st.selectbox(
            "Scenario",
            scenarios,
            index=scenarios.index(_worst_scenario(model, scenarios)),
            key=f"drill_scenario_{run.ref.run_id}",
        )

    try:
        result = _load_scenario_runs(str(run.ref.path), model_key, scenario_id)
    except br.BenchmarkReadError as exc:
        st.error(f"Cannot load the runs of {names[model_key]}: {exc}")
        return

    if not result.runs:
        st.info("No runs recorded for this model and scenario.")
    else:
        st.dataframe(scenario_runs_frame(result.runs), width="stretch", hide_index=True)
        for scenario_run in result.runs:
            icon = "✅" if scenario_run.passed else "❌"
            label = f"{icon} Repeat {scenario_run.repeat} · {scenario_run.status} · {scenario_run.partial:.0%}"
            with st.expander(label, expanded=not scenario_run.passed):
                if scenario_run.error:
                    st.error(scenario_run.error)
                if not scenario_run.checks:
                    st.caption("No checks recorded.")
                for check in scenario_run.checks:
                    st.markdown(f"{'✅' if check.passed else '❌'} **{check.type}**: `{check.reason}`")
    if result.skipped_lines:
        st.caption(f"{result.skipped_lines} malformed line(s) skipped in {model_key}.jsonl")

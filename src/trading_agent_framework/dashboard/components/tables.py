"""Styled table components."""

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from trading_agent_framework.dashboard.models import BenchmarkModel, ScenarioRun


def render_metric_table(metrics: dict[str, tuple[float, float]], title: str = "") -> None:
    """Render a styled two-column metric table (Strategy vs Benchmark)."""
    if title:
        st.subheader(title)

    rows = []
    for label, (s_val, b_val) in metrics.items():
        delta = s_val - b_val if isinstance(s_val, (int, float)) and isinstance(b_val, (int, float)) else ""
        rows.append({"Metric": label, "Strategy": s_val, "Benchmark": b_val, "Delta": delta})

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Strategy": st.column_config.NumberColumn(format="%.4f"),
            "Benchmark": st.column_config.NumberColumn(format="%.4f"),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
        },
    )


DECISION_OPTIONS = ["discarded", "study", "validated"]
DECISION_COLORS = ["darkred", "darkorange", "limegreen"]
REGIME_OPTIONS = ["Bearish", "Neutral", "Bullish", "All-Weather"]
REGIME_COLORS = ["darkred", "gray", "limegreen", "dodgerblue"]
# Editable colored-tag columns, rendered as MultiselectColumns (see render_scorecard_table):
# Decision holds at most one value, Regime any number.
PICKER_COLUMNS = ("Decision", "Regime")


def render_scorecard_table(runs_data: list[dict], key: str = "scorecard_table") -> tuple[list[int], pd.DataFrame]:
    """Render the scorecard comparison table with a "Select" checkbox column (for navigating
    to Run Detail / Side-by-side) and editable, colored Decision and Regime pickers.

    st.dataframe's native row-click selection and an editable per-cell picker can't coexist in
    one widget, so this uses st.data_editor throughout: "Select" replaces row-click selection,
    and "Decision"/"Regime" are editable ``MultiselectColumn`` (colored dropdown), used only for
    their per-option coloring -- ``st.column_config.SelectboxColumn`` has no color support.
    Regime keeps every tag selected; Decision allows only one value per cell, so the caller
    collapses extra Decision selections and re-renders under a fresh ``key`` (Streamlit
    disallows rewriting a data_editor's own session-state value directly).

    Returns a tuple of ``(selected_indices, edited_df)`` where ``selected_indices`` is the list
    of zero-based row indices with ``Select`` checked and ``edited_df`` is the edited DataFrame
    (without hidden columns like ``_ref``), reflecting any Decision change just made.
    """
    if not runs_data:
        st.info("No runs match the current filters.")
        return [], pd.DataFrame()

    df = pd.DataFrame(runs_data)
    df.insert(0, "Select", False)
    # MultiselectColumn cells are lists: a run's decision is a single value or unset (""), its
    # regime is already a list of any number of values.
    df["Decision"] = df["Decision"].apply(lambda v: [v] if v else [])
    df["Regime"] = df["Regime"].apply(list)

    # Remove hidden columns before passing to the Arrow serializer
    # (st.data_editor cannot serialize arbitrary Python objects like RunRef).
    display_cols = [c for c in df.columns if not c.startswith("_")]
    display_df = df[display_cols]
    disabled_cols = [c for c in display_cols if c not in ("Select", *PICKER_COLUMNS)]

    column_config = {
        "Select": st.column_config.CheckboxColumn("Select"),
        "Strategy": st.column_config.TextColumn("Strategy"),
        "Run Date": st.column_config.TextColumn("Run Date"),
        "Mode": st.column_config.TextColumn("Mode"),
        "Budget": st.column_config.NumberColumn("Budget", format="$%.0f"),
        "Period": st.column_config.TextColumn("Period"),
        "CAGR%": st.column_config.NumberColumn("CAGR%", format="%.2f%%"),
        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "Sortino": st.column_config.NumberColumn("Sortino", format="%.2f"),
        "Max DD%": st.column_config.NumberColumn("Max DD%", format="%.2f%%"),
        "Volatility%": st.column_config.NumberColumn("Volatility%", format="%.2f%%"),
        "Model": st.column_config.TextColumn("Model"),
        "Broker": st.column_config.TextColumn("Broker"),
        "Decision": st.column_config.MultiselectColumn(
            "Decision",
            options=DECISION_OPTIONS,
            color=DECISION_COLORS,
        ),
        "Regime": st.column_config.MultiselectColumn(
            "Regime",
            options=REGIME_OPTIONS,
            color=REGIME_COLORS,
        ),
        "Time": st.column_config.TextColumn("Time"),
        "Description": st.column_config.TextColumn("Description"),
    }

    edited_df = st.data_editor(
        display_df,
        width="stretch",
        height="stretch",
        hide_index=True,
        column_config=column_config,
        disabled=disabled_cols,
        num_rows="fixed",
        key=key,
    )

    selected_indices = edited_df.index[edited_df["Select"]].tolist()
    return selected_indices, edited_df


# --- vLLM benchmark (Models tab) ---------------------------------------------------------------


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100


def _ratio(a: int | None, b: int | None) -> str:
    return "—" if a is None or b is None else f"{a}/{b}"


def benchmark_summary_frame(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> pd.DataFrame:
    """One row per model, best overall first (marked 🏆); models that did not run go last."""
    ranked = sorted(models, key=lambda m: (not m.ran, -(m.overall or 0.0)))
    rows = []
    for i, model in enumerate(ranked):
        if not model.ran:
            name = f"{model.display_name} (did not run)"
        else:
            name = f"🏆 {model.display_name}" if i == 0 else model.display_name
        row: dict[str, object] = {"Model": name, "Overall": _pct(model.overall)}
        for category in categories:
            row[category.capitalize()] = _pct(model.categories.get(category))
        row |= {
            "Runs passed": _ratio(model.runs_passed, model.runs_total),
            "Text tool calls": model.text_tool_calls,
            "Avg tool calls": model.avg_tool_calls,
            "Median run (s)": model.median_run_s,
            "Tokens/s": model.median_tokens_per_s,
            "Timeouts / errors": _ratio(model.timeouts, model.errors),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_benchmark_summary_table(models: Sequence[BenchmarkModel], categories: Sequence[str]) -> None:
    frame = benchmark_summary_frame(models, categories)
    score_columns = ["Overall", *(category.capitalize() for category in categories)]
    column_config = {column: st.column_config.ProgressColumn(column, min_value=0, max_value=100, format="%.0f%%") for column in score_columns}
    column_config |= {
        "Avg tool calls": st.column_config.NumberColumn(format="%.2f"),
        "Median run (s)": st.column_config.NumberColumn(format="%.1f"),
        "Tokens/s": st.column_config.NumberColumn(format="%.1f"),
    }
    st.dataframe(frame, width="stretch", hide_index=True, column_config=column_config)


def scenario_runs_frame(runs: Sequence[ScenarioRun]) -> pd.DataFrame:
    """One row per repeat of a scenario for one model."""
    return pd.DataFrame(
        [
            {
                "Repeat": run.repeat,
                "Status": run.status,
                "Passed": "✅" if run.passed else "❌",
                "Partial %": run.partial * 100,
                "Total (s)": run.total_s,
                "Tool calls": run.tool_calls,
                "Model calls": run.model_calls,
                "Tokens/s": run.tokens_per_s,
            }
            for run in runs
        ]
    )

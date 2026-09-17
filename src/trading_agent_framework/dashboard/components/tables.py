"""Styled table components."""

import pandas as pd
import streamlit as st


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


def render_scorecard_table(runs_data: list[dict]) -> tuple[list[int], pd.DataFrame]:
    """Render the scorecard comparison table with native row-click selection.

    Returns a tuple of ``(selected_indices, display_df)`` where
    ``selected_indices`` is the list of zero-based row indices currently
    selected and ``display_df`` is the DataFrame shown to the user (without
    hidden columns like ``_ref``).
    """
    if not runs_data:
        st.info("No runs match the current filters.")
        return [], pd.DataFrame()

    df = pd.DataFrame(runs_data)

    # Remove hidden columns before passing to the Arrow serializer
    # (st.dataframe cannot serialize arbitrary Python objects like RunRef).
    display_cols = [c for c in df.columns if not c.startswith("_")]
    display_df = df[display_cols]

    column_config = {
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
        "Time": st.column_config.TextColumn("Time"),
        "Description": st.column_config.TextColumn("Description"),
    }

    selection = st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        key="scorecard_table",
        selection_mode="multi-row",
        on_select="rerun",
    )

    return selection.selection.rows, display_df

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


DECISION_OPTIONS = ["discarded", "study", "validated"]
DECISION_COLORS = ["darkred", "darkorange", "limegreen"]


def render_scorecard_table(runs_data: list[dict]) -> tuple[list[int], pd.DataFrame]:
    """Render the scorecard comparison table with a "Select" checkbox column (for navigating
    to Run Detail / Side-by-side) and an editable, colored Decision picker.

    st.dataframe's native row-click selection and an editable per-cell picker can't coexist in
    one widget, so this uses st.data_editor throughout: "Select" replaces row-click selection,
    and "Decision" is an editable ``MultiselectColumn`` (colored dropdown) constrained to at
    most one value per row -- Decision edits are resolved to a single string by the caller.

    Returns a tuple of ``(selected_indices, edited_df)`` where ``selected_indices`` is the list
    of zero-based row indices with ``Select`` checked and ``edited_df`` is the edited DataFrame
    (without hidden columns like ``_ref``), reflecting any Decision change just made.
    """
    if not runs_data:
        st.info("No runs match the current filters.")
        return [], pd.DataFrame()

    df = pd.DataFrame(runs_data)
    df.insert(0, "Select", False)
    # MultiselectColumn cells are lists; a run's decision is a single value or unset ("").
    df["Decision"] = df["Decision"].apply(lambda d: [d] if d else [])

    # Remove hidden columns before passing to the Arrow serializer
    # (st.data_editor cannot serialize arbitrary Python objects like RunRef).
    display_cols = [c for c in df.columns if not c.startswith("_")]
    display_df = df[display_cols]
    disabled_cols = [c for c in display_cols if c not in ("Select", "Decision")]

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
        "Decision": st.column_config.MultiselectColumn(
            "Decision",
            options=DECISION_OPTIONS,
            color=DECISION_COLORS,
        ),
        "Time": st.column_config.TextColumn("Time"),
        "Description": st.column_config.TextColumn("Description"),
    }

    edited_df = st.data_editor(
        display_df,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        disabled=disabled_cols,
        num_rows="fixed",
        key="scorecard_table",
    )

    selected_indices = edited_df.index[edited_df["Select"]].tolist()
    return selected_indices, edited_df

"""Metric KPI card components."""

from html import escape

import streamlit as st


def render_metric_card(
    label: str,
    value: float | str,
    fmt: str = ".2f",
    is_pct: bool = True,
    tone: str | None = None,
    help: str | None = None,
) -> None:
    """Render a single KPI metric card. Use inside st.columns().

    `tone` ("positive"/"negative") overrides the colour, which otherwise follows the sign of a numeric value.
    `help` adds an ⓘ next to the label whose hover tooltip shows that text.
    """
    if isinstance(value, (int, float)):
        if is_pct:
            display = f"{value * 100:{fmt}}%" if fmt else f"{value * 100:.1f}%"
        else:
            display = f"{value:{fmt}}" if fmt else f"{value:.2f}"
        css_class = "positive" if value >= 0 else "negative"
    else:
        display = str(value)
        css_class = ""

    if tone is not None:
        css_class = tone

    hint = f' <span title="{escape(help, quote=True)}" style="cursor: help;">ⓘ</span>' if help else ""

    st.markdown(
        f"""
    <div class="metric-card">
        <div class="label">{label}{hint}</div>
        <div class="value {css_class}">{display}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _universe_label(parameters: dict | None) -> str:
    universe = (parameters or {}).get("universe")
    if isinstance(universe, dict) and universe.get("mode") == "point_in_time":
        return "as it stood on each date (delisted stocks missing)"
    return "today's list"


def render_header_card(
    strategy_name: str,
    run_ts: str,
    mode: str,
    budget: float,
    period_start: str,
    period_end: str,
    model: str = "",
    data_source: str = "",
    backtest_time: float = 0.0,
    parameters: dict | None = None,
) -> None:
    """Render a run detail header card with metadata.

    `parameters` is the run's settings.json `parameters`; its `universe.mode == "point_in_time"` (cross_momentum and
    bull_bear backtests) selects the Universe line, anything else is today's list.
    """
    cols = st.columns([2, 2, 1, 1, 2])
    with cols[0]:
        st.markdown(f"**Strategy:** {strategy_name}")
        st.markdown(f"**Run:** {run_ts}")
    with cols[1]:
        st.markdown(f"**Period:** {period_start} → {period_end}")
        st.markdown(f"**Budget:** ${budget:,.0f}")
    with cols[2]:
        st.markdown(f"**Mode:** {mode}")
        st.markdown(f"**Data:** {data_source}")
        st.markdown(f"**Universe:** {_universe_label(parameters)}")
    with cols[3]:
        if model:
            st.markdown(f"**Model:** `{model}`")
        if backtest_time > 0:
            mins = int(backtest_time // 60)
            secs = int(backtest_time % 60)
            st.markdown(f"**Time:** {mins}m {secs}s")
    with cols[4]:
        st.markdown("**Status:** ✅ Complete")
    st.divider()

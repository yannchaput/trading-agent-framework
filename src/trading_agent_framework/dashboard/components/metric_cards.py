"""Metric KPI card components."""

import streamlit as st


def render_metric_card(label: str, value: float | str, fmt: str = ".2f", is_pct: bool = True) -> None:
    """Render a single KPI metric card. Use inside st.columns()."""
    if isinstance(value, (int, float)):
        if is_pct:
            display = f"{value * 100:{fmt}}%" if fmt else f"{value * 100:.1f}%"
        else:
            display = f"{value:{fmt}}" if fmt else f"{value:.2f}"
        css_class = "positive" if value >= 0 else "negative"
    else:
        display = str(value)
        css_class = ""

    st.markdown(
        f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{display}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


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
) -> None:
    """Render a run detail header card with metadata."""
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

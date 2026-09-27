"""Dark trading theme: CSS for the custom HTML components.

Streamlit's own colours come from cli.THEME_ARGS; charts use components.charts.CHART_TEMPLATE.
"""

import streamlit as st

THEME_CSS = """
<style>
.metric-card {
    background: #131722; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 16px; text-align: center;
}
.metric-card .label {
    font-size: 0.8rem; color: #8a8f98;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.metric-card .value {
    font-size: 1.4rem; font-weight: 600; color: #d1d4dc;
}
.metric-card .value.positive { color: #22c55e; }
.metric-card .value.negative { color: #ef4444; }
.strategy-badge {
    display: inline-block; background: rgba(34, 211, 238, 0.12);
    color: #22d3ee; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; font-weight: 500;
}
</style>
"""


def apply_theme():
    """Inject the dark theme CSS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

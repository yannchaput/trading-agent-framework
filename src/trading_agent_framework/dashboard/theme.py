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
[data-testid="stTopNavLink"][aria-current="page"] span {
    color: #22c55e !important;
    font-weight: 700 !important;
}
[data-testid="stTopNavLink"]:not([aria-current="page"]) span {
    color: #d1d4dc !important;
    font-weight: 400 !important;
}
/* Streamlit sizes a data_editor to its own content by default, even with height="stretch",
   because the page's own block container never flex-grows to the viewport. Force the
   scorecard table (only) to use the remaining browser height instead. */
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] {
    height: calc(100vh - 300px) !important;
    min-height: 320px;
}
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] [data-testid="stFullScreenFrame"],
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] [data-testid="stDataFrame"] {
    height: 100% !important;
}
</style>
"""


def apply_theme():
    """Inject the dark theme CSS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

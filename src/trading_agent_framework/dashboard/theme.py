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
/* st.logo only has three fixed heights; assets/logo.svg is drawn larger than "large", at its own 235 px width,
   and shrinks with a narrower sidebar. The sidebar header grows with it, and its collapse button floats at the
   right edge instead of squeezing the logo. */
[data-testid="stSidebarHeader"] {
    height: auto !important;
    min-height: 3.75rem;
    position: relative;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
img[data-testid="stSidebarLogo"] {
    height: auto !important;
    width: 100%;
    max-width: 235px;
}
[data-testid="stSidebarCollapseButton"] {
    position: absolute;
    top: 1.125rem;
    right: -0.5rem;
}
/* The icon shown in the header while the sidebar is collapsed (assets/logo-icon.svg), as tall as the sidebar's
   logo (assets/logo.svg is 48 px high), so the mascot keeps its size when the sidebar is collapsed. */
img[data-testid="stHeaderLogo"] {
    height: 3rem;
}
[data-testid="stTopNavLink"][aria-current="page"] span {
    color: #22c55e !important;
    font-weight: 700 !important;
}
[data-testid="stTopNavLink"]:not([aria-current="page"]) span {
    color: #d1d4dc !important;
    font-weight: 400 !important;
}
/* Streamlit's own block container never flex-grows to the viewport, so a data_editor's
   height="stretch" is a no-op at the page's top level -- it just sizes to its own content.
   Scoped to the scorecard page only (:has() guards every other page's layout): make the
   block chain down to the scorecard table a real flex column anchored to the tab's full
   height, then let the table (the only flexed child) absorb whatever space is left below
   the title and caption. */
div[data-testid="stMainBlockContainer"]:has(div[class*="st-key-scorecard_table"]) {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding-top: 48px !important;
    padding-bottom: 1.5rem !important;
}
div[data-testid="stMainBlockContainer"]:has(div[class*="st-key-scorecard_table"])
        > div[data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    min-height: 0;
}
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] {
    flex: 1 1 auto;
    min-height: 0;
}
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] [data-testid="stFullScreenFrame"],
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] [data-testid="stDataFrame"] {
    height: 100% !important;
}
/* The grid widget itself (glide-data-grid) bakes a content-fit pixel height as an inline
   style on this wrapper (e.g. "height: 400px; max-height: 527px"), independent of its DOM
   ancestors -- overriding just the ancestors above leaves this element, and therefore the
   visible grid/scrollbars, stuck at the old size. Force it to fill the space we just made. */
div[data-testid="stElementContainer"][class*="st-key-scorecard_table"] [data-testid="stDataFrameResizable"] {
    height: 100% !important;
    max-height: none !important;
}
/* Same oversized default top padding (128px) on the Models page; unlike the scorecard, this
   page just scrolls (many stacked charts), so only the padding needs trimming.
   models_page_marker (models.py, wrapping just the page title) exists purely as a CSS hook:
   a childless st.container() never reaches the DOM, and a plain st.dataframe's key (unlike
   st.data_editor's) isn't surfaced as an "st-key-" class either. */
div[data-testid="stMainBlockContainer"]:has(div[class*="st-key-models_page_marker"]) {
    padding-top: 48px !important;
}
/* The run description (scorecard.py, text_area keyed "run_description"): light box with
   dark text. The disabled textarea dims its text, so colour and opacity are forced too. */
div[class*="st-key-run_description"] div[data-baseweb="textarea"],
div[class*="st-key-run_description"] textarea {
    background-color: #131722 !important;
    color: #d1d4dc !important;
    -webkit-text-fill-color: #d1d4dc !important;
    opacity: 1 !important;
}
</style>
"""


def apply_theme():
    """Inject the dark theme CSS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)

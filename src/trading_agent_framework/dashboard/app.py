"""Streamlit dashboard — strategy comparison.

Run via: uv run dashboard
Or:      uv run streamlit run src/trading_agent_framework/dashboard/app.py
"""

import streamlit as st

from trading_agent_framework.dashboard.reader import load_description, save_description
from trading_agent_framework.dashboard.theme import apply_theme

# Must be first Streamlit call
st.set_page_config(
    page_title="Strategy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.dialog("Edit Description", width="large")
def edit_description_dialog():
    """Dialog for editing a run's description."""
    ref = st.session_state.get("description_editor_ref")
    if ref is None:
        st.error("No run selected.")
        if st.button("Close"):
            st.session_state.show_description_editor = False
            st.rerun()
        return

    st.caption(f"{ref.strategy_name} — {ref.run_ts}")
    current_desc = load_description(ref) or ""
    new_desc = st.text_area(
        "Description",
        value=current_desc,
        height=200,
        placeholder="Describe the purpose and parameters of this run...",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_description_editor = False
            st.rerun()
    with col2:
        if st.button("Save", use_container_width=True, type="primary"):
            save_description(ref, new_desc)
            st.toast("Description saved.")
            st.session_state.show_description_editor = False
            st.rerun()


def main():
    """Main dashboard with sidebar navigation."""
    st.sidebar.title("📊 Strategy Dashboard")
    st.sidebar.markdown("Compare backtesting runs across strategies.")
    st.sidebar.markdown("---")

    pages = {
        "Scorecard": "page_scorecard",
        "Run Detail": "page_detail",
        "Side-by-Side": "page_side_by_side",
    }
    icons = {"Scorecard": "📋", "Run Detail": "📈", "Side-by-Side": "⚖️"}

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Scorecard"

    for page_name in pages:
        icon = icons.get(page_name, "")
        if st.sidebar.button(f"{icon} {page_name}", use_container_width=True, key=f"nav_{page_name}"):
            # Validate checkbox selection before navigating to pages that
            # require selected runs.
            selected = st.session_state.get("selected_refs", [])
            if page_name == "Run Detail":
                if len(selected) == 0:
                    st.warning("Select a run to view its details.")
                elif len(selected) >= 2:
                    st.warning("Select exactly one run to view details. Use Side-by-Side for comparing multiple runs.")
                else:
                    st.session_state.detail_ref = selected[0]
                    st.session_state.current_page = page_name
            elif page_name == "Side-by-Side":
                if len(selected) < 2:
                    st.warning("Select at least two runs for side-by-side comparison.")
                else:
                    st.session_state.compare_refs = selected
                    st.session_state.current_page = page_name
            else:
                st.session_state.current_page = page_name

    # Edit Description button — only enabled for a single backtesting run
    selected = st.session_state.get("selected_refs", [])
    if st.sidebar.button(
        "✏️ Edit Description",
        use_container_width=True,
        key="nav_edit_description",
    ):
        if len(selected) != 1:
            st.warning("Select exactly one run to edit its description.")
        elif selected[0].mode != "backtesting":
            st.warning("Description editing is only available for backtesting runs.")
        else:
            st.session_state.description_editor_ref = selected[0]
            st.session_state.show_description_editor = True
            edit_description_dialog()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        if "run_index" in st.session_state:
            del st.session_state.run_index
        st.rerun()

    # Route to the selected page
    page_name = st.session_state.current_page
    if page_name == "Scorecard":
        from trading_agent_framework.dashboard._pages.scorecard import page_scorecard

        page_scorecard()
    elif page_name == "Run Detail":
        from trading_agent_framework.dashboard._pages.detail import page_detail

        page_detail()
    elif page_name == "Side-by-Side":
        from trading_agent_framework.dashboard._pages.side_by_side import page_side_by_side

        page_side_by_side()


if __name__ == "__main__":
    main()

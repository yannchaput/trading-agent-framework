"""Streamlit dashboard: Backtesting and Models tabs.

Run via: uv run dashboard [--benchmark-dir PATH]
Or:      uv run streamlit run src/trading_agent_framework/dashboard/app.py [-- --benchmark-dir PATH]
"""

from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard._pages.backtesting import page_backtesting
from trading_agent_framework.dashboard._pages.models import page_models
from trading_agent_framework.dashboard.theme import apply_theme

ASSETS = Path(__file__).parent / "assets"

# (title, url path, page function); the first one is the default tab. Plain data, so tests can
# check the tabs without a Streamlit runtime (st.Page cannot be built outside one).
NAV_PAGES = (
    ("Backtesting", "backtesting", page_backtesting),
    ("Models", "models", page_models),
)


def main():
    st.set_page_config(  # must be the first Streamlit call
        page_title="Yann's Trading Bots",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo-icon.svg"), size="large")
    pages = [st.Page(page, title=title, url_path=url_path, default=(i == 0)) for i, (title, url_path, page) in enumerate(NAV_PAGES)]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()

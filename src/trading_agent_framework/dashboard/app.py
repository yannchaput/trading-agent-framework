"""Streamlit dashboard: Live, Paper, Backtesting and Models tabs.

Run via: uv run dashboard [--benchmark-dir PATH]
Or:      uv run streamlit run src/trading_agent_framework/dashboard/app.py [-- --benchmark-dir PATH]
"""

from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard._pages.account import page_live, page_paper
from trading_agent_framework.dashboard._pages.backtesting import page_backtesting
from trading_agent_framework.dashboard._pages.models import page_models
from trading_agent_framework.dashboard.theme import apply_theme

ASSETS = Path(__file__).parent / "assets"

# (title, url path, page function), in tab order; DEFAULT_PAGE opens first. Plain data, so tests can
# check the tabs without a Streamlit runtime (st.Page cannot be built outside one).
NAV_PAGES = (
    ("Live", "live", page_live),
    ("Paper", "paper", page_paper),
    ("Backtesting", "backtesting", page_backtesting),
    ("Models", "models", page_models),
)
DEFAULT_PAGE = "Backtesting"


def main():
    st.set_page_config(  # must be the first Streamlit call
        page_title="Yann's Trading Bots",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo-icon.svg"), size="large")
    pages = [st.Page(page, title=title, url_path=url_path, default=(title == DEFAULT_PAGE)) for title, url_path, page in NAV_PAGES]
    st.navigation(pages, position="top").run()


if __name__ == "__main__":
    main()

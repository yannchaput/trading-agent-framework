"""White theme configuration with cyan accents."""

import streamlit as st


def apply_theme():
    """Apply white theme with cyan accent colors via custom CSS."""
    st.markdown(
        """
    <style>
    .stApp { background-color: #ffffff; }
    .metric-card {
        background: #ffffff; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 16px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card .label {
        font-size: 0.8rem; color: #888888;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.4rem; font-weight: 600; color: #1a1a1a;
    }
    .metric-card .value.positive { color: #27ae60; }
    .metric-card .value.negative { color: #e74c3c; }
    .strategy-badge {
        display: inline-block; background: #e8f4f8;
        color: #0891b2; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; font-weight: 500;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

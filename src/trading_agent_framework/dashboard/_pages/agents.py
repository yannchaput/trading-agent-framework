"""Agents tab: the `reviews.jsonl` of a backtesting run as a table, one row per strategy review."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from trading_agent_framework.dashboard import reviews_reader as rr
from trading_agent_framework.dashboard.models import RunRef

LOGS_DIR = Path("logs")  # relative to the working directory, like the Backtesting tab's scan


def page_agents() -> None:
    """Run picker in the sidebar; the review table and one review's raw record in the page."""
    st.title("Agents")
    refs = rr.scan_review_runs(LOGS_DIR)
    if not refs:
        st.info(f"No `reviews.jsonl` found under `{LOGS_DIR}/<strategy>/backtesting/`. Run an agent strategy such as `bill_ackman` or `bull_bear` in backtesting mode to produce one.")
        return

    ref = st.sidebar.selectbox("Run", refs, format_func=_run_label, key="agents_run")
    try:
        loaded = rr.load_reviews(ref)
    except OSError as exc:
        st.error(f"This run's reviews cannot be read: {exc}")
        return

    abandoned = sum(1 for record in loaded.records if record.get("abandoned"))
    parts = [f"{len(loaded.records)} reviews", f"{abandoned} abandoned"]
    if loaded.malformed:
        parts.append(f"{loaded.malformed} unreadable line{'s' if loaded.malformed != 1 else ''}")
    st.caption(" · ".join(parts))

    st.dataframe(rr.reviews_frame(loaded.records), width="stretch", hide_index=True)

    if loaded.records:
        dates = [str(record.get("date", "")) for record in loaded.records]
        date = st.selectbox("Review", dates, index=len(dates) - 1, key=f"agents_review_{ref.path}")
        st.json(loaded.records[dates.index(date)], expanded=True)


def _run_label(ref: RunRef) -> str:
    return f"{ref.strategy_name} · {ref.run_ts}"

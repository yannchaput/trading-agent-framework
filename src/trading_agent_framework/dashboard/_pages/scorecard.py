"""Page 1: Comparison Scorecard — filterable table of all runs."""

import streamlit as st

from lumibot_trading_agent.dashboard.components.tables import render_scorecard_table
from lumibot_trading_agent.dashboard.discovery import scan_runs
from lumibot_trading_agent.dashboard.reader import load_description, load_metrics, load_settings


def page_scorecard():
    """Scorecard page: filterable table of all backtesting runs."""
    st.title("Strategy Comparison")
    st.markdown("Compare backtesting runs across all strategies.")

    if "run_index" not in st.session_state:
        with st.spinner("Discovering runs..."):
            st.session_state.run_index = scan_runs("logs")
    index = st.session_state.run_index

    if not index.runs:
        st.warning("No backtesting runs found in logs/. Run a backtest first.")
        return

    all_strategies = index.strategy_names()
    selected_strategies = st.sidebar.multiselect("Strategies", all_strategies, default=all_strategies[: min(3, len(all_strategies))])
    if not selected_strategies:
        st.info("Select at least one strategy.")
        return

    rows = []
    for ref in index.runs:
        if ref.strategy_name not in selected_strategies:
            continue
        settings = load_settings(ref)
        metrics = load_metrics(ref)

        period = ""
        if settings and settings.backtesting_start and settings.backtesting_end:
            period = f"{settings.backtesting_start.strftime('%Y-%m-%d')} → {settings.backtesting_end.strftime('%Y-%m-%d')}"

        model = ""
        if settings and settings.parameters:
            for key, val in settings.parameters.items():
                if "model" in key.lower() and isinstance(val, str) and "/" in val:
                    model = val.split("/")[-1]
                    break

        backtest_time = ""
        if settings and settings.backtest_time_seconds:
            m = int(settings.backtest_time_seconds // 60)
            s = int(settings.backtest_time_seconds % 60)
            backtest_time = f"{m}m{s}s"

        rows.append(
            {
                "Strategy": ref.strategy_name,
                "Run Date": ref.run_ts.replace("_", " "),
                "Mode": ref.mode,
                "Budget": settings.budget if settings else 0,
                "Period": period,
                "CAGR%": (metrics.cagr_strategy * 100) if metrics else 0,
                "Sharpe": metrics.sharpe_strategy if metrics else 0,
                "Sortino": metrics.sortino_strategy if metrics else 0,
                "Max DD%": (metrics.max_drawdown_strategy * 100) if metrics else 0,
                "Volatility%": (metrics.volatility_strategy * 100) if metrics else 0,
                "Model": model,
                "Time": backtest_time,
                "Description": load_description(ref) or "",
                "_ref": ref,
            }
        )

    st.sidebar.metric("Total Runs", len(rows))

    if rows:
        selected_indices, _display_df = render_scorecard_table(rows)

        # Persist selection in session state so sidebar navigation buttons
        # can validate it before switching to Run Detail / Side-by-Side.
        if selected_indices:
            st.session_state.selected_refs = [rows[i]["_ref"] for i in selected_indices]
        else:
            st.session_state.selected_refs = []

        # Show description below the table when exactly one row is selected.
        if len(selected_indices) == 1:
            idx = selected_indices[0]
            desc = rows[idx].get("Description", "")
            if desc:
                st.text_area(
                    "Description",
                    value=desc,
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                )

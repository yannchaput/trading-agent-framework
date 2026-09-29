"""Page 1: Comparison Scorecard — filterable table of all runs."""

import streamlit as st

from trading_agent_framework.dashboard.components.tables import render_scorecard_table
from trading_agent_framework.dashboard.discovery import scan_runs
from trading_agent_framework.dashboard.reader import load_description, load_metrics, load_settings, save_decision


def _resolve_conflicting_decisions(rows: list[dict], table_key: str) -> bool:
    """Persist any row whose Decision cell in the previous widget instance ended up holding more
    than one tag, and report whether a correction was made.

    Streamlit's ``MultiselectColumn`` lets a cell hold more than one tag; only one decision may
    apply per row, so a two-tag edit is resolved here to the option most recently added. Once a
    correction is persisted, ``rows[i]["Decision"]`` is updated so the caller's next render starts
    from the resolved value. Streamlit refuses to let a caller rewrite a data_editor's own
    session-state value directly (``StreamlitValueAssignmentNotAllowedError``, even before the
    widget renders), so a `True` return tells the caller to render the table under a *new* widget
    key -- the only way to start it from the corrected data instead of the stale multi-tag edit.
    """
    state = st.session_state.get(table_key)
    if not state:
        return False

    changed = False
    for key, edits in state.get("edited_rows", {}).items():
        new_list = edits.get("Decision")
        if new_list is None or len(new_list) <= 1:
            continue

        i = int(key)
        old_decision = rows[i]["Decision"]
        old_list = [old_decision] if old_decision else []
        added = [d for d in new_list if d not in old_list]
        new_decision = added[-1] if added else new_list[-1]

        rows[i]["Decision"] = new_decision
        save_decision(rows[i]["_ref"], new_decision)
        changed = True

    return changed


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
        if settings and settings.agents:
            models = [agent["model"] for agent in settings.agents.values() if agent.get("model")]
            model = ", ".join(dict.fromkeys(models))

        broker = (settings.fees.get("broker") or "") if settings else ""

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
                "Broker": broker,
                "Decision": settings.dashboard_decision if settings else "",
                "Time": backtest_time,
                "Description": load_description(ref) or "",
                "_ref": ref,
            }
        )

    st.sidebar.metric("Total Runs", len(rows))

    if rows:
        if "scorecard_table_key" not in st.session_state:
            st.session_state.scorecard_table_key = 0
        table_key = f"scorecard_table_{st.session_state.scorecard_table_key}"

        if _resolve_conflicting_decisions(rows, table_key):
            st.session_state.scorecard_table_key += 1
            table_key = f"scorecard_table_{st.session_state.scorecard_table_key}"

        selected_indices, edited_df = render_scorecard_table(rows, key=table_key)

        # Persist Decision edits: MultiselectColumn cells are lists holding at most one value by
        # the time we get here (_resolve_conflicting_decisions already resolved any multi-tag
        # edit above), so this only needs to catch a genuine single-value change.
        for i, row in enumerate(rows):
            new_list = edited_df.iloc[i]["Decision"]
            new_decision = new_list[0] if new_list else ""
            if new_decision != row["Decision"]:
                save_decision(row["_ref"], new_decision)

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
                    height=300,
                    key="run_description",
                    disabled=True,
                    label_visibility="collapsed",
                )

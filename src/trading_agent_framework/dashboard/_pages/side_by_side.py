"""Page 3: Side-by-Side Comparison."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lumibot_trading_agent.dashboard.reader import load_portfolio_breakdown, load_run


def page_side_by_side():
    """Side-by-side comparison of selected runs."""
    st.title("Side-by-Side Comparison")

    if "compare_refs" not in st.session_state or not st.session_state.compare_refs:
        st.warning("No runs selected for comparison.")
        if st.button("← Back to Scorecard"):
            st.session_state.current_page = "Scorecard"
            st.rerun()
        return

    refs = st.session_state.compare_refs
    if st.button("← Back to Scorecard"):
        st.session_state.current_page = "Scorecard"
        st.rerun()

    runs = []
    for ref in refs:
        with st.spinner(f"Loading {ref.strategy_name}..."):
            run = load_run(ref)
            if run and run.metrics:
                runs.append(run)

    if not runs:
        st.error("Failed to load any of the selected runs.")
        return

    # Unified equity curve — uses stats.parquet (same source as Detail page)
    st.subheader("Equity Curves Overlay")
    colors = ["#0891b2", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad", "#2c3e50"]
    fig = go.Figure()
    for i, run in enumerate(runs):
        breakdown = load_portfolio_breakdown(run.ref)
        if breakdown and breakdown.get("dates") and breakdown.get("portfolio_value"):
            dates = pd.to_datetime(breakdown["dates"])
            values = breakdown["portfolio_value"]
        elif run.equity_curve:
            df = pd.DataFrame(run.equity_curve)
            date_col = "date" if "date" in df.columns else "datetime"
            val_col = "value" if "value" in df.columns else "close"
            dates = df[date_col]
            values = df[val_col]
        else:
            continue

        label = f"{run.ref.strategy_name} ({run.ref.run_ts[:10]})"
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name=label,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig.update_layout(
        title="Equity Curves Overlay",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    # Metric comparison
    st.subheader("Key Metrics Comparison")
    metrics_spec = [
        ("CAGR%", "cagr_strategy", True),
        ("Sharpe", "sharpe_strategy", False),
        ("Sortino", "sortino_strategy", False),
        ("Max DD%", "max_drawdown_strategy", True),
        ("Volatility%", "volatility_strategy", True),
        ("Win Days%", "win_days_pct_strategy", True),
        ("Calmar", "calmar_strategy", False),
        ("Omega", "omega_strategy", False),
    ]

    cols = st.columns(len(runs) + 1)
    with cols[0]:
        st.markdown("**Metric**")
        st.caption("Run date")
        for label, _, _ in metrics_spec:
            st.markdown(f"*{label}*")

    for i, run in enumerate(runs):
        with cols[i + 1]:
            st.markdown(f"**{run.ref.strategy_name}**")
            st.caption(run.ref.run_ts[:10])
            for label, attr, is_pct in metrics_spec:
                val = getattr(run.metrics, attr, 0)
                if is_pct:
                    st.markdown(f"{val * 100:.2f}%")
                else:
                    st.markdown(f"{val:.2f}")

    # Winner summary
    st.subheader("Winner Summary")
    summaries = []
    for label, attr, _ in metrics_spec:
        best = max(runs, key=lambda r, a=attr: getattr(r.metrics, a, -999))
        best_val = getattr(best.metrics, attr, 0)
        summaries.append(f"- **{label}**: {best.ref.strategy_name} - {best.ref.run_ts[:10]} ({best_val:.3f})")
    st.markdown("\n".join(summaries))

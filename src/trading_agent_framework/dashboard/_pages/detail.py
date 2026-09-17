"""Page 2: Run Detail — full metrics and charts for a single run."""

import pandas as pd
import streamlit as st

from trading_agent_framework.dashboard.components.charts import (
    cumulative_returns_chart,
    drawdown_chart,
    equity_curve_chart,
    monthly_returns_distribution,
    monthly_returns_heatmap,
    returns_distribution,
    rolling_sharpe_chart,
    rolling_sortino_chart,
    rolling_volatility_chart,
    trades_chart,
)
from trading_agent_framework.dashboard.components.metric_cards import render_header_card, render_metric_card
from trading_agent_framework.dashboard.reader import load_cumulative_returns, load_parameters, load_portfolio_breakdown, load_run, load_trades_curve, load_yearly_returns


def page_detail():
    """Run detail page: all metrics and charts for one run."""
    st.title("Run Detail")

    if "detail_ref" not in st.session_state:
        st.warning("No run selected. Go back to Scorecard.")
        if st.button("← Back to Scorecard"):
            st.session_state.current_page = "Scorecard"
            st.rerun()
        return

    ref = st.session_state.detail_ref
    if st.button("← Back to Scorecard"):
        st.session_state.current_page = "Scorecard"
        st.rerun()

    with st.spinner(f"Loading {ref.strategy_name} / {ref.run_ts}..."):
        run = load_run(ref)

    if run is None or run.metrics is None:
        st.error("Failed to load run data.")
        return

    m = run.metrics
    s = run.settings

    # Build a stats-based equity list for charts that derive from portfolio_value
    # (drawdown, heatmap, daily returns).  Falls back to indicators only when
    # stats.parquet is absent.
    breakdown = load_portfolio_breakdown(ref)
    bpv: list[dict] = []
    if breakdown:
        bpv = [{"date": d, "value": v} for d, v in zip(breakdown["dates"], breakdown["portfolio_value"])]

    # Load cumulative returns once — used by rolling-metric charts (Charts tab)
    # and the Returns tab.
    cum_ret = load_cumulative_returns(ref)

    period_start = s.backtesting_start.strftime("%Y-%m-%d") if s and s.backtesting_start else "?"
    period_end = s.backtesting_end.strftime("%Y-%m-%d") if s and s.backtesting_end else "?"
    model = ""
    if s and s.parameters:
        for key, val in s.parameters.items():
            if "model" in key.lower() and isinstance(val, str):
                model = val
                break

    render_header_card(
        strategy_name=ref.strategy_name,
        run_ts=ref.run_ts,
        mode=ref.mode,
        budget=s.budget if s else 0,
        period_start=period_start,
        period_end=period_end,
        model=model,
        data_source=s.backtesting_data_sources if s else "",
        backtest_time=s.backtest_time_seconds if s else 0,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance Metrics", "Charts", "Trades", "Returns", "Parameters"])

    with tab1:
        st.subheader("Returns")
        cols = st.columns(2)
        with cols[0]:
            render_metric_card("Total Return", m.total_return_strategy)
        with cols[1]:
            render_metric_card("CAGR", m.cagr_strategy)

        st.subheader("Risk-Adjusted")
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Sharpe", m.sharpe_strategy, is_pct=False)
        with cols[1]:
            render_metric_card("Sortino", m.sortino_strategy, is_pct=False)
        with cols[2]:
            render_metric_card("Calmar", m.calmar_strategy, is_pct=False)
        with cols[3]:
            render_metric_card("Omega", m.omega_strategy, is_pct=False)

        st.subheader("Risk")
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Max DD", m.max_drawdown_strategy)
        with cols[1]:
            render_metric_card("Volatility", m.volatility_strategy)
        with cols[2]:
            render_metric_card("Avg DD", m.avg_drawdown_strategy)
        with cols[3]:
            render_metric_card("Longest DD Days", m.longest_dd_days_strategy, fmt=".0f", is_pct=False)

        st.subheader("Win / Loss")
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Win Days %", m.win_days_pct_strategy)
        with cols[1]:
            render_metric_card("Win Month %", m.win_month_pct_strategy)
        with cols[2]:
            render_metric_card("Skew", m.skew_strategy, fmt=".2f", is_pct=False)
        with cols[3]:
            render_metric_card("Kurtosis", m.kurtosis_strategy, fmt=".2f", is_pct=False)

        st.subheader("Correlation")
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Beta", m.beta, fmt=".2f", is_pct=False)
        with cols[1]:
            render_metric_card("Alpha", m.alpha, fmt=".4f", is_pct=False)
        with cols[2]:
            render_metric_card("Correlation", m.correlation, fmt=".4f", is_pct=False)
        with cols[3]:
            render_metric_card("Treynor", m.treynor_ratio, fmt=".2f", is_pct=False)

        with st.expander("All Metrics (raw JSON)"):
            st.json(m.raw)

    with tab2:
        st.subheader("Equity Curve")
        st.plotly_chart(equity_curve_chart(run.equity_curve, breakdown=breakdown), width="stretch")
        st.subheader("Drawdown")
        st.plotly_chart(drawdown_chart(bpv or run.equity_curve), width="stretch")

        # Rolling metric charts — use daily returns from load_cumulative_returns
        if cum_ret and cum_ret.get("strategy_daily_returns"):
            sd = cum_ret["strategy_daily_returns"]
            bd = cum_ret.get("benchmark_daily_returns") or []
            dates = cum_ret["dates"]
            bm_label = cum_ret.get("benchmark_symbol", "SPY")

            st.subheader("Rolling Volatility (6M)")
            st.plotly_chart(rolling_volatility_chart(dates, sd, bd if bd else None, benchmark_label=bm_label), width="stretch")

            st.subheader("Rolling Sharpe (6M)")
            st.plotly_chart(rolling_sharpe_chart(dates, sd, bd if bd else None, benchmark_label=bm_label), width="stretch")

            st.subheader("Rolling Sortino (6M)")
            st.plotly_chart(rolling_sortino_chart(dates, sd, bd if bd else None, benchmark_label=bm_label), width="stretch")

            if cum_ret.get("benchmark_source"):
                st.caption(f"Benchmark data ({bm_label}): {cum_ret['benchmark_source']}")

    with tab3:
        st.subheader("Trade Activity")
        budget = s.budget if s else 10000.0
        trades_data = load_trades_curve(ref, budget)
        if trades_data and trades_data.get("trades"):
            st.plotly_chart(trades_chart(trades_data), width="stretch")
            n_buys = sum(1 for t in trades_data["trades"] if t["side"] == "buy")
            n_sells = sum(1 for t in trades_data["trades"] if t["side"] == "sell")
            st.caption(f"{len(trades_data['trades'])} trades — {n_buys} buys, {n_sells} sells")
        else:
            st.info("No trade data available for this run.")

    with tab4:
        st.subheader("Cumulative Returns vs Benchmark")
        if cum_ret:
            st.plotly_chart(cumulative_returns_chart(cum_ret), width="stretch")
            bs = cum_ret.get("benchmark_source", "")
            bm_label = cum_ret.get("benchmark_symbol", "SPY")
            if bs:
                st.caption(f"Benchmark data ({bm_label}): {bs}")
        else:
            st.info("No cumulative return data available (missing equity.parquet).")

        st.subheader("Monthly Returns Heatmap")
        st.caption(
            "Monthly returns are computed from `portfolio_value at end of month M / portfolio_value at end of month M-1 − 1`.\n\n"
            "Cash flow are accounted for, so monthly returns are not simply the sum of daily returns."
        )
        st.plotly_chart(monthly_returns_heatmap(bpv or run.equity_curve), width="stretch")

        st.subheader("Distribution of Monthly Returns")
        st.caption(
            "Monthly returns are computed from `portfolio_value at end of month M / portfolio_value at end of month M-1 − 1`, cash flow adjusted.\n\n"
            "The monthly returns are displayed as a histogram per buckets and density."
        )
        # Use pre-computed monthly returns from load_cumulative_returns —
        # these compound daily returns per month, matching quantstats exactly.
        strategy_monthly: list[float] = []
        benchmark_monthly: list[float] | None = None
        if cum_ret:
            strategy_monthly = cum_ret.get("strategy_monthly") or []
            benchmark_monthly = cum_ret.get("benchmark_monthly")
        bm_label = cum_ret.get("benchmark_symbol", "SPY") if cum_ret else "SPY"

        st.plotly_chart(
            monthly_returns_distribution(strategy_monthly, benchmark_monthly=benchmark_monthly, benchmark_label=bm_label),
            width="stretch",
        )

        st.subheader("Daily Returns")
        if cum_ret and cum_ret.get("strategy_daily_returns"):
            daily_returns_pct = [v * 100 for v in cum_ret["strategy_daily_returns"]]
            daily_dates = pd.to_datetime(cum_ret["dates"])
            st.plotly_chart(returns_distribution([], daily_returns=daily_returns_pct, daily_dates=daily_dates), width="stretch")
        else:
            st.plotly_chart(returns_distribution(bpv or run.equity_curve), width="stretch")

        st.subheader("Yearly Returns vs Benchmark")
        yearly = load_yearly_returns(ref)
        if yearly:
            yearly_df = pd.DataFrame(yearly)
            yearly_df["strategy"] = yearly_df["strategy"].apply(lambda x: f"{x * 100:.2f}%")
            yearly_df["benchmark"] = yearly_df["benchmark"].apply(lambda x: f"{x * 100:.2f}%" if x is not None else "—")
            yearly_df["won"] = yearly_df["won"].apply(lambda x: "✅" if x else "❌")
            st.dataframe(
                yearly_df.rename(
                    columns={
                        "year": "Year",
                        "strategy": "Strategy Return",
                        "benchmark": "Benchmark Return",
                        "won": "Beat Benchmark",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No yearly return data available.")

    with tab5:
        st.subheader("Parameters Used")
        params = load_parameters(ref)
        if params:
            from itertools import groupby

            for section, group in groupby(params, key=lambda r: r[0]):
                group_rows = [(p, v) for _, p, v in group]
                group_df = pd.DataFrame(group_rows, columns=["Parameter", "Value"])
                st.caption(f"**{section}**")
                st.dataframe(
                    group_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                        "Value": st.column_config.TextColumn("Value", width="large"),
                    },
                )
        else:
            st.info("No parameters available.")

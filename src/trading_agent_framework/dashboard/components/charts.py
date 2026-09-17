"""Plotly chart builders for the dashboard."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def equity_curve_chart(
    equity: list[dict[str, Any]],
    breakdown: dict[str, Any] | None = None,
    title: str = "Equity Curve",
) -> go.Figure:
    """Build a cumulative equity curve chart.

    When *breakdown* is provided (from load_portfolio_breakdown), all data —
    portfolio value, cash, and assets — is sourced from stats.parquet for
    consistency.  Falls back to the indicator-based *equity* list otherwise.
    """
    has_breakdown = bool(breakdown and breakdown.get("dates") and breakdown.get("portfolio_value"))

    if not has_breakdown and not equity:
        fig = go.Figure()
        fig.add_annotation(text="No equity data available", showarrow=False)
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = go.Figure()

    if has_breakdown:
        bd_dates = pd.to_datetime(breakdown["dates"])

        # Cash line (red)
        if breakdown.get("cash"):
            fig.add_trace(
                go.Scatter(
                    x=bd_dates,
                    y=breakdown["cash"],
                    mode="lines",
                    name="Cash",
                    line=dict(color="#ef4444", width=1.5),
                    hovertemplate="$%{y:,.0f}<extra>Cash</extra>",
                )
            )

        # Assets line (green)
        if breakdown.get("assets"):
            fig.add_trace(
                go.Scatter(
                    x=bd_dates,
                    y=breakdown["assets"],
                    mode="lines",
                    name="Assets",
                    line=dict(color="#16a34a", width=1.5),
                    hovertemplate="$%{y:,.0f}<extra>Assets</extra>",
                )
            )

        # Portfolio value line (sourced from stats.parquet, filled to zero)
        fig.add_trace(
            go.Scatter(
                x=bd_dates,
                y=breakdown["portfolio_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#0891b2", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(8, 145, 178, 0.1)",
                hovertemplate="$%{y:,.0f}<extra>Portfolio Value</extra>",
            )
        )
    else:
        # Fallback: indicator-based equity curve only
        df = pd.DataFrame(equity)
        date_col = "date" if "date" in df.columns else "datetime"
        val_col = "value" if "value" in df.columns else "close"

        if date_col in df.columns and val_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[date_col],
                    y=df[val_col],
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="#0891b2", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(8, 145, 178, 0.1)",
                    hovertemplate="$%{y:,.0f}<extra>Portfolio Value</extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def drawdown_chart(equity: list[dict[str, Any]], title: str = "Drawdown") -> go.Figure:
    """Build a drawdown chart from equity curve data."""
    if not equity:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        fig.update_layout(title=title, template="plotly_white")
        return fig

    df = pd.DataFrame(equity)
    val_col = "value" if "value" in df.columns else "close"
    date_col = "date" if "date" in df.columns else "datetime"

    if val_col not in df.columns:
        return go.Figure()

    values = df[val_col].values
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max * 100

    x_vals = df[date_col] if date_col in df.columns else list(range(len(drawdown)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=drawdown,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#e74c3c", width=1),
            fill="tozeroy",
            fillcolor="rgba(231, 76, 60, 0.15)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig.update_yaxes(tickformat=".1f")
    return fig


def monthly_returns_heatmap(equity: list[dict[str, Any]], title: str = "Monthly Returns") -> go.Figure:
    """Build a monthly returns heatmap from equity curve data."""
    if not equity or len(equity) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for heatmap", showarrow=False)
        return fig

    df = pd.DataFrame(equity)
    val_col = "value" if "value" in df.columns else "close"
    date_col = "date" if "date" in df.columns else "datetime"

    if date_col not in df.columns or val_col not in df.columns:
        return go.Figure()

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    monthly = df[val_col].resample("ME").last()
    returns = monthly.pct_change(fill_method=None).dropna() * 100

    if returns.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough monthly data", showarrow=False)
        return fig

    returns_df = returns.to_frame("return")
    returns_df["year"] = returns_df.index.year
    returns_df["month"] = returns_df.index.month
    pivot = returns_df.pivot_table(values="return", index="year", columns="month", aggfunc="sum")

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    pivot.columns = [month_names.get(c, str(c)) for c in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[str(y) for y in pivot.index],
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Return %"),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Year",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def returns_distribution(
    equity: list[dict[str, Any]],
    title: str = "Daily Returns",
    daily_returns: list[float] | None = None,
    daily_dates: "pd.DatetimeIndex | None" = None,
) -> go.Figure:
    """Build a daily returns time-series bar chart matching the tearsheet view.

    Each day's return is a bar (green if positive, red if negative) with the
    date on the X axis and percentage return on the Y axis.

    When *daily_returns* (list of percentage values) and *daily_dates* are
    provided they are used directly — this is the preferred path because the
    caller can source cash-flow-adjusted returns from stats.parquet's
    ``return`` column.  Otherwise falls back to computing ``pct_change()`` on
    the *equity* curve.
    """
    if daily_returns is not None and daily_dates is not None and len(daily_returns) > 0:
        daily_pct = pd.Series(daily_returns, index=daily_dates)
        daily_pct = daily_pct.dropna()
        if daily_pct.empty:
            fig = go.Figure()
            fig.add_annotation(text="Not enough data", showarrow=False)
            return fig
    elif not equity or len(equity) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", showarrow=False)
        return fig
    else:
        df = pd.DataFrame(equity)
        val_col = "value" if "value" in df.columns else "close"
        date_col = "date" if "date" in df.columns else "datetime"

        if date_col not in df.columns or val_col not in df.columns:
            return go.Figure()

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        daily_pct = df[val_col].pct_change(fill_method=None).dropna() * 100

        if daily_pct.empty:
            return go.Figure()

    # Solid, saturated colors that pop on a white background
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in daily_pct]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_pct.index,
            y=daily_pct.values,
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            name="Daily Return",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        )
    )

    # Zero reference line
    fig.add_hline(y=0, line_width=0.5, line_color="#9ca3af")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily Return (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        bargap=0,
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".2f")
    return fig


def monthly_returns_distribution(
    strategy_monthly: list[float],
    benchmark_monthly: list[float] | None = None,
    title: str = "Distribution of Monthly Returns",
    benchmark_label: str = "SPY",
) -> go.Figure:
    """Build a monthly returns distribution histogram matching the quantstats tearsheet.

    Uses density normalization (histnorm='probability density') and 20 bins
    to match the quantstats ``histogram()`` output.

    Args:
        strategy_monthly: Strategy monthly returns in percent (e.g. [2.5, -1.3, 3.1]).
        benchmark_monthly: Optional benchmark monthly returns in percent.
            Rendered as yellow bars + yellow KDE line.
        title: Chart title.
        benchmark_label: Ticker label for benchmark legend entries (e.g. "SPY", "QQQ").
    """
    s_series = pd.Series(strategy_monthly).dropna()
    bm_series = pd.Series(benchmark_monthly).dropna() if (benchmark_monthly and len(benchmark_monthly) > 0) else pd.Series([], dtype=float)

    if s_series.empty and bm_series.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough monthly data", showarrow=False)
        return fig

    # Shared bin edges so strategy and benchmark bars use identical widths
    all_values = pd.concat([s_series, bm_series])
    bin_start = all_values.min()
    bin_end = all_values.max()
    # Expand range slightly so edge bars aren't flush against the axes
    pad = (bin_end - bin_start) * 0.05 if bin_end > bin_start else 1.0
    bin_start -= pad
    bin_end += pad
    bin_size = (bin_end - bin_start) / 20

    fig = go.Figure()

    # Strategy histogram — density mode, matching quantstats bins=20
    if not s_series.empty:
        fig.add_trace(
            go.Histogram(
                x=s_series,
                xbins=dict(start=bin_start, end=bin_end, size=bin_size),
                histnorm="probability density",
                name="Strategy",
                marker_color="#0891b2",
                opacity=0.7,
            )
        )
    mean = s_series.mean() if not s_series.empty else 0.0
    std = s_series.std() if not s_series.empty else 0.0

    # Benchmark — yellow bars + yellow KDE line
    if not bm_series.empty:
        fig.add_trace(
            go.Histogram(
                x=bm_series,
                xbins=dict(start=bin_start, end=bin_end, size=bin_size),
                histnorm="probability density",
                name=benchmark_label,
                marker_color="#eab308",
                opacity=0.5,
            )
        )
        # Yellow KDE line for benchmark
        bm_std = bm_series.std()
        if bm_std > 0:
            bw = 1.06 * bm_std * len(bm_series) ** (-0.2)
            x_min = bm_series.min() - 3 * bw
            x_max = bm_series.max() + 3 * bw
            x_grid = np.linspace(x_min, x_max, 200)
            y_kde = np.zeros_like(x_grid)
            for xi in bm_series:
                y_kde += np.exp(-0.5 * ((x_grid - xi) / bw) ** 2)
            y_kde /= len(bm_series) * bw * np.sqrt(2 * np.pi)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y_kde,
                    mode="lines",
                    name=f"{benchmark_label} KDE",
                    line=dict(color="#eab308", width=2),
                )
            )

    # Normal fit overlay on strategy (in density space)
    if std > 0:
        x_range = np.linspace(mean - 4 * std, mean + 4 * std, 200)
        y_fit = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        label = f"Normal fit (μ={mean:.2f}%, σ={std:.2f}%)"
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_fit,
                mode="lines",
                name=label,
                line=dict(color="#e74c3c", width=2, dash="dash"),
            )
        )

    # Zero reference line
    fig.add_vline(x=0, line_dash="dot", line_color="#d1d5db", line_width=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Monthly Return (%)",
        yaxis_title="Density",
        template="plotly_white",
        bargap=0.05,
        barmode="overlay",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def cumulative_returns_chart(data: dict[str, Any], title: str = "Cumulative Returns vs Benchmark") -> go.Figure:
    """Build a cumulative returns comparison chart (strategy vs benchmark).

    Args:
        data: dict with keys 'dates' (list[str]), 'strategy' (list[float]),
              'benchmark' (list[float] or None), and optionally
              'benchmark_symbol' (str).  Values are fractional returns
              (0.10 = +10%).

    Returns a Plotly figure with both strategy and benchmark cumulative return lines.
    """
    if not data or not data.get("dates"):
        fig = go.Figure()
        fig.add_annotation(text="No cumulative return data available", showarrow=False)
        fig.update_layout(title=title, template="plotly_white")
        return fig

    dates = pd.to_datetime(data["dates"])
    strategy_pct = [v * 100 for v in data["strategy"]]
    benchmark = data.get("benchmark")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_pct,
            mode="lines",
            name="Strategy",
            line=dict(color="#0891b2", width=2),
            hovertemplate="%{y:.1f}%<extra>Strategy</extra>",
        )
    )

    if benchmark is not None:
        bm_symbol = data.get("benchmark_symbol", "Benchmark")
        benchmark_pct = [v * 100 for v in benchmark]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=benchmark_pct,
                mode="lines",
                name=bm_symbol,
                line=dict(color="#eab308", width=1.5, dash="dash"),
                hovertemplate=f"%{{y:.1f}}%<extra>{bm_symbol}</extra>",
            )
        )

    # Zero reference line
    fig.add_hline(y=0, line_dash="dot", line_color="#d1d5db", line_width=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_yaxes(tickformat=".1f")
    return fig


def _rolling_metric(
    dates: list[str],
    strategy_daily: list[float],
    benchmark_daily: list[float] | None,
    metric: str,
    window: int,
    title: str,
    y_label: str,
    benchmark_label: str = "SPY",
) -> go.Figure:
    """Shared helper for rolling-metric charts (volatility, Sharpe, Sortino).

    Args:
        dates: Daily date strings.
        strategy_daily: Strategy daily fractional returns (0.01 = +1%).
        benchmark_daily: Optional benchmark daily fractional returns.
        metric: One of 'volatility', 'sharpe', 'sortino'.
        window: Rolling window in trading days (126 ≈ 6 months).
        title: Chart title.
        y_label: Y-axis label.
        benchmark_label: Ticker label for benchmark legend (e.g. "SPY", "QQQ").
    """
    s = pd.Series(strategy_daily, index=pd.to_datetime(dates))
    ann = np.sqrt(252)

    if metric == "volatility":
        # Annualized rolling standard deviation, in percent
        s_rolling = s.rolling(window).std() * ann * 100
        if benchmark_daily:
            b = pd.Series(benchmark_daily, index=pd.to_datetime(dates))
            b_rolling = b.rolling(window).std() * ann * 100
    else:
        # Rolling annualized return
        s_ann_ret = s.rolling(window).mean() * 252
        if benchmark_daily:
            b = pd.Series(benchmark_daily, index=pd.to_datetime(dates))
            b_ann_ret = b.rolling(window).mean() * 252

        if metric == "sharpe":
            s_std = s.rolling(window).std() * ann
            s_rolling = s_ann_ret / s_std
            if benchmark_daily:
                b_std = b.rolling(window).std() * ann
                b_rolling = b_ann_ret / b_std
        else:  # sortino
            s_down = s.copy()
            s_down[s_down > 0] = 0
            s_down_std = s_down.rolling(window).std() * ann
            s_rolling = s_ann_ret / s_down_std
            if benchmark_daily:
                b_down = b.copy()
                b_down[b_down > 0] = 0
                b_down_std = b_down.rolling(window).std() * ann
                b_rolling = b_ann_ret / b_down_std

    # Trim NaN lead-in from rolling window
    s_rolling = s_rolling.dropna()
    if benchmark_daily:
        b_rolling = b_rolling.dropna()

    if s_rolling.empty:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for rolling window", showarrow=False)
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=s_rolling.index,
            y=s_rolling.values,
            mode="lines",
            name="Strategy",
            line=dict(color="#0891b2", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>Strategy</extra>",
        )
    )

    if benchmark_daily and not b_rolling.empty:
        fig.add_trace(
            go.Scatter(
                x=b_rolling.index,
                y=b_rolling.values,
                mode="lines",
                name=benchmark_label,
                line=dict(color="#eab308", width=1.5, dash="dash"),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:.2f}}<extra>{benchmark_label}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def rolling_volatility_chart(
    dates: list[str],
    strategy_daily: list[float],
    benchmark_daily: list[float] | None = None,
    window: int = 126,
    title: str = "Rolling Volatility (6M)",
    benchmark_label: str = "SPY",
) -> go.Figure:
    """Annualized rolling volatility over *window* trading days (6 months)."""
    return _rolling_metric(
        dates,
        strategy_daily,
        benchmark_daily,
        metric="volatility",
        window=window,
        title=title,
        y_label="Annualized Volatility (%)",
        benchmark_label=benchmark_label,
    )


def rolling_sharpe_chart(
    dates: list[str],
    strategy_daily: list[float],
    benchmark_daily: list[float] | None = None,
    window: int = 126,
    title: str = "Rolling Sharpe (6M)",
    benchmark_label: str = "SPY",
) -> go.Figure:
    """Annualized rolling Sharpe ratio over *window* trading days."""
    return _rolling_metric(
        dates,
        strategy_daily,
        benchmark_daily,
        metric="sharpe",
        window=window,
        title=title,
        y_label="Sharpe Ratio",
        benchmark_label=benchmark_label,
    )


def trades_chart(trades_data: dict[str, Any], title: str = "Trade Activity") -> go.Figure:
    """Build a trade activity chart with portfolio value curve and buy/sell markers.

    Args:
        trades_data: Dict with ``values`` (list of {time, portfolio_value})
            and ``trades`` (list of {time, side, symbol, qty, price, cost,
            portfolio_value}) from :func:`load_trades_curve`.
        title: Chart title.
    """
    values = trades_data.get("values", [])
    trades = trades_data.get("trades", [])

    if not values and not trades:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available", showarrow=False)
        fig.update_layout(title=title, template="plotly_white")
        return fig

    fig = go.Figure()

    # ── Portfolio value line ──
    if values:
        v_times = [v["time"] for v in values]
        v_vals = [v["portfolio_value"] for v in values]
        fig.add_trace(
            go.Scatter(
                x=v_times,
                y=v_vals,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#0891b2", width=2),
                hovertemplate="$%{y:,.0f}<extra>Portfolio Value</extra>",
            )
        )

    # ── Buy markers (green ▲) ──
    buys = [t for t in trades if t["side"] == "buy"]
    if buys:
        fig.add_trace(
            go.Scatter(
                x=[b["time"] for b in buys],
                y=[b["portfolio_value"] for b in buys],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=10, color="#16a34a", line=dict(width=1, color="#15803d")),
                hovertemplate=("<b>BUY</b> %{customdata[0]}<br>Qty: %{customdata[1]:.0f}<br>Price: $%{customdata[2]:,.2f}<br>Cost: $%{customdata[3]:,.2f}<br>Portfolio: $%{y:,.0f}<extra></extra>"),
                customdata=[(b["symbol"], b["qty"], b["price"], b["cost"]) for b in buys],
            )
        )

    # ── Sell markers (red ▼) ──
    sells = [t for t in trades if t["side"] == "sell"]
    if sells:
        fig.add_trace(
            go.Scatter(
                x=[s["time"] for s in sells],
                y=[s["portfolio_value"] for s in sells],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=10, color="#dc2626", line=dict(width=1, color="#b91c1c")),
                hovertemplate=("<b>SELL</b> %{customdata[0]}<br>Qty: %{customdata[1]:.0f}<br>Price: $%{customdata[2]:,.2f}<br>Cost: $%{customdata[3]:,.2f}<br>Portfolio: $%{y:,.0f}<extra></extra>"),
                customdata=[(s["symbol"], s["qty"], s["price"], s["cost"]) for s in sells],
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def rolling_sortino_chart(
    dates: list[str],
    strategy_daily: list[float],
    benchmark_daily: list[float] | None = None,
    window: int = 126,
    title: str = "Rolling Sortino (6M)",
    benchmark_label: str = "SPY",
) -> go.Figure:
    """Annualized rolling Sortino ratio over *window* trading days."""
    return _rolling_metric(
        dates,
        strategy_daily,
        benchmark_daily,
        metric="sortino",
        window=window,
        title=title,
        y_label="Sortino Ratio",
        benchmark_label=benchmark_label,
    )

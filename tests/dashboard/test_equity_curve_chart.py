from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pandas as pd
import plotly.graph_objects as go
import pytest

from trading_agent_framework.dashboard.components.charts import (
    REGIME_BAND_COLORS,
    benchmark_equity,
    drawdown_chart,
    equity_curve_chart,
    monthly_returns_heatmap,
)

T0 = datetime(2026, 9, 1, 20, 0, tzinfo=UTC)
DATES = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]
BREAKDOWN = {
    "dates": DATES,
    "portfolio_value": [1000.0, 1100.0, 1050.0, 1200.0],
    "cash": [1000.0, 200.0, 100.0, 300.0],
    "assets": [0.0, 900.0, 950.0, 900.0],
}
EQUITY = [{"date": d, "value": v} for d, v in zip(DATES, BREAKDOWN["portfolio_value"], strict=True)]
CUMULATIVE = {"dates": DATES, "benchmark": [0.0, 0.10, -0.05, 0.05], "benchmark_symbol": "SPY"}


def _trace(fig: go.Figure, name: str) -> Any:
    """The trace called `name` (typed Any: plotly's stubs do not expose trace attributes)."""
    return next(t for t in cast(Any, fig.data) if t.name == name)


def _regime(values: list[int]) -> dict:
    times = [T0 + timedelta(days=i) for i in range(len(values))]
    return {"Regime": [{"name": "Regime", "color": "#d1d4dc", "dash": "solid", "times": times, "values": [float(v) for v in values]}]}


def test_benchmark_equity_grows_the_budget_by_the_cumulative_benchmark_return() -> None:
    curve = benchmark_equity(CUMULATIVE, budget=1000.0)
    assert curve == {"label": "SPY", "dates": DATES, "values": pytest.approx([1000.0, 1100.0, 950.0, 1050.0])}


def test_benchmark_equity_is_none_without_a_recorded_benchmark() -> None:
    assert benchmark_equity({**CUMULATIVE, "benchmark": None}, budget=1000.0) is None
    assert benchmark_equity(None, budget=1000.0) is None


def test_equity_curve_draws_the_benchmark_as_a_dashed_orange_line() -> None:
    fig = equity_curve_chart(EQUITY, breakdown=BREAKDOWN, benchmark=benchmark_equity(CUMULATIVE, 1000.0))
    spy = _trace(fig, "SPY")
    assert spy.line.dash == "dash" and spy.line.color == "#f59e0b"
    assert list(spy.y) == pytest.approx([1000.0, 1100.0, 950.0, 1050.0])


def test_equity_curve_draws_the_benchmark_in_the_fallback_path_too() -> None:
    fig = equity_curve_chart(EQUITY, benchmark=benchmark_equity(CUMULATIVE, 1000.0))
    assert {t.name for t in cast(Any, fig.data)} == {"Portfolio Value", "SPY"}


def test_equity_curve_without_benchmark_has_no_spy_trace() -> None:
    fig = equity_curve_chart(EQUITY, breakdown=BREAKDOWN)
    assert "SPY" not in {t.name for t in cast(Any, fig.data)}


def test_equity_curve_without_indicators_is_a_single_pane() -> None:
    fig = equity_curve_chart(EQUITY, breakdown=BREAKDOWN)
    assert not fig.layout.shapes and "yaxis2" not in fig.to_dict()["layout"]


def test_the_regime_row_sits_under_the_equity_curve_with_bands_on_both_rows() -> None:
    fig = equity_curve_chart(EQUITY, breakdown=BREAKDOWN, indicators=_regime([1, 1, 0, -1]))
    rows = {t.name: t.yaxis for t in cast(Any, fig.data)}
    assert (rows["Portfolio Value"], rows["Regime"]) == ("y", "y2")
    assert tuple(fig.layout.yaxis2.ticktext) == ("Bearish", "Neutral", "Bullish")
    bands = [s for s in fig.layout.shapes if s.fillcolor]
    expected = [REGIME_BAND_COLORS[1], REGIME_BAND_COLORS[0], REGIME_BAND_COLORS[-1]]
    assert [s.fillcolor for s in bands if s.xref == "x"] == expected
    assert [s.fillcolor for s in bands if s.xref == "x2"] == expected
    assert all(pd.Timestamp(x).tzinfo is None for x in _trace(fig, "Regime").x)


def test_the_benchmark_and_the_regime_go_together_on_the_top_row() -> None:
    fig = equity_curve_chart(EQUITY, breakdown=BREAKDOWN, indicators=_regime([1, 0]), benchmark=benchmark_equity(CUMULATIVE, 1000.0))
    assert _trace(fig, "SPY").yaxis == "y"


def test_an_empty_equity_curve_ignores_the_regime() -> None:
    fig = equity_curve_chart([], indicators=_regime([1]))
    assert not fig.data


def test_drawdown_overlays_the_benchmark_drawdown() -> None:
    fig = drawdown_chart(EQUITY, benchmark=benchmark_equity(CUMULATIVE, 1000.0))
    spy = _trace(fig, "SPY Drawdown %")
    assert spy.line.dash == "dash" and spy.line.color == "#f59e0b"
    # peak 1100 -> 950 is -13.636..%, then 1050 is -4.545..%
    assert list(spy.y) == pytest.approx([0.0, 0.0, -150 / 1100 * 100, -50 / 1100 * 100])
    assert list(spy.x) == DATES


def test_drawdown_without_benchmark_is_unchanged() -> None:
    fig = drawdown_chart(EQUITY)
    assert [t.name for t in cast(Any, fig.data)] == ["Drawdown %"]


def test_drawdown_legend_overlays_the_plot_on_the_left() -> None:
    fig = drawdown_chart(EQUITY, benchmark=benchmark_equity(CUMULATIVE, 1000.0))
    legend = fig.layout.legend
    assert legend.xanchor == "left" and 0 <= legend.x < 0.5  # inside the plot, not in the right margin


def test_drawdown_and_equity_share_horizontal_margins_so_their_plots_align() -> None:
    equity = equity_curve_chart(EQUITY, breakdown=BREAKDOWN, indicators=_regime([1, 0]))
    drawdown = drawdown_chart(EQUITY)
    assert (equity.layout.margin.l, equity.layout.margin.r) == (drawdown.layout.margin.l, drawdown.layout.margin.r)
    # A tick label wider on one chart must not move its plot area.
    assert equity.layout.yaxis.automargin is False and drawdown.layout.yaxis.automargin is False
    # The regime row's "Bullish"/"Bearish" labels are wider than any tick: they must not grow the margin either.
    assert equity.layout.yaxis2.automargin is False


def test_drawdown_and_equity_span_the_same_dates_even_when_the_regime_starts_earlier() -> None:
    early = {"Regime": [{**_regime([1, 0, -1, 1, 0, 1])["Regime"][0], "times": [T0 - timedelta(days=30 - i) for i in range(6)]}]}
    equity = equity_curve_chart(EQUITY, breakdown=BREAKDOWN, indicators=early)
    drawdown = drawdown_chart(EQUITY)
    assert tuple(pd.Timestamp(x) for x in equity.layout.xaxis.range) == tuple(pd.Timestamp(x) for x in drawdown.layout.xaxis.range)
    assert pd.Timestamp(drawdown.layout.xaxis.range[0]) == pd.Timestamp(DATES[0])
    assert pd.Timestamp(drawdown.layout.xaxis.range[1]) == pd.Timestamp(DATES[-1])


def test_drawdown_takes_the_budget_as_the_first_peak() -> None:
    """A first-session loss was invisible: the first close was its own peak, while the Max DD card counted it."""
    fig = drawdown_chart(EQUITY, benchmark=benchmark_equity(CUMULATIVE, 1100.0), budget=1100.0)

    # strategy: 1000 is already under the 1100 budget; SPY starts on it: 1100, 1210, 1045, 1155
    assert list(_trace(fig, "Drawdown %").y) == pytest.approx([-100 / 1100 * 100, 0.0, -50 / 1100 * 100, 0.0])
    assert list(_trace(fig, "SPY Drawdown %").y) == pytest.approx([0.0, 0.0, -165 / 1210 * 100, -55 / 1210 * 100])


def test_monthly_heatmap_measures_the_first_month_from_the_budget() -> None:
    """The first month was missing: it had no month-end before it to be measured from."""
    equity = [{"date": "2026-01-30", "value": 1100.0}, {"date": "2026-02-27", "value": 1210.0}]

    fig = monthly_returns_heatmap(equity, budget=1000.0)

    heatmap = cast(Any, fig.data)[0]
    assert list(heatmap.x) == ["Jan", "Feb"]
    assert list(heatmap.z[0]) == pytest.approx([10.0, 10.0])


def test_monthly_heatmap_shows_a_run_shorter_than_two_months() -> None:
    fig = monthly_returns_heatmap([{"date": "2026-11-03", "value": 1010.0}], budget=1000.0)

    assert list(cast(Any, fig.data)[0].z[0]) == pytest.approx([1.0])

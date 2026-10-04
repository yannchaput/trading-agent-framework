from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from trading_agent_framework.dashboard.components.charts import REGIME_BAND_COLORS, cumulative_returns_chart

T0 = datetime(2026, 9, 1, 20, 0, tzinfo=UTC)  # a session close, as the regime line is stamped
CUMULATIVE = {
    "dates": ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-08"],
    "strategy": [0.0, 0.01, 0.02, 0.01, 0.03],
    "benchmark": [0.0, 0.005, 0.0, -0.01, 0.0],
    "benchmark_symbol": "SPY",
}


def _regime(values: list[int]) -> dict:
    times = [T0 + timedelta(days=i) for i in range(len(values))]
    return {"Regime": [{"name": "Regime", "color": "#d1d4dc", "dash": "solid", "times": times, "values": [float(v) for v in values]}]}


def test_without_indicators_the_chart_is_a_single_pane() -> None:
    fig = cumulative_returns_chart(CUMULATIVE)
    assert {t.name for t in fig.data} == {"Strategy", "SPY"}
    assert len(fig.layout.shapes) == 1  # the zero line only


def test_indicator_lines_other_than_the_regime_are_never_drawn() -> None:
    legacy = {"VIX": [{"name": "VIX", "color": "#fff", "dash": "solid", "times": [T0], "values": [18.0]}]}
    fig = cumulative_returns_chart(CUMULATIVE, indicators=legacy)
    assert {t.name for t in fig.data} == {"Strategy", "SPY"}
    assert len(fig.layout.shapes) == 1  # the zero line only: no regime bands


def test_the_regime_row_sits_under_the_cumulative_return_on_the_same_time_axis() -> None:
    fig = cumulative_returns_chart(CUMULATIVE, indicators=_regime([1, 1, 0, -1]))
    rows = {t.name: t.yaxis for t in fig.data}
    assert (rows["Strategy"], rows["SPY"], rows["Regime"]) == ("y", "y", "y2")
    assert fig.layout.xaxis.matches == "x2" and fig.layout.xaxis2.matches is None
    assert tuple(fig.layout.yaxis2.range) == (-1.3, 1.3)
    assert tuple(fig.layout.yaxis2.ticktext) == ("Bearish", "Neutral", "Bullish")
    regime = next(t for t in fig.data if t.name == "Regime")
    assert regime.line.shape == "hv" and list(regime.y) == [1, 1, 0, -1]
    portfolio, regime_row = fig.layout.yaxis.domain, fig.layout.yaxis2.domain
    assert (regime_row[1] - regime_row[0]) < (portfolio[1] - portfolio[0])


def test_each_regime_run_is_shaded_on_both_rows_up_to_the_last_date() -> None:
    fig = cumulative_returns_chart(CUMULATIVE, indicators=_regime([1, 1, 0, -1]))
    bands = [s for s in fig.layout.shapes if s.fillcolor]  # the zero line is not a band
    on_returns = [s for s in bands if s.xref == "x"]
    on_regime = [s for s in bands if s.xref == "x2"]
    expected = [REGIME_BAND_COLORS[1], REGIME_BAND_COLORS[0], REGIME_BAND_COLORS[-1]]
    assert [s.fillcolor for s in on_returns] == expected
    assert [s.fillcolor for s in on_regime] == expected
    assert on_returns[-1].x1 == pd.Timestamp("2026-09-08")  # the last run reaches the end of the cumulative curve


def test_the_regime_shares_the_naive_date_axis_of_the_cumulative_curve() -> None:
    # load_cumulative_returns gives naive "YYYY-MM-DD" dates; a tz-aware regime time would not line up with them.
    fig = cumulative_returns_chart(CUMULATIVE, indicators=_regime([1, 0]))
    regime = next(t for t in fig.data if t.name == "Regime")
    assert all(pd.Timestamp(x).tzinfo is None for x in regime.x)
    assert all(pd.Timestamp(s.x0).tzinfo is None for s in fig.layout.shapes if s.fillcolor)


def test_a_chart_without_data_ignores_the_regime() -> None:
    fig = cumulative_returns_chart({}, indicators=_regime([1]))
    assert not fig.data and not fig.layout.shapes


def test_the_zero_line_survives_the_regime_bands() -> None:
    fig = cumulative_returns_chart(CUMULATIVE, indicators=_regime([1, 0]))
    zero_lines = [s for s in fig.layout.shapes if not s.fillcolor]
    assert len(zero_lines) == 1 and zero_lines[0].y0 == 0 and zero_lines[0].y1 == 0

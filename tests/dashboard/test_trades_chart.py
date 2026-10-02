from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import perf_counter

from trading_agent_framework.dashboard.components.charts import REGIME_BAND_COLORS, trades_chart

T0 = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
TRADES = {
    "values": [{"time": T0, "portfolio_value": 10000.0}, {"time": T0 + timedelta(days=5), "portfolio_value": 10100.0}],
    "trades": [{"time": T0, "side": "buy", "symbol": "AAA", "qty": 1.0, "price": 10.0, "cost": 0.0, "portfolio_value": 10000.0}],
}


def _line(name: str) -> dict:
    return {"name": name, "color": "#ffffff", "dash": "solid", "times": [T0], "values": [1.0]}


def _regime(values: list[int]) -> dict:
    times = [T0 + timedelta(days=i) for i in range(len(values))]
    return {"Regime": [{"name": "Regime", "color": "#d1d4dc", "dash": "solid", "times": times, "values": [float(v) for v in values]}]}


def test_without_indicators_the_chart_is_a_single_pane() -> None:
    fig = trades_chart(TRADES)
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy"}
    assert len(fig._grid_ref) == 1
    assert not fig.layout.shapes


def test_indicator_lines_other_than_the_regime_are_never_drawn() -> None:
    # Runs recorded before the regime existed still hold ADX / RSI / VIX panes in indicators.parquet.
    legacy = {"ADX / RSI": [_line("ADX"), _line("RSI")], "VIX": [_line("VIX")], "Indicators": [_line("SMA")]}
    fig = trades_chart(TRADES, indicators=legacy)
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy"}
    assert len(fig._grid_ref) == 1 and not fig.layout.shapes  # no regime logged: the single portfolio pane


def test_a_legacy_run_shows_only_the_regime_row_when_the_regime_was_logged() -> None:
    legacy = {"ADX / RSI": [_line("ADX"), _line("RSI")], "VIX": [_line("VIX")]}
    fig = trades_chart(TRADES, indicators={**legacy, **_regime([1, 0])})
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy", "Regime"}
    assert len(fig._grid_ref) == 2  # portfolio + regime, nothing else


def test_the_regime_row_sits_right_under_the_portfolio_chart_on_the_same_time_axis() -> None:
    fig = trades_chart(TRADES, indicators=_regime([1, 1, 0, -1]))
    rows = {t.name: t.yaxis for t in fig.data}
    assert (rows["Portfolio Value"], rows["Regime"]) == ("y", "y2")
    # plotly's shared_xaxes links every upper row to the bottom axis (x2): one time scale
    assert fig.layout.xaxis.matches == "x2" and fig.layout.xaxis2.matches is None
    assert tuple(fig.layout.yaxis2.range) == (-1.3, 1.3)
    assert tuple(fig.layout.yaxis2.tickvals) == (-1, 0, 1)
    assert tuple(fig.layout.yaxis2.ticktext) == ("Bearish", "Neutral", "Bullish")
    regime = next(t for t in fig.data if t.name == "Regime")
    assert regime.line.shape == "hv" and list(regime.y) == [1, 1, 0, -1]
    portfolio, regime_row = fig.layout.yaxis.domain, fig.layout.yaxis2.domain
    assert (regime_row[1] - regime_row[0]) < (portfolio[1] - portfolio[0])


def test_the_regime_series_is_picked_by_name_and_another_series_in_that_pane_is_ignored() -> None:
    regime_series = _regime([1, 1, 0, -1])["Regime"][0]
    fig = trades_chart(TRADES, indicators={"Regime": [_line("Fast"), regime_series]})
    regime = next(t for t in fig.data if t.name == "Regime")
    assert regime.line.shape == "hv" and list(regime.y) == [1, 1, 0, -1] and regime.yaxis == "y2"
    assert "Fast" not in {t.name for t in fig.data}
    assert len(fig.layout.shapes) == 2 * 3  # the bands come from the regime series: 3 runs, 2 rows


def test_each_regime_run_is_shaded_on_both_rows() -> None:
    fig = trades_chart(TRADES, indicators=_regime([1, 1, 0, -1]))  # three runs: bullish, neutral, bearish
    on_portfolio = [s for s in fig.layout.shapes if s.xref == "x"]
    on_regime = [s for s in fig.layout.shapes if s.xref == "x2"]
    expected = [REGIME_BAND_COLORS[1], REGIME_BAND_COLORS[0], REGIME_BAND_COLORS[-1]]
    assert [s.fillcolor for s in on_portfolio] == expected
    assert [s.fillcolor for s in on_regime] == expected
    assert all(s.opacity < 0.2 for s in on_portfolio) and all(s.opacity > s2.opacity for s, s2 in zip(on_regime, on_portfolio, strict=True))
    # a run ends where the next one starts; the last one runs to the end of the chart (the last portfolio value)
    assert [(s.x0, s.x1) for s in on_portfolio] == [
        (T0, T0 + timedelta(days=2)),
        (T0 + timedelta(days=2), T0 + timedelta(days=3)),
        (T0 + timedelta(days=3), T0 + timedelta(days=5)),
    ]


def test_a_single_regime_point_makes_one_band_per_row() -> None:
    fig = trades_chart(TRADES, indicators=_regime([-1]))
    assert len(fig.layout.shapes) == 2
    assert {s.fillcolor for s in fig.layout.shapes} == {REGIME_BAND_COLORS[-1]}


def test_a_regime_with_many_runs_builds_quickly() -> None:
    runs = 300
    start = perf_counter()
    fig = trades_chart(TRADES, indicators=_regime([1 if i % 2 == 0 else -1 for i in range(runs)]))
    assert perf_counter() - start < 2.0  # per-run add_vrect was quadratic: ~4x per doubling
    assert len(fig.layout.shapes) == 2 * runs


def test_a_regime_only_chart_matches_its_upper_axis_to_the_bottom_one() -> None:
    fig = trades_chart(TRADES, indicators=_regime([1, 0]))
    assert fig.layout.xaxis.matches == "x2" and fig.layout.xaxis2.matches is None

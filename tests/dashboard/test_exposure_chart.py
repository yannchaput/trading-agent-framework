from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import intraday_exposure_chart

ROWS = [
    {"date": "2026-01-05", "peak_invested": 2000.0, "peak_pct": 20.0, "max_positions": 2},
    {"date": "2026-01-06", "peak_invested": 0.0, "peak_pct": 0.0, "max_positions": 0},
]


def test_one_bar_per_day_at_its_peak_percent_of_equity() -> None:
    fig = intraday_exposure_chart(ROWS)

    bars = fig.data[0]
    assert list(bars.x) == ["2026-01-05", "2026-01-06"]
    assert list(bars.y) == [20.0, 0.0]


def test_the_hover_carries_the_dollar_peak_and_the_position_count() -> None:
    fig = intraday_exposure_chart(ROWS)

    assert [tuple(c) for c in fig.data[0].customdata] == [(2000.0, 2), (0.0, 0)]


def test_no_rows_gives_an_empty_chart_with_a_note() -> None:
    fig = intraday_exposure_chart([])

    assert not fig.data
    assert fig.layout.annotations[0].text == "No trade data available"

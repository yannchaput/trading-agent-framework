from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import CHART_TEMPLATE, account_pnl_chart, signed_usd

DAYS = [
    {"date": "2026-10-05", "realized": 0.0, "unrealized": 49.0, "change": 49.0},
    {"date": "2026-10-06", "realized": -12.5, "unrealized": -3.0, "change": -64.5},
]


def test_one_line_with_markers_of_the_days_pnl_change() -> None:
    fig = account_pnl_chart(DAYS)

    (day,) = fig.data
    assert (day.type, day.name, day.mode) == ("scatter", "Day", "lines+markers")
    assert list(day.x) == ["2026-10-05", "2026-10-06"]
    assert list(day.y) == [49.0, -64.5]  # not realized + unrealized (-15.5): the open P&L is a level, not a day


def test_markers_are_green_for_a_gain_and_red_for_a_loss() -> None:
    (day,) = account_pnl_chart(DAYS).data

    assert list(day.marker.color) == ["#16a34a", "#dc2626"]


def test_hover_gives_the_realized_and_open_parts_of_each_day() -> None:
    (day,) = account_pnl_chart(DAYS).data

    assert [list(row) for row in day.customdata] == [[0.0, 49.0], [-12.5, -3.0]]
    assert all(word in day.hovertemplate for word in ("Realized", "Open"))


def test_the_chart_uses_the_dark_template() -> None:
    assert account_pnl_chart(DAYS).layout.template == CHART_TEMPLATE


def test_no_day_shows_the_no_data_annotation() -> None:
    fig = account_pnl_chart([])

    assert fig.data == ()
    assert fig.layout.annotations[0].text == "No trade data available"


def test_signed_usd_formats_both_signs() -> None:
    assert (signed_usd(1234.5), signed_usd(-0.05)) == ("+$1,234.50", "-$0.05")

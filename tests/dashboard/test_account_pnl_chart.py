from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import CHART_TEMPLATE, account_pnl_chart, signed_usd

DAYS = [
    {"date": "2026-10-05", "realized": 0.0, "unrealized": 49.0},
    {"date": "2026-10-06", "realized": -12.5, "unrealized": -3.0},
]


def test_two_grouped_bar_series_realized_then_unrealized() -> None:
    fig = account_pnl_chart(DAYS)

    realized, unrealized = fig.data
    assert (realized.type, realized.name, list(realized.x), list(realized.y)) == ("bar", "Realized", ["2026-10-05", "2026-10-06"], [0.0, -12.5])
    assert (unrealized.type, unrealized.name, list(unrealized.y)) == ("bar", "Unrealized", [49.0, -3.0])
    assert fig.layout.barmode == "group"


def test_bars_are_green_for_a_gain_and_red_for_a_loss_unrealized_hatched() -> None:
    realized, unrealized = account_pnl_chart(DAYS).data

    assert list(realized.marker.color) == ["#16a34a", "#dc2626"]
    assert list(unrealized.marker.color) == ["#16a34a", "#dc2626"]
    assert unrealized.marker.pattern.shape == "/"


def test_the_chart_uses_the_dark_template() -> None:
    assert account_pnl_chart(DAYS).layout.template == CHART_TEMPLATE


def test_no_day_shows_the_no_data_annotation() -> None:
    fig = account_pnl_chart([])

    assert fig.data == ()
    assert fig.layout.annotations[0].text == "No trade data available"


def test_signed_usd_formats_both_signs() -> None:
    assert (signed_usd(1234.5), signed_usd(-0.05)) == ("+$1,234.50", "-$0.05")

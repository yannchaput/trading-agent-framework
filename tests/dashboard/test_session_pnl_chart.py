from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import session_pnl_chart

ROWS = [
    {
        "date": "2026-01-05",
        "pnl": 50.0,
        "positions": [
            {"symbol": "AAA", "qty": 10.0, "avg_cost": 100.0, "sell_price": 110.0, "pnl": 100.0, "pnl_pct": 10.0},
            {"symbol": "BBB", "qty": 5.0, "avg_cost": 200.0, "sell_price": 190.0, "pnl": -50.0, "pnl_pct": -5.0},
        ],
    },
    {
        "date": "2026-01-06",
        "pnl": -30.0,
        "positions": [{"symbol": "CCC", "qty": 3.0, "avg_cost": 50.0, "sell_price": 40.0, "pnl": -30.0, "pnl_pct": -20.0}],
    },
]


def test_one_point_per_session_at_its_total_pnl_joined_by_a_line() -> None:
    fig = session_pnl_chart(ROWS)

    line = fig.data[0]
    assert list(line.x) == ["2026-01-05", "2026-01-06"]
    assert list(line.y) == [50.0, -30.0]
    assert line.mode == "lines+markers"


def test_the_hover_of_a_point_details_every_position_sold_that_session() -> None:
    fig = session_pnl_chart(ROWS)

    first, second = fig.data[0].hovertext
    assert "AAA" in first and "BBB" in first and "CCC" not in first
    assert "+$100.00" in first and "-$50.00" in first
    assert "CCC" in second and "-$30.00" in second


def test_a_gain_is_marked_green_and_a_loss_red() -> None:
    fig = session_pnl_chart(ROWS)

    green, red = fig.data[0].marker.color
    assert green != red


def test_no_rows_gives_an_empty_chart_with_a_note() -> None:
    fig = session_pnl_chart([])

    assert not fig.data
    assert fig.layout.annotations[0].text == "No trade data available"

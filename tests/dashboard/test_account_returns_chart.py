from __future__ import annotations

from trading_agent_framework.dashboard.components.charts import BENCHMARK_COLOR, CHART_TEMPLATE, STRATEGY_COLOR, account_returns_chart

ROWS = [
    {"date": "2026-10-12", "strategy": 2.0, "benchmark": 1.0},
    {"date": "2026-10-13", "strategy": -0.5, "benchmark": 0.25},
]


def test_one_line_per_series_in_the_backtest_colours() -> None:
    strategy, spy = account_returns_chart(ROWS).data

    assert (strategy.name, spy.name) == ("Strategy", "SPY")
    assert (strategy.line.color, spy.line.color) == (STRATEGY_COLOR, BENCHMARK_COLOR) == ("#0891b2", "#eab308")
    assert list(strategy.y) == [2.0, -0.5] and list(spy.y) == [1.0, 0.25]
    assert list(strategy.x) == ["2026-10-12", "2026-10-13"]


def test_the_last_value_of_each_line_is_written_at_its_end_in_the_lines_colour() -> None:
    strategy, spy = account_returns_chart(ROWS).data

    assert list(strategy.text) == ["", "Strategy -0.50%"] and list(spy.text) == ["", "SPY +0.25%"]
    assert (strategy.textfont.color, spy.textfont.color) == (STRATEGY_COLOR, BENCHMARK_COLOR)


def test_the_higher_end_label_goes_above_and_the_lower_below() -> None:
    strategy, spy = account_returns_chart(ROWS).data  # SPY ends higher

    assert (spy.textposition, strategy.textposition) == ("top right", "bottom right")


def test_a_day_without_a_value_is_a_gap_and_the_label_sits_on_the_last_known_point() -> None:
    rows = [{"date": "2026-10-12", "strategy": 1.5, "benchmark": 0.5}, {"date": "2026-10-13", "strategy": None, "benchmark": 0.75}]

    strategy, spy = account_returns_chart(rows).data

    assert list(strategy.text) == ["Strategy +1.50%", ""]
    assert spy.connectgaps is False and list(spy.text) == ["", "SPY +0.75%"]


def test_the_benchmark_label_is_the_one_given() -> None:
    assert account_returns_chart(ROWS, "QQQ").data[1].name == "QQQ"


def test_the_chart_uses_the_dark_template() -> None:
    assert account_returns_chart(ROWS).layout.template == CHART_TEMPLATE


def test_no_row_or_no_value_shows_the_no_data_annotation() -> None:
    for rows in ([], [{"date": "2026-10-12", "strategy": None, "benchmark": None}]):
        fig = account_returns_chart(rows)

        assert fig.data == ()
        assert fig.layout.annotations[0].text == "No return data available"

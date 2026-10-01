from __future__ import annotations

from datetime import UTC, datetime

from trading_agent_framework.dashboard.components.charts import trades_chart

T0 = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
TRADES = {
    "values": [{"time": T0, "portfolio_value": 10000.0}],
    "trades": [{"time": T0, "side": "buy", "symbol": "AAA", "qty": 1.0, "price": 10.0, "cost": 0.0, "portfolio_value": 10000.0}],
}


def _line(name: str) -> dict:
    return {"name": name, "color": "#ffffff", "dash": "solid", "times": [T0], "values": [1.0]}


def test_without_indicators_the_chart_is_a_single_pane() -> None:
    fig = trades_chart(TRADES)
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy"}
    assert len(fig._grid_ref) == 1


def test_each_indicator_pane_gets_its_own_sub_row_under_the_portfolio_chart() -> None:
    fig = trades_chart(TRADES, indicators={"VIX": [_line("VIX")], "ADX / RSI": [_line("ADX"), _line("RSI")]})
    assert {t.name for t in fig.data} == {"Portfolio Value", "Buy", "VIX", "ADX", "RSI"}
    assert len(fig._grid_ref) == 3  # portfolio + two panes
    rows = {t.name: t.yaxis for t in fig.data}
    assert rows["ADX"] == rows["RSI"] != rows["VIX"] != rows["Portfolio Value"]

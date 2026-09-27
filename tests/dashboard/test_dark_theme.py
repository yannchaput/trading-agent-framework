from __future__ import annotations

from pathlib import Path

import pytest

from trading_agent_framework.dashboard.components import charts
from trading_agent_framework.dashboard.theme import THEME_CSS

DASHBOARD = Path(charts.__file__).resolve().parents[1]


def test_css_has_no_white_background_left() -> None:
    assert "#ffffff" not in THEME_CSS.lower()
    assert "#131722" in THEME_CSS  # metric card background


@pytest.mark.parametrize(
    "build",
    [
        lambda: charts.equity_curve_chart([]),
        lambda: charts.drawdown_chart([]),
        lambda: charts.cumulative_returns_chart({}),
        lambda: charts.trades_chart({}),
    ],
)
def test_charts_use_the_transparent_dark_template(build) -> None:
    template = build().layout.template

    assert template.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert template.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_no_chart_is_left_on_the_white_template() -> None:
    for path in [DASHBOARD / "components" / "charts.py", *(DASHBOARD / "_pages").glob("*.py")]:
        assert "plotly_white" not in path.read_text(encoding="utf-8"), path.name

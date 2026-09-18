from __future__ import annotations

import importlib
from pathlib import Path

import pytest

MODULES = [
    "trading_agent_framework.dashboard.models",
    "trading_agent_framework.dashboard.discovery",
    "trading_agent_framework.dashboard.reader",
    "trading_agent_framework.dashboard.theme",
    "trading_agent_framework.dashboard.components.charts",
    "trading_agent_framework.dashboard.components.tables",
    "trading_agent_framework.dashboard.components.metric_cards",
    "trading_agent_framework.dashboard._pages.scorecard",
    "trading_agent_framework.dashboard._pages.detail",
    "trading_agent_framework.dashboard._pages.side_by_side",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_dashboard_module_imports_cleanly(module_name: str) -> None:
    importlib.import_module(module_name)


def test_pyproject_dashboard_script_points_at_the_real_package() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "trading_agent_framework.dashboard.cli:main" in text
    assert "lumibot_trading_agent" not in text

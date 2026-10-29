from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

STRATEGIES = ["alpha", "beta", "gamma", "delta"]


def _switching_script() -> None:
    # Stands in for the top navigation: session_state["tab"] picks the tab drawn on this run.
    import streamlit as st

    from trading_agent_framework.dashboard._pages.scorecard import page_scorecard

    if st.session_state.get("tab", "scorecard") == "scorecard":
        page_scorecard()
    else:
        st.write("another tab")


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # page_scorecard scans the relative "logs" directory.
    for strategy in STRATEGIES:
        run_dir = tmp_path / "logs" / strategy / "backtesting" / "2026-06-22_194053_backtesting"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
        (run_dir / "settings.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "logs"


def test_strategy_picker_defaults_to_the_first_three_strategies(logs: Path) -> None:
    at = AppTest.from_function(_switching_script, default_timeout=30).run()

    assert not at.exception
    assert at.sidebar.multiselect[0].value == ["alpha", "beta", "delta"]  # strategy_names() is sorted


def test_strategy_picker_keeps_the_choice_across_tabs(logs: Path) -> None:
    # Streamlit drops a widget's state on a run that does not draw it (another tab): the picker must keep the choice.
    at = AppTest.from_function(_switching_script, default_timeout=30).run()
    at.sidebar.multiselect[0].set_value(["gamma"]).run()
    assert at.sidebar.multiselect[0].value == ["gamma"]

    at.session_state["tab"] = "other"
    at.run()
    at.session_state["tab"] = "scorecard"
    at.run()

    assert not at.exception
    assert at.sidebar.multiselect[0].value == ["gamma"]

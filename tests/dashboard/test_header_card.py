from __future__ import annotations

from streamlit.testing.v1 import AppTest


def _markdown(parameters: dict | None) -> list[str]:
    def script() -> None:
        import streamlit as st

        from trading_agent_framework.dashboard.components.metric_cards import render_header_card

        render_header_card(
            strategy_name="cross_momentum",
            run_ts="2026-11-03_10-00-00",
            mode="backtesting",
            budget=100_000,
            period_start="2020-01-01",
            period_end="2025-12-31",
            parameters=st.session_state["parameters"],
        )

    at = AppTest.from_function(script, default_timeout=30)
    at.session_state["parameters"] = parameters
    at.run()
    assert not at.exception
    return [m.value for m in at.markdown]


def test_a_point_in_time_run_says_the_universe_is_as_it_stood_on_each_date() -> None:
    parameters = {"universe": {"mode": "point_in_time", "history_built": "2026-11-03", "pool_date": "2026-11-01", "top_n": 1200}}

    assert "**Universe:** as it stood on each date (delisted stocks missing)" in _markdown(parameters)


def test_a_run_without_the_universe_record_says_today_s_list() -> None:
    assert "**Universe:** today's list" in _markdown({"budget": 1})


def test_a_run_without_parameters_says_today_s_list() -> None:
    assert "**Universe:** today's list" in _markdown(None)


def test_another_universe_mode_says_today_s_list() -> None:
    assert "**Universe:** today's list" in _markdown({"universe": {"mode": "static"}})


def test_a_malformed_universe_record_says_today_s_list() -> None:
    assert "**Universe:** today's list" in _markdown({"universe": "point_in_time"})

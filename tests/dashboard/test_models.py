from __future__ import annotations

import pytest

from trading_agent_framework.dashboard.models import RunRef, Settings


def test_from_path_parses_this_projects_log_layout() -> None:
    ref = RunRef.from_path("logs/momentum/backtesting/2026-06-22_194053_backtesting")
    assert ref.strategy_name == "momentum"
    assert ref.mode == "backtesting"
    assert ref.run_ts == "2026-06-22_194053"
    assert ref.path == "logs/momentum/backtesting/2026-06-22_194053_backtesting"


def test_from_path_parses_paper_and_live_modes() -> None:
    assert RunRef.from_path("logs/momentum/paper/2026-06-22_194053_paper").mode == "paper"
    assert RunRef.from_path("logs/momentum/live/2026-06-22_194053_live").mode == "live"


def test_from_path_rejects_a_path_with_no_recognizable_mode() -> None:
    with pytest.raises(ValueError, match="Cannot parse"):
        RunRef.from_path("logs/momentum/2026-06-22_194053")


def test_from_path_handles_a_trailing_slash() -> None:
    ref = RunRef.from_path("logs/momentum/backtesting/2026-06-22_194053_backtesting/")
    assert ref.run_ts == "2026-06-22_194053"


def test_settings_dashboard_decision_defaults_to_unset() -> None:
    assert Settings.model_validate({}).dashboard_decision == ""


def test_settings_dashboard_regime_defaults_to_unset_and_parses() -> None:
    assert Settings.model_validate({}).dashboard_regime == ""
    assert Settings.model_validate({"dashboard_regime": "Neutral"}).dashboard_regime == "Neutral"


def test_settings_parses_dashboard_decision_and_agents() -> None:
    settings = Settings.model_validate({
        "dashboard_decision": "validated",
        "agents": {"trader": {"model": "qwen3.6-35b-a3b-awq"}},
    })
    assert settings.dashboard_decision == "validated"
    assert settings.agents["trader"]["model"] == "qwen3.6-35b-a3b-awq"


def test_settings_fees_default_to_empty_for_runs_written_before_fee_totals() -> None:
    assert Settings.model_validate({}).fees == {}


def test_settings_parses_the_fee_totals_broker() -> None:
    settings = Settings.model_validate({"fees": {"broker": "alpaca", "buy_orders": 3}})
    assert settings.fees["broker"] == "alpaca"

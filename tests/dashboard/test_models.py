from __future__ import annotations

import pytest

from trading_agent_framework.dashboard.models import RunRef


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

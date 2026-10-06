from __future__ import annotations

import pytest

from trading_agent_framework.strategies.earnings_drift.parameters import DriftParams


def test_defaults_match_the_spec() -> None:
    params = DriftParams()
    assert params.agent_enabled is True
    assert (params.max_positions, params.max_holding_sessions) == (8, 10)
    assert (params.min_trail_percent, params.default_trail_percent, params.max_trail_percent) == (3.0, 8.0, 15.0)
    assert (params.min_abnormal_pct, params.min_hold_ratio, params.min_close_location, params.min_rel_volume) == (0.03, 0.5, 0.5, 2.0)
    assert (params.min_price, params.min_dollar_volume) == (10.0, 20_000_000.0)
    assert params.live_bar_delay_seconds == 300.0
    assert params.tool_budget_per_item == 4
    assert (params.surprise_lookback_hours, params.surprise_window_hours, params.news_limit) == (2.0, 24.0, 50)
    assert not hasattr(params, "news_symbols_per_call")
    assert params.live_volume_share == 0.03


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_positions": 0}, "max_positions"),
        ({"max_holding_sessions": 0}, "max_holding_sessions"),
        ({"min_trail_percent": 9.0}, "trail"),
        ({"default_trail_percent": 16.0}, "trail"),
        ({"min_trail_percent": 0.0}, "trail"),
        ({"min_hold_ratio": 1.5}, "min_hold_ratio"),
        ({"min_close_location": -0.1}, "min_close_location"),
        ({"min_rel_volume": float("nan")}, "finite"),
        ({"volume_baseline_sessions": 1}, "volume_baseline_sessions"),
        ({"bars_lookback_sessions": 20}, "bars_lookback_sessions"),
        ({"news_limit": 0}, "news_limit"),
        ({"surprise_window_hours": 0.0}, "surprise_window_hours"),
        ({"surprise_window_hours": -1.0}, "surprise_window_hours"),
        ({"surprise_window_hours": float("inf")}, "finite"),
        ({"live_bar_delay_seconds": -1.0}, "live_bar_delay_seconds"),
        ({"live_volume_share": 0.0}, "live_volume_share"),
        ({"live_volume_share": 1.5}, "live_volume_share"),
        ({"live_volume_share": float("nan")}, "finite"),
        ({"sec_hollow_fraction": 1.5}, "sec_hollow_fraction"),
        ({"agent_temperature": 3.0}, "agent_temperature"),
    ],
)
def test_invalid_values_are_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DriftParams(**overrides)  # type: ignore[arg-type]


def test_zero_live_delay_and_no_temperature_are_allowed() -> None:
    DriftParams(live_bar_delay_seconds=0.0, agent_temperature=None)


def test_a_full_volume_share_is_allowed() -> None:
    assert DriftParams(live_volume_share=1.0).live_volume_share == 1.0

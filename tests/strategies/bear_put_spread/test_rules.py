"""Tests for the bear put spread entry, exit, strike, expiry and sizing rules."""

from datetime import date

import pytest

from trading_agent_framework.strategies.bear_put_spread.rules import (
    SpreadParameters,
    contracts_for,
    exit_reason,
    monthly_expiry,
    should_open,
    strikes,
    third_friday,
)


def test_third_fridays() -> None:
    assert third_friday(2026, 11) == date(2026, 11, 20)
    assert third_friday(2026, 12) == date(2026, 12, 18)
    assert third_friday(2027, 1) == date(2027, 1, 15)


def test_expiry_takes_the_nearer_of_two() -> None:
    assert monthly_expiry(date(2026, 11, 18), SpreadParameters()) == date(2026, 12, 18)  # 30 and 58 days


def test_expiry_skips_one_too_close() -> None:
    assert monthly_expiry(date(2026, 11, 19), SpreadParameters()) == date(2027, 1, 15)  # December is 29 days


def test_expiry_when_the_window_holds_none_takes_the_next_one() -> None:
    # Jan 15 is 29 days away and Feb 19 is 64: no third Friday falls in [30, 60]
    assert monthly_expiry(date(2026, 12, 17), SpreadParameters()) == date(2027, 2, 19)


def test_strikes_are_whole_dollars_five_apart() -> None:
    assert strikes(600.0, SpreadParameters()) == (588.0, 583.0)
    assert strikes(601.3, SpreadParameters()) == (589.0, 584.0)


def test_should_open() -> None:
    assert should_open(-1, False)
    assert not should_open(-1, True)
    assert not should_open(0, False)
    assert not should_open(1, False)


def test_exit_priority_time_then_target_then_regime() -> None:
    params, expiry = SpreadParameters(), date(2026, 12, 18)
    assert exit_reason(date(2026, 12, 11), expiry, 0, 4.5, 5.0, params) == "time"
    assert exit_reason(date(2026, 12, 1), expiry, 0, 4.0, 5.0, params) == "target"
    assert exit_reason(date(2026, 12, 1), expiry, 0, 3.9, 5.0, params) == "regime"
    assert exit_reason(date(2026, 12, 1), expiry, -1, 3.9, 5.0, params) is None


def test_sizing() -> None:
    params = SpreadParameters()
    assert contracts_for(1200.0, 1.35, 12.2, params) == 1
    assert contracts_for(1200.0, 2.05, 12.2, params) == 1  # 217.2 > 180 but <= 240: the 20% exception
    assert contracts_for(1200.0, 2.30, 12.2, params) == 0  # 242.2 > 240
    assert contracts_for(5000.0, 1.35, 12.2, params) == 5  # 750 / 147.2


def test_expiry_for_an_impossible_min_dte_raises() -> None:
    with pytest.raises(ValueError):
        monthly_expiry(date(2026, 11, 18), SpreadParameters(min_dte=400))

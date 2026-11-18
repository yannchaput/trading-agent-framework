"""Tests for the alternative bear put spread triggers (T1-T3)."""

import pytest

from trading_agent_framework.strategies.bear_put_spread.triggers import trigger_states


def test_t1_opens_after_the_close_below_the_50_day_and_closes_back_above() -> None:
    spy = [100.0] * 60 + [90.0] * 10 + [110.0] * 10
    states = trigger_states("t1", spy, [15.0] * len(spy))
    assert set(states[:61]) == {0}  # the first close under the average is index 60, acted on at index 61
    assert set(states[61:71]) == {-1}
    assert states[71] == 0


@pytest.mark.parametrize("name", ["t1", "t3"])
def test_a_high_vix_blocks_the_entry(name: str) -> None:
    spy = [100.0] * 30 + [100.0 - j for j in range(1, 31)] + [60.0] * 30
    assert set(trigger_states(name, spy, [30.0] * len(spy))) == {0}


def test_t2_needs_the_50_day_under_the_200_day_and_stays_until_they_cross_back() -> None:
    spy = [100.0] * 200 + [80.0] * 100 + [130.0] * 200
    states = trigger_states("t2", spy, [15.0] * len(spy))
    assert set(states[:201]) == {0}
    assert states[201] == -1
    assert states[299] == -1
    assert states[-1] == 0


def test_t3_opens_on_a_new_20_session_low_and_closes_above_the_10_day_average() -> None:
    spy = [100.0] * 30 + [100.0 - j for j in range(1, 11)] + [120.0] * 5
    states = trigger_states("t3", spy, [15.0] * len(spy))
    assert states[30] == 0  # a flat 100 is not a new low
    assert states[31] == -1
    assert states[40] == -1
    assert states[41] == 0


def test_an_unknown_trigger_raises() -> None:
    with pytest.raises(ValueError):
        trigger_states("t9", [100.0], [15.0])

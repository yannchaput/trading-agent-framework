from __future__ import annotations

import math

import pytest

from trading_agent_framework.utils.helpers import in_market_fall

CALM = 0.80  # VIX / VIX3M well under the 0.90 calm line
FLAT = 10.0  # a flat VIXY close


def fall(ratios: list[float], vixy: list[float] | None = None) -> bool:
    """`in_market_fall` on a series of VIX / VIX3M ratios (VIX3M fixed at 100 so VIX = ratio x 100)."""
    return in_market_fall([r * 100 for r in ratios], [100.0] * len(ratios), vixy if vixy is not None else [FLAT] * len(ratios))


def test_a_calm_market_is_not_a_fall() -> None:
    assert fall([CALM] * 10) is False


@pytest.mark.parametrize(("ratio", "expected"), [(0.95, True), (0.9499, False), (1.20, True)])
def test_the_ratio_entry_line_is_inclusive_at_095(ratio: float, expected: bool) -> None:
    assert fall([CALM] * 6 + [ratio]) is expected


@pytest.mark.parametrize(("last_vixy", "expected"), [(11.0, True), (10.99, False), (12.0, True)])
def test_a_ten_percent_vixy_rise_over_five_sessions_is_a_fall(last_vixy: float, expected: bool) -> None:
    vixy = [FLAT] * 5 + [last_vixy]  # the close 5 sessions earlier is the first FLAT
    assert fall([CALM] * 6, vixy) is expected


def test_the_vixy_rise_is_measured_over_exactly_five_sessions() -> None:
    assert fall([CALM] * 6, [10.0, 12.0, 12.0, 12.0, 12.0, 11.0]) is True  # +10% on the close 5 back (a 4-back read says -8%)
    assert fall([CALM] * 7, [10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 10.0]) is True  # +11% on the close 5 back (a 6-back read says 0%)


def test_a_missing_vixy_history_falls_back_to_the_ratio_alone() -> None:
    nan = math.nan
    assert fall([CALM] * 8, [nan] * 8) is False  # VIXY did not exist before 2011
    assert fall([CALM] * 7 + [1.0], [nan] * 8) is True
    assert fall([CALM, 1.0], [FLAT, FLAT]) is True  # under 6 sessions: no VIXY return, the ratio still counts


def test_a_ratio_of_exactly_090_is_not_a_calm_session() -> None:
    vix = [80.0] * 6 + [100.0] + [90.0] * 3  # 90 / 100 = 0.90: the calm line is exclusive
    assert in_market_fall(vix, [100.0] * 10, [FLAT] * 10) is True


def test_the_first_five_sessions_have_no_vixy_return_and_never_read_the_end_of_the_series() -> None:
    # session 4 against index -1 would read +22% (11 / 9): a negative index must not leak the last close into the start
    assert fall([CALM] * 6, [10.0, 10.0, 10.0, 10.0, 11.0, 9.0]) is False


def test_a_fall_lasts_until_three_calm_sessions() -> None:
    stressed = [CALM] * 6 + [1.0]
    assert fall(stressed + [0.92]) is True  # between the two lines: neither stressed nor calm
    assert fall(stressed + [CALM] * 2) is True
    assert fall(stressed + [CALM] * 3) is False


def test_the_trigger_session_does_not_count_as_a_calm_session() -> None:
    # VIXY alone triggers on the last session while the ratio is already calm: three more calm sessions are needed
    vixy = [FLAT] * 6 + [12.0]
    assert fall([CALM] * 7, vixy) is True
    assert fall([CALM] * 9, vixy + [12.0] * 2) is True
    assert fall([CALM] * 10, vixy + [12.0] * 3) is False


def test_a_new_stress_during_the_calm_count_restarts_it() -> None:
    assert fall([CALM] * 6 + [1.0, CALM, CALM, 1.0, CALM, CALM]) is True
    assert fall([CALM] * 6 + [1.0, CALM, CALM, 1.0, CALM, CALM, CALM]) is False


def test_a_half_calm_count_is_reset_by_an_in_between_session() -> None:
    assert fall([CALM] * 6 + [1.0, CALM, CALM, 0.92, CALM]) is True  # 0.92 is not calm: the count starts over


def test_only_the_latest_session_decides_after_an_old_episode_ended() -> None:
    assert fall([CALM] * 6 + [1.0] + [CALM] * 10) is False
    assert fall([CALM] * 6 + [1.0] + [CALM] * 10 + [1.0]) is True


def test_mismatched_series_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        in_market_fall([20.0, 21.0], [22.0], [10.0, 10.0])


def test_no_data_is_rejected_rather_than_read_as_calm() -> None:
    with pytest.raises(ValueError, match="at least one"):
        in_market_fall([], [], [])


@pytest.mark.parametrize("bad", [0.0, -1.0])
@pytest.mark.parametrize("series", ["vix", "vix3m", "vixy"])
def test_a_non_positive_close_is_rejected_with_a_clear_error(series: str, bad: float) -> None:
    closes = {"vix": [20.0] * 7, "vix3m": [25.0] * 7, "vixy": [10.0] * 7}
    closes[series][3] = bad
    with pytest.raises(ValueError, match="positive"):
        in_market_fall(closes["vix"], closes["vix3m"], closes["vixy"])


def test_nan_closes_are_missing_data_not_an_error() -> None:
    assert in_market_fall([math.nan] * 7, [25.0] * 7, [math.nan] * 7) is False

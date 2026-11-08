from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from trading_agent_framework.core.regime import (
    REGIME_LABELS,
    REGIME_LINE,
    RegimeParameters,
    RegimeReading,
    classify_regime,
    regime_by_session,
    regime_path,
)

SMALL = RegimeParameters(sma_window=2, high_window=3, vix_median_window=3, replay=6, neutral_hold=1)
CALM_VIX = 10.0
CALM_VIX3M = 12.0


def _run(spy: Sequence[float], vix: Sequence[float], vix3m: Sequence[float] | None = None, params: RegimeParameters = SMALL) -> list[RegimeReading]:
    """The readings from the first index with a full lookback, VIX3M calm (above VIX) unless given."""
    vix3m = vix3m if vix3m is not None else [CALM_VIX3M] * len(spy)
    return regime_path(spy, vix, vix3m, params, first=params.lookback - 1)


def test_defaults() -> None:
    params = RegimeParameters()
    assert (
        params.sma_window,
        params.high_window,
        params.vix_median_window,
        params.vix_spike_ratio,
        params.off_high,
        params.bear_score,
        params.bounce,
        params.neutral_hold,
        params.replay,
    ) == (50, 252, 252, 1.5, 0.05, 2, 0.08, 3, 252)
    assert params.lookback == 252
    assert params.min_bars == 503
    assert SMALL.min_bars == 8


def test_the_line_name_and_labels() -> None:
    assert REGIME_LINE == "Regime"
    assert REGIME_LABELS == {1: "bullish", 0: "neutral", -1: "bearish"}


@pytest.mark.parametrize(
    ("spy", "vix", "vix3m", "flag"),
    [
        ([100, 100, 100], [10, 10, 12], None, "inverted"),
        ([100, 100, 100], [10, 10, 15], [12, 12, 20], "vix_spike"),
        ([100, 100, 99], [10, 10, 10], None, "below_sma"),
    ],
)
def test_the_four_conditions_score_one_point_each(spy: Sequence[float], vix: Sequence[float], vix3m: Sequence[float] | None, flag: str) -> None:
    reading = _run(spy, vix, vix3m)[-1]
    assert getattr(reading, flag) is True
    assert reading.score == 1
    assert reading.regime == 0


def test_the_off_high_condition_scores_one_point() -> None:
    params = RegimeParameters(sma_window=1, high_window=3, vix_median_window=3, replay=6, neutral_hold=1)
    reading = _run([100, 100, 94], [10, 10, 10], params=params)[-1]
    assert reading.off_high is True
    assert not reading.below_sma
    assert reading.score == 1
    assert reading.regime == 0


def test_two_points_are_bearish_and_record_the_episode_low() -> None:
    readings = _run([100, 100, 100, 98, 97], [10, 10, 10, 13, 13])
    assert [r.regime for r in readings] == [1, -1, -1]
    assert readings[-1].score == 2
    assert readings[1].episode_low == 98
    assert readings[-1].episode_low == 97
    assert readings[0].episode_low is None


# Two bearish steps (episode low 97), then SPY closes at 105 (>= 1.08 x 97) with the score still 2.
_BOUNCE_SPY = [100, 100, 100, 98, 97, 105]
_BOUNCE_VIX = [10, 10, 10, 13, 13, 20]


def test_an_eight_percent_bounce_ends_the_episode_even_with_the_score_still_high() -> None:
    reading = _run(_BOUNCE_SPY, _BOUNCE_VIX)[-1]
    assert reading.score == 2
    assert reading.regime == 0
    assert reading.blocked is True
    assert reading.episode_low is None


def test_the_block_holds_until_a_new_low_or_a_zero_score() -> None:
    held = _run([*_BOUNCE_SPY, 104], [*_BOUNCE_VIX, 20])[-1]
    assert held.score == 2
    assert held.regime == 0
    assert held.blocked is True

    new_low = _run([*_BOUNCE_SPY, 104, 96], [*_BOUNCE_VIX, 20, 20])[-1]
    assert new_low.regime == -1
    assert new_low.blocked is False

    cleared = _run([*_BOUNCE_SPY, 105, 103], [*_BOUNCE_VIX, 10, 13])
    assert cleared[-2].score == 0
    assert cleared[-2].blocked is False
    assert cleared[-1].score == 2
    assert cleared[-1].regime == -1


def test_a_score_below_two_ends_the_episode() -> None:
    readings = _run([100, 100, 100, 98, 97, 97], [10, 10, 10, 13, 13, 13])
    assert readings[-2].regime == -1
    assert readings[-1].score == 1
    assert readings[-1].regime == 0
    assert readings[-1].blocked is False


def test_the_neutral_hold_delays_bullish_neutral_changes_only() -> None:
    params = RegimeParameters(sma_window=2, high_window=3, vix_median_window=3, replay=8, neutral_hold=3)
    # two calm readings, three score-1 steps (inverted), a score-2 step, then a score-0 step.
    spy = [100, 100, 100, 100, 100, 100, 100, 99, 99]
    vix = [10, 10, 10, 10, 13, 13, 13, 13, 10]
    readings = _run(spy, vix, params=params)
    assert [r.score for r in readings] == [0, 0, 1, 1, 1, 2, 0]
    assert [r.regime for r in readings] == [1, 1, 1, 1, 0, -1, 1]


def test_the_replay_ignores_closes_before_min_bars() -> None:
    spy = [100, 100, 100, 98, 97, 105, 104, 96]
    vix = [10, 10, 10, 13, 13, 20, 20, 20]
    vix3m = [CALM_VIX3M] * 8
    assert len(spy) == SMALL.min_bars
    expected = classify_regime(spy, vix, vix3m, SMALL)
    assert expected is not None
    assert classify_regime([1, 500, *spy], [90, 3, *vix], [2, 80, *vix3m], SMALL) == expected


def _sessions(count: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]


def test_regime_by_session_uses_closes_strictly_before_the_session() -> None:
    spy = [100, 100, 100, 98, 97, 105, 104, 96, 95, 94]
    vix = [10, 10, 10, 13, 13, 20, 20, 20, 20, 20]
    vix3m = [CALM_VIX3M] * 10
    sessions = _sessions(10)
    by_session = regime_by_session(sessions, spy, vix, vix3m, SMALL)
    assert list(by_session) == sessions[8:]
    assert by_session[sessions[9]] == classify_regime(spy[:9], vix[:9], vix3m[:9], SMALL)
    changed = regime_by_session(sessions, [*spy[:9], 1.0], vix, vix3m, SMALL)
    assert changed[sessions[9]] == by_session[sessions[9]]


def test_regime_by_session_rejects_misaligned_lengths() -> None:
    with pytest.raises(ValueError, match="aligned"):
        regime_by_session(_sessions(10), [100.0] * 9, [10.0] * 10, [20.0] * 10, SMALL)


def test_too_few_closes_give_no_reading() -> None:
    closes = [100.0] * (SMALL.min_bars - 1)
    assert classify_regime(closes, closes, closes, SMALL) is None


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("series", [0, 1, 2])
def test_mismatched_lengths_and_bad_closes_are_rejected(series: int, bad: float) -> None:
    data = [[100.0] * SMALL.min_bars for _ in range(3)]
    data[series][-2] = bad
    with pytest.raises(ValueError, match="finite and positive"):
        classify_regime(data[0], data[1], data[2], SMALL)


def test_unequal_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        regime_path([100.0] * 5, [10.0] * 4, [12.0] * 5, SMALL, first=2)
    with pytest.raises(ValueError, match="same length"):
        classify_regime([100.0] * 8, [10.0] * 8, [12.0] * 7, SMALL)


def test_first_must_leave_a_full_lookback() -> None:
    with pytest.raises(ValueError, match="first"):
        regime_path([100.0] * 5, [10.0] * 5, [12.0] * 5, SMALL, first=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sma_window": 0},
        {"high_window": 0},
        {"vix_median_window": 0},
        {"vix_spike_ratio": 1.0},
        {"off_high": 0.0},
        {"off_high": 1.0},
        {"bounce": 0.0},
        {"bounce": 1.0},
        {"bear_score": 0},
        {"bear_score": 5},
        {"neutral_hold": 0},
        {"replay": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RegimeParameters(**kwargs)

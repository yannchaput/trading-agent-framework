"""Tests for the regime scorecard: swing falls, reference labels and the scoring definitions."""

from datetime import date, timedelta

import pytest

from trading_agent_framework.core.regime_scorecard import reference_labels, score, zigzag


def _days(count: int) -> list[date]:
    return [date(2024, 1, 1) + timedelta(days=i) for i in range(count)]


def test_zigzag_finds_falls_from_recent_highs() -> None:
    closes = [100, 110, 99, 105, 95, 106, 115, 100, 130]
    days = _days(len(closes))

    episodes = zigzag(days, closes, 0.08)

    assert [(e.peak, e.trough) for e in episodes] == [(days[1], days[4]), (days[6], days[7])]
    assert episodes[0].depth == pytest.approx(95 / 110 - 1)  # the 105 bounce (+6%) did not end the swing
    assert episodes[1].depth == pytest.approx(100 / 115 - 1)


def test_zigzag_counts_an_unfinished_fall_and_handles_no_data() -> None:
    days = _days(4)

    episodes = zigzag(days, [100, 105, 98, 90], 0.08)

    assert [(e.peak, e.trough) for e in episodes] == [(days[1], days[3])]
    assert zigzag([], [], 0.08) == []


def test_reference_labels_mark_bear_neutral_and_bull() -> None:
    closes = [100, 110, 99, 105, 95, 106, 115, 100, 130]
    days = _days(len(closes))

    assert list(reference_labels(days, closes).values()) == [1, -1, -1, -1, -1, 1, -1, -1, 1]

    mild = [100, 105, 100, 106]  # a 4.8% fall: neutral, not bear
    assert list(reference_labels(_days(4), mild).values()) == [1, 0, 0, 1]


def _example() -> tuple[list[date], list[float], list[int]]:
    closes = [100, 102, 104, 100, 96, 94, 90, 88, 90, 93, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105]
    regimes = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 1, -1, -1, 1, 1, 1, 1, 1, 1]
    return _days(20), [float(c) for c in closes], regimes


def test_score_counts_switch_on_coverage_overstay_and_false_alarms() -> None:
    days, closes, regimes = _example()

    card = score(days, closes, dict(zip(days, regimes, strict=True)))

    (episode,) = card.episodes
    assert (episode.episode.peak, episode.episode.trough) == (days[2], days[7])
    assert episode.bear_on_fall == pytest.approx(90 / 104 - 1)  # first bearish session is index 6
    assert episode.coverage == pytest.approx(2 / 6)
    assert episode.overstay == 2
    assert card.recall == pytest.approx(2 / 6)
    assert card.false_bear_days == 2
    assert card.mean_on_fall == pytest.approx(90 / 104 - 1)
    assert card.median_overstay == 2
    assert card.changes == 5


def test_an_episode_never_read_bearish_uses_its_full_depth() -> None:
    days, closes, _ = _example()

    card = score(days, closes, dict.fromkeys(days, 1))

    (episode,) = card.episodes
    assert episode.bear_on_fall is None
    assert episode.coverage == 0
    assert card.mean_on_fall == pytest.approx(88 / 104 - 1)
    assert card.recall == 0
    assert card.false_bear_days == 0
    assert card.changes == 0


def test_score_rejects_misaligned_input() -> None:
    days, closes, regimes = _example()

    with pytest.raises(ValueError, match="same length"):
        score(days, closes[:-1], dict(zip(days, regimes, strict=True)))
    with pytest.raises(ValueError, match="no regime"):
        score(days, closes, {days[0]: 1})

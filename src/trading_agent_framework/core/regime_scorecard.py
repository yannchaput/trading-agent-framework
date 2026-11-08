"""Score the market-regime label against SPY's own swing falls (pure: lists and dates in, dataclasses out).

The reference is a zigzag of SPY closes: a fall of at least 8% from a swing high is a bear episode, one of at
least 4% a neutral one. A bear episode runs from its peak to its trough and ends on a rebound of the same size off
the low. The scorecard asks how early and how completely the label turns bearish on those episodes, how long it
stays bearish after the trough, and how many bearish days fall outside any episode.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class Episode:
    """A fall of SPY from a swing high to the swing low that followed.

    Attributes:
        peak: Session of the swing high.
        trough: Session of the swing low.
        depth: Trough close over peak close minus one; negative.
    """

    peak: date
    trough: date
    depth: float


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """How the regime read one bear episode.

    Attributes:
        episode: The reference episode.
        bear_on_fall: SPY's move from the peak at the first bearish session in `[peak, trough]`; None if none.
        coverage: Share of the `[peak, trough]` sessions read bearish, 0 to 1.
        overstay: Consecutive bearish sessions strictly after the trough.
    """

    episode: Episode
    bear_on_fall: float | None
    coverage: float
    overstay: int


@dataclass(frozen=True, slots=True)
class Scorecard:
    """The regime scored over a window of sessions.

    Attributes:
        episodes: One score per bear episode.
        recall: Share of reference bear sessions read bearish; 0 without any.
        false_bear_days: Bearish sessions outside every `[peak, trough + overstay]`.
        mean_on_fall: Mean `bear_on_fall`, an episode never read bearish counting its full depth; 0 without episodes.
        median_overstay: Median `overstay` over the episodes; 0 without episodes.
        changes: Sessions whose regime differs from the previous session's.
    """

    episodes: list[EpisodeScore]
    recall: float
    false_bear_days: int
    mean_on_fall: float
    median_overstay: float
    changes: int


def zigzag(sessions: Sequence[date], closes: Sequence[float], threshold: float) -> list[Episode]:
    """The falls of at least `threshold` from a swing high to a swing low.

    A swing high is armed from the first close. A close `threshold` below it starts a fall; the fall ends, and is
    recorded from the high to its lowest close, when a close rebounds `threshold` above that low, which also
    arms the next swing high. A fall still open at the end is recorded.

    Args:
        sessions: Session dates, oldest first.
        closes: Closes aligned with `sessions`.
        threshold: Fall (and rebound) size as a fraction, e.g. 0.08.

    Returns:
        The episodes, oldest first; empty for no sessions.
    """
    if not closes:
        return []
    episodes: list[Episode] = []
    high = low = 0
    falling = False
    for i, close in enumerate(closes):
        if not falling:
            if close > closes[high]:
                high = i
            elif close <= closes[high] * (1 - threshold):
                falling, low = True, i
        elif close < closes[low]:
            low = i
        elif close >= closes[low] * (1 + threshold):
            episodes.append(Episode(sessions[high], sessions[low], closes[low] / closes[high] - 1))
            falling, high = False, i
    if falling:
        episodes.append(Episode(sessions[high], sessions[low], closes[low] / closes[high] - 1))
    return episodes


def reference_labels(sessions: Sequence[date], closes: Sequence[float], bear: float = 0.08, neutral: float = 0.04) -> dict[date, int]:
    """The reference regime per session: -1 inside bear episodes, 0 inside other neutral ones, else 1.

    Args:
        sessions: Session dates, oldest first.
        closes: Closes aligned with `sessions`.
        bear: Fall size of a bear episode.
        neutral: Fall size of a neutral episode.

    Returns:
        One label per session.
    """
    labels = dict.fromkeys(sessions, 1)
    for threshold, label in ((neutral, 0), (bear, -1)):  # bear last: it overrides neutral
        for episode in zigzag(sessions, closes, threshold):
            for session in sessions:
                if episode.peak <= session <= episode.trough:
                    labels[session] = label
    return labels


def score(sessions: Sequence[date], closes: Sequence[float], regimes: Mapping[date, int]) -> Scorecard:
    """Score the regime against the 8% swing falls of `closes`.

    Args:
        sessions: Session dates of the window, oldest first.
        closes: SPY closes aligned with `sessions`.
        regimes: The regime (1, 0, -1) of every session in `sessions`.

    Returns:
        The scorecard.

    Raises:
        ValueError: If `sessions` and `closes` differ in length or a session has no regime.
    """
    if len(sessions) != len(closes):
        raise ValueError("sessions and closes must have the same length")
    if missing := [d for d in sessions if d not in regimes]:
        raise ValueError(f"no regime for {len(missing)} sessions, first {missing[0]}")
    position = {d: i for i, d in enumerate(sessions)}
    signal = [regimes[d] for d in sessions]

    scores: list[EpisodeScore] = []
    covered = [False] * len(sessions)
    for episode in zigzag(sessions, closes, 0.08):
        first, last = position[episode.peak], position[episode.trough]
        span = signal[first : last + 1]
        on = next((first + k for k, v in enumerate(span) if v == -1), None)
        overstay = 0
        while last + 1 + overstay < len(signal) and signal[last + 1 + overstay] == -1:
            overstay += 1
        for i in range(first, last + overstay + 1):
            covered[i] = True
        scores.append(
            EpisodeScore(
                episode,
                None if on is None else closes[on] / closes[first] - 1,
                span.count(-1) / len(span),
                overstay,
            )
        )

    labels = reference_labels(sessions, closes)
    reference_bear = [i for i, d in enumerate(sessions) if labels[d] == -1]
    recall = sum(signal[i] == -1 for i in reference_bear) / len(reference_bear) if reference_bear else 0.0
    on_fall = [s.episode.depth if s.bear_on_fall is None else s.bear_on_fall for s in scores]
    return Scorecard(
        episodes=scores,
        recall=recall,
        false_bear_days=sum(v == -1 and not covered[i] for i, v in enumerate(signal)),
        mean_on_fall=statistics.fmean(on_fall) if on_fall else 0.0,
        median_overstay=statistics.median(s.overstay for s in scores) if scores else 0.0,
        changes=sum(a != b for a, b in zip(signal, signal[1:], strict=False)),
    )

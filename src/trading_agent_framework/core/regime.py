"""Pure market-regime label from SPY, VIX and VIX3M daily closes (no I/O, no pandas, no `Strategy`).

Each session scores four conditions, one point each, on the closes up to and including that session:
VIX >= VIX3M (inverted term structure); VIX >= `vix_spike_ratio` x its median over `vix_median_window`
sessions; SPY below its `sma_window` SMA; SPY <= (1 - `off_high`) x its highest close over `high_window`.

The state machine reads the score: bearish at `bear_score` or more, neutral at 1, bullish at 0. Once
bearish it tracks SPY's lowest close of the episode. The episode ends when SPY closes `bounce` or more
above that low, or when the score falls below `bear_score`. After a bounce exit the bottom is presumed in:
it cannot turn bearish again until SPY closes below the episode low or the score reaches 0. A change
between bullish and neutral shows only after it has held `neutral_hold` sessions; changes into and out of
bearish are never delayed.

The state machine is stateful, but `classify_regime` is not: it replays the machine over the last
`min_bars` closes, so a restart or a backtest started on any date gives the same answer as an
uninterrupted run. `Strategy._refresh_regime` feeds it and logs the result; nothing trades on it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

REGIME_LINE = "Regime"  # the `add_line` name and pane; the dashboard's `charts.REGIME_PANE` must match
REGIME_LABELS = {1: "bullish", 0: "neutral", -1: "bearish"}


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    """Thresholds of the market-regime classifier.

    Attributes:
        sma_window: Sessions in SPY's simple moving average.
        high_window: Sessions in SPY's rolling high.
        vix_median_window: Sessions in the VIX median the spike is measured against.
        vix_spike_ratio: A VIX close at or above this multiple of its median is a spike; above 1.
        off_high: SPY at or below `(1 - off_high)` x its rolling high counts as off the high; in (0, 1).
        bear_score: Score from which the regime is bearish; 1 to 4.
        bounce: SPY rebound from the episode low that ends a bearish episode; in (0, 1).
        neutral_hold: Sessions a bullish/neutral change must hold before it shows.
        replay: Sessions the state machine is replayed over; the first one starts fresh.
    """

    sma_window: int = 50
    high_window: int = 252
    vix_median_window: int = 252
    vix_spike_ratio: float = 1.5
    off_high: float = 0.05
    bear_score: int = 2
    bounce: float = 0.08
    neutral_hold: int = 3
    replay: int = 252

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a window, ratio, fraction or count is out of range.
        """
        if min(self.sma_window, self.high_window, self.vix_median_window) < 1:
            raise ValueError("sma_window, high_window and vix_median_window must be at least 1")
        if self.vix_spike_ratio <= 1:
            raise ValueError(f"vix_spike_ratio must be above 1, got {self.vix_spike_ratio}")
        if not 0 < self.off_high < 1:
            raise ValueError(f"off_high must be strictly between 0 and 1, got {self.off_high}")
        if not 0 < self.bounce < 1:
            raise ValueError(f"bounce must be strictly between 0 and 1, got {self.bounce}")
        if not 1 <= self.bear_score <= 4:
            raise ValueError(f"bear_score must be between 1 and 4, got {self.bear_score}")
        if self.neutral_hold < 1:
            raise ValueError(f"neutral_hold must be at least 1, got {self.neutral_hold}")
        if self.replay < 1:
            raise ValueError(f"replay must be at least 1, got {self.replay}")

    @property
    def lookback(self) -> int:
        """Closes an indicator needs: the longest of the three windows."""
        return max(self.sma_window, self.high_window, self.vix_median_window)

    @property
    def min_bars(self) -> int:
        """Closes `classify_regime` needs (and the only ones it reads): the replay plus its lookback."""
        return self.lookback + self.replay - 1


_DEFAULTS = RegimeParameters()  # shared default of the functions below; the dataclass is frozen


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """The regime (1 bullish, 0 neutral, -1 bearish) and the figures behind it.

    Attributes:
        regime: 1 bullish, 0 neutral, -1 bearish.
        score: Conditions met, 0 to 4.
        spy: SPY close.
        sma: SPY's simple moving average.
        high: SPY's rolling high.
        vix: VIX close.
        vix3m: VIX3M close.
        vix_median: Median of the VIX over its window.
        inverted: VIX >= VIX3M.
        vix_spike: VIX >= `vix_spike_ratio` x its median.
        below_sma: SPY below its SMA.
        off_high: SPY at least `off_high` below its rolling high.
        episode_low: Lowest SPY close of the bearish episode; None unless bearish.
        blocked: The re-entry block after a bounce exit is active.
    """

    regime: int
    score: int
    spy: float
    sma: float
    high: float
    vix: float
    vix3m: float
    vix_median: float
    inverted: bool
    vix_spike: bool
    below_sma: bool
    off_high: bool
    episode_low: float | None
    blocked: bool


def regime_path(
    spy: Sequence[float],
    vix: Sequence[float],
    vix3m: Sequence[float],
    params: RegimeParameters = _DEFAULTS,
    *,
    first: int,
) -> list[RegimeReading]:
    """The readings after closes `first .. len - 1`, the state machine starting fresh at `first`.

    Closes before `first` only supply history: the indicators at index `i` include close `i`.

    Args:
        spy: SPY closes, oldest first.
        vix: VIX closes, aligned with `spy`.
        vix3m: VIX3M closes, aligned with `spy`.
        params: Thresholds.
        first: Index of the first reading; at least `params.lookback - 1`.

    Returns:
        One reading per index from `first`.

    Raises:
        ValueError: If the lengths differ, `first` leaves less than a lookback, or a close is not finite and positive.
    """
    if not len(spy) == len(vix) == len(vix3m):
        raise ValueError("spy, vix and vix3m must have the same length")
    if first < params.lookback - 1:
        raise ValueError(f"first must be at least lookback - 1 = {params.lookback - 1}, got {first}")
    series = [[float(close) for close in values] for values in (spy, vix, vix3m)]
    if any(not math.isfinite(close) or close <= 0 for values in series for close in values):
        raise ValueError("closes must be finite and positive")
    spy_closes, vix_closes, vix3m_closes = series

    readings: list[RegimeReading] = []
    bear = False
    low = 0.0
    blocked_at: float | None = None
    shown: int | None = None
    pending: int | None = None
    count = 0
    for i in range(first, len(spy_closes)):
        spy_i, vix_i, vix3m_i = spy_closes[i], vix_closes[i], vix3m_closes[i]
        sma = statistics.fmean(spy_closes[i + 1 - params.sma_window : i + 1])
        high = max(spy_closes[i + 1 - params.high_window : i + 1])
        median = statistics.median(vix_closes[i + 1 - params.vix_median_window : i + 1])
        inverted = vix_i >= vix3m_i
        spike = vix_i >= params.vix_spike_ratio * median
        below_sma = spy_i < sma
        off_high = spy_i <= (1 - params.off_high) * high
        score = inverted + spike + below_sma + off_high

        # block clears first, then the bear update, then entry (an exit step never re-enters)
        if blocked_at is not None and (spy_i < blocked_at or score == 0):
            blocked_at = None
        if bear:
            low = min(low, spy_i)
            if spy_i >= low * (1 + params.bounce):
                bear, blocked_at = False, low
            elif score < params.bear_score:
                bear = False
        elif score >= params.bear_score and blocked_at is None:
            bear, low = True, spy_i
        if bear:
            shown = -1
        else:
            raw = 0 if score >= 1 else 1
            if shown == -1 or shown is None:  # leaving bearish, or the replay's first step: no delay
                shown, pending, count = raw, raw, 0
            elif raw != shown:
                count = count + 1 if raw == pending else 1
                pending = raw
                if count >= params.neutral_hold:
                    shown, count = raw, 0
            else:
                count = 0
        readings.append(
            RegimeReading(
                regime=shown,
                score=score,
                spy=spy_i,
                sma=sma,
                high=high,
                vix=vix_i,
                vix3m=vix3m_i,
                vix_median=median,
                inverted=inverted,
                vix_spike=spike,
                below_sma=below_sma,
                off_high=off_high,
                episode_low=low if bear else None,
                blocked=blocked_at is not None,
            )
        )
    return readings


def classify_regime(
    spy: Sequence[float],
    vix: Sequence[float],
    vix3m: Sequence[float],
    params: RegimeParameters = _DEFAULTS,
) -> RegimeReading | None:
    """The regime at the last close, replayed over the last `params.min_bars` closes.

    Args:
        spy: SPY closes, oldest first.
        vix: VIX closes, aligned with `spy`.
        vix3m: VIX3M closes, aligned with `spy`.
        params: Thresholds.

    Returns:
        The last reading, or None with fewer than `params.min_bars` closes.

    Raises:
        ValueError: If the lengths differ or a close in the replayed window is not finite and positive.
    """
    if not len(spy) == len(vix) == len(vix3m):
        raise ValueError("spy, vix and vix3m must have the same length")
    if len(spy) < params.min_bars:
        return None
    tail = -params.min_bars
    readings = regime_path(spy[tail:], vix[tail:], vix3m[tail:], params, first=params.lookback - 1)
    return readings[-1]


def regime_by_session(
    sessions: Sequence[date],
    spy: Sequence[float],
    vix: Sequence[float],
    vix3m: Sequence[float],
    params: RegimeParameters = _DEFAULTS,
) -> dict[date, RegimeReading]:
    """The reading each session would have had, from the closes strictly before it.

    Args:
        sessions: Session dates, oldest first, aligned with the closes.
        spy: SPY closes.
        vix: VIX closes.
        vix3m: VIX3M closes.
        params: Thresholds.

    Returns:
        A reading per session from index `params.min_bars` on; none for the first `min_bars` sessions.

    Raises:
        ValueError: If the sessions and the closes do not have the same length.
    """
    if not len(sessions) == len(spy) == len(vix) == len(vix3m):
        raise ValueError("sessions and closes must be aligned (same length)")
    readings: dict[date, RegimeReading] = {}
    for j in range(params.min_bars, len(sessions)):
        reading = classify_regime(spy[:j], vix[:j], vix3m[:j], params)
        if reading is not None:
            readings[sessions[j]] = reading
    return readings

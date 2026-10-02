"""Pure market-regime estimate from a benchmark's daily closes (no I/O, no pandas, no `Strategy`).

Trend: the close and the fast SMA against the slow SMA. Volatility: when the latest realized volatility is
above its own high percentile over the lookback, a bullish read is capped at neutral. Volatility never
turns a read bearish. `Strategy._refresh_regime` feeds it and logs the result; nothing trades on it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

REGIME_LINE = "Regime"  # the `add_line` name and pane; the dashboard's `charts.REGIME_PANE` must match
REGIME_LABELS = {1: "bullish", 0: "neutral", -1: "bearish"}

_TRADING_DAYS_PER_YEAR = 252
_VOL_EPSILON = 1e-12  # float noise between equal volatilities must not read as a spike


@dataclass(frozen=True, slots=True)
class RegimeParameters:
    sma_fast: int = 50
    sma_slow: int = 200
    vol_window: int = 20
    vol_lookback: int = 252
    vol_percentile: float = 0.80

    def __post_init__(self) -> None:
        if not 1 <= self.sma_fast < self.sma_slow:
            raise ValueError(f"need 1 <= sma_fast < sma_slow, got {self.sma_fast} and {self.sma_slow}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be at least 2, got {self.vol_window}")
        if self.vol_lookback < 1:
            raise ValueError(f"vol_lookback must be at least 1, got {self.vol_lookback}")
        if not 0 < self.vol_percentile < 1:
            raise ValueError(f"vol_percentile must be strictly between 0 and 1, got {self.vol_percentile}")

    @property
    def min_bars(self) -> int:
        """Daily closes `classify_regime` needs (and the only ones it reads)."""
        return max(self.sma_slow, self.vol_window + self.vol_lookback) + 1


@dataclass(frozen=True, slots=True)
class RegimeReading:
    """The regime (1 bullish, 0 neutral, -1 bearish) and the figures behind it."""

    regime: int
    close: float
    sma_fast: float
    sma_slow: float
    vol: float
    vol_threshold: float
    stressed: bool


def classify_regime(closes: Sequence[float], params: RegimeParameters = RegimeParameters()) -> RegimeReading | None:  # noqa: B008 -- frozen dataclass default is safe
    """The regime at the last of `closes` (oldest first); None with fewer than `params.min_bars` closes."""
    if len(closes) < params.min_bars:
        return None
    window = [float(close) for close in closes[-params.min_bars :]]
    if any(not math.isfinite(close) or close <= 0 for close in window):
        raise ValueError("closes must be finite and positive")
    close = window[-1]
    sma_fast = statistics.fmean(window[-params.sma_fast :])
    sma_slow = statistics.fmean(window[-params.sma_slow :])
    if close > sma_slow and sma_fast > sma_slow:
        trend = 1
    elif close < sma_slow and sma_fast < sma_slow:
        trend = -1
    else:
        trend = 0
    vol, vol_threshold = _volatility(window, params)
    stressed = vol - vol_threshold > _VOL_EPSILON
    return RegimeReading(
        regime=0 if trend == 1 and stressed else trend,
        close=close,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        vol=vol,
        vol_threshold=vol_threshold,
        stressed=stressed,
    )


def _volatility(closes: Sequence[float], params: RegimeParameters) -> tuple[float, float]:
    """The latest annualized realized volatility, and its `vol_percentile` over the last `vol_lookback` values."""
    returns = [math.log(current / previous) for previous, current in pairwise(closes)]
    annualize = math.sqrt(_TRADING_DAYS_PER_YEAR)
    vols = [statistics.stdev(returns[end - params.vol_window : end]) * annualize for end in range(params.vol_window, len(returns) + 1)]
    return vols[-1], _percentile(vols[-params.vol_lookback :], params.vol_percentile)


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linear-interpolated percentile (numpy's default method), on plain floats."""
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

"""Approximates a trading-day lookback as a calendar-day buffer, for widening a
BacktestDataSource's fetch window before any trading calendar is available to
consult -- that calendar doesn't exist yet precisely because it's what the widened
window is used to build. 7/5 accounts for weekends; a fixed holiday buffer pads for
the ~9 NYSE holidays a year a weekday-only approximation would miss entirely.
Generous by design: oversizing the warm-up window costs a one-time wider fetch;
undersizing it silently starves a strategy's indicators of history near
`backtesting_start`.
"""

from __future__ import annotations

import math

DEFAULT_HOLIDAY_BUFFER_DAYS = 15


def warmup_calendar_days(trading_days: int, *, holiday_buffer_days: int = DEFAULT_HOLIDAY_BUFFER_DAYS) -> int:
    """Calendar days that very likely contain at least `trading_days` actual
    trading sessions. Returns 0 for `trading_days == 0` regardless of
    `holiday_buffer_days`, so a caller that doesn't ask for warm-up gets exactly
    the old, unwidened window. Raises `ValueError` for a negative `trading_days`.
    """
    if trading_days < 0:
        raise ValueError(f"trading_days must be >= 0, got {trading_days}")
    if trading_days == 0:
        return 0
    return math.ceil(trading_days * 7 / 5) + holiday_buffer_days

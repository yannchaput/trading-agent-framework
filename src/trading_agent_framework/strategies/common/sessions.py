"""Session helpers shared by the daily strategies (pure): today's partial bar, and the iteration's market time."""

from __future__ import annotations

import re
from datetime import date, time

import pandas as pd

from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import ConfigurationError

# A start at or after close - minutes_before_closing runs no iteration that session (the executor skips it), so
# the latest value assumes the strategy's default minutes_before_closing = 1.
_REBALANCE_EARLIEST = time(9, 30)
_REBALANCE_LATEST = time(15, 59)
_REBALANCE_TIME_PATTERN = re.compile(r"\d{2}:\d{2}")


def parse_rebalance_time(value: str) -> time:
    """`HH:MM` in market time (America/New_York), 09:30 <= t < 15:59: when the executor runs each session's iteration.

    On a half-day (13:00 close) a start at or after 12:59 runs no iteration that session, so such a Tuesday loses that
    week's rebalance (the default 12:00 is safe).
    """
    if not isinstance(value, str) or not _REBALANCE_TIME_PATTERN.fullmatch(value):
        raise ConfigurationError(f"rebalance_time must be HH:MM in market time, got {value!r}")
    try:
        parsed = time.fromisoformat(value)
    except ValueError:
        raise ConfigurationError(f"rebalance_time must be a valid HH:MM in market time, got {value!r}") from None
    if not _REBALANCE_EARLIEST <= parsed < _REBALANCE_LATEST:
        raise ConfigurationError(
            f"rebalance_time must be at or after {_REBALANCE_EARLIEST:%H:%M} and before {_REBALANCE_LATEST:%H:%M} "
            f"(market time), got {value!r}"
        )
    return parsed


def completed_bars(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """`df` without its bars dated `today` (market time): while a session is open its daily bar is partial.

    The strategy only iterates between its start time and the close, so a bar dated today is always partial
    there. In backtests the data gate already hides it and this is a no-op.
    """
    dates = [ts.date() for ts in pd.DatetimeIndex(df.index).tz_convert(MARKET_TZ)]
    return df.loc[[d != today for d in dates]]

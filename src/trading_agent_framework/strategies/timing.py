"""Pure timing helpers for the strategy executor: `sleeptime` parsing and tick maths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_agent_framework.errors import ConfigurationError

_PATTERN = re.compile(r"^\s*(\d+)\s*([smthd])\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "t": 60, "h": 3600}
_FORMAT_HELP = "an int (minutes) or '<n><S|M|T|H|D>', e.g. '30S', '5M', '2H', '1D'"


@dataclass(frozen=True, slots=True)
class SleepTime:
    """Either a fixed interval between iterations, or one iteration every N sessions."""

    interval: timedelta | None = None
    sessions: int | None = None

    def __post_init__(self) -> None:
        if (self.interval is None) == (self.sessions is None):
            raise ValueError("SleepTime needs exactly one of interval or sessions")


def parse_sleeptime(value: int | str) -> SleepTime:
    """Parse lumibot's `sleeptime` attribute."""
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise ConfigurationError(f"Invalid sleeptime {value!r}; expected {_FORMAT_HELP}")
    if isinstance(value, int):
        count, unit = value, "m"
    else:
        match = _PATTERN.match(value)
        if match is None:
            raise ConfigurationError(f"Invalid sleeptime {value!r}; expected {_FORMAT_HELP}")
        count, unit = int(match.group(1)), match.group(2).lower()
    if count <= 0:
        raise ConfigurationError(f"sleeptime must be positive, got {value!r}")
    if unit == "d":
        return SleepTime(sessions=count)
    return SleepTime(interval=timedelta(seconds=count * _UNIT_SECONDS[unit]))


def next_tick(last_tick: datetime, interval: timedelta, now: datetime) -> tuple[datetime, int]:
    """The first tick on `last_tick`'s grid strictly after `now`, and how many ticks were missed."""
    steps = max(1, (now - last_tick) // interval + 1)
    return last_tick + steps * interval, steps - 1

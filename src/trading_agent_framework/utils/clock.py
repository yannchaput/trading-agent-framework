"""Broker-agnostic market clock seam.

The strategy executor never calls `datetime.now()` or `time.sleep()` itself:
it asks a `MarketClock` what time it is, when the next session runs, and to
wait. Live clocks use the wall clock; a future backtesting clock will advance
simulated time instead, with no change to the executor.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class MarketSession:
    """One regular trading session (early closes included)."""

    open: datetime
    close: datetime

    def __post_init__(self) -> None:
        if self.open.tzinfo is None or self.close.tzinfo is None:
            raise ValueError("MarketSession open/close must be tz-aware")
        if self.close <= self.open:
            raise ValueError("MarketSession close must be after open")


class MarketClock(ABC):
    """Time source and session calendar for the strategy executor."""

    tz: ZoneInfo = MARKET_TZ

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def wait(self, seconds: float, wake: threading.Event) -> None:
        """Block for up to `seconds`, returning early as soon as `wake` is set."""
        if seconds > 0:
            wake.wait(seconds)

    @abstractmethod
    def next_session(self) -> MarketSession | None:
        """The first session whose close is after `now()`; None when there are no more."""

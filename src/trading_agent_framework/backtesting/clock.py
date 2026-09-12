"""`MarketClock` driving simulated time from injected sessions.

`wait()` jumps straight to the target time -- there is no real sleeping, so
`max_wait_slice` is infinite (Task 1's seam). Every successful jump calls
`on_advance(previous_now, new_now)` so `BacktestBroker` can process fills and
sample equity; `on_advance` is a plain settable attribute (not a constructor-only
argument) because the broker that needs to receive it is constructed *with* this
clock, creating a circular dependency the runner breaks by wiring it after both exist.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from trading_agent_framework.utils.clock import MarketClock, MarketSession


class BacktestClock(MarketClock):
    max_wait_slice: float = math.inf

    def __init__(
        self,
        start: datetime,
        sessions: Sequence[MarketSession],
        on_advance: Callable[[datetime, datetime], None] | None = None,
    ) -> None:
        self._now = start
        self._sessions = list(sessions)
        self.on_advance = on_advance

    def now(self) -> datetime:
        return self._now

    def wait(self, seconds: float, wake: threading.Event) -> None:
        if seconds <= 0 or wake.is_set():
            return
        previous = self._now
        self._now = previous + timedelta(seconds=seconds)
        if self.on_advance is not None:
            self.on_advance(previous, self._now)

    def next_session(self) -> MarketSession | None:
        return next((s for s in self._sessions if s.close > self._now), None)

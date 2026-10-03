"""Year-chunked minute bars for backtests longer than a year.

A minute frame per symbol over a multi-year window does not fit in memory: the vwap_pullback 5Y backtest
asked Alpaca for 150 symbols x 5 years of minutes in one request and was OOM-killed at 58 GB. Minute bars
are therefore fetched one year (`CHUNK`) at a time, each chunk from `OVERLAP` before its start so a
lookback early in the year never needs the previous one.
See docs/superpowers/specs/2026-10-03-year-chunked-minute-data-design.md.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

CHUNK = timedelta(days=365)
OVERLAP = timedelta(days=30)  # ~20 sessions; vwap's largest minute lookback is 11


def chunk_count(start: datetime, end: datetime) -> int:
    """Pure: how many chunks `[start, end]` splits into (1 when it is a year or less)."""
    if end - start <= CHUNK:
        return 1
    return math.ceil((end - start) / CHUNK)


def chunk_index(cutoff: datetime, start: datetime, end: datetime) -> int:
    """Pure: the chunk `cutoff` falls in, clamped to the window's chunks (a boundary opens the next one)."""
    return max(0, min((cutoff - start) // CHUNK, chunk_count(start, end) - 1))


def chunk_window(index: int, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    """Pure: the window chunk `index` is FETCHED over: `OVERLAP` before its start (never before `start`), to its end."""
    chunk_start = start + index * CHUNK
    chunk_end = end if index == chunk_count(start, end) - 1 else start + (index + 1) * CHUNK
    return max(start, chunk_start - OVERLAP), chunk_end


class YearChunkedData(BacktestDataSource):
    """Minute bars from a source built for the current chunk only; day bars and sessions from one source over
    the whole window. `inner` builds a source for a `(start, end)` window (e.g. the `AlpacaBacktestData` class).

    The current chunk follows the latest `cutoff` any `bars()` call has passed: a backtest clock only moves
    forward, and the scanner reads its daily bars before preloading minutes, so the position is on the right
    year by then. Crossing into the next chunk drops the previous source, and every frame it held.
    """

    def __init__(self, start: datetime, end: datetime, *, inner: Callable[[datetime, datetime], BacktestDataSource]) -> None:
        self._start = start
        self._end = end
        self._inner = inner
        self._whole = inner(start, end)
        self.name = self._whole.name
        self._position = start
        self._chunk = 0
        self._minute: BacktestDataSource | None = None
        self._lock = threading.Lock()  # a strategy may fan bars() out over threads (cross_momentum does)

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        """Day: the given window. Minute: the current chunk's window, whatever is asked (`load()` only pre-warms)."""
        if timestep != "minute":
            self._whole.load(assets, start, end, timestep)
            return
        with self._lock:
            source = self._minute_source(chunk_index(self._position, self._start, self._end))
            fetch_start, fetch_end = chunk_window(self._chunk, self._start, self._end)
        source.load(assets, fetch_start, fetch_end, timestep)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        with self._lock:
            self._position = max(self._position, cutoff)
            source = self._whole if timestep != "minute" else self._minute_for(cutoff)
        return source.bars(asset, cutoff, length, timestep)

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return self._whole.sessions(start, end)

    def _minute_for(self, cutoff: datetime) -> BacktestDataSource:
        """The minute source serving `cutoff` (lock held). An earlier chunk is served only within the current fetch window."""
        index = chunk_index(cutoff, self._start, self._end)
        if self._minute is not None and index < self._chunk:
            fetch_start, _ = chunk_window(self._chunk, self._start, self._end)
            if cutoff < fetch_start:
                raise BacktestDataError(
                    f"minute bars requested at {cutoff.isoformat()}, before the current chunk's window "
                    f"(from {fetch_start.isoformat()}): a backtest clock never goes back a year"
                )
            return self._minute
        return self._minute_source(index)

    def _minute_source(self, index: int) -> BacktestDataSource:
        """The source for chunk `index`, built (and the previous one dropped) when it is not the current one (lock held)."""
        if self._minute is None or index > self._chunk:
            self._minute = None  # drop the previous chunk before fetching the next one
            self._minute = self._inner(*chunk_window(index, self._start, self._end))
            self._chunk = index
        return self._minute

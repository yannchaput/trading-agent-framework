"""Year-chunked minute bars for backtests longer than a year.

A minute frame per symbol over a multi-year window does not fit in memory: the vwap_pullback 5Y backtest
asked Alpaca for 150 symbols x 5 years of minutes in one request and was OOM-killed at 58 GB. Minute bars
are therefore fetched one year (`CHUNK`) at a time, each chunk from `OVERLAP` before its start so a
lookback early in the year never needs the previous one.
See docs/superpowers/specs/2026-10-03-year-chunked-minute-data-design.md.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

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

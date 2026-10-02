"""Stock splits: restating a reported share count onto the basis of today's split-adjusted prices.

Bar prices are split-adjusted to today, but a share count in a filing is as reported on its date. A
2-for-1 split after the count would halve the computed market cap. `restate_shares` (pure) fixes the
count; `SplitHistory` is the I/O side and the only module that imports `yfinance` for splits (lazily).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple

from trading_agent_framework.utils.errors import FundamentalsError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "SplitHistory")

Split = tuple[date, float]


def restate_shares(shares: int, counted_on: date, splits: Sequence[Split]) -> Decimal:
    """`shares`, counted on `counted_on`, restated for every split dated after that day.

    Splits later than the caller's `as_of` are applied too, on purpose: the price this count gets
    multiplied by is already adjusted for them, so this undoes a price adjustment and leaks nothing.
    """
    restated = Decimal(shares)
    for split_date, ratio in splits:
        if split_date > counted_on:
            restated *= Decimal(str(ratio))
    return restated


def split_rows(history: Any) -> list[Split]:
    """The splits in a yfinance `Ticker.history(actions=True)` frame, oldest first.

    yfinance reports a failed download as an empty frame, which must not be read as "never split":
    an empty frame raises `FundamentalsError`. A row whose ratio is not a finite number above zero
    (a NaN, which `!= 0` would let through, or a bad negative) is not a split and is ignored.
    """
    if history.empty or "Stock Splits" not in history.columns:
        raise FundamentalsError("split lookup returned no history")
    column = history["Stock Splits"]
    return [(stamp.date(), float(ratio)) for stamp, ratio in column[column != 0].items() if _usable_ratio(ratio)]


def _usable_ratio(ratio: object) -> bool:
    return isinstance(ratio, int | float) and not isinstance(ratio, bool) and math.isfinite(ratio) and ratio > 0


def _fetch_from_yahoo(symbol: str) -> list[Split]:
    try:
        import yfinance as yf

        history = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
    except Exception as exc:  # yfinance raises many unrelated types; none may escape raw
        raise FundamentalsError(f"split lookup failed for {symbol}: {exc}") from exc
    try:
        return split_rows(history)
    except FundamentalsError as exc:
        raise FundamentalsError(f"{exc} for {symbol}") from exc


class _Entry(NamedTuple):
    fetched_at: datetime
    splits: list[Split]


def _decode_entry(raw: object) -> _Entry | None:
    """A cache entry as an `_Entry`, or `None` when it is malformed in any way (a cache miss)."""
    if not isinstance(raw, dict) or not isinstance(raw.get("fetched_at"), str) or not isinstance(raw.get("splits"), list):
        return None
    try:
        fetched_at = datetime.fromisoformat(raw["fetched_at"])
        splits = [_decode_split(row) for row in raw["splits"]]
    except (ValueError, TypeError):
        return None
    if fetched_at.tzinfo is None or None in splits:
        return None
    return _Entry(fetched_at, [split for split in splits if split is not None])


def _decode_split(row: object) -> Split | None:
    if not isinstance(row, list | tuple) or len(row) != 2 or not isinstance(row[0], str) or not _usable_ratio(row[1]):
        return None
    return date.fromisoformat(row[0]), float(row[1])


class SplitHistory:
    """Split history per symbol, fetched lazily and cached in one JSON file.

    Unlike the annual SEC figures, split history is not point-in-time data: it must match the basis
    of today's split-adjusted prices, which is a wall-clock fact. An entry is therefore stale when it
    was fetched more than `max_age_days` before `wall_clock()` (a timezone-aware callable, UTC by
    default), whatever date a backtest is simulating.

    The cache file is validated once, on load: a malformed entry is a cache miss, never an exception.
    """

    def __init__(
        self,
        cache_file: Path,
        *,
        fetch: Callable[[str], list[Split]] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._cache_file = cache_file
        self._fetch = fetch or _fetch_from_yahoo
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._entries: dict[str, _Entry] | None = None

    def splits(self, symbol: str, *, max_age_days: int) -> list[Split]:
        """`symbol`'s splits, oldest first (empty when it never split).

        Raises `FundamentalsError` when the lookup fails and there is no cached copy. A stale copy
        that cannot be refreshed is used, with a warning.
        """
        entries = self._load()
        entry = entries.get(symbol)
        now = self._wall_clock()
        if entry is not None and entry.fetched_at >= now - timedelta(days=max_age_days):
            return list(entry.splits)
        try:
            fetched = sorted(self._fetch(symbol))
        except FundamentalsError as exc:
            if entry is None:
                raise
            logger.log_warning(f"split history for {symbol} could not be refreshed, using the copy fetched {entry.fetched_at.isoformat()}: {exc}")
            return list(entry.splits)
        entries[symbol] = _Entry(now, fetched)
        self._save(entries)
        return fetched

    def _load(self) -> dict[str, _Entry]:
        if self._entries is None:
            try:
                loaded = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = {}  # no file yet, or one truncated by an interrupted write
            decoded = {symbol: _decode_entry(raw) for symbol, raw in loaded.items()} if isinstance(loaded, dict) else {}
            self._entries = {symbol: entry for symbol, entry in decoded.items() if entry is not None}
        return self._entries

    def _save(self, entries: dict[str, _Entry]) -> None:
        """Try to persist `entries` (already held in memory); a failed write only costs a refetch next process."""
        payload = {
            symbol: {"fetched_at": entry.fetched_at.isoformat(), "splits": [[split_date.isoformat(), ratio] for split_date, ratio in entry.splits]}
            for symbol, entry in entries.items()
        }
        temporary: str | None = None
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            # A unique name per write: parallel runs sharing the cache must not share a temp file.
            descriptor, temporary = tempfile.mkstemp(dir=self._cache_file.parent, prefix="splits.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temporary, self._cache_file)
        except OSError as exc:
            logger.log_warning(f"split history could not be saved to {self._cache_file}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

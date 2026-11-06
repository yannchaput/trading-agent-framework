"""The point-in-time stock universe: for each month, the top-N US stocks ranked only on data dated before that
month's first session.

cross_momentum and bull_bear backtest on a universe file built today, which holds today's winners and so leaks
the future into every past date. A `UniverseHistory` instead carries one list per month, each ranked on the
dollar volume stocks actually traded at (`traded_close * traded_volume`, split-unadjusted, from
`parse_yahoo_frame`) over the `lookback` sessions before the month began. `monthly_lists` is the pure ranking;
the history file is its saved result. No I/O beyond `save` and `load`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import FatalStrategyError, UniverseHistoryError

if TYPE_CHECKING:
    import pandas as pd


# A symbol's last bar before a month must fall within this many sessions before the month's first session, else it
# is no longer trading (acquired, delisted, a long halt or data hole) and its average is stale. 10 tolerates a Yahoo
# gap or a short halt.
_MAX_STALE_SESSIONS = 10


def _month_key(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def _month_keys(start: date, end: date) -> list[str]:
    """Every month key from `start`'s month to `end`'s month, inclusive."""
    keys: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        keys.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return keys


@dataclass(frozen=True)
class UniverseHistory:
    """The monthly universe lists and the parameters they were built with. `months` maps "YYYY-MM" to that month's
    symbols, best first.

    `pool_date`, `pool_size`, `unanswered` and `bars_absent` describe the candidate pool of the LATEST build, which
    also ranked every month of an earlier build that this one extended: an extension keeps the older months as they
    were saved. `pool_size` is the number of pool symbols; `unanswered` the candidates whose common-stock check got
    no answer from Yahoo (left out of the pool); `bars_absent` the pool symbols Yahoo had no bars for, even after the
    retry pass (left out of every month). Files saved before these fields existed load with their defaults (0, empty).
    """

    built: date
    pool_date: date
    top_n: int
    min_price: float
    min_dollar_volume: float
    lookback: int
    months: dict[str, tuple[str, ...]]
    pool_size: int = 0
    unanswered: tuple[str, ...] = ()
    bars_absent: tuple[str, ...] = ()

    def on(self, day: date) -> list[str]:
        """The list of `day`'s month. Never falls back to another month: a missing month is an error."""
        key = _month_key(day)
        try:
            return list(self.months[key])
        except KeyError:
            raise UniverseHistoryError(f"the universe history has no list for {key}") from None

    def covers(self, start: date, end: date) -> bool:
        """True when every month from `start`'s to `end`'s is present."""
        return all(key in self.months for key in _month_keys(start, end))

    def union(self, start: date, end: date) -> list[str]:
        """Every symbol listed in any month from `start`'s to `end`'s, sorted."""
        symbols: set[str] = set()
        for key in _month_keys(start, end):
            if key not in self.months:
                raise UniverseHistoryError(f"the universe history has no list for {key}")
            symbols.update(self.months[key])
        return sorted(symbols)

    def to_json(self) -> dict[str, Any]:
        return {
            "built": self.built.isoformat(),
            "pool_date": self.pool_date.isoformat(),
            "top_n": self.top_n,
            "min_price": self.min_price,
            "min_dollar_volume": self.min_dollar_volume,
            "lookback": self.lookback,
            "pool_size": self.pool_size,
            "unanswered": list(self.unanswered),
            "bars_absent": list(self.bars_absent),
            "months": {key: list(symbols) for key, symbols in self.months.items()},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> UniverseHistory:
        return cls(
            built=date.fromisoformat(data["built"]),
            pool_date=date.fromisoformat(data["pool_date"]),
            top_n=int(data["top_n"]),
            min_price=float(data["min_price"]),
            min_dollar_volume=float(data["min_dollar_volume"]),
            lookback=int(data["lookback"]),
            months={key: tuple(symbols) for key, symbols in data["months"].items()},
            pool_size=int(data.get("pool_size", 0)),
            unanswered=tuple(data.get("unanswered", ())),
            bars_absent=tuple(data.get("bars_absent", ())),
        )

    def save(self, path: Path) -> None:
        """Write to a temp file beside `path` and rename it over, so `path` is never half-written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(self.to_json(), indent=2) + "\n")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> UniverseHistory | None:
        """The history saved at `path`, or None when there is no such file."""
        if not path.exists():
            return None
        return cls.from_json(json.loads(path.read_text()))


def market_day(moment: datetime) -> date:
    """The market-time date of a backtest bound (bounds are timezone-aware by convention; a naive one is taken as is)."""
    return moment.astimezone(MARKET_TZ).date() if moment.tzinfo is not None else moment.date()


def window_symbols(history: UniverseHistory, start: datetime, end: datetime, *, exclude: Collection[str]) -> list[str]:
    """Every symbol the history lists in any month of the backtest window, sorted, minus `exclude` (the sleeve symbols)."""
    return [symbol for symbol in history.union(market_day(start), market_day(end)) if symbol not in exclude]


def month_universe(history: UniverseHistory, day: date, *, exclude: Collection[str]) -> list[str]:
    """The history's list for `day`'s month, minus `exclude`, in rank order.

    A month the history lacks raises `FatalStrategyError`: the executor would swallow anything else as a failed tick and
    drop the review without a trace in the results."""
    try:
        month_list = history.on(day)
    except UniverseHistoryError as exc:
        raise FatalStrategyError(str(exc)) from exc
    return [symbol for symbol in month_list if symbol not in exclude]


def monthly_lists(
    frames: Mapping[str, pd.DataFrame],
    sessions: Sequence[date],
    months: Sequence[str],
    *,
    min_price: float,
    min_dollar_volume: float,
    top_n: int,
    lookback: int = 63,
) -> dict[str, tuple[str, ...]]:
    """For each "YYYY-MM" in `months`, the top-`top_n` symbols ranked on data dated before that month's first session.

    `frames` are `parse_yahoo_frame` outputs (a row belongs to the session of its index date). A symbol is eligible
    for a month when it has at least `lookback` rows before the month's first session and its last such row is dated
    within the `_MAX_STALE_SESSIONS` sessions immediately before that session (a stock that stopped trading is not
    ranked on its old volume); its average is the mean of
    `traded_close * traded_volume` over its last `lookback` such rows and its price the last such `traded_close`.
    Symbols under `min_price` or `min_dollar_volume` (or with a non-finite average or price) are dropped; the rest
    are sorted by average, descending, ties by symbol, and cut to `top_n`.

    Raises `UniverseHistoryError` for a month with no session, or with fewer than `lookback` sessions before it.
    """
    first_sessions: dict[str, date] = {}
    for day in sorted(sessions):
        first_sessions.setdefault(_month_key(day), day)

    # Per symbol, once: the row dates (naive wall-clock dates of the session stamps), dollar volume and traded close.
    prepared: dict[str, tuple[Any, Any, Any]] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        frame = frame.sort_index()
        dates = frame.index.tz_localize(None).values.astype("datetime64[D]")
        closes = frame["traded_close"].to_numpy(dtype=float)
        dollars = closes * frame["traded_volume"].to_numpy(dtype=float)
        prepared[symbol] = (dates, closes, dollars)

    result: dict[str, tuple[str, ...]] = {}
    for month in months:
        first = first_sessions.get(month)
        if first is None:
            raise UniverseHistoryError(f"no session in {month}: cannot rank its universe")
        prior_sessions = sum(1 for day in sessions if day < first)
        if prior_sessions < lookback:
            raise UniverseHistoryError(f"{month} has {prior_sessions} sessions before its first ({first}), fewer than the lookback of {lookback}")
        cutoff = np.datetime64(first, "D")
        # The oldest date a symbol's last bar may carry: the 10th-last session before the month's first (the first
        # prior session when fewer than 10 exist, which only happens with a lookback under 10).
        prior = sorted(day for day in sessions if day < first)
        oldest_last_bar = np.datetime64(prior[max(0, len(prior) - _MAX_STALE_SESSIONS)], "D")
        ranked: list[tuple[float, str]] = []
        for symbol, (dates, closes, dollars) in prepared.items():
            count = int(np.searchsorted(dates, cutoff, side="left"))  # rows dated before the first session
            if count < lookback or dates[count - 1] < oldest_last_bar:
                continue
            average = float(np.mean(dollars[count - lookback : count]))
            price = float(closes[count - 1])
            if not (math.isfinite(average) and math.isfinite(price)):
                continue
            if price < min_price or average < min_dollar_volume:
                continue
            ranked.append((average, symbol))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        result[month] = tuple(symbol for _, symbol in ranked[:top_n])
    return result

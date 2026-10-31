"""Where a review's daily bars come from: one Yahoo batch in paper/live, the backtest gate otherwise.

Paper/live read Yahoo for the same reason as cross_momentum: Alpaca's IEX bars carry ~5% of consolidated volume, so
the $20M dollar-volume filter would empty the ranking. Both sources hand the pipeline completed sessions only (the
bar dated today is partial while the session is open), at most `HISTORY_BARS` of them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.common.sessions import completed_bars
from trading_agent_framework.utils.errors import BrokerError, YahooDataError

HISTORY_BARS = 300  # completed sessions scored: cross_momentum's own window


class DataUnavailable(Exception):
    """No usable daily bars for this review; the message says why."""


@dataclass(frozen=True, slots=True)
class DailySeries:
    closes: list[float]  # oldest first
    volumes: list[float]


class DailyBars(Protocol):
    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]: ...


class BarsSource(Protocol):
    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]: ...


def _series(bars: Bars, today: date) -> DailySeries | None:
    frame = completed_bars(bars.df, today).tail(HISTORY_BARS)
    if frame.empty:
        return None
    return DailySeries(closes=[float(value) for value in frame["close"]], volumes=[float(value) for value in frame["volume"]])


class YahooBars:
    """Paper/live: one batched Yahoo download per review, fetched again after each delay while it is unusable.

    Unusable when the download fails, covers less than `min_coverage` of the universe, or misses a held universe
    stock (it would be forced out as unranked on missing data). After the last attempt: `DataUnavailable`.
    """

    def __init__(
        self,
        source: BarsSource,
        *,
        today: Callable[[], date],
        sleep: Callable[[float], None],
        warn: Callable[[str], None],
        retry_delays: Sequence[float],
        min_coverage: float,
    ) -> None:
        self._source = source
        self._today = today
        self._sleep = sleep
        self._warn = warn
        self._retry_delays = tuple(retry_delays)
        self._min_coverage = min_coverage

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        problem = "no attempt made"
        for delay in (*self._retry_delays, None):
            bars, found = self._attempt(universe, held)
            if found is None:
                today = self._today()
                in_universe = set(universe)
                result: dict[str, DailySeries] = {}
                for symbol, symbol_bars in bars.items():
                    series = _series(symbol_bars, today) if symbol in in_universe else None
                    if series is not None:
                        result[symbol] = series
                return result
            problem = found
            if delay is not None:
                self._warn(f"{problem}: retry in {delay:.0f}s")
                self._sleep(delay)
        raise DataUnavailable(problem)

    def _attempt(self, universe: Sequence[str], held: Collection[str]) -> tuple[dict[str, Bars], str | None]:
        try:
            bars = self._source.bars(list(universe), self._today())
        except YahooDataError as exc:
            return {}, f"Yahoo lookup failed ({exc})"
        covered = sum(1 for symbol in universe if symbol in bars)
        if covered < len(universe) * self._min_coverage:
            return {}, f"Yahoo covers only {covered}/{len(universe)} symbols"
        missing_held = sorted((set(held) & set(universe)) - bars.keys())
        if missing_held:
            return {}, f"Yahoo has no bars for held {', '.join(missing_held)}"
        return bars, None


class GateBars:
    """Backtests: each symbol through `Strategy.get_historical_prices`, the no-look-ahead gate.

    A `BrokerError` for one symbol skips it (as cross_momentum does); a `BacktestError` propagates and abandons the
    review.
    """

    def __init__(self, fetch: Callable[[str, int], Bars | None], *, today: Callable[[], date], warn: Callable[[str], None]) -> None:
        self._fetch = fetch
        self._today = today
        self._warn = warn

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        today = self._today()
        result: dict[str, DailySeries] = {}
        for symbol in universe:
            try:
                bars = self._fetch(symbol, HISTORY_BARS + 1)  # one extra: today's bar, if any, is dropped
            except BrokerError as exc:
                self._warn(f"Skipping {symbol}: failed to fetch bars ({exc})")
                continue
            if bars is None or bars.empty:
                continue
            series = _series(bars, today)
            if series is not None:
                result[symbol] = series
        return result

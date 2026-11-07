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

from trading_agent_framework.backtesting.data.yahoo import traded_series
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.strategies.common.sessions import completed_bars
from trading_agent_framework.strategies.common.yahoo_daily_bars import BarsSource, UnusableBatch, fetch_usable_batch
from trading_agent_framework.utils.errors import BrokerError

HISTORY_BARS = 300  # completed sessions scored: cross_momentum's own window


class DataUnavailable(Exception):
    """No usable daily bars for this review; the message says why."""


@dataclass(frozen=True, slots=True)
class DailySeries:
    """A stock's daily history.

    Attributes:
        closes: Adjusted closes, oldest first.
        volumes: Adjusted volumes, same rows.
        traded_closes: The close as traded, same rows as `closes`; None when the source carries none.
        traded_volumes: The volume as traded, same rows; None when the source carries none.
    """

    closes: list[float]  # oldest first
    volumes: list[float]
    traded_closes: list[float] | None = None  # the close and volume as traded, same rows as `closes`; None when the source carries none
    traded_volumes: list[float] | None = None


class DailyBars(Protocol):
    """Source of the daily history of a universe."""

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        """Load the daily history of the universe.

        Args:
            universe: Tickers to load.
            held: Held tickers, which a batch source must be able to serve.

        Returns:
            The history per ticker; a ticker without a usable history is absent.
        """
        ...


def _series(bars: Bars, today: date) -> DailySeries | None:
    frame = completed_bars(bars.df, today).tail(HISTORY_BARS)
    if frame.empty:
        return None
    traded = traded_series(frame)  # after the cut: the same rows as closes
    return DailySeries(
        closes=[float(value) for value in frame["close"]],
        volumes=[float(value) for value in frame["volume"]],
        traded_closes=traded[0] if traded else None,
        traded_volumes=traded[1] if traded else None,
    )


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
        """Create a source over one Yahoo batch.

        Args:
            source: Batched daily-bars source.
            today: Returns the current session date.
            sleep: Waits between attempts at an unusable batch.
            warn: Logs a warning.
            retry_delays: Seconds before each new attempt.
            min_coverage: Share of the universe the batch must cover.
        """
        self._source = source
        self._today = today
        self._sleep = sleep
        self._warn = warn
        self._retry_delays = tuple(retry_delays)
        self._min_coverage = min_coverage

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        """Load the daily history of the universe.

        Args:
            universe: Tickers to load.
            held: Held tickers, which the batch must include.

        Returns:
            The history per ticker; a ticker without a usable history is absent.

        Raises:
            DataUnavailable: If the batch stays unusable after every retry.
        """
        try:
            bars = fetch_usable_batch(
                self._source,
                universe,
                today=self._today,
                held=lambda: held,
                min_coverage=self._min_coverage,
                retry_delays=self._retry_delays,
                sleep=self._sleep,
                on_retry=lambda problem, delay: self._warn(f"{problem}: retry in {delay:.0f}s"),
            )
        except UnusableBatch as exc:
            raise DataUnavailable(str(exc)) from exc
        today = self._today()
        in_universe = set(universe)
        result: dict[str, DailySeries] = {}
        for symbol, symbol_bars in bars.items():
            series = _series(symbol_bars, today) if symbol in in_universe else None
            if series is not None:
                result[symbol] = series
        return result


class GateBars:
    """Backtests: each symbol through `Strategy.get_historical_prices`, the no-look-ahead gate.

    A `BrokerError` for one symbol skips it (as cross_momentum does); a `BacktestError` propagates and abandons the
    review.
    """

    def __init__(self, fetch: Callable[[str, int], Bars | None], *, today: Callable[[], date], warn: Callable[[str], None]) -> None:
        """Create a source that fetches symbol by symbol.

        Args:
            fetch: Returns the bars of a symbol for a number of sessions, or None.
            today: Returns the current session date.
            warn: Logs a warning.
        """
        self._fetch = fetch
        self._today = today
        self._warn = warn

    def load(self, universe: Sequence[str], held: Collection[str]) -> dict[str, DailySeries]:
        """Load the daily history of the universe, one symbol at a time.

        Args:
            universe: Tickers to load.
            held: Held tickers (unused: every ticker is fetched on its own).

        Returns:
            The history per ticker; a ticker that fails or has no bars is skipped with a warning.
        """
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

"""SPY, VIX, VIX3M and VIXY daily closes from Yahoo, aligned on SPY's sessions, for the market regime.

One batched download feeds all four series. VIXY is optional: when it is missing or unreadable its column is `nan`
and the regime is computed as before. The download date is never a cutoff: Yahoo may return a bar dated
today (partial while the session is open), and only `RegimeSeries.before` decides which rows a caller sees, always
those dated strictly before the session being decided.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Protocol

from trading_agent_framework.backtesting.data.yahoo import extract_ticker_frame, parse_yahoo_frame, yfinance_download
from trading_agent_framework.utils.errors import RegimeDataError

if TYPE_CHECKING:
    import pandas as pd

REQUIRED = ("SPY", "^VIX", "^VIX3M")
SYMBOLS = (*REQUIRED, "VIXY")
_CALENDAR_DAYS_PER_BAR = 1.5  # sessions to calendar days, with slack for holidays
_CALENDAR_SLACK = 10

DownloadFn = Callable[..., "pd.DataFrame"]


def history_days(length: int) -> int:
    """Calendar days of history that cover `length` sessions, with slack for holidays.

    Args:
        length: Number of sessions wanted.

    Returns:
        The number of calendar days to download before the first session needed.
    """
    return math.ceil(length * _CALENDAR_DAYS_PER_BAR) + _CALENDAR_SLACK


@dataclass(frozen=True)
class RegimeSeries:
    """Aligned daily closes, oldest first.

    Attributes:
        sessions: SPY session dates, strictly increasing.
        spy: SPY adjusted closes.
        vix: VIX closes; `nan` where a gap exceeded the forward-fill limit.
        vix3m: VIX3M closes; `nan` where a gap exceeded the forward-fill limit.
        vixy: VIXY adjusted closes, `nan` where missing; empty when there is no VIXY data at all.
    """

    sessions: tuple[date, ...]
    spy: tuple[float, ...]
    vix: tuple[float, ...]
    vix3m: tuple[float, ...]
    vixy: tuple[float, ...] = ()

    def __len__(self) -> int:
        """Number of sessions."""
        return len(self.sessions)

    def before(self, session: date, length: int) -> RegimeSeries:
        """The last `length` rows dated strictly before `session`.

        Args:
            session: The session being decided; its own row and later ones are excluded.
            length: Maximum number of rows to keep.

        Returns:
            The slice (shorter when fewer rows exist).

        Raises:
            RegimeDataError: If a `nan` remains in the returned rows.
        """
        end = sum(1 for d in self.sessions if d < session)
        start = max(0, end - length)
        out = RegimeSeries(self.sessions[start:end], self.spy[start:end], self.vix[start:end], self.vix3m[start:end], self.vixy[start:end])
        for name, full, values in (("spy", self.spy, out.spy), ("vix", self.vix, out.vix), ("vix3m", self.vix3m, out.vix3m)):
            if not any(math.isnan(v) for v in values):
                continue
            first_valid = next((i for i, v in enumerate(full) if not math.isnan(v)), len(full))
            if start < first_valid:
                raise RegimeDataError(f"{name} has not enough aligned history before {session}: its closes start {self.sessions[first_valid] if first_valid < len(full) else 'never'}")
            raise RegimeDataError(f"{name} has a gap longer than the forward-fill limit before {session}")
        return out


class RegimeSource(Protocol):
    """Where the regime's closes come from."""

    def closes_before(self, session: date, length: int) -> RegimeSeries:
        """The last `length` aligned rows dated strictly before `session`."""
        ...


class YahooRegimeData:
    """`RegimeSource` backed by one batched Yahoo download per day."""

    def __init__(
        self,
        *,
        download: DownloadFn | None = None,
        today: Callable[[], date] = date.today,
        max_fill: int = 5,
    ) -> None:
        """Create a source.

        Args:
            download: Replacement for `yfinance.download` (injected in tests).
            today: The current date; a new date invalidates the cache.
            max_fill: Longest run of sessions a VIX or VIX3M gap is forward-filled over.
        """
        self._download = download
        self._today = today
        self._max_fill = max_fill
        self._cache: tuple[date, date, RegimeSeries] | None = None

    def series(self, start: date) -> RegimeSeries:
        """Aligned closes from `start` through the download date.

        Args:
            start: First date wanted.

        Returns:
            Rows on SPY's session dates; VIX/VIX3M forward-filled up to `max_fill` sessions.

        Raises:
            RegimeDataError: If the download fails or returns nothing for SPY.
        """
        today = self._today()
        cached = self._cache
        if cached is None or cached[0] != today or start < cached[1]:
            cached = (today, start, self._fetch(start, today))
            self._cache = cached
        full = cached[2]
        keep = [i for i, d in enumerate(full.sessions) if d >= start]
        return RegimeSeries(
            tuple(full.sessions[i] for i in keep),
            tuple(full.spy[i] for i in keep),
            tuple(full.vix[i] for i in keep),
            tuple(full.vix3m[i] for i in keep),
            tuple(full.vixy[i] for i in keep) if full.vixy else (),
        )

    def closes_before(self, session: date, length: int) -> RegimeSeries:
        """The last `length` aligned rows dated strictly before `session`.

        Raises:
            RegimeDataError: If the download fails or a gap remains in the rows.
        """
        start = session - timedelta(days=history_days(length))
        return self.series(start).before(session, length)

    def _fetch(self, start: date, today: date) -> RegimeSeries:
        try:
            download = self._download if self._download is not None else yfinance_download()
            raw = download(
                list(SYMBOLS),
                start=start.isoformat(),
                end=(today + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=False,
                actions=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            raise RegimeDataError(f"Yahoo download failed for {', '.join(SYMBOLS)}: {exc}") from exc
        if raw is None or raw.empty:
            raise RegimeDataError(f"Yahoo returned no data for {', '.join(SYMBOLS)}")

        closes = {symbol: self._closes(raw, symbol) for symbol in REQUIRED}
        spy = closes["SPY"]
        for symbol, values in closes.items():
            if values.empty:  # a partial download must not be cached: the next session retries
                raise RegimeDataError(f"Yahoo returned no {symbol} closes")
        vix = closes["^VIX"].reindex(spy.index).ffill(limit=self._max_fill)
        vix3m = closes["^VIX3M"].reindex(spy.index).ffill(limit=self._max_fill)
        try:
            vixy_closes = self._closes(raw, "VIXY")
        except RegimeDataError:  # optional: a missing VIXY must not cost the regime
            vixy_closes = spy.iloc[:0]
        vixy = vixy_closes.reindex(spy.index).ffill(limit=self._max_fill)
        return RegimeSeries(
            tuple(spy.index),
            tuple(float(v) for v in spy),
            tuple(float(v) for v in vix),
            tuple(float(v) for v in vix3m),
            tuple(float(v) for v in vixy),
        )

    @staticmethod
    def _closes(raw: pd.DataFrame, symbol: str) -> pd.Series:
        """One symbol's closes indexed by session date, sorted, last duplicate kept."""
        import pandas as pd  # not at module level: importing `core` must not import pandas

        try:
            frame = parse_yahoo_frame(extract_ticker_frame(raw, symbol))
        except Exception as exc:
            raise RegimeDataError(f"Unreadable Yahoo data for {symbol}: {exc}") from exc
        if frame.empty:
            return pd.Series(dtype="float64")
        index = pd.Index([ts.date() for ts in frame.index])
        close = pd.Series(frame["close"].to_numpy(), index=index)
        return close.groupby(level=0).last().sort_index()

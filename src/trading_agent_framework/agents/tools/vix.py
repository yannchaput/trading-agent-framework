"""VixSeries: the VIX's daily closes from Yahoo, for charting only (the Alpaca backtest data has no `^VIX`).

The strategy never trades on it. `previous_close(day)` returns the last close strictly BEFORE `day`, so a chart
point never shows a value that was not yet known during that session (no look-ahead).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from trading_agent_framework.backtesting.data.yahoo import MARKET_TZ, parse_yahoo_frame

VIX_SYMBOL = "^VIX"
_LOOKBACK_DAYS = 10  # calendar buffer, so the first session of a window still has a previous close

DownloadFn = Callable[..., Any]


class VixSeries:
    def __init__(self, download: DownloadFn | None = None) -> None:
        self._download = download
        self._days: list[date] = []
        self._closes: list[float] = []

    def load(self, start: date, end: date) -> None:
        """Fetch the closes for `[start - buffer, end]` (one call; raises on a Yahoo failure)."""
        download = self._download
        if download is None:
            import yfinance as yf  # deferred: only a backtest that charts the VIX pays for it

            download = yf.download
        raw = download(
            VIX_SYMBOL,
            start=(start - timedelta(days=_LOOKBACK_DAYS)).isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        frame = parse_yahoo_frame(raw)
        self._days = [ts.astimezone(MARKET_TZ).date() for ts in frame.index]
        self._closes = [float(v) for v in frame["close"]] if not frame.empty else []

    def previous_close(self, day: date) -> float | None:
        """The last close strictly before `day`; None when the series has none."""
        index = bisect_left(self._days, day)
        return self._closes[index - 1] if index > 0 else None

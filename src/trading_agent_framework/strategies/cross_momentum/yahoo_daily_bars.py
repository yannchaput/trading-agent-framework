"""Yahoo as the source of cross_momentum's daily bars in paper/live.

Alpaca's IEX bars carry ~5% of consolidated volume, skip a thin stock's quiet days, and close at the last IEX
trade rather than the official close, so the strategy's filter and ranks ran on other data than the backtest's
(whose bars are Yahoo's). One batched download per scan gives paper/live the backtest's own bars, split- and
dividend-adjusted. `yfinance` is imported lazily.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import date, timedelta

import pandas as pd

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import YahooDataError

logger = logging.getLogger(__name__)

HISTORY_BARS = 301  # the scan's 300 completed sessions plus the bar still forming
_CALENDAR_DAYS_PER_BAR = 1.5  # 301 sessions span ~440 days; the slack covers holidays and missing bars
_FIELDS = ["open", "high", "low", "close", "volume"]

DownloadFn = Callable[..., pd.DataFrame]


class YahooDailyBars:
    """Daily `Bars` per symbol from one batched Yahoo download."""

    def __init__(self, *, download: DownloadFn | None = None, history: int = HISTORY_BARS) -> None:
        self._download = download  # injected in tests; yfinance.download otherwise
        self._history = history

    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]:
        """{symbol: bars} for the symbols Yahoo has data for, oldest first, indexed at midnight market time (as
        live Alpaca bars are). The bar dated `today` is included while the session is open: the caller drops it.

        Raises YahooDataError when the download fails or returns nothing at all (an outage, not a missing symbol).
        """
        symbol_list = list(symbols)
        try:
            download = self._download if self._download is not None else self._real_download()
            raw = download(
                symbol_list,
                start=(today - timedelta(days=round(self._history * _CALENDAR_DAYS_PER_BAR))).isoformat(),
                end=(today + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=True,
                progress=False,
                group_by="ticker",
            )
        except Exception as exc:
            raise YahooDataError(f"Yahoo download failed for {len(symbol_list)} symbols: {exc}") from exc
        if raw is None or raw.empty:
            raise YahooDataError(f"Yahoo returned no data for {len(symbol_list)} symbols")

        result: dict[str, Bars] = {}
        for symbol in symbol_list:
            frame = _ticker_frame(raw, symbol)
            if frame is not None:
                result[symbol] = Bars(asset=Asset(symbol), timestep="day", df=frame)
        return result

    @staticmethod
    def _real_download() -> DownloadFn:
        import yfinance as yf

        # yfinance's own "Failed download" notices go to a handler-less logger, i.e. a raw dump on stderr that
        # bypasses the project's log files; the caller logs how many symbols had no data.
        yf_logger = logging.getLogger("yfinance")
        if not yf_logger.handlers:
            yf_logger.addHandler(logging.NullHandler())
            yf_logger.propagate = False
        return yf.download


def _ticker_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """One ticker's OHLCV out of a batched download in `Bars.df` shape, or None when Yahoo has nothing usable."""
    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        frame = raw[symbol]
    frame = frame.rename(columns=str.lower)
    if not set(_FIELDS) <= set(frame.columns):
        return None
    frame = frame[_FIELDS].dropna().astype("float64")  # a halt or a gap leaves NaN rows: no price, no information
    if frame.empty:
        return None
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(MARKET_TZ)
    frame.index.name = "timestamp"
    return frame.sort_index()

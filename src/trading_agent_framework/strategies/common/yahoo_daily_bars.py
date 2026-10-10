"""Yahoo as the source of daily bars in paper/live (cross_momentum, bull_bear).

Alpaca's IEX bars carry ~5% of consolidated volume, skip a thin stock's quiet days, and close at the last IEX
trade rather than the official close, so the strategy's filter and ranks ran on other data than the backtest's
(whose bars are Yahoo's). One batched download per scan gives paper/live the backtest's own bars: the same
download (split- and dividend-adjusted) and the same parse (`backtesting/data/yahoo.py`), stamped at the session
close. `yfinance` is imported lazily.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import date, timedelta

import pandas as pd

from trading_agent_framework.backtesting.data.yahoo import extract_ticker_frame, parse_yahoo_frame, yfinance_download
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.errors import YahooDataError

logger = logging.getLogger(__name__)

HISTORY_BARS = 301  # the scan's 300 completed sessions plus the bar still forming
_CALENDAR_DAYS_PER_BAR = 1.5  # 301 sessions span ~440 days; the slack covers holidays and missing bars

DownloadFn = Callable[..., pd.DataFrame]


class YahooDailyBars:
    """Daily `Bars` per symbol from one batched Yahoo download."""

    def __init__(self, *, download: DownloadFn | None = None, history: int = HISTORY_BARS) -> None:
        self._download = download  # injected in tests; yfinance.download otherwise
        self._history = history

    def bars(self, symbols: Sequence[str], today: date) -> dict[str, Bars]:
        """{symbol: bars} for the symbols Yahoo has data for, oldest first, each stamped at its session's close. The
        bar dated `today` is included while the session is open: the caller drops it.

        Raises YahooDataError when the download fails or returns nothing at all (an outage, not a missing symbol). A
        symbol whose data cannot be parsed is left out with a warning, like one Yahoo has nothing for.
        """
        symbol_list = list(symbols)
        try:
            download = self._download if self._download is not None else yfinance_download()
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
            try:
                frame = parse_yahoo_frame(extract_ticker_frame(raw, symbol))
            except Exception as exc:  # an unexpected layout for one ticker must not abort the whole scan
                logger.warning("Unreadable Yahoo data for %s: %s", symbol, exc)
                continue
            if not frame.empty:
                result[symbol] = Bars(asset=Asset(symbol), timestep="day", df=frame)
        return result

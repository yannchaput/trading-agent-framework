"""Yahoo as the source of every input of cross_momentum's `apply_filters` in paper/live.

Alpaca's IEX feed carries ~5% of consolidated volume, so `min_dollar_volume` was ~20x stricter live than in a
backtest (whose Yahoo bars are consolidated), and a thin stock's IEX bars skip the days without an IEX print.
Yahoo gives paper/live the same price, volume, volatility and history the backtest sees, computed the same way
(the scan's own formulas). Only the filter reads them: scores, ranks and weights still use Alpaca's
closes. `yfinance` is imported lazily.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from trading_agent_framework.strategies.cross_momentum.utils import annualized_volatility
from trading_agent_framework.utils.errors import FilterDataError

logger = logging.getLogger(__name__)

HISTORY_BARS = 300  # completed sessions behind the filter, as the scan's own history
VOLUME_WINDOW = 20  # sessions averaged for the dollar volume, as the backtest's own 20-bar average
VOLATILITY_WINDOW = 20
_CALENDAR_DAYS_PER_BAR = 1.5  # 300 sessions span ~436 days; the slack covers holidays and missing bars

DownloadFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class FilterInputs:
    """What `apply_filters` checks, for one symbol."""

    price: float
    avg_dollar_volume: float
    volatility: float | None
    trading_days: int


def filter_inputs(
    frame: pd.DataFrame,
    today: date,
    *,
    history: int = HISTORY_BARS,
    volume_window: int = VOLUME_WINDOW,
    volatility_window: int = VOLATILITY_WINDOW,
) -> FilterInputs | None:
    """The filter inputs from a Yahoo daily frame (`Close`, `Volume`, indexed by session date).

    Completed sessions only (a bar dated today is partial), the last `history` of them, rows with a missing close
    or volume dropped. None without a full volume window: a liquidity read from fewer sessions would not match
    the backtest's.
    """
    bars = frame[["Close", "Volume"]].dropna()
    bars = bars[[d < today for d in pd.DatetimeIndex(bars.index).date]].tail(history)
    if len(bars) < volume_window:
        return None
    closes = [float(close) for close in bars["Close"]]
    price = closes[-1]
    avg_volume = float(bars["Volume"].tail(volume_window).mean())
    return FilterInputs(
        price=price,
        avg_dollar_volume=avg_volume * price,
        volatility=annualized_volatility(closes, volatility_window),
        trading_days=len(closes),
    )


class YahooFilterSource:
    """`FilterInputs` per symbol from one batched Yahoo download."""

    def __init__(
        self,
        *,
        download: DownloadFn | None = None,
        history: int = HISTORY_BARS,
        volatility_window: int = VOLATILITY_WINDOW,
    ) -> None:
        self._download = download  # injected in tests; yfinance.download otherwise
        self._history = history
        self._volatility_window = volatility_window

    def inputs(self, symbols: Sequence[str], today: date) -> dict[str, FilterInputs]:
        """{symbol: inputs} for the symbols Yahoo has a full volume window for; the others are left out.

        Raises FilterDataError when the download fails or returns nothing at all (an outage, not a missing symbol).
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
            raise FilterDataError(f"Yahoo download failed for {len(symbol_list)} symbols: {exc}") from exc
        if raw is None or raw.empty:
            raise FilterDataError(f"Yahoo returned no data for {len(symbol_list)} symbols")

        result: dict[str, FilterInputs] = {}
        for symbol in symbol_list:
            frame = _ticker_frame(raw, symbol)
            if frame is None:
                continue
            values = filter_inputs(frame, today, history=self._history, volatility_window=self._volatility_window)
            if values is not None:
                result[symbol] = values
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
    """One ticker's columns out of a batched download, or None when Yahoo has nothing for it."""
    frame = raw
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        frame = raw[symbol]
    if "Close" not in frame.columns or "Volume" not in frame.columns:
        return None
    return frame

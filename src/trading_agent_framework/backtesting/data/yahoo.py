"""`BacktestDataSource` backed by yfinance daily OHLCV -- the default provider
(design spec, section 4.2): consolidated tape, official closing-auction closes,
decades of free history. `yfinance` is imported lazily so a strategy that never
backtests with Yahoo data never pays for its 12 transitive dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

if TYPE_CHECKING:
    import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)

DownloadFn = Callable[..., "pd.DataFrame"]


class YahooBacktestData(BacktestDataSource):
    """Daily OHLCV from Yahoo Finance. Only `timestep="day"` is supported (design
    spec, section 4.1 -- Yahoo has no usable minute history)."""

    name = "yahoo"

    def __init__(self, start: datetime, end: datetime, *, download: DownloadFn | None = None) -> None:
        self._start = start
        self._end = end
        self._download = download  # injected in tests; real yfinance.download otherwise
        self._frames: dict[Asset, pd.DataFrame] = {}

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._fetch(asset, timestep, start, end)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        if timestep != "day":
            raise BacktestDataError(f"YahooBacktestData only supports timestep='day', got {timestep!r}")
        df = self._frames.get(asset)
        if df is None:
            df = self._fetch(asset, timestep, self._start, self._end)
        if df is None or df.empty:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        """One 9:30-16:00 ET session per weekday in [start, end]. Half-days are not
        modelled (design spec, section 4.1) -- irrelevant for daily-bar strategies."""
        sessions: list[MarketSession] = []
        day = start.astimezone(MARKET_TZ).date()
        last = end.astimezone(MARKET_TZ).date()
        while day <= last:
            if day.weekday() < 5:
                sessions.append(
                    MarketSession(
                        open=datetime.combine(day, SESSION_OPEN, tzinfo=MARKET_TZ),
                        close=datetime.combine(day, SESSION_CLOSE, tzinfo=MARKET_TZ),
                    )
                )
            day += timedelta(days=1)
        return sessions

    def _fetch(self, asset: Asset, timestep: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        try:
            download = self._download if self._download is not None else self._real_download()
            raw = download(
                asset.symbol,
                start=start.date().isoformat(),
                end=(end.date() + timedelta(days=1)).isoformat(),
                auto_adjust=True,
                progress=False,
            )
            df = parse_yahoo_frame(raw)
        except ImportError as exc:
            raise BacktestDataError(
                "yfinance is required for YahooBacktestData; install the 'backtesting-yahoo' extra"
            ) from exc
        except Exception as exc:
            raise BacktestDataError(f"failed to fetch Yahoo data for {asset.symbol}: {exc}") from exc
        self._frames[asset] = df
        return df

    def _real_download(self) -> DownloadFn:
        import yfinance as yf

        return yf.download


def parse_yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure: normalise a yfinance download into `Bars.df` shape, indexed by bar CLOSE
    (each daily row's session date at 16:00 ET -- yfinance's own index is that
    session's date, naive)."""
    import pandas as pd

    if raw.empty:
        return raw
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    close_index = [datetime.combine(ts.date(), SESSION_CLOSE, tzinfo=MARKET_TZ) for ts in df.index]
    df.index = pd.DatetimeIndex(close_index, name="timestamp")
    return df.sort_index()

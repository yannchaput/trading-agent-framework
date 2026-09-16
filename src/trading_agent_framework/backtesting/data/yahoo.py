"""`BacktestDataSource` backed by yfinance daily OHLCV -- the default provider
(design spec, section 4.2): consolidated tape, official closing-auction closes,
decades of free history. `yfinance` is imported lazily so a strategy that never
backtests with Yahoo data never pays for its 12 transitive dependencies.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta
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

# The symbol whose daily bars stand in for the US equity trading calendar in
# `sessions()`. SPY prints on every regular session and has decades of history, so
# "a bar exists for this date" is a faithful proxy for "the market traded that day".
CALENDAR_SYMBOL = "SPY"

_OHLCV = ("open", "high", "low", "close", "volume")

DownloadFn = Callable[..., "pd.DataFrame"]


class YahooBacktestData(BacktestDataSource):
    """Daily OHLCV from Yahoo Finance. Only `timestep="day"` is supported (design
    spec, section 4.1 -- Yahoo has no usable minute history)."""

    name = "yahoo"

    def __init__(
        self,
        start: datetime,
        end: datetime,
        *,
        download: DownloadFn | None = None,
        calendar_symbol: str = CALENDAR_SYMBOL,
    ) -> None:
        self._start = start
        self._end = end
        self._download = download  # injected in tests; real yfinance.download otherwise
        self._frames: dict[Asset, pd.DataFrame] = {}
        self._calendar_asset = Asset(calendar_symbol)
        self._session_dates_cache: list[date] | None = None
        # Serializes calls into `download` on the lazy single-asset path (`_fetch`) --
        # an asset requested outside a `load()`-preloaded batch (e.g. a position held
        # from a prior run) still goes through `bars()`'s lazy fetch, possibly from a
        # strategy's own thread pool, and firing several `yf.download` calls at once
        # races yfinance's shared session/crumb handling.
        self._download_lock = threading.Lock()

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        """Batch-fetch `assets` in one `download` call instead of one per ticker --
        a strategy filtering a large universe (e.g. cross_momentum, hundreds of
        tickers) calling this once up front avoids ever falling back to `bars()`'s
        lazy per-asset fetch from its own thread pool, which would otherwise fire many
        concurrent `yf.download` calls, race yfinance's shared session/crumb handling,
        and trip Yahoo's rate limiting -- surfacing as the whole backtest stalling
        partway through the universe rather than a clean error.
        """
        missing = [asset for asset in assets if asset not in self._frames]
        if not missing:
            return
        symbols = [asset.symbol for asset in missing]
        try:
            download = self._download if self._download is not None else self._real_download()
            with self._download_lock:
                raw = download(
                    symbols,
                    start=start.date().isoformat(),
                    end=(end.date() + timedelta(days=1)).isoformat(),
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                )
        except ImportError as exc:
            raise BacktestDataError(
                "yfinance is required for YahooBacktestData; install the 'backtesting-yahoo' extra"
            ) from exc
        except Exception as exc:
            raise BacktestDataError(
                f"failed to batch-fetch Yahoo data for {len(symbols)} tickers "
                f"(starting {symbols[0]!r}): {exc}"
            ) from exc

        for asset in missing:
            self._frames[asset] = parse_yahoo_frame(_extract_ticker_frame(raw, asset.symbol))

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
        """One 9:30-16:00 ET session per ACTUAL daily bar in [start, end] (design spec,
        section 4.1: "One 9:30-16:00 ET session per daily bar").

        Dates come from `CALENDAR_SYMBOL`'s own fetched bar index, not from iterating
        weekdays: a weekday iteration counts the ~9 market holidays a year as trading
        sessions, and each spurious session drives a full, pointless strategy lifecycle
        (`before_market_opens` -> `on_trading_iteration` -> `after_market_closes`) plus
        a spurious equity sample, on a day when no bar exists and no price moved
        (Important whole-branch review finding).

        Half-days still close at 16:00 -- Yahoo's daily bars carry no session times, so
        early closes genuinely are not knowable here (design spec, section 4.1 says so
        explicitly; `AlpacaBacktestData` is the exact-calendar choice). Only *whether*
        a session happened is derived from the data, never its hours.

        Lazy, and normally free: the runner pre-loads the benchmark, which is SPY by
        default, so the calendar frame is already cached by the time this is called.
        A run with a non-SPY benchmark costs one extra daily download for the calendar,
        once per instance. Wrapped in `CachedDataSource`, a warm-cache rerun also pays
        that one download, since `CachedDataSource` caches bars but delegates
        `sessions()` straight through -- the same one-calendar-call-per-run cost
        `AlpacaBacktestData` has always had there.
        """
        first_date = start.astimezone(MARKET_TZ).date()
        last_date = end.astimezone(MARKET_TZ).date()
        return [
            MarketSession(
                open=datetime.combine(day, SESSION_OPEN, tzinfo=MARKET_TZ),
                close=datetime.combine(day, SESSION_CLOSE, tzinfo=MARKET_TZ),
            )
            for day in self._session_dates()
            if first_date <= day <= last_date
        ]

    def _session_dates(self) -> list[date]:
        """The calendar symbol's traded dates over the constructor's window, cached."""
        if self._session_dates_cache is None:
            df = self._frames.get(self._calendar_asset)
            if df is None:
                df = self._fetch(self._calendar_asset, "day", self._start, self._end)
            self._session_dates_cache = (
                [] if df is None or df.empty
                else [ts.astimezone(MARKET_TZ).date() for ts in df.index]
            )
        return self._session_dates_cache

    def _fetch(self, asset: Asset, timestep: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        try:
            download = self._download if self._download is not None else self._real_download()
            with self._download_lock:
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


def _extract_ticker_frame(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Slice one ticker's OHLCV out of a batched `download(symbols, group_by="ticker")`
    result. Columns come back as a `(ticker, field)` MultiIndex -- `raw[symbol]` drops
    the ticker level, leaving the same flat shape `parse_yahoo_frame` already expects
    from a single-ticker download. A symbol Yahoo had no data for at all is simply
    absent from the batch rather than raising (yfinance prints its own "Failed
    download" notice and moves on); treated the same as any other no-data ticker.
    """
    import pandas as pd

    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return pd.DataFrame()
        return raw[symbol]
    return raw  # a length-1 batch some yfinance versions return flat, not nested


def parse_yahoo_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Pure: normalise a yfinance download into `Bars.df` shape, indexed by bar CLOSE
    (each daily row's session date at 16:00 ET -- yfinance's own index is that
    session's date, naive).

    Rows with a NaN in any OHLCV column are dropped. yfinance legitimately emits them
    (trading halts, delistings, gaps in a multi-ticker frame), and they are not merely
    cosmetic downstream (Important whole-branch review finding): `BacktestBroker
    ._latest_bar_with_time` turns such a row into `Decimal('NaN')` prices, a MARKET
    order filling against one poisons `_cash` with NaN *permanently* -- surfacing much
    later as blank metrics rather than as an error -- and a LIMIT/STOP order instead
    raises `decimal.InvalidOperation` out of `fills.py`'s comparisons. This is a
    data-quality filter, not a no-look-ahead concern, so simply removing the bad rows
    is the right answer: a bar with no prices carries no information to preserve.
    """
    import pandas as pd

    if raw.empty:
        return raw
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[list(_OHLCV)].astype("float64").dropna(subset=list(_OHLCV))
    close_index = [datetime.combine(ts.date(), SESSION_CLOSE, tzinfo=MARKET_TZ) for ts in df.index]
    df.index = pd.DatetimeIndex(close_index, name="timestamp")
    return df.sort_index()

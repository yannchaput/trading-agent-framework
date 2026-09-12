"""`BacktestDataSource` backed by Alpaca's IEX feed, reusing the existing pure
`brokers/alpaca/market_data.py` translation and `brokers/alpaca/account.py`'s
calendar parsing. Exact sessions (early closes included) -- the choice when feed
parity with paper/live matters more than Yahoo's decades of free history (design
spec, section 4.2).

Bars are re-indexed from Alpaca's bar-START convention to this subsystem's
bar-CLOSE convention (see `data/base.py`): a daily bar's index becomes its
session's actual close (handling early closes correctly); a minute bar's index
becomes `timestamp + 1 minute` (Alpaca minute bars are exactly 1-minute windows
starting at `timestamp`).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from trading_agent_framework.backtesting.data.base import FULL_HISTORY, BacktestDataSource
from trading_agent_framework.brokers.alpaca import account, market_data
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BacktestDataError

if TYPE_CHECKING:
    import pandas as pd
    from alpaca.trading.requests import GetCalendarRequest


class AlpacaTradingCalendarClient(Protocol):
    """The one trading-client method this module needs -- narrower than the full
    `orders.AlpacaTradingClient` Protocol since this module never submits orders."""

    def get_calendar(self, filters: GetCalendarRequest) -> list[object]: ...


class AlpacaBacktestData(BacktestDataSource):
    """Exact Alpaca calendar sessions; IEX daily/minute bars, fetched lazily per asset
    over the fixed `[start, end]` window given at construction (`bars()`'s own
    lazy-fetch path, which carries no window of its own, uses that fixed window;
    `load()` always honors its own caller-supplied `start`/`end`)."""

    name = "alpaca"

    def __init__(
        self,
        client: market_data.AlpacaStockDataClient,
        trading_client: AlpacaTradingCalendarClient,
        start: datetime,
        end: datetime,
    ) -> None:
        self._client = client
        self._trading_client = trading_client
        self._start = start
        self._end = end
        self._frames: dict[Asset, pd.DataFrame] = {}
        self._sessions_cache: list[MarketSession] | None = None

    def load(self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str) -> None:
        for asset in assets:
            self._fetch(asset, timestep, start, end)

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
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
        if self._sessions_cache is None:
            request = account.build_calendar_request(self._start.date(), self._end.date())
            try:
                days = self._trading_client.get_calendar(filters=request)
            except Exception as exc:
                raise BacktestDataError(f"failed to fetch the Alpaca calendar: {exc}") from exc
            self._sessions_cache = account.parse_calendar(days, market_data.MARKET_TZ)
        return [s for s in self._sessions_cache if s.open >= start and s.close <= end]

    def _fetch(self, asset: Asset, timestep: str, start: datetime, end: datetime) -> pd.DataFrame | None:
        import pandas as pd

        request = market_data.build_bars_request([asset], timestep, start, end)
        try:
            barset = self._client.get_stock_bars(request)
            parsed = market_data.parse_bars(barset, [asset], timestep, FULL_HISTORY)
        except Exception as exc:
            raise BacktestDataError(f"failed to fetch Alpaca bars for {asset.symbol}: {exc}") from exc
        source_bars = parsed.get(asset)
        if source_bars is None:
            df = pd.DataFrame()
        else:
            # Calendar failures here surface as BacktestDataError from sessions()
            # itself (message already mentions "calendar") -- kept outside the
            # try/except above so that message isn't swallowed by the generic one.
            sessions = self.sessions(start, end)
            df = reindex_to_bar_close(source_bars.df, timestep, sessions)
        self._frames[asset] = df
        return df


def reindex_to_bar_close(df: pd.DataFrame, timestep: str, sessions: Sequence[MarketSession]) -> pd.DataFrame:
    """Pure: shift Alpaca's bar-START index to bar-CLOSE.

    Minute bars are exactly 1-minute windows starting at `timestamp`: shift by
    +1 minute. Daily bars are timestamped at midnight UTC of the *nominal*
    session date (Alpaca convention); `market_data._bars_frame` already
    converts that index to America/New_York, which -- since ET trails UTC --
    lands on the *previous* ET calendar day (e.g. `2026-01-05T00:00:00Z` becomes
    `2026-01-04T19:00:00-05:00`). Converting back to UTC before taking `.date()`
    recovers the correct nominal session date to look up each session's close.
    """
    import pandas as pd

    if df.empty:
        return df
    if timestep == "minute":
        df = df.copy()
        df.index = df.index + timedelta(minutes=1)
        return df.sort_index()
    close_by_date = {s.open.astimezone(market_data.MARKET_TZ).date(): s.close for s in sessions}
    new_index = [close_by_date.get(ts.astimezone(UTC).date(), ts) for ts in df.index]
    df = df.copy()
    df.index = pd.DatetimeIndex(new_index, name="timestamp")
    return df.sort_index()

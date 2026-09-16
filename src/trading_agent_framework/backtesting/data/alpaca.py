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
    `load()` always honors its own caller-supplied `start`/`end`). The calendar cache
    backing `sessions()` is NOT fixed to that window, though: it grows on demand to
    cover the union of every range it's ever been queried with (see `sessions()`)."""

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
        # The [start, end] the cached calendar actually covers -- starts as the
        # constructor's own window and grows (never shrinks) to the union of every
        # range `sessions()` has ever been asked about. See `sessions()`'s docstring
        # for why this must track queried ranges rather than staying fixed.
        self._calendar_start = start
        self._calendar_end = end

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
        """Sessions in `[start, end]`, fetched from Alpaca's calendar and cached.

        The cache is keyed by the UNION of every range ever asked for, not the fixed
        constructor window (Critical review finding, Task 6): `_fetch` calls this with
        whatever `start`/`end` its own caller used, which for `load()` can be wider
        than the constructor's `self._start`/`self._end` (e.g. `runner._run`'s
        warmup-widened eager benchmark load). A cache request fixed at the constructor
        window would silently omit any date `load()` widened past it, so
        `reindex_to_bar_close` would raise `BacktestDataError` for those bars('no
        trading session found') even though the requested warm-up range is completely
        valid -- not a cosmetic gap: it made `AlpacaBacktestData` incompatible with the
        exact widened-eager-load mechanism `warmup_trading_days` relies on. Re-fetches
        only when the requested range isn't already covered, so the common case (every
        query within `[self._start, self._end]`) still costs exactly one calendar call.
        """
        if (
            self._sessions_cache is None
            or start < self._calendar_start
            or end > self._calendar_end
        ):
            fetch_start = min(start, self._calendar_start)
            fetch_end = max(end, self._calendar_end)
            request = account.build_calendar_request(fetch_start.date(), fetch_end.date())
            try:
                days = self._trading_client.get_calendar(filters=request)
            except Exception as exc:
                raise BacktestDataError(f"failed to fetch the Alpaca calendar: {exc}") from exc
            # Assign the cache BEFORE widening the bounds it's keyed by, and preserve
            # this order: `CrossMomentumStrategy` fans out `get_historical_prices`
            # across a `ThreadPoolExecutor`, so `sessions()` can race here. A reader
            # that sees the new `_sessions_cache` but still the OLD `_calendar_start`/
            # `_calendar_end` only over-fetches (re-runs this branch for a range the
            # cache already covers) -- safe. The reverse order would let a racing
            # reader see bounds that claim coverage the cache doesn't actually contain
            # yet, and skip a fetch it needed -- an under-covering read, which is the
            # failure mode this whole module exists to prevent.
            self._sessions_cache = account.parse_calendar(days, market_data.MARKET_TZ)
            self._calendar_start, self._calendar_end = fetch_start, fetch_end
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
            df = reindex_to_bar_close(source_bars.df, timestep, sessions, symbol=asset.symbol)
        self._frames[asset] = df
        return df


def reindex_to_bar_close(
    df: pd.DataFrame,
    timestep: str,
    sessions: Sequence[MarketSession],
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Pure: shift Alpaca's bar-START index to bar-CLOSE.

    Minute bars are exactly 1-minute windows starting at `timestamp`: shift by
    +1 minute. Daily bars are timestamped at midnight *market time* -- a
    DST-aware UTC offset (`04:00Z` in EDT, `05:00Z` in EST), not a fixed
    `00:00:00Z` (see `brokers/alpaca/market_data.py`'s own documented
    convention). `market_data._bars_frame` already converts that index to
    America/New_York, which lands exactly on midnight ET of the nominal
    session date. Converting back to UTC before taking `.date()` is a
    defensive no-op for correctly-shaped data -- both `.date()` and
    `.astimezone(UTC).date()` agree on the nominal session date for any
    midnight-ET timestamp -- kept in case this function ever receives
    timestamps that aren't already market-time-aware.

    A daily bar whose date has NO session in `sessions` raises `BacktestDataError`.
    This used to fall back to the bar's ORIGINAL index instead, which is a real
    no-look-ahead violation and not a cosmetic one (Important whole-branch review
    finding): that original index is midnight ET, i.e. ~9.5 hours before the session
    it belongs to even OPENS, so the bar would become visible to the strategy through
    `bars(..., cutoff)` most of a day early -- silently, and looking entirely
    plausible. The situation is reachable whenever `sessions()`'s window filter drops
    a date the bars fetch kept (e.g. an `end` set mid-session, so that day's session
    fails the `s.close <= end` test while Alpaca still returns its partial daily bar),
    so failing loudly is the only safe answer -- there is no correct close time to
    re-index to.
    """
    import pandas as pd

    if df.empty:
        return df
    if timestep == "minute":
        df = df.copy()
        df.index = df.index + timedelta(minutes=1)
        return df.sort_index()
    close_by_date = {s.open.astimezone(market_data.MARKET_TZ).date(): s.close for s in sessions}
    new_index: list[datetime] = []
    for ts in df.index:
        bar_date = ts.astimezone(UTC).date()
        close = close_by_date.get(bar_date)
        if close is None:
            subject = f"{symbol} " if symbol else ""
            raise BacktestDataError(
                f"no trading session found for the {subject}daily bar dated {bar_date}: "
                "cannot re-index it from bar-open to bar-close, and keeping its "
                "bar-open index would expose the bar hours before its session closes. "
                "Construct the data source with a window that fully covers every "
                "session you request bars for (whole sessions, not a mid-session end)."
            )
        new_index.append(close)
    df = df.copy()
    df.index = pd.DatetimeIndex(new_index, name="timestamp")
    return df.sort_index()

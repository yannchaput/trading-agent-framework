"""In-memory BacktestDataSource fake shared by every backtesting test module."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession

_MARKET_TZ = ZoneInfo("America/New_York")


class FakeBacktestDataSource(BacktestDataSource):
    """Bars set directly by the test via `set_bars`; must already be indexed by bar
    CLOSE timestamp (tz-aware), matching the real contract in `data/base.py`."""

    name = "fake"

    def __init__(self) -> None:
        self._frames: dict[tuple[Asset, str | None], pd.DataFrame] = {}
        self._sessions: list[MarketSession] = []
        self.load_calls: list[tuple[Asset, ...]] = []
        self.load_windows: list[tuple[datetime, datetime]] = []
        self.bars_calls: list[tuple[Asset, datetime, int, str]] = []

    def set_bars(self, asset: Asset, df: pd.DataFrame, *, timestep: str | None = None) -> None:
        """`timestep=None` (the default) is the fallback frame `bars()` returns for a
        request at ANY timestep that has no more specific override -- every existing
        caller that doesn't care about timestep-specific data keeps working unchanged.
        Pass `timestep=` to make a request for that exact timestep see different bars,
        e.g. to catch a caller that fetches the wrong cadence (a real bug found in
        `runner.py`: the benchmark used to be fetched at the strategy's raw bar
        timestep instead of always at daily cadence)."""
        self._frames[(asset, timestep)] = df.sort_index()

    def set_sessions(self, sessions: list[MarketSession]) -> None:
        self._sessions = sessions

    def load(self, assets, start, end, timestep) -> None:
        self.load_calls.append(tuple(assets))
        self.load_windows.append((start, end))

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        self.bars_calls.append((asset, cutoff, length, timestep))
        df = self._frames.get((asset, timestep))
        if df is None:
            df = self._frames.get((asset, None))
        if df is None:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        # By CALENDAR DATE, exactly like YahooBacktestData/AlpacaBacktestData: a bare-midnight `end`
        # (the conventional way to spell a backtest's last day) means "through the end of that day".
        first_date = start.astimezone(_MARKET_TZ).date()
        last_date = end.astimezone(_MARKET_TZ).date()
        return [s for s in self._sessions if first_date <= s.open.astimezone(_MARKET_TZ).date() <= last_date]


def make_close_indexed_frame(
    closes: list[float], *, start: datetime, freq: str = "1D"
) -> pd.DataFrame:
    """An OHLCV frame indexed by bar CLOSE (see `data/base.py`'s convention).
    high = close + 1, low = close - 1, matching `tests/fakes.py:make_bars_frame`'s shape."""
    index = pd.date_range(start, periods=len(closes), freq=freq, name="timestamp")
    close = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": close,
            "volume": [1000.0] * len(close),
        },
        index=index,
    )

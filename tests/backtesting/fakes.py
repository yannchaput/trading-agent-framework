"""In-memory BacktestDataSource fake shared by every backtesting test module."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession


class FakeBacktestDataSource(BacktestDataSource):
    """Bars set directly by the test via `set_bars`; must already be indexed by bar
    CLOSE timestamp (tz-aware), matching the real contract in `data/base.py`."""

    name = "fake"

    def __init__(self) -> None:
        self._frames: dict[Asset, pd.DataFrame] = {}
        self._sessions: list[MarketSession] = []
        self.load_calls: list[tuple[Asset, ...]] = []
        self.bars_calls: list[tuple[Asset, datetime, int, str]] = []

    def set_bars(self, asset: Asset, df: pd.DataFrame) -> None:
        self._frames[asset] = df.sort_index()

    def set_sessions(self, sessions: list[MarketSession]) -> None:
        self._sessions = sessions

    def load(self, assets, start, end, timestep) -> None:
        self.load_calls.append(tuple(assets))

    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        self.bars_calls.append((asset, cutoff, length, timestep))
        df = self._frames.get(asset)
        if df is None:
            return None
        visible = df[df.index <= cutoff]
        if visible.empty:
            return None
        return Bars(asset=asset, timestep=timestep, df=visible.tail(length))

    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        return [s for s in self._sessions if s.open >= start and s.close <= end]


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

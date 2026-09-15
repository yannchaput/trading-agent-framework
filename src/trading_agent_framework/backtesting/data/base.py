"""Provider-agnostic backtest data source: the seam a future IBKR implementation
plugs into, and the only thing `BacktestBroker` depends on for prices and sessions.

`bars()` is the no-look-ahead chokepoint (design spec, section 5): its `cutoff`
parameter is always `clock.now()`, and every returned row's index value is that
bar's CLOSE timestamp (not its open/start -- a deliberate convention specific to
this subsystem, distinct from the live Alpaca `Bars` convention used by
`brokers/alpaca/market_data.py`, which indexes at bar open). A row at index `t`
means "this bar was fully formed and knowable as of `t`"; filtering `index <=
cutoff` is then sufficient and correct.

Implementations fetch lazily: `bars()` must return data for any asset even if
`load()` was never called for it -- `load()` is a pre-warming optimisation (e.g.
the runner uses it once for the benchmark), not a precondition. This lets a
strategy trade an asset picked dynamically inside `on_trading_iteration()`
without the runner knowing the universe in advance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import ClassVar

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.utils.clock import MarketSession

FULL_HISTORY = 10_000_000  # "return every bar you have" sentinel for bars(..., length=...)


class BacktestDataSource(ABC):
    """Bars and sessions for a backtest run. Never imports a broker; `BacktestBroker`
    depends on this, never the reverse."""

    name: ClassVar[str]

    @abstractmethod
    def load(
        self, assets: Sequence[Asset], start: datetime, end: datetime, timestep: str
    ) -> None:
        """Pre-fetch and cache `assets` for [start, end]. Optional to call; `bars()`
        fetches lazily for any asset it hasn't seen."""

    @abstractmethod
    def bars(self, asset: Asset, cutoff: datetime, length: int, timestep: str) -> Bars | None:
        """The last `length` bars closed at or before `cutoff`, oldest first, indexed
        by bar CLOSE time. None if the asset has no data at all."""

    @abstractmethod
    def sessions(self, start: datetime, end: datetime) -> list[MarketSession]:
        """Trading sessions in [start, end], early closes included where known."""

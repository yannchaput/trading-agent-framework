"""Pure Alpaca market-data translation: timesteps, the fetch window, requests, responses.

Same rules as `orders.py` and `account.py`: no I/O, no state, no client instances. The only
module allowed to import `alpaca.data.requests`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Protocol
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame

from trading_agent_framework.clock import MarketSession
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.errors import BrokerError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.data.models import Quote as AlpacaQuote
    from alpaca.data.models import Trade as AlpacaTrade

MARKET_TZ = ZoneInfo("America/New_York")
FEED = DataFeed.IEX
ADJUSTMENT = Adjustment.ALL  # split- and dividend-adjusted, like lumibot's default
MAX_SYMBOLS_PER_REQUEST = 150
TIMESTEPS = ("minute", "day")
_MINUTES_PER_SESSION = 390


class AlpacaStockDataClient(Protocol):
    """What `AlpacaBroker` calls on a `StockHistoricalDataClient` (or a test fake).

    Narrower than the SDK's `... | RawData` return types: this project never enables raw_data.
    """

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...
    def get_stock_latest_trade(
        self, request_params: StockLatestTradeRequest
    ) -> dict[str, AlpacaTrade]: ...
    def get_stock_latest_quote(
        self, request_params: StockLatestQuoteRequest
    ) -> dict[str, AlpacaQuote]: ...


def parse_timestep(timestep: str) -> TimeFrame:
    if timestep == "minute":
        return TimeFrame.Minute
    if timestep == "day":
        return TimeFrame.Day
    raise ValueError(f"Unsupported timestep {timestep!r}; expected one of {TIMESTEPS}")


def sessions_needed(length: int, timestep: str) -> int:
    """Trading sessions to fetch for `length` bars; the +1 covers a partial current session."""
    parse_timestep(timestep)
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}")
    if timestep == "day":
        return length + 1
    return math.ceil(length / _MINUTES_PER_SESSION) + 1


def calendar_lookback_start(end: datetime, length: int, timestep: str) -> date:
    """First day of the calendar request: comfortably more calendar days than sessions needed."""
    days = math.ceil(sessions_needed(length, timestep) * 1.5) + 10
    return (end.astimezone(MARKET_TZ) - timedelta(days=days)).date()


def bars_start(
    end: datetime, length: int, timestep: str, sessions: Sequence[MarketSession]
) -> datetime:
    """Midnight (market time) of the earliest session needed, so its pre-market bars count."""
    needed = sessions_needed(length, timestep)
    today = end.astimezone(MARKET_TZ).date()
    past = [s for s in sessions if s.open.astimezone(MARKET_TZ).date() <= today]
    if not past:
        raise BrokerError(f"The Alpaca calendar has no session on or before {today}")
    first = past[max(0, len(past) - needed)]
    return datetime.combine(first.open.astimezone(MARKET_TZ).date(), time(0), tzinfo=MARKET_TZ)


def chunk_assets(assets: Iterable[Asset]) -> Iterator[list[Asset]]:
    """Unique assets, in order, in batches small enough for one Alpaca request."""
    unique = list(dict.fromkeys(assets))
    for i in range(0, len(unique), MAX_SYMBOLS_PER_REQUEST):
        yield unique[i : i + MAX_SYMBOLS_PER_REQUEST]


def _symbols(assets: Sequence[Asset]) -> list[str]:
    return [asset.symbol for asset in assets]


def build_bars_request(
    assets: Sequence[Asset], timestep: str, start: datetime, end: datetime
) -> StockBarsRequest:
    return StockBarsRequest(
        symbol_or_symbols=_symbols(assets),
        timeframe=parse_timestep(timestep),
        start=start,
        end=end,
        feed=FEED,
        adjustment=ADJUSTMENT,
    )


def build_latest_trade_request(assets: Sequence[Asset]) -> StockLatestTradeRequest:
    return StockLatestTradeRequest(symbol_or_symbols=_symbols(assets), feed=FEED)


def build_latest_quote_request(assets: Sequence[Asset]) -> StockLatestQuoteRequest:
    return StockLatestQuoteRequest(symbol_or_symbols=_symbols(assets), feed=FEED)

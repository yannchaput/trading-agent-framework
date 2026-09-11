"""Pure Alpaca market-data translation: timesteps, the fetch window, requests, responses.

Same rules as `orders.py` and `account.py`: no I/O, no state, no client instances. The only
module allowed to import `alpaca.data.requests`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame
from pandas.core.indexes.datetimes import DatetimeIndex

from trading_agent_framework.brokers.alpaca.orders import _field, _to_decimal
from trading_agent_framework.clock import MarketSession
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
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
_OHLCV = ("open", "high", "low", "close", "volume")


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


# --- response parsing ------------------------------------------------------------


def parse_bars(
    barset: object,
    assets: Sequence[Asset],
    timestep: str,
    length: int,
    *,
    sessions: Sequence[MarketSession] | None = None,
) -> dict[Asset, Bars]:
    """The last `length` bars per asset, oldest first. Assets without bars are left out.

    When `sessions` is given, only bars inside them are kept (so early closes are handled).
    This filter runs before truncating, so the caller gets `length` in-session bars
    whenever the feed has that many.
    """
    data = cast(Mapping[str, Sequence[object]], _field(barset, "data") or {})
    result: dict[Asset, Bars] = {}
    for asset in assets:
        df = _bars_frame(data.get(asset.symbol) or [])
        if sessions is not None:
            df = _within_sessions(df, sessions)
        df = df.iloc[-length:]
        if not df.empty:
            result[asset] = Bars(asset=asset, timestep=timestep, df=df)
    return result


def _bars_frame(rows: Sequence[object]) -> pd.DataFrame:
    """The float64 boundary for bars (see `entities/bars.py`): indicators want native floats."""
    index: DatetimeIndex = pd.to_datetime([_field(row, "timestamp") for row in rows], utc=True)  # type: ignore[assignment]
    columns = {name: [float(cast(float, _field(row, name))) for row in rows] for name in _OHLCV}
    df = pd.DataFrame(columns, index=index.tz_convert(MARKET_TZ), dtype="float64")
    df.index.name = "timestamp"
    return df[~df.index.duplicated(keep="first")].sort_index()


def _within_sessions(df: pd.DataFrame, sessions: Sequence[MarketSession]) -> pd.DataFrame:
    keep = pd.Series(False, index=df.index)
    for session in sessions:
        keep |= (df.index >= session.open) & (df.index < session.close)
    return df[keep]


def parse_latest_trades(
    response: Mapping[str, object], assets: Sequence[Asset]
) -> dict[Asset, Decimal | None]:
    """Last traded price per asset; None when Alpaca returned no trade for the symbol."""
    return {asset: _to_decimal(_field(response.get(asset.symbol), "price")) for asset in assets}


def parse_quote(response: Mapping[str, object], asset: Asset) -> Quote | None:
    raw = response.get(asset.symbol)
    if raw is None:
        return None
    return Quote(
        asset=asset,
        bid=_book_price(_field(raw, "bid_price")),
        ask=_book_price(_field(raw, "ask_price")),
        bid_size=_to_decimal(_field(raw, "bid_size")),
        ask_size=_to_decimal(_field(raw, "ask_size")),
        timestamp=cast(datetime, _field(raw, "timestamp")).astimezone(MARKET_TZ),
    )


def _book_price(value: object) -> Decimal | None:
    """Alpaca reports an empty book side as 0: that means "no price", not a price of zero."""
    price = _to_decimal(value)
    return price if price is not None and price > 0 else None

"""Pure Alpaca market-data translation: timesteps, the fetch window, requests, responses.

Same rules as `orders.py` and `account.py`: no I/O, no state, no client instances. The only
module allowed to import `alpaca.data.requests`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import (
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from alpaca.data.timeframe import TimeFrame

from trading_agent_framework.brokers.alpaca.orders import _field, _to_decimal
from trading_agent_framework.brokers.alpaca.symbols import to_alpaca_symbol
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from alpaca.data.models import BarSet
    from alpaca.data.models import Quote as AlpacaQuote
    from alpaca.data.models import Trade as AlpacaTrade
    from alpaca.data.models.news import NewsSet

MARKET_TZ = ZoneInfo("America/New_York")
FEED = DataFeed.IEX
ADJUSTMENT = Adjustment.ALL  # split- and dividend-adjusted, like lumibot's default
MAX_SYMBOLS_PER_REQUEST = 150
MAX_NEWS_LIMIT = 50
MAX_NEWS_CONTENT_CHARS = 6000
TRUNCATION_MARKER = "... [truncated]"
TIMESTEPS = ("minute", "day")
_MINUTES_PER_SESSION = 390
_OHLCV = ("open", "high", "low", "close", "volume")


class AlpacaStockDataClient(Protocol):
    """What `AlpacaBroker` calls on a `StockHistoricalDataClient` (or a test fake).

    Narrower than the SDK's `... | RawData` return types: this project never enables raw_data.
    """

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet: ...
    def get_stock_latest_trade(self, request_params: StockLatestTradeRequest) -> dict[str, AlpacaTrade]: ...
    def get_stock_latest_quote(self, request_params: StockLatestQuoteRequest) -> dict[str, AlpacaQuote]: ...


class AlpacaNewsClient(Protocol):
    """What `AlpacaBroker` calls on a `NewsClient` (or a test fake)."""

    def get_news(self, request_params: NewsRequest) -> NewsSet: ...


def parse_timestep(timestep: str) -> TimeFrame:
    if timestep == "minute":
        return cast(TimeFrame, TimeFrame.Minute)
    if timestep == "day":
        return cast(TimeFrame, TimeFrame.Day)
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


def bars_start(end: datetime, length: int, timestep: str, sessions: Sequence[MarketSession]) -> datetime:
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
    return [to_alpaca_symbol(asset.symbol) for asset in assets]


def build_bars_request(assets: Sequence[Asset], timestep: str, start: datetime, end: datetime) -> StockBarsRequest:
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


def build_news_request(
    symbols: Sequence[str],
    *,
    start: datetime | None,
    end: datetime,
    limit: int,
    include_content: bool,
) -> NewsRequest:
    return NewsRequest(
        symbols=",".join(to_alpaca_symbol(symbol) for symbol in symbols) if symbols else None,
        start=start.astimezone(UTC).replace(tzinfo=None) if start else None,
        end=end.astimezone(UTC).replace(tzinfo=None),
        limit=min(max(int(limit), 1), MAX_NEWS_LIMIT),
        include_content=include_content,
    )


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
    whenever the feed has that many. No-op for `"day"` bars, which Alpaca timestamps at
    midnight market time -- always outside any session's 09:30-16:00 window.
    """
    data = cast(Mapping[str, Sequence[object]], _field(barset, "data") or {})
    result: dict[Asset, Bars] = {}
    for asset in assets:
        df = _bars_frame(data.get(to_alpaca_symbol(asset.symbol)) or [])
        if sessions is not None and timestep != "day":
            df = _within_sessions(df, sessions)
        df = df.iloc[-length:]
        if not df.empty:
            result[asset] = Bars(asset=asset, timestep=timestep, df=df)
    return result


def _bars_frame(rows: Sequence[object]) -> pd.DataFrame:
    """The float64 boundary for bars (see `entities/bars.py`): indicators want native floats."""
    index = pd.to_datetime([_field(row, "timestamp") for row in rows], utc=True)  # pyright: ignore[reportArgumentType, reportCallIssue]
    columns = {name: [float(cast(float, _field(row, name))) for row in rows] for name in _OHLCV}
    df = pd.DataFrame(columns, index=index.tz_convert(MARKET_TZ), dtype="float64")
    df.index.name = "timestamp"
    return df[~df.index.duplicated(keep="first")].sort_index()


def _within_sessions(df: pd.DataFrame, sessions: Sequence[MarketSession]) -> pd.DataFrame:
    keep = pd.Series(False, index=df.index)
    for session in sessions:
        keep |= (df.index >= session.open) & (df.index < session.close)
    return df[keep]


def parse_latest_trades(response: Mapping[str, object], assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
    """Last traded price per asset; None when Alpaca returned no trade for the symbol."""
    return {asset: _to_decimal(_field(response.get(to_alpaca_symbol(asset.symbol)), "price")) for asset in assets}


def parse_quote(response: Mapping[str, object], asset: Asset) -> Quote | None:
    raw = response.get(to_alpaca_symbol(asset.symbol))
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


def _truncate_content(content: str) -> str:
    """Cap a news article body so a full-content read stays within the LLM token budget."""
    if len(content) <= MAX_NEWS_CONTENT_CHARS:
        return content
    return content[:MAX_NEWS_CONTENT_CHARS] + TRUNCATION_MARKER


def parse_news(news_set: object) -> list[dict[str, object]]:
    """Lean articles: id, headline, summary, source, created_at, symbols, and content when present."""
    articles = cast(Mapping[str, Sequence[object]], _field(news_set, "data") or {}).get("news", [])
    parsed: list[dict[str, object]] = []
    for article in articles:
        item: dict[str, object] = {
            "id": _field(article, "id"),
            "headline": _field(article, "headline"),
            "summary": _field(article, "summary"),
            "source": _field(article, "source"),
            "created_at": cast(datetime, _field(article, "created_at")).isoformat(),
            "symbols": list(cast(Sequence[str], _field(article, "symbols") or [])),
        }
        content = _field(article, "content")
        if content:
            item["content"] = _truncate_content(str(content))
        parsed.append(item)
    return parsed

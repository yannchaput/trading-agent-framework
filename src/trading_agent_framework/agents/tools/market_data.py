"""Plain typed market-data tools for a LangChain agent: last price, quote, bars."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

_MIN_BARS = 1
_MAX_BARS = 200


def market_data_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Market-data tools bound to `strategy`."""

    def get_last_price(symbol: str) -> dict[str, Any]:
        """Get the last traded price for a symbol."""
        try:
            price = strategy.get_last_price(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        if price is None:
            return {"error": f"no trade data for {symbol!r}"}
        return {"symbol": symbol.upper(), "price": float(price)}

    def get_quote(symbol: str) -> dict[str, Any]:
        """Get the latest bid/ask quote for a symbol."""
        try:
            quote = strategy.get_quote(symbol)
        except BrokerError as exc:
            return {"error": str(exc)}
        if quote is None:
            return {"error": f"no quote for {symbol!r}"}
        return {
            "symbol": symbol.upper(),
            "bid": float(quote.bid) if quote.bid is not None else None,
            "ask": float(quote.ask) if quote.ask is not None else None,
            "mid": float(quote.mid) if quote.mid is not None else None,
            "timestamp": quote.timestamp.isoformat(),
        }

    def get_bars(symbol: str, length: int = 30, timestep: str = "day") -> dict[str, Any]:
        """Get recent OHLCV bars for a symbol, oldest first."""
        clamped_length = min(max(int(length), _MIN_BARS), _MAX_BARS)
        try:
            bars = strategy.get_historical_prices(symbol, clamped_length, timestep)
        except BrokerError as exc:
            return {"error": str(exc)}
        if bars is None:
            return {"error": f"no bars for {symbol!r}"}
        rows = [
            {
                "date": cast(Any, index).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for index, row in bars.df.iterrows()
        ]
        return {"symbol": symbol.upper(), "timestep": timestep, "bars": rows}

    return [get_last_price, get_quote, get_bars]

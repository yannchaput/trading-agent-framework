"""Plain typed news tool for a LangChain agent: Alpaca news search, gated on the strategy clock."""

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from trading_agent_framework.brokers.alpaca.broker import AlpacaBroker
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from trading_agent_framework.core.strategy import Strategy

MIN_LIMIT = 1
MAX_LIMIT = 50
_DEFAULT_LOOKBACK_DAYS = 7


def news_tools(strategy: "Strategy") -> list[Callable[..., dict[str, Any]]]:  # noqa: UP037
    """Alpaca news tool bound to `strategy`."""

    def search_news(
        symbols: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> dict[str, Any]:
        """Search recent news headlines and summaries, optionally filtered to symbols."""
        if not isinstance(strategy.broker, AlpacaBroker):
            return {"error": "news requires an Alpaca broker"}
        try:
            now = strategy.clock.now()
            if end:
                end_parsed = datetime.fromisoformat(end)
                if end_parsed.tzinfo is None:
                    return {"error": "end date must include timezone information"}
                end_dt = min(end_parsed, now)
            else:
                end_dt = now
            if start:
                start_parsed = datetime.fromisoformat(start)
                if start_parsed.tzinfo is None:
                    return {"error": "start date must include timezone information"}
                start_dt = start_parsed
            else:
                start_dt = end_dt - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        except ValueError as exc:
            return {"error": f"invalid date format: {exc}"}
        clamped_limit = min(max(int(limit), MIN_LIMIT), MAX_LIMIT)
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        try:
            articles = strategy.broker.get_news(
                symbol_list, start=start_dt, end=end_dt, limit=clamped_limit, include_content=include_content
            )
        except BrokerError as exc:
            return {"error": str(exc)}
        return {"count": len(articles), "articles": articles}

    return [search_news]

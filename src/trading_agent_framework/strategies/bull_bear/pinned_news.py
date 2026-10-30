"""The researcher's `search_news`, pinned to the stock it is researching (pure apart from the call it wraps).

The shared tool reads an empty `symbols` as "no filter" (news_builtin scans the whole market that way), so a model that leaves
it out gets headlines about unrelated companies; and a model that does not recognise a ticker searches a company name
instead (GGAL as "Gallo", UI as "Universal Insurance"), which finds nothing and spends a call of its budget. In the 2026-10-30
2Y run that was 134 of 2,170 calls (about one researcher run in ten). A symbol that is not a ticker is replaced by the stock
being researched, so the call is never lost and the shared tool stays as it is.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_TICKER = re.compile(r"[A-Z][A-Z0-9.\-]{0,9}")  # upper case on purpose: "Hesai" and "Gallo" are names, "HSAI" and "GGAL" tickers


def pin_search_news(search_news: Callable[..., dict[str, Any]], *, stock: Callable[[], str | None]) -> Callable[..., dict[str, Any]]:
    """`search_news` with the same name, docstring and signature, whose `symbols` keeps its tickers and drops the rest.

    A symbol list with no ticker in it (empty, a company name, a lower-case ticker) searches `stock()` instead; a name among
    tickers is dropped. Another stock's ticker (a competitor, SPY) is left alone. With no stock being researched the call
    is passed on unchanged. A rewritten call's result gets `symbols_searched`, so the model sees what was searched.
    """

    def pinned(symbols: str = "", start: str | None = None, end: str | None = None, limit: int = 10, include_content: bool = False) -> dict[str, Any]:
        current = stock()
        searched = symbols
        if current is not None:
            parts = [part.strip() for part in symbols.split(",") if part.strip()]
            tickers = [part for part in parts if _TICKER.fullmatch(part)]
            if len(tickers) != len(parts) or not tickers:
                searched = ",".join(tickers) or current
        result = search_news(searched, start, end, limit, include_content)
        if searched != symbols and "error" not in result:
            return {**result, "symbols_searched": searched}
        return result

    pinned.__name__ = search_news.__name__
    pinned.__doc__ = search_news.__doc__
    return pinned

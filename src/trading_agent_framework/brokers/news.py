"""Broker-agnostic news seam: what `news_tools` needs from a broker. No `alpaca` import, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol


class NewsProvider(Protocol):
    """A source of lean news articles (`id`, `headline`, `summary`, `source`, `created_at`, `symbols`, optional `content`).

    `sort` orders the articles by `created_at` BEFORE `limit` cuts the list: `None` is the provider's default
    (Alpaca: newest first), `"asc"` oldest first, `"desc"` newest first. A caller that needs the first articles after
    a moment in a busy window asks for `"asc"`: the default would drop exactly those once the window holds more than `limit`.
    """

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
        sort: str | None = None,
    ) -> list[dict[str, object]]: ...

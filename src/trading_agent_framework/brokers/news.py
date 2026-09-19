"""Broker-agnostic news seam: what `news_tools` needs from a broker. No `alpaca` import, no I/O."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol


class NewsProvider(Protocol):
    """A source of lean news articles (`id`, `headline`, `summary`, `source`, `created_at`, `symbols`, optional `content`)."""

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]: ...

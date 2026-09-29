"""Pure headline trimming: what the agents see of a symbol's news in a setup or trade row.

Headlines are sent to the local LLM on every run, so they are kept lean on purpose: no summary or body
(the agent can call `search_news` for more), duplicates removed, each capped in length.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_MAX_HEADLINE_CHARS = 200  # token budget: one headline never costs more than ~50 tokens


def lean_headlines(articles: Sequence[Mapping[str, object]], limit: int) -> list[dict[str, str]]:
    """The `limit` newest distinct headlines (case-insensitive), as `{headline, created_at, source}`."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    # Newest first; `created_at` is compared as text, which is chronological for the ISO-8601 timestamps
    # the news providers return. The same story syndicated by several sources is kept once.
    for article in sorted(articles, key=lambda a: str(a.get("created_at", "")), reverse=True):
        headline = str(article.get("headline") or "").strip()
        key = headline.casefold()
        if not headline or key in seen:
            continue
        seen.add(key)
        rows.append({"headline": headline[:_MAX_HEADLINE_CHARS], "created_at": str(article.get("created_at", "")), "source": str(article.get("source", ""))})
        if len(rows) >= limit:
            break
    return rows

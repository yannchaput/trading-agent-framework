"""Historical news for backtests from the GDELT GKG cache (`cache/gdelt/gkg_stockmarket/`), behind `NewsProvider`.

Alpaca's news archive starts on 2026-04-29, so a backtest simulating an earlier date gets nothing from it. This
module answers those dates from GDELT's cached stock-market articles (title, url, domain only: no summary, no body; each article is marked `titles_only`).
Every article is stamped with the earliest time GDELT knew it (`stamp`), so cutting the window at `end` keeps the
no-look-ahead guarantee.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import pyarrow.parquet as pq

from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.utils.errors import BrokerError

GDELT_CUTOFF = datetime(2026, 4, 29, tzinfo=UTC)  # Alpaca's news archive starts here (its older Benzinga history is gone)
DEFAULT_ROOT = Path("cache/gdelt/gkg_stockmarket")
DEFAULT_NAMES = Path("cache/sec/company_tickers.json")
MAX_WINDOW_DAYS = 31  # a window reads one file per day: this keeps one search under a second and inside the 64-day cache
_STAMP_FORMAT = "%Y%m%d%H%M%S"
_ID_TOKEN = re.compile(r"\d{5,}")  # article ids and dates inside a URL slug
_FILE_EXTENSION = re.compile(r"\.(html?|php|aspx?)$", re.I)
# Broad ETFs have no company name: the market words their news uses instead.
ALIASES = {"SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "QQQ": "Nasdaq", "DIA": "Dow"}
_LEGAL_SUFFIX = re.compile(r"[,\s]+(inc|corp|corporation|co|company|ltd|plc|p\.l\.c|holdings|group|class [abc]|n\.v|s\.a|/de/)\.?$", re.I)
_MIN_NAME_LENGTH = 4  # shorter names ("BP") hit unrelated words: those match by ticker tag only
# A title about the US market as a whole (no symbols asked): market words, and not a single-company item
# (a `(NASDAQ:XYZ)` tag). The filter of the news-sentiment study, `scripts/news/early_regime_spike.py`.
_US_MARKET = re.compile(r"wall street|\bs ?& ?p\b|\bs&p 500\b|\bdow\b|\bnasdaq\b|\bnyse\b|russell 2000|\bu\.?s\.? stocks|american stocks", re.I)
_TICKER_TAG = re.compile(r"\(\s*[A-Z][A-Z ]+:\s*[A-Z0-9.\-]+\s*\)", re.I)


def _separators(segment: str) -> int:
    """How many `-`/`_` a URL path segment holds: the slug is the segment with the most."""
    return len(re.findall(r"[-_]", segment))


def slug_title(url: str | None) -> str | None:
    """A readable title from a URL's slug, for articles GDELT has no title for (every one before 2019-09-23).

    The slug is the path segment with the most `-`/`_` separators. It loses its file extension, is split on the
    separators, and drops tokens of 5+ digits (ids and dates; `430` stays).

    Args:
        url: The article URL, or None.

    Returns:
        The words joined by single spaces, or None when the URL has no word-like segment.
    """
    if not isinstance(url, str) or not url:  # pandas hands a missing url over as NaN
        return None
    slug = max(urlparse(url).path.split("/"), key=_separators)
    if _separators(slug) == 0:
        return None
    words = [word for word in re.split(r"[-_]+", _FILE_EXTENSION.sub("", slug)) if word and not _ID_TOKEN.fullmatch(word)]
    return " ".join(words) or None


def short_name(sec_title: str) -> str | None:
    """A company's everyday name from SEC's registered title: legal suffixes and a leading "The" dropped.

    Args:
        sec_title: SEC's `title`, e.g. `Apple Inc.` or `Super Micro Computer, Inc.`.

    Returns:
        `Apple`, `Super Micro Computer`...; None when the result is under 4 characters (too ambiguous for a name match).
    """
    name = sec_title.strip()
    while (stripped := _LEGAL_SUFFIX.sub("", name)) != name:
        name = stripped
    name = re.sub(r"^the\s+", "", name, flags=re.I)
    return name if len(name) >= _MIN_NAME_LENGTH else None


def _load_names(path: Path) -> dict[str, str]:
    """`{ticker: short name}` from SEC's `company_tickers.json`; empty when the file is missing (ticker tags and aliases still work)."""
    if not path.exists():
        return {}
    entries = json.loads(path.read_text()).values()
    names = {entry["ticker"].upper(): short_name(entry["title"]) for entry in entries}
    return {ticker: name for ticker, name in names.items() if name}


@lru_cache(maxsize=64)
def _day_frame(root: Path, day: date) -> pd.DataFrame:
    """One cached UTC day: columns `stamp, domain, url, title`, titles filled from URLs, rows without a title dropped.

    Cached because a backtest asks for the same days many times. A missing day file (GDELT has gaps) gives an empty frame,
    and so does a day file without a `stamp` column (the cache holds no-column files for 2025-06-15..07-01).
    """
    path = root / f"{day:%Y}" / f"{day:%Y-%m-%d}.parquet"
    if not path.exists() or "stamp" not in pq.read_schema(path).names:
        return pd.DataFrame(columns=["stamp", "domain", "url", "title"])
    frame = pd.read_parquet(path, columns=["stamp", "domain", "url", "title"])
    frame["title"] = frame["title"].where(frame["title"].notna(), frame["url"].map(slug_title))
    return frame.dropna(subset=["title"])


class GdeltNewsProvider:
    """A `NewsProvider` reading the GDELT GKG day files; the 15-minute `stamp` of an article is its `created_at`."""

    def __init__(self, root: Path = DEFAULT_ROOT, names: Path = DEFAULT_NAMES) -> None:
        """Create a provider.

        Args:
            root: The cache directory holding `<YYYY>/<YYYY-MM-DD>.parquet` day files (UTC days).
            names: SEC's `company_tickers.json`, to find a company by name as well as by ticker tag. These are
                today's names: an older article under a former name matches by ticker tag only.
        """
        self._root = root
        self._names = _load_names(names)

    def _patterns(self, symbols: Sequence[str]) -> dict[str, re.Pattern[str]]:
        """Per ticker, what a matching title holds: its `(EXCHANGE:TICKER)` tag, or its alias / company name as whole words."""
        patterns: dict[str, re.Pattern[str]] = {}
        for ticker in (symbol.strip().upper() for symbol in symbols):
            parts = [rf"\(\s*[A-Z][A-Z ]*:\s*{re.escape(ticker)}\s*\)"]
            name = ALIASES.get(ticker) or self._names.get(ticker)
            if name:
                parts.append(rf"(?i:\b{re.escape(name)}\b)")
            patterns[ticker] = re.compile("|".join(parts))
        return patterns

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
        sort: str | None = None,
    ) -> list[dict[str, object]]:
        """Articles stamped within `[start, end]`, identical titles collapsed to the earliest.

        Args:
            symbols: Tickers to filter on; empty means the US-market news without single-company items.
            start: Earliest stamp; None reads only `end`'s UTC day. A window wider than `MAX_WINDOW_DAYS` is cut to
                the `MAX_WINDOW_DAYS` days before `end`.
            end: Latest stamp (never exceeded, so no look-ahead).
            limit: Maximum number of articles.
            include_content: Ignored: GDELT has titles only.
            sort: `"asc"` for oldest first, else newest first.

        Returns:
            One Alpaca-shaped dict per article (`id`, `headline`, `source`, `created_at`, `symbols`, `titles_only`: no summary).

        Raises:
            BrokerError: If the cache directory does not exist.
        """
        if not self._root.is_dir():
            raise BrokerError(f"GDELT news cache not found at {self._root}")
        end_utc = end.astimezone(UTC)
        start_utc = start.astimezone(UTC) if start else end_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        # shortcut: a wider window is cut to the last MAX_WINDOW_DAYS, so `sort="asc"` gives the oldest of those, not
        # the oldest overall; upgrade (read newest days first, stop at `limit`) if a strategy needs older news.
        start_utc = max(start_utc, end_utc - timedelta(days=MAX_WINDOW_DAYS))
        days = [start_utc.date() + timedelta(days=i) for i in range((end_utc.date() - start_utc.date()).days + 1)]
        frames = [_day_frame(self._root, day) for day in days]
        articles = pd.concat(frames, ignore_index=True).sort_values("stamp", kind="mergesort")
        in_window = (articles["stamp"] >= f"{start_utc:{_STAMP_FORMAT}}") & (articles["stamp"] <= f"{end_utc:{_STAMP_FORMAT}}")
        articles = articles.loc[in_window]
        patterns = self._patterns(symbols)
        if patterns:
            articles = articles.loc[articles["title"].map(lambda title: any(p.search(title) for p in patterns.values()))]
        else:
            articles = articles.loc[articles["title"].map(lambda title: bool(_US_MARKET.search(title)) and not _TICKER_TAG.search(title))]
        articles = articles.drop_duplicates("title")  # sorted by stamp, so the earliest copy stays
        articles = articles.iloc[::-1] if sort != "asc" else articles
        found = articles.head(limit).to_dict("records")
        return [_article(row, [ticker for ticker, p in patterns.items() if p.search(row["title"])]) for row in found]


def _article(row: dict[str, str], symbols: list[str]) -> dict[str, object]:
    """One cache row in Alpaca's article shape, without a summary and marked `titles_only`; `symbols` are the requested tickers its title matched.

    The marker is how the source says it has titles only: `search_news` turns it into an instruction for the model
    and removes it from the facts it hands over.
    """
    created = datetime.strptime(row["stamp"], _STAMP_FORMAT).replace(tzinfo=UTC)
    return {"id": row["url"], "headline": row["title"], "source": row["domain"], "created_at": created.isoformat(), "symbols": symbols, "titles_only": True}


class CutoffNewsProvider:
    """A `NewsProvider` answering from `before` until the cutoff and from `after` from the cutoff on.

    A window spanning the cutoff asks both and merges, so no article is dropped or counted twice: `before` gets the
    part up to one microsecond short of the cutoff, `after` the part from the cutoff.
    """

    def __init__(self, before: NewsProvider, after: Callable[[], NewsProvider], cutoff: datetime = GDELT_CUTOFF) -> None:
        """Create a router.

        Args:
            before: Answers windows before the cutoff (GDELT).
            after: Factory of the provider for the cutoff on (Alpaca), called once when first needed, so a backtest
                that stays before the cutoff needs no news credentials.
            cutoff: The first instant `after` answers for.
        """
        self._before = before
        self._make_after = after
        self._after: NewsProvider | None = None
        self._cutoff = cutoff

    def _after_provider(self) -> NewsProvider:
        """The `after` provider, built on first use."""
        if self._after is None:
            self._after = self._make_after()
        return self._after

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
        sort: str | None = None,
    ) -> list[dict[str, object]]:
        """Articles from whichever provider(s) cover `[start, end]`; arguments as `NewsProvider.get_news`."""
        ask = {"limit": limit, "include_content": include_content, "sort": sort}
        if end < self._cutoff:
            return self._before.get_news(symbols, start=start, end=end, **ask)
        if start is None or start >= self._cutoff:
            return self._after_provider().get_news(symbols, start=start, end=end, **ask)
        older = self._before.get_news(symbols, start=start, end=self._cutoff - timedelta(microseconds=1), **ask)
        newer = self._after_provider().get_news(symbols, start=self._cutoff, end=end, **ask)
        merged = sorted(older + newer, key=lambda article: datetime.fromisoformat(str(article["created_at"])), reverse=sort != "asc")
        return merged[:limit]

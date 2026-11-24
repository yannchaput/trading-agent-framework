"""Tests for `GdeltNewsProvider`: tiny parquet day files and SEC name files in `tmp_path`, never the real cache."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trading_agent_framework.backtesting.news import GdeltNewsProvider, short_name, slug_title
from trading_agent_framework.utils.errors import BrokerError

_ET = ZoneInfo("America/New_York")


def _day(root: Path, day: str, rows: list[dict[str, str | None]]) -> None:
    """Write `root/<YYYY>/<day>.parquet` with the cache's columns (`title` may be None)."""
    path = root / day[:4] / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["stamp", "domain", "url", "title"]).astype("object")
    frame.to_parquet(path)


def _row(stamp: str, title: str | None, url: str | None = "http://x.com/a-b", domain: str = "x.com") -> dict[str, str | None]:
    return {"stamp": stamp, "domain": domain, "url": url, "title": title}


def _names(path: Path, names: dict[str, str]) -> Path:
    """Write SEC's `company_tickers.json` shape."""
    path.write_text(json.dumps({str(i): {"cik_str": i, "ticker": t, "title": n} for i, (t, n) in enumerate(names.items())}))
    return path


def _provider(tmp_path: Path, names: dict[str, str] | None = None) -> GdeltNewsProvider:
    return GdeltNewsProvider(root=tmp_path / "gkg", names=_names(tmp_path / "names.json", names or {}))


def _headlines(articles: list[dict[str, object]]) -> list[object]:
    return [a["headline"] for a in articles]


def test_slug_title_uses_the_segment_with_most_separators_and_drops_long_ids() -> None:
    url = "http://www.ptinews.com/news/9542091_Mkt-takes-U-turn--Sensex-tumbles-430-pts-as-bank-stocks-drag.html"
    assert slug_title(url) == "Mkt takes U turn Sensex tumbles 430 pts as bank stocks drag"
    assert slug_title("http://x.com/news/ap-top-stories/republicans-want-trump-to-back-off/711910290") == "republicans want trump to back off"
    assert slug_title("https://x.com/story/5267827/?cs=233") is None
    assert slug_title(None) is None


def test_a_null_title_is_filled_from_the_url_and_a_row_with_neither_is_dropped(tmp_path: Path) -> None:
    _day(
        tmp_path / "gkg",
        "2018-03-06",
        [
            _row("20180306100000", None, "http://x.com/wall-street-rallies-on-jobs/123456"),
            _row("20180306110000", None, None),
            _row("20180306120000", None, "http://x.com/story/5267827/"),
        ],
    )
    articles = _provider(tmp_path).get_news(end=datetime(2018, 3, 6, 23, tzinfo=UTC))
    assert _headlines(articles) == ["wall street rallies on jobs"]


def test_only_rows_inside_the_window_are_returned(tmp_path: Path) -> None:
    _day(
        tmp_path / "gkg",
        "2024-03-05",
        [
            _row("20240305093000", "Wall Street early"),
            _row("20240305150000", "Wall Street late"),
            _row("20240305160001", "Wall Street after"),
        ],
    )
    articles = _provider(tmp_path).get_news(start=datetime(2024, 3, 5, 9, tzinfo=UTC), end=datetime(2024, 3, 5, 16, tzinfo=UTC))
    assert _headlines(articles) == ["Wall Street late", "Wall Street early"]


def test_an_et_window_reads_the_next_utc_day_file(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-03-06", [_row("20240306010000", "Wall Street evening")])
    articles = _provider(tmp_path).get_news(
        start=datetime(2024, 3, 5, 19, tzinfo=_ET),
        end=datetime(2024, 3, 5, 21, tzinfo=_ET),
    )
    assert _headlines(articles) == ["Wall Street evening"]


def test_start_none_reads_only_the_end_day(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-03-04", [_row("20240304100000", "Wall Street yesterday")])
    _day(tmp_path / "gkg", "2024-03-05", [_row("20240305100000", "Wall Street today")])
    articles = _provider(tmp_path).get_news(end=datetime(2024, 3, 5, 23, tzinfo=UTC))
    assert _headlines(articles) == ["Wall Street today"]


def test_missing_day_files_and_empty_files_are_skipped(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-03-04", [])
    _day(tmp_path / "gkg", "2024-03-06", [_row("20240306100000", "Wall Street ok")])
    articles = _provider(tmp_path).get_news(start=datetime(2024, 3, 4, tzinfo=UTC), end=datetime(2024, 3, 6, 23, tzinfo=UTC))
    assert _headlines(articles) == ["Wall Street ok"]


def test_a_missing_cache_root_raises_broker_error(tmp_path: Path) -> None:
    with pytest.raises(BrokerError, match="GDELT news cache not found"):
        GdeltNewsProvider(root=tmp_path / "nope").get_news(end=datetime(2024, 3, 5, tzinfo=UTC))


def test_identical_titles_keep_the_earliest_then_newest_first_and_limit(tmp_path: Path) -> None:
    _day(
        tmp_path / "gkg",
        "2024-03-05",
        [
            _row("20240305100000", "Wall Street same"),
            _row("20240305110000", "Wall Street same"),
            _row("20240305120000", "Wall Street b"),
            _row("20240305130000", "Wall Street c"),
        ],
    )
    provider = _provider(tmp_path)
    end = datetime(2024, 3, 5, 23, tzinfo=UTC)
    assert _headlines(provider.get_news(end=end, limit=2)) == ["Wall Street c", "Wall Street b"]
    everything = provider.get_news(end=end)
    assert _headlines(everything) == ["Wall Street c", "Wall Street b", "Wall Street same"]
    assert everything[-1]["created_at"] == "2024-03-05T10:00:00+00:00"


def test_sort_asc_returns_oldest_first(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-03-05", [_row("20240305100000", "Wall Street a"), _row("20240305110000", "Wall Street b")])
    articles = _provider(tmp_path).get_news(end=datetime(2024, 3, 5, 23, tzinfo=UTC), sort="asc")
    assert _headlines(articles) == ["Wall Street a", "Wall Street b"]


def test_article_shape_matches_alpaca(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-03-05", [_row("20240305093000", "Wall Street up", "http://x.com/wall", "x.com")])
    article = _provider(tmp_path).get_news(end=datetime(2024, 3, 5, 23, tzinfo=UTC), include_content=True)[0]
    assert article == {
        "id": "http://x.com/wall",
        "headline": "Wall Street up",
        "summary": "",
        "source": "x.com",
        "created_at": "2024-03-05T09:30:00+00:00",
        "symbols": [],
    }


_END = datetime(2024, 3, 5, 23, tzinfo=UTC)


def _titles_day(tmp_path: Path, titles: list[str]) -> None:
    """One 2024-03-05 day file with `titles` stamped 10:00, 10:01, ..."""
    _day(tmp_path / "gkg", "2024-03-05", [_row(f"202403051{i:02d}000", title) for i, title in enumerate(titles)])


def test_short_name_drops_legal_suffixes_and_refuses_short_names() -> None:
    assert short_name("NVIDIA CORP") == "NVIDIA"
    assert short_name("Apple Inc.") == "Apple"
    assert short_name("Apple Inc. /DE/") == "Apple"
    assert short_name("Alphabet Inc.") == "Alphabet"
    assert short_name("Super Micro Computer, Inc.") == "Super Micro Computer"
    assert short_name("BP p.l.c.") is None  # under 4 characters


def test_a_symbol_matches_its_ticker_tag_or_its_name_case_insensitive(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["Why Nvidia Stock Was Popping", "Fund buys more (NASDAQ:NVDA) shares", "Apple fined", "Cooking tips"])
    articles = _provider(tmp_path, {"NVDA": "NVIDIA CORP", "AAPL": "Apple Inc."}).get_news(["NVDA"], end=_END)
    assert _headlines(articles) == ["Fund buys more (NASDAQ:NVDA) shares", "Why Nvidia Stock Was Popping"]
    assert all(a["symbols"] == ["NVDA"] for a in articles)


def test_a_name_matches_whole_words_only(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["Pineapple prices rise", "Apple launches a phone"])
    articles = _provider(tmp_path, {"AAPL": "Apple Inc."}).get_news(["AAPL"], end=_END)
    assert _headlines(articles) == ["Apple launches a phone"]


def test_a_short_name_matches_by_ticker_tag_only(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["BP sees profit", "Shares rise (NYSE:BP)"])
    articles = _provider(tmp_path, {"BP": "BP p.l.c."}).get_news(["BP"], end=_END)
    assert _headlines(articles) == ["Shares rise (NYSE:BP)"]


def test_aliases_cover_broad_etfs(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["S&P 500 hits a record", "Gold slips"])
    articles = _provider(tmp_path).get_news(["SPY"], end=_END)
    assert _headlines(articles) == ["S&P 500 hits a record"]
    assert articles[0]["symbols"] == ["SPY"]


def test_an_article_lists_every_requested_symbol_it_matched(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["Nvidia and Apple rally", "Apple alone"])
    articles = _provider(tmp_path, {"NVDA": "NVIDIA CORP", "AAPL": "Apple Inc."}).get_news(["NVDA", "AAPL"], end=_END, sort="asc")
    assert [a["symbols"] for a in articles] == [["NVDA", "AAPL"], ["AAPL"]]


def test_without_symbols_only_us_market_titles_without_ticker_tags(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["Wall Street closes higher", "Nasdaq falls - NVIDIA (NASDAQ:NVDA)", "2024 Subaru Solterra review"])
    assert _headlines(_provider(tmp_path).get_news(end=_END)) == ["Wall Street closes higher"]


def test_a_missing_names_file_falls_back_to_ticker_tags_and_aliases(tmp_path: Path) -> None:
    _titles_day(tmp_path, ["Nvidia rallies", "Fund buys (NASDAQ:NVDA)", "S&P 500 up"])
    provider = GdeltNewsProvider(root=tmp_path / "gkg", names=tmp_path / "missing.json")
    assert _headlines(provider.get_news(["NVDA", "SPY"], end=_END, sort="asc")) == ["Fund buys (NASDAQ:NVDA)", "S&P 500 up"]


def test_a_window_wider_than_31_days_is_cut_to_the_31_days_before_end(tmp_path: Path) -> None:
    _day(tmp_path / "gkg", "2024-01-25", [_row("20240125100000", "Wall Street forty days ago")])
    _day(tmp_path / "gkg", "2024-03-01", [_row("20240301100000", "Wall Street four days ago")])
    articles = _provider(tmp_path).get_news(start=datetime(2023, 3, 5, tzinfo=UTC), end=datetime(2024, 3, 5, 12, tzinfo=UTC), sort="asc")
    assert _headlines(articles) == ["Wall Street four days ago"]

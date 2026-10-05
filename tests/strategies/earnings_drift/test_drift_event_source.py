from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.strategies.earnings_drift.event_source import EventSource, LoadReport

_TICKERS = {"0": {"cik_str": 29989, "ticker": "OMC"}, "1": {"cik_str": 320193, "ticker": "AAPL"}}
_RECENT = {
    "filings": {
        "recent": {
            "form": ["8-K"],
            "items": ["2.02,9.01"],
            "acceptanceDateTime": ["2026-07-28T20:07:49.000Z"],
            "accessionNumber": ["acc-new"],
            "primaryDocument": ["new.htm"],
        },
        "files": [{"name": "CIK0000029989-submissions-001.json", "filingFrom": "2016-01-01", "filingTo": "2026-01-31"}],
    }
}
_PAGE = {"form": ["8-K"], "items": ["2.02"], "acceptanceDateTime": ["2026-01-20T21:01:00.000Z"], "accessionNumber": ["acc-old"], "primaryDocument": ["old.htm"]}


def _source(tmp_path: Path, requests: list[str], *, reload_every_cycle: bool = False) -> EventSource:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json=_TICKERS)
        if request.url.path == "/submissions/CIK0000029989.json":
            return httpx.Response(200, json=_RECENT)
        if request.url.path == "/submissions/CIK0000029989-submissions-001.json":
            return httpx.Response(200, json=_PAGE)
        return httpx.Response(404)

    client = SecEdgarClient("TestApp test@example.com", tmp_path, min_request_interval_seconds=0.0, transport=httpx.MockTransport(handler))
    return EventSource(client, reload_every_cycle=reload_every_cycle)


NOW = datetime(2026, 9, 1, 20, 0, tzinfo=UTC)


def test_load_reads_recent_filings_and_the_older_pages_reaching_since(tmp_path: Path) -> None:
    requests: list[str] = []
    source = _source(tmp_path, requests)
    report = source.load(["OMC"], as_of=NOW, since=date(2025, 12, 1))
    assert report == LoadReport(loaded=1, failed={})
    assert [e.accession_number for e in source.all_events()] == ["acc-old", "acc-new"]
    assert "/submissions/CIK0000029989-submissions-001.json" in requests


def test_an_older_page_before_since_is_not_fetched(tmp_path: Path) -> None:
    requests: list[str] = []
    source = _source(tmp_path, requests)
    source.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert [e.accession_number for e in source.all_events()] == ["acc-new"]
    assert "/submissions/CIK0000029989-submissions-001.json" not in requests


def test_a_failing_symbol_is_reported_not_raised(tmp_path: Path) -> None:
    source = _source(tmp_path, [])
    report = source.load(["OMC", "NOPE", "AAPL"], as_of=NOW, since=date(2026, 6, 1))
    assert report.loaded == 1
    assert set(report.failed) == {"NOPE", "AAPL"}  # unknown ticker; AAPL's submissions answer 404


def test_needs_load_once_in_a_backtest_every_cycle_live_and_after_discard(tmp_path: Path) -> None:
    backtest = _source(tmp_path / "bt", [])
    assert backtest.needs_load()
    backtest.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert not backtest.needs_load()
    backtest.discard()
    assert backtest.needs_load() and backtest.all_events() == []
    live = _source(tmp_path / "live", [], reload_every_cycle=True)
    live.load(["OMC"], as_of=NOW, since=date(2026, 6, 1))
    assert live.needs_load()


@pytest.mark.parametrize(
    ("loaded", "failed", "hollow"),
    [(10, 25, True), (10, 19, False), (35, 25, False), (0, 0, False)],
)
def test_is_hollow(loaded: int, failed: int, hollow: bool) -> None:
    report = LoadReport(loaded=loaded, failed={f"S{i}": "error" for i in range(failed)})
    assert report.is_hollow(0.5, 20) is hollow


def test_a_cached_submissions_file_is_refetched_only_for_an_as_of_after_its_fetch_time(tmp_path: Path) -> None:
    """`max_age_days=0`: a file is fresh for every `as_of` up to its fetch time, stale after it."""
    requests: list[str] = []
    source = _source(tmp_path, requests)
    submissions = "/submissions/CIK0000029989.json"
    source.load(["OMC"], as_of=datetime(2026, 5, 1, tzinfo=UTC), since=date(2026, 6, 1))
    assert requests.count(submissions) == 1
    (cached,) = tmp_path.rglob("CIK0000029989.json")
    fetched_at = datetime(2026, 6, 1, 12, 0, tzinfo=UTC).timestamp()
    os.utime(cached, (fetched_at, fetched_at))  # as if fetched on 2026-06-01
    source.load(["OMC"], as_of=datetime(2026, 5, 15, tzinfo=UTC), since=date(2026, 6, 1))
    source.load(["OMC"], as_of=datetime(2026, 6, 1, 11, 0, tzinfo=UTC), since=date(2026, 6, 1))
    assert requests.count(submissions) == 1  # an as_of before the fetch: served from the cache
    source.load(["OMC"], as_of=datetime(2026, 9, 23, tzinfo=UTC), since=date(2026, 6, 1))
    assert requests.count(submissions) == 2  # an as_of after the fetch: the 8-Ks filed since may be missing, fetched again

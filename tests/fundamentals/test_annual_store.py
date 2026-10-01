from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.annual_store import AnnualFiguresStore
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import FundamentalsError

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL"}}
FACTS = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"val": 100, "start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"}]}}}}}
FETCHED = datetime(2026, 10, 2, tzinfo=UTC)
LATER = datetime(2026, 11, 15, tzinfo=UTC)  # more than 30 days after FETCHED


class FakeSec:
    """A fake SEC: records every request and answers each endpoint with a settable status."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.facts_status = 200
        self.submissions_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "/companyfacts/" in url:
            return httpx.Response(200, json=FACTS) if self.facts_status == 200 else httpx.Response(self.facts_status)
        if "/submissions/" in url:
            return httpx.Response(200, json={"sic": "3571"}) if self.submissions_status == 200 else httpx.Response(self.submissions_status)
        return httpx.Response(404)

    def count(self, fragment: str) -> int:
        return sum(fragment in url for url in self.requests)


def _store(tmp_path: Path, sec: FakeSec) -> AnnualFiguresStore:
    client = SecEdgarClient("TestApp test@example.com", tmp_path / "sec", min_request_interval_seconds=0.0, transport=httpx.MockTransport(sec))
    return AnnualFiguresStore(client, tmp_path / "sec" / "annual", wall_clock=lambda: FETCHED)


def _figures(store: AnnualFiguresStore, symbol: str = "AAPL", as_of: datetime = FETCHED) -> dict[str, object] | None:
    return store.figures(symbol, as_of=as_of, max_age_days=30)


def _sic(store: AnnualFiguresStore, as_of: datetime = FETCHED) -> int | None:
    return store.sic("AAPL", as_of=as_of, max_age_days=30)


def _record_file(tmp_path: Path) -> Path:
    return tmp_path / "sec" / "annual" / "CIK0000320193.json"


def test_the_first_call_fetches_and_writes_the_reduced_record_only(tmp_path: Path) -> None:
    sec = FakeSec()

    record = _figures(_store(tmp_path, sec))

    assert record is not None
    assert record["status"] == "ok"
    assert record["cik"] == "0000320193"
    assert record["fetched_at"] == FETCHED.isoformat()
    assert [row["value"] for row in record["flows"]] == [100]  # ty: ignore[not-iterable]
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8")) == record
    assert not (tmp_path / "sec" / "companyfacts").exists()  # the 4 MB raw payload is not kept


def test_a_second_call_makes_no_request_in_memory_or_from_disk(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store)
    _figures(store)
    _figures(_store(tmp_path, sec))  # a new process reads the file

    assert sec.count("/companyfacts/") == 1


def test_a_record_fetched_today_serves_every_past_date(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store, as_of=datetime(2024, 1, 2, tzinfo=UTC))
    _figures(store, as_of=datetime(2025, 6, 2, tzinfo=UTC))

    assert sec.count("/companyfacts/") == 1


def test_a_stale_record_is_refetched(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    _figures(store)
    _figures(store, as_of=LATER)

    assert sec.count("/companyfacts/") == 2


def test_a_404_is_cached_as_an_absent_company(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 404
    store = _store(tmp_path, sec)

    assert _figures(store) is None
    assert _figures(store) is None
    assert sec.count("/companyfacts/") == 1
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))["status"] == "absent"


def test_a_transport_error_raises_and_is_not_cached(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 500
    store = _store(tmp_path, sec)

    with pytest.raises(FundamentalsError):
        _figures(store)
    assert not _record_file(tmp_path).exists()

    sec.facts_status = 200
    assert _figures(store) is not None


def test_a_stale_record_survives_a_failed_refetch_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    _figures(store)
    sec.facts_status = 500

    with caplog.at_level(logging.WARNING):
        record = _figures(store, as_of=LATER)

    assert record is not None
    assert record["status"] == "ok"
    assert "0000320193" in caplog.text


def test_a_corrupt_record_file_is_refetched(tmp_path: Path) -> None:
    sec = FakeSec()
    _record_file(tmp_path).parent.mkdir(parents=True)
    _record_file(tmp_path).write_text("{not json", encoding="utf-8")

    record = _figures(_store(tmp_path, sec))

    assert record is not None
    assert sec.count("/companyfacts/") == 1


def test_a_ticker_sec_does_not_know_is_absent_without_a_request_or_a_file(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert store.cik("ZZZZ") is None
    assert _figures(store, "ZZZZ") is None
    assert sec.count("/companyfacts/") == 0
    assert not (tmp_path / "sec" / "annual").exists()


def test_the_ticker_map_is_requested_once(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert store.cik("AAPL") == "0000320193"
    assert store.cik("AAPL") == "0000320193"
    assert sec.count("company_tickers.json") == 1


def test_the_sic_code_is_fetched_once_and_saved_in_the_record(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)

    assert _sic(store) == 3571
    assert _sic(store) == 3571
    assert _sic(_store(tmp_path, sec)) == 3571

    assert sec.count("/submissions/") == 1
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))["sic"] == 3571
    assert not (tmp_path / "sec" / "submissions").exists()


def test_the_sic_code_survives_a_refresh_of_the_record(tmp_path: Path) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    _sic(store)

    assert _sic(store, as_of=LATER) == 3571
    assert sec.count("/companyfacts/") == 2
    assert sec.count("/submissions/") == 1


def test_a_company_with_no_submissions_has_no_sic_and_is_not_asked_again(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.submissions_status = 404
    store = _store(tmp_path, sec)

    assert _sic(store) is None
    assert _sic(store) is None
    assert sec.count("/submissions/") == 1


def test_a_transport_error_on_the_sic_lookup_raises_and_is_retried(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.submissions_status = 500
    store = _store(tmp_path, sec)

    with pytest.raises(FundamentalsError):
        _sic(store)

    sec.submissions_status = 200
    assert _sic(store) == 3571


def test_the_sic_of_an_absent_company_is_none_without_a_request(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 404
    store = _store(tmp_path, sec)

    assert _sic(store) is None
    assert sec.count("/submissions/") == 0

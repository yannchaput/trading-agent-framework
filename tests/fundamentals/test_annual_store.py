from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals import annual_store
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
        self.tickers: dict[str, dict[str, object]] = TICKERS

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=self.tickers)
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


def _valid_record() -> dict[str, object]:
    return {"cik": "0000320193", "fetched_at": FETCHED.isoformat(), "status": "ok", "flows": [], "balances": [], "shares": []}


def _without(key: str) -> dict[str, object]:
    record = _valid_record()
    del record[key]
    return record


@pytest.mark.parametrize(
    "record",
    [
        _without("fetched_at"),
        {**_valid_record(), "fetched_at": "not a date"},
        {**_valid_record(), "fetched_at": "2026-10-02T00:00:00"},
        {**_valid_record(), "fetched_at": 20261002},
        _without("flows"),
        {**_valid_record(), "flows": {"value": 1}},
        {**_valid_record(), "status": "weird"},
        {**_valid_record(), "cik": "0000000001"},
        _without("cik"),
    ],
    ids=["no-fetched-at", "unparseable-fetched-at", "naive-fetched-at", "non-string-fetched-at", "no-flows", "flows-not-a-list", "bad-status", "other-cik", "no-cik"],
)
def test_a_record_file_of_the_wrong_shape_is_a_cache_miss(tmp_path: Path, record: dict[str, object]) -> None:
    sec = FakeSec()
    _record_file(tmp_path).parent.mkdir(parents=True)
    _record_file(tmp_path).write_text(json.dumps(record), encoding="utf-8")

    result = _figures(_store(tmp_path, sec))

    assert result is not None
    assert result["status"] == "ok"
    assert [row["value"] for row in result["flows"]] == [100]  # ty: ignore[not-iterable]
    assert sec.count("/companyfacts/") == 1


def test_a_record_file_that_is_not_a_json_object_is_a_cache_miss(tmp_path: Path) -> None:
    sec = FakeSec()
    _record_file(tmp_path).parent.mkdir(parents=True)
    _record_file(tmp_path).write_text("[1, 2]", encoding="utf-8")

    assert _figures(_store(tmp_path, sec)) is not None
    assert sec.count("/companyfacts/") == 1


def test_a_failed_replace_never_escapes_and_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    real_replace = os.replace

    def failing_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name.startswith("CIK"):
            raise FileNotFoundError("the temporary file is already gone")
        real_replace(source, destination)

    monkeypatch.setattr(annual_store.os, "replace", failing_replace)

    with caplog.at_level(logging.WARNING):
        record = _figures(store)
    _figures(store)

    assert record is not None
    assert record["status"] == "ok"
    assert "0000320193" in caplog.text
    assert sec.count("/companyfacts/") == 1  # served from memory
    assert not _record_file(tmp_path).exists()
    assert list(_record_file(tmp_path).parent.glob("*.tmp")) == []


def test_an_unwritable_cache_directory_never_escapes(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sec = FakeSec()
    client = SecEdgarClient("TestApp test@example.com", tmp_path / "sec", min_request_interval_seconds=0.0, transport=httpx.MockTransport(sec))
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a regular file where the cache directory should be", encoding="utf-8")
    store = AnnualFiguresStore(client, blocker, wall_clock=lambda: FETCHED)

    with caplog.at_level(logging.WARNING):
        record = _figures(store)
    _figures(store)

    assert record is not None
    assert record["status"] == "ok"
    assert "could not be saved" in caplog.text
    assert sec.count("/companyfacts/") == 1


def test_a_successful_write_leaves_no_temp_file_and_uses_no_fixed_temp_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    sources: list[str] = []
    real_replace = os.replace

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination).name.startswith("CIK"):
            sources.append(str(source))
        real_replace(source, destination)

    monkeypatch.setattr(annual_store.os, "replace", recording_replace)

    _figures(store)
    _sic(store)  # a second write for the same CIK

    assert len(sources) == 2
    assert sources[0] != sources[1]
    assert list(_record_file(tmp_path).parent.glob("*.tmp")) == []


def test_two_share_classes_of_one_company_share_one_record(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.tickers = {"0": {"cik_str": 320193, "ticker": "GOOGL"}, "1": {"cik_str": 320193, "ticker": "GOOG"}}
    store = _store(tmp_path, sec)

    first = _figures(store, "GOOGL")
    second = _figures(store, "GOOG")

    assert first is not None
    assert first == second
    assert sec.count("/companyfacts/") == 1
    assert store.sic("GOOGL", as_of=FETCHED, max_age_days=30) == 3571
    assert store.sic("GOOG", as_of=FETCHED, max_age_days=30) == 3571
    assert sec.count("/submissions/") == 1


def test_an_absent_company_is_requested_again_only_once_stale(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.facts_status = 404
    store = _store(tmp_path, sec)

    assert _figures(store) is None
    assert _figures(store, as_of=datetime(2026, 10, 20, tzinfo=UTC)) is None  # within 30 days
    assert sec.count("/companyfacts/") == 1

    assert _figures(store, as_of=LATER) is None
    assert sec.count("/companyfacts/") == 2


def test_a_stale_company_that_turns_404_becomes_absent_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sec = FakeSec()
    store = _store(tmp_path, sec)
    assert _figures(store) is not None
    sec.facts_status = 404

    with caplog.at_level(logging.WARNING):
        assert _figures(store, as_of=LATER) is None

    assert "0000320193" in caplog.text
    assert json.loads(_record_file(tmp_path).read_text(encoding="utf-8"))["status"] == "absent"

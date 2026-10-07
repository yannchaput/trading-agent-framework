from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest
from tests.congress.pdfs import make_image_only_pdf, make_text_pdf, make_zip

from trading_agent_framework.congress.clerk_client import ClerkClient, filing_url, index_url
from trading_agent_framework.congress.ptr import FilingRef
from trading_agent_framework.utils.errors import ConfigurationError, CongressDataError, CongressNotFoundError

PTR = FilingRef(doc_id="20026590", member="Nancy Pelosi", kind="ptr", filed=date(2025, 2, 20), year=2025)
ANNUAL = FilingRef(doc_id="10063900", member="Nancy Pelosi", kind="annual", filed=date(2025, 5, 15), year=2024)
XML = "<FinancialDisclosure></FinancialDisclosure>"


def _client(tmp_path: Path, handler) -> ClerkClient:
    return ClerkClient("TestApp test@example.com", tmp_path, min_request_interval_seconds=0.0, transport=httpx.MockTransport(handler))


def _at(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def _age(path: Path, when: datetime) -> None:
    os.utime(path, (when.timestamp(), when.timestamp()))


def test_blank_user_agent_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="CONGRESS_USER_AGENT"):
        ClerkClient("  ", tmp_path)


def test_urls_follow_the_clerk_layout() -> None:
    assert index_url(2025).endswith("/financial-pdfs/2025FD.zip")
    assert filing_url(PTR).endswith("/ptr-pdfs/2025/20026590.pdf")
    assert filing_url(ANNUAL).endswith("/financial-pdfs/2024/10063900.pdf")  # the folder is the index year (the Year field), not the filing year


def test_year_index_is_read_from_the_zip_and_cached(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["User-Agent"] == "TestApp test@example.com"
        return httpx.Response(200, content=make_zip(2025, XML))

    client = _client(tmp_path, handler)

    assert client.year_index_xml(2025, _at(2025, 6, 1)) == XML
    assert client.year_index_xml(2025, _at(2025, 6, 1)) == XML
    assert len(calls) == 1


def test_a_past_year_index_fetched_after_that_year_is_never_refetched(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=make_zip(2024, XML))

    client = _client(tmp_path, handler)
    client.year_index_xml(2024, _at(2026, 3, 1))
    _age(tmp_path / "index" / "2024FD.xml", _at(2026, 3, 1))  # fetched in 2026: the 2024 index is complete

    client.year_index_xml(2024, _at(2026, 12, 31))

    assert len(calls) == 1


def test_the_current_year_index_is_refetched_after_a_day_by_the_callers_clock(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=make_zip(2026, XML))

    client = _client(tmp_path, handler)
    client.year_index_xml(2026, _at(2026, 3, 1))
    _age(tmp_path / "index" / "2026FD.xml", _at(2026, 3, 1))

    client.year_index_xml(2026, _at(2026, 3, 1, 12))  # within a day by the caller's clock
    assert len(calls) == 1
    client.year_index_xml(2026, _at(2026, 3, 3))  # more than a day later
    assert len(calls) == 2


def test_a_backtest_clock_never_refetches_an_index_fetched_today(tmp_path: Path) -> None:
    """In a backtest `as_of` is simulated and earlier than the file's mtime: the wall clock is never compared with data."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=make_zip(2026, XML))

    client = _client(tmp_path, handler)
    client.year_index_xml(2026, _at(2026, 1, 15))
    client.year_index_xml(2026, _at(2026, 1, 20))

    assert len(calls) == 1


def test_a_stale_index_that_cannot_refresh_is_served_with_a_warning(tmp_path: Path) -> None:
    responses = [httpx.Response(200, content=make_zip(2026, XML)), httpx.Response(500)]
    client = _client(tmp_path, lambda request: responses.pop(0))
    client.year_index_xml(2026, _at(2026, 3, 1))
    _age(tmp_path / "index" / "2026FD.xml", _at(2026, 3, 1))

    assert client.year_index_xml(2026, _at(2026, 3, 5)) == XML


def test_a_missing_index_is_a_not_found_error(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(CongressNotFoundError):
        client.year_index_xml(2027, _at(2027, 1, 2))


def test_a_naive_as_of_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, content=make_zip(2025, XML)))

    with pytest.raises(ValueError, match="timezone-aware"):
        client.year_index_xml(2025, datetime(2025, 6, 1))


def test_filing_text_is_extracted_from_the_pdf_and_cached_forever(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.path.endswith("/ptr-pdfs/2025/20026590.pdf")
        return httpx.Response(200, content=make_text_pdf("Hello PTR"))

    client = _client(tmp_path, handler)

    assert "Hello PTR" in client.filing_text(PTR)
    assert "Hello PTR" in client.filing_text(PTR)
    assert len(calls) == 1


def test_a_yearly_report_is_fetched_from_financial_pdfs(tmp_path: Path) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, content=make_text_pdf("Hello annual"))

    _client(tmp_path, handler).filing_text(ANNUAL)

    assert seen == ["/public_disc/financial-pdfs/2024/10063900.pdf"]


def test_an_image_only_pdf_returns_empty_text_and_is_cached(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=make_image_only_pdf())

    client = _client(tmp_path, handler)

    assert client.filing_text(PTR) == ""
    assert client.filing_text(PTR) == ""
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500), httpx.Response(200, content=b"not a pdf at all"), httpx.Response(200, content=b"%PDF-1.4 truncated")],
)
def test_failures_are_wrapped_in_congress_data_error(tmp_path: Path, response: httpx.Response) -> None:
    client = _client(tmp_path, lambda request: response)

    with pytest.raises(CongressDataError):
        client.filing_text(PTR)

    assert not list(tmp_path.rglob("*.txt"))  # a failure caches nothing


def test_a_missing_filing_is_a_not_found_error(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(CongressNotFoundError):
        client.filing_text(PTR)


@pytest.mark.parametrize("content", [b"not a zip", make_zip(2025, XML)[:40]])
def test_a_broken_index_archive_is_a_congress_data_error(tmp_path: Path, content: bytes) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, content=content))

    with pytest.raises(CongressDataError):
        client.year_index_xml(2025, _at(2025, 6, 1))


def test_transport_errors_are_wrapped(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(CongressDataError):
        _client(tmp_path, handler).year_index_xml(2025, _at(2025, 6, 1))

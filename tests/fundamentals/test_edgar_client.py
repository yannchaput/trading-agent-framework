from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL"}}


def _client(tmp_path: Path, handler) -> SecEdgarClient:
    transport = httpx.MockTransport(handler)
    return SecEdgarClient("TestApp test@example.com", tmp_path, min_request_interval_seconds=0.0, transport=transport)


def test_blank_user_agent_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="SEC_EDGAR_USER_AGENT"):
        SecEdgarClient("  ", tmp_path)


def test_get_json_fetches_and_caches(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["User-Agent"] == "TestApp test@example.com"
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, handler)

    first = client.get_json("https://data.sec.gov/x.json", ("x.json",))
    second = client.get_json("https://data.sec.gov/x.json", ("x.json",))

    assert first == second == {"ok": True}
    assert len(calls) == 1  # second call was a cache hit
    assert (tmp_path / "x.json").exists()


def test_get_json_wraps_http_errors_as_fundamentals_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _client(tmp_path, handler)

    with pytest.raises(FundamentalsError):
        client.get_json("https://data.sec.gov/missing.json", ("missing.json",))


def test_get_text_fetches_and_caches(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>filing</html>")

    client = _client(tmp_path, handler)

    text = client.get_text("https://www.sec.gov/f.htm", ("filings", "f.htm"))

    assert text == "<html>filing</html>"
    assert (tmp_path / "filings" / "f.htm").exists()


def test_ticker_to_cik_uses_sec_parse_company_tickers(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_TICKERS)

    client = _client(tmp_path, handler)

    assert client.ticker_to_cik("AAPL") == "0000320193"


def test_ticker_to_cik_unknown_symbol_raises_fundamentals_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_TICKERS)

    client = _client(tmp_path, handler)

    with pytest.raises(FundamentalsError, match="ZZZZ"):
        client.ticker_to_cik("ZZZZ")


def test_get_company_facts_payload_hits_the_xbrl_endpoint(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"facts": {}})

    client = _client(tmp_path, handler)

    client.get_company_facts_payload("0000320193")

    assert seen_urls == ["https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]


def test_get_submissions_payload_hits_the_submissions_endpoint(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"filings": {}})

    client = _client(tmp_path, handler)

    client.get_submissions_payload("0000320193")

    assert seen_urls == ["https://data.sec.gov/submissions/CIK0000320193.json"]

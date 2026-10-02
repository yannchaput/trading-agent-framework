from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError

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


def test_get_json_wraps_a_non_json_2xx_body_as_fundamentals_error(tmp_path: Path) -> None:
    """SEC's fair-access throttling sometimes serves an HTML block page with a 2xx status
    instead of JSON; `response.json()` then raises a raw `json.JSONDecodeError` (a
    `ValueError` subclass) that must not escape as a raw exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>Request Rate Threshold Exceeded</html>")

    client = _client(tmp_path, handler)

    with pytest.raises(FundamentalsError):
        client.get_json("https://data.sec.gov/x.json", ("x.json",))


def test_get_json_treats_a_corrupt_cache_file_as_a_miss_and_self_heals(tmp_path: Path) -> None:
    """A cache file truncated by an interrupted write must not permanently wedge the
    client -- it should be treated as a cache miss, re-fetched, and rewritten as valid."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    cache_path = tmp_path / "x.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    client = _client(tmp_path, handler)
    result = client.get_json("https://data.sec.gov/x.json", ("x.json",))

    assert result == {"ok": True}
    assert len(calls) == 1
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"ok": True}


def test_close_closes_the_underlying_httpx_client(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, handler)

    client.close()

    assert client._client.is_closed


def test_context_manager_closes_the_underlying_httpx_client(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with _client(tmp_path, handler) as client:
        assert not client._client.is_closed

    assert client._client.is_closed


def test_get_submissions_payload_hits_the_submissions_endpoint(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"filings": {}})

    client = _client(tmp_path, handler)

    client.get_submissions_payload("0000320193")

    assert seen_urls == ["https://data.sec.gov/submissions/CIK0000320193.json"]


def test_fetch_json_makes_a_request_every_time_and_writes_nothing(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _client(tmp_path, handler)

    assert client.fetch_json("https://data.sec.gov/x.json") == {"ok": True}
    assert client.fetch_json("https://data.sec.gov/x.json") == {"ok": True}
    assert len(calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_fetch_json_raises_not_found_on_a_404(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(FundamentalsNotFoundError):
        client.fetch_json("https://data.sec.gov/missing.json")


def test_fetch_json_raises_a_plain_fundamentals_error_on_a_server_error(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(500))

    with pytest.raises(FundamentalsError) as raised:
        client.fetch_json("https://data.sec.gov/x.json")
    assert not isinstance(raised.value, FundamentalsNotFoundError)


def test_fetch_json_raises_a_plain_fundamentals_error_on_a_throttle_page(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, text="<html>Request Rate Threshold Exceeded</html>"))

    with pytest.raises(FundamentalsError) as raised:
        client.fetch_json("https://data.sec.gov/x.json")
    assert not isinstance(raised.value, FundamentalsNotFoundError)


def test_get_json_also_raises_not_found_on_a_404(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(404))

    with pytest.raises(FundamentalsNotFoundError):
        client.get_json("https://data.sec.gov/missing.json", ("missing.json",))


def test_ticker_to_cik_raises_not_found_for_an_unknown_symbol(tmp_path: Path) -> None:
    client = _client(tmp_path, lambda request: httpx.Response(200, json=_TICKERS))

    with pytest.raises(FundamentalsNotFoundError, match="ZZZZ"):
        client.ticker_to_cik("ZZZZ")


def test_fetch_payload_methods_hit_the_sec_endpoints_without_caching(tmp_path: Path) -> None:
    seen_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={})

    client = _client(tmp_path, handler)

    client.fetch_company_facts_payload("0000320193")
    client.fetch_submissions_payload("0000320193")

    assert seen_urls == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        "https://data.sec.gov/submissions/CIK0000320193.json",
    ]
    assert list(tmp_path.iterdir()) == []


# --- freshness of the cached payloads (as_of / max_age_days) ---------------------------------------

_FACTS_FILE = ("companyfacts", "CIK0000320193.json")


def _age_file(path: Path, *, days: float) -> None:
    """Make `path` look like it was written `days` ago (its modification time is what the client reads)."""
    then = (datetime.now(UTC) - timedelta(days=days)).timestamp()
    os.utime(path, (then, then))


def _seed_facts(tmp_path: Path, payload: dict[str, object], *, age_days: float) -> Path:
    path = tmp_path.joinpath(*_FACTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _age_file(path, days=age_days)
    return path


def test_a_fresh_cached_payload_is_served_without_a_request(tmp_path: Path) -> None:
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=5)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert calls == []


def test_a_stale_cached_payload_is_refetched_and_rewritten(tmp_path: Path) -> None:
    calls = []
    path = _seed_facts(tmp_path, {"v": "cached"}, age_days=40)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "new"}
    assert len(calls) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": "new"}


def test_a_file_written_today_is_fresh_for_every_past_as_of(tmp_path: Path) -> None:
    # A backtest asks with a simulated, past `as_of`: the file's own date is today, so it never refetches.
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=0)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime(2024, 1, 2, tzinfo=UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert calls == []


def test_a_stale_payload_survives_a_failed_refresh_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _seed_facts(tmp_path, {"v": "cached"}, age_days=40)
    client = _client(tmp_path, lambda request: httpx.Response(500))

    with caplog.at_level(logging.WARNING):
        payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "cached"}
    assert "could not be refreshed" in caplog.text


def test_without_as_of_and_max_age_an_old_cached_payload_is_still_served(tmp_path: Path) -> None:
    calls = []
    _seed_facts(tmp_path, {"v": "cached"}, age_days=400)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    assert client.get_company_facts_payload("0000320193") == {"v": "cached"}
    assert calls == []


def test_a_naive_as_of_is_refused_when_a_cached_file_exists(tmp_path: Path) -> None:
    _seed_facts(tmp_path, {"v": "cached"}, age_days=1)
    client = _client(tmp_path, lambda request: httpx.Response(200, json={"v": "new"}))

    with pytest.raises(ValueError, match="timezone-aware"):
        client.get_company_facts_payload("0000320193", as_of=datetime(2026, 9, 14), max_age_days=30)


def test_a_corrupt_cached_file_is_refetched_even_with_freshness_arguments(tmp_path: Path) -> None:
    calls = []
    path = _seed_facts(tmp_path, {}, age_days=1)
    path.write_text("{not json", encoding="utf-8")
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"v": "new"}))

    payload = client.get_company_facts_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"v": "new"}
    assert len(calls) == 1


def test_the_submissions_payload_takes_the_same_freshness_arguments(tmp_path: Path) -> None:
    calls = []
    path = tmp_path / "submissions" / "CIK0000320193.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"filings": "old"}), encoding="utf-8")
    _age_file(path, days=40)
    client = _client(tmp_path, lambda request: calls.append(request) or httpx.Response(200, json={"filings": "new"}))

    payload = client.get_submissions_payload("0000320193", as_of=datetime.now(UTC), max_age_days=30)

    assert payload == {"filings": "new"}
    assert len(calls) == 1

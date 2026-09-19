"""SEC EDGAR I/O: `data.sec.gov` requests, cached to disk, rate-limited.

The only module allowed to import `httpx` for SEC access. Wraps every network failure as
`FundamentalsError` (repo rule: no raw library exception escapes) and requires a non-blank
`user_agent` (SEC's fair-access policy blocks generic/missing ones).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


class SecEdgarClient:
    """Cached, rate-limited SEC EDGAR REST client."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        min_request_interval_seconds: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise ConfigurationError("Missing or blank SEC_EDGAR_USER_AGENT environment variable")
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_request_interval_seconds = max(float(min_request_interval_seconds), 0.0)
        self._last_request_at = 0.0
        self._client = httpx.Client(transport=transport, timeout=30.0)

    def close(self) -> None:
        """Close the underlying `httpx.Client` and its connection pool."""
        self._client.close()

    def __enter__(self) -> SecEdgarClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    def _cache_path(self, *parts: str) -> Path:
        safe = [_SAFE_CHARS.sub("_", str(part)).strip("_") for part in parts]
        return self.cache_dir.joinpath(*safe)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def get_json(self, url: str, cache_key: tuple[str, ...]) -> dict[str, Any]:
        cache_path = self._cache_path(*cache_key)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except ValueError:
                # A corrupt/truncated cache entry (e.g. from an interrupted write) is
                # treated as a cache miss so it self-heals on the next fetch, rather than
                # permanently wedging the tool until someone deletes the file by hand.
                pass
        self._rate_limit()
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError also covers json.JSONDecodeError: SEC's fair-access throttling
            # sometimes serves an HTML block page with a 2xx status instead of JSON.
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def get_text(self, url: str, cache_key: tuple[str, ...]) -> str:
        cache_path = self._cache_path(*cache_key)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")
        self._rate_limit()
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            text = response.text
        except httpx.HTTPError as exc:
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8", errors="replace")
        return text

    def ticker_to_cik(self, symbol: str) -> str:
        payload = self.get_json(SEC_COMPANY_TICKERS_URL, ("company_tickers.json",))
        try:
            return sec.parse_company_tickers(payload, symbol)
        except ValueError as exc:
            raise FundamentalsError(str(exc)) from exc

    def get_company_facts_payload(self, cik: str) -> dict[str, Any]:
        url = f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
        return self.get_json(url, ("companyfacts", f"CIK{cik}.json"))

    def get_submissions_payload(self, cik: str) -> dict[str, Any]:
        url = f"{SEC_DATA_BASE_URL}/submissions/CIK{cik}.json"
        return self.get_json(url, ("submissions", f"CIK{cik}.json"))

    def get_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        url = sec.filing_url(cik, accession_number, primary_document)
        return self.get_text(url, ("filings", cik, accession_number, primary_document))

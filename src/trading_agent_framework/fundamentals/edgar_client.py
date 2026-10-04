"""SEC EDGAR I/O: `data.sec.gov` requests, cached to disk, rate-limited.

The only module allowed to import `httpx` for SEC access. Wraps every network failure as
`FundamentalsError` (repo rule: no raw library exception escapes) and requires a non-blank
`user_agent` (SEC's fair-access policy blocks generic/missing ones).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import ConfigurationError, FundamentalsError, FundamentalsNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "SecEdgarClient")

SEC_DATA_BASE_URL = "https://data.sec.gov"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


def _company_facts_url(cik: str) -> str:
    return f"{SEC_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"


def _submissions_url(cik: str) -> str:
    return f"{SEC_DATA_BASE_URL}/submissions/CIK{cik}.json"


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

    def fetch_json(self, url: str) -> dict[str, Any]:
        """One uncached request. HTTP 404 raises `FundamentalsNotFoundError`, any other failure `FundamentalsError`."""
        self._rate_limit()
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise FundamentalsNotFoundError(f"Not found: {url}") from exc
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            # ValueError also covers json.JSONDecodeError: SEC's fair-access throttling
            # sometimes serves an HTML block page with a 2xx status instead of JSON.
            raise FundamentalsError(f"Failed to fetch {url}: {exc}") from exc

    def get_json(self, url: str, cache_key: tuple[str, ...], *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict[str, Any]:
        """The payload at `url`, cached to disk.

        Without `as_of` and `max_age_days` a cached file is served forever (the original behaviour). With both,
        a cached file whose modification time is more than `max_age_days` before `as_of` is stale and is
        fetched again (`freshness.is_stale`, the screen's rule): in a backtest `as_of` is simulated, and a file
        written today is fresh for every past date; live refreshes monthly. A stale file that cannot be
        refreshed is served with a warning rather than failing the caller.
        """
        cache_path = self._cache_path(*cache_key)
        cached = self._read_cached_json(cache_path)
        if cached is not None and not self._cache_is_stale(cache_path, as_of, max_age_days):
            return cached
        try:
            payload = self.fetch_json(url)
        except FundamentalsError as exc:
            if cached is None:
                raise
            logger.log_warning(f"{url} could not be refreshed, using the cached copy: {exc}")
            return cached
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    @staticmethod
    def _read_cached_json(cache_path: Path) -> dict[str, Any] | None:
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            # A corrupt/truncated cache entry (e.g. from an interrupted write) is
            # treated as a cache miss so it self-heals on the next fetch, rather than
            # permanently wedging the tool until someone deletes the file by hand.
            return None

    @staticmethod
    def _cache_is_stale(cache_path: Path, as_of: datetime | None, max_age_days: int | None) -> bool:
        if as_of is None or max_age_days is None:
            return False
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        return is_stale(datetime.fromtimestamp(cache_path.stat().st_mtime, tz=UTC), as_of, max_age_days)

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
            raise FundamentalsNotFoundError(str(exc)) from exc

    def get_company_facts_payload(self, cik: str, *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict[str, Any]:
        return self.get_json(_company_facts_url(cik), ("companyfacts", f"CIK{cik}.json"), as_of=as_of, max_age_days=max_age_days)

    def fetch_company_facts_payload(self, cik: str) -> dict[str, Any]:
        """Uncached: the payload is about 4 MB, and the quality screen keeps only a reduced copy."""
        return self.fetch_json(_company_facts_url(cik))

    def get_submissions_payload(self, cik: str, *, as_of: datetime | None = None, max_age_days: int | None = None) -> dict[str, Any]:
        return self.get_json(_submissions_url(cik), ("submissions", f"CIK{cik}.json"), as_of=as_of, max_age_days=max_age_days)

    def fetch_submissions_payload(self, cik: str) -> dict[str, Any]:
        """Uncached, for a caller that needs one field of it."""
        return self.fetch_json(_submissions_url(cik))

    def get_filing_text(self, cik: str, accession_number: str, primary_document: str) -> str:
        url = sec.filing_url(cik, accession_number, primary_document)
        return self.get_text(url, ("filings", cik, accession_number, primary_document))

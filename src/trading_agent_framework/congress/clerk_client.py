"""House Clerk I/O: the yearly filing index and the filings' PDF text, cached to disk and rate-limited.

The only module allowed to import `httpx` and `pypdf` for the Clerk. Wraps every network and PDF failure as
`CongressDataError` (repo rule: no raw library exception escapes) and requires a non-blank `user_agent`.

Caching: a filing's text is immutable once filed (an amendment is a new DocID), so it is cached forever; an image-only
PDF is cached as an empty file. A year's index is refetched when it is stale by the CALLER'S clock (`as_of`, never the
wall clock): fetched no later than that year and more than a day before `as_of` (`freshness.is_stale`). So a backtest
never refetches an index it already holds, and live refreshes the current year's index daily.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from trading_agent_framework.congress.ptr import FilingRef
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import ConfigurationError, CongressDataError, CongressNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "ClerkClient")

CLERK_BASE_URL = "https://disclosures-clerk.house.gov/public_disc"
INDEX_MAX_AGE_DAYS = 1

_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.=-]+")


def index_url(year: int) -> str:
    return f"{CLERK_BASE_URL}/financial-pdfs/{year}FD.zip"


def filing_url(ref: FilingRef) -> str:
    """A PTR lives under `ptr-pdfs/`, a yearly report under `financial-pdfs/`, in the folder of the index year (the Year field)."""
    folder = "ptr-pdfs" if ref.kind == "ptr" else "financial-pdfs"
    return f"{CLERK_BASE_URL}/{folder}/{ref.year}/{ref.doc_id}.pdf"


class ClerkClient:
    """Cached, rate-limited House Clerk client."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path,
        *,
        min_request_interval_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not user_agent or not user_agent.strip():
            raise ConfigurationError("Missing or blank CONGRESS_USER_AGENT environment variable")
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self.min_request_interval_seconds = max(float(min_request_interval_seconds), 0.0)
        self._last_request_at = 0.0
        self._client = httpx.Client(transport=transport, timeout=60.0, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ClerkClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- plumbing ---------------------------------------------------------------------------------

    def _cache_path(self, *parts: str) -> Path:
        return self.cache_dir.joinpath(*(_SAFE_CHARS.sub("_", str(part)).strip("_") for part in parts))

    def _get(self, url: str) -> bytes:
        if self.min_request_interval_seconds:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_request_interval_seconds:
                time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()
        try:
            response = self._client.get(url, headers={"User-Agent": self.user_agent})
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise CongressNotFoundError(f"Not found: {url}") from exc
            raise CongressDataError(f"Failed to fetch {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise CongressDataError(f"Failed to fetch {url}: {exc}") from exc

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        """Write via a temp file so an interrupted download is never mistaken for a cache hit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- the year index ---------------------------------------------------------------------------

    def year_index_xml(self, year: int, as_of: datetime) -> str:
        """The year's filing index XML. `as_of` (timezone-aware, the caller's clock) decides whether a cached copy is stale."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        path = self._cache_path("index", f"{year}FD.xml")
        cached = path.read_text(encoding="utf-8") if path.exists() else None
        if cached is not None and not self._index_is_stale(path, year, as_of):
            return cached
        try:
            xml = self._download_index(year)
        except CongressDataError as exc:
            if cached is None or isinstance(exc, CongressNotFoundError):
                raise
            logger.log_warning(f"The {year} index could not be refreshed, using the cached copy: {exc}")
            return cached
        self._write_atomic(path, xml)
        return xml

    @staticmethod
    def _index_is_stale(path: Path, year: int, as_of: datetime) -> bool:
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        return fetched_at.year <= year and is_stale(fetched_at, as_of, INDEX_MAX_AGE_DAYS)

    def _download_index(self, year: int) -> str:
        data = self._get(index_url(year))
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                name = next((n for n in archive.namelist() if n.lower().endswith(".xml")), None)
                if name is None:
                    raise CongressDataError(f"The {year} index archive has no XML file")
                return archive.read(name).decode("utf-8-sig")
        except (zipfile.BadZipFile, UnicodeDecodeError, OSError) as exc:
            raise CongressDataError(f"The {year} index archive is unreadable: {exc}") from exc

    # --- filings ----------------------------------------------------------------------------------

    def filing_text(self, ref: FilingRef) -> str:
        """The extracted text of a filing's PDF; `""` for an image-only PDF. Cached forever."""
        path = self._cache_path(ref.kind, str(ref.year), f"{ref.doc_id}.txt")
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = self._pdf_text(self._get(filing_url(ref)), ref)
        self._write_atomic(path, text)
        return text

    @staticmethod
    def _pdf_text(data: bytes, ref: FilingRef) -> str:
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as exc:  # pypdf raises a wide range of types on a damaged file; none may escape
            raise CongressDataError(f"Filing {ref.doc_id} is not a readable PDF: {exc}") from exc

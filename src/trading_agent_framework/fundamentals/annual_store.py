"""`AnnualFiguresStore`: each company's reduced annual SEC figures, fetched lazily and cached on disk.

The I/O side of the quality screen. A company-facts payload is about 4 MB; the store fetches it
uncached, keeps only `sec.annual_figures(...)` of it (a few KB) in `<cache_dir>/CIK<cik>.json`, and
holds every record it has read in memory for the life of the process.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.fundamentals import sec
from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "AnnualFiguresStore")

_EMPTY_FIGURES: dict[str, list[dict[str, Any]]] = {"flows": [], "balances": [], "shares": []}


class AnnualFiguresStore:
    def __init__(self, client: SecEdgarClient, cache_dir: Path, *, wall_clock: Callable[[], datetime] | None = None) -> None:
        self._client = client
        self._cache_dir = cache_dir
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._ciks: dict[str, str | None] = {}
        self._records: dict[str, dict[str, Any]] = {}

    def cik(self, symbol: str) -> str | None:
        """`symbol`'s CIK, or `None` for a ticker SEC does not know. Two share classes share one CIK."""
        if symbol not in self._ciks:
            # Kept in memory: the client re-reads and re-parses its 900 KB ticker map on every call.
            try:
                self._ciks[symbol] = self._client.ticker_to_cik(symbol)
            except FundamentalsNotFoundError:
                self._ciks[symbol] = None
        return self._ciks[symbol]

    def figures(self, symbol: str, *, as_of: datetime, max_age_days: int) -> dict[str, Any] | None:
        """The reduced record, or `None` when SEC has no such company.

        Raises `FundamentalsError` when the fetch fails and there is no cached copy. A stale copy that
        cannot be refreshed is used, with a warning.
        """
        record = self._record(symbol, as_of, max_age_days)
        return None if record is None or record["status"] == "absent" else record

    def sic(self, symbol: str, *, as_of: datetime, max_age_days: int) -> int | None:
        """The company's SIC industry code, fetched once and saved in its record."""
        record = self._record(symbol, as_of, max_age_days)
        if record is None or record["status"] == "absent":
            return None
        if "sic" not in record:
            try:
                code = sec.parse_sic(self._client.fetch_submissions_payload(record["cik"]))
            except FundamentalsNotFoundError:
                code = None
            self._write({**record, "sic": code})
        return self._records[record["cik"]]["sic"]

    def _record(self, symbol: str, as_of: datetime, max_age_days: int) -> dict[str, Any] | None:
        cik = self.cik(symbol)
        if cik is None:
            return None
        record = self._records.get(cik) or self._read(cik)
        if record is not None and not is_stale(datetime.fromisoformat(record["fetched_at"]), as_of, max_age_days):
            self._records[cik] = record
            return record
        base = {"cik": cik, "fetched_at": self._wall_clock().isoformat()}
        try:
            fresh = {**base, "status": "ok", **sec.annual_figures(self._client.fetch_company_facts_payload(cik))}
        except FundamentalsNotFoundError:
            fresh = {**base, "status": "absent", **_EMPTY_FIGURES}
        except FundamentalsError as exc:
            if record is None:
                raise
            logger.log_warning(f"annual figures for {symbol} (CIK {cik}) could not be refreshed, using the copy fetched {record['fetched_at']}: {exc}")
            self._records[cik] = record
            return record
        if record is not None and "sic" in record:
            fresh["sic"] = record["sic"]  # an industry code does not move with a new filing
        self._write(fresh)
        return fresh

    def _path(self, cik: str) -> Path:
        return self._cache_dir / f"CIK{cik}.json"

    def _read(self, cik: str) -> dict[str, Any] | None:
        try:
            record = json.loads(self._path(cik).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # no file yet, or one truncated by an interrupted write: a cache miss
        if not isinstance(record, dict) or "fetched_at" not in record or "status" not in record:
            return None
        return record

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(record["cik"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temporary, path)
        self._records[record["cik"]] = record

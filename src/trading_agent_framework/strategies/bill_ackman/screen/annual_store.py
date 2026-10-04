"""`AnnualFiguresStore`: each company's reduced annual SEC figures, fetched lazily and cached on disk.

The I/O side of the quality screen. A company-facts payload is about 4 MB; the store fetches it
uncached, keeps only `annual_figures(...)` of it (a few KB) in `<cache_dir>/CIK<cik>.json`, and
holds every record it has read in memory for the life of the process.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.freshness import is_stale
from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import annual_figures, parse_sic
from trading_agent_framework.utils.errors import FundamentalsError, FundamentalsNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "AnnualFiguresStore")

# The version of the reduced record. It MUST be bumped whenever the tag lists or the row shape in
# `annual_figures.py` change: a record fetched today is fresh for every past date in a
# backtest (see `freshness.is_stale`), so without a bump an old cache would hide a tag fix for as long
# as the file exists. A record with a missing or different `schema` is a cache miss.
SCHEMA_VERSION = 2

_STATUSES = frozenset({"ok", "absent"})
_FIGURE_KEYS = ("flows", "balances", "shares")


def _empty_figures() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in _FIGURE_KEYS}


def _parse_fetched_at(value: object) -> datetime | None:
    """`value` as a timezone-aware datetime, or `None` when it is not an ISO string or is naive."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


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
                code = parse_sic(self._client.fetch_submissions_payload(record["cik"]))
            except FundamentalsNotFoundError:
                code = None
            self._write({**record, "sic": code})
        return self._records[record["cik"]]["sic"]

    def _record(self, symbol: str, as_of: datetime, max_age_days: int) -> dict[str, Any] | None:
        cik = self.cik(symbol)
        if cik is None:
            return None
        record = self._records.get(cik) or self._read(cik)
        if record is not None and not is_stale(_fetched_at(record), as_of, max_age_days):
            self._records[cik] = record
            return record
        base = {"cik": cik, "schema": SCHEMA_VERSION, "fetched_at": self._wall_clock().isoformat()}
        try:
            fresh = {**base, "status": "ok", **annual_figures(self._client.fetch_company_facts_payload(cik))}
        except FundamentalsNotFoundError:
            fresh = {**base, "status": "absent", **_empty_figures()}
            if record is not None and record["status"] == "ok":
                logger.log_warning(f"{symbol} (CIK {cik}) had annual figures but SEC now has no company facts for it: recorded as absent")
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
        except OSError, ValueError:
            return None  # no file yet, or one truncated by an interrupted write: a cache miss
        if not isinstance(record, dict):
            return None
        if record.get("status") not in _STATUSES or record.get("cik") != cik:
            return None
        if record.get("schema") != SCHEMA_VERSION:
            return None  # written by older tag lists or another row shape
        if _parse_fetched_at(record.get("fetched_at")) is None:
            return None
        if not all(isinstance(record.get(key), list) for key in _FIGURE_KEYS):
            return None
        return record

    def _write(self, record: dict[str, Any]) -> None:
        """Keep `record` in memory, then try to persist it. A failed write only costs a refetch next process."""
        cik = record["cik"]
        self._records[cik] = record
        path = self._path(cik)
        temporary: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A unique name per write: parallel runs sharing the cache must not share a temp file.
            descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"CIK{cik}.", suffix=".tmp")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle)
            os.replace(temporary, path)
        except OSError as exc:
            logger.log_warning(f"annual figures for CIK {cik} could not be saved to {path}: {exc}")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def _fetched_at(record: dict[str, Any]) -> datetime:
    """A record's fetch time; safe because `_read` validated it and the store writes only valid ones."""
    parsed = _parse_fetched_at(record["fetched_at"])
    assert parsed is not None
    return parsed

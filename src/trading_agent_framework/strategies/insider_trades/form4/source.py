"""Form 4 filings of the watchlist from SEC EDGAR, kept in memory: the only SEC I/O of the strategy.

`load` reads, per ticker, the submissions feed (the recent block, then the older pages reaching back to `since`) and fetches and
parses the raw XML of every Form 4 filed on or after `since`. Only form exactly "4" is read: amendments ("4/A") and Form 5 are
counted in `ignored_forms`. A feed also lists filings where the company is only a reporting owner of another issuer's shares: those
are counted in `other_issuer` and dropped. A ticker is all or nothing: one bad filing (a throttle page instead of XML, a failed
request) fails the ticker, which is then absent from the result and named in `failed`, so a half-read history never labels trades.

`known` is a cheap filter of the last `load`, safe to call every simulated day: a filing is known strictly after its filing date,
the market date of `as_of` being the first day it can be acted on. Every payload goes through `SecEdgarClient`'s cache; the
submissions block is refreshed against the load's `as_of` (`max_age_days=0`, as earnings_drift does), the XML and older pages never
change and are cached for good.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from trading_agent_framework.fundamentals.edgar_client import SecEdgarClient
from trading_agent_framework.fundamentals.sec import older_pages, raw_form4_document
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing, parse_form4
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import FundamentalsError, InsiderDataError

_IGNORED_FORMS = frozenset({"4/A", "5", "5/A"})


@dataclass(frozen=True, slots=True)
class LoadReport:
    loaded: tuple[str, ...]  # the tickers that loaded
    failed: dict[str, str] = field(default_factory=dict)  # ticker -> error message
    ignored_forms: int = 0  # 4/A and 5 rows filed since the history start, seen and not read
    other_issuer: int = 0  # Form 4 filings read and dropped because their issuer is not the ticker's company


@dataclass(frozen=True, slots=True)
class KnownInsiderData:
    filings: dict[str, list[Form4Filing]]  # ticker -> filings filed strictly before the market date, oldest first; a failed ticker is absent
    failed: dict[str, str]  # ticker -> why its last load failed


@dataclass(frozen=True, slots=True)
class _Row:
    form: str
    filed: date
    accession: str
    primary_document: str


def _rows(payload: Mapping[str, Any]) -> list[_Row]:
    """The filing rows of a submissions payload: `filings.recent`, or the top level of an older page."""
    filings = payload.get("filings")
    columns = filings.get("recent") if isinstance(filings, Mapping) else payload
    if not isinstance(columns, Mapping):
        return []

    def at(key: str, index: int) -> str:
        values = columns.get(key) or []
        return str(values[index]) if index < len(values) else ""

    rows: list[_Row] = []
    for index, form in enumerate(columns.get("form") or []):
        try:
            filed = date.fromisoformat(at("filingDate", index))
        except ValueError:
            continue
        rows.append(_Row(str(form), filed, at("accessionNumber", index), at("primaryDocument", index)))
    return rows


def _market_date(moment: datetime) -> date:
    if moment.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return moment.astimezone(MARKET_TZ).date()


def _same_cik(left: str, right: str) -> bool:
    """CIKs are numbers: the feed pads them to 10 digits, a filing may not."""
    return left.lstrip("0") == right.lstrip("0")


class Form4Source:
    def __init__(self, client: SecEdgarClient, watchlist: Sequence[str]) -> None:
        self._client = client
        self._watchlist = tuple(ticker.upper() for ticker in watchlist)
        self._filings: dict[str, list[Form4Filing]] = {}
        self._failed: dict[str, str] = {}
        self.loaded_on: date | None = None  # market date of the last load's `as_of`; the pipeline loads once per market date

    def load(self, *, as_of: datetime, since: date) -> LoadReport:
        loaded_on = _market_date(as_of)
        filings: dict[str, list[Form4Filing]] = {}
        failed: dict[str, str] = {}
        ignored_forms = other_issuer = 0
        for ticker in self._watchlist:
            try:
                read, ignored, other = self._ticker_filings(ticker, as_of=as_of, since=since)
            except (FundamentalsError, InsiderDataError, OSError, AttributeError, TypeError) as exc:  # a malformed 200 body or a cache I/O error fails one ticker
                failed[ticker] = str(exc) or type(exc).__name__
                continue
            filings[ticker] = read
            ignored_forms += ignored
            other_issuer += other
        self._filings, self._failed, self.loaded_on = filings, failed, loaded_on
        return LoadReport(loaded=tuple(filings), failed=failed, ignored_forms=ignored_forms, other_issuer=other_issuer)

    def known(self, as_of: datetime) -> KnownInsiderData:
        today = _market_date(as_of)
        filings = {ticker: [f for f in read if f.filed < today] for ticker, read in self._filings.items()}
        return KnownInsiderData(filings=filings, failed=dict(self._failed))

    def _ticker_filings(self, ticker: str, *, as_of: datetime, since: date) -> tuple[list[Form4Filing], int, int]:
        cik = self._client.ticker_to_cik(ticker)
        recent = self._client.get_submissions_payload(cik, as_of=as_of, max_age_days=0)
        payloads = [recent, *(self._client.get_submissions_page(name) for name in older_pages(recent, since))]
        filings: list[Form4Filing] = []
        seen: set[str] = set()
        ignored = other = 0
        for payload in payloads:
            for row in _rows(payload):
                if row.filed < since or row.accession in seen:
                    continue
                if row.form in _IGNORED_FORMS:
                    ignored += 1
                    continue
                if row.form != "4":
                    continue
                seen.add(row.accession)
                xml = self._client.get_filing_text(cik, row.accession, raw_form4_document(row.primary_document))
                filing = parse_form4(xml, accession=row.accession, filed=row.filed, ticker=ticker)
                if _same_cik(filing.issuer_cik, cik):
                    filings.append(filing)
                else:
                    other += 1
        return sorted(filings, key=lambda f: (f.filed, f.accession)), ignored, other

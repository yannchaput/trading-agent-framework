"""The filings of one member that are KNOWN on a given day, parsed: the join of the Clerk client and the pure parsers.

A filing is known only when its filing DATE is strictly before the market date of `as_of` (the Clerk gives a date and
no time). That one rule is the no-look-ahead gate for everything built on this module, in backtests and live alike.

`known()` finds the newest yearly report (the latest reporting year, then the latest filing; the next one down if that
one is an image-only scan), then every PTR filed after that report's period end. The Clerk lists a yearly report in
the index of its REPORTING year (the one filed in May 2026 holds the Dec 31, 2025 positions and sits in the 2025 index). The
trades since the period end are then applied by `holdings.reconstruct`; the filing date of a PTR is irrelevant to that.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from trading_agent_framework.strategies.congress_trades.congress import ptr
from trading_agent_framework.strategies.congress_trades.congress.annual import AssetHolding, parse_annual, period_end
from trading_agent_framework.strategies.congress_trades.congress.ptr import FilingRef, Transaction
from trading_agent_framework.utils.clock import MARKET_TZ
from trading_agent_framework.utils.errors import CongressDataError, CongressNotFoundError
from trading_agent_framework.utils.log import ColorLogger

logger = ColorLogger(logging.getLogger(__name__), "CongressSource")

_ANNUAL_LOOKBACK_YEARS = 5  # index years searched for a readable yearly report, newest first


class ClerkLike(Protocol):
    def year_index_xml(self, year: int, as_of: datetime) -> str: ...

    def filing_text(self, ref: FilingRef) -> str: ...


@dataclass(frozen=True, slots=True)
class KnownFilings:
    annual_ref: FilingRef
    period_end: date
    assets: list[AssetHolding]
    transactions: list[Transaction]
    refs: list[FilingRef]  # the yearly report and every PTR since it, newest filing first
    unparsed_filings: int  # image-only filings that were skipped
    skipped_non_stock: int  # rows that were not stock holdings or trades (options, bonds, rows that did not parse)


class CongressSource:
    def __init__(self, client: ClerkLike, politician: str) -> None:
        self._client = client
        self._politician = politician

    def known(self, as_of: datetime) -> KnownFilings:
        """What is known on `as_of`'s market date. Raises `CongressDataError` when no readable yearly report is known."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        today = as_of.astimezone(MARKET_TZ).date()
        by_year: dict[int, list[FilingRef]] = {}

        def refs_of(year: int) -> list[FilingRef]:
            if year not in by_year:
                try:
                    xml = self._client.year_index_xml(year, as_of)
                except CongressNotFoundError:
                    if year != today.year:
                        raise
                    logger.log_info(f"The {year} filing index is not published yet")
                    by_year[year] = []
                    return by_year[year]
                by_year[year] = [r for r in ptr.filings_for(ptr.parse_index(xml), self._politician) if r.filed < today]
            return by_year[year]

        unparsed = skipped = 0
        chosen: tuple[FilingRef, list[AssetHolding]] | None = None
        for year in range(today.year, today.year - _ANNUAL_LOOKBACK_YEARS, -1):
            for ref in sorted((r for r in refs_of(year) if r.kind == "annual"), key=lambda r: (r.filed, r.doc_id), reverse=True):
                parsed = parse_annual(self._client.filing_text(ref), ref)
                if parsed is None:
                    unparsed += 1
                    continue
                chosen = (ref, parsed.assets)
                skipped += parsed.skipped_non_stock
                break
            if chosen is not None:
                break
        if chosen is None:
            raise CongressDataError(f"No readable yearly report of {self._politician} is known before {today.isoformat()}")
        annual_ref, assets = chosen
        end = period_end(annual_ref)

        ptr_refs = sorted(
            (r for year in range(end.year + 1, today.year + 1) for r in refs_of(year) if r.kind == "ptr" and r.filed > end),
            key=lambda r: (r.filed, r.doc_id),
            reverse=True,
        )
        transactions: list[Transaction] = []
        for ref in ptr_refs:
            parsed_ptr = ptr.parse_ptr(self._client.filing_text(ref), ref)
            if parsed_ptr is None:
                unparsed += 1
                continue
            transactions.extend(parsed_ptr.transactions)
            skipped += parsed_ptr.skipped_non_stock
        return KnownFilings(
            annual_ref=annual_ref,
            period_end=end,
            assets=assets,
            transactions=transactions,
            refs=sorted([annual_ref, *ptr_refs], key=lambda r: (r.filed, r.doc_id), reverse=True),
            unparsed_filings=unparsed,
            skipped_non_stock=skipped,
        )

    @staticmethod
    def new_since(known: KnownFilings, processed: Collection[str]) -> list[FilingRef]:
        """The known filings whose DocID is not in `processed`, newest filing first."""
        seen = set(processed)
        return [ref for ref in sorted(known.refs, key=lambda r: (r.filed, r.doc_id), reverse=True) if ref.doc_id not in seen]

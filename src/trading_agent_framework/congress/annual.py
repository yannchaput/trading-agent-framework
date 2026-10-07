"""PURE parsing of a House yearly financial disclosure (Schedule A) and the value-band table.

A yearly report lists each asset held on the reporting year's December 31 as a VALUE BAND, never as shares. This
module reads those rows (stocks only), and maps a dollar value to a band's TIER (its index in `VALUE_BANDS`, 0 = the
smallest), which is what the portfolio weights are ordered by.

The Schedule A row layout assumed here (whitespace collapsed) is
`[owner] <asset name> (<TICKER>) [ST] <value band> [<income type(s)> [<income amount>]]` and it has NOT been checked
against a real report yet (see the congress_trades plan, Task 5 checkpoint). As in `ptr.py`, text with no
recognisable row raises `CongressDataError` and only blank text (an image-only scan) returns `None`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from trading_agent_framework.congress.ptr import FilingRef, parse_amount, parse_row_head
from trading_agent_framework.utils.errors import CongressDataError


@dataclass(frozen=True, slots=True)
class Band:
    low: Decimal
    high: Decimal | None  # None = open-ended (the top band)
    label: str


def _band(low: int, high: int | None, label: str) -> Band:
    return Band(Decimal(low), None if high is None else Decimal(high), label)


# The value bands of the House form, smallest first. Index = tier.
VALUE_BANDS: tuple[Band, ...] = (
    _band(1, 1_000, "$1 - $1,000"),
    _band(1_001, 15_000, "$1,001 - $15,000"),
    _band(15_001, 50_000, "$15,001 - $50,000"),
    _band(50_001, 100_000, "$50,001 - $100,000"),
    _band(100_001, 250_000, "$100,001 - $250,000"),
    _band(250_001, 500_000, "$250,001 - $500,000"),
    _band(500_001, 1_000_000, "$500,001 - $1,000,000"),
    _band(1_000_001, 5_000_000, "$1,000,001 - $5,000,000"),
    _band(5_000_001, 25_000_000, "$5,000,001 - $25,000,000"),
    _band(25_000_001, 50_000_000, "$25,000,001 - $50,000,000"),
    _band(50_000_001, None, "Over $50,000,000"),
)

_AMOUNT = r"(?:\$[\d,]+\s*-\s*\$[\d,]+|Over\s+\$[\d,]+)"
_INCOME = r"(?:Dividends|Interest|Capital Gains|Rent and Royalties|Rent|Royalties|Partnership Income|Tax-Deferred|Excepted Investment Fund|None)"
_ROW = re.compile(rf"\[(?P<type>[A-Z]{{2}})\]\s*(?P<value>{_AMOUNT}|None)(?:\s+{_INCOME}(?:\s*,\s*{_INCOME})*(?:\s+{_AMOUNT})?)?")
_TAG = re.compile(r"\[[A-Z]{2}\]")
_OVER = re.compile(r"Over\s+\$([\d,]+)")


@dataclass(frozen=True, slots=True)
class AssetHolding:
    doc_id: str
    owner: str  # "self" | "spouse" | "joint" | "dependent"
    ticker: str
    asset_name: str
    value_low: Decimal
    value_high: Decimal
    tier: int


@dataclass(frozen=True, slots=True)
class AnnualParse:
    assets: list[AssetHolding]
    skipped_non_stock: int  # tagged rows not read as a stock holding: options, bonds, funds, rows with no value or ticker


def tier_of(value: Decimal) -> int:
    """The tier of the band containing `value`; a value in a gap or below the table is the band below, above it the top band."""
    tier = 0
    for index, band in enumerate(VALUE_BANDS):
        if value >= band.low:
            tier = index
    return tier


def period_end(ref: FilingRef) -> date:
    """The date a yearly report's holdings are as of: December 31 of its reporting year."""
    return date(ref.year, 12, 31)


def _value(text: str) -> tuple[Decimal, Decimal]:
    """`$5,000,001 - $25,000,000` -> (5000001, 25000000); `Over $1,000,000` -> (1000001, 1000001), the floor of the open band."""
    over = _OVER.fullmatch(text.strip())
    if over is not None:
        floor = Decimal(over.group(1).replace(",", "")) + 1
        return floor, floor
    return parse_amount(text)


def parse_annual(text: str, ref: FilingRef) -> AnnualParse | None:
    """The stock holdings in a yearly report's extracted text; `None` for blank text (an image-only filing).

    Raises `CongressDataError` when the text has no recognisable asset row at all: a changed layout must not read as
    "she owns nothing". A report whose rows are all options or bonds is a valid, empty result.
    """
    flat = " ".join(text.split())
    if not flat:
        return None
    rows = list(_ROW.finditer(flat))
    if not rows:
        raise CongressDataError(f"Report {ref.doc_id}: no asset row found in {len(flat)} characters of text (layout changed?)")
    assets = []
    previous_end = 0
    for row in rows:
        head = flat[previous_end : row.start()]
        previous_end = row.end()
        parsed_head = parse_row_head(head)
        if row["type"] != "ST" or row["value"] == "None" or parsed_head is None:
            continue
        try:
            low, high = _value(row["value"])
        except ValueError:
            continue
        owner, name, ticker = parsed_head
        assets.append(AssetHolding(doc_id=ref.doc_id, owner=owner, ticker=ticker, asset_name=name, value_low=low, value_high=high, tier=tier_of(low)))
    return AnnualParse(assets=assets, skipped_non_stock=len(_TAG.findall(flat)) - len(assets))

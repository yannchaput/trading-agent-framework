"""PURE parsing of the House Clerk's filing index and Periodic Transaction Reports (PTRs).

No I/O and no clock. The index (`<YYYY>FD.xml`) lists every filing; `parse_index` keeps the PTRs and the yearly
reports (and their amendments). `parse_ptr` turns a PTR's extracted text into stock `Transaction`s.

The row layout assumed here (whitespace collapsed) is
`[owner] <asset name> (<TICKER>) [ST] <P|S|S (partial)|E> <MM/DD/YYYY> <MM/DD/YYYY> $<low> - $<high>` and it has NOT
been checked against a real filing yet (see the congress_trades plan, Task 5 checkpoint). Nothing here ever turns an
unreadable filing into "no trades": text with no recognisable row raises `CongressDataError`; only blank text (an
image-only scan) returns `None`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from trading_agent_framework.utils.errors import CongressDataError

# Filing types in the Clerk index: P = periodic transaction report, C = annual report, A = amendment of one.
_KINDS = {"P": "ptr", "C": "annual", "A": "annual"}
_HONORIFICS = frozenset({"hon", "mr", "mrs", "ms", "dr", "jr", "sr", "ii", "iii", "iv"})
_OWNERS = {"SP": "spouse", "JT": "joint", "DC": "dependent"}
_SIDES = {"P": "buy", "S": "sell", "S (partial)": "sell_partial"}

_TAG = re.compile(r"\[[A-Z]{2}\]")
_AMOUNT = r"(?:\$[\d,]+\s*-\s*\$[\d,]+|Over\s+\$[\d,]+)"
_DATE = r"\d{2}/\d{2}/\d{4}"
_ROW = re.compile(rf"\[(?P<type>[A-Z]{{2}})\]\s*(?P<side>S \(partial\)|[PSE])\s+(?P<tdate>{_DATE})\s+(?P<ndate>{_DATE})\s+(?P<amount>{_AMOUNT})")
# What precedes a row's type tag: an optional owner code, the asset name, then the ticker in parentheses.
_HEAD = re.compile(r"(?P<name>[^:()\[\]]*?)\s*\((?P<ticker>[A-Z][A-Z0-9.\-]{0,9})\)\s*$")
_OWNER_CODE = re.compile(r"(?:^|\s)(SP|JT|DC)\s+")
_MONEY = re.compile(r"\$([\d,]+)")


@dataclass(frozen=True, slots=True)
class FilingRef:
    doc_id: str
    member: str  # "First Last", no honorific or suffix
    kind: str  # "ptr" | "annual"
    filed: date
    year: int  # the Clerk's Year field: the reporting year of an annual report


@dataclass(frozen=True, slots=True)
class Transaction:
    doc_id: str
    owner: str  # "self" | "spouse" | "joint" | "dependent"
    ticker: str
    asset_name: str
    side: str  # "buy" | "sell" | "sell_partial"
    transaction_date: date
    notification_date: date
    amount_low: Decimal
    amount_high: Decimal
    filed: date


@dataclass(frozen=True, slots=True)
class PtrParse:
    transactions: list[Transaction]
    skipped_non_stock: int  # rows not turned into a transaction: options, bonds, funds, exchanges, rows that did not parse


# --- names and the index ------------------------------------------------------------------------


def normalize_name(name: str) -> str:
    """Lower case, punctuation (except hyphens) dropped, honorifics and suffixes removed, spaces collapsed."""
    tokens = re.sub(r"[.,]", " ", name).lower().split()
    return " ".join(token for token in tokens if token not in _HONORIFICS)


def _text(member: ET.Element, tag: str) -> str:
    return (member.findtext(tag) or "").strip()


def parse_index(xml_text: str) -> list[FilingRef]:
    """The PTRs and yearly reports (with amendments) in a year index, in file order. Other filing types are dropped."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise CongressDataError(f"The filing index is not valid XML: {exc}") from exc
    refs = []
    for member in root.iter("Member"):
        kind = _KINDS.get(_text(member, "FilingType"))
        if kind is None:
            continue
        doc_id = _text(member, "DocID")
        if not doc_id:
            raise CongressDataError("A filing in the index has no DocID")
        try:
            filed = datetime.strptime(_text(member, "FilingDate"), "%m/%d/%Y").date()
            year = int(_text(member, "Year"))
        except ValueError as exc:
            raise CongressDataError(f"Filing {doc_id} has an unreadable date or year in the index: {exc}") from exc
        name = " ".join(part for part in (_text(member, "First"), _text(member, "Last")) if part)
        refs.append(FilingRef(doc_id=doc_id, member=name, kind=kind, filed=filed, year=year))
    return refs


def filings_for(refs: Iterable[FilingRef], politician: str) -> list[FilingRef]:
    """The refs filed by `politician` (exact match on the normalized first and last name)."""
    wanted = normalize_name(politician)
    return [ref for ref in refs if normalize_name(ref.member) == wanted]


# --- amounts and PTR rows -----------------------------------------------------------------------


def parse_amount(text: str) -> tuple[Decimal, Decimal]:
    """`$1,001 - $15,000` -> (1001, 15000); the open-ended `Over $50,000,000` -> (50000000, 50000000)."""
    try:
        values = [Decimal(match.replace(",", "")) for match in _MONEY.findall(text)]
    except InvalidOperation as exc:
        raise ValueError(f"unreadable amount: {text!r}") from exc
    if len(values) == 2:
        return values[0], values[1]
    if len(values) == 1:
        return values[0], values[0]
    raise ValueError(f"unreadable amount: {text!r}")


def parse_row_head(head: str) -> tuple[str, str, str] | None:
    """(owner, asset name, ticker) from the text before a row's type tag, or None if it does not end in a ticker.

    The last owner code before the name wins: earlier text is the previous row's description. Shared with the yearly
    report parser (`annual.py`), whose Schedule A rows start the same way.
    """
    ticker_match = _HEAD.search(head)
    if ticker_match is None:
        return None
    candidate = " " + ticker_match["name"].strip() + " "
    owners = list(_OWNER_CODE.finditer(candidate))
    owner = "self"
    if owners:
        owner = _OWNERS[owners[-1].group(1)]
        candidate = candidate[owners[-1].end() :]
    return owner, candidate.strip(), ticker_match["ticker"]


def _row_transaction(ref: FilingRef, head: str, row: re.Match[str]) -> Transaction | None:
    """The stock transaction a type-tagged row describes, or None if it is not a readable stock purchase or sale."""
    side = _SIDES.get(row["side"])
    parsed_head = parse_row_head(head)
    if row["type"] != "ST" or side is None or parsed_head is None:
        return None
    try:
        transaction_date = datetime.strptime(row["tdate"], "%m/%d/%Y").date()
        notification_date = datetime.strptime(row["ndate"], "%m/%d/%Y").date()
        low, high = parse_amount(row["amount"])
    except ValueError:
        return None
    owner, asset_name, ticker = parsed_head
    return Transaction(
        doc_id=ref.doc_id,
        owner=owner,
        ticker=ticker,
        asset_name=asset_name,
        side=side,
        transaction_date=transaction_date,
        notification_date=notification_date,
        amount_low=low,
        amount_high=high,
        filed=ref.filed,
    )


def parse_ptr(text: str, ref: FilingRef) -> PtrParse | None:
    """The stock transactions of a PTR's extracted text; `None` for blank text (an image-only filing).

    Raises `CongressDataError` when the text has no recognisable transaction row at all: a changed layout must not
    read as "no trades". A filing whose rows are all options or bonds is a valid, empty result.
    """
    flat = " ".join(text.split())
    if not flat:
        return None
    rows = list(_ROW.finditer(flat))
    if not rows:
        raise CongressDataError(f"Filing {ref.doc_id}: no transaction row found in {len(flat)} characters of text (layout changed?)")
    transactions = []
    previous_end = 0
    for row in rows:
        head = flat[previous_end : row.start()]
        previous_end = row.end()
        transaction = _row_transaction(ref, head, row)
        if transaction is not None:
            transactions.append(transaction)
    return PtrParse(transactions=transactions, skipped_non_stock=len(_TAG.findall(flat)) - len(transactions))

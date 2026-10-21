"""PURE parsing of a Form 4 (`ownershipDocument` XML) into the open-market trades of its insider.

A Form 4 lists one or more reporting owners and one set of transaction rows that belongs to the filing, not to a single
owner (a joint filing by a fund and a director reports each row once). So the rows are attributed to the first listed owner
who is an insider (director or officer); the other owners are named as co-filers. A filing with no insider owner (a 10% holder
alone) yields no trades.

Only open-market purchases (`P`) and sales (`S`) are kept as trades. Every other non-derivative code (awards, exercises, tax
withholding, gifts, ...) and every derivative row is counted in `other_codes` and dropped. A `P`/`S` row missing shares, price,
trade date or the holding after the trade cannot be measured and is counted in `unpriced`.

Optional elements are tolerated: a missing `aff10b5One` is "no plan", a missing `footnotes` block is none. XML that does not
parse, or whose root is not `ownershipDocument` (SEC answers a throttled request with an HTML page), raises `InsiderDataError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree

from trading_agent_framework.utils.errors import InsiderDataError

_TRADE_CODES = frozenset({"P", "S"})


@dataclass(frozen=True, slots=True)
class Owner:
    cik: str
    name: str
    is_director: bool
    is_officer: bool
    officer_title: str
    is_ten_percent_owner: bool
    is_other: bool

    @property
    def is_insider(self) -> bool:
        return self.is_director or self.is_officer

    @property
    def role(self) -> str:
        parts: list[str] = []
        if self.is_officer:
            parts.append(self.officer_title or "officer")
        if self.is_director:
            parts.append("director")
        if parts:
            return ", ".join(parts)
        return "10% owner" if self.is_ten_percent_owner else "other"


@dataclass(frozen=True, slots=True)
class Form4Trade:
    accession: str
    filed: date
    ticker: str
    owner_cik: str
    owner_name: str
    role: str
    code: str  # "P" or "S"
    trade_date: date
    shares: Decimal
    price: Decimal
    owned_after: Decimal
    plan: bool
    footnotes: tuple[str, ...]

    @property
    def dollars(self) -> Decimal:
        return self.shares * self.price

    @property
    def holding_fraction(self) -> Decimal:
        """The share of the holding the trade moved: a purchase against the holding after it (1 when it opened the position), a sale against the holding before it."""
        base = self.owned_after if self.code == "P" else self.shares + self.owned_after
        return self.shares / base if base else Decimal(1)


@dataclass(frozen=True, slots=True)
class Form4Filing:
    accession: str
    filed: date
    ticker: str
    owners: tuple[Owner, ...]
    attributed_to: Owner | None
    co_filers: tuple[str, ...]
    trades: tuple[Form4Trade, ...]
    plan: bool
    other_codes: int
    unpriced: int
    footnotes: dict[str, str]


def _text(node: ElementTree.Element | None, path: str) -> str:
    found = node.find(path) if node is not None else None
    return (found.text or "").strip() if found is not None else ""


def _flag(node: ElementTree.Element | None, path: str) -> bool:
    return _text(node, path).lower() in ("1", "true")


def _value(row: ElementTree.Element, path: str) -> str:
    return _text(row, f"{path}/value")


def _decimal(row: ElementTree.Element, path: str) -> Decimal | None:
    raw = _value(row, path)
    try:
        return Decimal(raw) if raw else None
    except InvalidOperation:
        return None


def _day(row: ElementTree.Element, path: str) -> date | None:
    raw = _value(row, path)[:10]  # some filings carry a UTC offset after the date
    try:
        return date.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def _owner(node: ElementTree.Element) -> Owner:
    relationship = node.find("reportingOwnerRelationship")
    return Owner(
        cik=_text(node, "reportingOwnerId/rptOwnerCik"),
        name=_text(node, "reportingOwnerId/rptOwnerName"),
        is_director=_flag(relationship, "isDirector"),
        is_officer=_flag(relationship, "isOfficer"),
        officer_title=_text(relationship, "officerTitle"),
        is_ten_percent_owner=_flag(relationship, "isTenPercentOwner"),
        is_other=_flag(relationship, "isOther"),
    )


def _row_footnotes(row: ElementTree.Element, footnotes: dict[str, str]) -> tuple[str, ...]:
    """The text of every footnote the row's elements reference, in document order, each once."""
    ids = dict.fromkeys(ref.get("id", "") for ref in row.iter("footnoteId"))
    return tuple(footnotes[i] for i in ids if i in footnotes)


def parse_form4(xml: str, *, accession: str, filed: date, ticker: str) -> Form4Filing:
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise InsiderDataError(f"Form 4 {accession} is not valid XML: {exc}") from exc
    if root.tag != "ownershipDocument":
        raise InsiderDataError(f"Form 4 {accession} has no ownershipDocument root (got <{root.tag}>)")

    owners = tuple(_owner(node) for node in root.findall("reportingOwner"))
    attributed_to = next((owner for owner in owners if owner.is_insider), None)
    co_filers = tuple(owner.name for owner in owners if owner is not attributed_to)
    plan = _flag(root, "aff10b5One")
    footnotes = {note.get("id", ""): "".join(note.itertext()).strip() for note in root.findall("footnotes/footnote")}

    trades: list[Form4Trade] = []
    unpriced = 0
    other_codes = len(root.findall("derivativeTable/derivativeTransaction"))
    for row in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(row, "transactionCoding/transactionCode").upper()
        if code not in _TRADE_CODES:
            other_codes += 1
            continue
        shares = _decimal(row, "transactionAmounts/transactionShares")
        price = _decimal(row, "transactionAmounts/transactionPricePerShare")
        trade_date = _day(row, "transactionDate")
        owned_after = _decimal(row, "postTransactionAmounts/sharesOwnedFollowingTransaction")
        if shares is None or price is None or trade_date is None or owned_after is None:
            unpriced += 1
            continue
        if attributed_to is None:
            continue
        trades.append(
            Form4Trade(
                accession=accession,
                filed=filed,
                ticker=ticker,
                owner_cik=attributed_to.cik,
                owner_name=attributed_to.name,
                role=attributed_to.role,
                code=code,
                trade_date=trade_date,
                shares=shares,
                price=price,
                owned_after=owned_after,
                plan=plan,
                footnotes=_row_footnotes(row, footnotes),
            )
        )

    return Form4Filing(
        accession=accession,
        filed=filed,
        ticker=ticker,
        owners=owners,
        attributed_to=attributed_to,
        co_filers=co_filers,
        trades=tuple(trades),
        plan=plan,
        other_codes=other_codes,
        unpriced=unpriced,
        footnotes=footnotes,
    )

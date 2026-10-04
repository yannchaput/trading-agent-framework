"""The quality screen's reduction of a SEC company-facts payload to annual figures.

Pure translation (no I/O, no state, no clock): `annual_figures` turns a company-facts payload (about 4 MB)
into the few KB of rows the screen needs, and `parse_sic` reads the industry code from a submissions
payload. A fiscal year is identified by its period END date, never by XBRL's `fy` field: each 10-K repeats
three years of figures, all tagged with the filing's own `fy`.
"""

from __future__ import annotations

from typing import Any

from trading_agent_framework.fundamentals.sec import BALANCE_SHEET_TAGS, INCOME_STATEMENT_TAGS, parse_dt

MIN_FISCAL_YEAR_DAYS = 350
MAX_FISCAL_YEAR_DAYS = 380
_ANNUAL_FORMS = frozenset({"10-K", "10-K/A"})
_SHARE_COUNT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})

ANNUAL_FLOW_TAGS: dict[str, list[str]] = {
    # Revenue is the one field where the filing's LARGEST value among these tags wins (see
    # `_largest_per_period`): a filer may tag only part of its top line with the ASC 606 tag (URI's
    # equipment rentals) while `Revenues` is the total, and utilities (NEE) report the whole company
    # under the last two. Every other field takes the first tag present.
    "revenue": [
        *INCOME_STATEMENT_TAGS["revenue"],
        "RegulatedAndUnregulatedOperatingRevenue",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "operating_income": INCOME_STATEMENT_TAGS["operating_income"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}


def _tag_rows(facts: dict[str, Any], tag: str, unit: str, forms: frozenset[str]) -> list[dict[str, Any]]:
    """One tag's facts in one unit, from `forms` only, that carry a value, a period end and a filing date."""
    rows = (facts.get(tag) or {}).get("units", {}).get(unit, [])
    return [row for row in rows if isinstance(row, dict) and row.get("form") in forms and row.get("val") is not None and row.get("end") and row.get("filed")]


def _is_fiscal_year(row: dict[str, Any]) -> bool:
    start, end = parse_dt(row.get("start")), parse_dt(row.get("end"))
    return start is not None and end is not None and MIN_FISCAL_YEAR_DAYS <= (end - start).days <= MAX_FISCAL_YEAR_DAYS


def _slim(row: dict[str, Any], *, with_start: bool = False) -> dict[str, Any]:
    slim = {"start": row["start"]} if with_start else {}
    return {**slim, "end": row["end"], "value": row["val"], "filed": row["filed"]}


def _first_source_per_period(sources: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every filed version of each period, sorted by (end, filed); within one filing the first source wins.

    A later source only fills (period end, filing date) pairs that no earlier source has, so two tags
    reporting the same year in the same filing never mix. Across filings every version is kept whichever
    tag it came from: a later 10-K restates earlier years (possibly under another tag), and the as-of
    selection needs each version's own `filed` date.
    """
    owner: dict[tuple[str, str], int] = {}
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for index, rows in enumerate(sources):
        for row in rows:
            key = (row["end"], row["filed"])
            if owner.setdefault(key, index) == index:
                kept[key] = row
    return sorted(kept.values(), key=lambda row: (row["end"], row["filed"]))


def _largest_per_period(sources: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Like `_first_source_per_period`, but within one filing the largest value wins, whichever source it is.

    For revenue: several tags can cover one fiscal year in one filing, and a partial tag (an ASC 606
    subset, net sales without memberships) is smaller than the total. Where the tags agree, or only one
    exists, the result is the same as taking the first.
    """
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for rows in sources:
        for row in rows:
            key = (row["end"], row["filed"])
            if key not in kept or row["value"] > kept[key]["value"]:
                kept[key] = row
    return sorted(kept.values(), key=lambda row: (row["end"], row["filed"]))


def _summed(gaap: dict[str, Any], noncurrent_tag: str, current_tag: str) -> list[dict[str, Any]]:
    """Noncurrent + current debt per filing; the current part counts as 0 when the filing lacks it."""
    current = {(row["end"], row["filed"]): row["val"] for row in _tag_rows(gaap, current_tag, "USD", _ANNUAL_FORMS)}
    return [{"end": row["end"], "value": row["val"] + current.get((row["end"], row["filed"]), 0), "filed": row["filed"]} for row in _tag_rows(gaap, noncurrent_tag, "USD", _ANNUAL_FORMS)]


def _debt_sources(gaap: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Total debt, in order of preference: the total tags, then noncurrent + current sums, then the combined tag.

    Real filers use `LongTermDebt` (LOW), the lease-inclusive total (KO, MDLZ), `LongTermDebtNoncurrent` +
    `LongTermDebtCurrent` (QSR, NEE), or only the lease-inclusive pair (HLT). Known limitation: commercial
    paper and other short-term borrowings (`CommercialPaper`, `ShortTermBorrowings`) are not added, since
    filers disagree on whether the long-term tags already include them.
    """
    return [
        [_slim(row) for row in _tag_rows(gaap, "LongTermDebt", "USD", _ANNUAL_FORMS)],
        [_slim(row) for row in _tag_rows(gaap, "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities", "USD", _ANNUAL_FORMS)],
        _summed(gaap, "LongTermDebtNoncurrent", "LongTermDebtCurrent"),
        _summed(gaap, "LongTermDebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
        [_slim(row) for row in _tag_rows(gaap, "DebtLongtermAndShorttermCombinedAmount", "USD", _ANNUAL_FORMS)],
    ]


def annual_figures(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Reduce a company-facts payload (about 4 MB) to the rows the quality screen needs (a few KB).

    `flows` and `balances` come from annual reports only. `shares` keeps both the cover-page count
    (`dei`, `"kind": "cover"`, listed first) and the weighted-average diluted count (`"kind": "weighted"`),
    from annual and quarterly reports: a multi-class company has no usable cover-page count in this API.
    """
    root = payload.get("facts", {})
    gaap, dei = root.get("us-gaap", {}), root.get("dei", {})

    flows: list[dict[str, Any]] = []
    for field, tags in ANNUAL_FLOW_TAGS.items():
        sources = [[_slim(row, with_start=True) for row in _tag_rows(gaap, tag, "USD", _ANNUAL_FORMS) if _is_fiscal_year(row)] for tag in tags]
        pick = _largest_per_period if field == "revenue" else _first_source_per_period
        flows.extend({"field": field, **row} for row in pick(sources))

    cash_sources = [[_slim(row) for row in _tag_rows(gaap, tag, "USD", _ANNUAL_FORMS)] for tag in BALANCE_SHEET_TAGS["cash"]]
    balances: list[dict[str, Any]] = []
    for field, sources in (("debt", _debt_sources(gaap)), ("cash", cash_sources)):
        balances.extend({"field": field, **row} for row in _first_source_per_period(sources))

    cover = [_slim(row) for row in _tag_rows(dei, "EntityCommonStockSharesOutstanding", "shares", _SHARE_COUNT_FORMS)]
    # A 10-Q reports a 3-month and a year-to-date count under one (end, filed), and `_first_source_per_period`
    # keeps the last row of a source: order by start so the shortest period (the latest start) is last.
    weighted_facts = sorted(_tag_rows(gaap, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", _SHARE_COUNT_FORMS), key=lambda row: row.get("start") or "")
    weighted = [_slim(row) for row in weighted_facts]
    shares = [
        *({**row, "kind": "cover"} for row in _first_source_per_period([cover])),
        *({**row, "kind": "weighted"} for row in _first_source_per_period([weighted])),
    ]

    return {"flows": flows, "balances": balances, "shares": shares}


def parse_sic(submissions_payload: dict[str, Any]) -> int | None:
    """The company's SIC industry code from a submissions payload, or `None` when absent or malformed."""
    text = str(submissions_payload.get("sic") or "").strip()
    return int(text) if text.isdigit() else None

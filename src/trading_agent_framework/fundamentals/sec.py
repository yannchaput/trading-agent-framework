"""Pure SEC EDGAR translation: CIK lookup, income-statement/balance-sheet tag maps, as-of
candidate filtering and statement-period matching, filings parsing, URL building, HTML stripping.

No I/O, no state, no strategy/clock knowledge (same rules as `brokers/alpaca/market_data.py`).
Ported and trimmed from lumibot's `SECFundamentals` (`lumibot/fundamentals/sec.py`).
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime
from typing import Any

SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data/"

INCOME_STATEMENT_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_basic": ["EarningsPerShareBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

BALANCE_SHEET_TAGS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "liabilities": ["Liabilities"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "debt": ["LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtCurrent", "LongTermDebt"],
    "equity": ["StockholdersEquity"],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStocksIncludingAdditionalPaidInCapital",
    ],
}

# The same fields for filers of 20-F / 40-F reports, who tag under the `ifrs-full` taxonomy. Separate constants: the us-gaap
# maps above are imported by the bill_ackman screen and must not change. Tag names checked against real payloads (UMC,
# Kinross, Equinox Gold, Gold Fields, Abivax, ...). `RevenueFromSaleOfGoods` is left out: it can be a component of revenue.
IFRS_INCOME_STATEMENT_TAGS: dict[str, list[str]] = {
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers"],
    "cost_of_revenue": ["CostOfSales"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["ProfitLossFromOperatingActivities"],
    "net_income": ["ProfitLossAttributableToOwnersOfParent", "ProfitLoss"],
    "eps_basic": ["BasicEarningsLossPerShare"],
    "eps_diluted": ["DilutedEarningsLossPerShare"],
}

IFRS_BALANCE_SHEET_TAGS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "current_assets": ["CurrentAssets"],
    "cash": ["CashAndCashEquivalents"],
    "liabilities": ["Liabilities"],
    "current_liabilities": ["CurrentLiabilities"],
    "debt": ["Borrowings"],
    "equity": ["EquityAttributableToOwnersOfParent", "Equity"],
    "shares_outstanding": ["NumberOfSharesOutstanding"],
}

_PRIORITY_COMPANY_FACT_TAGS = tuple(dict.fromkeys(tag for tags in (*INCOME_STATEMENT_TAGS.values(), *BALANCE_SHEET_TAGS.values()) for tag in tags))
_IFRS_PRIORITY_COMPANY_FACT_TAGS = tuple(dict.fromkeys(tag for tags in (*IFRS_INCOME_STATEMENT_TAGS.values(), *IFRS_BALANCE_SHEET_TAGS.values()) for tag in tags))

_FORM_PRIORITY = {"10-K": 5, "20-F": 5, "40-F": 5, "10-Q": 4, "8-K": 2}


def parse_company_tickers(payload: dict[str, Any], symbol: str) -> str:
    """The zero-padded 10-digit CIK for `symbol`; raises `ValueError` if not found."""
    symbol_upper = symbol.upper().strip()
    for entry in payload.values():
        if str(entry.get("ticker", "")).upper() == symbol_upper:
            return f"{int(entry['cik_str']):010d}"
    raise ValueError(f"No SEC CIK found for ticker {symbol!r}")


def parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _same_tz(value: datetime, reference: datetime) -> datetime:
    if value.tzinfo is None and reference.tzinfo is not None:
        return value.replace(tzinfo=reference.tzinfo)
    if value.tzinfo is not None and reference.tzinfo is None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _candidate_sort_key(row: dict[str, Any], tags: list[str] | None = None) -> tuple[Any, ...]:
    form_priority = _FORM_PRIORITY.get(str(row.get("form") or "").upper(), 1)
    tag_priority = 0
    if tags:
        try:
            tag_priority = len(tags) - tags.index(str(row.get("tag") or ""))
        except ValueError:
            tag_priority = 0
    return (
        str(row.get("filed") or ""),
        str(row.get("end") or ""),
        str(row.get("start") or ""),
        form_priority,
        tag_priority,
    )


def filter_facts_as_of(facts: dict[str, Any], tags: list[str], as_of: datetime) -> list[dict[str, Any]]:
    """Candidates for `tags` filed on or before `as_of`, best (most recent, highest-priority form) first."""
    candidates: list[dict[str, Any]] = []
    for tag in tags:
        tag_payload = facts.get(tag)
        if not tag_payload:
            continue
        for unit, unit_facts in tag_payload.get("units", {}).items():
            if not isinstance(unit_facts, list):
                continue
            for fact in unit_facts:
                if not isinstance(fact, dict):
                    continue
                filed = parse_dt(fact.get("filed") or fact.get("acceptanceDateTime"))
                if filed is None or _same_tz(filed, as_of) > as_of:
                    continue
                candidates.append(
                    {
                        "tag": tag,
                        "value": fact.get("val"),
                        "unit": unit,
                        "filed": fact.get("filed"),
                        "form": fact.get("form"),
                        "fy": fact.get("fy"),
                        "fp": fact.get("fp"),
                        "start": fact.get("start"),
                        "end": fact.get("end"),
                        "accession_number": fact.get("accn"),
                    }
                )
    candidates.sort(key=lambda row: _candidate_sort_key(row, tags), reverse=True)
    return candidates


def latest_fact(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most relevant candidate; `filter_facts_as_of` already sorts best-first."""
    return candidates[0] if candidates else None


def compact_company_facts(payload: dict[str, Any], *, as_of: datetime, max_facts: int = 80) -> dict[str, Any]:
    """The latest as-of value for each priority tag present, capped at `max_facts`.

    US GAAP facts come first, then `ifrs-full` facts (20-F / 40-F filers), each marked `"taxonomy": "ifrs-full"`. A tag
    name used by both taxonomies keeps its more recently filed fact.
    """
    taxonomies = payload.get("facts", {})
    gaap, ifrs = taxonomies.get("us-gaap", {}), taxonomies.get("ifrs-full", {})
    gaap_priority = [tag for tag in _PRIORITY_COMPANY_FACT_TAGS if tag in gaap]
    ifrs_priority = [tag for tag in _IFRS_PRIORITY_COMPANY_FACT_TAGS if tag in ifrs]
    ordered = [
        *((gaap, tag, None) for tag in gaap_priority),
        *((ifrs, tag, "ifrs-full") for tag in ifrs_priority),
        *((gaap, tag, None) for tag in sorted(gaap) if tag not in gaap_priority),
        *((ifrs, tag, "ifrs-full") for tag in sorted(ifrs) if tag not in ifrs_priority),
    ]
    limit = max(int(max_facts), 1)
    compact: dict[str, Any] = {}
    truncated = False
    for facts, tag, taxonomy in ordered:
        if len(compact) >= limit:
            truncated = True
            break
        fact = latest_fact(filter_facts_as_of(facts, [tag], as_of))
        if fact is None:
            continue
        entry = {"value": fact["value"], "unit": fact["unit"], "filed": fact["filed"], "form": fact["form"]}
        if taxonomy is not None:
            entry["taxonomy"] = taxonomy
        if tag in compact:
            # Both taxonomies use this tag name (`Assets`, `Liabilities`, ...): keep the more recently filed fact.
            if str(fact["filed"] or "") > str(compact[tag]["filed"] or ""):
                compact[tag] = entry
            continue
        compact[tag] = entry
    return {"facts": compact, "fact_count": len(compact), "truncated": truncated}


def _statement_anchor(field_candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = [row for rows in field_candidates.values() for row in rows]
    if not candidates:
        return None
    return sorted(candidates, key=_candidate_sort_key, reverse=True)[0]


def _same_statement_period(candidate: dict[str, Any], anchor: dict[str, Any]) -> bool:
    candidate_accn = str(candidate.get("accession_number") or "").strip()
    anchor_accn = str(anchor.get("accession_number") or "").strip()
    if candidate_accn and anchor_accn and candidate_accn == anchor_accn:
        return True
    candidate_end = str(candidate.get("end") or "").strip()
    anchor_end = str(anchor.get("end") or "").strip()
    if not candidate_end or not anchor_end or candidate_end != anchor_end:
        return False
    for key in ("start", "fy", "fp", "form"):
        candidate_value = str(candidate.get(key) or "").strip()
        anchor_value = str(anchor.get(key) or "").strip()
        if candidate_value and anchor_value and candidate_value != anchor_value:
            return False
    return True


def statement_values(
    payload: dict[str, Any],
    tag_map: dict[str, list[str]],
    *,
    as_of: datetime,
    ifrs_tag_map: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Income-statement / balance-sheet field values as of `as_of`, from a raw company-facts payload.

    A field whose best candidate doesn't match the statement's period anchor is omitted, rather
    than mixing facts pulled from different SEC filings or periods.

    `tag_map` is read from the `us-gaap` taxonomy. `ifrs_tag_map` (the same field names mapped to `ifrs-full` tags)
    adds filers of 20-F / 40-F reports: both taxonomies' candidates compete and the most recently filed wins, so a
    filer that changed framework reads from its current one. None reads `us-gaap` only.
    """
    taxonomies = payload.get("facts", {})
    gaap = taxonomies.get("us-gaap", {})
    ifrs = taxonomies.get("ifrs-full", {}) if ifrs_tag_map is not None else {}
    field_candidates: dict[str, list[dict[str, Any]]] = {}
    for field, tags in tag_map.items():
        candidates = filter_facts_as_of(gaap, tags, as_of)
        ifrs_tags = (ifrs_tag_map or {}).get(field, [])
        ifrs_candidates = filter_facts_as_of(ifrs, ifrs_tags, as_of) if ifrs_tags else []
        if ifrs_candidates:
            every_tag = [*tags, *ifrs_tags]
            candidates = sorted([*candidates, *ifrs_candidates], key=lambda row: _candidate_sort_key(row, every_tag), reverse=True)
        if candidates:
            field_candidates[field] = candidates
    anchor = _statement_anchor(field_candidates)
    if anchor is None:
        return {}
    values: dict[str, dict[str, Any]] = {}
    for field, candidates in field_candidates.items():
        for candidate in candidates:
            if _same_statement_period(candidate, anchor):
                values[field] = {
                    "value": candidate["value"],
                    "unit": candidate["unit"],
                    "filed": candidate["filed"],
                    "form": candidate["form"],
                }
                break
    return values


def parse_filings(
    submissions_payload: dict[str, Any],
    *,
    cik: str,
    as_of: datetime,
    form: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Recent filings filed on or before `as_of`, optionally restricted to one `form`."""
    recent = submissions_payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    acceptances = recent.get("acceptanceDateTime", [])
    primary_docs = recent.get("primaryDocument", [])
    rows: list[dict[str, Any]] = []
    for idx, filing_form in enumerate(forms):
        if form and str(filing_form).upper() != form.upper():
            continue
        filed_raw = acceptances[idx] if idx < len(acceptances) and acceptances[idx] else filing_dates[idx]
        filed_dt = parse_dt(filed_raw)
        if filed_dt is None or _same_tz(filed_dt, as_of) > as_of:
            continue
        accession = accession_numbers[idx]
        primary_document = primary_docs[idx] if idx < len(primary_docs) else ""
        rows.append(
            {
                "form": filing_form,
                "accession_number": accession,
                "filing_date": filing_dates[idx] if idx < len(filing_dates) else None,
                "report_date": report_dates[idx] if idx < len(report_dates) else None,
                "primary_document": primary_document,
                "document_url": filing_url(cik, accession, primary_document),
            }
        )
        if len(rows) >= max(int(limit), 1):
            break
    return rows


def filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(str(cik).lstrip("0") or "0"))
    accession_clean = accession_number.replace("-", "")
    return f"{SEC_ARCHIVES_BASE_URL}{cik_int}/{accession_clean}/{primary_document}"


def strip_html(raw: str) -> str:
    """Filing HTML/XBRL to readable text."""
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    text = re.sub(r"(?is)<ix:hidden.*?</ix:hidden>", " ", text)
    text = re.sub(r"(?is)</?(?:p|div|br|tr|table|section|article|h[1-6])\b[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

"""Pure SEC EDGAR translation: CIK lookup, income-statement/balance-sheet tag maps, as-of
candidate filtering and statement-period matching, filings parsing, URL building, HTML stripping.

No I/O, no state, no strategy/clock knowledge (same rules as `brokers/alpaca/market_data.py`).
Ported and trimmed from lumibot's `SECFundamentals` (`lumibot/fundamentals/sec.py`).
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
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

# The same fields for filers of 20-F / 40-F reports, who tag under the `ifrs-full` taxonomy. bill_ackman's screen imports
# INCOME_STATEMENT_TAGS["revenue"] and ["operating_income"], BALANCE_SHEET_TAGS["cash"], CASH_FLOW_TAGS and
# `total_debt_sources`: changing those changes its records (bump its `annual_store.SCHEMA_VERSION`). Tag names checked
# against real payloads (UMC, Kinross, Equinox Gold, Gold Fields, Abivax, ...). `RevenueFromSaleOfGoods` is left out:
# it can be a component of revenue.
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

# Total debt, in order of preference: the total tags, then noncurrent + current sums, then the combined tag. Real filers
# use `LongTermDebt` (LOW), the lease-inclusive total (KO, MDLZ), `LongTermDebtNoncurrent` + `LongTermDebtCurrent` (QSR,
# NEE), or only the lease-inclusive pair (HLT). Known limitation: commercial paper and other short-term borrowings
# (`CommercialPaper`, `ShortTermBorrowings`) are not added, since filers disagree on whether the long-term tags include them.
TOTAL_DEBT_TAGS: tuple[str, ...] = ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities")
SUMMED_DEBT_TAGS: tuple[tuple[str, str], ...] = (
    ("LongTermDebtNoncurrent", "LongTermDebtCurrent"),
    ("LongTermDebtAndCapitalLeaseObligations", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
)
COMBINED_DEBT_TAG = "DebtLongtermAndShorttermCombinedAmount"

# Cash flow: the screen reads annual reports, the agent tools quarterly ones too.
CASH_FLOW_TAGS: dict[str, list[str]] = {
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}

# Checked against the 17 cached ifrs-full payloads (2026-10-11): all 17 tag the operating total, 16 the PP&E purchase,
# 4 only the broader PP&E-and-other-assets purchase.
IFRS_CASH_FLOW_TAGS: dict[str, list[str]] = {
    "operating_cash_flow": ["CashFlowsFromUsedInOperatingActivities"],
    "capex": [
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
    ],
}

# The agent tools' forms: quarterly and annual reports, and the 20-F / 40-F reports of foreign filers that tag us-gaap.
STATEMENT_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"})
STALE_STATEMENT_DAYS = 548  # about 18 months: a statement whose newest period ended earlier is flagged `stale`

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


def unit_rows(facts: dict[str, Any], tag: str, unit: str, forms: frozenset[str]) -> list[dict[str, Any]]:
    """One tag's facts in one unit, from `forms` only, that carry a value, a period end and a filing date."""
    rows = (facts.get(tag) or {}).get("units", {}).get(unit, [])
    return [row for row in rows if isinstance(row, dict) and row.get("form") in forms and row.get("val") is not None and row.get("end") and row.get("filed")]


def _debt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"end": row["end"], "value": row["val"], "filed": row["filed"]}


def _summed_debt(gaap: dict[str, Any], noncurrent_tag: str, current_tag: str, forms: frozenset[str], unit: str) -> list[dict[str, Any]]:
    """Noncurrent + current debt per filing; the current part counts as 0 when the filing lacks it."""
    current = {(row["end"], row["filed"]): row["val"] for row in unit_rows(gaap, current_tag, unit, forms)}
    return [{"end": row["end"], "value": row["val"] + current.get((row["end"], row["filed"]), 0), "filed": row["filed"]} for row in unit_rows(gaap, noncurrent_tag, unit, forms)]


def total_debt_sources(gaap: dict[str, Any], forms: frozenset[str], *, unit: str = "USD") -> list[list[dict[str, Any]]]:
    """Total-debt rows (`end`, `value`, `filed`; one `unit`, USD by default) from `forms`, one list per source, in order of preference."""
    return [
        *([_debt_row(row) for row in unit_rows(gaap, tag, unit, forms)] for tag in TOTAL_DEBT_TAGS),
        *(_summed_debt(gaap, noncurrent, current, forms, unit) for noncurrent, current in SUMMED_DEBT_TAGS),
        [_debt_row(row) for row in unit_rows(gaap, COMBINED_DEBT_TAG, unit, forms)],
    ]


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


_YEAR_DAYS = 365
_PERIOD_TOLERANCE_DAYS = 15  # fiscal periods can end a few days off the calendar (52/53-week years)


def _period(row: dict[str, Any]) -> dict[str, Any]:
    """The `start` (flows only) and `end` a value covers."""
    return {**({"start": row["start"]} if row.get("start") else {}), **({"end": row["end"]} if row.get("end") else {})}


def _a_year_before(end: datetime, other_end: datetime) -> bool:
    return abs((end - other_end).days - _YEAR_DAYS) <= _PERIOD_TOLERANCE_DAYS


def _prior_year(candidates: list[dict[str, Any]], chosen: dict[str, Any]) -> dict[str, Any] | None:
    """The same-length period ending about a year before `chosen`, in its unit, from `candidates` (best first).

    A flow (with a `start`) matches only a flow of the same length, so a quarter is never paired with a year-to-date
    figure; a balance (no `start`) matches only a balance.
    """
    end, start = parse_dt(chosen.get("end")), parse_dt(chosen.get("start"))
    if end is None:
        return None
    for row in candidates:
        row_end, row_start = parse_dt(row.get("end")), parse_dt(row.get("start"))
        if row_end is None or row.get("unit") != chosen.get("unit") or not _a_year_before(end, row_end):
            continue
        if (start is None) != (row_start is None):
            continue
        if start is not None and row_start is not None and abs((row_end - row_start).days - (end - start).days) > _PERIOD_TOLERANCE_DAYS:
            continue
        return {"value": row["value"], **_period(row)}
    return None


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

    Each value carries its period (`start` for a flow, `end`) and, when the filings known at `as_of` report it,
    `prior_year`: the same-length period ending about a year earlier (`value`, `start`, `end`).

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
                value = {"value": candidate["value"], "unit": candidate["unit"], "filed": candidate["filed"], "form": candidate["form"], **_period(candidate)}
                prior = _prior_year(candidates, candidate)
                if prior is not None:
                    value["prior_year"] = prior
                values[field] = value
                break
    return values


def _known_debt(gaap: dict[str, Any], as_of: datetime, unit: str, end_matches: Callable[[str], bool]) -> dict[str, Any] | None:
    """The newest-filed total-debt row in `unit` known at `as_of` whose end satisfies `end_matches`, from the most preferred source with one."""
    for rows in total_debt_sources(gaap, STATEMENT_FORMS, unit=unit):
        known = [row for row in rows if end_matches(row["end"]) and (filed := parse_dt(row["filed"])) is not None and _same_tz(filed, as_of) <= as_of]
        if known:
            return max(known, key=lambda row: row["filed"])
    return None


def total_debt(gaap: dict[str, Any], *, end: str, as_of: datetime, unit: str = "USD") -> dict[str, Any] | None:
    """Total debt in `unit` at the balance-sheet date `end`, with the figure a year earlier when reported; None without one."""
    target = parse_dt(end)
    row = _known_debt(gaap, as_of, unit, lambda row_end: row_end == end)
    if row is None or target is None:
        return None
    value: dict[str, Any] = {"value": row["value"], "unit": unit, "filed": row["filed"], "end": row["end"]}
    prior = _known_debt(gaap, as_of, unit, lambda row_end: (other := parse_dt(row_end)) is not None and _a_year_before(target, other))
    if prior is not None:
        value["prior_year"] = {"value": prior["value"], "end": prior["end"]}
    return value


def balance_sheet(payload: dict[str, Any], *, as_of: datetime) -> dict[str, dict[str, Any]]:
    """The balance sheet as of `as_of`; a US GAAP filer's `debt` is its total debt (`total_debt`) in the statement's reporting
    currency, never a current portion."""
    values = statement_values(payload, {**BALANCE_SHEET_TAGS, "debt": []}, as_of=as_of, ifrs_tag_map=IFRS_BALANCE_SHEET_TAGS)
    gaap = payload.get("facts", {}).get("us-gaap", {})
    end = next((value["end"] for value in values.values() if value.get("end")), None)
    unit = (values.get("assets") or next((value for value in values.values() if value.get("unit")), {})).get("unit", "USD")
    if gaap and end is not None and "debt" not in values:
        debt = total_debt(gaap, end=end, as_of=as_of, unit=unit)
        if debt is not None:
            values["debt"] = debt
    return values


def free_cash_flow(values: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Operating cash flow minus capex over the same period and unit, with its prior year when both have one."""
    operating, capex = values.get("operating_cash_flow"), values.get("capex")
    if operating is None or capex is None or _period(operating) != _period(capex) or operating["unit"] != capex["unit"]:
        return None
    value: dict[str, Any] = {"value": operating["value"] - capex["value"], "unit": operating["unit"], **_period(operating)}
    before, capex_before = operating.get("prior_year"), capex.get("prior_year")
    if before is not None and capex_before is not None and _period(before) == _period(capex_before):
        value["prior_year"] = {"value": before["value"] - capex_before["value"], **_period(before)}
    return value


def income_statement(payload: dict[str, Any], *, as_of: datetime) -> dict[str, dict[str, Any]]:
    """The income statement and cash flow as of `as_of`, with `free_cash_flow` when both cash-flow figures cover one period."""
    values = statement_values(payload, {**INCOME_STATEMENT_TAGS, **CASH_FLOW_TAGS}, as_of=as_of, ifrs_tag_map={**IFRS_INCOME_STATEMENT_TAGS, **IFRS_CASH_FLOW_TAGS})
    fcf = free_cash_flow(values)
    if fcf is not None:
        values["free_cash_flow"] = fcf
    return values


def is_stale_statement(values: dict[str, dict[str, Any]], *, as_of: datetime) -> bool:
    """Whether the newest period in `values` ended more than `STALE_STATEMENT_DAYS` before `as_of`."""
    ends = [end for value in values.values() if (end := parse_dt(value.get("end"))) is not None]
    return bool(ends) and (as_of - _same_tz(max(ends), as_of)).days > STALE_STATEMENT_DAYS


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


def raw_form4_document(primary_document: str) -> str:
    """A Form 4's raw XML file name: the submissions feed's primary document sits in an `xslF345X05/` folder (the rendered HTML view)."""
    return primary_document.rsplit("/", 1)[-1]


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

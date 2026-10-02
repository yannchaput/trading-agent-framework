from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from trading_agent_framework.fundamentals import sec

_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


def test_parse_company_tickers_finds_the_cik_case_insensitively() -> None:
    assert sec.parse_company_tickers(_TICKERS_PAYLOAD, "aapl") == "0000320193"


def test_parse_company_tickers_raises_for_an_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="ZZZZ"):
        sec.parse_company_tickers(_TICKERS_PAYLOAD, "ZZZZ")


def _facts(tag: str, *rows: dict[str, object]) -> dict[str, object]:
    return {tag: {"units": {"USD": list(rows)}}}


def test_filter_facts_as_of_excludes_post_cutoff_facts() -> None:
    facts = _facts(
        "NetIncomeLoss",
        {"val": 100, "filed": "2026-01-01", "form": "10-K", "accn": "a1"},
        {"val": 200, "filed": "2026-06-01", "form": "10-K", "accn": "a2"},
    )
    as_of = datetime(2026, 3, 1, tzinfo=UTC)

    candidates = sec.filter_facts_as_of(facts, ["NetIncomeLoss"], as_of)

    assert [c["value"] for c in candidates] == [100]


def test_filter_facts_as_of_prefers_10k_then_most_recent() -> None:
    facts = _facts(
        "NetIncomeLoss",
        {"val": 50, "filed": "2026-01-01", "form": "8-K", "accn": "a1"},
        {"val": 60, "filed": "2026-01-01", "form": "10-K", "accn": "a2"},
    )
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    candidates = sec.filter_facts_as_of(facts, ["NetIncomeLoss"], as_of)
    fact = sec.latest_fact(candidates)

    assert fact is not None
    assert fact["value"] == 60


def test_latest_fact_of_no_candidates_is_none() -> None:
    assert sec.latest_fact([]) is None


def test_compact_company_facts_orders_priority_tags_first_and_caps_at_max_facts() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K"}]}},
                "ZzzCustomTag": {"units": {"USD": [{"val": 1, "filed": "2026-01-01", "form": "10-K"}]}},
                "Assets": {"units": {"USD": [{"val": 20, "filed": "2026-01-01", "form": "10-K"}]}},
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    result = sec.compact_company_facts(payload, as_of=as_of, max_facts=2)

    assert result["fact_count"] == 2
    assert result["truncated"] is True
    assert set(result["facts"]) <= {"NetIncomeLoss", "Assets", "ZzzCustomTag"}


def test_statement_values_omits_a_field_whose_only_candidate_mismatches_the_anchor_period() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}
                },
                "GrossProfit": {
                    "units": {"USD": [{"val": 99, "filed": "2026-01-01", "form": "10-K", "end": "2024-12-31", "accn": "a2"}]}
                },
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    values = sec.statement_values(payload, {"net_income": ["NetIncomeLoss"], "gross_profit": ["GrossProfit"]}, as_of=as_of)

    # anchor is whichever candidate sorts first (NetIncomeLoss, filed/end used as sort keys);
    # gross_profit's different `end` means it can't match that anchor and is omitted.
    assert "net_income" in values
    assert "gross_profit" not in values


def test_parse_filings_filters_by_form_and_as_of() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-K"],
                "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002", "0000320193-26-000003"],
                "filingDate": ["2026-01-01", "2026-02-01", "2026-12-01"],
                "reportDate": ["2025-12-31", "2026-01-31", "2026-11-30"],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z", "2026-02-01T20:00:00Z", "2026-12-01T20:00:00Z"],
                "primaryDocument": ["aapl-10k.htm", "aapl-8k.htm", "aapl-10k2.htm"],
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    rows = sec.parse_filings(submissions, cik="0000320193", as_of=as_of, form="10-K", limit=10)

    assert [row["accession_number"] for row in rows] == ["0000320193-26-000001"]
    assert rows[0]["document_url"] == sec.filing_url("0000320193", "0000320193-26-000001", "aapl-10k.htm")


def test_parse_filings_respects_the_limit() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-Q", "10-Q"],
                "accessionNumber": ["a1", "a2", "a3"],
                "filingDate": ["2026-01-01", "2026-02-01", "2026-03-01"],
                "reportDate": [None, None, None],
                "acceptanceDateTime": ["2026-01-01T20:00:00Z", "2026-02-01T20:00:00Z", "2026-03-01T20:00:00Z"],
                "primaryDocument": ["d1.htm", "d2.htm", "d3.htm"],
            }
        }
    }
    as_of = datetime(2026, 6, 1, tzinfo=UTC)

    rows = sec.parse_filings(submissions, cik="1", as_of=as_of, limit=2)

    assert len(rows) == 2


def test_filing_url_strips_dashes_and_leading_zeros() -> None:
    url = sec.filing_url("0000320193", "0000320193-26-000001", "aapl-10k.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl-10k.htm"


def test_strip_html_removes_tags_and_collapses_whitespace() -> None:
    raw = "<html><body><p>Hello &amp; welcome</p><script>bad()</script></body></html>"
    assert sec.strip_html(raw) == "Hello & welcome"


def _annual(val: int, year: int, filed: str, *, form: str = "10-K") -> dict[str, object]:
    return {"val": val, "start": f"{year}-01-01", "end": f"{year}-12-31", "filed": filed, "form": form}


def _instant(val: int, end: str, filed: str, *, form: str = "10-K") -> dict[str, object]:
    return {"val": val, "end": end, "filed": filed, "form": form}


def _usd(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"USD": list(rows)}}


def _share_units(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"shares": list(rows)}}


def _company(gaap: dict[str, object] | None = None, dei: dict[str, object] | None = None) -> dict[str, object]:
    return {"facts": {"us-gaap": gaap or {}, "dei": dei or {}}}


def _rows(figures: dict[str, list[dict[str, object]]], kind: str, field: str) -> list[dict[str, object]]:
    return [row for row in figures[kind] if row["field"] == field]


def test_annual_figures_keeps_every_filed_version_of_a_fiscal_year() -> None:
    # Each 10-K repeats earlier years; the 2025 filing restates 2023.
    payload = _company(
        {
            "Revenues": _usd(
                _annual(100, 2023, "2024-02-15"),
                _annual(110, 2024, "2025-02-15"),
                _annual(101, 2023, "2026-02-15"),
                _annual(120, 2025, "2026-02-15"),
            )
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["filed"], row["value"]) for row in revenue] == [
        ("2023-12-31", "2024-02-15", 100),
        ("2023-12-31", "2026-02-15", 101),
        ("2024-12-31", "2025-02-15", 110),
        ("2025-12-31", "2026-02-15", 120),
    ]
    assert revenue[0]["start"] == "2023-01-01"


def test_annual_figures_drops_quarters_and_filings_that_are_not_annual_reports() -> None:
    payload = _company(
        {
            "Revenues": _usd(
                {"val": 30, "start": "2025-10-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"},
                _annual(999, 2025, "2026-01-20", form="8-K"),
                _annual(998, 2025, "2026-02-01", form="10-Q"),
                _annual(120, 2025, "2026-03-01", form="10-K/A"),
            )
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["filed"], row["value"]) for row in revenue] == [("2026-03-01", 120)]


def test_annual_figures_takes_the_largest_whole_company_revenue_tag_per_period() -> None:
    # A filer may tag only part of its top line with the ASC 606 tag (URI: equipment rentals are
    # outside it): within one filing the largest value among the revenue tags is the total.
    payload = _company(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(3695, 2025, "2026-01-28"), _annual(3588, 2024, "2026-01-28")),
            "Revenues": _usd(_annual(16099, 2025, "2026-01-28"), _annual(15345, 2024, "2026-01-28")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["value"]) for row in revenue] == [("2024-12-31", 15345), ("2025-12-31", 16099)]


def test_annual_figures_takes_the_largest_revenue_tag_whichever_one_it_is() -> None:
    payload = _company(
        {
            "Revenues": _usd(_annual(90, 2025, "2026-02-15")),
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(120, 2025, "2026-02-15")),
            "SalesRevenueNet": _usd(_annual(130, 2025, "2026-02-15")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "flows", "revenue")] == [130]


def test_annual_figures_leaves_revenue_unchanged_where_the_tags_agree_or_only_one_exists() -> None:
    payload = _company(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(12039, 2025, "2026-02-11"), _annual(11174, 2024, "2026-02-11")),
            "Revenues": _usd(_annual(12039, 2025, "2026-02-11")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["filed"], row["value"]) for row in revenue] == [("2024-12-31", "2026-02-11", 11174), ("2025-12-31", "2026-02-11", 12039)]


def test_annual_figures_picks_the_largest_revenue_tag_filing_by_filing() -> None:
    # The earlier filing only knew the part tag; the later one adds the total for the same year.
    payload = _company(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(3500, 2024, "2025-01-29"), _annual(3588, 2024, "2026-01-28")),
            "Revenues": _usd(_annual(15345, 2024, "2026-01-28")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["filed"], row["value"]) for row in revenue] == [("2025-01-29", 3500), ("2026-01-28", 15345)]


def test_annual_figures_reads_the_four_flow_fields_with_their_fallback_tags() -> None:
    payload = _company(
        {
            "Revenues": _usd(_annual(100, 2025, "2026-02-15")),
            "OperatingIncomeLoss": _usd(_annual(20, 2025, "2026-02-15")),
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": _usd(_annual(25, 2025, "2026-02-15")),
            "PaymentsToAcquireProductiveAssets": _usd(_annual(5, 2025, "2026-02-15")),
        }
    )

    flows = sec.annual_figures(payload)["flows"]

    assert {row["field"]: row["value"] for row in flows} == {"revenue": 100, "operating_income": 20, "operating_cash_flow": 25, "capex": 5}


def test_annual_figures_prefers_total_long_term_debt() -> None:
    payload = _company(
        {
            "LongTermDebt": _usd(_instant(90, "2025-12-31", "2026-02-15")),
            "LongTermDebtNoncurrent": _usd(_instant(70, "2025-12-31", "2026-02-15")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [90]


def test_annual_figures_sums_noncurrent_and_current_debt_when_there_is_no_total() -> None:
    payload = _company(
        {
            "LongTermDebtNoncurrent": _usd(_instant(70, "2024-12-31", "2025-02-15"), _instant(80, "2025-12-31", "2026-02-15")),
            "LongTermDebtCurrent": _usd(_instant(12, "2025-12-31", "2026-02-15")),
        }
    )

    debt = _rows(sec.annual_figures(payload), "balances", "debt")

    assert [(row["end"], row["value"]) for row in debt] == [("2024-12-31", 70), ("2025-12-31", 92)]


def test_annual_figures_falls_back_to_the_combined_debt_tag() -> None:
    payload = _company({"DebtLongtermAndShorttermCombinedAmount": _usd(_instant(55, "2025-12-31", "2026-02-15"))})

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [55]


def test_annual_figures_reads_the_lease_inclusive_total_debt_tag() -> None:
    # KO, MDLZ: the total tag; the noncurrent + current pair of the same filing is ignored.
    payload = _company(
        {
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": _usd(_instant(18_517, "2025-12-31", "2026-02-04")),
            "LongTermDebtAndCapitalLeaseObligations": _usd(_instant(17_222, "2025-12-31", "2026-02-04")),
            "LongTermDebtAndCapitalLeaseObligationsCurrent": _usd(_instant(1_295, "2025-12-31", "2026-02-04")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [18_517]


def test_annual_figures_sums_the_lease_inclusive_noncurrent_and_current_pair() -> None:
    # HLT: only the pair exists.
    payload = _company(
        {
            "LongTermDebtAndCapitalLeaseObligations": _usd(_instant(12_338, "2025-12-31", "2026-02-11")),
            "LongTermDebtAndCapitalLeaseObligationsCurrent": _usd(_instant(25, "2025-12-31", "2026-02-11")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [12_363]


def test_annual_figures_counts_a_missing_current_part_of_the_lease_inclusive_pair_as_zero() -> None:
    payload = _company({"LongTermDebtAndCapitalLeaseObligations": _usd(_instant(12_338, "2024-12-31", "2025-02-11"))})

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [12_338]


def test_annual_figures_prefers_the_plain_noncurrent_pair_over_the_lease_inclusive_pair() -> None:
    # QSR carries both pairs; only one may count, never a mix of the two.
    payload = _company(
        {
            "LongTermDebtNoncurrent": _usd(_instant(13_250, "2025-12-31", "2026-02-20")),
            "LongTermDebtCurrent": _usd(_instant(32, "2025-12-31", "2026-02-20")),
            "LongTermDebtAndCapitalLeaseObligations": _usd(_instant(13_300, "2025-12-31", "2026-02-20")),
            "LongTermDebtAndCapitalLeaseObligationsCurrent": _usd(_instant(68, "2025-12-31", "2026-02-20")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [13_282]


def test_annual_figures_uses_the_new_debt_tag_for_a_later_year_and_the_old_one_for_earlier_years() -> None:
    # KO's pattern: an older 10-K under LongTermDebt, the latest under the lease-inclusive total.
    payload = _company(
        {
            "LongTermDebt": _usd(_instant(30, "2022-12-31", "2023-02-20")),
            "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities": _usd(_instant(44, "2025-12-31", "2026-02-20")),
        }
    )

    assert [(row["end"], row["value"]) for row in _rows(sec.annual_figures(payload), "balances", "debt")] == [("2022-12-31", 30), ("2025-12-31", 44)]


def test_annual_figures_reads_utility_revenue_tags() -> None:
    # NEE: no `Revenues` since 2012; the whole-company tag wins over the contract-only subset.
    payload = _company(
        {
            "RegulatedAndUnregulatedOperatingRevenue": _usd(_annual(27_412, 2025, "2026-02-13")),
            "RevenueFromContractWithCustomerIncludingAssessedTax": _usd(_annual(25_800, 2025, "2026-02-13"), _annual(24_000, 2024, "2025-02-14")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["value"]) for row in revenue] == [("2024-12-31", 24_000), ("2025-12-31", 27_412)]


def test_annual_figures_takes_the_larger_of_the_contract_and_the_tax_inclusive_revenue_tags() -> None:
    payload = _company(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(120, 2025, "2026-02-15")),
            "RevenueFromContractWithCustomerIncludingAssessedTax": _usd(_annual(125, 2025, "2026-02-15")),
        }
    )

    assert [row["value"] for row in _rows(sec.annual_figures(payload), "flows", "revenue")] == [125]


def test_annual_figures_reads_cash_from_annual_reports_only() -> None:
    payload = _company(
        {
            "CashAndCashEquivalentsAtCarryingValue": _usd(
                _instant(40, "2025-12-31", "2026-02-15"),
                _instant(45, "2026-03-31", "2026-05-01", form="10-Q"),
            )
        }
    )

    assert [(row["end"], row["value"]) for row in _rows(sec.annual_figures(payload), "balances", "cash")] == [("2025-12-31", 40)]


def test_annual_figures_keeps_both_share_counts_cover_page_first() -> None:
    payload = _company(
        gaap={
            "WeightedAverageNumberOfDilutedSharesOutstanding": _share_units(
                {"val": 1010, "start": "2026-01-01", "end": "2026-03-31", "filed": "2026-05-01", "form": "10-Q"}
            )
        },
        dei={
            "EntityCommonStockSharesOutstanding": _share_units(
                _instant(1000, "2026-01-31", "2026-02-15"),
                _instant(1005, "2026-04-20", "2026-05-01", form="10-Q"),
                _instant(9999, "2026-05-01", "2026-05-02", form="8-K"),
            )
        },
    )

    shares = sec.annual_figures(payload)["shares"]

    assert shares == [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15", "kind": "cover"},
        {"end": "2026-04-20", "value": 1005, "filed": "2026-05-01", "kind": "cover"},
        {"end": "2026-03-31", "value": 1010, "filed": "2026-05-01", "kind": "weighted"},
    ]


@pytest.mark.parametrize("three_month_first", [True, False])
def test_annual_figures_keeps_the_three_month_weighted_count_when_a_quarterly_filing_also_reports_year_to_date(three_month_first: bool) -> None:
    # A 10-Q carries the 3-month and the year-to-date weighted average under one (end, filed); the
    # 3-month one is closest to the filing, and the payload's row order must not decide.
    quarter = {"val": 940, "start": "2025-07-01", "end": "2025-09-30", "filed": "2025-11-05", "form": "10-Q"}
    year_to_date = {"val": 930, "start": "2025-01-01", "end": "2025-09-30", "filed": "2025-11-05", "form": "10-Q"}
    rows = [quarter, year_to_date] if three_month_first else [year_to_date, quarter]
    payload = _company(gaap={"WeightedAverageNumberOfDilutedSharesOutstanding": _share_units(*rows)})

    assert sec.annual_figures(payload)["shares"] == [{"end": "2025-09-30", "value": 940, "filed": "2025-11-05", "kind": "weighted"}]


def test_annual_figures_skips_rows_without_a_value_or_a_filing_date() -> None:
    payload = _company(
        {
            "Revenues": _usd(
                {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-15", "form": "10-K"},
                {"val": 100, "start": "2025-01-01", "end": "2025-12-31", "form": "10-K"},
            )
        }
    )

    assert sec.annual_figures(payload) == {"flows": [], "balances": [], "shares": []}


def test_annual_figures_of_an_empty_payload_is_empty() -> None:
    assert sec.annual_figures({}) == {"flows": [], "balances": [], "shares": []}


def test_parse_sic_reads_the_code_as_an_integer() -> None:
    assert sec.parse_sic({"sic": "3571", "sicDescription": "Electronic Computers"}) == 3571


@pytest.mark.parametrize("payload", [{}, {"sic": ""}, {"sic": None}, {"sic": "n/a"}])
def test_parse_sic_of_a_missing_or_malformed_code_is_none(payload: dict[str, object]) -> None:
    assert sec.parse_sic(payload) is None


def test_annual_figures_keeps_versions_filed_under_a_different_tag() -> None:
    # The company switched revenue tags: the 2025 filing restates FY2023 under the new tag.
    payload = _company(
        {
            "Revenues": _usd(_annual(100, 2023, "2024-02-15")),
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(102, 2023, "2025-02-15")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    assert [(row["end"], row["filed"], row["value"]) for row in revenue] == [
        ("2023-12-31", "2024-02-15", 100),
        ("2023-12-31", "2025-02-15", 102),
    ]


def test_annual_figures_never_mixes_two_tags_within_one_filing_row() -> None:
    payload = _company(
        {
            "Revenues": _usd(_annual(90, 2025, "2026-02-15")),
            "RevenueFromContractWithCustomerExcludingAssessedTax": _usd(_annual(120, 2025, "2026-02-15")),
        }
    )

    revenue = _rows(sec.annual_figures(payload), "flows", "revenue")

    # One value per (period end, filing): the larger tag's, never a blend.
    assert [(row["end"], row["filed"], row["value"]) for row in revenue] == [("2025-12-31", "2026-02-15", 120)]


def test_annual_figures_keeps_debt_versions_filed_under_a_different_tag() -> None:
    payload = _company(
        {
            "LongTermDebtNoncurrent": _usd(_instant(70, "2024-12-31", "2025-02-15")),
            "LongTermDebt": _usd(_instant(75, "2024-12-31", "2026-02-15")),
        }
    )

    debt = _rows(sec.annual_figures(payload), "balances", "debt")

    assert [(row["end"], row["filed"], row["value"]) for row in debt] == [
        ("2024-12-31", "2025-02-15", 70),
        ("2024-12-31", "2026-02-15", 75),
    ]


def test_annual_figures_prefers_total_debt_within_one_filing() -> None:
    payload = _company(
        {
            "LongTermDebt": _usd(_instant(90, "2025-12-31", "2026-02-15")),
            "LongTermDebtNoncurrent": _usd(_instant(70, "2025-12-31", "2026-02-15")),
            "LongTermDebtCurrent": _usd(_instant(12, "2025-12-31", "2026-02-15")),
        }
    )

    debt = _rows(sec.annual_figures(payload), "balances", "debt")

    assert [(row["end"], row["filed"], row["value"]) for row in debt] == [("2025-12-31", "2026-02-15", 90)]


@pytest.mark.parametrize(
    ("days", "kept"),
    [(349, False), (350, True), (364, True), (365, True), (366, True), (371, True), (380, True), (381, False)],
)
def test_annual_figures_fiscal_year_length_boundaries(days: int, kept: bool) -> None:
    end = date(2025, 12, 31)
    start = end - timedelta(days=days)
    row = {"val": 100, "start": start.isoformat(), "end": end.isoformat(), "filed": "2026-02-15", "form": "10-K"}

    revenue = _rows(sec.annual_figures(_company({"Revenues": _usd(row)})), "flows", "revenue")

    assert bool(revenue) is kept

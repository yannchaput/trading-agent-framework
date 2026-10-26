from __future__ import annotations

from datetime import UTC, datetime

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
                "NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K", "end": "2025-12-31", "accn": "a1"}]}},
                "GrossProfit": {"units": {"USD": [{"val": 99, "filed": "2026-01-01", "form": "10-K", "end": "2024-12-31", "accn": "a2"}]}},
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


# --- IFRS filers (20-F / 40-F): the `ifrs-full` taxonomy ---------------------------------------------------------
# Shapes follow real company-facts payloads: Kinross (40-F) tags revenue `RevenueFromContractsWithCustomers`, UMC reports
# in TWD and USD, and Celestica / SSR Mining carry both taxonomies because they changed reporting framework.


def _row(val: object, filed: str, end: str, form: str = "20-F", accn: str = "a1") -> dict[str, object]:
    return {"val": val, "filed": filed, "end": end, "form": form, "accn": accn}


def _ifrs_payload() -> dict[str, object]:
    return {
        "facts": {
            "dei": {},
            "ifrs-full": {
                "RevenueFromContractsWithCustomers": {"units": {"EUR": [_row(7_051, "2026-03-26", "2025-12-31")]}},
                "ProfitLossFromOperatingActivities": {"units": {"EUR": [_row(3_277, "2026-03-26", "2025-12-31")]}},
                "ProfitLossAttributableToOwnersOfParent": {"units": {"EUR": [_row(2_390, "2026-03-26", "2025-12-31")]}},
                "BasicEarningsLossPerShare": {"units": {"EUR/shares": [_row(1.96, "2026-03-26", "2025-12-31")]}},
                "Assets": {"units": {"EUR": [_row(12_411, "2026-03-26", "2025-12-31")]}},
                "EquityAttributableToOwnersOfParent": {"units": {"EUR": [_row(8_576, "2026-03-26", "2025-12-31")]}},
                "Borrowings": {"units": {"EUR": [_row(738, "2026-03-26", "2025-12-31")]}},
            },
        }
    }


_AS_OF = datetime(2026, 6, 1, tzinfo=UTC)


def test_an_ifrs_filers_income_statement_is_read_from_the_ifrs_full_taxonomy() -> None:
    values = sec.statement_values(_ifrs_payload(), sec.INCOME_STATEMENT_TAGS, as_of=_AS_OF, ifrs_tag_map=sec.IFRS_INCOME_STATEMENT_TAGS)

    assert values["revenue"]["value"] == 7_051 and values["revenue"]["unit"] == "EUR"
    assert values["operating_income"]["value"] == 3_277
    assert values["net_income"]["value"] == 2_390
    assert values["eps_basic"]["value"] == 1.96 and values["eps_basic"]["unit"] == "EUR/shares"


def test_an_ifrs_filers_balance_sheet_is_read_from_the_ifrs_full_taxonomy() -> None:
    values = sec.statement_values(_ifrs_payload(), sec.BALANCE_SHEET_TAGS, as_of=_AS_OF, ifrs_tag_map=sec.IFRS_BALANCE_SHEET_TAGS)

    assert values["assets"]["value"] == 12_411
    assert values["equity"]["value"] == 8_576
    assert values["debt"]["value"] == 738


def test_without_an_ifrs_tag_map_statement_values_reads_us_gaap_only() -> None:
    """The default keeps every existing caller exactly as it was."""
    assert sec.statement_values(_ifrs_payload(), sec.INCOME_STATEMENT_TAGS, as_of=_AS_OF) == {}


def test_an_ifrs_tag_map_changes_nothing_for_a_us_gaap_filer() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [_row(500, "2026-02-01", "2025-12-31", "10-K")]}},
                "NetIncomeLoss": {"units": {"USD": [_row(40, "2026-02-01", "2025-12-31", "10-K")]}},
            }
        }
    }

    plain = sec.statement_values(payload, sec.INCOME_STATEMENT_TAGS, as_of=_AS_OF)
    with_ifrs = sec.statement_values(payload, sec.INCOME_STATEMENT_TAGS, as_of=_AS_OF, ifrs_tag_map=sec.IFRS_INCOME_STATEMENT_TAGS)

    assert plain == with_ifrs and plain["revenue"]["value"] == 500


def _both_taxonomies() -> dict[str, object]:
    """A filer that reported under IFRS until its 2024-03 20-F and under US GAAP since (Celestica's shape)."""
    return {
        "facts": {
            "ifrs-full": {"Assets": {"units": {"USD": [_row(5_890, "2024-03-11", "2023-12-31", accn="old")]}}},
            "us-gaap": {"Assets": {"units": {"USD": [_row(6_700, "2026-02-05", "2025-12-31", "10-K", accn="new")]}}},
        }
    }


def test_when_a_filer_has_both_taxonomies_the_most_recently_filed_fact_wins() -> None:
    values = sec.statement_values(_both_taxonomies(), sec.BALANCE_SHEET_TAGS, as_of=_AS_OF, ifrs_tag_map=sec.IFRS_BALANCE_SHEET_TAGS)

    assert values["assets"]["value"] == 6_700


def test_ifrs_facts_are_filtered_by_as_of_like_us_gaap_facts() -> None:
    """No look-ahead: a backtest dated before the switch must see the IFRS figure, not the later US GAAP one."""
    values = sec.statement_values(_both_taxonomies(), sec.BALANCE_SHEET_TAGS, as_of=datetime(2025, 1, 1, tzinfo=UTC), ifrs_tag_map=sec.IFRS_BALANCE_SHEET_TAGS)

    assert values["assets"]["value"] == 5_890


def test_the_same_tag_name_in_both_taxonomies_is_never_read_from_the_wrong_one() -> None:
    """`Assets` exists in us-gaap and ifrs-full: a us-gaap tag map must not pick up the ifrs-full fact on its own."""
    payload = {"facts": {"ifrs-full": {"Assets": {"units": {"USD": [_row(1, "2026-03-01", "2025-12-31")]}}}}}

    assert sec.statement_values(payload, sec.BALANCE_SHEET_TAGS, as_of=_AS_OF) == {}


def test_compact_company_facts_includes_an_ifrs_filers_facts_marked_with_their_taxonomy() -> None:
    result = sec.compact_company_facts(_ifrs_payload(), as_of=_AS_OF)

    assert result["fact_count"] == 7
    assert result["facts"]["RevenueFromContractsWithCustomers"] == {"value": 7_051, "unit": "EUR", "filed": "2026-03-26", "form": "20-F", "taxonomy": "ifrs-full"}


def test_compact_company_facts_for_a_us_gaap_filer_has_no_taxonomy_key() -> None:
    payload = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [{"val": 10, "filed": "2026-01-01", "form": "10-K"}]}}}}}

    result = sec.compact_company_facts(payload, as_of=_AS_OF)

    assert result["facts"]["NetIncomeLoss"] == {"value": 10, "unit": "USD", "filed": "2026-01-01", "form": "10-K"}


def test_compact_company_facts_keeps_the_newer_fact_when_both_taxonomies_use_a_tag_name() -> None:
    result = sec.compact_company_facts(_both_taxonomies(), as_of=_AS_OF)

    assert result["facts"]["Assets"]["value"] == 6_700 and "taxonomy" not in result["facts"]["Assets"]


# --- total debt and cash flow: one definition for the agent tools and bill_ackman's screen ---------------------------

_ANNUAL = frozenset({"10-K", "10-K/A"})


def _usd(*rows: dict[str, object]) -> dict[str, object]:
    return {"units": {"USD": list(rows)}}


def test_total_debt_sources_follow_the_order_of_preference() -> None:
    year_end = {"end": "2025-12-31", "filed": "2026-02-01", "form": "10-K"}
    gaap = {
        "LongTermDebt": _usd({"val": 900, **year_end}),
        "LongTermDebtNoncurrent": _usd({"val": 700, **year_end}),
        "LongTermDebtCurrent": _usd({"val": 200, **year_end}),
        "DebtLongtermAndShorttermCombinedAmount": _usd({"val": 950, **year_end}),
    }

    sources = sec.total_debt_sources(gaap, _ANNUAL)

    assert [rows[0]["value"] if rows else None for rows in sources] == [900, None, 900, None, 950]
    assert sources[0][0] == {"end": "2025-12-31", "value": 900, "filed": "2026-02-01"}


def test_a_noncurrent_debt_without_its_current_part_counts_the_current_part_as_zero() -> None:
    gaap = {"LongTermDebtNoncurrent": _usd({"val": 700, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K"})}

    assert sec.total_debt_sources(gaap, _ANNUAL)[2] == [{"end": "2025-12-31", "value": 700, "filed": "2026-02-01"}]


def test_total_debt_sources_read_only_the_forms_asked_for() -> None:
    gaap = {"LongTermDebt": _usd({"val": 900, "end": "2026-03-31", "filed": "2026-05-01", "form": "10-Q"})}

    assert sec.total_debt_sources(gaap, _ANNUAL)[0] == []
    assert sec.total_debt_sources(gaap, frozenset({"10-Q"}))[0] == [{"end": "2026-03-31", "value": 900, "filed": "2026-05-01"}]


def test_the_cash_flow_tags_are_shared_with_bill_ackmans_screen() -> None:
    from trading_agent_framework.strategies.bill_ackman.screen.annual_figures import ANNUAL_FLOW_TAGS

    assert ANNUAL_FLOW_TAGS["operating_cash_flow"] is sec.CASH_FLOW_TAGS["operating_cash_flow"]
    assert ANNUAL_FLOW_TAGS["capex"] is sec.CASH_FLOW_TAGS["capex"]


# --- statement periods and the prior year -------------------------------------------------------------------------


def _flow(val: object, start: str, end: str, filed: str = "2026-05-01", form: str = "10-Q", accn: str = "q1") -> dict[str, object]:
    return {"val": val, "start": start, "end": end, "filed": filed, "form": form, "accn": accn}


def test_a_value_carries_its_period_and_the_same_quarter_a_year_earlier() -> None:
    payload = {"facts": {"us-gaap": {"Revenues": _usd(_flow(120, "2026-01-01", "2026-03-31"), _flow(100, "2025-01-01", "2025-03-31"))}}}

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert values["revenue"] == {
        "value": 120,
        "unit": "USD",
        "filed": "2026-05-01",
        "form": "10-Q",
        "start": "2026-01-01",
        "end": "2026-03-31",
        "prior_year": {"value": 100, "start": "2025-01-01", "end": "2025-03-31"},
    }


def test_a_quarter_is_never_compared_with_a_year_to_date_figure() -> None:
    """A 10-Q tags a 3-month and a 9-month figure ending the same day: each has its own prior year."""
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": _usd(
                    _flow(120, "2025-07-01", "2025-09-30", filed="2025-11-01"),
                    _flow(330, "2025-01-01", "2025-09-30", filed="2025-11-01"),
                    _flow(270, "2024-01-01", "2024-09-30", filed="2025-11-01"),  # only the 9-month prior year is reported
                )
            }
        }
    }

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert values["revenue"]["value"] == 120 and values["revenue"]["start"] == "2025-07-01"
    assert "prior_year" not in values["revenue"]  # 270 covers nine months, not three


def test_a_balance_compares_with_the_balance_a_year_earlier() -> None:
    k1 = {"filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {"facts": {"us-gaap": {"Assets": _usd({"val": 50, "end": "2025-12-31", **k1}, {"val": 40, "end": "2024-12-31", **k1})}}}

    values = sec.statement_values(payload, {"assets": ["Assets"]}, as_of=_AS_OF)

    assert values["assets"] == {"value": 50, "unit": "USD", "filed": "2026-02-01", "form": "10-K", "end": "2025-12-31", "prior_year": {"value": 40, "end": "2024-12-31"}}


def test_the_prior_year_is_point_in_time() -> None:
    """No look-ahead: a prior-year figure first filed after `as_of` (a later restatement) is not read."""
    payload = {"facts": {"us-gaap": {"Revenues": _usd(_flow(120, "2026-01-01", "2026-03-31"), _flow(100, "2025-01-01", "2025-03-31", filed="2026-07-01", accn="later"))}}}

    values = sec.statement_values(payload, {"revenue": ["Revenues"]}, as_of=_AS_OF)

    assert "prior_year" not in values["revenue"]


# --- the agent statements: total debt, cash flow, staleness -------------------------------------------------------


def _q(val: object, end: str = "2026-03-31", filed: str = "2026-05-01", accn: str = "q1") -> dict[str, object]:
    return {"val": val, "end": end, "filed": filed, "form": "10-Q", "accn": accn}


def test_the_balance_sheets_debt_is_total_debt_not_the_current_portion() -> None:
    """VSXY's shape: the agent tool used to show $4M (the current portion) against $3.93B of liabilities."""
    payload = {"facts": {"us-gaap": {"Liabilities": _usd(_q(3_930)), "LongTermDebtCurrent": _usd(_q(4)), "LongTermDebtNoncurrent": _usd(_q(1_200))}}}

    values = sec.balance_sheet(payload, as_of=_AS_OF)

    assert values["debt"] == {"value": 1_204, "unit": "USD", "filed": "2026-05-01", "end": "2026-03-31"}
    assert values["liabilities"]["value"] == 3_930


def test_total_debt_carries_the_balance_a_year_earlier() -> None:
    payload = {"facts": {"us-gaap": {"Assets": _usd(_q(9_000)), "LongTermDebt": _usd(_q(900), _q(800, end="2025-03-31", filed="2025-05-01", accn="q0"))}}}

    assert sec.balance_sheet(payload, as_of=_AS_OF)["debt"]["prior_year"] == {"value": 800, "end": "2025-03-31"}


def test_an_ifrs_filers_debt_is_still_its_borrowings() -> None:
    assert sec.balance_sheet(_ifrs_payload(), as_of=_AS_OF)["debt"]["value"] == 738


def test_a_company_without_a_debt_figure_has_no_debt_field() -> None:
    assert "debt" not in sec.balance_sheet({"facts": {"us-gaap": {"Assets": _usd(_q(9_000))}}}, as_of=_AS_OF)


def test_the_income_statement_adds_cash_flow_and_free_cash_flow() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    before = {"start": "2024-01-01", "end": "2024-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": _usd({"val": 1_000, **year}),
                "NetCashProvidedByUsedInOperatingActivities": _usd({"val": 300, **year}, {"val": 250, **before}),
                "PaymentsToAcquirePropertyPlantAndEquipment": _usd({"val": 100, **year}, {"val": 90, **before}),
            }
        }
    }

    values = sec.income_statement(payload, as_of=_AS_OF)

    assert values["operating_cash_flow"]["value"] == 300 and values["capex"]["value"] == 100
    assert values["free_cash_flow"] == {
        "value": 200,
        "unit": "USD",
        "start": "2025-01-01",
        "end": "2025-12-31",
        "prior_year": {"value": 160, "start": "2024-01-01", "end": "2024-12-31"},
    }


def test_free_cash_flow_needs_both_figures_over_the_same_period() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "k1"}
    payload = {"facts": {"us-gaap": {"NetCashProvidedByUsedInOperatingActivities": _usd({"val": 300, **year})}}}

    assert "free_cash_flow" not in sec.income_statement(payload, as_of=_AS_OF)


def test_an_ifrs_filers_cash_flow_is_read_from_the_ifrs_full_taxonomy() -> None:
    year = {"start": "2025-01-01", "end": "2025-12-31", "filed": "2026-03-26", "form": "20-F", "accn": "a1"}
    payload = {
        "facts": {
            "ifrs-full": {
                "CashFlowsFromUsedInOperatingActivities": {"units": {"EUR": [{"val": 500, **year}]}},
                "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": {"units": {"EUR": [{"val": 120, **year}]}},
            }
        }
    }

    values = sec.income_statement(payload, as_of=_AS_OF)

    assert values["free_cash_flow"]["value"] == 380 and values["free_cash_flow"]["unit"] == "EUR"


def test_a_statement_whose_newest_period_ended_over_18_months_ago_is_stale() -> None:
    """Ticker B (2026-02-17) returned 2011 figures, and the bear called the company's data 'unknowable'."""
    old = {"facts": {"us-gaap": {"Assets": _usd({"val": 1, "end": "2011-03-31", "filed": "2011-04-05", "form": "10-K", "accn": "o"})}}}
    fresh = {"facts": {"us-gaap": {"Assets": _usd(_q(1))}}}

    assert sec.is_stale_statement(sec.balance_sheet(old, as_of=_AS_OF), as_of=_AS_OF)
    assert not sec.is_stale_statement(sec.balance_sheet(fresh, as_of=_AS_OF), as_of=_AS_OF)
    assert not sec.is_stale_statement({}, as_of=_AS_OF)


# --- total debt for 20-F / 40-F filers and non-USD reporters ------------------------------------------------------


def _in(unit: str, *rows: dict[str, object]) -> dict[str, object]:
    return {"units": {unit: list(rows)}}


def _annual_report(val: object, form: str, end: str = "2025-12-31", filed: str = "2026-03-01", accn: str = "a1") -> dict[str, object]:
    return {"val": val, "end": end, "filed": filed, "form": form, "accn": accn}


def test_a_us_gaap_filer_of_a_20f_still_has_debt() -> None:
    """Nebius's shape: us-gaap debt facts exist only under 20-F; the statement used to drop the debt field."""
    payload = {"facts": {"us-gaap": {"Assets": _usd(_annual_report(9_000, "20-F")), "LongTermDebt": _usd(_annual_report(2_400, "20-F"))}}}

    assert sec.balance_sheet(payload, as_of=_AS_OF)["debt"] == {"value": 2_400, "unit": "USD", "filed": "2026-03-01", "end": "2025-12-31"}


def test_a_40f_filers_debt_is_read_too() -> None:
    payload = {"facts": {"us-gaap": {"Assets": _usd(_annual_report(9_000, "40-F")), "LongTermDebt": _usd(_annual_report(700, "40-F"))}}}

    assert sec.balance_sheet(payload, as_of=_AS_OF)["debt"]["value"] == 700


def test_a_non_usd_reporters_debt_carries_its_currency() -> None:
    """Hesai's shape: assets and debt both in CNY; the debt is read in that unit, never converted or dropped."""
    payload = {"facts": {"us-gaap": {"Assets": _in("CNY", _annual_report(9_000, "20-F")), "LongTermDebt": _in("CNY", _annual_report(116, "20-F"))}}}

    values = sec.balance_sheet(payload, as_of=_AS_OF)

    assert values["assets"]["unit"] == "CNY"
    assert values["debt"] == {"value": 116, "unit": "CNY", "filed": "2026-03-01", "end": "2025-12-31"}


def test_debt_in_another_currency_than_the_statements_is_not_mixed_in() -> None:
    payload = {"facts": {"us-gaap": {"Assets": _usd(_annual_report(9_000, "10-K")), "LongTermDebt": _in("CNY", _annual_report(116, "10-K"))}}}

    assert "debt" not in sec.balance_sheet(payload, as_of=_AS_OF)


def test_a_10k_filers_debt_still_resolves_in_usd() -> None:
    payload = {"facts": {"us-gaap": {"Assets": _usd(_annual_report(9_000, "10-K")), "LongTermDebt": _usd(_annual_report(2_400, "10-K"))}}}

    assert sec.balance_sheet(payload, as_of=_AS_OF)["debt"]["unit"] == "USD"


def test_the_screens_total_debt_sources_stay_usd_only() -> None:
    gaap = {"LongTermDebt": _in("CNY", _annual_report(116, "10-K")), "LongTermDebtNoncurrent": _in("CNY", _annual_report(50, "10-K"))}

    assert sec.total_debt_sources(gaap, _ANNUAL) == [[], [], [], [], []]
    assert sec.total_debt_sources(gaap, _ANNUAL, unit="CNY")[0] == [{"end": "2025-12-31", "value": 116, "filed": "2026-03-01"}]


def test_revenue_tagged_including_assessed_tax_is_read() -> None:
    # Hecla (HL) and Lumentum (LITE) tag revenue only as RevenueFromContractWithCustomerIncludingAssessedTax: without it
    # the income statement had operating income but no revenue, so no growth and no margin.
    payload = {"facts": {"us-gaap": {
        **_facts(
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            _flow(1_423_019_000, "2025-01-01", "2025-12-31", "2026-02-17", "10-K", "a26"),
            _flow(930_000_000, "2024-01-01", "2024-12-31", "2026-02-17", "10-K", "a26"),
        ),
        **_facts("OperatingIncomeLoss", _flow(514_795_000, "2025-01-01", "2025-12-31", "2026-02-17", "10-K", "a26")),
    }}}

    values = sec.income_statement(payload, as_of=datetime(2026, 3, 1, tzinfo=UTC))

    assert values["revenue"]["value"] == 1_423_019_000
    assert values["revenue"]["prior_year"]["value"] == 930_000_000


def test_a_filer_tagging_revenue_both_ways_keeps_the_net_figure() -> None:
    # Within one filing and period, the tag order decides: the net-of-tax ASC 606 tag stays ahead of the gross one.
    payload = {"facts": {"us-gaap": {
        **_facts("RevenueFromContractWithCustomerExcludingAssessedTax", _flow(100, "2025-01-01", "2025-12-31", "2026-02-17", "10-K", "a26")),
        **_facts("RevenueFromContractWithCustomerIncludingAssessedTax", _flow(104, "2025-01-01", "2025-12-31", "2026-02-17", "10-K", "a26")),
    }}}

    values = sec.income_statement(payload, as_of=datetime(2026, 3, 1, tzinfo=UTC))

    assert values["revenue"]["value"] == 100

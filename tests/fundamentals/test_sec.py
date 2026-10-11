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

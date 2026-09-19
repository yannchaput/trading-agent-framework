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

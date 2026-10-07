from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest
from tests.fakes import et

from trading_agent_framework.congress.ptr import FilingRef
from trading_agent_framework.congress.source import CongressSource
from trading_agent_framework.utils.errors import CongressDataError, CongressNotFoundError

ANNUAL_TEXT = "SP Apple Inc. (AAPL) [ST] $5,000,001 - $25,000,000 Dividends $1 - $200 Tesla Call (TSLA) [OP] $1,001 - $15,000 None"
PTR_TEXT = "NVIDIA Corp (NVDA) [ST] P 02/03/2025 02/03/2025 $15,001 - $50,000 Option (X) [OP] P 02/03/2025 02/03/2025 $1,001 - $15,000"


def _member(doc_id: str, kind: str, filed: str, year: int, *, first: str = "Nancy", last: str = "Pelosi") -> str:
    return f"<Member><Last>{last}</Last><First>{first}</First><FilingType>{kind}</FilingType><Year>{year}</Year><FilingDate>{filed}</FilingDate><DocID>{doc_id}</DocID></Member>"


def _index(*members: str) -> str:
    return "<FinancialDisclosure>" + "".join(members) + "</FinancialDisclosure>"


class FakeClerk:
    def __init__(self, indexes: dict[int, str], texts: dict[str, str]) -> None:
        self.indexes = indexes
        self.texts = texts
        self.index_calls: list[int] = []
        self.text_calls: list[str] = []

    def year_index_xml(self, year: int, as_of: datetime) -> str:
        self.index_calls.append(year)
        if year not in self.indexes:
            raise CongressNotFoundError(f"no index for {year}")
        return self.indexes[year]

    def filing_text(self, ref: FilingRef) -> str:
        self.text_calls.append(ref.doc_id)
        return self.texts[ref.doc_id]


def _clerk() -> FakeClerk:
    return FakeClerk(
        {
            2025: _index(
                _member("A25", "C", "8/14/2025", 2024),
                _member("P1", "P", "2/20/2025", 2025),
                _member("P2", "P", "3/10/2025", 2025),
                _member("PX", "P", "3/11/2025", 2025, first="Paul"),
            ),
            2024: _index(_member("A24", "C", "8/15/2024", 2023), _member("P0", "P", "6/1/2024", 2024)),
            2026: _index(_member("P3", "P", "1/15/2026", 2026), _member("P4", "P", "9/14/2026", 2026)),
        },
        {"A25": ANNUAL_TEXT, "A24": ANNUAL_TEXT, "P0": PTR_TEXT, "P1": PTR_TEXT, "P2": PTR_TEXT, "P3": PTR_TEXT, "P4": PTR_TEXT},
    )


def _source(clerk: FakeClerk) -> CongressSource:
    return CongressSource(clerk, "Nancy Pelosi")


def test_known_filings_exclude_anything_filed_today_or_later() -> None:
    known = _source(_clerk()).known(et(2026, 9, 14, 10))  # P4 was filed this very day

    assert "P4" not in {r.doc_id for r in known.refs}
    assert "P3" in {r.doc_id for r in known.refs}

    later = _source(_clerk()).known(et(2026, 9, 15, 10))
    assert "P4" in {r.doc_id for r in later.refs}


def test_the_market_date_decides_not_utc() -> None:
    """22:00 ET on the 14th is already the 15th in UTC: a filing dated the 14th is still 'today' in market time."""
    known = _source(_clerk()).known(datetime.fromisoformat("2026-09-14T22:00:00-04:00"))

    assert "P4" not in {r.doc_id for r in known.refs}


def test_newest_yearly_report_is_the_base_and_older_ones_are_ignored() -> None:
    clerk = _clerk()

    known = _source(clerk).known(et(2026, 9, 14, 10))

    assert known.annual_ref.doc_id == "A25"
    assert known.period_end == date(2024, 12, 31)
    assert "A24" not in clerk.text_calls
    assert {a.ticker for a in known.assets} == {"AAPL"}


def test_an_amendment_of_the_newest_reporting_year_wins_and_an_old_year_amendment_does_not() -> None:
    clerk = _clerk()
    clerk.indexes[2026] = _index(
        _member("A25b", "A", "2/1/2026", 2024),  # amends the 2024 report
        _member("A23b", "A", "3/1/2026", 2022),  # a late amendment of an older year
    )
    clerk.texts.update({"A25b": ANNUAL_TEXT, "A23b": ANNUAL_TEXT})

    known = _source(clerk).known(et(2026, 9, 14, 10))

    assert known.annual_ref.doc_id == "A25b"


def test_an_image_only_newest_report_falls_back_to_the_next_one_and_is_counted() -> None:
    clerk = _clerk()
    clerk.texts["A25"] = ""

    known = _source(clerk).known(et(2026, 9, 14, 10))

    assert known.annual_ref.doc_id == "A24"
    assert known.unparsed_filings == 1
    assert known.period_end == date(2023, 12, 31)


def test_ptrs_are_those_filed_after_the_period_end_of_the_chosen_report() -> None:
    known = _source(_clerk()).known(et(2026, 9, 14, 10))

    ptr_ids = {r.doc_id for r in known.refs if r.kind == "ptr"}

    assert ptr_ids == {"P1", "P2", "P3"}  # P0 (2024) is inside the report; PX is another member
    assert {t.doc_id for t in known.transactions} == {"P1", "P2", "P3"}


def test_refs_are_newest_filing_first() -> None:
    known = _source(_clerk()).known(et(2026, 9, 14, 10))

    assert [r.doc_id for r in known.refs] == ["P3", "A25", "P2", "P1"]


def test_no_yearly_report_known_is_an_error_not_an_empty_book() -> None:
    clerk = FakeClerk({2025: _index(_member("P1", "P", "2/20/2025", 2025)), 2024: _index(), 2023: _index()}, {"P1": PTR_TEXT})

    with pytest.raises(CongressDataError, match="yearly report"):
        _source(clerk).known(et(2025, 9, 1, 10))


def test_a_yearly_report_filed_today_is_not_known_yet() -> None:
    clerk = FakeClerk({2025: _index(_member("A25", "C", "8/14/2025", 2024)), 2024: _index(), 2023: _index()}, {"A25": ANNUAL_TEXT})

    with pytest.raises(CongressDataError, match="yearly report"):
        _source(clerk).known(et(2025, 8, 14, 10))
    assert _source(clerk).known(et(2025, 8, 15, 10)).annual_ref.doc_id == "A25"


def test_a_missing_current_year_index_early_in_the_year_is_empty_not_an_error() -> None:
    clerk = _clerk()
    del clerk.indexes[2026]

    known = _source(clerk).known(et(2026, 1, 2, 10))

    assert known.annual_ref.doc_id == "A25"
    assert {r.doc_id for r in known.refs if r.kind == "ptr"} == {"P1", "P2"}


def test_a_missing_older_index_is_an_error() -> None:
    clerk = _clerk()
    del clerk.indexes[2025]

    with pytest.raises(CongressNotFoundError):
        _source(clerk).known(et(2026, 9, 14, 10))


def test_counts_image_only_filings_and_skipped_rows() -> None:
    clerk = _clerk()
    clerk.texts["P2"] = ""

    known = _source(clerk).known(et(2026, 9, 14, 10))

    assert known.unparsed_filings == 1
    assert known.skipped_non_stock == 1 + 1 + 1  # the annual's option, P1's option, P3's option
    assert {t.ticker for t in known.transactions} == {"NVDA"}


def test_a_filing_that_cannot_be_parsed_aborts_the_lookup() -> None:
    clerk = _clerk()
    clerk.texts["P1"] = "layout changed: nothing resembling a transaction row"

    with pytest.raises(CongressDataError):
        _source(clerk).known(et(2026, 9, 14, 10))


def test_the_transactions_carry_their_filing_date() -> None:
    known = _source(_clerk()).known(et(2026, 9, 14, 10))

    nvda = {t.doc_id: t for t in known.transactions}
    assert nvda["P1"].filed == date(2025, 2, 20)
    assert nvda["P1"].amount_low == Decimal(15001)


def test_a_naive_as_of_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _source(_clerk()).known(datetime(2026, 9, 14, 10))


def test_new_filings_are_the_known_doc_ids_minus_the_processed_ones() -> None:
    source = _source(_clerk())
    known = source.known(et(2026, 9, 14, 10))

    assert [r.doc_id for r in source.new_since(known, [])] == ["P3", "A25", "P2", "P1"]
    assert [r.doc_id for r in source.new_since(known, ["A25", "P1", "P2"])] == ["P3"]
    assert source.new_since(known, ["A25", "P1", "P2", "P3", "stale-id"]) == []

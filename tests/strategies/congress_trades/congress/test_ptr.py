from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.strategies.congress_trades.congress import ptr
from trading_agent_framework.strategies.congress_trades.congress.ptr import FilingRef
from trading_agent_framework.utils.errors import CongressDataError

FIXTURES = Path(__file__).parent / "fixtures"
REF = FilingRef(doc_id="20026590", member="Nancy Pelosi", kind="ptr", filed=date(2025, 2, 20), year=2025)


def _member(last="Pelosi", first="Nancy", filing_type="P", filing_date="1/5/2025", doc_id="1", year="2025", prefix="", suffix="") -> str:
    return (
        f"<Member><Prefix>{prefix}</Prefix><Last>{last}</Last><First>{first}</First><Suffix>{suffix}</Suffix>"
        f"<FilingType>{filing_type}</FilingType><StateDst>CA11</StateDst><Year>{year}</Year>"
        f"<FilingDate>{filing_date}</FilingDate><DocID>{doc_id}</DocID></Member>"
    )


def _index(*members: str) -> str:
    return "<?xml version='1.0'?><FinancialDisclosure>" + "".join(members) + "</FinancialDisclosure>"


def _fixture() -> str:
    return (FIXTURES / "ptr_synthetic.txt").read_text(encoding="utf-8")


def _parse(text: str | None = None) -> ptr.PtrParse:
    result = ptr.parse_ptr(_fixture() if text is None else text, REF)
    assert result is not None
    return result


def _by_ticker() -> dict[str, ptr.Transaction]:
    return {t.ticker: t for t in _parse().transactions}


# --- index ----------------------------------------------------------------------------------------


def test_parse_index_returns_only_ptr_and_annual_refs_with_kind() -> None:
    xml = _index(
        _member(filing_type="P", doc_id="10"),
        _member(filing_type="O", doc_id="11", filing_date="5/15/2025", year="2024"),
        _member(filing_type="C", doc_id="12"),
        _member(filing_type="A", doc_id="13"),
        _member(filing_type="X", doc_id="14"),
        _member(filing_type="T", doc_id="15"),
    )

    refs = ptr.parse_index(xml)

    assert [(r.doc_id, r.kind) for r in refs] == [("10", "ptr"), ("11", "annual")]
    annual = refs[1]
    assert annual.filed == date(2025, 5, 15)
    assert annual.year == 2024
    assert annual.member == "Nancy Pelosi"


def test_filing_date_parses_month_first_without_padding() -> None:
    [ref] = ptr.parse_index(_index(_member(filing_date="1/5/2025")))

    assert ref.filed == date(2025, 1, 5)


def test_member_match_is_exact_on_normalized_first_and_last() -> None:
    refs = ptr.parse_index(
        _index(
            _member(doc_id="1", prefix="Hon.", suffix=""),
            _member(doc_id="2", first="Paul"),
            _member(doc_id="3", last="Pelosi-Smith"),
            _member(doc_id="4", first="NANCY", last="pelosi"),
        )
    )

    assert [r.doc_id for r in ptr.filings_for(refs, "Nancy Pelosi")] == ["1", "4"]
    assert [r.doc_id for r in ptr.filings_for(refs, "  nancy   PELOSI ")] == ["1", "4"]


def test_normalize_name_drops_honorifics_and_suffixes() -> None:
    assert ptr.normalize_name("Hon. Nancy  Pelosi, Jr.") == "nancy pelosi"


@pytest.mark.parametrize(
    "xml",
    [
        "not xml at all",
        _index("<Member><Last>Pelosi</Last><First>Nancy</First><FilingType>P</FilingType><Year>2025</Year><FilingDate>1/5/2025</FilingDate></Member>"),
        _index(_member(filing_date="not a date")),
        _index(_member(year="twenty")),
    ],
)
def test_malformed_index_raises_congress_data_error(xml: str) -> None:
    with pytest.raises(CongressDataError):
        ptr.parse_index(xml)


# --- transactions ---------------------------------------------------------------------------------


def test_parses_a_purchase_with_owner_and_range() -> None:
    nvda = _by_ticker()["NVDA"]

    assert nvda.owner == "spouse"
    assert nvda.side == "buy"
    assert nvda.transaction_date == date(2025, 1, 14)
    assert nvda.notification_date == date(2025, 1, 14)
    assert (nvda.amount_low, nvda.amount_high) == (Decimal("250001"), Decimal("500000"))
    assert nvda.doc_id == REF.doc_id
    assert nvda.filed == REF.filed
    assert "NVIDIA" in nvda.asset_name


def test_a_row_without_an_owner_code_belongs_to_the_filer() -> None:
    assert _by_ticker()["AAPL"].owner == "self"


def test_sale_and_partial_sale_are_kept_apart() -> None:
    by = _by_ticker()

    assert by["AAPL"].side == "sell"
    assert by["GOOGL"].side == "sell_partial"


def test_options_bonds_exchanges_are_counted_not_parsed() -> None:
    result = _parse()

    assert {t.ticker for t in result.transactions} == {"NVDA", "AAPL", "GOOGL", "BRK.B", "BF-B", "VST"}
    assert result.skipped_non_stock == 3  # the option, the Treasury bill and the exchange


def test_open_ended_top_band_uses_the_floor_for_both_bounds() -> None:
    vst = _by_ticker()["VST"]

    assert (vst.amount_low, vst.amount_high) == (Decimal("50000000"), Decimal("50000000"))


def test_a_row_split_across_lines_parses() -> None:
    brk = _by_ticker()["BRK.B"]

    assert brk.owner == "joint"
    assert brk.side == "buy"
    assert brk.notification_date == date(2025, 2, 4)


def test_ticker_with_dot_or_dash_parses() -> None:
    by = _by_ticker()

    assert by["BRK.B"].ticker == "BRK.B"
    assert by["BF-B"].ticker == "BF-B"
    assert by["BF-B"].owner == "dependent"


def test_text_with_type_tags_but_no_parseable_row_raises() -> None:
    with pytest.raises(CongressDataError):
        ptr.parse_ptr("Apple Inc. (AAPL) [ST] bought some shares last week", REF)


def test_text_with_no_type_tag_raises() -> None:
    with pytest.raises(CongressDataError):
        ptr.parse_ptr("PERIODIC TRANSACTION REPORT  Name: Hon. Nancy Pelosi  nothing else here", REF)


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_blank_text_is_an_image_only_filing(text: str) -> None:
    assert ptr.parse_ptr(text, REF) is None


def test_a_filing_of_non_stock_rows_only_is_an_empty_result_not_an_error() -> None:
    result = ptr.parse_ptr("US Treasury Bill [GS] P 02/06/2025 02/06/2025 $1,000,001 - $5,000,000", REF)

    assert result is not None
    assert result.transactions == []
    assert result.skipped_non_stock == 1


def test_a_row_with_an_impossible_date_is_skipped_and_counted() -> None:
    text = "Apple Inc. (AAPL) [ST] P 13/45/2025 01/15/2025 $1,001 - $15,000\nNVIDIA Corp (NVDA) [ST] P 01/14/2025 01/14/2025 $1,001 - $15,000"

    result = ptr.parse_ptr(text, REF)

    assert result is not None
    assert [t.ticker for t in result.transactions] == ["NVDA"]
    assert result.skipped_non_stock == 1


# --- the real layout (a PTR filed 2026-10-02) ----------------------------------------------------


def test_real_ptr_with_only_a_non_stock_row_is_an_empty_result() -> None:
    result = ptr.parse_ptr((FIXTURES / "ptr_real_excerpt.txt").read_text(encoding="utf-8"), REF)

    assert result is not None
    assert result.transactions == []
    assert result.skipped_non_stock == 1  # the LLC investment, tagged [AB]


def test_description_lines_between_rows_do_not_pollute_the_next_name() -> None:
    text = (
        "SP NVIDIA Corporation (NVDA) [ST] P 01/14/2026 01/14/2026 $250,001 -\n$500,000\nF S: New\nD: Bought more shares of the AI chip maker in\nthe brokerage account.\n"
        "Apple Inc. (AAPL) [ST] S 01/15/2026 01/15/2026 $1,001 -\n$15,000\n"
    )

    result = ptr.parse_ptr(text, REF)

    assert result is not None
    assert [(t.ticker, t.owner, t.asset_name) for t in result.transactions] == [("NVDA", "spouse", "NVIDIA Corporation"), ("AAPL", "self", "Apple Inc.")]

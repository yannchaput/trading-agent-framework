from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.fundamentals.sec import raw_form4_document
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Filing, Owner, parse_form4
from trading_agent_framework.utils.errors import InsiderDataError

FIXTURES = Path(__file__).parent / "fixtures"
FILED = date(2025, 10, 3)


def _parse(name: str) -> Form4Filing:
    return parse_form4((FIXTURES / name).read_text(), accession="0001-25-000001", filed=FILED, ticker="AAPL")


def test_an_open_market_sale_by_the_ceo():
    f = _parse("sale_ceo.xml")
    (t,) = f.trades
    assert (t.code, t.trade_date, t.shares, t.price, t.owned_after) == ("S", date(2025, 10, 1), Decimal(100), Decimal("255.10"), Decimal(3_280_295))
    assert t.role == "CEO, director" and t.owner_cik == "0001214156" and not t.plan
    assert (t.accession, t.filed, t.ticker, t.owner_name) == ("0001-25-000001", FILED, "AAPL", "COOK TIMOTHY D")
    assert t.dollars == Decimal("25510.00")
    assert t.holding_fraction == Decimal(100) / Decimal(3_280_395)
    assert not f.plan and f.other_codes == 0 and f.unpriced == 0 and f.co_filers == ()


def test_a_purchase_is_measured_against_the_holding_after_it():
    (t,) = _parse("purchase_director.xml").trades
    assert (t.code, t.role, t.shares, t.owned_after) == ("P", "director", Decimal(2000), Decimal(10_000))
    assert t.holding_fraction == Decimal(2000) / Decimal(10_000)


def test_a_purchase_that_opens_a_position_is_a_whole_holding():
    (t,) = _parse("purchase_director.xml").trades
    assert replace(t, owned_after=Decimal(0)).holding_fraction == Decimal(1)


def test_a_joint_filing_counts_its_rows_once_for_the_first_insider():
    f = _parse("joint_fund_then_director.xml")
    assert f.attributed_to is not None and f.attributed_to.name == "DOE JANE" and f.co_filers == ("BIG FUND LP",)
    assert {t.owner_name for t in f.trades} == {"DOE JANE"} and len(f.trades) == 2
    assert [o.name for o in f.owners] == ["BIG FUND LP", "DOE JANE"]


def test_awards_exercises_and_derivative_rows_are_counted_not_kept():
    f = _parse("mixed_codes.xml")  # rows A, M, F, S + one derivative row
    assert [t.code for t in f.trades] == ["S"] and f.other_codes == 4
    assert f.trades[0].role == "CFO"


def test_a_row_without_a_price_is_skipped_and_counted():
    f = _parse("unpriced_row.xml")  # one S without <transactionPricePerShare><value>, one priced S
    assert len(f.trades) == 1 and f.unpriced == 1
    assert f.trades[0].price == Decimal("12.00")


def test_the_10b5_1_box_marks_every_trade_as_plan():
    f = _parse("plan_checked.xml")  # <aff10b5One>true</aff10b5One>
    assert f.plan and len(f.trades) == 2 and all(t.plan for t in f.trades)


@pytest.mark.parametrize(("flag", "expected"), [("1", True), ("true", True), ("TRUE", True), ("0", False), ("false", False), ("", False)])
def test_the_10b5_1_box_accepts_one_and_true(flag: str, expected: bool):
    xml = (FIXTURES / "plan_checked.xml").read_text().replace("<aff10b5One>true</aff10b5One>", f"<aff10b5One>{flag}</aff10b5One>")
    f = parse_form4(xml, accession="x", filed=FILED, ticker="AAPL")
    assert f.plan is expected and all(t.plan is expected for t in f.trades)


def test_a_missing_10b5_1_box_and_footnotes_block_mean_no_plan_and_no_footnotes():
    f = _parse("purchase_director.xml")
    assert not f.plan and f.footnotes == {} and f.trades[0].footnotes == ()


def test_a_filing_by_a_ten_percent_owner_alone_has_no_trades():
    f = _parse("ten_percent_only.xml")
    assert f.attributed_to is None and f.trades == ()
    assert f.co_filers == ("BIG FUND LP",) and [o.role for o in f.owners] == ["10% owner"]


def test_footnotes_referenced_by_a_row_travel_with_the_trade():
    f = _parse("sale_ceo.xml")
    (t,) = f.trades
    assert t.footnotes == ("Sold under a Rule 10b5-1 plan adopted on 2025-03-01.",)
    assert f.footnotes == {"F1": "Sold under a Rule 10b5-1 plan adopted on 2025-03-01."}


def test_a_footnote_referenced_twice_by_a_row_is_kept_once():
    xml = (FIXTURES / "sale_ceo.xml").read_text().replace(
        "<transactionShares><value>100</value></transactionShares>",
        '<transactionShares><value>100</value><footnoteId id="F1"/></transactionShares>',
    )
    (t,) = parse_form4(xml, accession="x", filed=FILED, ticker="AAPL").trades
    assert len(t.footnotes) == 1


@pytest.mark.parametrize(
    ("owner", "role"),
    [
        (Owner("1", "A", True, True, "CEO", False, False), "CEO, director"),
        (Owner("1", "A", True, False, "", False, False), "director"),
        (Owner("1", "A", False, True, "CFO", False, False), "CFO"),
        (Owner("1", "A", False, True, "", False, False), "officer"),
        (Owner("1", "A", True, True, "", False, False), "officer, director"),
        (Owner("1", "A", False, False, "", True, False), "10% owner"),
        (Owner("1", "A", False, False, "", False, True), "other"),
    ],
)
def test_an_owner_role(owner: Owner, role: str):
    assert owner.role == role
    assert owner.is_insider is (owner.is_director or owner.is_officer)


@pytest.mark.parametrize("text", ["<html>Request Rate Threshold Exceeded</html>", "", "<ownershipDocument><unclosed>", "<other/>"])
def test_garbage_raises(text: str):
    with pytest.raises(InsiderDataError):
        parse_form4(text, accession="x", filed=date(2025, 1, 2), ticker="AAPL")


def test_the_issuer_is_parsed_from_the_document():
    f = _parse("sale_ceo.xml")
    assert (f.issuer_cik, f.issuer_symbol) == ("0000320193", "AAPL")


def test_a_missing_issuer_symbol_is_blank():
    xml = (FIXTURES / "sale_ceo.xml").read_text().replace("<issuerTradingSymbol>AAPL</issuerTradingSymbol>", "")
    assert parse_form4(xml, accession="x", filed=FILED, ticker="AAPL").issuer_symbol == ""


@pytest.mark.parametrize("replacement", ["", "<issuerCik>  </issuerCik>"])
def test_a_document_without_an_issuer_cik_raises(replacement: str):
    xml = (FIXTURES / "sale_ceo.xml").read_text().replace("<issuerCik>0000320193</issuerCik>", replacement)
    with pytest.raises(InsiderDataError, match="issuerCik"):
        parse_form4(xml, accession="x", filed=FILED, ticker="AAPL")


def test_another_issuers_filing_is_parsed_the_same_and_left_to_the_caller_to_filter():
    xml = (FIXTURES / "sale_ceo.xml").read_text().replace("0000320193", "0000789019").replace("AAPL", "MSFT")
    f = parse_form4(xml, accession="x", filed=FILED, ticker="AAPL")
    assert (f.issuer_cik, f.issuer_symbol) == ("0000789019", "MSFT")
    assert [t.ticker for t in f.trades] == ["AAPL"] and len(f.trades) == 1


def test_raw_document_strips_the_xsl_folder():
    assert raw_form4_document("xslF345X05/wk-form4_1.xml") == "wk-form4_1.xml"
    assert raw_form4_document("form4.xml") == "form4.xml"

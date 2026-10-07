from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from trading_agent_framework.congress import annual
from trading_agent_framework.congress.ptr import FilingRef
from trading_agent_framework.utils.errors import CongressDataError

FIXTURES = Path(__file__).parent / "fixtures"
REF = FilingRef(doc_id="10063900", member="Nancy Pelosi", kind="annual", filed=date(2025, 8, 14), year=2024)


def _fixture() -> str:
    return (FIXTURES / "annual_synthetic.txt").read_text(encoding="utf-8")


def _parse(text: str | None = None) -> annual.AnnualParse:
    result = annual.parse_annual(_fixture() if text is None else text, REF)
    assert result is not None
    return result


def _by_ticker() -> dict[str, annual.AssetHolding]:
    return {a.ticker: a for a in _parse().assets}


# --- bands and tiers ------------------------------------------------------------------------------


def test_bands_are_ordered_and_contiguous_in_dollars() -> None:
    bands = annual.VALUE_BANDS

    assert [b.low for b in bands] == sorted(b.low for b in bands)
    for below, above in zip(bands, bands[1:], strict=False):
        assert below.high is not None
        assert above.low == below.high + 1  # no gap and no overlap, in whole dollars
    assert bands[-1].high is None


def test_tier_of_a_value_is_the_band_that_contains_it() -> None:
    assert annual.tier_of(Decimal("5000001")) == annual.tier_of(Decimal("25000000"))
    assert annual.tier_of(Decimal("5000001")) == annual.tier_of(Decimal("5000000")) + 1
    assert annual.tier_of(Decimal("1000001")) == annual.tier_of(Decimal("5000000"))
    assert annual.tier_of(Decimal("1000000")) == annual.tier_of(Decimal("1000001")) - 1


def test_tier_of_handles_cents_below_and_above_the_table() -> None:
    top = len(annual.VALUE_BANDS) - 1

    assert annual.tier_of(Decimal("15000.50")) == annual.tier_of(Decimal("15000"))  # a gap value belongs to the band below
    assert annual.tier_of(Decimal("0")) == 0
    assert annual.tier_of(Decimal("-5")) == 0
    assert annual.tier_of(Decimal("999999999")) == top


# --- the yearly report ---------------------------------------------------------------------------


def test_parse_annual_assets_reads_stock_rows_with_owner_and_value_band() -> None:
    by = _by_ticker()
    aapl = by["AAPL"]

    assert aapl.owner == "spouse"
    assert (aapl.value_low, aapl.value_high) == (Decimal("5000001"), Decimal("25000000"))
    assert aapl.tier == annual.tier_of(Decimal("5000001"))
    assert aapl.doc_id == REF.doc_id
    assert "Apple" in aapl.asset_name
    assert by["NVDA"].owner == "self"
    assert by["V"].owner == "dependent"
    assert by["COST"].tier == 0


def test_a_higher_value_band_has_a_higher_tier() -> None:
    by = _by_ticker()

    assert by["V"].tier < by["NVDA"].tier < by["AAPL"].tier < by["GOOGL"].tier


def test_a_row_split_across_lines_parses() -> None:
    googl = _by_ticker()["GOOGL"]

    assert googl.owner == "joint"
    assert (googl.value_low, googl.value_high) == (Decimal("25000001"), Decimal("50000000"))


def test_income_columns_do_not_leak_into_the_next_asset_name() -> None:
    assert "Dividends" not in _by_ticker()["NVDA"].asset_name
    assert "Capital" not in _by_ticker()["GOOGL"].asset_name


def test_over_one_million_spouse_band_maps_to_the_one_to_five_million_tier() -> None:
    vst = _by_ticker()["VST"]

    assert vst.tier == annual.tier_of(Decimal("5000000"))
    assert vst.value_low == vst.value_high == Decimal("1000001")  # the floor only: the real value is unknown


def test_over_fifty_million_is_the_top_tier() -> None:
    amzn = _by_ticker()["AMZN"]

    assert amzn.tier == len(annual.VALUE_BANDS) - 1
    assert amzn.value_low == amzn.value_high == Decimal("50000001")


def test_other_asset_tags_options_and_valueless_rows_are_counted() -> None:
    result = _parse()

    assert {a.ticker for a in result.assets} == {"AAPL", "NVDA", "GOOGL", "V", "VST", "AMZN", "COST"}
    assert result.skipped_non_stock == 3  # the option, the Treasury bill and the row with no value


def test_period_end_is_december_31_of_the_reporting_year() -> None:
    assert annual.period_end(REF) == date(2024, 12, 31)


def test_text_with_type_tags_but_no_parseable_asset_raises() -> None:
    with pytest.raises(CongressDataError):
        annual.parse_annual("Apple Inc. (AAPL) [ST] worth a lot", REF)


def test_text_with_no_type_tag_raises() -> None:
    with pytest.raises(CongressDataError):
        annual.parse_annual("FINANCIAL DISCLOSURE REPORT  Name: Hon. Nancy Pelosi  nothing else", REF)


@pytest.mark.parametrize("text", ["", "  \n "])
def test_blank_text_is_an_image_only_filing(text: str) -> None:
    assert annual.parse_annual(text, REF) is None


def test_a_report_of_non_stock_assets_only_is_empty_not_an_error() -> None:
    result = annual.parse_annual("US Treasury Bill [GS] $250,001 - $500,000 Interest $1,001 - $2,500", REF)

    assert result is not None
    assert result.assets == []
    assert result.skipped_non_stock == 1


# --- the real layout (an excerpt of Nancy Pelosi's report filed 2026-05-15) ------------------------


def _real() -> annual.AnnualParse:
    return _parse((FIXTURES / "annual_real_excerpt.txt").read_text(encoding="utf-8"))


def test_real_excerpt_finds_the_one_stock_among_property_partnership_and_option_rows() -> None:
    result = _real()

    [googl] = result.assets
    assert googl.ticker == "GOOGL"
    assert googl.owner == "spouse"
    assert (googl.value_low, googl.value_high) == (Decimal("5000001"), Decimal("25000000"))
    assert result.skipped_non_stock == 5  # three properties [RP], the partnership [OL] and the option [OP] on the same ticker


def test_real_excerpt_name_is_not_polluted_by_the_description_lines_before_it() -> None:
    [googl] = _real().assets

    assert googl.asset_name == "Alphabet Inc. - Class A"


def test_the_option_on_the_same_ticker_is_not_counted_as_a_stock_holding() -> None:
    assert [a.tier for a in _real().assets] == [annual.tier_of(Decimal("5000001"))]  # only the [ST] row, not the $1M-$5M [OP] row


def test_unread_stock_rows_show_the_stock_tags_that_did_not_become_holdings() -> None:
    unread = annual.unread_stock_rows(_fixture(), REF)

    assert len(unread) == 1
    assert "Broadcom" in unread[0]  # the [ST] row whose value is None


def test_unread_stock_rows_is_empty_when_every_stock_row_parsed() -> None:
    assert annual.unread_stock_rows((FIXTURES / "annual_real_excerpt.txt").read_text(encoding="utf-8"), REF) == []

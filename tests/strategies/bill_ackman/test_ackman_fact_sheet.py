from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from trading_agent_framework.strategies.bill_ackman.fact_sheet import fact_sheet, price_return, unavailable_fact_sheet
from trading_agent_framework.strategies.bill_ackman.screen import Candidate


def _candidate(**overrides: object) -> Candidate:
    values: dict[str, object] = {
        "symbol": "AAA",
        "rank": 1,
        "sic": 5812,
        "market_cap": Decimal("123456789012"),
        "fcf_yield": 0.0456789,
        "fcf_margin": 0.1834567,
        "operating_margin": 0.2512345,
        "operating_margin_stdev": 0.0212345,
        "revenue_growth": 0.0812345,
        "net_debt_to_operating_income": 1.23456,
        "debt_reported": True,
        "fiscal_year_end": date(2025, 12, 31),
        "filed": date(2026, 2, 15),
    }
    values.update(overrides)
    return Candidate(**values)  # type: ignore[arg-type]


def test_a_fact_sheet_has_the_agreed_fields_rounded() -> None:
    sheet = fact_sheet(_candidate(), price=123.456, price_return_12m=0.123456)

    assert sheet == {
        "symbol": "AAA",
        "sic": 5812,
        "market_cap_usd_bn": 123.46,
        "fcf_yield": 0.0457,
        "fcf_margin_5y": 0.1835,
        "operating_margin": 0.2512,
        "operating_margin_stdev": 0.0212,
        "revenue_cagr_5y": 0.0812,
        "net_debt_to_operating_income": 1.23,
        "fiscal_year_end": "2025-12-31",
        "filed": "2026-02-15",
        "price": 123.456,
        "price_return_12m": 0.1235,
    }


def test_a_missing_debt_figure_is_flagged_and_a_reported_one_is_not() -> None:
    assert fact_sheet(_candidate(debt_reported=False), price=1.0, price_return_12m=None)["debt_reported"] is False
    assert "debt_reported" not in fact_sheet(_candidate(), price=1.0, price_return_12m=None)


def test_a_price_fact_that_cannot_be_computed_is_none() -> None:
    sheet = fact_sheet(_candidate(), price=None, price_return_12m=None)

    assert sheet["price"] is None
    assert sheet["price_return_12m"] is None


def test_a_missing_sic_code_is_none() -> None:
    assert fact_sheet(_candidate(sic=None), price=1.0, price_return_12m=None)["sic"] is None


def test_a_holding_the_screen_could_not_describe_carries_only_price_facts_and_the_reason() -> None:
    sheet = unavailable_fact_sheet("hlt", reason="no_data", price=150.5, price_return_12m=-0.05)

    assert sheet == {"symbol": "HLT", "screen_unavailable": "no_data", "price": 150.5, "price_return_12m": -0.05}


def test_price_return_compares_the_last_close_with_the_close_a_year_earlier() -> None:
    closes = [100.0] + [110.0] * 251 + [120.0]  # 253 closes: 252 bars back is the first one

    assert price_return(closes) == pytest.approx(0.2)


@pytest.mark.parametrize("closes", [[], [100.0] * 252])
def test_price_return_is_none_without_enough_history(closes: list[float]) -> None:
    assert price_return(closes) is None


def test_price_return_is_none_when_the_starting_close_is_not_positive() -> None:
    assert price_return([0.0] + [10.0] * 252) is None


def test_price_return_takes_a_shorter_lookback() -> None:
    assert price_return([10.0, 11.0, 12.0, 15.0], lookback=3) == pytest.approx(0.5)

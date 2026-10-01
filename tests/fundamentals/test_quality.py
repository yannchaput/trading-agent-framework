from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from tests.fundamentals.annual_fixtures import FLOW_FIELDS, healthy_figures

from trading_agent_framework.fundamentals.quality import ScreenParams, Survivor, assess

AS_OF = datetime(2026, 6, 1, tzinfo=UTC)
PARAMS = ScreenParams()


def _assess(figures: dict[str, object] | None, as_of: datetime = AS_OF) -> Survivor | str:
    return assess("ACME", figures, as_of=as_of, params=PARAMS)


def _survivor(figures: dict[str, object] | None, as_of: datetime = AS_OF) -> Survivor:
    outcome = _assess(figures, as_of)
    assert isinstance(outcome, Survivor), outcome
    return outcome


def test_a_healthy_company_survives_with_its_metrics() -> None:
    survivor = _survivor(healthy_figures())

    assert survivor.symbol == "ACME"
    assert survivor.fiscal_year_end == date(2025, 12, 31)
    assert survivor.filed == date(2026, 2, 15)
    assert survivor.free_cash_flow == 28
    assert survivor.fcf_margin == pytest.approx(0.2)
    assert survivor.operating_margin == pytest.approx(0.2)
    assert survivor.operating_margin_stdev == pytest.approx(0.0, abs=1e-12)
    assert survivor.revenue_growth == pytest.approx((140 / 100) ** (1 / 4) - 1)
    assert survivor.net_debt_to_operating_income == pytest.approx(40 / 28)
    assert survivor.debt_reported is True
    assert survivor.shares == 1000
    assert survivor.counted_on == date(2026, 1, 31)


def test_no_record_is_no_data() -> None:
    assert _assess(None) == "no_data"


def test_an_absent_record_is_no_data() -> None:
    assert _assess({"status": "absent", "flows": [], "balances": [], "shares": []}) == "no_data"


def test_nothing_filed_before_the_date_is_no_data() -> None:
    assert _assess(healthy_figures(), datetime(2021, 6, 1, tzinfo=UTC)) == "no_data"


def test_a_last_fiscal_year_older_than_18_months_is_stale() -> None:
    assert _assess(healthy_figures(), datetime(2027, 9, 1, tzinfo=UTC)) == "stale_filing"


def test_four_years_of_history_is_insufficient() -> None:
    figures = healthy_figures(drop={(field, 2021) for field in FLOW_FIELDS})

    assert _assess(figures) == "insufficient_history"


def test_a_gap_between_fiscal_years_is_insufficient() -> None:
    assert _assess(healthy_figures(years=(2020, 2022, 2023, 2024, 2025))) == "insufficient_history"


@pytest.mark.parametrize("field", ["operating_income", "operating_cash_flow", "capex"])
def test_a_year_missing_one_field_is_insufficient(field: str) -> None:
    assert _assess(healthy_figures(drop={(field, 2023)})) == "insufficient_history"


@pytest.mark.parametrize("revenue", [0, -5])
def test_a_year_without_positive_revenue_is_insufficient_and_never_divides_by_zero(revenue: int) -> None:
    assert _assess(healthy_figures(revenue=(100, 110, revenue, 130, 140))) == "insufficient_history"


def test_an_operating_loss_in_any_year_is_rejected() -> None:
    assert _assess(healthy_figures(operating_income=(20, 22, -1, 26, 28))) == "operating_loss"


def test_zero_operating_income_counts_as_a_loss() -> None:
    assert _assess(healthy_figures(operating_income=(20, 22, 0, 26, 28))) == "operating_loss"


def test_negative_free_cash_flow_in_any_year_is_rejected() -> None:
    assert _assess(healthy_figures(capex=(5, 5, 40, 5, 5))) == "negative_fcf"


def test_the_first_failed_gate_is_the_reason() -> None:
    figures = healthy_figures(operating_income=(20, 22, -1, 26, 28), capex=(5, 5, 40, 5, 5))

    assert _assess(figures) == "operating_loss"


def test_revenue_up_in_only_two_of_four_years_is_shrinking() -> None:
    assert _assess(healthy_figures(revenue=(100, 90, 85, 95, 140))) == "shrinking_revenue"


def test_revenue_ending_below_where_it_started_is_shrinking() -> None:
    assert _assess(healthy_figures(revenue=(100, 110, 120, 130, 95))) == "shrinking_revenue"


def test_revenue_up_in_three_of_four_years_passes() -> None:
    assert isinstance(_assess(healthy_figures(revenue=(100, 110, 105, 130, 140))), Survivor)


def test_net_debt_above_four_times_operating_income_is_too_much() -> None:
    assert _assess(healthy_figures(debt=200, cash=10)) == "too_much_debt"


def test_net_debt_of_exactly_four_times_operating_income_passes() -> None:
    assert isinstance(_assess(healthy_figures(debt=122, cash=10)), Survivor)


def test_net_cash_passes_with_a_negative_multiple() -> None:
    survivor = _survivor(healthy_figures(debt=0, cash=56))

    assert survivor.net_debt_to_operating_income == pytest.approx(-2.0)


def test_a_missing_debt_figure_counts_as_zero_and_is_flagged() -> None:
    survivor = _survivor(healthy_figures(debt=None, cash=10))

    assert survivor.debt_reported is False
    assert survivor.net_debt_to_operating_income == pytest.approx(-10 / 28)


def test_missing_cash_counts_as_zero() -> None:
    survivor = _survivor(healthy_figures(debt=50, cash=None))

    assert survivor.net_debt_to_operating_income == pytest.approx(50 / 28)


def test_a_filing_dated_today_is_not_known_yet() -> None:
    # The 2025 annual report is filed on 2026-02-15; SEC gives no time of day.
    assert _assess(healthy_figures(), datetime(2026, 2, 15, 23, 0, tzinfo=UTC)) == "insufficient_history"
    assert isinstance(_assess(healthy_figures(), datetime(2026, 2, 16, 0, 1, tzinfo=UTC)), Survivor)


def test_a_restatement_filed_after_the_date_is_ignored() -> None:
    figures = healthy_figures()
    figures["flows"].append({"field": "revenue", "start": "2025-01-01", "end": "2025-12-31", "value": 150, "filed": "2026-08-01"})

    before = _survivor(figures, datetime(2026, 6, 1, tzinfo=UTC))
    after = _survivor(figures, datetime(2026, 9, 1, tzinfo=UTC))

    assert before.operating_margin == pytest.approx(28 / 140)
    assert after.operating_margin == pytest.approx(28 / 150)
    assert after.filed == date(2026, 8, 1)


def test_the_share_count_is_the_latest_one_known_on_the_date() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15"},
        {"end": "2026-04-30", "value": 1100, "filed": "2026-05-10"},
        {"end": "2026-07-31", "value": 1200, "filed": "2026-08-10"},
    ]

    survivor = _survivor(figures)

    assert (survivor.shares, survivor.counted_on) == (1100, date(2026, 4, 30))


def test_the_cover_page_count_wins_a_tie_on_the_period_end() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15"},  # cover page: listed first by annual_figures
        {"end": "2026-01-31", "value": 990, "filed": "2026-02-15"},
    ]

    assert _survivor(figures).shares == 1000


def test_a_company_with_no_share_count_still_survives_the_numeric_gates() -> None:
    survivor = _survivor(healthy_figures(shares=None))

    assert (survivor.shares, survivor.counted_on) == (None, None)

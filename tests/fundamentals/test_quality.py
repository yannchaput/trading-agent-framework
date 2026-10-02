from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from tests.fundamentals.annual_fixtures import FLOW_FIELDS, healthy_figures

from trading_agent_framework.fundamentals.quality import Priced, ScreenParams, Survivor, assess, rank, sector_excluded

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


def test_operating_income_above_revenue_in_any_year_is_implausible_figures() -> None:
    # A partial revenue tag or a tagging error: the margins would be nonsense.
    assert _assess(healthy_figures(operating_income=(20, 22, 130, 26, 28))) == "insufficient_history"


def test_free_cash_flow_above_revenue_in_any_year_is_implausible_figures() -> None:
    assert _assess(healthy_figures(operating_cash_flow=(25, 27, 29, 31, 150))) == "insufficient_history"


def test_operating_income_equal_to_revenue_is_not_implausible() -> None:
    assert isinstance(_assess(healthy_figures(operating_income=(20, 22, 120, 26, 28))), Survivor)


def test_implausible_figures_are_reported_before_a_loss_gate() -> None:
    figures = healthy_figures(operating_income=(20, -4, 130, 26, 28))

    assert _assess(figures) == "insufficient_history"


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


def test_a_missing_debt_figure_in_every_year_is_still_flagged_not_rejected() -> None:
    survivor = _survivor(healthy_figures(debt_by_year={}, cash=10))

    assert survivor.debt_reported is False


def test_debt_reported_in_the_latest_year_is_unchanged_by_earlier_years() -> None:
    survivor = _survivor(healthy_figures(debt_by_year={2023: 10, 2025: 50}, cash=10))

    assert survivor.debt_reported is True
    assert survivor.net_debt_to_operating_income == pytest.approx(40 / 28)


def test_no_debt_in_the_latest_year_but_some_in_an_earlier_one_is_unknown() -> None:
    assert _assess(healthy_figures(debt_by_year={2022: 50, 2023: 50}, cash=10)) == "debt_unknown"


def test_debt_unknown_is_checked_just_before_too_much_debt() -> None:
    # latest-year debt present and huge: too_much_debt; absent with an earlier year present: debt_unknown
    assert _assess(healthy_figures(debt_by_year={2025: 900}, cash=10)) == "too_much_debt"
    assert _assess(healthy_figures(debt_by_year={2021: 900}, cash=10)) == "debt_unknown"


def test_debt_filed_after_the_date_does_not_make_an_earlier_year_count() -> None:
    figures = healthy_figures(debt_by_year={2022: 50}, cash=10)
    figures["balances"][0]["filed"] = "2026-07-01"

    assert _survivor(figures).debt_reported is False


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
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15", "kind": "cover"},
        {"end": "2026-04-30", "value": 1100, "filed": "2026-05-10", "kind": "cover"},
        {"end": "2026-07-31", "value": 1200, "filed": "2026-08-10", "kind": "cover"},
    ]

    survivor = _survivor(figures)

    assert (survivor.shares, survivor.counted_on) == (1100, date(2026, 4, 30))


def test_the_cover_page_count_wins_a_tie_on_the_period_end() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15", "kind": "cover"},
        {"end": "2026-01-31", "value": 990, "filed": "2026-02-15", "kind": "weighted"},
    ]

    assert _survivor(figures).shares == 1000


def test_the_cover_page_count_wins_a_full_tie_whatever_the_list_order() -> None:
    figures = healthy_figures()
    figures["shares"] = [
        {"end": "2026-01-31", "value": 990, "filed": "2026-02-15", "kind": "weighted"},
        {"end": "2026-01-31", "value": 1000, "filed": "2026-02-15", "kind": "cover"},
    ]

    assert _survivor(figures).shares == 1000


def test_a_cover_page_count_is_counted_on_its_period_end() -> None:
    figures = healthy_figures()
    figures["shares"] = [{"end": "2026-04-20", "value": 1005, "filed": "2026-05-01", "kind": "cover"}]

    assert _survivor(figures).counted_on == date(2026, 4, 20)


def test_a_weighted_average_count_is_counted_on_its_filing_date() -> None:
    # ASC 260 restates a weighted-average count for splits between the period end and issuance, so
    # the count already reflects every split up to the day it was filed.
    figures = healthy_figures()
    figures["shares"] = [{"end": "2026-03-31", "value": 1010, "filed": "2026-05-01", "kind": "weighted"}]

    survivor = _survivor(figures)

    assert (survivor.shares, survivor.counted_on) == (1010, date(2026, 5, 1))


def test_a_company_with_no_share_count_still_survives_the_numeric_gates() -> None:
    survivor = _survivor(healthy_figures(shares=None))

    assert (survivor.shares, survivor.counted_on) == (None, None)


def _priced(symbol: str, *, market_cap: str, fcf: int = 28, fcf_margin: float = 0.2, stdev: float = 0.01, sic: int | None = 3571) -> Priced:
    survivor = Survivor(
        symbol=symbol,
        fiscal_year_end=date(2025, 12, 31),
        filed=date(2026, 2, 15),
        free_cash_flow=fcf,
        fcf_margin=fcf_margin,
        operating_margin=0.25,
        operating_margin_stdev=stdev,
        revenue_growth=0.08,
        net_debt_to_operating_income=1.5,
        debt_reported=True,
        shares=1000,
        counted_on=date(2026, 1, 31),
    )
    return Priced(survivor=survivor, sic=sic, market_cap=Decimal(market_cap))


@pytest.mark.parametrize(("sic", "excluded"), [(6021, True), (6000, True), (6799, True), (4911, True), (6800, False), (4899, False), (3571, False), (None, False)])
def test_sector_excluded_covers_utilities_and_finance_and_lets_an_unknown_code_pass(sic: int | None, excluded: bool) -> None:
    assert sector_excluded(sic, PARAMS) is excluded


def test_rank_scores_by_weighted_percentiles() -> None:
    priced = [
        _priced("AAA", market_cap="280", fcf_margin=0.2, stdev=0.00),  # yield 0.100: pct 1.0 | margin pct 0.5 | stdev pct 0.0
        _priced("BBB", market_cap="560", fcf_margin=0.3, stdev=0.02),  # yield 0.050: pct 0.5 | margin pct 1.0 | stdev pct 1.0
        _priced("CCC", market_cap="1120", fcf_margin=0.1, stdev=0.01),  # yield 0.025: pct 0.0 | margin pct 0.0 | stdev pct 0.5
    ]

    candidates = rank(priced, PARAMS)

    assert [(c.symbol, c.rank) for c in candidates] == [("AAA", 1), ("BBB", 2), ("CCC", 3)]
    assert [c.score for c in candidates] == pytest.approx([0.4 * 1.0 + 0.3 * 0.5 + 0.3 * 1.0, 0.4 * 0.5 + 0.3 * 1.0 + 0.3 * 0.0, 0.3 * 0.5])
    assert [c.fcf_yield for c in candidates] == pytest.approx([0.1, 0.05, 0.025])


def test_rank_carries_the_survivor_metrics_into_the_candidate() -> None:
    (candidate,) = rank([_priced("AAA", market_cap="280", sic=5812)], PARAMS)

    assert candidate.sic == 5812
    assert candidate.market_cap == Decimal("280")
    assert candidate.fcf_margin == 0.2
    assert candidate.operating_margin == 0.25
    assert candidate.operating_margin_stdev == 0.01
    assert candidate.revenue_growth == 0.08
    assert candidate.net_debt_to_operating_income == 1.5
    assert candidate.debt_reported is True
    assert candidate.fiscal_year_end == date(2025, 12, 31)
    assert candidate.filed == date(2026, 2, 15)


def test_a_single_survivor_gets_percentile_one_on_every_metric() -> None:
    (candidate,) = rank([_priced("AAA", market_cap="280")], PARAMS)

    assert candidate.score == pytest.approx(0.4 + 0.3 + 0.3 * (1 - 1.0))


def test_tied_metrics_share_the_average_percentile() -> None:
    # Same yield (28/280 and 56/560), same margin, same stdev: every percentile is 0.5 for both.
    candidates = rank([_priced("AAA", market_cap="280", fcf=28), _priced("BBB", market_cap="560", fcf=56)], PARAMS)

    assert [c.score for c in candidates] == pytest.approx([0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5] * 2)


def test_equal_scores_are_ordered_by_market_cap_then_symbol() -> None:
    priced = [
        _priced("ZZZ", market_cap="280", fcf=28),
        _priced("MMM", market_cap="560", fcf=56),
        _priced("AAA", market_cap="280", fcf=28),
    ]

    assert [c.symbol for c in rank(priced, PARAMS)] == ["MMM", "AAA", "ZZZ"]


def test_rank_returns_at_most_top_n() -> None:
    priced = [_priced("AAA", market_cap="280"), _priced("BBB", market_cap="560"), _priced("CCC", market_cap="1120")]

    candidates = rank(priced, ScreenParams(top_n=2))

    assert [(c.symbol, c.rank) for c in candidates] == [("AAA", 1), ("BBB", 2)]


def test_rank_of_nothing_is_empty() -> None:
    assert rank([], PARAMS) == []

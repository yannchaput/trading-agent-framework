from __future__ import annotations

from datetime import date
from decimal import Decimal

from trading_agent_framework.strategies.insider_trades.form4.classify import Label, label_trades
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Trade


def trade(day: date, *, cik: str = "0001", name: str = "COOK TIMOTHY D", code: str = "S", plan: bool = False) -> Form4Trade:
    return Form4Trade(
        accession=f"0001-{day.isoformat()}",
        filed=day,
        ticker="AAPL",
        owner_cik=cik,
        owner_name=name,
        role="CEO",
        code=code,
        trade_date=day,
        shares=Decimal(100),
        price=Decimal(10),
        owned_after=Decimal(1000),
        plan=plan,
        footnotes=(),
    )


def history(*days: date, **kwargs) -> list[Form4Trade]:
    return [trade(d, **kwargs) for d in days]


def test_same_month_three_years_running_is_routine():
    h = history(date(2022, 3, 4), date(2023, 3, 20), date(2024, 3, 1))
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.ROUTINE and label.rule == "routine: traded in March in each of 2022, 2023, 2024"
    assert label.months_by_year == {2022: (3,), 2023: (3,), 2024: (3,)}


def test_every_year_but_never_the_same_month_is_opportunistic():
    h = history(date(2022, 1, 4), date(2023, 5, 4), date(2024, 11, 4))
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.OPPORTUNISTIC
    assert label.rule == "opportunistic: traded in each of 2022, 2023, 2024, never in the same month every year"
    assert label.months_by_year == {2022: (1,), 2023: (5,), 2024: (11,)}


def test_a_missing_year_is_unclassified():
    h = history(date(2022, 3, 4), date(2024, 3, 4))
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.UNCLASSIFIED and label.rule == "unclassified: no trade in 2023"
    assert label.months_by_year == {2022: (3,), 2023: (), 2024: (3,)}


def test_several_missing_years_are_listed_ascending():
    (label,) = label_trades([trade(date(2025, 6, 2))], history(date(2023, 3, 4)), years=3)
    assert label.rule == "unclassified: no trade in 2022, 2024"


def test_no_history_at_all_is_unclassified():
    (label,) = label_trades([trade(date(2025, 6, 2))], [], years=3)
    assert label.label is Label.UNCLASSIFIED and label.rule == "unclassified: no trade in 2022, 2023, 2024"
    assert label.months_by_year == {2022: (), 2023: (), 2024: ()}


def test_a_plan_trade_is_routine_whatever_the_history():
    h = history(date(2022, 1, 4), date(2023, 5, 4), date(2024, 11, 4))
    (label,) = label_trades([trade(date(2025, 6, 2), plan=True)], h, years=3)
    assert label.label is Label.ROUTINE and label.rule == "plan: the 10b5-1 box is checked"
    assert label.months_by_year == {2022: (1,), 2023: (5,), 2024: (11,)}


def test_a_plan_trade_with_no_history_is_still_routine():
    (label,) = label_trades([trade(date(2025, 6, 2), plan=True)], [], years=3)
    assert label.label is Label.ROUTINE and label.months_by_year == {2022: (), 2023: (), 2024: ()}


def test_trades_either_side_of_new_year_use_their_own_years():
    h = history(date(2022, 3, 4), date(2023, 3, 4), date(2024, 3, 4), date(2025, 8, 4))
    labels = label_trades([trade(date(2025, 12, 20)), trade(date(2026, 1, 5))], h, years=3)
    assert [x.label for x in labels] == [Label.ROUTINE, Label.OPPORTUNISTIC]
    assert [tuple(x.months_by_year) for x in labels] == [(2022, 2023, 2024), (2023, 2024, 2025)]


def test_history_is_keyed_on_the_owner_cik_not_the_name():
    h = [
        trade(date(2022, 3, 4), name="COOK TIMOTHY D"),
        trade(date(2023, 3, 4), name="Cook Tim"),
        trade(date(2024, 3, 4), name="Cook, Timothy"),
    ]
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.ROUTINE


def test_another_insiders_trades_do_not_count():
    h = history(date(2022, 3, 4), date(2023, 3, 4), date(2024, 3, 4), cik="0002")
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.UNCLASSIFIED


def test_buys_and_sales_both_count_as_history():
    h = [trade(date(2022, 3, 4), code="P"), trade(date(2023, 3, 4), code="S"), trade(date(2024, 3, 4), code="P")]
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.label is Label.ROUTINE


def test_trades_passed_in_are_not_part_of_the_history():
    trades = history(date(2022, 3, 4), date(2023, 3, 4), date(2024, 3, 4), date(2025, 3, 4))
    labels = label_trades(trades, [], years=3)
    assert {x.label for x in labels} == {Label.UNCLASSIFIED}


def test_trades_of_the_same_year_or_later_are_not_look_back():
    h = history(date(2022, 3, 4), date(2023, 3, 4), date(2024, 3, 4), date(2025, 1, 4), date(2026, 3, 4))
    (label,) = label_trades([trade(date(2025, 6, 2))], h, years=3)
    assert label.months_by_year == {2022: (3,), 2023: (3,), 2024: (3,)}


def test_the_earliest_shared_month_is_named_whatever_the_history_order():
    days = [date(2022, 9, 1), date(2022, 2, 1), date(2023, 9, 1), date(2023, 2, 1), date(2024, 2, 1), date(2024, 9, 1)]
    forward = label_trades([trade(date(2025, 6, 2))], history(*days), years=3)
    backward = label_trades([trade(date(2025, 6, 2))], history(*reversed(days)), years=3)
    assert forward == backward
    assert forward[0].rule == "routine: traded in February in each of 2022, 2023, 2024"
    assert forward[0].months_by_year == {2022: (2, 9), 2023: (2, 9), 2024: (2, 9)}


def test_months_are_distinct_and_sorted():
    h = history(date(2022, 7, 1), date(2022, 7, 9), date(2022, 2, 1))
    (label,) = label_trades([trade(date(2023, 6, 2))], h, years=1)
    assert label.months_by_year == {2022: (2, 7)}
    assert label.label is Label.ROUTINE and label.rule == "routine: traded in February in each of 2022"


def test_one_label_per_trade_in_input_order():
    h = history(date(2022, 3, 4), date(2023, 3, 4), date(2024, 3, 4))
    other = trade(date(2025, 5, 1), cik="0002")
    mine = trade(date(2025, 6, 2))
    labels = label_trades([other, mine], h, years=3)
    assert [x.trade for x in labels] == [other, mine]
    assert [x.label for x in labels] == [Label.UNCLASSIFIED, Label.ROUTINE]

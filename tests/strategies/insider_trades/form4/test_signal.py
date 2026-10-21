from __future__ import annotations

from datetime import date
from decimal import Decimal

from trading_agent_framework.strategies.insider_trades.form4.classify import Label, Labelled
from trading_agent_framework.strategies.insider_trades.form4.parse import Form4Trade
from trading_agent_framework.strategies.insider_trades.form4.signal import (
    Override,
    SignalClass,
    apply_overrides,
    signal_class,
    summarise,
    window_trades,
)

TODAY = date(2025, 10, 14)
MIN_BUY = Decimal("0.01")
MIN_SELL = Decimal("0.10")


def labelled(
    label: Label,
    *,
    code: str = "S",
    shares: int = 100,
    price: str = "10",
    owned_after: int = 900,
    cik: str = "0001",
    name: str = "COOK TIMOTHY D",
    filed: date = date(2025, 10, 1),
    plan: bool = False,
    accession: str = "0001-a",
) -> Labelled:
    trade = Form4Trade(
        accession=accession,
        filed=filed,
        ticker="AAPL",
        owner_cik=cik,
        owner_name=name,
        role="CEO",
        code=code,
        trade_date=filed,
        shares=Decimal(shares),
        price=Decimal(price),
        owned_after=Decimal(owned_after),
        plan=plan,
        footnotes=(),
    )
    return Labelled(trade, label, f"{label}: test", {})


def sale(label: Label, fraction_percent: int, **kwargs) -> Labelled:
    """A sale moving `fraction_percent` % of the holding before it."""
    return labelled(label, code="S", shares=fraction_percent, owned_after=100 - fraction_percent, **kwargs)


def purchase(label: Label, shares: int, owned_after: int, **kwargs) -> Labelled:
    return labelled(label, code="P", shares=shares, owned_after=owned_after, **kwargs)


def classify(*window: Labelled) -> SignalClass:
    facts = summarise("AAPL", window, today=TODAY, window_start=date(2025, 9, 14))
    return signal_class(facts, min_buy_fraction=MIN_BUY, min_sell_fraction=MIN_SELL)


def test_the_window_is_the_30_days_before_today_by_filing_date():
    trades = [
        labelled(Label.ROUTINE, filed=date(2025, 9, 13), accession="too-old"),
        labelled(Label.ROUTINE, filed=date(2025, 9, 14), accession="first-day"),
        labelled(Label.ROUTINE, filed=date(2025, 10, 13), accession="yesterday"),
        labelled(Label.ROUTINE, filed=date(2025, 10, 14), accession="today"),
    ]
    kept = window_trades(trades, today=TODAY, window_days=30)
    assert [t.trade.accession for t in kept] == ["first-day", "yesterday"]


def test_one_opportunistic_buy_above_the_threshold_is_buying_even_against_bigger_sales():
    buy = purchase(Label.OPPORTUNISTIC, shares=2, owned_after=100, price="25000")  # $50k, fraction 0.02
    big_sale = labelled(Label.OPPORTUNISTIC, shares=30, owned_after=70, price="66666.67", cik="0002")  # ~$2M, fraction 0.30
    assert classify(buy, big_sale) is SignalClass.BUYING


def test_a_negligible_opportunistic_buy_does_not_count():
    assert classify(purchase(Label.OPPORTUNISTIC, shares=5, owned_after=995)) is SignalClass.NONE  # 0.005 < 0.01


def test_heavy_opportunistic_selling_is_selling():
    assert classify(sale(Label.OPPORTUNISTIC, 12)) is SignalClass.SELLING


def test_light_opportunistic_selling_is_none():
    assert classify(sale(Label.OPPORTUNISTIC, 5)) is SignalClass.NONE


def test_routine_and_unclassified_dollars_never_move_the_class():
    assert classify(purchase(Label.ROUTINE, shares=500, owned_after=1000), sale(Label.UNCLASSIFIED, 90)) is SignalClass.NONE


def test_selling_needs_more_dollars_sold_than_bought_by_opportunistic_insiders():
    buy = purchase(Label.OPPORTUNISTIC, shares=1, owned_after=1000, price="1000000", cik="0002")  # fraction 0.001, $1M
    assert classify(sale(Label.OPPORTUNISTIC, 12), buy) is SignalClass.NONE


def test_an_override_relabels_one_insiders_trades_of_one_label_but_never_a_plan_trade():
    opportunistic = sale(Label.OPPORTUNISTIC, 12, cik="0009", accession="opp")
    planned = sale(Label.ROUTINE, 12, cik="0009", plan=True, accession="planned")
    other = sale(Label.OPPORTUNISTIC, 12, cik="0008", accession="other")
    trades = [opportunistic, planned]
    assert classify(*trades) is SignalClass.SELLING

    # a plan trade is never relabelled, and nobody else's trade matches the insider
    assert apply_overrides(trades, [Override("0009", Label.ROUTINE, Label.OPPORTUNISTIC, "no")]) == trades

    reason = "plan named in footnote F2"
    result = apply_overrides([opportunistic, planned, other], [Override("0009", Label.OPPORTUNISTIC, Label.ROUTINE, reason)])
    assert [r.label for r in result] == [Label.ROUTINE, Label.ROUTINE, Label.OPPORTUNISTIC]
    assert result[0].rule == f"override: {reason}"
    assert result[1] is planned and result[2] is other
    assert classify(result[0], result[1]) is SignalClass.NONE
    assert opportunistic.label is Label.OPPORTUNISTIC  # the input is not mutated


def test_overrides_apply_in_order():
    first = sale(Label.OPPORTUNISTIC, 12, cik="0009")
    result = apply_overrides(
        [first],
        [
            Override("0009", Label.OPPORTUNISTIC, Label.ROUTINE, "one"),
            Override("0009", Label.ROUTINE, Label.UNCLASSIFIED, "two"),
        ],
    )
    assert [(r.label, r.rule) for r in result] == [(Label.UNCLASSIFIED, "override: two")]


def test_the_summary_splits_dollars_by_label():
    window = [
        purchase(Label.OPPORTUNISTIC, shares=2, owned_after=100, price="25000"),
        labelled(Label.ROUTINE, shares=100, price="1200", owned_after=900, accession="r"),
        sale(Label.OPPORTUNISTIC, 10, accession="o1"),
        sale(Label.OPPORTUNISTIC, 9, accession="o2"),
    ]
    facts = summarise("AAPL", window, today=TODAY, window_start=date(2025, 9, 14))
    (summary,) = facts.insiders
    assert summary.bought == {Label.OPPORTUNISTIC: Decimal("50000.00")}
    assert summary.sold[Label.ROUTINE] == Decimal("120000.00")
    assert summary.sold[Label.OPPORTUNISTIC] == Decimal("190")
    assert summary.max_buy_fraction == Decimal("2") / Decimal("100")
    assert summary.sell_fraction == Decimal("0.19")
    assert (summary.insider_id, summary.name, summary.role) == ("0001", "COOK TIMOTHY D", "CEO")
    assert summary.labelled == tuple(window)
    assert facts.accessions == frozenset({"0001-a", "r", "o1", "o2"})


def test_insiders_are_ordered_by_total_dollars_then_name():
    window = [
        labelled(Label.ROUTINE, cik="0001", name="SMALL", price="1"),
        labelled(Label.ROUTINE, cik="0002", name="BIG", price="100"),
        labelled(Label.ROUTINE, cik="0003", name="AARDVARK", price="1"),
    ]
    facts = summarise("AAPL", window, today=TODAY, window_start=date(2025, 9, 14))
    assert [i.name for i in facts.insiders] == ["BIG", "AARDVARK", "SMALL"]


def test_an_empty_window_summarises_to_no_insiders_and_no_signal():
    facts = summarise("AAPL", [], today=TODAY, window_start=date(2025, 9, 14))
    assert facts.insiders == () and facts.accessions == frozenset()
    assert signal_class(facts, min_buy_fraction=MIN_BUY, min_sell_fraction=MIN_SELL) is SignalClass.NONE

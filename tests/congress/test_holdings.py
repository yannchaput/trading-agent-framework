from __future__ import annotations

from datetime import date
from decimal import Decimal

from trading_agent_framework.congress import holdings
from trading_agent_framework.congress.annual import AssetHolding, tier_of
from trading_agent_framework.congress.ptr import Transaction

PERIOD_END = date(2024, 12, 31)


def asset(ticker: str, low: int, high: int, *, owner: str = "self", doc_id: str = "A1") -> AssetHolding:
    return AssetHolding(doc_id=doc_id, owner=owner, ticker=ticker, asset_name=f"{ticker} Inc.", value_low=Decimal(low), value_high=Decimal(high), tier=tier_of(Decimal(low)))


def trade(
    ticker: str,
    side: str,
    low: int,
    high: int,
    *,
    on: date = date(2025, 2, 3),
    owner: str = "self",
    doc_id: str = "P1",
    filed: date = date(2025, 2, 20),
) -> Transaction:
    return Transaction(
        doc_id=doc_id,
        owner=owner,
        ticker=ticker,
        asset_name=f"{ticker} Inc.",
        side=side,
        transaction_date=on,
        notification_date=on,
        amount_low=Decimal(low),
        amount_high=Decimal(high),
        filed=filed,
    )


def _by_ticker(result: list[holdings.Holding]) -> dict[str, holdings.Holding]:
    return {h.ticker: h for h in result}


# --- reconstruct ----------------------------------------------------------------------------------


def test_annual_assets_alone_give_holdings_at_the_band_midpoint() -> None:
    [aapl] = holdings.reconstruct([asset("AAPL", 5_000_001, 25_000_000)], [], period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(5_000_001), Decimal(25_000_000))
    assert aapl.midpoint == Decimal("15000000.5")
    assert aapl.tier == tier_of(Decimal(5_000_001))
    assert aapl.sources == ("A1",)


def test_ptr_buy_after_period_end_adds_the_range() -> None:
    [aapl] = holdings.reconstruct([asset("AAPL", 100_001, 250_000)], [trade("AAPL", "buy", 15_001, 50_000)], period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(115_002), Decimal(300_000))
    assert aapl.sources == ("A1", "P1")


def test_ptr_trade_dated_on_or_before_period_end_is_ignored() -> None:
    """The yearly report already contains it (disclosure lag), so applying it would count the trade twice."""
    old = [trade("AAPL", "buy", 15_001, 50_000, on=PERIOD_END, filed=date(2025, 1, 20)), trade("AAPL", "sell", 15_001, 50_000, on=date(2024, 12, 30))]

    [aapl] = holdings.reconstruct([asset("AAPL", 100_001, 250_000)], old, period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(100_001), Decimal(250_000))
    assert aapl.sources == ("A1",)


def test_ptr_filed_before_the_annual_report_but_traded_after_period_end_is_applied() -> None:
    """Filing date is irrelevant here: a PTR filed in March for a February trade belongs to a report filed in August."""
    early = trade("AAPL", "buy", 15_001, 50_000, on=date(2025, 2, 3), filed=date(2025, 3, 1))

    [aapl] = holdings.reconstruct([asset("AAPL", 100_001, 250_000)], [early], period_end=PERIOD_END)

    assert aapl.value_high == Decimal(300_000)


def test_full_sale_sets_the_holding_to_zero_and_removes_it() -> None:
    result = holdings.reconstruct([asset("AAPL", 100_001, 250_000), asset("MSFT", 15_001, 50_000)], [trade("AAPL", "sell", 100_001, 250_000)], period_end=PERIOD_END)

    assert [h.ticker for h in result] == ["MSFT"]


def test_partial_sale_subtracts_the_range() -> None:
    [aapl] = holdings.reconstruct([asset("AAPL", 100_001, 250_000)], [trade("AAPL", "sell_partial", 15_001, 50_000)], period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(50_001), Decimal(234_999))


def test_partial_sale_floors_the_low_bound_at_zero_and_keeps_the_holding() -> None:
    [aapl] = holdings.reconstruct([asset("AAPL", 15_001, 50_000)], [trade("AAPL", "sell_partial", 15_001, 50_000)], period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(0), Decimal(34_999))


def test_partial_sale_that_can_only_empty_the_holding_removes_it() -> None:
    assert holdings.reconstruct([asset("AAPL", 15_001, 50_000)], [trade("AAPL", "sell_partial", 250_001, 500_000)], period_end=PERIOD_END) == []


def test_a_buy_of_a_ticker_not_in_the_annual_report_creates_a_holding() -> None:
    [nvda] = holdings.reconstruct([], [trade("NVDA", "buy", 250_001, 500_000, owner="spouse")], period_end=PERIOD_END)

    assert nvda.ticker == "NVDA"
    assert (nvda.value_low, nvda.value_high) == (Decimal(250_001), Decimal(500_000))
    assert nvda.sources == ("P1",)


def test_a_sale_of_something_never_held_is_ignored() -> None:
    assert [h.ticker for h in holdings.reconstruct([asset("MSFT", 15_001, 50_000)], [trade("AAPL", "sell", 15_001, 50_000)], period_end=PERIOD_END)] == ["MSFT"]


def test_tier_is_recomputed_from_the_estimated_value() -> None:
    before = holdings.reconstruct([asset("NVDA", 1_000_001, 5_000_000)], [], period_end=PERIOD_END)
    after = holdings.reconstruct([asset("NVDA", 1_000_001, 5_000_000)], [trade("NVDA", "buy", 5_000_001, 25_000_000)], period_end=PERIOD_END)

    assert after[0].tier == before[0].tier + 1


def test_two_owners_of_the_same_ticker_add_up() -> None:
    [aapl] = holdings.reconstruct([asset("AAPL", 100_001, 250_000, owner="self"), asset("AAPL", 250_001, 500_000, owner="spouse")], [], period_end=PERIOD_END)

    assert (aapl.value_low, aapl.value_high) == (Decimal(350_002), Decimal(750_000))


def test_a_full_sale_only_empties_the_owner_who_sold() -> None:
    [aapl] = holdings.reconstruct(
        [asset("AAPL", 100_001, 250_000, owner="self"), asset("AAPL", 250_001, 500_000, owner="spouse")],
        [trade("AAPL", "sell", 100_001, 250_000, owner="spouse")],
        period_end=PERIOD_END,
    )

    assert (aapl.value_low, aapl.value_high) == (Decimal(100_001), Decimal(250_000))


def test_trades_apply_in_date_order_whatever_the_input_order() -> None:
    sale = trade("AAPL", "sell_partial", 15_001, 50_000, on=date(2025, 3, 1), doc_id="P2")
    buy = trade("AAPL", "buy", 100_001, 250_000, on=date(2025, 2, 1), doc_id="P1")

    [a] = holdings.reconstruct([], [sale, buy], period_end=PERIOD_END)
    [b] = holdings.reconstruct([], [buy, sale], period_end=PERIOD_END)

    assert a == b
    assert (a.value_low, a.value_high) == (Decimal(50_001), Decimal(234_999))  # buy first, then the sale; sale-first would leave the full buy


def test_a_buy_and_a_full_sale_on_the_same_day_net_to_nothing_held() -> None:
    day = date(2025, 4, 1)

    result = holdings.reconstruct([], [trade("AAPL", "sell", 15_001, 50_000, on=day, doc_id="P2"), trade("AAPL", "buy", 15_001, 50_000, on=day, doc_id="P1")], period_end=PERIOD_END)

    assert result == []


def test_holdings_are_sorted_by_ticker() -> None:
    result = holdings.reconstruct([asset("MSFT", 15_001, 50_000), asset("AAPL", 15_001, 50_000)], [], period_end=PERIOD_END)

    assert [h.ticker for h in result] == ["AAPL", "MSFT"]


# --- baseline weights -----------------------------------------------------------------------------


def _holding(ticker: str, mid: int) -> holdings.Holding:
    return holdings.Holding(ticker=ticker, asset_name=ticker, value_low=Decimal(mid), value_high=Decimal(mid), tier=tier_of(Decimal(mid)), sources=("A1",))


def test_baseline_weights_are_proportional_to_midpoints_and_sum_to_max_total() -> None:
    weights = holdings.baseline_weights([_holding("AAA", 3_000_000), _holding("BBB", 1_000_000)], max_total=Decimal("0.9"), max_position=Decimal("0.9"))

    assert weights == {"AAA": Decimal("0.6750"), "BBB": Decimal("0.2250")}


def test_baseline_weight_cap_is_not_redistributed() -> None:
    weights = holdings.baseline_weights([_holding("AAA", 9_000_000), _holding("BBB", 1_000_000)], max_total=Decimal("0.9"), max_position=Decimal("0.15"))

    assert weights == {"AAA": Decimal("0.1500"), "BBB": Decimal("0.0900")}
    assert sum(weights.values()) < Decimal("0.9")


def test_baseline_weights_never_exceed_max_total_after_rounding() -> None:
    many = [_holding(f"T{i}", 1_000_000 + i) for i in range(7)]

    weights = holdings.baseline_weights(many, max_total=Decimal("0.95"), max_position=Decimal("0.95"))

    assert sum(weights.values()) <= Decimal("0.95")


def test_baseline_weights_of_nothing_is_empty() -> None:
    assert holdings.baseline_weights([], max_total=Decimal("0.9"), max_position=Decimal("0.15")) == {}

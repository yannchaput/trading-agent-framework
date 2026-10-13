from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from trading_agent_framework.dashboard.pnl import AverageCostBook, BrokerPosition, Fee, Fill, daily_account_pnl

T0 = datetime(2026, 10, 5, 14, tzinfo=UTC)


def _fill(symbol: str, side: str, qty: float, price: float, fee: float = 0.0) -> Fill:
    return Fill(time=T0, symbol=symbol, side=side, qty=qty, price=price, fee=fee)


# --- AverageCostBook ------------------------------------------------------------------------------


def test_a_buy_is_held_at_its_cost_with_the_fee() -> None:
    book = AverageCostBook()

    assert book.apply(_fill("AAA", "buy", 10, 100, fee=1)) is None
    assert book.holdings() == {"AAA": (10, 1001)}


def test_a_sale_leaves_at_the_average_cost_of_several_buys_net_of_its_fee() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 10, 100))
    book.apply(_fill("AAA", "buy", 10, 120))

    sale = book.apply(_fill("AAA", "sell", 5, 130, fee=2))

    assert sale is not None
    assert (sale.symbol, sale.qty, sale.gross, sale.fees, sale.basis) == ("AAA", 5, 650, 2, 550)
    assert sale.pnl == pytest.approx(98)
    assert book.holdings() == {"AAA": (15, pytest.approx(1650))}


def test_selling_everything_empties_the_holding() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 10, 100))
    book.apply(_fill("AAA", "sell", 10, 90))

    assert book.holdings() == {}
    assert book.unmatched == {}


def test_a_sale_with_no_held_shares_is_unmatched_and_books_nothing() -> None:
    book = AverageCostBook()

    assert book.apply(_fill("BBB", "sell", 5, 50)) is None
    assert book.unmatched == {"BBB": 5}
    assert book.holdings() == {}


def test_selling_more_than_held_closes_the_held_shares_and_counts_the_rest_unmatched() -> None:
    book = AverageCostBook()
    book.apply(_fill("AAA", "buy", 4, 100))

    sale = book.apply(_fill("AAA", "sell", 6, 110, fee=3))

    assert sale is not None
    assert (sale.qty, sale.fees, sale.basis) == (4, pytest.approx(2), 400)
    assert book.unmatched == {"AAA": pytest.approx(2)}
    assert book.holdings() == {}


# --- daily_account_pnl --------------------------------------------------------------------------

MON, TUE, WED, THU = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8)


def _at(day: date, hour_utc: int = 15) -> datetime:
    return datetime(day.year, day.month, day.day, hour_utc, tzinfo=UTC)


def _position(symbol: str, qty: float, unrealized_pl: float) -> BrokerPosition:
    return BrokerPosition(symbol=symbol, qty=qty, avg_entry_price=100.0, current_price=None, market_value=None, unrealized_pl=unrealized_pl, unrealized_plpc=None)


def test_realized_is_booked_on_the_sale_day_net_of_fills_fees_and_fee_activities() -> None:
    fills = [
        Fill(_at(MON), "AAA", "buy", 10, 100, fee=1),
        Fill(_at(TUE), "AAA", "sell", 10, 110, fee=2),
    ]
    fees = [Fee(TUE, -0.05)]
    closes = {"AAA": {MON: 105.0, TUE: 111.0}}

    result = daily_account_pnl(fills, fees, closes, [], WED)

    assert result.days == [
        {"date": "2026-10-05", "realized": 0.0, "unrealized": 49.0},  # 10 * 105 - 1,001
        {"date": "2026-10-06", "realized": 96.95, "unrealized": 0.0},  # 1,100 - 2 - 1,001 - 0.05
        {"date": "2026-10-07", "realized": 0.0, "unrealized": 0.0},
    ]
    assert result.realized_total == pytest.approx(96.95)


def test_today_unrealized_comes_from_the_positions_not_a_close() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100)]
    closes = {"AAA": {MON: 101.0, TUE: 999.0}}  # a partial bar for today must be ignored

    result = daily_account_pnl(fills, [], closes, [_position("AAA", 10, 42.0)], TUE)

    assert result.days[-1] == {"date": "2026-10-06", "realized": 0.0, "unrealized": 42.0}
    assert result.unrealized_now == 42.0


def test_a_fill_in_the_evening_utc_belongs_to_the_new_york_day() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(datetime(2026, 10, 6, 1, 30, tzinfo=UTC), "AAA", "sell", 10, 110)]

    result = daily_account_pnl(fills, [], {"AAA": {MON: 100.0}}, [], WED)

    assert result.days[0] == {"date": "2026-10-05", "realized": 100.0, "unrealized": 0.0}


def test_a_missing_close_is_carried_forward_and_a_symbol_never_priced_is_named() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(_at(MON), "ZZZ", "buy", 1, 50)]
    closes = {"AAA": {MON: 102.0}, "BBB": {TUE: 7.0}}  # AAA has no Tuesday bar; ZZZ has none at all

    result = daily_account_pnl(fills, [], closes, [_position("AAA", 10, 0.0), _position("ZZZ", 1, 0.0)], WED)

    assert result.days[1] == {"date": "2026-10-06", "realized": 0.0, "unrealized": 20.0}  # 10 * 102 - 1,000
    assert result.unpriced == ["ZZZ"]


def test_replayed_holdings_differing_from_the_broker_are_reported() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(_at(MON), "BBB", "sell", 5, 50)]

    result = daily_account_pnl(fills, [], {}, [_position("AAA", 15, 0.0), _position("CCC", 3, 0.0)], TUE)

    assert result.mismatches == [("AAA", 10, 15), ("CCC", 0.0, 3)]
    assert result.unmatched_sales == {"BBB": 5}


def test_positions_without_fills_give_no_days_but_today_unrealized() -> None:
    result = daily_account_pnl([], [], {}, [_position("AAA", 10, -12.5)], TUE)

    assert result.days == []
    assert result.realized_total == 0.0
    assert result.unrealized_now == -12.5


def _priced_position(symbol: str, qty: float, current_price: float, unrealized_pl: float) -> BrokerPosition:
    return BrokerPosition(symbol=symbol, qty=qty, avg_entry_price=100.0, current_price=current_price, market_value=None, unrealized_pl=unrealized_pl, unrealized_plpc=None)


def test_today_unrealized_keeps_the_cost_basis_the_realized_pnl_uses_not_the_brokers_fifo_one() -> None:
    # Alpaca re-bases overnight positions to compressed FIFO: after selling 10 of 20 shares bought at 100 and 200,
    # it keeps the lot bought at 200 (loss of 500 at 150); the average cost kept by the replay is 150 (loss of 0).
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100), Fill(_at(TUE), "AAA", "buy", 10, 200), Fill(_at(WED), "AAA", "sell", 10, 150)]
    closes = {"AAA": {MON: 100.0, TUE: 200.0, WED: 150.0}}

    result = daily_account_pnl(fills, [], closes, [_priced_position("AAA", 10, 150.0, -500.0)], THU)

    assert result.days[-1] == {"date": "2026-10-08", "realized": 0.0, "unrealized": 0.0}
    assert result.unrealized_now == 0.0
    assert result.broker_unrealized == -500.0


def test_today_unrealized_falls_back_to_the_brokers_when_the_fills_do_not_explain_the_position() -> None:
    fills = [Fill(_at(MON), "AAA", "buy", 10, 100)]  # the broker holds 15

    result = daily_account_pnl(fills, [], {"AAA": {MON: 100.0}}, [_priced_position("AAA", 15, 110.0, 99.0)], TUE)

    assert result.unrealized_now == 99.0
    assert result.broker_unrealized == 99.0

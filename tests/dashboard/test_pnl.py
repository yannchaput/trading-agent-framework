from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trading_agent_framework.dashboard.pnl import AverageCostBook, Fill

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

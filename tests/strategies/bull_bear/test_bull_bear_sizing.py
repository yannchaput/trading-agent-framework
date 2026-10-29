from __future__ import annotations

from trading_agent_framework.strategies.bull_bear.sizing import equal_weights


def test_every_stock_gets_the_same_weight_in_input_order() -> None:
    w = equal_weights(["B", "A", "C"], weight=0.098)

    assert w == {"B": 0.098, "A": 0.098, "C": 0.098}
    assert list(w) == ["B", "A", "C"]


def test_a_short_book_is_not_scaled_up() -> None:
    assert sum(equal_weights(["A", "B"], weight=0.098).values()) == 0.196  # the rest is parked, not spread over two stocks


def test_no_stock_is_an_empty_book() -> None:
    assert equal_weights([], weight=0.098) == {}

from __future__ import annotations

from decimal import Decimal as D

import pytest

from trading_agent_framework.backtesting import fills
from trading_agent_framework.entities.enums import OrderSide


def _bar(o: str, h: str, low: str, c: str) -> fills.Bar:
    return fills.Bar(open=D(o), high=D(h), low=D(low), close=D(c))


def test_sell_trail_fills_at_the_level_when_touched() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("101", "101.5", "98.9", "99.5"), reference=D("101"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("99"))
    assert reference == D("101")


def test_sell_trail_gapping_through_fills_at_the_open() -> None:
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("97", "97.5", "96", "96.5"), reference=D("101"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("97"))


def test_sell_trail_ratchets_the_reference_on_a_bar_that_does_not_trigger() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("101", "104", "100", "103.5"), reference=D("101"), trail_price=D("2"))
    assert result is None
    assert reference == D("104")


def test_sell_trail_tests_the_previous_level_before_this_bars_high_can_raise_it() -> None:
    # Level 98 from reference 100. The bar's 105 high would lift the level to 103 if applied first;
    # pessimistically it is not, so a 97.9 low fills at 98.
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("100", "105", "97.9", "104"), reference=D("100"), trail_price=D("2"))
    assert result == fills.FillResult(price=D("98"))


def test_sell_trail_percent() -> None:
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("100", "100", "94.9", "95"), reference=D("100"), trail_percent=D("5"))
    assert result == fills.FillResult(price=D("95"))


def test_buy_trail_mirrors_with_a_low_water_mark() -> None:
    result, reference = fills.evaluate_trailing_stop(side=OrderSide.BUY, bar=_bar("99", "99.5", "96", "96.5"), reference=D("99"), trail_price=D("1"))
    assert result is None
    assert reference == D("96")
    result, _ = fills.evaluate_trailing_stop(side=OrderSide.BUY, bar=_bar("100", "100.5", "97", "98"), reference=D("99"), trail_price=D("1"))
    assert result == fills.FillResult(price=D("100"))


def test_trailing_stop_needs_exactly_one_trail_field() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        fills.evaluate_trailing_stop(side=OrderSide.SELL, bar=_bar("1", "1", "1", "1"), reference=D("1"))

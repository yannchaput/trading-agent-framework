from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.backtesting.fills import Bar, apply_commission_and_slippage, evaluate_fill
from trading_agent_framework.entities.enums import OrderSide, OrderType

D = Decimal
BAR = Bar(open=D(100), high=D(105), low=D(95), close=D(102))


def test_market_order_always_fills_at_open() -> None:
    result = evaluate_fill(order_type=OrderType.MARKET, side=OrderSide.BUY, bar=BAR)
    assert result is not None
    assert result.price == D(100)


@pytest.mark.parametrize(
    "side,limit_price,expected",
    [
        (OrderSide.BUY, D(96), D(96)),  # touches low (95), doesn't gap through open (100) -> limit price
        (OrderSide.BUY, D(101), D(100)),  # open already satisfies (100 <= 101) -> gapped, better price
        (OrderSide.SELL, D(104), D(104)),  # touches high (105), doesn't gap through open -> limit price
        (OrderSide.SELL, D(99), D(100)),  # open already satisfies (100 >= 99) -> gapped, better price
    ],
)
def test_limit_fills_when_touched_pessimistically(
    side: OrderSide, limit_price: Decimal, expected: Decimal
) -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=side, bar=BAR, limit_price=limit_price)
    assert result is not None
    assert result.price == expected


def test_buy_limit_does_not_fill_when_low_never_touches_it() -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.BUY, bar=BAR, limit_price=D(90))
    assert result is None


def test_sell_limit_does_not_fill_when_high_never_touches_it() -> None:
    result = evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.SELL, bar=BAR, limit_price=D(110))
    assert result is None


@pytest.mark.parametrize(
    "side,stop_price,expected",
    [
        (OrderSide.BUY, D(103), D(103)),  # touches high (105), open (100) below stop -> stop price
        (OrderSide.BUY, D(99), D(100)),  # open already above stop -> gapped, worse price (pessimistic)
        (OrderSide.SELL, D(97), D(97)),  # touches low (95), open above stop -> stop price
        (OrderSide.SELL, D(101), D(100)),  # open already below stop -> gapped, worse price
    ],
)
def test_stop_triggers_pessimistically(side: OrderSide, stop_price: Decimal, expected: Decimal) -> None:
    result = evaluate_fill(order_type=OrderType.STOP, side=side, bar=BAR, stop_price=stop_price)
    assert result is not None
    assert result.price == expected


def test_buy_stop_does_not_trigger_when_high_never_reaches_it() -> None:
    result = evaluate_fill(order_type=OrderType.STOP, side=OrderSide.BUY, bar=BAR, stop_price=D(110))
    assert result is None


def test_stop_limit_buy_needs_both_the_stop_trigger_and_the_limit_touch() -> None:
    # Stop at 103 (triggers, high=105), limit at 96 (also touched, low=95): fills.
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.BUY, bar=BAR,
        stop_price=D(103), stop_limit_price=D(96),
    )
    assert result is not None
    assert result.price == D(96)

    # Stop triggers (105 >= 103) but the limit (99) is never touched by the low (95 <= 99 is TRUE,
    # so use a limit that is NOT touched: low=95 means anything >= 95 IS touched; pick a limit
    # below the low to prove the "not touched" branch).
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.BUY, bar=BAR,
        stop_price=D(103), stop_limit_price=D(90),
    )
    assert result is None  # low (95) never reaches down to 90


def test_stop_limit_sell_needs_both_the_stop_trigger_and_the_limit_touch() -> None:
    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.SELL, bar=BAR,
        stop_price=D(97), stop_limit_price=D(104),
    )
    assert result is not None
    assert result.price == D(104)

    result = evaluate_fill(
        order_type=OrderType.STOP_LIMIT, side=OrderSide.SELL, bar=BAR,
        stop_price=D(97), stop_limit_price=D(110),
    )
    assert result is None  # high (105) never reaches up to 110


def test_trailing_stop_is_not_supported() -> None:
    with pytest.raises(ValueError, match="TRAIL"):
        evaluate_fill(order_type=OrderType.TRAIL, side=OrderSide.BUY, bar=BAR)


def test_limit_order_without_a_limit_price_raises() -> None:
    with pytest.raises(ValueError, match="limit_price"):
        evaluate_fill(order_type=OrderType.LIMIT, side=OrderSide.BUY, bar=BAR)


@pytest.mark.parametrize(
    "side,commission,slippage,expected_price,expected_commission",
    [
        (OrderSide.BUY, D(0), D(0), D(100), D(0)),
        (OrderSide.BUY, D("0.001"), D(0), D(100), D("0.1")),  # 10bps of 100
        (OrderSide.BUY, D(0), D("0.01"), D(101), D(0)),  # buys pay more
        (OrderSide.SELL, D(0), D("0.01"), D(99), D(0)),  # sells receive less
    ],
)
def test_commission_and_slippage(
    side: OrderSide, commission: Decimal, slippage: Decimal,
    expected_price: Decimal, expected_commission: Decimal,
) -> None:
    price, commission_per_share = apply_commission_and_slippage(
        D(100), side, commission=commission, slippage=slippage
    )
    assert price == expected_price
    assert commission_per_share == expected_commission

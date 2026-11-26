from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D
from typing import Any

import pytest

from trading_agent_framework.strategies.fallout_defensive_swing.rule import Action, Phase, SwingState, decide, trading_days_since

TODAY = date(2026, 11, 30)
FRIDAY = date(2026, 11, 20)


def _decide(state: SwingState, **overrides: Any) -> Action:
    arguments: dict[str, Any] = {
        "value": D("768"),
        "sell_fees": D(0),
        "prices": {"XLP": D("100"), "XLU": D("50")},
        "session": TODAY,
        "take_profit": D("0.01"),
        "buyback_dip": D("0.01"),
        "cash_reset_days": None,
    }
    return decide(state, **{**arguments, **overrides})


def _holding(*, last_sale: date | None = None, selling: bool = False) -> SwingState:
    return SwingState(phase=Phase.HOLDING, basis=D("768"), last_sale_session=last_sale, selling=selling)


def _cash(last_sale: date | None = TODAY - timedelta(days=1)) -> SwingState:
    return SwingState(phase=Phase.CASH, sale_prices={"XLP": D("100"), "XLU": D("50")}, last_sale_session=last_sale)


def test_fresh_buys() -> None:
    assert _decide(SwingState()) is Action.BUY


def test_holding_below_target_holds() -> None:
    assert _decide(_holding(), value=D("775.90"), sell_fees=D("0.24")) is Action.HOLD  # 775.66 < 775.68


def test_holding_at_target_net_of_fees_sells() -> None:
    assert _decide(_holding(), value=D("776.00"), sell_fees=D("0.24")) is Action.SELL  # 775.76 >= 775.68


def test_no_sale_on_the_last_sale_session() -> None:
    assert _decide(_holding(last_sale=TODAY), value=D("800")) is Action.HOLD


def test_sale_allowed_the_next_session() -> None:
    assert _decide(_holding(last_sale=TODAY - timedelta(days=1)), value=D("800")) is Action.SELL


def test_unfinished_sale_sells_the_rest_whatever_the_price() -> None:
    assert _decide(_holding(selling=True, last_sale=TODAY - timedelta(days=3)), value=D("700")) is Action.SELL


def test_unfinished_sale_still_waits_for_settlement() -> None:
    assert _decide(_holding(selling=True, last_sale=TODAY), value=D("700")) is Action.HOLD


def test_cash_above_the_dip_holds() -> None:
    assert _decide(_cash(), prices={"XLP": D("99.5"), "XLU": D("49.5")}) is Action.HOLD  # mean ratio 0.9925


def test_cash_at_the_dip_buys() -> None:
    assert _decide(_cash(), prices={"XLP": D("99"), "XLU": D("49.5")}) is Action.BUY  # mean ratio 0.99


@pytest.mark.parametrize(
    ("sold", "session", "expected"),
    [
        (date(2027, 10, 8), date(2027, 10, 11), Action.HOLD),  # Columbus Day: the market trades, nothing settles
        (date(2027, 10, 8), date(2027, 10, 12), Action.SELL),
        (date(2026, 11, 10), date(2026, 11, 11), Action.HOLD),  # Veterans Day
        (date(2026, 11, 10), date(2026, 11, 12), Action.SELL),
        (date(2029, 11, 9), date(2029, 11, 12), Action.HOLD),  # Veterans Day on a Sunday: banks close on Monday
        (date(2026, 11, 25), date(2026, 11, 27), Action.SELL),  # an exchange holiday in between changes nothing
        (date(2026, 11, 20), date(2026, 11, 23), Action.SELL),  # Friday sale, Monday session
    ],
)
def test_no_sale_before_the_last_sale_has_settled(sold: date, session: date, expected: Action) -> None:
    assert _decide(_holding(last_sale=sold), value=D("800"), session=session) is expected


def test_trading_days_since_skips_the_weekend() -> None:
    assert trading_days_since(FRIDAY, date(2026, 11, 23)) == 1
    assert trading_days_since(FRIDAY, date(2026, 11, 27)) == 5


def test_cash_without_a_dip_holds_before_the_reset_day() -> None:
    assert _decide(_cash(FRIDAY), session=date(2026, 11, 26), cash_reset_days=5) is Action.HOLD


def test_cash_without_a_dip_resets_on_the_reset_day() -> None:
    assert _decide(_cash(FRIDAY), session=date(2026, 11, 27), cash_reset_days=5) is Action.RESET


def test_without_a_reset_cash_waits_for_the_dip() -> None:
    assert _decide(_cash(FRIDAY), session=date(2027, 11, 26), cash_reset_days=None) is Action.HOLD


def test_the_dip_still_buys_before_the_reset() -> None:
    dipped = {"XLP": D("99"), "XLU": D("49.5")}

    assert _decide(_cash(FRIDAY), session=date(2026, 11, 23), prices=dipped, cash_reset_days=5) is Action.BUY


def test_holding_never_resets() -> None:
    assert _decide(_holding(last_sale=FRIDAY), session=date(2027, 11, 26), cash_reset_days=5) is Action.HOLD


def test_cash_without_a_sale_date_only_waits_for_the_dip() -> None:
    assert _decide(_cash(None), session=date(2027, 11, 26), cash_reset_days=5) is Action.HOLD

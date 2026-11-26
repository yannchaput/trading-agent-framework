from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal as D
from typing import Any

from trading_agent_framework.strategies.fallout_defensive_swing.rule import Action, Phase, SwingState, decide

TODAY = date(2026, 11, 30)


def _decide(state: SwingState, **overrides: Any) -> Action:
    arguments: dict[str, Any] = {
        "value": D("768"),
        "sell_fees": D(0),
        "prices": {"XLP": D("100"), "XLU": D("50")},
        "session": TODAY,
        "take_profit": D("0.01"),
        "buyback_dip": D("0.01"),
    }
    return decide(state, **{**arguments, **overrides})


def _holding(*, last_sale: date | None = None, selling: bool = False) -> SwingState:
    return SwingState(phase=Phase.HOLDING, basis=D("768"), last_sale_session=last_sale, selling=selling)


def _cash() -> SwingState:
    return SwingState(phase=Phase.CASH, sale_prices={"XLP": D("100"), "XLU": D("50")}, last_sale_session=TODAY - timedelta(days=1))


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

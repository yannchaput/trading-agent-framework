"""Pure Alpaca account and calendar translation.

Same rules as `orders.py`: no I/O, no state, no client instances. Together
with `orders.py`, the only module allowed to import `alpaca.trading.requests`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import cast
from zoneinfo import ZoneInfo

from alpaca.trading.requests import GetCalendarRequest

from trading_agent_framework.brokers.alpaca.orders import _field, _to_decimal
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.clock import MarketSession
from trading_agent_framework.utils.errors import BrokerError


def parse_account(response: object) -> AccountBalances:
    """Map an Alpaca TradeAccount to AccountBalances; equity is the portfolio value."""
    cash = _to_decimal(_field(response, "cash"))
    equity = _to_decimal(_field(response, "equity"))
    if equity is None:
        equity = _to_decimal(_field(response, "portfolio_value"))
    buying_power = _to_decimal(_field(response, "buying_power"))
    if cash is None or equity is None or buying_power is None:
        raise BrokerError("Alpaca account response is missing cash, equity or buying_power")
    return AccountBalances(cash=cash, portfolio_value=equity, buying_power=buying_power)


def build_calendar_request(start: date, end: date) -> GetCalendarRequest:
    return GetCalendarRequest(start=start, end=end)


def parse_calendar(responses: Iterable[object], tz: ZoneInfo) -> list[MarketSession]:
    """Alpaca calendar days -> sessions sorted by open. The SDK returns naive market times."""
    sessions = [
        MarketSession(
            open=_localize(cast(datetime, _field(day, "open")), tz),
            close=_localize(cast(datetime, _field(day, "close")), tz),
        )
        for day in responses
    ]
    return sorted(sessions, key=lambda session: session.open)


def _localize(value: datetime, tz: ZoneInfo) -> datetime:
    return value.replace(tzinfo=tz) if value.tzinfo is None else value.astimezone(tz)

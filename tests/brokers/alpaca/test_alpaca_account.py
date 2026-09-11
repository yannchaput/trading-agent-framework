from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from tests.fakes import ET, make_alpaca_account, make_alpaca_calendar, make_session

from trading_agent_framework.brokers.alpaca import account
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.errors import BrokerError


def test_parse_account_maps_equity_to_portfolio_value() -> None:
    assert account.parse_account(make_alpaca_account()) == AccountBalances(
        cash=Decimal("10000.50"),
        portfolio_value=Decimal("25000.25"),
        buying_power=Decimal("20000"),
    )


def test_parse_account_falls_back_to_portfolio_value_without_equity() -> None:
    response = make_alpaca_account(equity=None, portfolio_value="123.45")
    assert account.parse_account(response).portfolio_value == Decimal("123.45")


def test_parse_account_rejects_missing_cash() -> None:
    with pytest.raises(BrokerError, match="cash"):
        account.parse_account(make_alpaca_account(cash=None))


def test_build_calendar_request_sets_the_date_range() -> None:
    request = account.build_calendar_request(date(2026, 9, 14), date(2026, 9, 28))
    assert request.start == date(2026, 9, 14)
    assert request.end == date(2026, 9, 28)


def test_parse_calendar_localizes_sorts_and_keeps_early_closes() -> None:
    responses = [
        make_alpaca_calendar("2024-12-02"),
        make_alpaca_calendar("2024-11-29", close_at="13:00"),  # day after Thanksgiving
    ]
    sessions = account.parse_calendar(responses, ET)
    assert sessions == [
        make_session(date(2024, 11, 29), close_at=time(13, 0)),
        make_session(date(2024, 12, 2)),
    ]
    assert sessions[0].open.tzinfo is ET


def test_parse_calendar_converts_aware_times() -> None:
    day = SimpleNamespace(
        open=datetime(2024, 11, 29, 14, 30, tzinfo=UTC),
        close=datetime(2024, 11, 29, 18, 0, tzinfo=UTC),
    )
    [session] = account.parse_calendar([day], ET)
    assert session == make_session(date(2024, 11, 29), close_at=time(13, 0))

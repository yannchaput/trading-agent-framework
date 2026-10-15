from __future__ import annotations

from decimal import Decimal

import pytest

from trading_agent_framework.brokers.tastytrade import account
from trading_agent_framework.brokers.tastytrade.values import decimal_of, optional_decimal
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

D = Decimal
BALANCES = {"account-number": "5WT00001", "cash-balance": "1000.50", "net-liquidating-value": "2500.25", "equity-buying-power": "1000.50", "margin-equity": "2500.25"}


def _item(number: str = "5WT00001", kind: str = "Cash", closed: bool = False) -> dict[str, object]:
    return {"account": {"account-number": number, "margin-or-cash": kind, "is-closed": closed, "nickname": "Individual"}, "authority-level": "owner"}


# --- values ----------------------------------------------------------------------------


def test_decimal_of_parses_strings_and_numbers_exactly() -> None:
    assert decimal_of("150.10", "price") == D("150.10")
    assert decimal_of(3, "quantity") == D(3)


@pytest.mark.parametrize(("raw", "message"), [(None, "missing"), ("abc", "not a number"), ("NaN", "not a finite number")])
def test_decimal_of_rejects_bad_values(raw: object, message: str) -> None:
    with pytest.raises(BrokerError, match=f"Tastytrade price is {message}"):
        decimal_of(raw, "price")


def test_optional_decimal_passes_none_through() -> None:
    assert optional_decimal(None, "price") is None
    assert optional_decimal("1.5", "price") == D("1.5")


# --- balances --------------------------------------------------------------------------


def test_balances_map_cash_net_liquidating_value_and_equity_buying_power() -> None:
    assert account.parse_balances(BALANCES) == AccountBalances(cash=D("1000.50"), portfolio_value=D("2500.25"), buying_power=D("1000.50"))


def test_negative_cash_is_kept() -> None:
    assert account.parse_balances({**BALANCES, "cash-balance": "-12.50"}).cash == D("-12.50")


@pytest.mark.parametrize("key", ["cash-balance", "net-liquidating-value", "equity-buying-power"])
def test_a_missing_balance_field_is_a_broker_error(key: str) -> None:
    with pytest.raises(BrokerError, match=key):
        account.parse_balances({k: v for k, v in BALANCES.items() if k != key})


# --- account check ---------------------------------------------------------------------


def test_a_cash_account_passes_without_warnings() -> None:
    assert account.check_account([_item(), _item("5WT00002")], "5WT00001", is_paper=False) == []


def test_an_account_the_login_does_not_own_is_refused_listing_the_owned_ones() -> None:
    with pytest.raises(ConfigurationError, match=r"TASTYTRADE_ACCOUNT_NUMBER=5WT99999 .*5WT00001, 5WT00002"):
        account.check_account([_item(), _item("5WT00002")], "5WT99999", is_paper=False)


def test_a_closed_account_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="closed"):
        account.check_account([_item(closed=True)], "5WT00001", is_paper=True)


def test_a_margin_account_is_refused_live() -> None:
    with pytest.raises(ConfigurationError, match="margin account"):
        account.check_account([_item(kind="Margin")], "5WT00001", is_paper=False)


def test_a_margin_account_only_warns_in_the_sandbox() -> None:
    [warning] = account.check_account([_item(kind="Margin")], "5WT00001", is_paper=True)

    assert "margin" in warning


def test_unwrapped_account_items_are_accepted_too() -> None:
    assert account.check_account([{"account-number": "5WT00001", "margin-or-cash": "Cash", "is-closed": False}], "5WT00001", is_paper=False) == []

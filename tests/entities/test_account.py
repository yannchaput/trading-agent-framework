from __future__ import annotations

import dataclasses
from decimal import Decimal

import pytest

from trading_agent_framework.entities.account import AccountBalances


def test_account_balances_holds_decimals() -> None:
    balances = AccountBalances(
        cash=Decimal("1000.50"), portfolio_value=Decimal("2500"), buying_power=Decimal("4000")
    )
    assert balances.cash == Decimal("1000.50")
    assert balances.portfolio_value == Decimal("2500")
    assert balances.buying_power == Decimal("4000")


def test_account_balances_is_frozen() -> None:
    balances = AccountBalances(
        cash=Decimal("1"), portfolio_value=Decimal("1"), buying_power=Decimal("1")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        balances.cash = Decimal("2")  # type: ignore

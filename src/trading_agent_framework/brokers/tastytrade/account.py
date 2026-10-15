"""Pure Tastytrade account translation: balances, and the start-up account check.

No I/O: the broker passes in what `GET /accounts/{a}/balances` and `GET /customers/me/accounts` returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trading_agent_framework.brokers.tastytrade.values import decimal_of
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import ConfigurationError


def parse_balances(data: Mapping[str, Any]) -> AccountBalances:
    return AccountBalances(
        cash=decimal_of(data.get("cash-balance"), "balance cash-balance"),
        portfolio_value=decimal_of(data.get("net-liquidating-value"), "balance net-liquidating-value"),
        buying_power=decimal_of(data.get("equity-buying-power"), "balance equity-buying-power"),
    )


def check_account(items: Sequence[Mapping[str, Any]], account_number: str, *, is_paper: bool) -> list[str]:
    """Refuse an account this login doesn't own, a closed one, and a live margin account; returns warnings.

    `items` are `GET /customers/me/accounts`'s `data.items`: `{"account": {...}, "authority-level": ...}`.
    """
    accounts = [item.get("account", item) for item in items]
    found = next((a for a in accounts if a.get("account-number") == account_number), None)
    if found is None:
        owned = ", ".join(sorted(str(a.get("account-number")) for a in accounts)) or "none"
        raise ConfigurationError(f"TASTYTRADE_ACCOUNT_NUMBER={account_number} is not an account of this Tastytrade login (its accounts: {owned})")
    if found.get("is-closed"):
        raise ConfigurationError(f"Tastytrade account {account_number} is closed")
    warnings: list[str] = []
    if found.get("margin-or-cash") == "Margin":
        if not is_paper:
            raise ConfigurationError(f"Tastytrade account {account_number} is a margin account; this framework expects a cash account (no margin, no shorting)")
        warnings.append(f"Tastytrade sandbox account {account_number} is a margin account (accepted in the sandbox only)")
    return warnings

"""Pure IBKR account translation and start-up checks (same rules as `brokers/alpaca/account.py`).

IBKR cannot switch margin or shorting off through the API, so `IbkrBroker.configure_account`
only CHECKS the account with `check_account` and refuses to trade live on a margin account.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, InvalidOperation

from ib_async import AccountValue

from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError

PAPER_ACCOUNT_PREFIX = "DU"
BASE_CURRENCY = "USD"
_MARGIN_TOLERANCE = Decimal("1.05")  # a cash account's buying power is its settled cash


def is_paper_account(account_id: str) -> bool:
    return account_id.startswith(PAPER_ACCOUNT_PREFIX)


def _tags(values: Iterable[AccountValue], account_id: str) -> dict[str, AccountValue]:
    return {v.tag: v for v in values if v.account == account_id}


def _amount(tags: dict[str, AccountValue], tag: str) -> Decimal:
    value = tags.get(tag)
    if value is None:
        raise BrokerError(f"IBKR account summary has no {tag}")
    try:
        return Decimal(value.value)
    except InvalidOperation:
        raise BrokerError(f"IBKR account summary {tag} is not a number: {value.value!r}") from None


def parse_account(values: Iterable[AccountValue], account_id: str) -> AccountBalances:
    tags = _tags(values, account_id)
    return AccountBalances(
        cash=_amount(tags, "TotalCashValue"),
        portfolio_value=_amount(tags, "NetLiquidation"),
        buying_power=_amount(tags, "BuyingPower"),
    )


def check_account(values: Iterable[AccountValue], account_id: str, *, is_paper: bool) -> list[str]:
    """Raise `ConfigurationError` when the account must not trade; return warnings to log."""
    if is_paper_account(account_id) != is_paper:
        kind = "a paper" if is_paper_account(account_id) else "a live"
        raise ConfigurationError(
            f"IB Gateway is logged into {kind} account ({account_id}) but BROKER_API_IS_PAPER={str(is_paper).lower()}"
        )
    tags = _tags(values, account_id)
    currency = tags["NetLiquidation"].currency if "NetLiquidation" in tags else ""
    if currency != BASE_CURRENCY:
        raise ConfigurationError(f"IBKR account {account_id} has base currency {currency!r}; only USD is supported")
    cash = _amount(tags, "TotalCashValue")
    buying_power = _amount(tags, "BuyingPower")
    if buying_power > cash * _MARGIN_TOLERANCE + 1:
        message = (
            f"IBKR account {account_id} looks like a margin account (buying power {buying_power} > cash {cash}); "
            "this framework expects a cash account (no margin, no shorting)"
        )
        if not is_paper:
            raise ConfigurationError(message)
        return [message]
    return []

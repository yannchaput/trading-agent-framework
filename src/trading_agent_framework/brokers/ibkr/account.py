"""Pure IBKR account translation and start-up checks (same rules as `brokers/alpaca/account.py`).

IBKR cannot switch margin or shorting off through the API, so `IbkrBroker.configure_account`
only CHECKS the account with `check_account` and refuses to trade live on a margin account.

The margin check itself is a heuristic (`BuyingPower` vs `TotalCashValue`): the TWS API has no
field that says whether an account is cash or margin. It only runs live, where it matters and
where it holds -- a live cash account extends no leverage, so `BuyingPower` tracks cash. IBKR's
paper-trading simulator was observed reporting `BuyingPower` as ~6.67x cash on an account
verified (via the web portal, and via zero `InitMarginReq`/`MaintMarginReq`/`Cushion`=1 in this
same summary) to be a genuine cash account -- paper's simulated buying power doesn't reflect the
linked account's real cash/margin restriction, so the heuristic is guaranteed noise there.
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


def check_account(values: Iterable[AccountValue], account_id: str, *, is_paper: bool) -> None:
    """Raise `ConfigurationError` when the account must not trade.

    The margin heuristic below only runs live: `BuyingPower` is not a meaningful signal on
    IBKR's paper-trading simulator (see the module docstring), so checking it there only ever
    produces false positives.
    """
    if is_paper_account(account_id) != is_paper:
        kind = "a paper" if is_paper_account(account_id) else "a live"
        raise ConfigurationError(
            f"IB Gateway is logged into {kind} account ({account_id}) but BROKER_API_IS_PAPER={str(is_paper).lower()}"
        )
    tags = _tags(values, account_id)
    currency = tags["NetLiquidation"].currency if "NetLiquidation" in tags else ""
    if currency != BASE_CURRENCY:
        raise ConfigurationError(f"IBKR account {account_id} has base currency {currency!r}; only USD is supported")
    if is_paper:
        return
    cash = _amount(tags, "TotalCashValue")
    buying_power = _amount(tags, "BuyingPower")
    if buying_power > cash * _MARGIN_TOLERANCE + 1:
        raise ConfigurationError(
            f"IBKR account {account_id} looks like a margin account (buying power {buying_power} > cash {cash}); "
            "this framework expects a cash account (no margin, no shorting)"
        )

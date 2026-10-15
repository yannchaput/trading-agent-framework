"""Number parsing shared by the pure Tastytrade modules: the API sends money and quantities as strings.

Pure: no I/O, no state.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from trading_agent_framework.utils.errors import BrokerError


def decimal_of(raw: object, what: str) -> Decimal:
    """`raw` as an exact `Decimal`; `BrokerError` naming `what` when it is missing or not a finite number."""
    if raw is None or isinstance(raw, bool):
        raise BrokerError(f"Tastytrade {what} is missing")
    try:
        value = Decimal(str(raw))
    except InvalidOperation:
        raise BrokerError(f"Tastytrade {what} is not a number: {raw!r}") from None
    if not value.is_finite():
        raise BrokerError(f"Tastytrade {what} is not a finite number: {raw!r}")
    return value


def optional_decimal(raw: object, what: str) -> Decimal | None:
    return None if raw is None else decimal_of(raw, what)

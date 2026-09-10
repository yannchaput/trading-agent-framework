from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class AccountBalances:
    """Broker account snapshot. `portfolio_value` is equity: cash + positions."""

    cash: Decimal
    portfolio_value: Decimal
    buying_power: Decimal

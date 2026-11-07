"""Latest bid/ask quote entity."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_agent_framework.entities.asset import Asset


@dataclass(frozen=True, slots=True)
class Quote:
    """Latest bid/ask for an asset. A side is None when the book is empty on that side."""

    asset: Asset
    bid: Decimal | None
    ask: Decimal | None
    bid_size: Decimal | None
    ask_size: Decimal | None
    timestamp: datetime

    @property
    def mid(self) -> Decimal | None:
        """Return the mid price between bid and ask.

        Returns:
            ``(bid + ask) / 2``, or None when either side of the book is empty.
        """
        if self.bid is None or self.ask is None:
            return None
        return (self.bid + self.ask) / 2

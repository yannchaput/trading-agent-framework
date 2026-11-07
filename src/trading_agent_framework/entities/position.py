"""Open position entity."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide


@dataclass(slots=True)
class Position:
    """An open position held at the broker.

    Attributes:
        strategy_name: Name of the strategy that owns the position.
        asset: The instrument held.
        quantity: Number of shares (positive for both sides; see ``side``).
        side: Long or short.
        avg_fill_price: Average cost per share, when the broker reports one.
        current_price: Latest price per share, when known.
        market_value: Current market value of the position, when known.
        unrealized_pnl: Unrealized profit or loss, when known.
        raw: The broker's native position object, kept for debugging.
    """

    strategy_name: str
    asset: Asset
    quantity: Decimal
    side: PositionSide
    avg_fill_price: Decimal | None = None
    current_price: Decimal | None = None
    market_value: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    raw: Any = None

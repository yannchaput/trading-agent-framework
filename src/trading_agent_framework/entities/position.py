from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide


@dataclass(slots=True)
class Position:
    strategy_name: str
    asset: Asset
    quantity: Decimal
    side: PositionSide
    avg_fill_price: Decimal | None = None
    current_price: Decimal | None = None
    market_value: Decimal | None = None
    unrealized_pnl: Decimal | None = None
    raw: Any = None

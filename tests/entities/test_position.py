from __future__ import annotations

from decimal import Decimal

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import PositionSide
from trading_agent_framework.entities.position import Position


def test_position_holds_a_decimal_quantity() -> None:
    position = Position(
        strategy_name="momentum",
        asset=Asset(symbol="AAPL"),
        quantity=Decimal("10"),
        side=PositionSide.LONG,
    )
    assert position.quantity == Decimal("10")
    assert isinstance(position.quantity, Decimal)

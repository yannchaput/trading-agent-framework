from __future__ import annotations

from trading_agent_framework import entities
from trading_agent_framework.entities import asset as asset_module
from trading_agent_framework.entities import enums as enums_module
from trading_agent_framework.entities import order as order_module
from trading_agent_framework.entities import position as position_module


def test_entities_reexports_match_source_modules() -> None:
    assert entities.Asset is asset_module.Asset
    assert entities.AssetType is enums_module.AssetType
    assert entities.OrderSide is enums_module.OrderSide
    assert entities.OrderType is enums_module.OrderType
    assert entities.TimeInForce is enums_module.TimeInForce
    assert entities.PositionSide is enums_module.PositionSide
    assert entities.OrderStatus is enums_module.OrderStatus
    assert entities.OrderEvent is enums_module.OrderEvent
    assert entities.ACTIVE_ORDER_STATUSES is enums_module.ACTIVE_ORDER_STATUSES
    assert entities.is_equivalent_status is enums_module.is_equivalent_status
    assert entities.Order is order_module.Order
    assert entities.Transaction is order_module.Transaction
    assert entities.Position is position_module.Position

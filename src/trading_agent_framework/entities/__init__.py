from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    ACTIVE_ORDER_STATUSES,
    AssetType,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
    is_equivalent_status,
)
from trading_agent_framework.entities.order import Order, Transaction
from trading_agent_framework.entities.position import Position

try:
    __version__ = version("trading_agent_framework")
except PackageNotFoundError:
    # Package is not installed (e.g., running from local source)
    __version__ = "unknown"

__all__ = [
    "ACTIVE_ORDER_STATUSES",
    "Asset",
    "AssetType",
    "Order",
    "OrderEvent",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "PositionSide",
    "TimeInForce",
    "Transaction",
    "is_equivalent_status",
]

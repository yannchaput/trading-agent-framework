"""Enumerations shared by every broker: asset, order and position vocabulary."""

from __future__ import annotations

from enum import StrEnum


class AssetType(StrEnum):
    """Kind of tradable instrument."""

    STOCK = "stock"


class OrderSide(StrEnum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order execution style (market, limit, stop, stop-limit, trailing stop)."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAIL = "trailing_stop"


class TimeInForce(StrEnum):
    """How long an order stays working before the broker cancels it."""

    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


class PositionSide(StrEnum):
    """Direction of an open position."""

    LONG = "long"
    SHORT = "short"


class OrderStatus(StrEnum):
    """Lifecycle state of an order, normalised across brokers."""

    UNPROCESSED = "unprocessed"
    SUBMITTED = "submitted"
    OPEN = "open"
    NEW = "new"
    CANCELLING = "cancelling"
    CANCELED = "canceled"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    ERROR = "error"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class OrderEvent(StrEnum):
    """Order lifecycle event delivered to strategy hooks."""

    NEW = "new"
    CANCELED = "canceled"
    FILLED = "fill"
    PARTIALLY_FILLED = "partial_fill"
    MODIFIED = "modified"
    ERROR = "error"


ACTIVE_ORDER_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.UNPROCESSED,
        OrderStatus.SUBMITTED,
        OrderStatus.OPEN,
        OrderStatus.NEW,
        OrderStatus.CANCELLING,
        OrderStatus.PARTIAL_FILL,
    }
)

_OPEN_EQUIVALENCE_CLASS = frozenset({OrderStatus.OPEN, OrderStatus.NEW, OrderStatus.SUBMITTED})


def is_equivalent_status(a: OrderStatus, b: OrderStatus) -> bool:
    """Treat OPEN, NEW, and SUBMITTED as one equivalence class; otherwise compare directly."""
    if a in _OPEN_EQUIVALENCE_CLASS and b in _OPEN_EQUIVALENCE_CLASS:
        return True
    return a == b

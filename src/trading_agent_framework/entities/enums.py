from __future__ import annotations

from enum import StrEnum


class AssetType(StrEnum):
    STOCK = "stock"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAIL = "trailing_stop"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"
    IOC = "ioc"
    FOK = "fok"


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderStatus(StrEnum):
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

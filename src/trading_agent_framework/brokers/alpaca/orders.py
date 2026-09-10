"""Pure Alpaca order translation logic: status/event mapping, price conforming,
and pre-submission validation.

This module holds no I/O, no state, and no client instances -- it is a
collection of module-level functions and constants operating on our own
`Order` entity and plain Python values. Per the project's global constraints,
`orders.py` is the only module in the codebase permitted to import
`alpaca.trading.requests` (needed once `build_order_request` lands here);
Tasks 7-9 need nothing from the `alpaca` package at all, since the status/event
maps are keyed by plain lowercase strings and accept either a raw string or an
enum member duck-typed via `.value`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType

from trading_agent_framework.entities.enums import OrderEvent, OrderStatus, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderValidationError

logger = logging.getLogger(__name__)

# --- Task 7: status / event maps --------------------------------------------

ALPACA_STATUS_MAP: MappingProxyType[str, OrderStatus] = MappingProxyType(
    {
        "filled": OrderStatus.FILL,
        "partially_filled": OrderStatus.PARTIAL_FILL,
        "canceled": OrderStatus.CANCELED,
        "cancelled": OrderStatus.CANCELED,
        "cancel": OrderStatus.CANCELED,
        "done_for_day": OrderStatus.CANCELED,
        "replaced": OrderStatus.CANCELED,
        "stopped": OrderStatus.CANCELED,
        "suspended": OrderStatus.CANCELED,
        "pending_cancel": OrderStatus.CANCELED,
        "pending_replace": OrderStatus.CANCELED,
        "new": OrderStatus.NEW,
        "pending_new": OrderStatus.NEW,
        "presubmitted": OrderStatus.NEW,
        "accepted": OrderStatus.OPEN,
        "accepted_for_bidding": OrderStatus.OPEN,
        "calculated": OrderStatus.OPEN,
        "held": OrderStatus.OPEN,
        "pending_review": OrderStatus.OPEN,
        "rejected": OrderStatus.ERROR,
        "expired": OrderStatus.EXPIRED,
    }
)

ALPACA_EVENT_MAP: MappingProxyType[str, OrderEvent | None] = MappingProxyType(
    {
        "new": OrderEvent.NEW,
        "accepted": OrderEvent.NEW,
        "fill": OrderEvent.FILLED,
        "partial_fill": OrderEvent.PARTIALLY_FILLED,
        "canceled": OrderEvent.CANCELED,
        "expired": OrderEvent.CANCELED,
        "rejected": OrderEvent.ERROR,
        "replaced": OrderEvent.MODIFIED,
        "pending_new": None,
        "pending_cancel": None,
        "pending_replace": None,
        "restated": None,
    }
)


def _normalize_key(raw: object) -> str:
    """Reduce a raw status/event -- a plain string or a `(str, Enum)` member -- to
    the lowercase string used as a map key."""
    value = getattr(raw, "value", raw)
    return str(value).lower()


def map_status(raw: object) -> OrderStatus:
    """Map a raw Alpaca order status (str or enum member) to our internal OrderStatus.

    Never raises: an unrecognised status is logged and mapped to UNKNOWN so a
    single unexpected payload can't kill the streaming thread.
    """
    key = _normalize_key(raw)
    status = ALPACA_STATUS_MAP.get(key)
    if status is None:
        logger.warning("Unrecognised Alpaca order status: %r", raw)
        return OrderStatus.UNKNOWN
    return status


def map_event(raw: object) -> OrderEvent | None:
    """Map a raw Alpaca trade event (str or enum member) to our internal OrderEvent.

    Returns None both for events we deliberately ignore (e.g. pending_new,
    already a key with a None value) and for events we don't recognise at all
    (which additionally logs a warning). Never raises.
    """
    key = _normalize_key(raw)
    if key not in ALPACA_EVENT_MAP:
        logger.warning("Unrecognised Alpaca trade event: %r", raw)
        return None
    return ALPACA_EVENT_MAP[key]


# --- Task 8: price conforming ------------------------------------------------

_PENNY = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")


def round_price(price: Decimal) -> Decimal:
    """Round a limit price to Alpaca's tick size: 2dp at/above $1, 4dp below $1.

    The bucket is chosen from the pre-rounded value, never from an
    already-quantized intermediate -- rounding first and bucketing second
    misclassifies boundary values such as Decimal("0.99999").
    """
    quantum = _PENNY if price >= 1 else _SUB_PENNY
    return price.quantize(quantum, rounding=ROUND_HALF_UP)


def round_stop_price(price: Decimal) -> Decimal:
    """Round a stop price to Alpaca's tick size: always 2 decimal places."""
    return price.quantize(_PENNY, rounding=ROUND_HALF_UP)


def _conform_field(order: Order, field_name: str, rounder: Callable[[Decimal], Decimal]) -> None:
    current: Decimal | None = getattr(order, field_name)
    if current is None:
        return
    rounded = rounder(current)
    if rounded != current:
        logger.warning("conform_order: %s rounded from %s to %s", field_name, current, rounded)
        setattr(order, field_name, rounded)


def conform_order(order: Order) -> Order:
    """Round an order's prices to Alpaca's tick size, in place.

    Mutates and returns the same `order` instance: the caller and the tracker
    must observe the same object, or stream updates never reach what the
    caller holds. Logs one warning per field actually changed; logs nothing
    when nothing changes.
    """
    _conform_field(order, "limit_price", round_price)
    _conform_field(order, "stop_price", round_stop_price)
    _conform_field(order, "stop_limit_price", round_stop_price)
    return order


# --- Task 9: pre-submission validation --------------------------------------

_OPENING_CLOSING_TIF = frozenset({TimeInForce.OPG, TimeInForce.CLS})
_OPENING_CLOSING_ORDER_TYPES = frozenset({OrderType.MARKET, OrderType.LIMIT})


def validate_order(order: Order) -> None:
    """Enforce Alpaca server-side order rules that alpaca-py's pydantic request
    models do not check.

    Covers exactly four rules -- deliberately not an attempt to mirror every
    Alpaca constraint, which is how lumibot grew its dead code. Raises
    OrderValidationError; never a raw pydantic ValidationError.
    """
    if order.notional is not None and order.order_type is not OrderType.MARKET:
        raise OrderValidationError(
            f"notional is only supported for MARKET orders (order_type={order.order_type})"
        )

    if order.quantity is not None and order.quantity % 1 != 0:
        if order.order_type is not OrderType.MARKET:
            raise OrderValidationError(
                "fractional quantity is only supported for MARKET orders "
                f"(order_type={order.order_type})"
            )
        if order.time_in_force is not TimeInForce.DAY:
            raise OrderValidationError(
                "fractional quantity requires time_in_force=DAY "
                f"(time_in_force={order.time_in_force})"
            )

    if (
        order.time_in_force in _OPENING_CLOSING_TIF
        and order.order_type not in _OPENING_CLOSING_ORDER_TYPES
    ):
        raise OrderValidationError(
            f"time_in_force={order.time_in_force} requires order_type MARKET or LIMIT "
            f"(order_type={order.order_type})"
        )

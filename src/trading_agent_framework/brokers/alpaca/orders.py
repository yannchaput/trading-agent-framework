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
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from uuid import uuid4

from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    OrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from pydantic import ValidationError

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    AssetType,
    OrderEvent,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeInForce,
)
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
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


# --- Task 10: build_order_request --------------------------------------------

_REQUEST_BY_TYPE: MappingProxyType[OrderType, type[OrderRequest]] = MappingProxyType(
    {
        OrderType.MARKET: MarketOrderRequest,
        OrderType.LIMIT: LimitOrderRequest,
        OrderType.STOP: StopOrderRequest,
        OrderType.STOP_LIMIT: StopLimitOrderRequest,
        OrderType.TRAIL: TrailingStopOrderRequest,
    }
)


def _to_api_number(value: Decimal | None) -> float | None:
    """Convert a Decimal to the float alpaca-py's pydantic requests expect.

    The only place in the codebase where Decimal becomes float -- see
    global constraint 5.
    """
    return None if value is None else float(value)


def build_order_request(order: Order) -> OrderRequest:
    """Build the alpaca-py request object matching `order.order_type`.

    Pure translation: no I/O, no validation beyond what alpaca-py's own
    pydantic models enforce. Any ValidationError or ValueError raised while
    constructing the request is wrapped in OrderValidationError so callers
    never see a raw pydantic exception (global constraint 8).
    """
    request_cls = _REQUEST_BY_TYPE[order.order_type]

    kwargs: dict[str, object] = {
        "symbol": order.asset.symbol,
        "side": AlpacaOrderSide(order.side.value),
        "time_in_force": AlpacaTimeInForce(order.time_in_force.value),
        "extended_hours": order.extended_hours,
        "client_order_id": order.client_order_id,
    }
    if order.quantity is not None:
        kwargs["qty"] = _to_api_number(order.quantity)
    else:
        kwargs["notional"] = _to_api_number(order.notional)

    if order.order_type is OrderType.LIMIT:
        kwargs["limit_price"] = _to_api_number(order.limit_price)
    elif order.order_type is OrderType.STOP:
        kwargs["stop_price"] = _to_api_number(order.stop_price)
    elif order.order_type is OrderType.STOP_LIMIT:
        kwargs["stop_price"] = _to_api_number(order.stop_price)
        kwargs["limit_price"] = _to_api_number(order.stop_limit_price)
    elif order.order_type is OrderType.TRAIL:
        if order.trail_price is not None:
            kwargs["trail_price"] = _to_api_number(order.trail_price)
        if order.trail_percent is not None:
            kwargs["trail_percent"] = _to_api_number(order.trail_percent)

    try:
        return request_cls(**kwargs)
    except (ValidationError, ValueError) as exc:
        raise OrderValidationError(str(exc)) from exc


# --- Task 11: parsing broker responses ---------------------------------------

_SENTINEL = object()


def _field(response: object, name: str, default: object = None) -> object:
    """Read `name` off `response`, whether it's an attribute-bearing object
    (a real alpaca-py model) or a plain Mapping (a raw dict payload).

    getattr's own default only kicks in when the attribute is genuinely
    missing, never when it is present but None -- so a present-but-None
    attribute is returned as None, not silently swapped for the Mapping path.
    """
    value = getattr(response, name, _SENTINEL)
    if value is not _SENTINEL:
        return value
    if isinstance(response, Mapping):
        return response.get(name, default)
    return default


def _to_decimal(value: object) -> Decimal | None:
    """Coerce a raw Alpaca numeric (str | float | None) to Decimal.

    Never Decimal(float) directly -- always via str -- to avoid binary float
    noise leaking into our authoritative Decimal fields.
    """
    return None if value is None else Decimal(str(value))


def parse_broker_order(response: object, strategy_name: str) -> Order | None:
    """Parse an Alpaca order response (a real `alpaca.trading.models.Order`
    instance or an equivalent raw dict) into our `Order` entity.

    Returns None -- logging a warning -- only when both `qty` and `notional`
    are absent, which shouldn't happen for a real order. This deliberately
    diverges from a naive lumibot-style port that returns None whenever `qty`
    alone is None: Alpaca returns `qty: null, notional: "500"` for notional
    orders, and a naive `qty is None` check would silently drop them.
    """
    qty = _field(response, "qty")
    notional = _field(response, "notional")
    if qty is None and notional is None:
        logger.warning(
            "parse_broker_order: response has neither qty nor notional, skipping: %r",
            response,
        )
        return None

    raw_type = _field(response, "type")
    if raw_type is None:
        raw_type = _field(response, "order_type")
    order_type = OrderType(_normalize_key(raw_type))

    stop_price = _to_decimal(_field(response, "stop_price"))
    limit_price = _to_decimal(_field(response, "limit_price"))
    stop_limit_price = None
    if order_type is OrderType.STOP_LIMIT:
        stop_limit_price = limit_price
        limit_price = None

    raw_id = _field(response, "id")
    filled_quantity = _to_decimal(_field(response, "filled_qty"))

    return Order(
        strategy_name=strategy_name,
        asset=Asset(symbol=_field(response, "symbol"), asset_type=AssetType.STOCK),
        side=OrderSide(_normalize_key(_field(response, "side"))),
        order_type=order_type,
        quantity=_to_decimal(qty),
        notional=_to_decimal(notional),
        time_in_force=TimeInForce(_normalize_key(_field(response, "time_in_force"))),
        limit_price=limit_price,
        stop_price=stop_price,
        stop_limit_price=stop_limit_price,
        trail_price=_to_decimal(_field(response, "trail_price")),
        trail_percent=_to_decimal(_field(response, "trail_percent")),
        status=map_status(_field(response, "status")),
        identifier=str(raw_id) if raw_id is not None else uuid4().hex,
        client_order_id=_field(response, "client_order_id"),
        filled_quantity=filled_quantity if filled_quantity is not None else Decimal(0),
        avg_fill_price=_to_decimal(_field(response, "filled_avg_price")),
        created_at=_field(response, "created_at"),
        updated_at=_field(response, "updated_at"),
        raw=response,
    )


def parse_broker_orders(responses: object, strategy_name: str) -> list[Order]:
    """Parse an iterable of broker order responses, dropping any that
    `parse_broker_order` returns None for."""
    return [
        order
        for order in (parse_broker_order(response, strategy_name) for response in responses)
        if order is not None
    ]


def parse_broker_position(response: object, strategy_name: str) -> Position:
    """Parse an Alpaca position response (a real `alpaca.trading.models.Position`
    instance or an equivalent raw dict) into our `Position` entity."""
    return Position(
        strategy_name=strategy_name,
        asset=Asset(symbol=_field(response, "symbol"), asset_type=AssetType.STOCK),
        quantity=_to_decimal(_field(response, "qty")),
        side=PositionSide(_normalize_key(_field(response, "side"))),
        avg_fill_price=_to_decimal(_field(response, "avg_entry_price")),
        current_price=_to_decimal(_field(response, "current_price")),
        market_value=_to_decimal(_field(response, "market_value")),
        unrealized_pnl=_to_decimal(_field(response, "unrealized_pl")),
        raw=response,
    )

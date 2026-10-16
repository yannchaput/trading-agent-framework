"""Pure translation between our entities and Tastytrade's REST JSON (dasherized keys).

No I/O, no state, no client instances: the broker passes in what it fetched (fractional
eligibility, a last price). Money stays `Decimal`, sent and parsed as strings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any

from trading_agent_framework.brokers.tastytrade.values import decimal_of, optional_decimal
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.utils.errors import BrokerError, OrderValidationError

logger = logging.getLogger(__name__)

SOURCE = "trading-agent-framework"
# Smallest fractional step of an eligible equity (probe: GET /instruments/quantity-decimal-precisions).
# PROVISIONAL (brief default): the sandbox probe output was not available; confirm against spec section 11.
FRACTIONAL_QUANTUM = Decimal("0.00001")

_PENNY = Decimal("0.01")
_SUB_PENNY = Decimal("0.0001")

# Ours -> (regular hours, extended hours); None: Tastytrade has no extended-hours variant.
_TIME_IN_FORCE: dict[TimeInForce, tuple[str, str | None]] = {
    TimeInForce.DAY: ("Day", "Ext"),
    TimeInForce.GTC: ("GTC", "GTC Ext"),
    TimeInForce.IOC: ("IOC", None),
}


# --- identifiers and symbols -------------------------------------------------------------


def external_identifier_for(strategy_name: str, identifier: str) -> str:
    return f"{strategy_name}:{identifier}"


def identifier_from_external(external: str | None, strategy_name: str) -> str | None:
    prefix = f"{strategy_name}:"
    if external and external.startswith(prefix) and len(external) > len(prefix):
        return external[len(prefix) :]
    return None


def to_tastytrade_symbol(symbol: str) -> str:
    """Class shares: our (Alpaca's) `BRK.B` is Tastytrade's `BRK/B`.

    PROVISIONAL (brief default): confirm against the probe findings (spec section 11).
    """
    return symbol.replace(".", "/")


def from_tastytrade_symbol(symbol: str) -> str:
    return symbol.replace("/", ".")


# --- conform -------------------------------------------------------------------------------


def conform_order(order: Order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order:
    """Apply Tastytrade's order rules: reject what it can't take, floor what it can't split.

    `fractional_eligible` is the equity's `is-fractional-quantity-eligible`; `last_price` is only
    needed to turn a non-eligible notional order into whole shares.
    """
    if order.order_type is OrderType.TRAIL:
        raise OrderValidationError("Tastytrade has no trailing stops; use a STOP order")
    time_in_force = _TIME_IN_FORCE.get(order.time_in_force)
    if time_in_force is None:
        raise OrderValidationError(f"Tastytrade does not support time_in_force={order.time_in_force}")
    if order.extended_hours and time_in_force[1] is None:
        raise OrderValidationError(f"Tastytrade has no extended-hours {order.time_in_force} orders")
    if order.order_type is OrderType.MARKET and order.time_in_force is TimeInForce.GTC:
        raise OrderValidationError("Tastytrade does not accept GTC market orders")
    if order.notional is not None:
        return _conform_notional(order, fractional_eligible=fractional_eligible, last_price=last_price)
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    if order.quantity % 1 == 0:
        return order
    if order.order_type is not OrderType.MARKET:
        raise OrderValidationError(f"fractional quantity is only supported for MARKET orders (order_type={order.order_type}): a fractional stop or limit would leave the remainder behind")
    if fractional_eligible:
        floored = order.quantity.quantize(FRACTIONAL_QUANTUM, rounding=ROUND_FLOOR)
        if floored != order.quantity:  # keep 4.57 as "4.57", not "4.57000"
            order.quantity = floored
        if order.quantity <= 0:
            raise OrderValidationError(f"order {order.identifier}: quantity rounds to zero at Tastytrade's precision {FRACTIONAL_QUANTUM}")
        return order
    order.quantity = _whole_shares(order, order.quantity, f"quantity {order.quantity}")
    return order


def _conform_notional(order: Order, *, fractional_eligible: bool, last_price: Decimal | None) -> Order:
    if order.order_type is not OrderType.MARKET:
        raise OrderValidationError(f"notional is only supported for MARKET orders (order_type={order.order_type})")
    if fractional_eligible:
        return order
    assert order.notional is not None
    if last_price is None or last_price <= 0:
        raise OrderValidationError(f"{order.asset.symbol} can't be traded by dollar amount at Tastytrade and has no last price to convert {order.notional} into shares")
    order.quantity = _whole_shares(order, order.notional / last_price, f"notional {order.notional} at {last_price}")
    order.notional = None
    return order


def _whole_shares(order: Order, shares: Decimal, what: str) -> Decimal:
    whole = shares.to_integral_value(rounding=ROUND_FLOOR)
    if whole < 1:
        raise OrderValidationError(f"order {order.identifier}: {what} is below one whole share of {order.asset.symbol}, which Tastytrade trades in whole shares only")
    logger.warning("Tastytrade trades %s in whole shares only: %s -> %s shares", order.asset.symbol, what, whole)
    return whole


# --- order JSON ----------------------------------------------------------------------------


def round_price(price: Decimal) -> Decimal:
    return price.quantize(_PENNY if price >= 1 else _SUB_PENNY, rounding=ROUND_HALF_UP)


def _price(price: Decimal) -> str:
    return format(round_price(price), "f")


def _require(value: Decimal | None, name: str, order: Order) -> Decimal:
    if value is None:
        raise OrderValidationError(f"{order.order_type.value} order {order.identifier} needs {name}")
    return value


def _pricing(order: Order, effect: str) -> dict[str, Any]:
    match order.order_type:
        case OrderType.MARKET:
            return {"order-type": "Market"}
        case OrderType.LIMIT:
            return {"order-type": "Limit", "price": _price(_require(order.limit_price, "limit_price", order)), "price-effect": effect}
        case OrderType.STOP:
            return {"order-type": "Stop", "stop-trigger": _price(_require(order.stop_price, "stop_price", order))}
        case OrderType.STOP_LIMIT:
            return {
                "order-type": "Stop Limit",
                "stop-trigger": _price(_require(order.stop_price, "stop_price", order)),
                "price": _price(_require(order.stop_limit_price, "stop_limit_price", order)),
                "price-effect": effect,
            }
    raise OrderValidationError(f"Tastytrade cannot place {order.order_type} orders")


def build_order_json(order: Order) -> dict[str, Any]:
    """The `POST /accounts/{a}/orders` body for an already-conformed `order`."""
    buy = order.side is OrderSide.BUY
    effect = "Debit" if buy else "Credit"
    regular, extended = _TIME_IN_FORCE[order.time_in_force]
    leg: dict[str, Any] = {"instrument-type": "Equity", "symbol": to_tastytrade_symbol(order.asset.symbol), "action": "Buy to Open" if buy else "Sell to Close"}
    body: dict[str, Any] = {"time-in-force": extended if order.extended_hours and extended else regular, "source": SOURCE}
    if order.client_order_id:
        body["external-identifier"] = order.client_order_id
    if order.notional is not None:
        body.update({"order-type": "Notional Market", "value": format(order.notional.quantize(_PENNY, rounding=ROUND_FLOOR), "f"), "value-effect": effect})
    else:
        leg["quantity"] = format(_require(order.quantity, "quantity", order), "f")
        body.update(_pricing(order, effect))
    body["legs"] = [leg]
    return body


def build_replace_json(order: Order) -> dict[str, Any]:
    """The `PUT /accounts/{a}/orders/{id}` body: the order without its legs (they can't change on a replace)."""
    body = build_order_json(order)
    del body["legs"]
    return body


# --- statuses ------------------------------------------------------------------------------

TERMINAL_STATUSES: frozenset[str] = frozenset({"Filled", "Cancelled", "Rejected", "Expired", "Removed", "Partially Removed"})
# Statuses that end an order without (more) fills. `Expired` is in it, but `map_status` returns
# `OrderStatus.EXPIRED` for it (not CANCELED): callers treating this set as "cancelled" must not
# compare against `OrderStatus.CANCELED` alone.
CANCELED_STATUSES: frozenset[str] = frozenset({"Cancelled", "Removed", "Partially Removed", "Expired"})

_SUBMITTED = frozenset({"Received", "Routed", "In Flight", "Contingent"})
_WORKING = frozenset({"Live", "Replace Requested"})
_STATUSES: dict[str, OrderStatus] = {
    "Cancel Requested": OrderStatus.CANCELLING,
    "Filled": OrderStatus.FILL,
    "Cancelled": OrderStatus.CANCELED,
    "Removed": OrderStatus.CANCELED,
    "Partially Removed": OrderStatus.CANCELED,
    "Expired": OrderStatus.EXPIRED,
    "Rejected": OrderStatus.ERROR,
}


def map_status(status: str, *, has_fills: bool) -> OrderStatus:
    if status in _SUBMITTED:
        return OrderStatus.SUBMITTED
    if status in _WORKING:
        return OrderStatus.PARTIAL_FILL if has_fills else OrderStatus.NEW
    mapped = _STATUSES.get(status)
    if mapped is None:
        logger.warning("Unknown Tastytrade order status %r", status)
        return OrderStatus.UNKNOWN
    return mapped


# --- parsing -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    quantity: Decimal
    price: Decimal


def parse_fills(data: Mapping[str, Any]) -> list[Fill]:
    fills: list[Fill] = []
    for leg in data.get("legs") or []:
        if not isinstance(leg, Mapping):
            raise BrokerError(f"Tastytrade order {data.get('id')} has a leg that is not an object: {leg!r}")
        for fill in leg.get("fills") or []:
            if not isinstance(fill, Mapping):
                raise BrokerError(f"Tastytrade order {data.get('id')} has a fill that is not an object: {fill!r}")
            fill_id = fill.get("fill-id")
            if fill_id is None:
                raise BrokerError(f"Tastytrade order {data.get('id')} has a fill without a fill-id")
            fills.append(Fill(str(fill_id), abs(decimal_of(fill.get("quantity"), "fill quantity")), abs(decimal_of(fill.get("fill-price"), "fill price"))))
    return fills


_ORDER_TYPES: dict[str, OrderType] = {
    "Market": OrderType.MARKET,
    "Notional Market": OrderType.MARKET,
    "Limit": OrderType.LIMIT,
    "Stop": OrderType.STOP,
    "Stop Limit": OrderType.STOP_LIMIT,
}
_TIME_IN_FORCE_BACK: dict[str, tuple[TimeInForce, bool]] = {
    "Day": (TimeInForce.DAY, False),
    "Ext": (TimeInForce.DAY, True),
    "GTC": (TimeInForce.GTC, False),
    "GTC Ext": (TimeInForce.GTC, True),
    "IOC": (TimeInForce.IOC, False),
}


_SIDES: dict[str, OrderSide] = {
    "Buy to Open": OrderSide.BUY,
    "Buy to Close": OrderSide.BUY,
    "Sell to Open": OrderSide.SELL,
    "Sell to Close": OrderSide.SELL,
}
_DIRECTIONS = frozenset({"Long", "Short", "Zero"})


def is_parsable_order(data: Mapping[str, Any]) -> bool:
    """A single-leg equity order of a type we model (manual option or `Marketable Limit` orders are not)."""
    legs = data.get("legs")
    return (
        isinstance(legs, list)
        and len(legs) == 1
        and isinstance(legs[0], Mapping)
        and legs[0].get("instrument-type") == "Equity"
        and isinstance(data.get("order-type"), str)
        and data["order-type"] in _ORDER_TYPES
    )


def parse_order(data: Mapping[str, Any], strategy_name: str) -> Order:
    """An `Order` from a Tastytrade order snapshot (REST or account streamer)."""
    if not is_parsable_order(data):
        raise BrokerError(f"Tastytrade order {data.get('id')} ({data.get('order-type')!r}) is not a single-leg equity order of a type this framework models")
    raw_type = str(data["order-type"])
    order_type = _ORDER_TYPES[raw_type]
    leg = data["legs"][0]
    raw_tif = str(data.get("time-in-force"))
    tif = _TIME_IN_FORCE_BACK.get(raw_tif)
    if tif is None:
        logger.warning("Unknown Tastytrade time-in-force %r on order %s; treating it as Day", raw_tif, data.get("id"))
        tif = (TimeInForce.DAY, False)
    time_in_force, extended = tif
    action = leg.get("action")
    side = _SIDES.get(action) if isinstance(action, str) else None
    if side is None:
        raise BrokerError(f"Tastytrade order {data.get('id')} has an unknown action: {action!r}")
    asset = Asset(from_tastytrade_symbol(str(leg.get("symbol") or data.get("underlying-symbol"))))
    if raw_type == "Notional Market":
        order = Order(strategy_name, asset, side, order_type, notional=abs(decimal_of(data.get("value"), "order value")), time_in_force=time_in_force, extended_hours=extended)
    else:
        quantity = abs(decimal_of(leg.get("quantity", data.get("size")), "order quantity"))
        order = Order(strategy_name, asset, side, order_type, quantity=quantity, time_in_force=time_in_force, extended_hours=extended)
    price = optional_decimal(data.get("price"), "order price")
    trigger = optional_decimal(data.get("stop-trigger"), "order stop-trigger")
    if order_type is OrderType.LIMIT:
        order.limit_price = abs(price) if price is not None else None
    elif order_type is OrderType.STOP:
        order.stop_price = abs(trigger) if trigger is not None else None
    elif order_type is OrderType.STOP_LIMIT:
        order.stop_price = abs(trigger) if trigger is not None else None
        order.stop_limit_price = abs(price) if price is not None else None
    raw_external = data.get("external-identifier")
    external = raw_external if isinstance(raw_external, str) else None
    order.identifier = identifier_from_external(external, strategy_name) or str(data.get("id"))
    order.client_order_id = external or None
    fills = parse_fills(data)
    if fills:
        order.filled_quantity = sum((f.quantity for f in fills), Decimal(0))
        if order.filled_quantity > 0:
            order.avg_fill_price = sum((f.quantity * f.price for f in fills), Decimal(0)) / order.filled_quantity
    order.status = map_status(str(data.get("status", "")), has_fills=bool(fills))
    if order.status is OrderStatus.ERROR:
        order.error_message = str(data.get("reject-reason") or "rejected by Tastytrade")
    order.update_raw(dict(data))
    return order


def parse_position(data: Mapping[str, Any], strategy_name: str) -> Position | None:
    """An equity `Position` (shorts have a positive quantity and `side=SHORT`); None for closed or non-equity ones."""
    if data.get("instrument-type") != "Equity":
        return None
    direction = data.get("quantity-direction")
    if not isinstance(direction, str) or direction not in _DIRECTIONS:
        raise BrokerError(f"Tastytrade position {data.get('symbol')} has an unknown quantity-direction: {direction!r}")
    quantity = abs(decimal_of(data.get("quantity"), "position quantity"))
    if direction == "Zero" or quantity == 0:
        return None
    return Position(
        strategy_name,
        Asset(from_tastytrade_symbol(str(data.get("symbol")))),
        quantity,
        PositionSide.SHORT if direction == "Short" else PositionSide.LONG,
        avg_fill_price=optional_decimal(data.get("average-open-price"), "position average-open-price"),
        raw=dict(data),
    )

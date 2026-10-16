"""Pure translation between our entities and Tastytrade's REST JSON (dasherized keys).

No I/O, no state, no client instances: the broker passes in what it fetched (fractional
eligibility, a last price). Money stays `Decimal`, sent and parsed as strings.
"""

from __future__ import annotations

import logging
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any

from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

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

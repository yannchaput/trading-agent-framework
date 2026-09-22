"""Pure IBKR order translation: our `Order` <-> `ib_async` `Contract`/`Order`/`Trade`.

No I/O, no state, no client instances (same rules as `brokers/alpaca/orders.py`). This is the
IBKR counterpart of Alpaca's float boundary: `ib_async` order fields are floats; everything
handed back to the framework is `Decimal`.
"""

from __future__ import annotations

import logging
from decimal import ROUND_FLOOR, Decimal

from ib_async import Order as IbOrder
from ib_async import Stock

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderType, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.utils.errors import OrderValidationError

logger = logging.getLogger(__name__)

EXCHANGE = "SMART"
CURRENCY = "USD"

_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "MKT",
    OrderType.LIMIT: "LMT",
    OrderType.STOP: "STP",
    OrderType.STOP_LIMIT: "STP LMT",
    OrderType.TRAIL: "TRAIL",
}
_AT_THE_CLOSE: dict[OrderType, str] = {OrderType.MARKET: "MOC", OrderType.LIMIT: "LOC"}
_TIME_IN_FORCE: dict[TimeInForce, str] = {
    TimeInForce.DAY: "DAY",
    TimeInForce.GTC: "GTC",
    TimeInForce.IOC: "IOC",
    TimeInForce.FOK: "FOK",
    TimeInForce.OPG: "OPG",
}


def build_contract(asset: Asset) -> Stock:
    return Stock(asset.symbol, EXCHANGE, CURRENCY)


def client_order_id_for(strategy_name: str, identifier: str) -> str:
    return f"{strategy_name}:{identifier}"


def conform_order(order: Order) -> Order:
    """IBKR's API trades whole shares: reject dollar amounts, floor fractional share counts."""
    if order.notional is not None:
        raise OrderValidationError("IBKR does not accept notional (dollar-amount) orders; pass a share quantity")
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    whole = order.quantity.to_integral_value(rounding=ROUND_FLOOR)
    if whole < 1:
        raise OrderValidationError(
            f"order {order.identifier} for {order.quantity} {order.asset.symbol} is below one whole share (IBKR trades whole shares only)"
        )
    if whole != order.quantity:
        logger.warning("IBKR trades whole shares only: %s quantity %s floored to %s", order.asset.symbol, order.quantity, whole)
        order.quantity = whole
    return order


def _require(value: Decimal | None, name: str, order: Order) -> float:
    if value is None:
        raise OrderValidationError(f"{order.order_type.value} order {order.identifier} needs {name}")
    return float(value)


def build_order(order: Order) -> IbOrder:
    """The `ib_async` order for an already-conformed `order` (quantity in whole shares)."""
    if order.quantity is None:
        raise OrderValidationError(f"order {order.identifier} has no quantity")
    ib_order = IbOrder()
    ib_order.action = "BUY" if order.side is OrderSide.BUY else "SELL"
    ib_order.totalQuantity = float(order.quantity)
    if order.time_in_force is TimeInForce.CLS:
        at_the_close = _AT_THE_CLOSE.get(order.order_type)
        if at_the_close is None:
            raise OrderValidationError("an at-the-close (CLS) order must be a market or limit order")
        ib_order.orderType = at_the_close
        ib_order.tif = "DAY"
    else:
        ib_order.orderType = _ORDER_TYPES[order.order_type]
        ib_order.tif = _TIME_IN_FORCE[order.time_in_force]
    if order.order_type is OrderType.LIMIT:
        ib_order.lmtPrice = _require(order.limit_price, "limit_price", order)
    elif order.order_type is OrderType.STOP:
        ib_order.auxPrice = _require(order.stop_price, "stop_price", order)
    elif order.order_type is OrderType.STOP_LIMIT:
        ib_order.auxPrice = _require(order.stop_price, "stop_price", order)
        ib_order.lmtPrice = _require(order.stop_limit_price, "stop_limit_price", order)
    elif order.order_type is OrderType.TRAIL:
        if order.trail_price is not None:
            ib_order.auxPrice = float(order.trail_price)
        elif order.trail_percent is not None:
            ib_order.trailingPercent = float(order.trail_percent)
        else:
            raise OrderValidationError(f"trailing stop order {order.identifier} needs trail_price or trail_percent")
    ib_order.outsideRth = order.extended_hours
    ib_order.orderRef = order.client_order_id or ""
    ib_order.transmit = True
    return ib_order

"""Pure IBKR order translation: our `Order` <-> `ib_async` `Contract`/`Order`/`Trade`.

No I/O, no state, no client instances (same rules as `brokers/alpaca/orders.py`). This is the
IBKR counterpart of Alpaca's float boundary: `ib_async` order fields are floats; everything
handed back to the framework is `Decimal`.
"""

from __future__ import annotations

import logging
import math
from decimal import ROUND_FLOOR, Decimal

from ib_async import Order as IbOrder
from ib_async import PortfolioItem, Stock, Trade
from ib_async.util import UNSET_DOUBLE

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide, TimeInForce
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
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


PENDING_STATUSES: frozenset[str] = frozenset({"", "PendingSubmit", "ApiPending"})
_ALWAYS_REJECTED_STATUSES = frozenset({"Inactive"})
_CANCELLED_STATUSES = frozenset({"Cancelled", "ApiCancelled"})

_STATUS: dict[str, OrderStatus] = {
    "PendingSubmit": OrderStatus.NEW,
    "PreSubmitted": OrderStatus.NEW,
    "Submitted": OrderStatus.NEW,
    "Filled": OrderStatus.FILL,
    "Cancelled": OrderStatus.CANCELED,
    "ApiCancelled": OrderStatus.CANCELED,
    "Inactive": OrderStatus.ERROR,
    "PendingCancel": OrderStatus.CANCELLING,
}
_STATUS_EVENT: dict[str, OrderEvent] = {
    "PreSubmitted": OrderEvent.NEW,
    "Submitted": OrderEvent.NEW,
    "Cancelled": OrderEvent.CANCELED,
    "ApiCancelled": OrderEvent.CANCELED,
    "Inactive": OrderEvent.ERROR,
}
_PARSED_ORDER_TYPES: dict[str, tuple[OrderType, TimeInForce | None]] = {
    **{ib: (ours, None) for ours, ib in _ORDER_TYPES.items()},
    "MOC": (OrderType.MARKET, TimeInForce.CLS),
    "LOC": (OrderType.LIMIT, TimeInForce.CLS),
}
_PARSED_TIME_IN_FORCE: dict[str, TimeInForce] = {ib: ours for ours, ib in _TIME_IN_FORCE.items()}


def map_status(status: str) -> OrderStatus:
    return _STATUS.get(status, OrderStatus.UNKNOWN)


def map_status_event(status: str) -> OrderEvent | None:
    """The tracker event an order-status update means; fills come from executions instead."""
    return _STATUS_EVENT.get(status)


def identifier_from_order_ref(order_ref: str, strategy_name: str) -> str | None:
    prefix = f"{strategy_name}:"
    if order_ref.startswith(prefix) and len(order_ref) > len(prefix):
        return order_ref[len(prefix):]
    return None


def to_decimal(value: float | None) -> Decimal | None:
    if value is None or math.isnan(value) or value == UNSET_DOUBLE:
        return None
    return Decimal(str(value))


def parse_trade(trade: Trade, strategy_name: str) -> Order:
    ib_order = trade.order
    order_type, forced_tif = _PARSED_ORDER_TYPES.get(ib_order.orderType, (OrderType.MARKET, None))
    identifier = identifier_from_order_ref(ib_order.orderRef, strategy_name)
    limit = to_decimal(ib_order.lmtPrice)
    aux = to_decimal(ib_order.auxPrice)
    return Order(
        strategy_name=strategy_name,
        asset=Asset(trade.contract.symbol),
        side=OrderSide.BUY if ib_order.action == "BUY" else OrderSide.SELL,
        order_type=order_type,
        quantity=to_decimal(ib_order.totalQuantity),
        time_in_force=forced_tif or _PARSED_TIME_IN_FORCE.get(ib_order.tif, TimeInForce.DAY),
        limit_price=limit if order_type is OrderType.LIMIT else None,
        stop_price=aux if order_type in (OrderType.STOP, OrderType.STOP_LIMIT) else None,
        stop_limit_price=limit if order_type is OrderType.STOP_LIMIT else None,
        trail_price=aux if order_type is OrderType.TRAIL else None,
        trail_percent=to_decimal(ib_order.trailingPercent) if order_type is OrderType.TRAIL else None,
        extended_hours=bool(ib_order.outsideRth),
        status=map_status(trade.orderStatus.status),
        identifier=identifier if identifier is not None else f"ibkr:{ib_order.permId}",
        client_order_id=ib_order.orderRef if identifier is not None else None,
        filled_quantity=to_decimal(trade.orderStatus.filled) or Decimal(0),
        avg_fill_price=to_decimal(trade.orderStatus.avgFillPrice) or None,  # IBKR reports 0.0 before any fill
        raw=trade,
    )


def parse_portfolio_item(item: PortfolioItem, strategy_name: str) -> Position | None:
    quantity = to_decimal(item.position) or Decimal(0)
    if quantity == 0:
        return None
    return Position(
        strategy_name=strategy_name,
        asset=Asset(item.contract.symbol),
        quantity=abs(quantity),
        side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
        avg_fill_price=to_decimal(item.averageCost),
        current_price=to_decimal(item.marketPrice),
        market_value=to_decimal(item.marketValue),
        unrealized_pnl=to_decimal(item.unrealizedPNL),
        raw=item,
    )


def _is_rejection(trade: Trade) -> bool:
    """`Inactive` is IBKR's own-initiative rejection status, always a rejection. `Cancelled`/
    `ApiCancelled` only count as a rejection when the log also carries a real IBKR error code --
    an IOC/FOK order that simply didn't fill cancels the exact same way, with no error at all.
    """
    status = trade.orderStatus.status
    if status in _ALWAYS_REJECTED_STATUSES:
        return True
    if status in _CANCELLED_STATUSES:
        return any(entry.errorCode for entry in trade.log)
    return False


def rejection_message(trade: Trade) -> str | None:
    """Why IBKR refused `trade`, or None while it is pending, working, or normally cancelled."""
    if not _is_rejection(trade):
        return None
    status = trade.orderStatus.status
    for entry in reversed(trade.log):
        if entry.message:
            suffix = f" (IBKR error {entry.errorCode})" if entry.errorCode else ""
            return f"{entry.message}{suffix}"
    return f"order {status} by IBKR"

"""Pure Alpaca order translation logic: status/event mapping, price conforming,
and pre-submission validation.

This module holds no I/O, no state, and no client instances -- it is a
collection of module-level functions and constants operating on our own
`Order` entity and plain Python values. Per the project's global constraints,
`orders.py` is the only module in the codebase (besides `account.py`)
permitted to import `alpaca.trading.requests`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    OrderRequest,
    ReplaceOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TrailingStopOrderRequest,
)
from pydantic import ValidationError

from trading_agent_framework.brokers.alpaca.symbols import from_alpaca_symbol, to_alpaca_symbol
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
from trading_agent_framework.utils.errors import OrderValidationError

if TYPE_CHECKING:
    from datetime import datetime

    from alpaca.trading.models import AccountConfiguration as AlpacaAccountConfiguration
    from alpaca.trading.models import Calendar as AlpacaCalendarModel
    from alpaca.trading.models import ClosePositionResponse as AlpacaClosePositionResponse
    from alpaca.trading.models import Order as AlpacaOrderModel
    from alpaca.trading.models import Position as AlpacaPositionModel
    from alpaca.trading.models import TradeAccount as AlpacaTradeAccount

logger = logging.getLogger(__name__)


class AlpacaTradingClient(Protocol):
    """Structural contract for what `AlpacaBroker` actually calls on a
    `alpaca.trading.client.TradingClient` (or a hand-written test double).

    A real `TradingClient` satisfies this automatically (structural typing);
    a fake only needs to implement the methods below, not inherit anything.

    Narrower than `TradingClient`'s own `submit_order`/`get_order_by_id`
    signatures in one respect: both are typed here as always returning a
    parsed `Order` model, since this project never passes `raw_data=True` and
    therefore never receives the `RawData` (dict) half of alpaca-py's real
    `Order | RawData` return type.
    """

    def submit_order(self, order_data: OrderRequest) -> AlpacaOrderModel: ...
    def cancel_order_by_id(self, order_id: str) -> None: ...
    def get_orders(self, filter: GetOrdersRequest) -> list[AlpacaOrderModel]: ...
    def get_order_by_id(self, order_id: str) -> AlpacaOrderModel: ...
    def get_all_positions(self) -> list[AlpacaPositionModel]: ...
    def get_account(self) -> AlpacaTradeAccount: ...
    def get_calendar(self, filters: GetCalendarRequest) -> list[AlpacaCalendarModel]: ...
    def replace_order_by_id(
        self, order_id: str, order_data: ReplaceOrderRequest
    ) -> AlpacaOrderModel: ...
    def close_position(
        self, symbol_or_asset_id: str, close_options: ClosePositionRequest
    ) -> AlpacaOrderModel: ...
    def close_all_positions(self, cancel_orders: bool) -> list[AlpacaClosePositionResponse]: ...
    def get_account_configurations(self) -> AlpacaAccountConfiguration: ...
    def set_account_configurations(
        self, account_configurations: AlpacaAccountConfiguration
    ) -> AlpacaAccountConfiguration: ...


# --- status / event maps -----------------------------------------------------

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
        # matches ALPACA_STATUS_MAP's treatment of the same wire value
        "done_for_day": OrderEvent.CANCELED,
        "stopped": None,
        "suspended": None,
        "calculated": None,
        "order_replace_rejected": None,
        "order_cancel_rejected": None,
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


# --- price conforming ---------------------------------------------------------

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


# --- pre-submission validation -------------------------------------------------

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


# --- request builders ---------------------------------------------------------

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
        "symbol": to_alpaca_symbol(order.asset.symbol),
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
        # kwargs is heterogeneous by construction -- it's built once per order
        # type from a shared dict and splatted into whichever of the 5 request
        # classes _REQUEST_BY_TYPE selected, each with different fields. A
        # static checker can't verify that dynamic spread against a specific
        # constructor's signature; pydantic validates it at runtime instead,
        # and any mismatch surfaces as the ValidationError caught below.
        return request_cls(**kwargs)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    except (ValidationError, ValueError) as exc:
        raise OrderValidationError(str(exc)) from exc


def build_get_orders_request(limit: int = 100, *, open_only: bool = False) -> GetOrdersRequest:
    """Build a GetOrdersRequest for every order (default) or only the open ones."""
    status = QueryOrderStatus.OPEN if open_only else QueryOrderStatus.ALL
    return GetOrdersRequest(status=status, limit=limit)


def build_replace_order_request(
    *, limit_price: Decimal | None = None, stop_price: Decimal | None = None
) -> ReplaceOrderRequest:
    """Build the PATCH /orders/{id} body; prices are rounded to Alpaca's ticks first."""
    if limit_price is None and stop_price is None:
        raise OrderValidationError("modify_order needs a new limit_price and/or stop_price")
    try:
        return ReplaceOrderRequest(
            limit_price=_to_api_number(None if limit_price is None else round_price(limit_price)),
            stop_price=_to_api_number(
                None if stop_price is None else round_stop_price(stop_price)
            ),
        )
    except (ValidationError, ValueError) as exc:
        raise OrderValidationError(str(exc)) from exc


_CLOSE_PERCENTAGE_PRECISION = Decimal("0.000000001")  # Alpaca caps `percentage` at 9dp


def build_close_position_request(fraction: Decimal) -> ClosePositionRequest:
    """Close `fraction` (0 < fraction <= 1) of a position, sent as Alpaca's percentage string."""
    if not Decimal(0) < fraction <= Decimal(1):
        raise OrderValidationError(f"close fraction must be in (0, 1], got {fraction}")
    percentage = (fraction * 100).quantize(_CLOSE_PERCENTAGE_PRECISION)
    return ClosePositionRequest(percentage=format(percentage, "f"))


# --- response parsing -----------------------------------------------------------

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
        asset=Asset(
            symbol=from_alpaca_symbol(cast(str, _field(response, "symbol"))), asset_type=AssetType.STOCK
        ),
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
        client_order_id=cast("str | None", _field(response, "client_order_id")),
        filled_quantity=filled_quantity if filled_quantity is not None else Decimal(0),
        avg_fill_price=_to_decimal(_field(response, "filled_avg_price")),
        created_at=cast("datetime | None", _field(response, "created_at")),
        updated_at=cast("datetime | None", _field(response, "updated_at")),
        raw=response,
    )


def parse_broker_orders(responses: Iterable[object], strategy_name: str) -> list[Order]:
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
    quantity = _to_decimal(_field(response, "qty"))
    # alpaca.trading.models.Position.qty is a required field -- unlike an Order,
    # which can be notional-only, a position always has a share quantity.
    assert quantity is not None, "Alpaca position response is missing qty"
    return Position(
        strategy_name=strategy_name,
        asset=Asset(
            symbol=from_alpaca_symbol(cast(str, _field(response, "symbol"))), asset_type=AssetType.STOCK
        ),
        quantity=quantity,
        side=PositionSide(_normalize_key(_field(response, "side"))),
        avg_fill_price=_to_decimal(_field(response, "avg_entry_price")),
        current_price=_to_decimal(_field(response, "current_price")),
        market_value=_to_decimal(_field(response, "market_value")),
        unrealized_pnl=_to_decimal(_field(response, "unrealized_pl")),
        raw=response,
    )


def parse_close_all_responses(responses: Iterable[object], strategy_name: str) -> list[Order]:
    """Orders placed by close_all_positions. Each response `body` is either an order or a
    FailedClosePositionDetails (code/message, no id); failures are logged and skipped."""
    closed: list[Order] = []
    for response in responses:
        body = _field(response, "body")
        if _field(body, "id") is None:
            logger.warning(
                "Alpaca could not close position %s: %s",
                _field(response, "symbol"),
                _field(body, "message"),
            )
            continue
        order = parse_broker_order(body, strategy_name)
        if order is not None:
            closed.append(order)
    return closed

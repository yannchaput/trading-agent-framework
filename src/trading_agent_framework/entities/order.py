from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import (
    ACTIVE_ORDER_STATUSES,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trading_agent_framework.errors import OrderValidationError

_DECIMAL_FIELDS = (
    "quantity",
    "notional",
    "limit_price",
    "stop_price",
    "stop_limit_price",
    "trail_price",
    "trail_percent",
    "filled_quantity",
    "avg_fill_price",
)


def _to_decimal(value: int | float | str | Decimal | None) -> Decimal | None:
    """Coerce a numeric input to an exact Decimal, or pass through None.

    Uses Decimal(str(value)) for int/float so binary float noise never leaks in;
    a str input is already exact.
    """
    if value is None or isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class Transaction:
    quantity: Decimal
    price: Decimal
    timestamp: datetime


@dataclass(eq=False)
class Order:
    """An order to be submitted to / tracked against a broker.

    eq=False: identity is by `identifier`, which the tracker depends on.
    No slots=True: carries an untyped `raw` and is mutated constantly.
    """

    strategy_name: str
    asset: Asset
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal | None = None
    notional: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    stop_limit_price: Decimal | None = None
    trail_price: Decimal | None = None
    trail_percent: Decimal | None = None
    extended_hours: bool = False
    status: OrderStatus = OrderStatus.UNPROCESSED
    identifier: str = field(default_factory=lambda: uuid4().hex)
    client_order_id: str | None = None
    filled_quantity: Decimal = Decimal(0)
    avg_fill_price: Decimal | None = None
    transactions: list[Transaction] = field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    error_message: str | None = None
    raw: Any = None

    def __post_init__(self) -> None:
        if (self.quantity is None) == (self.notional is None):
            raise OrderValidationError(
                "Exactly one of `quantity` or `notional` must be set on an Order."
            )

        for field_name in _DECIMAL_FIELDS:
            setattr(self, field_name, _to_decimal(getattr(self, field_name)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Order):
            return NotImplemented
        return self.identifier == other.identifier

    def __hash__(self) -> int:
        return hash(self.identifier)

    def set_identifier(self, identifier: str | UUID) -> None:
        self.identifier = str(identifier)

    def set_error(self, error: BaseException | str) -> None:
        self.status = OrderStatus.ERROR
        self.error_message = str(error)

    def update_raw(self, raw: Any) -> None:
        self.raw = raw

    def add_transaction(
        self,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        price = _to_decimal(price)
        quantity = _to_decimal(quantity)
        self.transactions.append(
            Transaction(
                quantity=quantity,
                price=price,
                timestamp=timestamp if timestamp is not None else datetime.now(),
            )
        )
        self.filled_quantity += quantity

    def is_active(self) -> bool:
        return self.status in ACTIVE_ORDER_STATUSES

    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILL

    def is_canceled(self) -> bool:
        return self.status == OrderStatus.CANCELED

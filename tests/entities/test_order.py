from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderSide, OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.errors import OrderValidationError


def make_order(**overrides: object) -> Order:
    defaults: dict[str, object] = {
        "strategy_name": "momentum",
        "asset": Asset(symbol="AAPL"),
        "side": OrderSide.BUY,
        "quantity": Decimal("1"),
    }
    defaults.update(overrides)
    return Order(**defaults)  # ty: ignore[invalid-argument-type]


def test_fresh_order_gets_a_32_char_hex_identifier_and_differs() -> None:
    order1 = make_order()
    order2 = make_order()
    assert len(order1.identifier) == 32
    assert all(c in "0123456789abcdef" for c in order1.identifier)
    assert order1.identifier != order2.identifier


def test_set_identifier_with_uuid_stores_a_str_and_drives_equality() -> None:
    order = make_order()
    new_id = uuid4()
    order.set_identifier(new_id)
    assert isinstance(order.identifier, str)
    assert order.identifier == str(new_id)


def test_two_orders_sharing_identifier_are_equal_and_collide_in_set() -> None:
    order1 = make_order()
    order2 = make_order()
    shared_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    order1.set_identifier(shared_id)
    order2.set_identifier(shared_id)
    assert order1 == order2
    assert len({order1, order2}) == 1


def test_both_quantity_and_notional_set_raises() -> None:
    with pytest.raises(OrderValidationError):
        make_order(quantity=Decimal("1"), notional=Decimal("100"))


def test_neither_quantity_nor_notional_set_raises() -> None:
    with pytest.raises(OrderValidationError):
        make_order(quantity=None, notional=None)


def test_quantity_coercion_from_str_and_float_are_exact() -> None:
    order_from_str = make_order(quantity="0.001")
    order_from_float = make_order(quantity=0.001)
    assert order_from_str.quantity == Decimal("0.001")
    assert order_from_float.quantity == Decimal("0.001")


def test_add_transaction_appends_and_bumps_filled_quantity() -> None:
    order = make_order(quantity=Decimal("10"))
    order.add_transaction(price=Decimal("100"), quantity=Decimal("3"))
    order.add_transaction(price=Decimal("101"), quantity=Decimal("2"))
    assert len(order.transactions) == 2
    assert order.filled_quantity == Decimal("5")


def test_set_error_sets_status_and_message() -> None:
    order = make_order()
    order.set_error(ValueError("boom"))
    assert order.status == OrderStatus.ERROR
    assert order.error_message == "boom"


def test_avg_fill_price_regression_no_rounding() -> None:
    order = make_order(avg_fill_price=Decimal("101.123456"))
    assert order.avg_fill_price == Decimal("101.123456")


def test_add_transaction_default_timestamp_and_explicit_timestamp() -> None:
    order = make_order(quantity=Decimal("5"))
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    order.add_transaction(price=Decimal("10"), quantity=Decimal("1"), timestamp=ts)
    assert order.transactions[0].timestamp == ts


def test_is_active_is_filled_is_canceled() -> None:
    order = make_order()
    assert order.is_active() is True
    assert order.is_filled() is False
    assert order.is_canceled() is False

    order.status = OrderStatus.FILL
    assert order.is_active() is False
    assert order.is_filled() is True

    order.status = OrderStatus.CANCELED
    assert order.is_canceled() is True


def test_update_raw_sets_raw() -> None:
    order = make_order()
    order.update_raw({"foo": "bar"})
    assert order.raw == {"foo": "bar"}


def test_set_error_with_string_message() -> None:
    order = make_order()
    order.set_error("boom")
    assert order.status == OrderStatus.ERROR
    assert order.error_message == "boom"


def test_set_identifier_accepts_str() -> None:
    order = make_order()
    order.set_identifier("some-broker-id")
    assert order.identifier == "some-broker-id"


def test_transaction_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    from trading_agent_framework.entities.order import Transaction

    txn = Transaction(quantity=Decimal("1"), price=Decimal("1"), timestamp=datetime.now(UTC))
    with pytest.raises(FrozenInstanceError):
        txn.quantity = Decimal("2")  # ty: ignore[invalid-assignment]

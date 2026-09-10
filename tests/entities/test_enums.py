from __future__ import annotations

from trading_agent_framework.entities.enums import (
    ACTIVE_ORDER_STATUSES,
    OrderStatus,
    OrderType,
    is_equivalent_status,
)


def test_active_order_statuses_is_exactly_those_six_members() -> None:
    assert ACTIVE_ORDER_STATUSES == frozenset(
        {
            OrderStatus.UNPROCESSED,
            OrderStatus.SUBMITTED,
            OrderStatus.OPEN,
            OrderStatus.NEW,
            OrderStatus.CANCELLING,
            OrderStatus.PARTIAL_FILL,
        }
    )


def test_is_equivalent_status_true_cases() -> None:
    assert is_equivalent_status(OrderStatus.OPEN, OrderStatus.NEW) is True
    assert is_equivalent_status(OrderStatus.NEW, OrderStatus.SUBMITTED) is True
    assert is_equivalent_status(OrderStatus.SUBMITTED, OrderStatus.OPEN) is True
    assert is_equivalent_status(OrderStatus.FILL, OrderStatus.FILL) is True


def test_is_equivalent_status_false_case() -> None:
    assert is_equivalent_status(OrderStatus.OPEN, OrderStatus.FILL) is False


def test_order_type_trail_wire_value_is_pinned() -> None:
    assert OrderType.TRAIL == "trailing_stop"

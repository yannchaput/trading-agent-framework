from __future__ import annotations

import logging

import alpaca.trading.enums as alpaca_enums
import pytest

from trading_agent_framework.brokers.alpaca.orders import (
    ALPACA_STATUS_MAP,
    map_event,
    map_status,
)
from trading_agent_framework.entities.enums import OrderEvent, OrderStatus

# Task 7: status and event maps -----------------------------------------------

EXPECTED_STATUS_BY_ALPACA_VALUE: dict[str, OrderStatus] = {
    "new": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIAL_FILL,
    "filled": OrderStatus.FILL,
    "done_for_day": OrderStatus.CANCELED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.CANCELED,
    "pending_replace": OrderStatus.CANCELED,
    "pending_review": OrderStatus.OPEN,
    "accepted": OrderStatus.OPEN,
    "pending_new": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.OPEN,
    "stopped": OrderStatus.CANCELED,
    "rejected": OrderStatus.ERROR,
    "suspended": OrderStatus.CANCELED,
    "calculated": OrderStatus.OPEN,
    "held": OrderStatus.OPEN,
}

EXPECTED_EVENT_BY_ALPACA_VALUE: dict[str, OrderEvent | None] = {
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


# Test 32
@pytest.mark.parametrize("member", list(alpaca_enums.OrderStatus))
def test_map_status_covers_every_alpaca_order_status(member: alpaca_enums.OrderStatus) -> None:
    expected = EXPECTED_STATUS_BY_ALPACA_VALUE[member.value]
    assert map_status(member) == expected


# Test 33
def test_map_status_aliases() -> None:
    assert map_status("cancelled") == OrderStatus.CANCELED
    assert map_status("cancel") == OrderStatus.CANCELED
    assert map_status("presubmitted") == OrderStatus.NEW


# Test 34
def test_map_status_unknown_returns_unknown_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = map_status("wat")
    assert result == OrderStatus.UNKNOWN
    assert any(record.levelname == "WARNING" for record in caplog.records)


# Test 35
def test_map_status_accepts_alpaca_enum_member_directly() -> None:
    assert map_status(alpaca_enums.OrderStatus.FILLED) == OrderStatus.FILL


# Test 36: upgrade tripwire
def test_alpaca_status_upgrade_tripwire() -> None:
    assert {s.value for s in alpaca_enums.OrderStatus} <= set(ALPACA_STATUS_MAP)


# Test 37
@pytest.mark.parametrize("member", list(alpaca_enums.TradeEvent))
def test_map_event_covers_every_alpaca_trade_event(member: alpaca_enums.TradeEvent) -> None:
    expected = EXPECTED_EVENT_BY_ALPACA_VALUE[member.value]
    assert map_event(member) == expected

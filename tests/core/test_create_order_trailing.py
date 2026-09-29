from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from tests.fakes import FakeBroker, FakeClock, et

from trading_agent_framework.core import Strategy
from trading_agent_framework.entities.enums import OrderType
from trading_agent_framework.utils.errors import OrderValidationError


def _strategy(tmp_path: Path) -> Strategy:
    return Strategy(FakeBroker(FakeClock(et(2026, 9, 1, 10))), project_root=tmp_path)


def test_create_order_with_a_trail_price_builds_a_trailing_stop(tmp_path: Path) -> None:
    order = _strategy(tmp_path).create_order("AAPL", 10, "sell", trail_price=1.5)
    assert order.order_type is OrderType.TRAIL
    assert order.trail_price == Decimal("1.5")
    assert order.stop_price is None and order.limit_price is None


def test_create_order_with_a_trail_percent(tmp_path: Path) -> None:
    order = _strategy(tmp_path).create_order("AAPL", 10, "sell", trail_percent=2)
    assert order.order_type is OrderType.TRAIL
    assert order.trail_percent == Decimal("2")


def test_create_order_refuses_a_trail_with_a_stop_price(tmp_path: Path) -> None:
    with pytest.raises(OrderValidationError, match="trailing stop"):
        _strategy(tmp_path).create_order("AAPL", 10, "sell", stop_price=99, trail_price=1)

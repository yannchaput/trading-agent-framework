"""Shared hand-written test doubles for broker-layer tests.

Hand-written over `MagicMock` throughout: several tests assert on *which
request object was built* (e.g. the exact `GetOrdersRequest` passed to
`get_orders`), and a fake makes each such test a couple of lines instead of
several lines of mock configuration.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import alpaca.trading.enums as alpaca_enums
import alpaca.trading.models as alpaca_models
from alpaca.common.exceptions import APIError

_NOW = datetime(2024, 1, 1, tzinfo=UTC)


def make_alpaca_order(**overrides: object) -> alpaca_models.Order:
    """Build a real `alpaca.trading.models.Order` with sensible defaults for
    every required field, merged with `**overrides`."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "client_order_id": "cid-1",
        "created_at": _NOW,
        "updated_at": _NOW,
        "submitted_at": _NOW,
        "symbol": "AAPL",
        "asset_class": alpaca_enums.AssetClass.US_EQUITY,
        "qty": "10",
        "notional": None,
        "filled_qty": "0",
        "filled_avg_price": None,
        "type": alpaca_enums.OrderType.MARKET,
        "side": alpaca_enums.OrderSide.BUY,
        "time_in_force": alpaca_enums.TimeInForce.DAY,
        "limit_price": None,
        "stop_price": None,
        "status": alpaca_enums.OrderStatus.NEW,
        "extended_hours": False,
    }
    defaults.update(overrides)
    return alpaca_models.Order(**defaults)  # type: ignore[arg-type]


def make_alpaca_position(**overrides: object) -> alpaca_models.Position:
    """Build a real `alpaca.trading.models.Position` with sensible defaults for
    every required field, merged with `**overrides`."""
    defaults: dict[str, object] = {
        "asset_id": uuid4(),
        "symbol": "AAPL",
        "exchange": alpaca_enums.AssetExchange.NASDAQ,
        "asset_class": alpaca_enums.AssetClass.US_EQUITY,
        "avg_entry_price": "100.00",
        "qty": "10",
        "side": alpaca_enums.PositionSide.LONG,
        "market_value": "1010.00",
        "cost_basis": "1000.00",
        "unrealized_pl": "10.00",
        "current_price": "101.00",
    }
    defaults.update(overrides)
    return alpaca_models.Position(**defaults)  # type: ignore[arg-type]


def make_api_error(status_code: int | None) -> APIError:
    """Build a real `alpaca.common.exceptions.APIError` reporting `status_code`.

    `APIError.status_code` reads `self._http_error.response.status_code`; when
    `status_code` is None here, `http_error` is left None entirely so the
    property's own `is not None` guard kicks in, matching a connection-level
    failure with no underlying HTTP response.
    """
    http_error = None
    if status_code is not None:
        http_error = SimpleNamespace(response=SimpleNamespace(status_code=status_code))
    return APIError("simulated error", http_error=http_error)


class FakeTradingClient:
    """A hand-written stand-in for `alpaca.trading.client.TradingClient`.

    Records every `submit_order` call and returns canned responses for the
    other methods `AlpacaBroker` calls. Not a general-purpose alpaca client
    fake -- only what `AlpacaBroker`'s tests need.
    """

    def __init__(self) -> None:
        self.submitted: list[object] = []
        self.raise_on_submit: BaseException | type[BaseException] | None = None
        self.submit_response: object = None

        self.canceled: list[str] = []

        self.orders_response: list[object] = []
        self.last_orders_request: object = None

        self.order_by_id_response: object = None
        self.get_order_by_id_raises: BaseException | type[BaseException] | None = None
        self.get_order_by_id_calls: list[str] = []

        self.positions_response: list[object] = []

    def submit_order(self, order_data: object) -> object:
        self.submitted.append(order_data)
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        return self.submit_response

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled.append(order_id)

    def get_orders(self, filter: object = None) -> list[object]:
        self.last_orders_request = filter
        return self.orders_response

    def get_order_by_id(self, order_id: str) -> object:
        self.get_order_by_id_calls.append(order_id)
        if self.get_order_by_id_raises is not None:
            raise self.get_order_by_id_raises
        return self.order_by_id_response

    def get_all_positions(self) -> list[object]:
        return self.positions_response

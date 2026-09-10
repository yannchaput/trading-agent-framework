"""Shared hand-written test doubles for broker-layer tests.

Hand-written over `MagicMock` throughout: several tests assert on *which
request object was built* (e.g. the exact `GetOrdersRequest` passed to
`get_orders`), and a fake makes each such test a couple of lines instead of
several lines of mock configuration.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, time, timedelta
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import alpaca.trading.enums as alpaca_enums
import alpaca.trading.models as alpaca_models
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import OrderRequest

from trading_agent_framework.clock import MarketClock, MarketSession

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
    return alpaca_models.Order(**defaults)


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
    return alpaca_models.Position(**defaults)  # ty: ignore[invalid-argument-type]


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

    Response fields are typed as the real alpaca models (never `None` at the
    point a test's code path actually reaches them) so this class structurally
    satisfies `orders.AlpacaTradingClient` -- every test sets the relevant
    `*_response` field before triggering the call that returns it; the asserts
    below turn a forgotten setup into a clear failure instead of a `None`
    silently flowing downstream.
    """

    def __init__(self) -> None:
        self.submitted: list[OrderRequest] = []
        self.raise_on_submit: BaseException | type[BaseException] | None = None
        self.submit_response: alpaca_models.Order | None = None

        self.canceled: list[str] = []

        self.orders_response: list[alpaca_models.Order] = []
        self.last_orders_request: object = None

        self.order_by_id_response: alpaca_models.Order | None = None
        self.get_order_by_id_raises: BaseException | type[BaseException] | None = None
        self.get_order_by_id_calls: list[str] = []

        self.positions_response: list[alpaca_models.Position] = []

    def submit_order(self, order_data: OrderRequest) -> alpaca_models.Order:
        self.submitted.append(order_data)
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        assert self.submit_response is not None, "test must set client.submit_response"
        return self.submit_response

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled.append(order_id)

    def get_orders(self, filter: object = None) -> list[alpaca_models.Order]:
        self.last_orders_request = filter
        return self.orders_response

    def get_order_by_id(self, order_id: str) -> alpaca_models.Order:
        self.get_order_by_id_calls.append(order_id)
        if self.get_order_by_id_raises is not None:
            raise self.get_order_by_id_raises
        assert self.order_by_id_response is not None, "test must set client.order_by_id_response"
        return self.order_by_id_response

    def get_all_positions(self) -> list[alpaca_models.Position]:
        return self.positions_response


ET = ZoneInfo("America/New_York")


def et(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """A tz-aware datetime in market (Eastern) time."""
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def make_session(
    day: date, open_at: time = time(9, 30), close_at: time = time(16, 0)
) -> MarketSession:
    return MarketSession(
        open=datetime.combine(day, open_at, tzinfo=ET),
        close=datetime.combine(day, close_at, tzinfo=ET),
    )


def weekday_sessions(first_day: date, count: int) -> list[MarketSession]:
    """`count` regular 09:30-16:00 sessions on consecutive weekdays from
    `first_day`."""
    sessions: list[MarketSession] = []
    day = first_day
    while len(sessions) < count:
        if day.weekday() < 5:
            sessions.append(make_session(day))
        day += timedelta(days=1)
    return sessions


class FakeClock(MarketClock):
    """Manual `MarketClock`: `wait` advances fake time instantly unless `wake`
    is set.

    `on_wait` runs at the start of every `wait` call -- tests use it to inject
    order events or call `stop()` from "outside" the executor loop.
    `next_session_errors` are raised (in order) by `next_session` before any
    session is returned.
    """

    def __init__(self, now: datetime, sessions: Iterable[MarketSession] = ()) -> None:
        self._now = now
        self.sessions = list(sessions)
        self.waits: list[float] = []
        self.on_wait: Callable[[], None] | None = None
        self.next_session_errors: list[BaseException] = []

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def wait(self, seconds: float, wake: threading.Event) -> None:
        self.waits.append(seconds)
        if self.on_wait is not None:
            self.on_wait()
        if not wake.is_set():
            self.advance(seconds)

    def next_session(self) -> MarketSession | None:
        if self.next_session_errors:
            raise self.next_session_errors.pop(0)
        return next((s for s in self.sessions if s.close > self._now), None)

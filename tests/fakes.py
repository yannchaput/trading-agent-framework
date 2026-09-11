"""Shared hand-written test doubles for broker-layer tests.

Hand-written over `MagicMock` throughout: several tests assert on *which
request object was built* (e.g. the exact `GetOrdersRequest` passed to
`get_orders`), and a fake makes each such test a couple of lines instead of
several lines of mock configuration.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import ClassVar
from uuid import uuid4
from zoneinfo import ZoneInfo

import alpaca.data.models as alpaca_data_models
import alpaca.trading.enums as alpaca_enums
import alpaca.trading.models as alpaca_models
from alpaca.common.exceptions import APIError
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetCalendarRequest,
    OrderRequest,
    ReplaceOrderRequest,
)

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.clock import MarketClock, MarketSession
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.enums import OrderStatus
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position

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


def make_alpaca_account(**overrides: object) -> alpaca_models.TradeAccount:
    """Build a real `alpaca.trading.models.TradeAccount` (string money fields, as Alpaca sends)."""
    defaults: dict[str, object] = {
        "id": uuid4(),
        "account_number": "PA0000000",
        "status": alpaca_enums.AccountStatus.ACTIVE,
        "cash": "10000.50",
        "equity": "25000.25",
        "portfolio_value": "25000.25",
        "buying_power": "20000",
    }
    defaults.update(overrides)
    return alpaca_models.TradeAccount(**defaults)  # ty: ignore[invalid-argument-type]


def make_alpaca_calendar(
    day: str, open_at: str = "09:30", close_at: str = "16:00"
) -> alpaca_models.Calendar:
    """Build a real `Calendar` the way the SDK does from the API payload (naive times)."""
    return alpaca_models.Calendar(date=day, open=open_at, close=close_at)


def make_failed_close_details(symbol: str = "AAPL") -> alpaca_models.FailedClosePositionDetails:
    return alpaca_models.FailedClosePositionDetails(
        code=40310000, message="insufficient qty available for order", symbol=symbol
    )


def make_close_position_response(
    body: alpaca_models.Order | alpaca_models.FailedClosePositionDetails, symbol: str = "AAPL"
) -> alpaca_models.ClosePositionResponse:
    is_order = isinstance(body, alpaca_models.Order)
    return alpaca_models.ClosePositionResponse(
        order_id=body.id if isinstance(body, alpaca_models.Order) else None,
        status=200 if is_order else 403,
        symbol=symbol,
        body=body,
    )


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


def bar_payload(timestamp: str, close: float, volume: float = 1000.0) -> dict[str, object]:
    """One bar as Alpaca's API sends it; `BarSet` is built from these raw payloads."""
    return {
        "t": timestamp,
        "o": close - 0.5,
        "h": close + 1.0,
        "l": close - 1.0,
        "c": close,
        "v": volume,
        "n": 10,
        "vw": close,
    }


def make_alpaca_barset(bars: dict[str, list[dict[str, object]]]) -> alpaca_data_models.BarSet:
    return alpaca_data_models.BarSet(bars)


def make_alpaca_trade(
    symbol: str = "AAPL", price: float = 100.15, timestamp: str = "2026-09-10T13:30:00Z"
) -> alpaca_data_models.Trade:
    payload = {"t": timestamp, "x": "V", "p": price, "s": 50, "i": 1, "c": ["@"], "z": "C"}
    return alpaca_data_models.Trade(symbol, payload)


def make_alpaca_quote(
    symbol: str = "AAPL",
    bid: float = 100.1,
    ask: float = 100.2,
    bid_size: float = 3.0,
    ask_size: float = 4.0,
    timestamp: str = "2026-09-10T13:30:00Z",
) -> alpaca_data_models.Quote:
    payload = {
        "t": timestamp,
        "bp": bid,
        "bs": bid_size,
        "bx": "V",
        "ap": ask,
        "as": ask_size,
        "ax": "V",
        "c": ["R"],
        "z": "C",
    }
    return alpaca_data_models.Quote(symbol, payload)


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

        self.raises: dict[str, BaseException] = {}

        self.account_response: alpaca_models.TradeAccount | None = None

        self.calendar_response: list[alpaca_models.Calendar] = []
        self.calendar_requests: list[GetCalendarRequest] = []

        self.replace_calls: list[tuple[str, ReplaceOrderRequest]] = []
        self.replace_response: alpaca_models.Order | None = None

        self.close_position_calls: list[tuple[str, ClosePositionRequest]] = []
        self.close_position_response: alpaca_models.Order | None = None

        self.close_all_calls: list[bool] = []
        self.close_all_response: list[alpaca_models.ClosePositionResponse] = []

    def submit_order(self, order_data: OrderRequest) -> alpaca_models.Order:
        self.submitted.append(order_data)
        if self.raise_on_submit is not None:
            raise self.raise_on_submit
        assert self.submit_response is not None, "test must set client.submit_response"
        return self.submit_response

    def cancel_order_by_id(self, order_id: str) -> None:
        self.canceled.append(order_id)

    def get_orders(self, filter: object = None) -> list[alpaca_models.Order]:
        self._maybe_raise("get_orders")
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

    def _maybe_raise(self, method: str) -> None:
        error = self.raises.get(method)
        if error is not None:
            raise error

    def get_account(self) -> alpaca_models.TradeAccount:
        self._maybe_raise("get_account")
        assert self.account_response is not None, "test must set client.account_response"
        return self.account_response

    def get_calendar(self, filters: GetCalendarRequest) -> list[alpaca_models.Calendar]:
        self.calendar_requests.append(filters)
        self._maybe_raise("get_calendar")
        return self.calendar_response

    def replace_order_by_id(
        self, order_id: str, order_data: ReplaceOrderRequest
    ) -> alpaca_models.Order:
        self.replace_calls.append((order_id, order_data))
        self._maybe_raise("replace_order_by_id")
        assert self.replace_response is not None, "test must set client.replace_response"
        return self.replace_response

    def close_position(
        self, symbol_or_asset_id: str, close_options: ClosePositionRequest
    ) -> alpaca_models.Order:
        self.close_position_calls.append((symbol_or_asset_id, close_options))
        self._maybe_raise("close_position")
        assert self.close_position_response is not None, "test must set close_position_response"
        return self.close_position_response

    def close_all_positions(self, cancel_orders: bool) -> list[alpaca_models.ClosePositionResponse]:
        self.close_all_calls.append(cancel_orders)
        self._maybe_raise("close_all_positions")
        return self.close_all_response


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


class FakeBroker(Broker):
    """In-memory `Broker` for strategy/executor tests: records every call, no I/O.

    Submitted and synced orders go through the real `OrderTracker`, so tests can
    drive fills with `broker.tracker.process_trade_event(...)`.
    """

    name: ClassVar[str] = "fake"

    def __init__(
        self, clock: MarketClock, strategy_name: str = "momentum", *, is_paper: bool = True
    ) -> None:
        super().__init__(strategy_name, clock=clock, is_paper=is_paper)
        self.account = AccountBalances(
            cash=Decimal("10000"), portfolio_value=Decimal("25000"), buying_power=Decimal("20000")
        )
        self.positions: list[Position] = []
        self.remote_orders: dict[str, Order] = {}
        self.orders_to_sync: list[Order] = []
        self.submitted: list[Order] = []
        self.canceled: list[Order] = []
        self.modified: list[tuple[Order, Decimal | None, Decimal | None]] = []
        self.closed: list[tuple[Asset, Decimal]] = []
        self.close_all_calls: list[bool] = []
        self.calls: list[str] = []

    def _conform_order(self, order: Order) -> Order:
        return order

    def _submit_order(self, order: Order) -> Order:
        self.submitted.append(order)
        self.tracker.track_unprocessed(order)
        return order

    def cancel_order(self, order: Order) -> None:
        self.canceled.append(order)

    def pull_order(self, identifier: str) -> Order | None:
        return self.remote_orders.get(identifier)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        return list(self.remote_orders.values())[:limit]

    def pull_positions(self) -> list[Position]:
        return list(self.positions)

    def get_account(self) -> AccountBalances:
        return self.account

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        self.modified.append((order, limit_price, stop_price))
        replacement = dataclasses.replace(
            order,
            identifier=uuid4().hex,
            limit_price=limit_price if limit_price is not None else order.limit_price,
            stop_price=stop_price if stop_price is not None else order.stop_price,
            status=OrderStatus.UNPROCESSED,
            transactions=[],
            filled_quantity=Decimal(0),
        )
        self.tracker.mark_replaced(order, replacement)
        return replacement

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        self.closed.append((asset, fraction))
        return None

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        self.close_all_calls.append(cancel_orders)
        return []

    def sync_open_orders(self) -> list[Order]:
        self.calls.append("sync_open_orders")
        for order in self.orders_to_sync:
            self.tracker.track_unprocessed(order)
        return list(self.orders_to_sync)

    def start_stream(self) -> None:
        self.calls.append("start_stream")

    def stop_stream(self, timeout: float = 5.0) -> None:
        self.calls.append("stop_stream")

"""Concrete `Broker` implementation wiring the Alpaca trading client, order
translation helpers (`orders.py`), and `OrderTracker` into real I/O.

`AlpacaBroker` itself performs no translation logic: it delegates every bit of
that to the pure functions in `orders.py` (global constraint 4) and only adds
the I/O calls against a `TradingClient` plus the bookkeeping (`tracker`,
`client_order_id` stamping) that a broker-agnostic `orders.py` cannot own.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from alpaca.common.exceptions import APIError
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trading_agent_framework.brokers.alpaca import orders
from trading_agent_framework.brokers.alpaca.client import build_trading_client, build_trading_stream
from trading_agent_framework.brokers.alpaca.stream import AlpacaTradeStream
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.stream import TradingStream

    from trading_agent_framework.config.env import AlpacaCredentials

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """`Broker` implementation backed by Alpaca's `TradingClient`."""

    name: ClassVar[str] = "alpaca"

    def __init__(
        self,
        strategy_name: str,
        client: TradingClient,
        tracker: OrderTracker | None = None,
        stream: TradingStream | None = None,
    ) -> None:
        super().__init__(strategy_name, tracker)
        self._client = client
        self._stream = stream
        self._alpaca_stream: AlpacaTradeStream | None = None

    @classmethod
    def from_credentials(
        cls,
        strategy_name: str,
        creds: AlpacaCredentials,
        with_stream: bool = True,
    ) -> AlpacaBroker:
        client = build_trading_client(creds)
        stream = build_trading_stream(creds) if with_stream else None
        return cls(strategy_name, client, stream=stream)

    def _conform_order(self, order: Order) -> Order:
        return orders.conform_order(order)

    def _submit_order(self, order: Order) -> Order:
        if not order.client_order_id:
            order.client_order_id = f"{self.strategy_name}:{order.identifier}"

        orders.validate_order(order)
        request = orders.build_order_request(order)
        try:
            response = self._client.submit_order(order_data=request)
        except Exception as exc:
            order.set_error(exc)
            logger.exception("Failed to submit order %s", order.identifier)
            raise  # set_error BEFORE re-raising -- lumibot's contract
        order.set_identifier(response.id)
        order.status = orders.map_status(response.status)
        order.update_raw(response)
        self.tracker.track_unprocessed(order)
        return order

    def cancel_order(self, order: Order) -> None:
        self._client.cancel_order_by_id(order.identifier)

    def pull_order(self, identifier: str) -> Order | None:
        try:
            response = self._client.get_order_by_id(identifier)
        except APIError as exc:
            # status_code is None for a connection-level failure with no
            # underlying HTTP response -- that's not a 404, so it re-raises too.
            if exc.status_code == 404:
                return None
            raise
        return orders.parse_broker_order(response, self.strategy_name)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        responses = self._client.get_orders(filter=request)
        return orders.parse_broker_orders(responses, self.strategy_name)

    def pull_positions(self) -> list[Position]:
        return [
            orders.parse_broker_position(p, self.strategy_name)
            for p in self._client.get_all_positions()
        ]

    def _ensure_alpaca_stream(self) -> AlpacaTradeStream:
        if self._alpaca_stream is None:
            self._alpaca_stream = AlpacaTradeStream(self.tracker, self._stream)
        return self._alpaca_stream

    def start_stream(self) -> None:
        self._ensure_alpaca_stream().start()

    def stop_stream(self, timeout: float = 5.0) -> None:
        self._ensure_alpaca_stream().stop(timeout)

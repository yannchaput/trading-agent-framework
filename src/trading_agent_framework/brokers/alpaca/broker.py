"""Concrete `Broker` implementation wiring the Alpaca trading client, order
translation helpers (`orders.py`), and `OrderTracker` into real I/O.

`AlpacaBroker` itself performs no translation logic: it delegates every bit of
that to the pure functions in `orders.py` (global constraint 4) and only adds
the I/O calls against a `TradingClient` plus the bookkeeping (`tracker`,
`client_order_id` stamping) that a broker-agnostic `orders.py` cannot own.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar, cast

from alpaca.common.exceptions import APIError

from trading_agent_framework.brokers.alpaca import account, orders
from trading_agent_framework.brokers.alpaca.client import (
    build_trading_client,
    build_trading_stream,
)
from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.alpaca.news import AlpacaNewsProvider, lazy_news_provider
from trading_agent_framework.brokers.alpaca.stream import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    AlpacaTradeStream,
)
from trading_agent_framework.brokers.alpaca.symbols import to_alpaca_symbol
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError

if TYPE_CHECKING:
    from alpaca.trading.stream import TradingStream

    from trading_agent_framework.brokers.alpaca import market_data
    from trading_agent_framework.config.env import AlpacaCredentials

logger = logging.getLogger(__name__)


class AlpacaBroker(Broker):
    """`Broker` implementation backed by Alpaca's `TradingClient`."""

    name: ClassVar[str] = "alpaca"

    _SYNC_LIMIT = 500  # Alpaca's maximum page size for GET /orders

    def __init__(
        self,
        strategy_name: str,
        client: orders.AlpacaTradingClient,
        tracker: OrderTracker | None = None,
        stream: TradingStream | None = None,
        *,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        data_client: market_data.AlpacaStockDataClient | None = None,
        news_client: market_data.AlpacaNewsClient | None = None,
        market_data: AlpacaMarketData | None = None,
        news_provider_factory: Callable[[], NewsProvider] | None = None,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(client),
            is_paper=is_paper,
        )
        self._client = client
        self._market_data = market_data if market_data is not None else AlpacaMarketData(data_client, client)
        self._news_provider: NewsProvider | None = AlpacaNewsProvider(news_client) if news_client is not None else None
        self._news_provider_factory = news_provider_factory
        self._stream = stream
        self._alpaca_stream: AlpacaTradeStream | None = None

    @classmethod
    def from_credentials(
        cls,
        strategy_name: str,
        *,
        trading: AlpacaCredentials,
        data: AlpacaCredentials,
        news: Callable[[], AlpacaCredentials] | None = None,
        with_stream: bool = True,
    ) -> AlpacaBroker:
        # TradingClient's own signatures declare `T | RawData` (a dict) because
        # the SDK supports a raw_data mode; this project never enables it, so
        # every call actually returns the parsed model. Narrow once, here, at
        # the single point a real client is constructed.
        client = cast("orders.AlpacaTradingClient", build_trading_client(trading))
        stream = build_trading_stream(trading) if with_stream else None
        broker = cls(
            strategy_name,
            client,
            stream=stream,
            is_paper=trading.is_paper,
            market_data=AlpacaMarketData.from_credentials(data),
            news_provider_factory=lazy_news_provider(news) if news is not None else None,
        )
        broker.configure_account()
        return broker

    def _conform_order(self, order: Order) -> Order:
        return orders.conform_order(order)

    def _submit_order(self, order: Order) -> Order:
        if not order.client_order_id:
            order.client_order_id = f"{self.strategy_name}:{order.identifier}"

        orders.validate_order(order)
        request = orders.build_order_request(order)
        # Tracked (by client_order_id) before the call goes out -- the trade stream
        # can report this order's "new" event before submit_order() returns, and it
        # needs to find the order already known or that event is silently dropped.
        self.tracker.track_unprocessed(order)
        try:
            response = self._client.submit_order(order_data=request)
        except Exception as exc:
            order.set_error(exc)
            logger.exception("Failed to submit order %s", order.identifier)
            self.tracker.untrack(order)
            raise  # set_error BEFORE re-raising -- lumibot's contract
        order.set_identifier(response.id)
        order.status = orders.map_status(response.status)
        order.update_raw(response)
        return order

    def cancel_order(self, order: Order) -> None:
        try:
            self._client.cancel_order_by_id(order.identifier)
        except Exception as exc:
            raise BrokerError(f"Failed to cancel order {order.identifier}: {exc}") from exc

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
        request = orders.build_get_orders_request(limit)
        try:
            responses = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch orders: {exc}") from exc
        return orders.parse_broker_orders(responses, self.strategy_name)

    def pull_positions(self) -> list[Position]:
        try:
            responses = self._client.get_all_positions()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch positions: {exc}") from exc
        return [orders.parse_broker_position(p, self.strategy_name) for p in responses]

    def get_account(self) -> AccountBalances:
        try:
            response = self._client.get_account()
        except Exception as exc:
            raise BrokerError(f"Failed to fetch the Alpaca account: {exc}") from exc
        return account.parse_account(response)

    def configure_account(
        self,
        *,
        no_shorting: bool = True,
        max_margin_multiplier: str = "1",
        fractional_trading: bool = True,
    ) -> None:
        """Restrict the Alpaca account: no shorting, no margin (by default), fractional shares."""
        try:
            configuration = self._client.get_account_configurations()
            account.apply_account_restrictions(
                configuration,
                no_shorting=no_shorting,
                max_margin_multiplier=max_margin_multiplier,
                fractional_trading=fractional_trading,
            )
            updated = self._client.set_account_configurations(configuration)
        except Exception as exc:
            raise BrokerError(f"Failed to configure the Alpaca account: {exc}") from exc
        logger.info("Account configuration: %s", updated.model_dump_json())

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        request = orders.build_replace_order_request(limit_price=limit_price, stop_price=stop_price)
        try:
            response = self._client.replace_order_by_id(order.identifier, order_data=request)
        except Exception as exc:
            raise BrokerError(f"Failed to modify order {order.identifier}: {exc}") from exc
        replacement = orders.parse_broker_order(response, self.strategy_name)
        if replacement is None:
            raise BrokerError(f"Alpaca returned no usable replacement for order {order.identifier}")
        self.tracker.mark_replaced(order, replacement)
        return replacement

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        request = orders.build_close_position_request(fraction)
        try:
            response = self._client.close_position(to_alpaca_symbol(asset.symbol), close_options=request)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise BrokerError(f"Failed to close position {asset.symbol}: {exc}") from exc
        except Exception as exc:
            raise BrokerError(f"Failed to close position {asset.symbol}: {exc}") from exc
        order = orders.parse_broker_order(response, self.strategy_name)
        if order is not None:
            self.tracker.track_unprocessed(order)
        return order

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        try:
            responses = self._client.close_all_positions(cancel_orders=cancel_orders)
        except Exception as exc:
            raise BrokerError(f"Failed to close all positions: {exc}") from exc
        closed = orders.parse_close_all_responses(responses, self.strategy_name)
        for order in closed:
            self.tracker.track_unprocessed(order)
        return closed

    def sync_open_orders(self) -> list[Order]:
        request = orders.build_get_orders_request(self._SYNC_LIMIT, open_only=True)
        try:
            responses = self._client.get_orders(filter=request)
        except Exception as exc:
            raise BrokerError(f"Failed to fetch open orders: {exc}") from exc
        prefix = f"{self.strategy_name}:"
        adopted: list[Order] = []
        for order in orders.parse_broker_orders(responses, self.strategy_name):
            if not (order.client_order_id or "").startswith(prefix):
                continue
            if self.tracker.get_tracked_order(order.identifier) is not None:
                continue
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted

    # --- market data -------------------------------------------------------------------

    def news_provider(self) -> NewsProvider | None:
        if self._news_provider is None and self._news_provider_factory is not None:
            self._news_provider = self._news_provider_factory()
        return self._news_provider

    def get_news(
        self,
        symbols: Sequence[str] = (),
        *,
        start: datetime | None = None,
        end: datetime,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, object]]:
        provider = self.news_provider()
        if provider is None:
            raise BrokerError(
                "no news client configured; construct the broker with news_client=... "
                "or use AlpacaBroker.from_credentials(...)"
            )
        return provider.get_news(symbols, start=start, end=end, limit=limit, include_content=include_content)

    def get_last_price(self, asset: Asset) -> Decimal | None:
        return self._market_data.get_last_price(asset)

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return self._market_data.get_last_prices(assets)

    def get_quote(self, asset: Asset) -> Quote | None:
        return self._market_data.get_quote(asset)

    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self._market_data.get_bars(
            assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours
        )

    def _ensure_alpaca_stream(self) -> AlpacaTradeStream:
        if self._alpaca_stream is None:
            self._alpaca_stream = AlpacaTradeStream(self.tracker, self._stream)
        return self._alpaca_stream

    def start_stream(self, connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS) -> None:
        if self._stream is None:
            raise BrokerError(
                "no TradingStream configured; construct the broker with a stream "
                "or use AlpacaBroker.from_credentials(..., with_stream=True)"
            )
        self._ensure_alpaca_stream().start(connect_timeout=connect_timeout)

    def stop_stream(self, timeout: float = 5.0) -> None:
        self._ensure_alpaca_stream().stop(timeout)

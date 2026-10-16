"""`TastytradeBroker`: Tastytrade (REST API + account streamer) for trading, account and positions;
Alpaca for market data, the calendar and news.

Wiring only: translation lives in the pure `orders`/`account` modules, HTTP in `TastytradeSession`,
the websocket in `TastytradeAccountStream`, and event handling in `TastytradeOrderEvents`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn
from urllib.parse import quote

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.alpaca.news import lazy_news_provider
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tastytrade import account, orders
from trading_agent_framework.brokers.tastytrade.client import TastytradeHTTPError, TastytradeSession
from trading_agent_framework.brokers.tastytrade.events import TastytradeOrderEvents
from trading_agent_framework.brokers.tastytrade.stream import StreamSocket, TastytradeAccountStream
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError, BrokerUncertainError, OrderValidationError

if TYPE_CHECKING:
    from trading_agent_framework.config.env import AlpacaCredentials, TastytradeSettings

logger = logging.getLogger(__name__)

_MODIFIABLE = frozenset({OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT})


class TastytradeBroker(Broker):
    name: ClassVar[str] = "tastytrade"

    def __init__(
        self,
        strategy_name: str,
        session: TastytradeSession,
        *,
        account_number: str,
        streamer_url: str,
        market_data: AlpacaMarketData,
        tracker: OrderTracker | None = None,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        news_provider_factory: Callable[[], NewsProvider] | None = None,
        stream_connect: Callable[[str], StreamSocket] | None = None,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(market_data.calendar_client),
            is_paper=is_paper,
        )
        self._session = session
        self._account_number = account_number
        self._market_data = market_data
        self._news_provider: NewsProvider | None = None
        self._news_provider_factory = news_provider_factory
        self._events = TastytradeOrderEvents(self.tracker)
        self._eligible: dict[str, bool] = {}
        connect: dict[str, Any] = {"connect": stream_connect} if stream_connect is not None else {}
        self._stream = TastytradeAccountStream(streamer_url, account_number, session.access_token, self._events.apply, self.reconcile, **connect)

    @classmethod
    def from_settings(
        cls,
        strategy_name: str,
        settings: TastytradeSettings,
        *,
        data: AlpacaCredentials,
        news: Callable[[], AlpacaCredentials] | None = None,
        session: TastytradeSession | None = None,
        market_data: AlpacaMarketData | None = None,
    ) -> TastytradeBroker:
        """Open the session and check the account; any failure closes the session before raising."""
        market_data = market_data if market_data is not None else AlpacaMarketData.from_credentials(data)
        session = session if session is not None else TastytradeSession(settings)
        try:
            broker = cls(
                strategy_name,
                session,
                account_number=settings.account_number,
                streamer_url=settings.streamer_url,
                market_data=market_data,
                is_paper=settings.is_paper,
                news_provider_factory=lazy_news_provider(news) if news is not None else None,
            )
            broker.configure_account()
        except BaseException:
            session.close()
            raise
        return broker

    # --- account ---------------------------------------------------------------------

    @property
    def account_number(self) -> str:
        return self._account_number

    def _path(self, suffix: str) -> str:
        return f"/accounts/{self._account_number}{suffix}"

    def configure_account(self) -> None:
        """Tastytrade cannot switch margin off via the API: check the account instead."""
        items = self._session.get("/customers/me/accounts").get("items") or []
        for warning in account.check_account(items, self._account_number, is_paper=self.is_paper):
            logger.warning(warning)
        logger.info("Tastytrade account %s checked (sandbox=%s)", self._account_number, self.is_paper)

    def get_account(self) -> AccountBalances:
        return account.parse_balances(self._session.get(self._path("/balances")))

    # --- orders ----------------------------------------------------------------------

    def _fractional_eligible(self, asset: Asset) -> bool:
        symbol = asset.symbol
        if symbol not in self._eligible:
            path = f"/instruments/equities/{quote(orders.to_tastytrade_symbol(symbol), safe='')}"
            try:
                data = self._session.get(path)
            except TastytradeHTTPError as exc:
                if exc.status_code == 404:
                    raise OrderValidationError(f"Tastytrade does not know the US stock {symbol}") from exc
                raise
            self._eligible[symbol] = bool(data.get("is-fractional-quantity-eligible"))
        return self._eligible[symbol]

    def _conform_order(self, order: Order) -> Order:
        eligible = self._fractional_eligible(order.asset)
        last_price = self._market_data.get_last_price(order.asset) if order.notional is not None and not eligible else None
        return orders.conform_order(order, fractional_eligible=eligible, last_price=last_price)

    def _external_identifier(self, order: Order) -> str:
        """Always `{strategy}:{identifier-or-client-id}`: `sync_open_orders` adopts by that prefix after a restart."""
        if not order.client_order_id:
            return orders.external_identifier_for(self.strategy_name, order.identifier)
        if order.client_order_id.startswith(f"{self.strategy_name}:"):
            return order.client_order_id
        return orders.external_identifier_for(self.strategy_name, order.client_order_id)

    @staticmethod
    def _with_id(data: object) -> Mapping[str, Any] | None:
        """`data` when it is an order snapshot carrying an id, else None."""
        return data if isinstance(data, Mapping) and data.get("id") is not None else None

    def _apply(self, data: Mapping[str, Any]) -> bool:
        """`TastytradeOrderEvents.apply`, with a malformed snapshot logged instead of raised."""
        try:
            return self._events.apply(data)
        except BrokerError:
            logger.exception("Applying the Tastytrade order snapshot %s failed", data.get("id"))
            return False

    def _submit_order(self, order: Order) -> Order:
        order.client_order_id = self._external_identifier(order)
        body = orders.build_order_json(order)
        external = order.client_order_id
        self.tracker.track_unprocessed(order)  # before posting: the stream can report the order before the call returns
        placed: Mapping[str, Any] | None = None
        unknown: str | None = None  # why the outcome is unknown; None once Tastytrade's answer is known
        try:
            response = self._session.post(self._path("/orders"), json=body)
        except TastytradeHTTPError as exc:  # a definite rejection (4xx): the order does not exist
            self._fail(order, str(exc), cause=exc)
        except BrokerUncertainError as exc:  # timeout, transport failure, 5xx, unreadable 2xx
            unknown = str(exc)
        except Exception as exc:
            self._fail(order, str(exc), cause=exc)
        else:
            placed = self._with_id(response.get("order"))
            if placed is None:  # a 2xx that names no order: it may well have been accepted
                unknown = "Tastytrade answered without an order id"
        if placed is None:
            found = self._find_order(lambda item: item.get("external-identifier") == external)
            placed = self._with_id(found)
            if placed is None:
                self._fail(order, f"submission outcome unknown ({unknown}); Tastytrade does not show it, so it was not resent -- check the account before resubmitting")
            logger.warning("Order %s: the submission outcome was unknown, but Tastytrade has it (id %s)", order.identifier, placed.get("id"))
        self._events.register(order, str(placed["id"]))
        order.update_raw(dict(placed))
        if order.status is OrderStatus.UNPROCESSED:
            order.status = OrderStatus.SUBMITTED  # from here on, TastytradeOrderEvents owns its status
        self._apply(placed)  # a sandbox $2 limit is already Filled here
        return order

    def _fail(self, order: Order, message: str, *, cause: BaseException | None = None) -> NoReturn:
        order.set_error(message)  # BEFORE raising -- lumibot's contract
        self.tracker.untrack(order)
        raise BrokerError(f"Tastytrade order {order.identifier} failed: {message}") from cause

    def _find_order(self, matches: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any] | None:
        """The first live, then today's, order matching `matches`; None when absent or every lookup failed."""
        today = self.clock.now().date().isoformat()
        for path, params in ((self._path("/orders/live"), None), (self._path("/orders"), {"start-date": today, "sort": "Desc"})):
            try:
                items = self._session.get_all(path, params=params)
            except BrokerError:
                logger.exception("Looking up Tastytrade orders at %s failed", path)
                continue
            found = next((item for item in items if matches(item)), None)
            if found is not None:
                return found
        return None

    def _tt_id(self, order: Order) -> str:
        tt_id = self._events.tt_id_of(order)
        if tt_id is None:
            raise BrokerError(f"no Tastytrade order id for {order.identifier}")
        return tt_id

    def cancel_order(self, order: Order) -> None:
        self._session.delete(self._path(f"/orders/{self._tt_id(order)}"))

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Tastytrade replaces the order (new id, same `external-identifier`); the same `Order` comes back."""
        if order.notional is not None or order.order_type not in _MODIFIABLE:
            raise OrderValidationError(f"only limit, stop and stop-limit orders can be modified (order {order.identifier} is {order.order_type.value})")
        tt_id = self._tt_id(order)
        previous = (order.limit_price, order.stop_price, order.stop_limit_price)
        if limit_price is not None:
            if order.order_type is OrderType.STOP_LIMIT:
                order.stop_limit_price = limit_price
            else:
                order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        body = orders.build_replace_json(order)
        self._events.begin_replace(order)  # the original's Cancelled may arrive before the PUT returns
        unknown: str | None = None
        replacement: Mapping[str, Any] | None = None
        try:
            response = self._session.put(self._path(f"/orders/{tt_id}"), json=body)
        except BrokerUncertainError as exc:
            unknown = str(exc)
        except Exception:
            self._restore(order, tt_id, previous)
            raise
        else:
            replacement = self._with_id(response.get("order")) or self._with_id(response)  # the order itself, or wrapped
            if replacement is None:
                unknown = "Tastytrade answered without an order id"
        if replacement is None:
            external = order.client_order_id
            replacement = self._with_id(self._find_order(lambda item: str(item.get("replaces-order-id")) == tt_id and item.get("external-identifier") == external))
            if replacement is None:
                self._restore(order, tt_id, previous)
                raise BrokerUncertainError(f"modify of Tastytrade order {tt_id} has an unknown outcome ({unknown}) and no replacement was found")
        self._events.finish_replace(order, str(replacement["id"]))
        self._apply(replacement)
        self.tracker.process_trade_event(order, OrderEvent.MODIFIED)
        return order

    def _restore(self, order: Order, tt_id: str, previous: tuple[Decimal | None, Decimal | None, Decimal | None]) -> None:
        """A replace that did not happen: restore the prices and the original id, then re-read the original.

        While it was superseded its statuses were dropped, so it may have been cancelled or filled meanwhile.
        """
        self._events.abort_replace(tt_id)
        order.limit_price, order.stop_price, order.stop_limit_price = previous
        try:
            self._apply(self._session.get(self._path(f"/orders/{tt_id}")))
        except BrokerError:
            logger.exception("Re-reading Tastytrade order %s after a failed replace failed", tt_id)

    def pull_order(self, identifier: str) -> Order | None:
        external = orders.external_identifier_for(self.strategy_name, identifier)
        items = self._session.get_all(self._path("/orders"), params={"sort": "Desc"}, max_items=1000)
        found = next((item for item in items if item.get("external-identifier") == external or str(item.get("id")) == identifier), None)
        return orders.parse_order(found, self.strategy_name) if found is not None and orders.is_parsable_order(found) else None

    def pull_orders(self, limit: int = 100) -> list[Order]:
        """The newest `limit` orders of the account (equity orders this framework models)."""
        items = self._session.get_all(self._path("/orders"), params={"sort": "Desc"}, max_items=limit)
        parsed: list[Order] = []
        for item in items:
            if not orders.is_parsable_order(item):
                continue
            try:
                parsed.append(orders.parse_order(item, self.strategy_name))
            except BrokerError:
                logger.exception("Skipping the malformed Tastytrade order %s", item.get("id"))
        return parsed

    def pull_positions(self) -> list[Position]:
        items = self._session.get(self._path("/positions")).get("items") or []
        parsed = (orders.parse_position(item, self.strategy_name) for item in items)
        return [p for p in parsed if p is not None]

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = next((p for p in self.pull_positions() if p.asset == asset), None)
        if position is None:
            return None
        quantity = position.quantity * fraction
        if not self._fractional_eligible(asset) and quantity.to_integral_value(rounding=ROUND_FLOOR) < 1:
            logger.warning("close_position(%s, %s): below one whole share, nothing sent", asset.symbol, fraction)
            return None
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        return self.submit_order(Order(self.strategy_name, asset, side, OrderType.MARKET, quantity=quantity))

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            for item in self._session.get_all(self._path("/orders/live")):
                if item.get("cancellable"):
                    try:
                        self._session.delete(self._path(f"/orders/{item['id']}"))
                    except BrokerError:
                        logger.exception("Cancelling Tastytrade order %s failed", item.get("id"))
        closed = (self.close_position(p.asset) for p in self.pull_positions())
        return [o for o in closed if o is not None]

    def sync_open_orders(self) -> list[Order]:
        adopted: list[Order] = []
        for item in self._session.get_all(self._path("/orders/live")):
            identifier = orders.identifier_from_external(item.get("external-identifier"), self.strategy_name)
            if identifier is None or item.get("status") in orders.TERMINAL_STATUSES or item.get("replacing-order-id") is not None or not orders.is_parsable_order(item):
                continue
            if self.tracker.get_tracked_order(identifier) is not None:
                continue
            try:
                order = orders.parse_order(item, self.strategy_name)
            except BrokerError:
                logger.exception("Not adopting the malformed Tastytrade order %s", item.get("id"))
                continue
            self._events.register(order, str(item["id"]), snapshot=item)  # its existing fills must not count twice
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted

    def reconcile(self) -> None:
        """After a stream reconnect: adopt open orders, then re-read every active order we track.

        Each snapshot goes through `TastytradeOrderEvents`, which decides what changed; polled and tracked
        statuses are never compared here (`Expired` maps to `OrderStatus.EXPIRED` but ends the order as CANCELED).
        """
        try:
            self.sync_open_orders()
        except BrokerError:
            logger.exception("Adopting Tastytrade open orders after a reconnect failed")
        for order in self.tracker.get_active_orders():
            tt_id = self._events.tt_id_of(order)
            if tt_id is None:
                continue
            try:
                self._apply(self._session.get(self._path(f"/orders/{tt_id}")))
            except BrokerError:
                logger.exception("Re-reading Tastytrade order %s failed", tt_id)

    # --- market data, clock, news (Alpaca) ---------------------------------------------

    def news_provider(self) -> NewsProvider | None:
        if self._news_provider is None and self._news_provider_factory is not None:
            self._news_provider = self._news_provider_factory()
        return self._news_provider

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
        return self._market_data.get_bars(assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours)

    # --- stream ----------------------------------------------------------------------

    def start_stream(self) -> None:
        self._stream.start()

    def stop_stream(self, timeout: float = 5.0) -> None:
        try:
            self._stream.stop(timeout)
        finally:
            self._session.close()

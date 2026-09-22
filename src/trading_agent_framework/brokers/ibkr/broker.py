"""`IbkrBroker`: IBKR (TWS API via IB Gateway) for trading, account and positions; Alpaca for
market data, the calendar and news.

Wiring only: translation lives in the pure `orders`/`account` modules, threads and asyncio in
`IbkrConnection`, and event handling in `IbkrOrderEvents`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, ClassVar

from trading_agent_framework.brokers.alpaca.clock import AlpacaMarketClock
from trading_agent_framework.brokers.alpaca.data import AlpacaMarketData
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.ibkr import account, orders
from trading_agent_framework.brokers.ibkr.client import IbkrConnection
from trading_agent_framework.brokers.ibkr.events import IbkrOrderEvents
from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, OrderStatus, OrderType, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BrokerError, ConfigurationError, OrderValidationError

if TYPE_CHECKING:
    from ib_async import Contract, Trade

logger = logging.getLogger(__name__)


async def _place_and_wait(ib: Any, contract: Contract, ib_order: Any, timeout: float) -> Trade:
    """Place the order, then wait (up to `timeout`) until IBKR acknowledges or rejects it."""
    trade = ib.placeOrder(contract, ib_order)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while trade.orderStatus.status in orders.PENDING_STATUSES and loop.time() < deadline:
        await asyncio.sleep(0.05)
    return trade


class IbkrBroker(Broker):
    name: ClassVar[str] = "ibkr"

    def __init__(
        self,
        strategy_name: str,
        connection: IbkrConnection,
        *,
        market_data: AlpacaMarketData,
        client_id: int,
        tracker: OrderTracker | None = None,
        clock: MarketClock | None = None,
        is_paper: bool = True,
        news_provider_factory: Callable[[], NewsProvider] | None = None,
        ack_timeout: float = 5.0,
    ) -> None:
        super().__init__(
            strategy_name,
            tracker,
            clock=clock if clock is not None else AlpacaMarketClock(market_data.calendar_client),
            is_paper=is_paper,
        )
        self._connection = connection
        self._market_data = market_data
        self._client_id = client_id
        self._news_provider: NewsProvider | None = None
        self._news_provider_factory = news_provider_factory
        self._ack_timeout = ack_timeout
        self._events = IbkrOrderEvents(self.tracker)
        self._contracts: dict[str, Contract] = {}
        self._account_id: str | None = None
        connection.on_reconnect = self.reconcile

    # --- account ---------------------------------------------------------------------

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            accounts = self._connection.call(lambda ib: ib.managedAccounts())
            if len(accounts) != 1:
                raise ConfigurationError(
                    f"IB Gateway manages {len(accounts)} accounts ({', '.join(accounts)}); this framework needs exactly one account"
                )
            self._account_id = accounts[0]
        return self._account_id

    def _summary(self) -> list[Any]:
        account_id = self.account_id
        return self._connection.call(lambda ib: ib.accountSummaryAsync(account_id))

    def get_account(self) -> AccountBalances:
        return account.parse_account(self._summary(), self.account_id)

    def configure_account(self) -> None:
        """IBKR cannot switch margin/shorting off via the API: check the account instead."""
        for warning in account.check_account(self._summary(), self.account_id, is_paper=self.is_paper):
            logger.warning(warning)
        logger.info("IBKR account %s checked (paper=%s)", self.account_id, self.is_paper)

    # --- orders ----------------------------------------------------------------------

    def _conform_order(self, order: Order) -> Order:
        return orders.conform_order(order)

    def _qualified(self, asset: Asset) -> Contract:
        contract = self._contracts.get(asset.symbol)
        if contract is None:
            results = self._connection.call(lambda ib: ib.qualifyContractsAsync(orders.build_contract(asset)))
            contract = results[0] if results else None
            if contract is None:
                raise OrderValidationError(f"IBKR does not know the US stock {asset.symbol}")
            self._contracts[asset.symbol] = contract
        return contract

    def _submit_order(self, order: Order) -> Order:
        if not order.client_order_id:
            order.client_order_id = orders.client_order_id_for(self.strategy_name, order.identifier)
        ib_order = orders.build_order(order)
        contract = self._qualified(order.asset)
        # Tracked before placing: the event handlers can see this order before the call returns.
        self.tracker.track_unprocessed(order)
        try:
            trade = self._connection.call(lambda ib: _place_and_wait(ib, contract, ib_order, self._ack_timeout))
        except Exception as exc:
            order.set_error(exc)
            logger.exception("Failed to submit order %s", order.identifier)
            self.tracker.untrack(order)
            raise  # set_error BEFORE re-raising -- lumibot's contract
        rejection = orders.rejection_message(trade)
        if rejection is not None:
            order.set_error(rejection)
            self.tracker.untrack(order)
            raise BrokerError(f"IBKR rejected order {order.identifier}: {rejection}")
        order.update_raw(trade)
        if order.status is OrderStatus.UNPROCESSED:
            order.status = OrderStatus.SUBMITTED  # from here on, IbkrOrderEvents owns its status
        return order

    def _open_trade(self, order: Order) -> Trade:
        client_order_id = order.client_order_id
        trade = self._connection.call(
            lambda ib: next((t for t in ib.openTrades() if client_order_id and t.order.orderRef == client_order_id), None)
        )
        if trade is None:
            raise BrokerError(f"no open IBKR order for {order.identifier}")
        if trade.order.clientId != self._client_id:
            raise BrokerError(
                f"order {order.identifier} was placed by client id {trade.order.clientId}; "
                f"only that client id can change it (this broker is client id {self._client_id})"
            )
        return trade

    def cancel_order(self, order: Order) -> None:
        trade = self._open_trade(order)
        self._connection.call(lambda ib: ib.cancelOrder(trade.order))

    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """IBKR edits the order in place: the same `Order` comes back with the new prices."""
        trade = self._open_trade(order)

        def replace(ib: Any) -> Trade:
            if limit_price is not None:
                trade.order.lmtPrice = float(limit_price)
            if stop_price is not None:
                trade.order.auxPrice = float(stop_price)
            return ib.placeOrder(trade.contract, trade.order)

        self._connection.call(replace)
        if limit_price is not None:
            if order.order_type is OrderType.STOP_LIMIT:
                order.stop_limit_price = limit_price
            else:
                order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        self.tracker.process_trade_event(order, OrderEvent.MODIFIED)
        return order

    def pull_order(self, identifier: str) -> Order | None:
        return next((o for o in self.pull_orders(limit=10_000) if o.identifier == identifier), None)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        """Open orders plus those completed this session (IBKR's API keeps no deeper history)."""
        trades = self._connection.call(lambda ib: ib.trades())
        return [orders.parse_trade(t, self.strategy_name) for t in trades][-limit:]

    def pull_positions(self) -> list[Position]:
        items = self._connection.call(lambda ib: ib.portfolio())
        parsed = (orders.parse_portfolio_item(item, self.strategy_name) for item in items)
        return [p for p in parsed if p is not None]

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = next((p for p in self.pull_positions() if p.asset == asset), None)
        if position is None:
            return None
        quantity = (position.quantity * fraction).to_integral_value(rounding=ROUND_FLOOR)
        if quantity < 1:
            logger.warning("close_position(%s, %s): below one whole share, nothing sent", asset.symbol, fraction)
            return None
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        return self.submit_order(Order(self.strategy_name, asset, side, OrderType.MARKET, quantity=quantity))

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            self._connection.call(lambda ib: ib.reqGlobalCancel())
        closed = (self.close_position(p.asset) for p in self.pull_positions())
        return [o for o in closed if o is not None]

    def sync_open_orders(self) -> list[Order]:
        trades = self._connection.call(lambda ib: ib.reqAllOpenOrdersAsync())
        adopted: list[Order] = []
        for trade in trades:
            identifier = orders.identifier_from_order_ref(trade.order.orderRef, self.strategy_name)
            if identifier is None or self.tracker.get_tracked_order(identifier) is not None:
                continue
            order = orders.parse_trade(trade, self.strategy_name)
            self.tracker.track_unprocessed(order)
            adopted.append(order)
        return adopted

    def reconcile(self) -> None:
        """After a reconnect: adopt open orders, then apply fills missed while disconnected."""
        self.sync_open_orders()
        fills = self._connection.call(lambda ib: ib.reqExecutionsAsync())
        for fill in fills:
            self._events.apply_fill(fill)

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
        return self._market_data.get_bars(
            assets, length, timestep, end=self.clock.now(), include_after_hours=include_after_hours
        )

    # --- stream ----------------------------------------------------------------------

    def start_stream(self) -> None:
        self._connection.call(lambda ib: self._events.register(ib))

    def stop_stream(self, timeout: float = 5.0) -> None:
        try:
            self._connection.call(lambda ib: self._events.unregister(ib), timeout=timeout)
        except BrokerError:
            logger.exception("error unregistering IBKR order events")
        self._connection.disconnect()

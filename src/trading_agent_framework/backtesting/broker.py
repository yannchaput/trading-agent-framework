"""`Broker` implementation simulating fills against a `BacktestDataSource`, with no
network calls. Cash and positions are Decimal-exact. Order submission here tracks the
order and queues it for the next bar; `on_advance`/`process_pending` (this module,
added alongside the fill engine) are what actually fill it -- see the module's second
half below the "--- fills" marker.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from trading_agent_framework.backtesting import fills
from trading_agent_framework.backtesting.data.base import BacktestDataSource
from trading_agent_framework.backtesting.ledger import Ledger
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import OrderEvent, OrderSide, PositionSide
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import BacktestError, OrderValidationError

logger = logging.getLogger(__name__)


@dataclass
class _PendingOrder:
    order: Order
    asset: Asset
    last_evaluated: datetime


class BacktestBroker(Broker):
    """`Broker` implementation backed by simulated fills against a `BacktestDataSource`."""

    name: ClassVar[str] = "backtest"

    def __init__(
        self,
        strategy_name: str,
        *,
        data_source: BacktestDataSource,
        clock: MarketClock,
        budget: Decimal,
        timestep: str = "day",
        commission: Decimal = Decimal(0),
        slippage: Decimal = Decimal(0),
        tracker: OrderTracker | None = None,
    ) -> None:
        super().__init__(strategy_name, tracker, clock=clock, is_paper=True)
        self._data_source = data_source
        self._timestep = timestep
        self._commission = commission
        self._slippage = slippage
        self._cash = budget
        self._positions: dict[Asset, Position] = {}
        self._pending: dict[str, _PendingOrder] = {}
        self.ledger = Ledger()

    # --- Broker ABC: orders ------------------------------------------------------------

    def _conform_order(self, order: Order) -> Order:
        return order  # no broker-specific tick rounding to simulate

    def _submit_order(self, order: Order) -> Order:
        if order.notional is not None:
            raise OrderValidationError(
                "backtesting only supports quantity-based orders, not notional orders"
            )
        if not order.client_order_id:
            order.client_order_id = f"{self.strategy_name}:{order.identifier}"
        self.tracker.track_unprocessed(order)
        self.tracker.process_trade_event(order, OrderEvent.NEW)
        self._pending[order.identifier] = _PendingOrder(
            order=order, asset=order.asset, last_evaluated=self.clock.now()
        )
        return order

    def cancel_order(self, order: Order) -> None:
        self._pending.pop(order.identifier, None)
        self.tracker.process_trade_event(order, OrderEvent.CANCELED)

    def pull_order(self, identifier: str) -> Order | None:
        return self.tracker.get_tracked_order(identifier)

    def pull_orders(self, limit: int = 100) -> list[Order]:
        return self.tracker.get_all_tracked_orders()[:limit]

    def pull_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account(self) -> AccountBalances:
        portfolio_value = self._portfolio_value(self.clock.now())
        return AccountBalances(
            cash=self._cash, portfolio_value=portfolio_value, buying_power=self._cash
        )

    def modify_order(
        self, order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None
    ) -> Order:
        if order.identifier not in self._pending:
            raise BacktestError(f"order {order.identifier} is not pending; cannot modify")
        if limit_price is not None:
            order.limit_price = limit_price
        if stop_price is not None:
            order.stop_price = stop_price
        return order

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        position = self._positions.get(asset)
        if position is None:
            return None
        quantity = (position.quantity * fraction).copy_abs()
        side = OrderSide.SELL if position.side is PositionSide.LONG else OrderSide.BUY
        order = Order(strategy_name=self.strategy_name, asset=asset, side=side, quantity=quantity)
        return self.submit_order(order)

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        if cancel_orders:
            for pending in list(self._pending.values()):
                self.cancel_order(pending.order)
        closed: list[Order] = []
        for asset in list(self._positions):
            order = self.close_position(asset)
            if order is not None:
                closed.append(order)
        return closed

    def sync_open_orders(self) -> list[Order]:
        return []  # a fresh backtest has no prior state to adopt

    # --- Broker ABC: market data --------------------------------------------------------

    def get_last_price(self, asset: Asset) -> Decimal | None:
        bar = self._latest_bar(asset, self.clock.now())
        return bar.close if bar is not None else None

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        return {asset: self.get_last_price(asset) for asset in assets}

    def get_quote(self, asset: Asset) -> Quote | None:
        bar = self._latest_bar(asset, self.clock.now())
        if bar is None:
            return None
        return Quote(
            asset=asset, bid=bar.close, ask=bar.close, bid_size=None, ask_size=None,
            timestamp=self.clock.now(),
        )

    def get_bars(
        self, assets: Sequence[Asset], length: int, timestep: str = "day", *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        result: dict[Asset, Bars] = {}
        for asset in assets:
            bars = self._data_source.bars(asset, self.clock.now(), length, timestep)
            if bars is not None:
                result[asset] = bars
        return result

    def start_stream(self) -> None:
        pass

    def stop_stream(self, timeout: float = 5.0) -> None:
        pass

    # --- shared bar lookup -----------------------------------------------------------------

    def _latest_bar_with_time(
        self, asset: Asset, cutoff: datetime
    ) -> tuple[fills.Bar, datetime] | None:
        bars = self._data_source.bars(asset, cutoff, 1, self._timestep)
        if bars is None or bars.df.empty:
            return None
        row = bars.df.iloc[-1]
        bar = fills.Bar(
            open=Decimal(str(row["open"])), high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])), close=Decimal(str(row["close"])),
        )
        return bar, bars.df.index[-1].to_pydatetime()

    def _latest_bar(self, asset: Asset, cutoff: datetime) -> fills.Bar | None:
        found = self._latest_bar_with_time(asset, cutoff)
        return None if found is None else found[0]

    def _portfolio_value(self, cutoff: datetime) -> Decimal:
        return self._cash + self._positions_value(cutoff)

    def _positions_value(self, cutoff: datetime) -> Decimal:
        total = Decimal(0)
        for asset, position in self._positions.items():
            bar = self._latest_bar(asset, cutoff)
            price = bar.close if bar is not None else (position.avg_fill_price or Decimal(0))
            signed_qty = position.quantity if position.side is PositionSide.LONG else -position.quantity
            total += signed_qty * price
        return total

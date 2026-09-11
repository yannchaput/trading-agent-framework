"""Broker-agnostic abstract base class for order submission and tracking.

`Broker` fixes the template-method ordering (conform, then submit) and wires a
default `OrderTracker`, plus an account snapshot, a `MarketClock`, market
data, and optional order-stream hooks, but leaves everything broker-specific
-- how an order is conformed, how it's actually submitted, how
positions/orders/account are pulled -- to concrete subclasses (e.g.
`AlpacaBroker`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import ClassVar

from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock


class Broker(ABC):
    """Broker-agnostic interface for submitting and tracking orders."""

    name: ClassVar[str]

    def __init__(
        self,
        strategy_name: str,
        tracker: OrderTracker | None = None,
        *,
        clock: MarketClock,
        is_paper: bool = True,
    ) -> None:
        self.strategy_name = strategy_name
        self.tracker = tracker if tracker is not None else OrderTracker()
        self.clock = clock
        self.is_paper = is_paper

    def submit_order(self, order: Order) -> Order:
        order = self._conform_order(order)
        return self._submit_order(order)

    def submit_orders(self, orders: Sequence[Order]) -> list[Order]:
        return [self.submit_order(o) for o in orders]

    def get_tracked_order(self, identifier: str) -> Order | None:
        return self.tracker.get_tracked_order(identifier)

    @abstractmethod
    def _conform_order(self, order: Order) -> Order: ...

    @abstractmethod
    def _submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order: Order) -> None: ...

    @abstractmethod
    def pull_order(self, identifier: str) -> Order | None: ...

    @abstractmethod
    def pull_orders(self, limit: int = 100) -> list[Order]: ...

    @abstractmethod
    def pull_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_account(self) -> AccountBalances: ...

    @abstractmethod
    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Replace `order`'s prices at the broker; returns the replacement order."""

    @abstractmethod
    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        """Close `fraction` of the position in `asset`; None when there is no position."""

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]: ...

    @abstractmethod
    def sync_open_orders(self) -> list[Order]:
        """Track this strategy's open broker orders (e.g. after a restart).

        Returns the adopted ones.
        """

    # --- market data -------------------------------------------------------------------

    @abstractmethod
    def get_last_price(self, asset: Asset) -> Decimal | None:
        """Last traded price; None when the data source has no trade for the asset."""

    @abstractmethod
    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]: ...

    @abstractmethod
    def get_quote(self, asset: Asset) -> Quote | None: ...

    @abstractmethod
    def get_bars(
        self,
        assets: Sequence[Asset],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        """The last `length` bars per asset, oldest first; assets without data are left out."""

    def start_stream(self) -> None:  # noqa: B027 -- optional hook, deliberately not abstract
        """Start pushing order events into `tracker`. No-op for brokers without a stream."""

    def stop_stream(self, timeout: float = 5.0) -> None:  # noqa: B027 -- optional hook
        """Stop the order-event stream. No-op for brokers without a stream."""

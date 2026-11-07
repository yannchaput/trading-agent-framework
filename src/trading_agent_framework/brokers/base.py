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

from trading_agent_framework.brokers.news import NewsProvider
from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.clock import MarketClock
from trading_agent_framework.utils.errors import OrderValidationError


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
        """Set up the order tracker, clock and paper/live flag shared by every broker.

        Args:
            strategy_name: Name that prefixes every client order id.
            tracker: Order tracker; a fresh one is created when omitted.
            clock: Market clock the broker (and the executor) reads time from.
            is_paper: Whether the account is a paper account.
        """
        self.strategy_name = strategy_name
        self.tracker = tracker if tracker is not None else OrderTracker()
        self.clock = clock
        self.is_paper = is_paper

    def submit_order(self, order: Order) -> Order:
        """Validate, conform and submit `order`, then return it as tracked by the broker."""
        # Checked here, not in Order: orders parsed from broker responses can legitimately carry a 0 size.
        for field_name, size in (("quantity", order.quantity), ("notional", order.notional)):
            if size is not None and size <= 0:
                raise OrderValidationError(f"{field_name} must be positive, got {size}; the direction goes in `side`, not the sign")
        order = self._conform_order(order)
        return self._submit_order(order)

    def submit_orders(self, orders: Sequence[Order]) -> list[Order]:
        """Submit each order in turn and return them in the same order."""
        return [self.submit_order(o) for o in orders]

    def get_tracked_order(self, identifier: str) -> Order | None:
        """Look an order up in the tracker by identifier; `None` when it is not tracked."""
        return self.tracker.get_tracked_order(identifier)

    def news_provider(self) -> NewsProvider | None:
        """This broker's news source, or `None` when it has none (`news_tools` then reports an error)."""
        return None

    @abstractmethod
    def _conform_order(self, order: Order) -> Order:
        """Adjust `order` to what the venue accepts (rounding, fractional rules) before it is sent."""
        ...

    @abstractmethod
    def _submit_order(self, order: Order) -> Order:
        """Send an already conformed order to the venue and return it with broker-assigned state."""
        ...

    @abstractmethod
    def cancel_order(self, order: Order) -> None:
        """Cancel one open order at the venue."""
        ...

    @abstractmethod
    def pull_order(self, identifier: str) -> Order | None:
        """Fetch one order from the venue by identifier; `None` when unknown."""
        ...

    @abstractmethod
    def pull_orders(self, limit: int = 100) -> list[Order]:
        """Fetch up to `limit` recent orders from the venue."""
        ...

    @abstractmethod
    def pull_positions(self) -> list[Position]:
        """Fetch every open position from the venue."""
        ...

    @abstractmethod
    def get_account(self) -> AccountBalances:
        """Fetch the account balances (cash, portfolio value, buying power)."""
        ...

    @abstractmethod
    def modify_order(
        self,
        order: Order,
        *,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Replace `order`'s prices at the broker; returns the replacement order."""
        ...

    @abstractmethod
    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        """Close `fraction` of the position in `asset`; None when there is no position."""
        ...

    @abstractmethod
    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        """Close every position, cancelling open orders first by default; return the closing orders."""
        ...

    @abstractmethod
    def sync_open_orders(self) -> list[Order]:
        """Track this strategy's open broker orders (e.g. after a restart).

        Returns the adopted ones.
        """
        ...

    # --- market data -------------------------------------------------------------------

    @abstractmethod
    def get_last_price(self, asset: Asset) -> Decimal | None:
        """Last traded price; None when the data source has no trade for the asset."""
        ...

    @abstractmethod
    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        """Last traded price per asset; `None` where none is available."""
        ...

    @abstractmethod
    def get_quote(self, asset: Asset) -> Quote | None:
        """Latest bid/ask quote for `asset`, or `None` when unavailable."""
        ...

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
        ...

    def start_stream(self) -> None:  # noqa: B027 -- optional hook, deliberately not abstract
        """Start pushing order events into `tracker`. No-op for brokers without a stream."""
        ...

    def stop_stream(self, timeout: float = 5.0) -> None:  # noqa: B027 -- optional hook
        """Stop the order-event stream. No-op for brokers without a stream."""
        ...

    def settle_unknown_orders(self) -> None:  # noqa: B027 -- optional hook
        """Decide the orders whose submission outcome is unknown (`BrokerUncertainError`); the executor calls it before every tick.

        No-op for brokers whose submits are never uncertain. A broker that has them looks each up and adopts the real ones
        and fails the missing ones (never resending), so a strategy's next tick sees the outcome.
        """
        ...

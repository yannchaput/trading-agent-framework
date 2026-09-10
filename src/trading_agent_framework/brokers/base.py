"""Broker-agnostic abstract base class for order submission and tracking.

`Broker` fixes the template-method ordering (conform, then submit) and wires a
default `OrderTracker`, but leaves everything broker-specific -- how an order
is conformed, how it's actually submitted, how positions/orders are pulled --
to concrete subclasses (e.g. `AlpacaBroker`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from trading_agent_framework.brokers.tracker import OrderTracker
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position


class Broker(ABC):
    """Broker-agnostic interface for submitting and tracking orders."""

    name: ClassVar[str]

    def __init__(self, strategy_name: str, tracker: OrderTracker | None = None) -> None:
        self.strategy_name = strategy_name
        self.tracker = tracker if tracker is not None else OrderTracker()

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

"""`PlaceholderBroker`: what `main.py` builds a strategy with in backtesting mode, before
`run_backtest` swaps in the real `BacktestBroker`. It only carries the strategy name and a
simulated clock and touches no network -- building a live broker just to start a backtest used
to reconfigure the live Alpaca account, and would need IB Gateway running under `BROKER=ibkr`."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import ClassVar, NoReturn

from trading_agent_framework.backtesting.clock import BacktestClock
from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.entities.account import AccountBalances
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.utils.errors import BrokerError

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)  # never read: run_backtest replaces the clock too


def _unavailable() -> NoReturn:
    raise BrokerError("not available before the backtest starts")


class PlaceholderBroker(Broker):
    name: ClassVar[str] = "placeholder"

    def __init__(self, strategy_name: str) -> None:
        super().__init__(strategy_name, clock=BacktestClock(start=_EPOCH, sessions=[]), is_paper=True)

    def _conform_order(self, order: Order) -> Order:
        _unavailable()

    def _submit_order(self, order: Order) -> Order:
        _unavailable()

    def cancel_order(self, order: Order) -> None:
        _unavailable()

    def pull_order(self, identifier: str) -> Order | None:
        _unavailable()

    def pull_orders(self, limit: int = 100) -> list[Order]:
        _unavailable()

    def pull_positions(self) -> list[Position]:
        _unavailable()

    def get_account(self) -> AccountBalances:
        _unavailable()

    def modify_order(self, order: Order, *, limit_price: Decimal | None = None, stop_price: Decimal | None = None) -> Order:
        _unavailable()

    def close_position(self, asset: Asset, fraction: Decimal = Decimal(1)) -> Order | None:
        _unavailable()

    def close_all_positions(self, cancel_orders: bool = True) -> list[Order]:
        _unavailable()

    def sync_open_orders(self) -> list[Order]:
        _unavailable()

    def get_last_price(self, asset: Asset) -> Decimal | None:
        _unavailable()

    def get_last_prices(self, assets: Sequence[Asset]) -> dict[Asset, Decimal | None]:
        _unavailable()

    def get_quote(self, asset: Asset) -> Quote | None:
        _unavailable()

    def get_bars(self, assets: Sequence[Asset], length: int, timestep: str = "day", *, include_after_hours: bool = True) -> dict[Asset, Bars]:
        _unavailable()

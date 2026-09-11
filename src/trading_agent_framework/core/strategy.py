"""Lean port of lumibot's `Strategy` template.

Subclasses override the lifecycle hooks (`initialize`, `on_trading_iteration`,
`before_market_opens`, ...) and the order-event hooks. Everything else is a thin
facade over the broker, so strategy code reads like lumibot strategy code.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from trading_agent_framework.brokers.base import Broker
from trading_agent_framework.clock import MarketClock
from trading_agent_framework.config.env import TradingMode
from trading_agent_framework.core.executor import StrategyExecutor
from trading_agent_framework.core.indicators import Indicators
from trading_agent_framework.entities.asset import Asset
from trading_agent_framework.entities.bars import Bars
from trading_agent_framework.entities.enums import (
    ACTIVE_ORDER_STATUSES,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trading_agent_framework.entities.order import Order
from trading_agent_framework.entities.position import Position
from trading_agent_framework.entities.quote import Quote
from trading_agent_framework.errors import BrokerError, ConfigurationError
from trading_agent_framework.log import ColorLogger, setup_strategy_logging

logger = logging.getLogger(__name__)

Number = Decimal | int | float | str

_FINAL_STATUSES = frozenset(OrderStatus) - ACTIVE_ORDER_STATUSES


def _to_asset(asset: Asset | str) -> Asset:
    return asset if isinstance(asset, Asset) else Asset(asset)


def _to_decimal(value: Number) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _to_optional_decimal(value: Number | None) -> Decimal | None:
    return None if value is None else _to_decimal(value)


def _format_duration(delta: timedelta) -> str:
    """HH:MM:SS, with hours allowed past 24 (a weekend wait reads 65:30:00)."""
    total = max(0, int(delta.total_seconds()))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Strategy:
    """Base class for strategies, with lumibot's hook names and signatures."""

    sleeptime: int | str = "1M"
    minutes_before_opening: int = 60
    minutes_before_closing: int = 1
    minutes_after_closing: int = 0
    parameters: Mapping[str, Any] = MappingProxyType({})

    def __init__(
        self,
        broker: Broker,
        *,
        mode: TradingMode = TradingMode.PAPER,
        parameters: Mapping[str, Any] | None = None,
        clock: MarketClock | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.broker = broker
        self.trading_mode = mode
        self.parameters = {**type(self).parameters, **(parameters or {})}
        self.clock = clock if clock is not None else broker.clock
        self.project_root = project_root
        self.vars = SimpleNamespace()
        self.first_iteration = True
        self._log = ColorLogger(logger, self.name)
        self._indicators: Indicators | None = None
        self.executor = StrategyExecutor(self)

    @property
    def name(self) -> str:
        return self.broker.strategy_name

    @property
    def is_backtesting(self) -> bool:
        return self.trading_mode is TradingMode.BACKTESTING

    @property
    def indicators(self) -> Indicators:
        """pandas-ta-classic indicators over this strategy's bars, e.g. `indicators.sma(a, length=20)`."""
        if self._indicators is None:
            self._indicators = Indicators(self)
        return self._indicators

    # --- lifecycle hooks -------------------------------------------------------

    def initialize(self) -> None:
        """Called once before trading starts; receives matching `parameters` as kwargs."""

    def on_trading_iteration(self) -> None:
        """Called every `sleeptime` while the market is open."""

    def before_market_opens(self) -> None:
        """Called `minutes_before_opening` before each session opens."""

    def before_starting_trading(self) -> None:
        """Called when each session opens, before its first iteration."""

    def before_market_closes(self) -> None:
        """Called `minutes_before_closing` before each session closes."""

    def after_market_closes(self) -> None:
        """Called `minutes_after_closing` after each session closes."""

    def on_strategy_end(self) -> None:
        """Called once when the run ends (stop, interrupt, or no more sessions)."""

    def on_bot_crash(self, error: BaseException) -> None:
        """Called when a hook raises; like lumibot, defaults to `on_abrupt_closing()`."""
        self.on_abrupt_closing()

    def on_abrupt_closing(self) -> None:
        """Called on Ctrl+C / SIGTERM, and by the default `on_bot_crash`."""

    # --- order-event hooks -----------------------------------------------------

    def on_new_order(self, order: Order) -> None:
        """Called when the broker accepts an order."""

    def on_canceled_order(self, order: Order) -> None:
        """Called when an order is canceled or expires."""

    def on_partially_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called on each partial fill; `quantity` is the size of this fill."""

    def on_filled_order(
        self,
        position: Position | None,
        order: Order,
        price: Decimal,
        quantity: Decimal,
        multiplier: int,
    ) -> None:
        """Called when an order is completely filled."""

    # --- logging ---------------------------------------------------------------

    def log_debug(self, message: object) -> str:
        return self._log.log_debug(message, stacklevel=2)

    def log_info(self, message: object) -> str:
        return self._log.log_info(message, stacklevel=2)

    def log_warning(self, message: object) -> str:
        return self._log.log_warning(message, stacklevel=2)

    def log_error(self, message: object) -> str:
        return self._log.log_error(message, stacklevel=2)

    def log_critical(self, message: object) -> str:
        return self._log.log_critical(message, stacklevel=2)

    # --- control -------------------------------------------------------------------

    def sleep(self, seconds: float) -> None:
        """Pause for `seconds` of clock time; order hooks still fire meanwhile."""
        self.executor.wait_until(self.get_datetime() + timedelta(seconds=seconds))

    def stop(self) -> None:
        """End the run once the current hook returns; `on_strategy_end` still runs."""
        self.executor.stop()

    def wait_for_order_execution(self, order: Order, timeout: float | None = None) -> bool:
        """Wait until `order` is filled, canceled, expired or rejected; False on timeout/stop.

        Needs the broker's trade stream (the runners start it). Order hooks keep firing
        while waiting. A modified order is replaced: wait on the order `modify_order` returned.
        """
        return self.wait_for_orders_execution([order], timeout)

    def wait_for_orders_execution(
        self, orders: Sequence[Order], timeout: float | None = None
    ) -> bool:
        return self.executor.wait_for(
            lambda: all(order.status in _FINAL_STATUSES for order in orders), timeout
        )

    # --- accounting --------------------------------------------------------------

    def get_datetime(self) -> datetime:
        return self.clock.now()

    def get_cash(self) -> Decimal:
        return self.broker.get_account().cash

    def get_portfolio_value(self) -> Decimal:
        return self.broker.get_account().portfolio_value

    @property
    def cash(self) -> Decimal:
        return self.get_cash()

    @property
    def portfolio_value(self) -> Decimal:
        return self.get_portfolio_value()

    def get_positions(self) -> list[Position]:
        return self.broker.pull_positions()

    def get_position(self, asset: Asset | str) -> Position | None:
        symbol = _to_asset(asset).symbol
        return next((p for p in self.get_positions() if p.asset.symbol == symbol), None)

    def get_orders(self) -> list[Order]:
        return self.broker.tracker.get_all_tracked_orders()

    def get_order(self, identifier: str) -> Order | None:
        tracked = self.broker.get_tracked_order(identifier)
        return tracked if tracked is not None else self.broker.pull_order(identifier)

    # --- market data -----------------------------------------------------------------

    def get_last_price(self, asset: Asset | str) -> Decimal | None:
        """Last traded price; for the quote midpoint use `get_quote(asset).mid`."""
        return self.broker.get_last_price(_to_asset(asset))

    def get_last_prices(self, assets: Iterable[Asset | str]) -> dict[Asset, Decimal | None]:
        return self.broker.get_last_prices([_to_asset(asset) for asset in assets])

    def get_quote(self, asset: Asset | str) -> Quote | None:
        return self.broker.get_quote(_to_asset(asset))

    def get_historical_prices(
        self,
        asset: Asset | str,
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> Bars | None:
        """The last `length` "minute" or "day" bars, oldest first; None without data."""
        target = _to_asset(asset)
        bars = self.broker.get_bars(
            [target], length, timestep, include_after_hours=include_after_hours
        )
        return bars.get(target)

    def get_historical_prices_for_assets(
        self,
        assets: Iterable[Asset | str],
        length: int,
        timestep: str = "day",
        *,
        include_after_hours: bool = True,
    ) -> dict[Asset, Bars]:
        return self.broker.get_bars(
            [_to_asset(asset) for asset in assets],
            length,
            timestep,
            include_after_hours=include_after_hours,
        )

    # --- trading -----------------------------------------------------------------

    def create_order(
        self,
        asset: Asset | str,
        quantity: Number,
        side: OrderSide | str,
        *,
        limit_price: Number | None = None,
        stop_price: Number | None = None,
        time_in_force: TimeInForce | str = TimeInForce.DAY,
    ) -> Order:
        """Build (not submit) an order; the type follows from the prices given."""
        limit = _to_optional_decimal(limit_price)
        stop = _to_optional_decimal(stop_price)
        if limit is not None and stop is not None:
            order_type = OrderType.STOP_LIMIT
        elif limit is not None:
            order_type = OrderType.LIMIT
        elif stop is not None:
            order_type = OrderType.STOP
        else:
            order_type = OrderType.MARKET
        is_stop_limit = order_type is OrderType.STOP_LIMIT
        return Order(
            strategy_name=self.name,
            asset=_to_asset(asset),
            side=OrderSide(side),
            order_type=order_type,
            quantity=_to_decimal(quantity),
            time_in_force=TimeInForce(time_in_force),
            limit_price=None if is_stop_limit else limit,
            stop_price=stop,
            stop_limit_price=limit if is_stop_limit else None,
        )

    def submit_order(self, order: Order) -> Order:
        return self.broker.submit_order(order)

    def submit_orders(self, orders: Sequence[Order]) -> list[Order]:
        return self.broker.submit_orders(orders)

    def cancel_order(self, order: Order) -> None:
        self.broker.cancel_order(order)

    def cancel_orders(self, orders: Iterable[Order]) -> None:
        for order in orders:
            self.cancel_order(order)

    def cancel_open_orders(self) -> None:
        self.cancel_orders(self.broker.tracker.get_active_orders())

    def modify_order(
        self, order: Order, limit_price: Number | None = None, stop_price: Number | None = None
    ) -> Order:
        """Change an open order's prices; returns the replacement order (new identifier)."""
        return self.broker.modify_order(
            order,
            limit_price=_to_optional_decimal(limit_price),
            stop_price=_to_optional_decimal(stop_price),
        )

    def sell_all(self, cancel_open_orders: bool = True) -> list[Order]:
        return self.broker.close_all_positions(cancel_orders=cancel_open_orders)

    def close_position(self, asset: Asset | str, fraction: Number = 1) -> Order | None:
        return self.broker.close_position(_to_asset(asset), _to_decimal(fraction))

    def close_positions(self, assets: Iterable[Asset | str] | None = None) -> list[Order]:
        """Close each given asset's position; `None` means every position currently held."""
        targets = [p.asset for p in self.get_positions()] if assets is None else assets
        orders = [self.close_position(asset) for asset in targets]
        return [order for order in orders if order is not None]

    # --- runners (the lumibot-agent `WrappingStrategy` entry points) -------------------

    def run_strategy(self) -> None:
        if self.trading_mode is TradingMode.LIVE:
            self.run_live_trading()
        elif self.trading_mode is TradingMode.PAPER:
            self.run_paper_trading()
        else:
            self.run_backtesting()

    def run_paper_trading(self) -> None:
        self._run_trading(TradingMode.PAPER)

    def run_live_trading(self) -> None:
        self._run_trading(TradingMode.LIVE)

    def run_backtesting(self) -> None:
        raise NotImplementedError(
            "backtesting is not implemented yet; it ships with the backtesting subproject"
        )

    def _run_trading(self, mode: TradingMode) -> None:
        if self.broker.is_paper != (mode is TradingMode.PAPER):
            account_kind = "paper" if self.broker.is_paper else "live"
            raise ConfigurationError(
                f"Refusing to run strategy {self.name!r} in {mode} mode "
                f"against a {account_kind} broker account"
            )
        self.trading_mode = mode
        log_file = setup_strategy_logging(self.name, mode, project_root=self.project_root)
        self._log_startup_banner(mode, log_file)
        self.executor.run()

    def _log_startup_banner(self, mode: TradingMode, log_file: Path) -> None:
        self.log_info(f"======== {mode.value.upper()} TRADING MODE ========")
        self.log_info(f"Logs will be saved to: {log_file.parent}")
        self.log_info(f"Broker account: {'PAPER' if self.broker.is_paper else 'LIVE'}")
        self.log_info(f"Parameters: {dict(self.parameters)}")
        self._log_market_conditions()
        try:
            self.log_info(f"Initial cash: {self.get_cash()}")
            for position in self.get_positions():
                self.log_info(f"Position: {position.quantity} {position.asset}")
        except BrokerError as exc:
            self.log_warning(f"Could not fetch account/position info: {exc}")

    def _log_market_conditions(self) -> None:
        try:
            session = self.clock.next_session()
        except BrokerError as exc:
            self.log_warning(f"Market calendar unavailable: {exc}")
            return
        if session is None:
            self.log_warning("No upcoming market session")
            return
        now = self.clock.now()
        if session.open > now:
            self.log_info(f"{_format_duration(session.open - now)} until market opens")
        else:
            self.log_info("Market is open")
        self.log_info(f"{_format_duration(session.close - now)} until market closes")
